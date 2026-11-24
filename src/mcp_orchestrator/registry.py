"""Tool registry: aggregates tools from many MCP servers behind one namespaced catalogue."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from jsonschema.protocols import Validator
from jsonschema.validators import validator_for

from .config import ServerConfig, build_transport
from .errors import (
    InvalidToolArguments,
    MCPError,
    ToolExecutionError,
    ToolNotFound,
)
from .models import CallToolResult, Tool
from .session import ClientSession
from .tracing import NoopTracer, Tracer

log = logging.getLogger(__name__)

_INVALID_NAME_CHARS = re.compile(r"[^A-Za-z0-9_-]")
_MAX_NAME_LEN = 64  # the Anthropic Messages API limit for tool names


def qualified_name(server: str, tool: str, *, separator: str = "__") -> str:
    """``server__tool``, restricted to ``[A-Za-z0-9_-]`` and at most 64 characters."""
    raw = f"{_INVALID_NAME_CHARS.sub('_', server)}{separator}{_INVALID_NAME_CHARS.sub('_', tool)}"
    if len(raw) <= _MAX_NAME_LEN:
        return raw
    digest = hashlib.sha1(raw.encode()).hexdigest()[:8]
    return f"{raw[: _MAX_NAME_LEN - 9]}_{digest}"


def validate_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> list[str]:
    """Return human-readable validation errors (empty if the arguments are valid)."""
    return _errors_from(_compile(schema), arguments)


def _compile(schema: dict[str, Any]) -> Validator | None:
    try:
        cls = validator_for(schema, default=Draft202012Validator)
        cls.check_schema(schema)
        return cls(schema)
    except SchemaError as exc:
        log.warning("tool has an invalid input schema; skipping validation: %s", exc.message)
        return None


def _errors_from(validator: Validator | None, arguments: dict[str, Any]) -> list[str]:
    if validator is None:
        return []
    found = sorted(
        validator.iter_errors(arguments), key=lambda e: [str(p) for p in e.absolute_path]
    )
    return [f"{'.'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}" for e in found]


@dataclass(frozen=True)
class RegisteredTool:
    name: str  # qualified, unique across servers
    server: str
    tool: Tool
    validator: Validator | None = None

    def to_anthropic(self) -> dict[str, Any]:
        """The tool definition in the format the Anthropic Messages API expects."""
        return {
            "name": self.name,
            "description": self.tool.description or self.tool.title or self.tool.name,
            "input_schema": self.tool.input_schema,
        }


class ToolRegistry:
    def __init__(self, tracer: Tracer | None = None, *, separator: str = "__") -> None:
        self._tracer: Tracer = tracer or NoopTracer()
        self._separator = separator
        self._sessions: dict[str, ClientSession] = {}
        self._tools: dict[str, RegisteredTool] = {}
        self._refresh_tasks: set[asyncio.Task[None]] = set()
        self.failures: dict[str, str] = {}

    # -- connecting --------------------------------------------------------------------------

    async def __aenter__(self) -> ToolRegistry:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def add_server(self, config: ServerConfig) -> list[RegisteredTool]:
        """Connect to a server described by ``config`` and register its tools."""
        session = ClientSession(build_transport(config), request_timeout=config.request_timeout)
        await session.start()
        try:
            return await self.add_session(config.name, session)
        except BaseException:
            await session.close()
            raise

    async def add_session(self, name: str, session: ClientSession) -> list[RegisteredTool]:
        """Register an already-started session."""
        if name in self._sessions:
            raise ValueError(f"a server named '{name}' is already registered")
        tools = await session.list_tools()
        self._sessions[name] = session
        session.on_notification = lambda method, _params: self._on_notification(name, method)
        return self._install_tools(name, tools)

    async def connect_all(self, configs: Iterable[ServerConfig]) -> None:
        """Connect to every server concurrently. One server failing does not stop the others."""
        configs = list(configs)
        results = await asyncio.gather(
            *(self.add_server(c) for c in configs), return_exceptions=True
        )
        for config, result in zip(configs, results, strict=True):
            if isinstance(result, BaseException):
                self.failures[config.name] = f"{type(result).__name__}: {result}"
                log.error("server '%s' failed to connect: %s", config.name, result)

    async def refresh(self, server: str) -> list[RegisteredTool]:
        """Re-read a server's tool list (called automatically on ``tools/list_changed``)."""
        tools = await self._sessions[server].list_tools()
        return self._install_tools(server, tools)

    def _install_tools(self, server: str, tools: list[Tool]) -> list[RegisteredTool]:
        self._tools = {n: t for n, t in self._tools.items() if t.server != server}
        installed: list[RegisteredTool] = []
        for tool in tools:
            name = qualified_name(server, tool.name, separator=self._separator)
            if name in self._tools:  # two names collided after sanitising
                digest = hashlib.sha1(f"{server}/{tool.name}".encode()).hexdigest()[:8]
                name = f"{name[: _MAX_NAME_LEN - 9]}_{digest}"
            entry = RegisteredTool(name, server, tool, _compile(tool.input_schema))
            self._tools[name] = entry
            installed.append(entry)
        return installed

    def _on_notification(self, server: str, method: str) -> None:
        if method != "notifications/tools/list_changed":
            return
        task = asyncio.create_task(self._safe_refresh(server))
        self._refresh_tasks.add(task)
        task.add_done_callback(self._refresh_tasks.discard)

    async def _safe_refresh(self, server: str) -> None:
        try:
            await self.refresh(server)
        except MCPError:
            log.exception("could not refresh tools for '%s'", server)

    # -- catalogue ---------------------------------------------------------------------------

    def tools(self, server: str | None = None) -> list[RegisteredTool]:
        entries = list(self._tools.values())
        return [t for t in entries if server is None or t.server == server]

    def get(self, name: str) -> RegisteredTool:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolNotFound(name) from None

    def to_anthropic_tools(self) -> list[dict[str, Any]]:
        return [t.to_anthropic() for t in self._tools.values()]

    def status(self) -> dict[str, dict[str, Any]]:
        report: dict[str, dict[str, Any]] = {
            name: {"connected": session.is_connected, "tools": len(self.tools(name))}
            for name, session in self._sessions.items()
        }
        for name, error in self.failures.items():
            report[name] = {"connected": False, "tools": 0, "error": error}
        return report

    # -- calling -----------------------------------------------------------------------------

    async def call(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
        raise_on_error: bool = True,
    ) -> CallToolResult:
        """Validate arguments, call the tool on its server, and trace the whole call.

        Raises ``ToolNotFound``, ``InvalidToolArguments``, a transport/protocol error, or
        ``ToolExecutionError`` when the tool reports ``isError`` (unless ``raise_on_error=False``).
        """
        entry = self.get(name)
        args = arguments or {}
        metadata: dict[str, Any] = {"server": entry.server, "tool": entry.tool.name}
        with self._tracer.span(
            f"mcp.call_tool.{name}", kind="tool", input=args, metadata=metadata
        ) as span:
            errors = _errors_from(entry.validator, args)
            if errors:
                span.update(
                    level="ERROR",
                    status_message="invalid arguments: " + "; ".join(errors),
                    metadata=metadata,
                )
                raise InvalidToolArguments(name, errors)

            started = time.perf_counter()
            try:
                result = await self._sessions[entry.server].call_tool(
                    entry.tool.name, args, timeout=timeout
                )
            except MCPError as exc:
                span.update(
                    level="ERROR",
                    status_message=f"{type(exc).__name__}: {exc}",
                    metadata={**metadata, "duration_ms": _ms(started)},
                )
                raise
            span.update(
                output=result.structured_content or result.text,
                level="ERROR" if result.is_error else "DEFAULT",
                status_message="tool reported an error" if result.is_error else None,
                metadata={**metadata, "duration_ms": _ms(started), "is_error": result.is_error},
            )

        if result.is_error and raise_on_error:
            raise ToolExecutionError(name, result.text or "no error message")
        return result

    # -- shutdown ----------------------------------------------------------------------------

    async def aclose(self) -> None:
        for task in list(self._refresh_tasks):
            task.cancel()
        await asyncio.gather(*(s.close() for s in self._sessions.values()), return_exceptions=True)
        self._sessions.clear()
        self._tools.clear()
        self._tracer.flush()


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)
