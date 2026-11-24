"""MCP client session: request/response correlation, handshake and the tool calls."""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from . import __version__
from .errors import (
    JSONRPCError,
    MCPError,
    ProtocolError,
    RequestTimeout,
    SessionClosed,
    TransportError,
)
from .jsonrpc import (
    METHOD_NOT_FOUND,
    Notification,
    Request,
    RequestId,
    Response,
    make_error,
    make_notification,
    make_request,
    make_result,
    parse_message,
)
from .models import CallToolResult, Implementation, InitializeResult, ListToolsResult, Tool
from .transports.base import Transport

log = logging.getLogger(__name__)

SUPPORTED_PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
DEFAULT_PROTOCOL_VERSION = "2025-03-26"
_MAX_TOOL_PAGES = 100

NotificationHandler = Callable[[str, dict[str, Any] | None], None]


class ClientSession:
    """One connection to one MCP server.

    Use as an async context manager::

        async with ClientSession(StdioTransport("python", ["server.py"])) as session:
            tools = await session.list_tools()
            result = await session.call_tool("add", {"a": 1, "b": 2})
    """

    def __init__(
        self,
        transport: Transport,
        *,
        client_info: Implementation | None = None,
        request_timeout: float = 30.0,
        protocol_version: str = DEFAULT_PROTOCOL_VERSION,
        on_notification: NotificationHandler | None = None,
    ) -> None:
        self._transport = transport
        self._client_info = client_info or Implementation(
            name="mcp-orchestrator", version=__version__
        )
        self._request_timeout = request_timeout
        self._requested_version = protocol_version
        self.on_notification = on_notification

        self._ids = itertools.count(1)
        self._pending: dict[RequestId, asyncio.Future[Any]] = {}
        self._reader: asyncio.Task[None] | None = None
        self._closed = False

        self.server_info: Implementation | None = None
        self.server_capabilities: dict[str, Any] = {}
        self.protocol_version: str | None = None
        self.instructions: str | None = None

    # -- lifecycle ---------------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._reader is not None and not self._closed

    async def __aenter__(self) -> ClientSession:
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    async def start(self) -> None:
        """Connect the transport and perform the ``initialize`` handshake."""
        await self._transport.start()
        self._reader = asyncio.create_task(self._read_loop(), name="mcp-session-reader")
        try:
            await self._initialize()
        except BaseException:
            await self.close()
            raise

    async def close(self) -> None:
        if self._closed and self._reader is None:
            return
        self._closed = True
        reader, self._reader = self._reader, None
        if reader:
            reader.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await reader
        self._fail_pending(SessionClosed("session closed"))
        await self._transport.close()

    async def _initialize(self) -> None:
        raw = await self.request(
            "initialize",
            {
                "protocolVersion": self._requested_version,
                "capabilities": {},
                "clientInfo": self._client_info.model_dump(exclude_none=True),
            },
        )
        try:
            result = InitializeResult.model_validate(raw)
        except ValidationError as exc:
            raise ProtocolError(f"malformed initialize result: {exc}") from exc
        if result.protocol_version not in SUPPORTED_PROTOCOL_VERSIONS:
            raise ProtocolError(
                f"server selected unsupported protocol version '{result.protocol_version}' "
                f"(supported: {', '.join(SUPPORTED_PROTOCOL_VERSIONS)})"
            )
        self.protocol_version = result.protocol_version
        self.server_info = result.server_info
        self.server_capabilities = result.capabilities
        self.instructions = result.instructions
        await self.notify("notifications/initialized")

    # -- JSON-RPC primitives -----------------------------------------------------------------

    async def request(
        self, method: str, params: dict[str, Any] | None = None, *, timeout: float | None = None
    ) -> Any:
        """Send a request and wait for its response."""
        if self._closed:
            raise SessionClosed("session is closed")
        request_id = next(self._ids)
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        limit = self._request_timeout if timeout is None else timeout
        try:
            await self._transport.send(make_request(request_id, method, params))
            return await asyncio.wait_for(future, limit)
        except TimeoutError:
            with contextlib.suppress(MCPError):
                await self.notify(
                    "notifications/cancelled", {"requestId": request_id, "reason": "timeout"}
                )
            raise RequestTimeout(method, limit) from None
        finally:
            self._pending.pop(request_id, None)

    async def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        if self._closed:
            raise SessionClosed("session is closed")
        await self._transport.send(make_notification(method, params))

    # -- MCP operations ----------------------------------------------------------------------

    async def ping(self, *, timeout: float | None = None) -> None:
        await self.request("ping", timeout=timeout)

    async def list_tools(self) -> list[Tool]:
        """All tools the server exposes, following pagination cursors."""
        tools: list[Tool] = []
        cursor: str | None = None
        for _ in range(_MAX_TOOL_PAGES):
            raw = await self.request("tools/list", {"cursor": cursor} if cursor else None)
            try:
                page = ListToolsResult.model_validate(raw)
            except ValidationError as exc:
                raise ProtocolError(f"malformed tools/list result: {exc}") from exc
            tools.extend(page.tools)
            cursor = page.next_cursor
            if not cursor:
                return tools
        raise ProtocolError(f"tools/list did not finish after {_MAX_TOOL_PAGES} pages")

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> CallToolResult:
        raw = await self.request(
            "tools/call", {"name": name, "arguments": arguments or {}}, timeout=timeout
        )
        try:
            return CallToolResult.model_validate(raw)
        except ValidationError as exc:
            raise ProtocolError(f"malformed tools/call result: {exc}") from exc

    # -- inbound messages --------------------------------------------------------------------

    async def _read_loop(self) -> None:
        reason: MCPError = SessionClosed("connection closed by the server")
        try:
            async for raw in self._transport.messages():
                await self._dispatch(raw)
        except TransportError as exc:
            reason = exc
        finally:
            self._closed = True
            self._fail_pending(reason)

    async def _dispatch(self, raw: dict[str, Any]) -> None:
        try:
            message = parse_message(raw)
        except ProtocolError as exc:
            log.warning("dropping invalid message: %s", exc)
            return
        if isinstance(message, Response):
            self._handle_response(message)
        elif isinstance(message, Request):
            await self._handle_server_request(message)
        else:
            self._handle_notification(message)

    def _handle_response(self, response: Response) -> None:
        future = self._pending.get(response.id) if response.id is not None else None
        if future is None:
            log.warning("response for unknown request id %r", response.id)
            return
        if future.done():
            return
        if response.error is not None:
            err = response.error
            future.set_exception(JSONRPCError(err.code, err.message, err.data))
        else:
            future.set_result(response.result)

    async def _handle_server_request(self, request: Request) -> None:
        # Servers may ping us. Anything else (sampling, roots, ...) is not supported.
        if request.method == "ping":
            reply = make_result(request.id, {})
        else:
            reply = make_error(
                request.id, METHOD_NOT_FOUND, f"method '{request.method}' not supported"
            )
        try:
            await self._transport.send(reply)
        except TransportError as exc:
            log.warning("could not answer server request '%s': %s", request.method, exc)

    def _handle_notification(self, notification: Notification) -> None:
        params = notification.params if isinstance(notification.params, dict) else None
        if self.on_notification is None:
            return
        try:
            self.on_notification(notification.method, params)
        except Exception:
            log.exception("notification handler raised for '%s'", notification.method)

    def _fail_pending(self, reason: MCPError) -> None:
        for future in list(self._pending.values()):
            if not future.done():
                future.set_exception(reason)
        self._pending.clear()
