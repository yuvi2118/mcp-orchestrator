"""Exception hierarchy. Everything the library raises derives from :class:`MCPError`."""

from __future__ import annotations

from typing import Any


class MCPError(Exception):
    """Base class for all mcp-orchestrator errors."""


class ProtocolError(MCPError):
    """A message violated JSON-RPC 2.0 or the MCP wire format."""


class JSONRPCError(MCPError):
    """The peer answered a request with a JSON-RPC error object."""

    def __init__(self, code: int, message: str, data: Any | None = None) -> None:
        super().__init__(f"JSON-RPC error {code}: {message}")
        self.code = code
        self.message = message
        self.data = data


class TransportError(MCPError):
    """The underlying transport (subprocess pipe or HTTP stream) failed."""


class SessionClosed(MCPError):
    """The session is closed or the connection dropped while a request was pending."""


class RequestTimeout(MCPError):
    """A request did not receive a response within its timeout."""

    def __init__(self, method: str, timeout: float) -> None:
        super().__init__(f"request '{method}' timed out after {timeout:g}s")
        self.method = method
        self.timeout = timeout


class ToolNotFound(MCPError):
    """No registered tool has the requested name."""

    def __init__(self, name: str) -> None:
        super().__init__(f"unknown tool '{name}'")
        self.name = name


class InvalidToolArguments(MCPError):
    """Tool arguments failed validation against the tool's JSON Schema."""

    def __init__(self, tool: str, errors: list[str]) -> None:
        super().__init__(f"invalid arguments for '{tool}': " + "; ".join(errors))
        self.tool = tool
        self.errors = errors


class ToolExecutionError(MCPError):
    """The server executed the tool and reported ``isError: true``."""

    def __init__(self, tool: str, message: str) -> None:
        super().__init__(f"tool '{tool}' failed: {message}")
        self.tool = tool
        self.message = message


class ServerConfigError(MCPError):
    """A server definition or config file is invalid."""
