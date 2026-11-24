"""mcp-orchestrator: a production-style Model Context Protocol client."""

__version__ = "0.1.0"

from .agent import AgentResult, AgentRunner
from .config import ServerConfig, build_transport, load_config
from .errors import (
    InvalidToolArguments,
    JSONRPCError,
    MCPError,
    ProtocolError,
    RequestTimeout,
    ServerConfigError,
    SessionClosed,
    ToolExecutionError,
    ToolNotFound,
    TransportError,
)
from .models import CallToolResult, Tool
from .registry import RegisteredTool, ToolRegistry
from .session import ClientSession
from .tracing import LangfuseTracer, NoopTracer, RecordingTracer, Tracer, tracer_from_env
from .transports import MemoryTransport, SSETransport, StdioTransport, Transport

__all__ = [
    "AgentResult",
    "AgentRunner",
    "CallToolResult",
    "ClientSession",
    "InvalidToolArguments",
    "JSONRPCError",
    "LangfuseTracer",
    "MCPError",
    "MemoryTransport",
    "NoopTracer",
    "ProtocolError",
    "RecordingTracer",
    "RegisteredTool",
    "RequestTimeout",
    "SSETransport",
    "ServerConfig",
    "ServerConfigError",
    "SessionClosed",
    "StdioTransport",
    "Tool",
    "ToolExecutionError",
    "ToolNotFound",
    "ToolRegistry",
    "Tracer",
    "Transport",
    "TransportError",
    "build_transport",
    "load_config",
    "tracer_from_env",
]
