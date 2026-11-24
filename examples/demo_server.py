"""A tiny MCP server built with the official Python SDK, used for demos and integration tests.

python examples/demo_server.py                          # stdio
python examples/demo_server.py --transport sse --port 8001
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime

from mcp.server.mcpserver import MCPServer

server = MCPServer("demo")


@server.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@server.tool()
def word_count(text: str) -> int:
    """Count the words in a piece of text."""
    return len(text.split())


@server.tool()
def utc_now() -> str:
    """The current UTC time in ISO 8601 format."""
    return datetime.now(UTC).isoformat()


@server.tool()
def fail(reason: str = "requested failure") -> str:
    """Always fails. Useful for exercising error handling."""
    raise ValueError(reason)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["stdio", "sse"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    if args.transport == "stdio":
        server.run("stdio")
    else:
        server.run("sse", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
