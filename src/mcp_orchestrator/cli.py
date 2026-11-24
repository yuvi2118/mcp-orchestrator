"""Command line: list tools, call a tool, or run the HTTP service."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections.abc import Sequence

from .config import load_config
from .errors import MCPError
from .registry import ToolRegistry
from .tracing import tracer_from_env


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mcp-orchestrator", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    list_cmd = sub.add_parser("list-tools", help="connect to every server and print its tools")
    list_cmd.add_argument("--config", required=True)

    call_cmd = sub.add_parser("call", help="call one tool")
    call_cmd.add_argument("--config", required=True)
    call_cmd.add_argument("tool", help="qualified tool name, e.g. demo__add")
    call_cmd.add_argument("--args", default="{}", help="arguments as a JSON object")

    serve_cmd = sub.add_parser("serve", help="run the FastAPI service")
    serve_cmd.add_argument("--config", required=True)
    serve_cmd.add_argument("--host", default="127.0.0.1")
    serve_cmd.add_argument("--port", type=int, default=8000)
    return parser


async def _run(args: argparse.Namespace) -> int:
    registry = ToolRegistry(tracer=tracer_from_env())
    async with registry:
        await registry.connect_all(load_config(args.config))
        for name, message in registry.failures.items():
            print(f"warning: server '{name}' unavailable: {message}", file=sys.stderr)

        if args.command == "list-tools":
            for tool in registry.tools():
                print(f"{tool.name}\t{tool.tool.description or ''}")
            return 0

        try:
            arguments = json.loads(args.args)
        except json.JSONDecodeError as exc:
            print(f"error: --args is not valid JSON: {exc}", file=sys.stderr)
            return 2
        try:
            result = await registry.call(args.tool, arguments)
        except MCPError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(result.text or json.dumps(result.structured_content))
        return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)

    if args.command == "serve":
        import uvicorn

        from .api import create_app

        uvicorn.run(create_app(config_path=args.config), host=args.host, port=args.port)
        return 0
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
