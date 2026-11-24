"""FastAPI service exposing the registry over HTTP."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .agent import AgentRunner
from .config import load_config
from .errors import (
    InvalidToolArguments,
    MCPError,
    RequestTimeout,
    SessionClosed,
    ToolExecutionError,
    ToolNotFound,
    TransportError,
)
from .registry import ToolRegistry
from .tracing import Tracer, tracer_from_env

log = logging.getLogger(__name__)


class CallRequest(BaseModel):
    arguments: dict[str, Any] = Field(default_factory=dict)
    timeout: float | None = Field(default=None, gt=0)


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_turns: int = Field(default=8, ge=1, le=25)


def create_app(
    registry: ToolRegistry | None = None,
    *,
    config_path: str | None = None,
    tracer: Tracer | None = None,
    agent_client: Any | None = None,
) -> FastAPI:
    """Build the app.

    Pass a ready ``registry`` (the caller then owns its lifecycle), or let the app load
    ``config_path`` / ``$MCP_ORCHESTRATOR_CONFIG`` and manage connections itself.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if registry is not None:
            app.state.registry = registry
            yield
            return
        path = config_path or os.environ.get("MCP_ORCHESTRATOR_CONFIG")
        active_tracer = tracer or tracer_from_env()
        owned = ToolRegistry(tracer=active_tracer)
        if path:
            await owned.connect_all(load_config(path))
        else:
            log.warning("no config path given; starting with no MCP servers")
        app.state.registry = owned
        app.state.tracer = active_tracer
        try:
            yield
        finally:
            await owned.aclose()

    app = FastAPI(title="mcp-orchestrator", version="0.1.0", lifespan=lifespan)
    app.state.agent_client = agent_client
    if registry is not None:
        app.state.registry = registry
        app.state.tracer = tracer

    def reg(request: Request) -> ToolRegistry:
        return request.app.state.registry  # type: ignore[no-any-return]

    @app.exception_handler(ToolNotFound)
    async def _not_found(_: Request, exc: ToolNotFound) -> JSONResponse:
        return JSONResponse({"error": "tool_not_found", "detail": str(exc)}, status_code=404)

    @app.exception_handler(InvalidToolArguments)
    async def _invalid(_: Request, exc: InvalidToolArguments) -> JSONResponse:
        return JSONResponse(
            {"error": "invalid_arguments", "detail": exc.errors, "tool": exc.tool}, status_code=422
        )

    @app.exception_handler(RequestTimeout)
    async def _timeout(_: Request, exc: RequestTimeout) -> JSONResponse:
        return JSONResponse({"error": "timeout", "detail": str(exc)}, status_code=504)

    @app.exception_handler(ToolExecutionError)
    async def _tool_failed(_: Request, exc: ToolExecutionError) -> JSONResponse:
        return JSONResponse(
            {"error": "tool_error", "tool": exc.tool, "detail": exc.message}, status_code=502
        )

    @app.exception_handler(SessionClosed)
    @app.exception_handler(TransportError)
    async def _unavailable(_: Request, exc: MCPError) -> JSONResponse:
        return JSONResponse({"error": "server_unavailable", "detail": str(exc)}, status_code=503)

    @app.exception_handler(MCPError)
    async def _mcp_error(_: Request, exc: MCPError) -> JSONResponse:
        return JSONResponse({"error": "mcp_error", "detail": str(exc)}, status_code=502)

    @app.get("/healthz")
    async def healthz(request: Request) -> dict[str, Any]:
        status = reg(request).status()
        healthy = all(s["connected"] for s in status.values())
        return {"status": "ok" if healthy else "degraded", "servers": status}

    @app.get("/servers")
    async def servers(request: Request) -> dict[str, Any]:
        return reg(request).status()

    @app.get("/tools")
    async def tools(request: Request, server: str | None = None) -> list[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "server": t.server,
                "description": t.tool.description,
                "input_schema": t.tool.input_schema,
            }
            for t in reg(request).tools(server)
        ]

    @app.post("/tools/{name}/call")
    async def call_tool(name: str, body: CallRequest, request: Request) -> dict[str, Any]:
        result = await reg(request).call(name, body.arguments, timeout=body.timeout)
        return {
            "text": result.text,
            "structured_content": result.structured_content,
            "content": [block.model_dump(by_alias=True) for block in result.content],
        }

    @app.post("/chat")
    async def chat(body: ChatRequest, request: Request) -> dict[str, Any]:
        client = request.app.state.agent_client
        if client is None:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise HTTPException(503, "ANTHROPIC_API_KEY is not set; /chat is unavailable")
            from anthropic import AsyncAnthropic

            client = request.app.state.agent_client = AsyncAnthropic()
        runner = AgentRunner(
            reg(request),
            client,
            max_turns=body.max_turns,
            tracer=getattr(request.app.state, "tracer", tracer),
        )
        result = await runner.run(body.prompt)
        return {
            "text": result.text,
            "stop_reason": result.stop_reason,
            "turns": result.turns,
            "tool_calls": [
                {
                    "name": c.name,
                    "arguments": c.arguments,
                    "output": c.output,
                    "is_error": c.is_error,
                }
                for c in result.tool_calls
            ],
        }

    return app
