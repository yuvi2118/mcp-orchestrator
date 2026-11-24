"""HTTP+SSE flavour of mini_server (Starlette), served by uvicorn inside the test process."""

from __future__ import annotations

import asyncio
import json
import socket
from collections.abc import AsyncIterator
from typing import Any

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from .mini_server import handle


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class SSEServer:
    """Runs the app on a free local port. Use as ``async with SSEServer() as srv: srv.url``."""

    def __init__(
        self,
        *,
        endpoint: str = "/messages",
        advertise: str | None = None,
        auth_token: str | None = None,
    ) -> None:
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}/sse"
        self._endpoint = endpoint  # the route that accepts POSTs
        self._advertised = advertise or endpoint  # what the 'endpoint' event tells the client
        self._auth = auth_token
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self.received: list[dict[str, Any]] = []
        self._server: uvicorn.Server | None = None
        self._task: asyncio.Task[None] | None = None

    def push(self, message: dict[str, Any]) -> None:
        """Send an unsolicited message (e.g. a notification) down the SSE stream."""
        self._queue.put_nowait(message)

    def _authorised(self, request: Request) -> bool:
        return self._auth is None or request.headers.get("authorization") == f"Bearer {self._auth}"

    def _app(self) -> Starlette:
        async def sse(request: Request) -> Response:
            if not self._authorised(request):
                return Response(status_code=401)

            async def stream() -> AsyncIterator[str]:
                yield f"event: endpoint\ndata: {self._advertised}\n\n"
                yield ": keep-alive comment\n\n"
                while True:
                    message = await self._queue.get()
                    yield f"event: message\ndata: {json.dumps(message)}\n\n"

            return StreamingResponse(stream(), media_type="text/event-stream")

        async def post(request: Request) -> Response:
            if not self._authorised(request):
                return Response(status_code=401)
            message = await request.json()
            self.received.append(message)
            for reply in handle(message):
                self._queue.put_nowait(reply)
            return JSONResponse({"accepted": True}, status_code=202)

        return Starlette(routes=[Route("/sse", sse), Route(self._endpoint, post, methods=["POST"])])

    async def __aenter__(self) -> SSEServer:
        config = uvicorn.Config(self._app(), host="127.0.0.1", port=self.port, log_level="error")
        self._server = uvicorn.Server(config)
        self._task = asyncio.create_task(self._server.serve())
        for _ in range(200):
            if self._server.started:
                return self
            await asyncio.sleep(0.01)
        raise RuntimeError("test SSE server did not start")

    async def __aexit__(self, *exc_info: object) -> None:
        assert self._server and self._task
        self._server.should_exit = True
        self._server.force_exit = True
        await self._task
