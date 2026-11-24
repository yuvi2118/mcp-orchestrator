from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest

from mcp_orchestrator import ClientSession, MemoryTransport, StdioTransport
from mcp_orchestrator.config import ServerConfig

from .fixtures.mini_server import handle

FIXTURES = Path(__file__).parent / "fixtures"
STDIO_SERVER = str(FIXTURES / "stdio_server.py")


def stdio_config(name: str = "mini", **overrides: Any) -> ServerConfig:
    return ServerConfig(
        name=name, transport="stdio", command=sys.executable, args=[STDIO_SERVER], **overrides
    )


@pytest.fixture
def stdio_server_config() -> Callable[..., ServerConfig]:
    return stdio_config


@pytest.fixture
async def stdio_session() -> AsyncIterator[ClientSession]:
    async with ClientSession(StdioTransport(sys.executable, [STDIO_SERVER])) as session:
        yield session


class MemoryServer:
    """Drives the server end of a memory transport with the mini_server logic.

    Hooks let a test intercept messages to delay, drop, or reorder replies.
    """

    def __init__(self, transport: MemoryTransport) -> None:
        self.transport = transport
        self.received: list[dict[str, Any]] = []
        self.interceptor: Callable[[dict[str, Any]], bool] | None = None
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        async for message in self.transport.messages():
            self.received.append(message)
            if self.interceptor and self.interceptor(message):
                continue  # the test handled (or deliberately ignored) this message
            for reply in handle(message):
                await self.transport.send(reply)

    async def send(self, payload: dict[str, Any]) -> None:
        await self.transport.send(payload)

    async def stop(self) -> None:
        await self.transport.close()
        await self._task


@pytest.fixture
async def memory_pair() -> AsyncIterator[tuple[MemoryTransport, MemoryServer]]:
    client_transport, server_transport = MemoryTransport.pair()
    server = MemoryServer(server_transport)
    yield client_transport, server
    await server.stop()
