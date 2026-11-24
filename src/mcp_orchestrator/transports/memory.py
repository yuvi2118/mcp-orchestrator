"""In-process transport pair. Used by the test-suite and handy for embedding a server."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from ..errors import TransportError
from .base import Transport

_EOF = object()


class MemoryTransport(Transport):
    def __init__(self, inbox: asyncio.Queue[Any], outbox: asyncio.Queue[Any]) -> None:
        self._inbox = inbox
        self._outbox = outbox
        self._closed = False

    @classmethod
    def pair(cls) -> tuple[MemoryTransport, MemoryTransport]:
        """Two connected endpoints: whatever one sends, the other receives."""
        a_to_b: asyncio.Queue[Any] = asyncio.Queue()
        b_to_a: asyncio.Queue[Any] = asyncio.Queue()
        return cls(inbox=b_to_a, outbox=a_to_b), cls(inbox=a_to_b, outbox=b_to_a)

    async def start(self) -> None:
        return None

    async def send(self, payload: dict[str, Any]) -> None:
        if self._closed:
            raise TransportError("memory transport is closed")
        self._outbox.put_nowait(payload)

    async def messages(self) -> AsyncIterator[dict[str, Any]]:
        while True:
            item = await self._inbox.get()
            if item is _EOF:
                self._inbox.put_nowait(_EOF)
                return
            yield item

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._outbox.put_nowait(_EOF)  # the peer sees a clean end of stream
        self._inbox.put_nowait(_EOF)  # and so does our own reader
