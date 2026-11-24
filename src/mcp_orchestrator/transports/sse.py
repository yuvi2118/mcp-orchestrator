"""HTTP+SSE transport (the original MCP remote transport).

Protocol:
1. ``GET <sse_url>`` opens an event stream.
2. The first event is ``endpoint``; its data is the URL to ``POST`` client messages to.
3. Server messages arrive as ``message`` events whose data is one JSON-RPC message.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterable, AsyncIterator, Mapping
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx

from ..errors import TransportError
from .base import Transport

log = logging.getLogger(__name__)

_CLOSED = object()


async def parse_sse(lines: AsyncIterable[str]) -> AsyncIterator[tuple[str, str]]:
    """Parse a Server-Sent Events stream into ``(event, data)`` pairs.

    ``lines`` yields text lines without line terminators. Follows the HTML spec:
    blank line dispatches an event, ``data`` lines are joined with ``\\n``, lines
    starting with ``:`` are comments, and the default event type is ``message``.
    """
    event = ""
    data: list[str] = []
    async for line in lines:
        if line == "":
            if data:
                yield (event or "message", "\n".join(data))
            event, data = "", []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if field == "event":
            event = value
        elif field == "data":
            data.append(value)
        # 'id' and 'retry' are valid SSE fields but irrelevant to MCP.
    if data:  # stream ended without a trailing blank line
        yield (event or "message", "\n".join(data))


def _origin(url: str) -> tuple[str, str]:
    parts = urlsplit(url)
    return parts.scheme, parts.netloc


class SSETransport(Transport):
    def __init__(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        connect_timeout: float = 10.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._url = url
        self._headers = dict(headers or {})
        self._connect_timeout = connect_timeout
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=None))
        self._endpoint: str | None = None
        self._endpoint_ready = asyncio.Event()
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._reader: asyncio.Task[None] | None = None
        self._failure: TransportError | None = None

    async def start(self) -> None:
        self._reader = asyncio.create_task(self._read_stream())
        try:
            await asyncio.wait_for(self._endpoint_ready.wait(), self._connect_timeout)
        except TimeoutError:
            await self.close()
            raise TransportError(
                f"no 'endpoint' event from {self._url} within {self._connect_timeout:g}s"
            ) from None
        if self._endpoint is None:
            # The stream ended (or failed) before the server told us where to POST.
            failure = self._failure or TransportError(
                f"{self._url} closed the event stream before sending an 'endpoint' event"
            )
            await self.close()
            raise failure

    async def _read_stream(self) -> None:
        headers = {"Accept": "text/event-stream", "Cache-Control": "no-store", **self._headers}
        try:
            async with self._client.stream("GET", self._url, headers=headers) as response:
                response.raise_for_status()
                async for event, data in parse_sse(response.aiter_lines()):
                    if event == "endpoint":
                        self._set_endpoint(data)
                    elif event == "message":
                        try:
                            decoded = json.loads(data)
                        except json.JSONDecodeError:
                            log.warning("ignoring non-JSON SSE message: %.120s", data)
                            continue
                        if isinstance(decoded, dict):
                            await self._queue.put(decoded)
                    else:
                        log.debug("ignoring SSE event '%s'", event)
        except asyncio.CancelledError:
            raise
        except httpx.HTTPStatusError as exc:
            self._failure = TransportError(
                f"SSE connection to {self._url} failed with HTTP {exc.response.status_code}"
            )
        except httpx.HTTPError as exc:
            self._failure = TransportError(f"SSE connection to {self._url} failed: {exc!r}")
        except TransportError as exc:
            self._failure = exc
        finally:
            self._endpoint_ready.set()  # unblock start() if we failed before the endpoint event
            await self._queue.put(_CLOSED)

    def _set_endpoint(self, data: str) -> None:
        endpoint = urljoin(self._url, data.strip())
        if _origin(endpoint) != _origin(self._url):
            raise TransportError(
                f"server sent an endpoint on a different origin ({endpoint}); refusing to use it"
            )
        self._endpoint = endpoint
        self._endpoint_ready.set()

    async def send(self, payload: dict[str, Any]) -> None:
        if self._endpoint is None:
            raise TransportError("SSE transport is not connected")
        try:
            response = await self._client.post(
                self._endpoint,
                json=payload,
                headers={"Content-Type": "application/json", **self._headers},
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise TransportError(
                f"POST to {self._endpoint} failed with HTTP {exc.response.status_code}"
            ) from exc
        except httpx.HTTPError as exc:
            raise TransportError(f"POST to {self._endpoint} failed: {exc!r}") from exc

    async def messages(self) -> AsyncIterator[dict[str, Any]]:
        while True:
            item = await self._queue.get()
            if item is _CLOSED:
                # Re-queue so a second consumer also terminates.
                self._queue.put_nowait(_CLOSED)
                if self._failure:
                    raise self._failure
                return
            yield item

    async def close(self) -> None:
        reader, self._reader = self._reader, None
        if reader:
            reader.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await reader
        if self._owns_client:
            await self._client.aclose()
