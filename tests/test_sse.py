from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import httpx
import pytest

from mcp_orchestrator import ClientSession, SSETransport
from mcp_orchestrator.errors import TransportError
from mcp_orchestrator.transports import parse_sse

from .fixtures.sse_server import SSEServer, free_port


async def _lines(*lines: str) -> AsyncIterator[str]:
    for line in lines:
        yield line


async def _collect(*lines: str) -> list[tuple[str, str]]:
    return [event async for event in parse_sse(_lines(*lines))]


# --- the SSE parser ---------------------------------------------------------------------------


async def test_parse_basic_event():
    assert await _collect("event: endpoint", "data: /messages", "") == [("endpoint", "/messages")]


async def test_parse_defaults_to_message_event():
    assert await _collect("data: {}", "") == [("message", "{}")]


async def test_parse_joins_multiple_data_lines():
    assert await _collect("data: a", "data: b", "") == [("message", "a\nb")]


async def test_parse_ignores_comments_ids_and_retry():
    events = await _collect(": keep-alive", "id: 7", "retry: 1000", "data: x", "")
    assert events == [("message", "x")]


async def test_parse_handles_missing_space_after_colon():
    assert await _collect("data:x", "") == [("message", "x")]


async def test_parse_flushes_event_without_trailing_blank_line():
    assert await _collect("event: message", "data: tail") == [("message", "tail")]


async def test_parse_skips_events_without_data_and_resets_event_name():
    events = await _collect("event: ping", "", "data: after", "")
    assert events == [("message", "after")]


async def test_parse_multiple_events():
    events = await _collect("data: 1", "", "event: x", "data: 2", "")
    assert events == [("message", "1"), ("x", "2")]


# --- a real HTTP+SSE round trip --------------------------------------------------------------


async def test_full_session_over_sse():
    async with SSEServer() as server:
        async with ClientSession(SSETransport(server.url), request_timeout=5.0) as session:
            assert session.server_info.name == "mini"
            assert len(await session.list_tools()) == 5
            result = await session.call_tool("add", {"a": 5, "b": 6})
            assert result.text == "11"
        methods = [m.get("method") for m in server.received]
        assert methods[:2] == ["initialize", "notifications/initialized"]


async def test_concurrent_calls_over_sse():
    async with SSEServer() as server:
        async with ClientSession(SSETransport(server.url), request_timeout=5.0) as session:
            results = await asyncio.gather(
                *(session.call_tool("echo", {"text": str(i)}) for i in range(20))
            )
            assert [r.text for r in results] == [str(i) for i in range(20)]


async def test_server_pushed_notifications_are_delivered():
    async with SSEServer() as server:
        async with ClientSession(SSETransport(server.url), request_timeout=5.0) as session:
            seen: list[str] = []
            session.on_notification = lambda method, params: seen.append(method)
            server.push({"jsonrpc": "2.0", "method": "notifications/tools/list_changed"})
            for _ in range(100):
                if seen:
                    break
                await asyncio.sleep(0.01)
            assert seen == ["notifications/tools/list_changed"]


async def test_custom_headers_are_sent_on_both_requests():
    async with SSEServer(auth_token="s3cret") as server:
        transport = SSETransport(server.url, headers={"Authorization": "Bearer s3cret"})
        async with ClientSession(transport, request_timeout=5.0) as session:
            await session.ping()


async def test_rejected_credentials_fail_fast_with_the_status_code():
    async with SSEServer(auth_token="s3cret") as server:
        transport = SSETransport(server.url, headers={"Authorization": "Bearer wrong"})
        with pytest.raises(TransportError, match="HTTP 401"):
            await ClientSession(transport).start()


async def test_unreachable_server_raises_transport_error():
    transport = SSETransport(f"http://127.0.0.1:{free_port()}/sse", connect_timeout=2.0)
    with pytest.raises(TransportError, match="failed"):
        await ClientSession(transport).start()


async def test_endpoint_on_another_origin_is_refused():
    async with SSEServer(advertise="http://attacker.example/messages") as server:
        with pytest.raises(TransportError, match="different origin"):
            await ClientSession(SSETransport(server.url)).start()


async def test_relative_endpoint_is_resolved_against_the_stream_url():
    async with SSEServer(endpoint="/custom/path") as server:
        transport = SSETransport(server.url)
        await transport.start()
        try:
            assert transport._endpoint == f"http://127.0.0.1:{server.port}/custom/path"
        finally:
            await transport.close()


# --- failure modes using a mocked HTTP layer --------------------------------------------------


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_stream_that_ends_before_the_endpoint_event():
    client = _client(lambda request: httpx.Response(200, content=b": nothing useful\n\n"))
    with pytest.raises(TransportError, match="before sending an 'endpoint' event"):
        await SSETransport("http://x/sse", client=client).start()


async def test_timeout_waiting_for_endpoint_event():
    class Hang(httpx.AsyncByteStream):
        async def __aiter__(self):
            await asyncio.sleep(30)
            yield b""

    client = _client(lambda request: httpx.Response(200, stream=Hang()))
    transport = SSETransport("http://x/sse", client=client, connect_timeout=0.1)
    with pytest.raises(TransportError, match="no 'endpoint' event"):
        await transport.start()


async def test_post_failure_raises_transport_error():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, content=b"event: endpoint\ndata: /m\n\n")
        return httpx.Response(500)

    transport = SSETransport("http://x/sse", client=_client(handler))
    await transport.start()
    try:
        with pytest.raises(TransportError, match="HTTP 500"):
            await transport.send({"jsonrpc": "2.0", "method": "x"})
    finally:
        await transport.close()


async def test_post_network_error_raises_transport_error():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, content=b"event: endpoint\ndata: /m\n\n")
        raise httpx.ConnectError("boom")

    transport = SSETransport("http://x/sse", client=_client(handler))
    await transport.start()
    try:
        with pytest.raises(TransportError, match="failed"):
            await transport.send({"jsonrpc": "2.0", "method": "x"})
    finally:
        await transport.close()


async def test_send_before_start_raises():
    with pytest.raises(TransportError, match="not connected"):
        await SSETransport("http://x/sse").send({})


async def test_messages_end_cleanly_when_the_stream_closes_and_bad_events_are_skipped():
    body = (
        b"event: endpoint\ndata: /m\n\n"
        b"event: message\ndata: not json\n\n"
        b"event: message\ndata: [1]\n\n"
        b"event: other\ndata: ignored\n\n"
        b'event: message\ndata: {"hello": "world"}\n\n'
    )
    transport = SSETransport(
        "http://x/sse", client=_client(lambda r: httpx.Response(200, content=body))
    )
    await transport.start()
    try:
        received = [m async for m in transport.messages()]
        assert received == [{"hello": "world"}]
    finally:
        await transport.close()
        await transport.close()  # idempotent


async def test_stream_failure_after_start_is_raised_to_the_reader():
    class Broken(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"event: endpoint\ndata: /m\n\n"
            raise httpx.ReadError("connection reset")

    transport = SSETransport(
        "http://x/sse", client=_client(lambda r: httpx.Response(200, stream=Broken()))
    )
    await transport.start()
    try:
        with pytest.raises(TransportError, match="failed"):
            async for _ in transport.messages():
                pass
    finally:
        await transport.close()
