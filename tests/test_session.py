from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest

from mcp_orchestrator import ClientSession, MemoryTransport
from mcp_orchestrator.errors import (
    JSONRPCError,
    ProtocolError,
    RequestTimeout,
    SessionClosed,
    TransportError,
)

from .conftest import MemoryServer


async def eventually(predicate: Callable[[], bool], timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.005)


@pytest.fixture
async def session(memory_pair) -> AsyncIterator[ClientSession]:
    transport, _server = memory_pair
    async with ClientSession(transport, request_timeout=2.0) as s:
        yield s


@pytest.fixture
def server(memory_pair) -> MemoryServer:
    return memory_pair[1]


async def test_handshake_populates_server_details(session, server):
    assert session.server_info.name == "mini"
    assert session.protocol_version == "2025-03-26"
    assert session.server_capabilities == {"tools": {"listChanged": True}}
    assert session.instructions == "test server"
    assert session.is_connected
    methods = [m.get("method") for m in server.received]
    assert methods[:2] == ["initialize", "notifications/initialized"]
    assert server.received[0]["params"]["clientInfo"]["name"] == "mcp-orchestrator"


async def test_handshake_rejects_unsupported_protocol_version(memory_pair):
    transport, server = memory_pair

    def reply_with_old_version(message: dict[str, Any]) -> bool:
        if message.get("method") == "initialize":
            asyncio.get_running_loop().create_task(
                server.send(
                    {
                        "jsonrpc": "2.0",
                        "id": message["id"],
                        "result": {
                            "protocolVersion": "1999-01-01",
                            "capabilities": {},
                            "serverInfo": {"name": "old"},
                        },
                    }
                )
            )
            return True
        return False

    server.interceptor = reply_with_old_version
    session = ClientSession(transport)
    with pytest.raises(ProtocolError, match="unsupported protocol version"):
        await session.start()
    assert not session.is_connected


async def test_handshake_rejects_malformed_initialize_result(memory_pair):
    transport, server = memory_pair

    def bad_reply(message: dict[str, Any]) -> bool:
        if message.get("method") == "initialize":
            asyncio.get_running_loop().create_task(
                server.send({"jsonrpc": "2.0", "id": message["id"], "result": {"nope": 1}})
            )
            return True
        return False

    server.interceptor = bad_reply
    with pytest.raises(ProtocolError, match="malformed initialize"):
        await ClientSession(transport).start()


async def test_list_tools_follows_pagination(session):
    tools = await session.list_tools()
    assert [t.name for t in tools] == ["echo", "add", "fail", "rich", "crash"]


async def test_list_tools_gives_up_on_endless_pagination(session, server):
    def always_more(message: dict[str, Any]) -> bool:
        if message.get("method") == "tools/list":
            asyncio.get_running_loop().create_task(
                server.send(
                    {
                        "jsonrpc": "2.0",
                        "id": message["id"],
                        "result": {"tools": [], "nextCursor": "again"},
                    }
                )
            )
            return True
        return False

    server.interceptor = always_more
    with pytest.raises(ProtocolError, match="did not finish"):
        await session.list_tools()


async def test_call_tool_returns_typed_result(session):
    result = await session.call_tool("add", {"a": 2, "b": 3})
    assert result.text == "5"
    assert result.structured_content == {"result": 5}
    assert result.is_error is False


async def test_call_tool_reports_is_error_without_raising(session):
    result = await session.call_tool("fail")
    assert result.is_error is True
    assert result.text == "something went wrong"


async def test_jsonrpc_error_response_raises(session):
    with pytest.raises(JSONRPCError) as info:
        await session.call_tool("does-not-exist")
    assert info.value.code == -32602
    assert "does-not-exist" in info.value.message


async def test_unknown_method_raises_method_not_found(session):
    with pytest.raises(JSONRPCError) as info:
        await session.request("resources/list")
    assert info.value.code == -32601


async def test_malformed_tool_result_raises_protocol_error(session, server):
    def bad(message: dict[str, Any]) -> bool:
        if message.get("method") == "tools/call":
            asyncio.get_running_loop().create_task(
                server.send({"jsonrpc": "2.0", "id": message["id"], "result": {"content": "oops"}})
            )
            return True
        return False

    server.interceptor = bad
    with pytest.raises(ProtocolError, match="malformed tools/call"):
        await session.call_tool("echo", {"text": "x"})


async def test_concurrent_requests_are_matched_even_if_answered_out_of_order(session, server):
    held: list[dict[str, Any]] = []

    def hold(message: dict[str, Any]) -> bool:
        if message.get("method") == "tools/call":
            held.append(message)
            return True
        return False

    server.interceptor = hold
    first = asyncio.create_task(session.call_tool("echo", {"text": "one"}))
    second = asyncio.create_task(session.call_tool("echo", {"text": "two"}))
    await eventually(lambda: len(held) == 2)

    for message in reversed(held):  # answer the second request first
        text = message["params"]["arguments"]["text"]
        await server.send(
            {
                "jsonrpc": "2.0",
                "id": message["id"],
                "result": {"content": [{"type": "text", "text": text}]},
            }
        )
    assert (await first).text == "one"
    assert (await second).text == "two"


async def test_timeout_raises_and_sends_cancellation(session, server):
    server.interceptor = lambda m: m.get("method") == "tools/call"  # never answer
    with pytest.raises(RequestTimeout) as info:
        await session.call_tool("echo", {"text": "x"}, timeout=0.05)
    assert info.value.method == "tools/call"
    await eventually(
        lambda: any(m.get("method") == "notifications/cancelled" for m in server.received)
    )
    cancelled = next(m for m in server.received if m.get("method") == "notifications/cancelled")
    assert cancelled["params"]["reason"] == "timeout"
    assert session.is_connected  # one slow call does not kill the session


async def test_late_response_after_timeout_is_ignored(session, server):
    held: list[dict[str, Any]] = []
    server.interceptor = lambda m: (
        (held.append(m) or True) if m.get("method") == "tools/call" else False
    )
    with pytest.raises(RequestTimeout):
        await session.call_tool("echo", {"text": "x"}, timeout=0.05)
    await server.send({"jsonrpc": "2.0", "id": held[0]["id"], "result": {"content": []}})
    server.interceptor = None
    assert (await session.call_tool("echo", {"text": "still works"})).text == "still works"


async def test_ping_round_trip(session):
    await session.ping()


async def test_server_ping_is_answered(session, server):
    await server.send({"jsonrpc": "2.0", "id": "srv-1", "method": "ping"})
    await eventually(lambda: any(m.get("id") == "srv-1" for m in server.received))
    reply = next(m for m in server.received if m.get("id") == "srv-1")
    assert reply["result"] == {}


async def test_unsupported_server_request_gets_method_not_found(session, server):
    await server.send({"jsonrpc": "2.0", "id": "s2", "method": "sampling/createMessage"})
    await eventually(lambda: any(m.get("id") == "s2" for m in server.received))
    reply = next(m for m in server.received if m.get("id") == "s2")
    assert reply["error"]["code"] == -32601


async def test_notifications_reach_the_handler(session, server):
    seen: list[tuple[str, Any]] = []
    session.on_notification = lambda method, params: seen.append((method, params))
    await server.send(
        {"jsonrpc": "2.0", "method": "notifications/message", "params": {"level": "info"}}
    )
    await eventually(lambda: bool(seen))
    assert seen == [("notifications/message", {"level": "info"})]


async def test_failing_notification_handler_does_not_kill_the_session(session, server):
    def boom(method: str, params: Any) -> None:
        raise RuntimeError("handler bug")

    session.on_notification = boom
    await server.send({"jsonrpc": "2.0", "method": "notifications/x"})
    await asyncio.sleep(0.02)
    await session.ping()


async def test_garbage_messages_are_dropped(session, server):
    await server.send({"not": "jsonrpc"})
    await server.send({"jsonrpc": "2.0", "id": 9999, "result": {}})  # unknown id
    await server.send({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "x"}})
    await session.ping()


async def test_closing_fails_pending_requests(session, server):
    server.interceptor = lambda m: m.get("method") == "tools/call"
    pending = asyncio.create_task(session.call_tool("echo", {"text": "x"}))
    await eventually(lambda: any(m.get("method") == "tools/call" for m in server.received))
    await session.close()
    with pytest.raises(SessionClosed):
        await pending


async def test_requests_after_close_raise(session):
    await session.close()
    with pytest.raises(SessionClosed):
        await session.call_tool("add", {"a": 1, "b": 1})
    with pytest.raises(SessionClosed):
        await session.notify("notifications/x")
    await session.close()  # idempotent


async def test_server_disconnect_fails_pending_and_future_requests(session, server):
    server.interceptor = lambda m: m.get("method") == "tools/call"
    pending = asyncio.create_task(session.call_tool("echo", {"text": "x"}))
    await eventually(lambda: any(m.get("method") == "tools/call" for m in server.received))
    await server.transport.close()
    with pytest.raises(SessionClosed, match="closed by the server"):
        await pending
    assert not session.is_connected
    with pytest.raises(SessionClosed):
        await session.ping()


async def test_transport_failure_is_reported_to_pending_requests():
    class Flaky(MemoryTransport):
        async def messages(self):  # type: ignore[override]
            async for message in super().messages():
                yield message
                if message.get("id") == 1:
                    raise TransportError("link dropped")

    a_to_b: asyncio.Queue[Any] = asyncio.Queue()
    b_to_a: asyncio.Queue[Any] = asyncio.Queue()
    client = Flaky(inbox=b_to_a, outbox=a_to_b)
    server_side = MemoryTransport(inbox=a_to_b, outbox=b_to_a)
    server = MemoryServer(server_side)
    try:
        with pytest.raises((TransportError, SessionClosed)):
            await ClientSession(client, request_timeout=1.0).start()
    finally:
        await server.stop()
