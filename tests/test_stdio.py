from __future__ import annotations

import sys

import pytest

from mcp_orchestrator import ClientSession, StdioTransport
from mcp_orchestrator.errors import SessionClosed, TransportError
from mcp_orchestrator.models import ImageContent, TextContent, UnknownContent

from .conftest import STDIO_SERVER


async def test_end_to_end_over_a_real_subprocess(stdio_session):
    assert stdio_session.server_info.name == "mini"
    tools = await stdio_session.list_tools()
    assert [t.name for t in tools][:2] == ["echo", "add"]
    result = await stdio_session.call_tool("add", {"a": 20, "b": 22})
    assert result.text == "42"


async def test_non_json_stdout_noise_is_ignored(stdio_session):
    # The fixture prints a banner before speaking JSON; getting here means it was skipped.
    assert (await stdio_session.call_tool("echo", {"text": "ok"})).text == "ok"


async def test_mixed_content_blocks(stdio_session):
    result = await stdio_session.call_tool("rich")
    assert [type(b) for b in result.content] == [
        TextContent,
        ImageContent,
        UnknownContent,
        TextContent,
    ]
    assert result.text == "hello\nworld"


async def test_large_messages_survive_the_pipe(stdio_session):
    payload = "x" * 300_000  # larger than asyncio's default 64 KiB line limit
    assert (await stdio_session.call_tool("echo", {"text": payload})).text == payload


async def test_many_concurrent_calls(stdio_session):
    import asyncio

    results = await asyncio.gather(
        *(stdio_session.call_tool("add", {"a": i, "b": i}) for i in range(50))
    )
    assert [r.text for r in results] == [str(2 * i) for i in range(50)]


async def test_server_crash_surfaces_as_session_closed():
    async with ClientSession(StdioTransport(sys.executable, [STDIO_SERVER])) as session:
        with pytest.raises(SessionClosed):
            await session.call_tool("crash")
        assert not session.is_connected


async def test_missing_executable_raises_transport_error():
    with pytest.raises(TransportError, match="could not start"):
        await ClientSession(StdioTransport("/no/such/binary")).start()


async def test_send_before_start_raises():
    with pytest.raises(TransportError, match="not connected"):
        await StdioTransport(sys.executable).send({"jsonrpc": "2.0"})


async def test_messages_before_start_raises():
    with pytest.raises(TransportError, match="not connected"):
        async for _ in StdioTransport(sys.executable).messages():
            pass


async def test_close_stops_the_process():
    transport = StdioTransport(sys.executable, [STDIO_SERVER])
    await transport.start()
    proc = transport._proc
    assert proc is not None and proc.returncode is None
    await transport.close()
    assert proc.returncode is not None
    await transport.close()  # idempotent


async def test_close_terminates_a_process_that_ignores_stdin_closing():
    transport = StdioTransport(
        sys.executable, ["-c", "import time; time.sleep(60)"], shutdown_timeout=0.2
    )
    await transport.start()
    proc = transport._proc
    await transport.close()
    assert proc.returncode is not None


async def test_close_kills_a_process_that_ignores_sigterm():
    code = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
    transport = StdioTransport(sys.executable, ["-c", code], shutdown_timeout=0.3)
    await transport.start()
    import asyncio

    await asyncio.sleep(0.3)  # let the child install its signal handler
    proc = transport._proc
    await transport.close()
    assert proc.returncode is not None


async def test_environment_is_passed_to_the_child():
    code = (
        "import os, json, sys; "
        "print(json.dumps({'jsonrpc': '2.0', 'method': 'notifications/env', "
        "'params': {'value': os.environ['MCP_TEST_VALUE']}}), flush=True)"
    )
    transport = StdioTransport(sys.executable, ["-c", code], env={"MCP_TEST_VALUE": "hello"})
    await transport.start()
    try:
        message = await anext(transport.messages())
        assert message["params"]["value"] == "hello"
    finally:
        await transport.close()


async def test_non_object_json_lines_are_skipped():
    code = "print('[1, 2, 3]'); print('{\"ok\": true}', flush=True)"
    transport = StdioTransport(sys.executable, ["-c", code])
    await transport.start()
    try:
        assert await anext(transport.messages()) == {"ok": True}
    finally:
        await transport.close()
