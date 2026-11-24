"""Integration tests against a server built with the official MCP Python SDK.

They prove the client interoperates with an independent implementation of the protocol,
over both transports.
"""

from __future__ import annotations

import asyncio
import socket
import subprocess
import sys
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from mcp_orchestrator import (
    ClientSession,
    InvalidToolArguments,
    RecordingTracer,
    SSETransport,
    StdioTransport,
    ToolExecutionError,
    ToolRegistry,
)
from mcp_orchestrator.config import ServerConfig

pytest.importorskip("mcp.server.mcpserver")
pytestmark = pytest.mark.integration

DEMO = str(Path(__file__).parent.parent / "examples" / "demo_server.py")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(params=["stdio", "sse"])
async def sdk_session(request) -> AsyncIterator[ClientSession]:
    if request.param == "stdio":
        async with ClientSession(StdioTransport(sys.executable, [DEMO]), request_timeout=20) as s:
            yield s
        return

    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, DEMO, "--transport", "sse", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(300):
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                    break
            except OSError:
                await asyncio.sleep(0.05)
        else:
            pytest.fail("SDK SSE server did not start")
        async with ClientSession(
            SSETransport(f"http://127.0.0.1:{port}/sse"), request_timeout=20
        ) as s:
            yield s
    finally:
        proc.terminate()
        proc.wait(timeout=10)


async def test_handshake_with_the_official_sdk(sdk_session):
    assert sdk_session.server_info.name == "demo"
    assert sdk_session.protocol_version in ("2025-06-18", "2025-03-26", "2024-11-05")
    assert "tools" in sdk_session.server_capabilities


async def test_tools_are_discovered(sdk_session):
    names = {t.name for t in await sdk_session.list_tools()}
    assert {"add", "word_count", "utc_now", "fail"} <= names


async def test_tool_calls_return_typed_results(sdk_session):
    assert (await sdk_session.call_tool("add", {"a": 2, "b": 40})).text == "42"
    assert (await sdk_session.call_tool("word_count", {"text": "one two three"})).text == "3"
    assert (await sdk_session.call_tool("utc_now")).text.startswith("20")


async def test_tool_exceptions_come_back_as_is_error(sdk_session):
    result = await sdk_session.call_tool("fail", {"reason": "boom"})
    assert result.is_error is True


async def test_concurrent_calls(sdk_session):
    results = await asyncio.gather(
        *(sdk_session.call_tool("add", {"a": i, "b": 1}) for i in range(25))
    )
    assert [r.text for r in results] == [str(i + 1) for i in range(25)]


async def test_registry_end_to_end_with_tracing_and_validation():
    tracer = RecordingTracer()
    config = ServerConfig(name="demo", transport="stdio", command=sys.executable, args=[DEMO])
    async with ToolRegistry(tracer=tracer) as registry:
        await registry.connect_all([config])
        assert not registry.failures
        assert (await registry.call("demo__add", {"a": 1, "b": 2})).text == "3"
        with pytest.raises(InvalidToolArguments):
            await registry.call("demo__add", {"a": "1", "b": 2})
        with pytest.raises(ToolExecutionError):
            await registry.call("demo__fail", {})
    assert [s.level for s in tracer.spans] == ["DEFAULT", "ERROR", "ERROR"]
