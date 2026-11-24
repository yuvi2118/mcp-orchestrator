from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from mcp_orchestrator import (
    ClientSession,
    InvalidToolArguments,
    RecordingTracer,
    RequestTimeout,
    ToolExecutionError,
    ToolNotFound,
    ToolRegistry,
)
from mcp_orchestrator.config import ServerConfig
from mcp_orchestrator.registry import qualified_name, validate_arguments

from .conftest import MemoryServer, stdio_config
from .fixtures.mini_server import TOOLS


@pytest.fixture
async def tracer() -> RecordingTracer:
    return RecordingTracer()


@pytest.fixture
async def registry(tracer) -> AsyncIterator[ToolRegistry]:
    async with ToolRegistry(tracer=tracer) as reg:
        await reg.connect_all([stdio_config("alpha"), stdio_config("beta")])
        assert not reg.failures
        yield reg


# --- naming ----------------------------------------------------------------------------------


def test_qualified_name_basic():
    assert qualified_name("srv", "tool") == "srv__tool"


def test_qualified_name_replaces_invalid_characters():
    assert qualified_name("my server", "do.thing/now") == "my_server__do_thing_now"


def test_qualified_name_is_limited_to_64_chars_and_stays_unique():
    a = qualified_name("s", "x" * 100 + "a")
    b = qualified_name("s", "x" * 100 + "b")
    assert len(a) <= 64 and len(b) <= 64 and a != b


# --- argument validation ----------------------------------------------------------------------


def test_validate_arguments_ok():
    schema = TOOLS[1]["inputSchema"]
    assert validate_arguments(schema, {"a": 1, "b": 2}) == []


def test_validate_arguments_reports_each_problem_with_its_path():
    errors = validate_arguments(TOOLS[1]["inputSchema"], {"a": "x"})
    assert any(e.startswith("a:") and "integer" in e for e in errors)
    assert any("'b' is a required property" in e for e in errors)


def test_validate_arguments_skips_broken_schemas():
    assert validate_arguments({"type": "not-a-type"}, {"anything": 1}) == []


def test_validate_arguments_honours_additional_properties():
    errors = validate_arguments(TOOLS[0]["inputSchema"], {"text": "x", "extra": 1})
    assert errors and "extra" in errors[0]


# --- connecting and catalogue ------------------------------------------------------------------


async def test_tools_from_every_server_are_namespaced(registry):
    names = {t.name for t in registry.tools()}
    assert {"alpha__echo", "beta__echo", "alpha__add", "beta__add"} <= names
    assert len(registry.tools("alpha")) == 5
    assert registry.get("alpha__add").server == "alpha"


async def test_anthropic_tool_format(registry):
    tool = next(t for t in registry.to_anthropic_tools() if t["name"] == "alpha__add")
    assert tool["description"] == "Add two integers."
    assert tool["input_schema"]["required"] == ["a", "b"]
    assert set(tool) == {"name", "description", "input_schema"}


async def test_status_reports_connection_state(registry):
    status = registry.status()
    assert status["alpha"] == {"connected": True, "tools": 5}


async def test_one_failing_server_does_not_stop_the_others(tracer):
    bad = ServerConfig(name="broken", transport="stdio", command="/no/such/binary")
    async with ToolRegistry(tracer=tracer) as reg:
        await reg.connect_all([stdio_config("good"), bad])
        assert "good__add" in {t.name for t in reg.tools()}
        assert "TransportError" in reg.failures["broken"]
        status = reg.status()
        assert status["good"]["connected"] is True
        assert status["broken"]["connected"] is False and "error" in status["broken"]


async def test_duplicate_server_names_are_rejected(registry):
    with pytest.raises(ValueError, match="already registered"):
        await registry.add_server(stdio_config("alpha"))


async def test_server_that_fails_during_tool_listing_is_closed_again(tracer, memory_pair):
    transport, server = memory_pair
    server.interceptor = lambda m: m.get("method") == "tools/list"  # never answers
    session = ClientSession(transport, request_timeout=0.1)
    await session.start()
    async with ToolRegistry(tracer=tracer) as reg:
        with pytest.raises(RequestTimeout):
            await reg.add_session("slow", session)
        assert reg.tools() == []


async def test_name_collisions_after_sanitising_get_distinct_names(memory_pair):
    transport, server = memory_pair

    def two_lookalikes(message: dict[str, Any]) -> bool:
        if message.get("method") == "tools/list":
            tools = [
                {"name": "a.b", "inputSchema": {"type": "object"}},
                {"name": "a_b", "inputSchema": {"type": "object"}},
            ]
            asyncio.get_running_loop().create_task(
                server.send({"jsonrpc": "2.0", "id": message["id"], "result": {"tools": tools}})
            )
            return True
        return False

    server.interceptor = two_lookalikes
    async with ToolRegistry() as reg:
        session = ClientSession(transport)
        await session.start()
        installed = await reg.add_session("s", session)
        assert len({t.name for t in installed}) == 2


async def test_tools_list_changed_triggers_a_refresh(memory_pair):
    transport, server = memory_pair
    state = {"extra": False}

    def maybe_extend(message: dict[str, Any]) -> bool:
        if message.get("method") == "tools/list" and state["extra"]:
            tools = [*TOOLS, {"name": "brand_new", "inputSchema": {"type": "object"}}]
            asyncio.get_running_loop().create_task(
                server.send({"jsonrpc": "2.0", "id": message["id"], "result": {"tools": tools}})
            )
            return True
        return False

    server.interceptor = maybe_extend
    async with ToolRegistry() as reg:
        session = ClientSession(transport)
        await session.start()
        await reg.add_session("live", session)
        assert "live__brand_new" not in {t.name for t in reg.tools()}

        state["extra"] = True
        await server.send({"jsonrpc": "2.0", "method": "notifications/tools/list_changed"})
        for _ in range(200):
            if "live__brand_new" in {t.name for t in reg.tools()}:
                break
            await asyncio.sleep(0.01)
        assert "live__brand_new" in {t.name for t in reg.tools()}


async def test_unrelated_notifications_do_not_refresh(memory_pair):
    transport, server = memory_pair
    async with ToolRegistry() as reg:
        session = ClientSession(transport)
        await session.start()
        await reg.add_session("s", session)
        before = sum(1 for m in server.received if m.get("method") == "tools/list")
        await server.send({"jsonrpc": "2.0", "method": "notifications/message"})
        await asyncio.sleep(0.05)
        after = sum(1 for m in server.received if m.get("method") == "tools/list")
        assert before == after


# --- calling ----------------------------------------------------------------------------------


async def test_call_routes_to_the_right_server(registry):
    result = await registry.call("beta__add", {"a": 1, "b": 2})
    assert result.text == "3"


async def test_unknown_tool(registry):
    with pytest.raises(ToolNotFound, match="nope"):
        await registry.call("alpha__nope", {})


async def test_invalid_arguments_never_reach_the_server(registry):
    with pytest.raises(InvalidToolArguments) as info:
        await registry.call("alpha__add", {"a": "one"})
    assert info.value.tool == "alpha__add"
    assert any("required property" in e for e in info.value.errors)


async def test_tool_errors_raise_by_default(registry):
    with pytest.raises(ToolExecutionError, match="something went wrong"):
        await registry.call("alpha__fail")


async def test_tool_errors_can_be_returned_instead(registry):
    result = await registry.call("alpha__fail", raise_on_error=False)
    assert result.is_error and result.text == "something went wrong"


async def test_missing_arguments_default_to_empty_object(registry):
    assert (await registry.call("alpha__rich")).text == "hello\nworld"


async def test_timeouts_propagate(tracer, memory_pair):
    transport, server = memory_pair
    async with ToolRegistry(tracer=tracer) as reg:
        session = ClientSession(transport)
        await session.start()
        await reg.add_session("s", session)
        server.interceptor = lambda m: m.get("method") == "tools/call"
        with pytest.raises(RequestTimeout):
            await reg.call("s__echo", {"text": "x"}, timeout=0.05)
    span = tracer.spans[0]
    assert span.level == "ERROR" and span.updates[-1]["status_message"].startswith("RequestTimeout")


# --- tracing ----------------------------------------------------------------------------------


async def test_successful_call_is_traced(registry, tracer):
    await registry.call("alpha__add", {"a": 2, "b": 3})
    (span,) = tracer.spans
    assert span.name == "mcp.call_tool.alpha__add"
    assert span.kind == "tool"
    assert span.input == {"a": 2, "b": 3}
    assert span.output == {"result": 5}
    assert span.level == "DEFAULT"
    final = span.updates[-1]["metadata"]
    assert final["server"] == "alpha" and final["tool"] == "add" and final["is_error"] is False
    assert final["duration_ms"] >= 0
    assert span.error is None


async def test_tool_error_marks_the_span_as_error(registry, tracer):
    await registry.call("alpha__fail", raise_on_error=False)
    span = tracer.spans[0]
    assert span.level == "ERROR"
    assert span.updates[-1]["metadata"]["is_error"] is True


async def test_invalid_arguments_are_traced_too(registry, tracer):
    with pytest.raises(InvalidToolArguments):
        await registry.call("alpha__add", {})
    span = tracer.spans[0]
    assert span.level == "ERROR"
    assert "invalid arguments" in span.updates[-1]["status_message"]
    assert span.error and "InvalidToolArguments" in span.error


async def test_every_concurrent_call_gets_its_own_span(registry, tracer):
    await asyncio.gather(*(registry.call("alpha__add", {"a": i, "b": 1}) for i in range(10)))
    assert len(tracer.spans) == 10
    assert sorted(s.output["result"] for s in tracer.spans) == list(range(1, 11))


async def test_aclose_flushes_the_tracer_and_closes_sessions():
    flushed: list[bool] = []

    class Flushing(RecordingTracer):
        def flush(self) -> None:
            flushed.append(True)

    reg = ToolRegistry(tracer=Flushing())
    await reg.connect_all([stdio_config("one")])
    session = reg._sessions["one"]
    await reg.aclose()
    assert flushed == [True]
    assert not session.is_connected
    assert reg.tools() == []


async def test_memory_server_helper_is_usable_directly(memory_pair):
    transport, server = memory_pair
    assert isinstance(server, MemoryServer)
    async with ClientSession(transport) as session:
        assert (await session.call_tool("echo", {"text": "hi"})).text == "hi"
