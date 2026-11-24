from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest

from mcp_orchestrator import AgentRunner, ClientSession, RecordingTracer, ToolRegistry
from mcp_orchestrator.agent import DEFAULT_MODEL, _block_to_param


def text(value: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=value)


def tool_use(call_id: str, name: str, arguments: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=call_id, name=name, input=arguments)


def reply(*blocks: SimpleNamespace, stop: str = "end_turn") -> SimpleNamespace:
    return SimpleNamespace(content=list(blocks), stop_reason=stop)


class ScriptedClient:
    """Stands in for anthropic.AsyncAnthropic: returns scripted replies and records requests."""

    def __init__(self, *replies: SimpleNamespace) -> None:
        self._replies = list(replies)
        self.requests: list[dict[str, Any]] = []
        self.messages = SimpleNamespace(create=self._create)

    async def _create(self, **kwargs: Any) -> SimpleNamespace:
        # Snapshot: the runner keeps appending to the same list.
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self._replies.pop(0)


@pytest.fixture
async def registry(memory_pair) -> AsyncIterator[ToolRegistry]:
    transport, _ = memory_pair
    async with ToolRegistry() as reg:
        session = ClientSession(transport)
        await session.start()
        await reg.add_session("mini", session)
        yield reg


async def test_plain_answer_without_tools(registry):
    client = ScriptedClient(reply(text("Hello!")))
    result = await AgentRunner(registry, client).run("hi")
    assert result.text == "Hello!" and result.turns == 1 and result.tool_calls == []
    request = client.requests[0]
    assert request["model"] == DEFAULT_MODEL
    assert request["messages"] == [{"role": "user", "content": "hi"}]
    assert {t["name"] for t in request["tools"]} >= {"mini__add", "mini__echo"}


async def test_tool_use_loop(registry):
    client = ScriptedClient(
        reply(text("Let me add."), tool_use("t1", "mini__add", {"a": 2, "b": 3}), stop="tool_use"),
        reply(text("The answer is 5.")),
    )
    result = await AgentRunner(registry, client).run("what is 2+3?")

    assert result.text == "The answer is 5."
    assert result.turns == 2 and result.stop_reason == "end_turn"
    (call,) = result.tool_calls
    assert (call.name, call.arguments, call.output, call.is_error) == (
        "mini__add",
        {"a": 2, "b": 3},
        "5",
        False,
    )

    second_request = client.requests[1]["messages"]
    assert second_request[1]["role"] == "assistant"
    assert second_request[1]["content"][1] == {
        "type": "tool_use",
        "id": "t1",
        "name": "mini__add",
        "input": {"a": 2, "b": 3},
    }
    assert second_request[2] == {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "5", "is_error": False}
        ],
    }


async def test_parallel_tool_calls_are_all_answered(registry):
    client = ScriptedClient(
        reply(
            tool_use("a", "mini__add", {"a": 1, "b": 1}),
            tool_use("b", "mini__echo", {"text": "hi"}),
            stop="tool_use",
        ),
        reply(text("done")),
    )
    result = await AgentRunner(registry, client).run("go")
    assert [c.output for c in result.tool_calls] == ["2", "hi"]
    tool_results = client.requests[1]["messages"][2]["content"]
    assert [r["tool_use_id"] for r in tool_results] == ["a", "b"]


async def test_invalid_arguments_are_reported_to_the_model_not_raised(registry):
    client = ScriptedClient(
        reply(tool_use("t1", "mini__add", {"a": "oops"}), stop="tool_use"),
        reply(text("Sorry, retrying.")),
    )
    result = await AgentRunner(registry, client).run("go")
    (call,) = result.tool_calls
    assert call.is_error and "InvalidToolArguments" in call.output
    sent = client.requests[1]["messages"][2]["content"][0]
    assert sent["is_error"] is True


async def test_unknown_tools_are_reported_to_the_model(registry):
    client = ScriptedClient(
        reply(tool_use("t1", "mini__nope", {}), stop="tool_use"), reply(text("ok"))
    )
    result = await AgentRunner(registry, client).run("go")
    assert result.tool_calls[0].is_error and "ToolNotFound" in result.tool_calls[0].output


async def test_tool_reported_errors_are_marked(registry):
    client = ScriptedClient(
        reply(tool_use("t1", "mini__fail", {}), stop="tool_use"), reply(text("ok"))
    )
    result = await AgentRunner(registry, client).run("go")
    assert result.tool_calls[0].is_error and result.tool_calls[0].output == "something went wrong"


async def test_stops_after_max_turns(registry):
    looping = [
        reply(text("thinking"), tool_use(f"t{i}", "mini__echo", {"text": "x"}), stop="tool_use")
        for i in range(5)
    ]
    client = ScriptedClient(*looping)
    result = await AgentRunner(registry, client, max_turns=3).run("go")
    assert result.stop_reason == "max_turns" and result.turns == 3
    assert len(result.tool_calls) == 3 and len(client.requests) == 3
    assert result.text == "thinking"


async def test_system_prompt_model_and_limits_are_forwarded(registry):
    client = ScriptedClient(reply(text("ok")))
    await AgentRunner(registry, client, model="my-model", system="be brief", max_tokens=77).run("x")
    request = client.requests[0]
    assert (request["model"], request["system"], request["max_tokens"]) == (
        "my-model",
        "be brief",
        77,
    )


async def test_model_can_come_from_the_environment(registry, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", "env-model")
    client = ScriptedClient(reply(text("ok")))
    await AgentRunner(registry, client).run("x")
    assert client.requests[0]["model"] == "env-model"


async def test_no_tools_key_when_registry_is_empty():
    client = ScriptedClient(reply(text("ok")))
    async with ToolRegistry() as empty:
        await AgentRunner(empty, client).run("x")
    assert "tools" not in client.requests[0]


async def test_agent_run_is_traced(registry):
    tracer = RecordingTracer()
    client = ScriptedClient(reply(text("hi there")))
    await AgentRunner(registry, client, tracer=tracer).run("hello")
    (span,) = tracer.spans
    assert (span.name, span.kind, span.input, span.output) == (
        "agent.run",
        "agent",
        "hello",
        "hi there",
    )


async def test_max_turns_run_is_traced_as_a_warning(registry):
    tracer = RecordingTracer()
    client = ScriptedClient(reply(tool_use("t", "mini__echo", {"text": "x"}), stop="tool_use"))
    await AgentRunner(registry, client, max_turns=1, tracer=tracer).run("go")
    assert tracer.spans[0].level == "WARNING"


def test_block_conversion_accepts_dicts_and_objects_and_drops_unknown_types():
    assert _block_to_param({"type": "text", "text": "a"}) == {"type": "text", "text": "a"}
    assert _block_to_param(text("b")) == {"type": "text", "text": "b"}
    assert _block_to_param({"type": "tool_use", "id": "i", "name": "n", "input": {"k": 1}})[
        "input"
    ] == {"k": 1}
    assert _block_to_param(SimpleNamespace(type="thinking")) == {"type": "text", "text": ""}
