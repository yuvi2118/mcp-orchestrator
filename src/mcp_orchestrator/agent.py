"""A small tool-use loop: Claude decides which MCP tools to call, the registry executes them."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from .errors import MCPError
from .registry import ToolRegistry
from .tracing import NoopTracer, Tracer

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-5-5"


@dataclass
class ToolCallRecord:
    id: str
    name: str
    arguments: dict[str, Any]
    output: str
    is_error: bool


@dataclass
class AgentResult:
    text: str
    stop_reason: str | None
    turns: int
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)


class AgentRunner:
    """Runs the Messages API tool-use loop against a :class:`ToolRegistry`.

    ``client`` is anything shaped like ``anthropic.AsyncAnthropic`` (an object with an async
    ``messages.create``); the tests pass a scripted fake.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        client: Any,
        *,
        model: str | None = None,
        system: str | None = None,
        max_turns: int = 8,
        max_tokens: int = 1024,
        tracer: Tracer | None = None,
    ) -> None:
        self._registry = registry
        self._client = client
        self._model = model or os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)
        self._system = system
        self._max_turns = max_turns
        self._max_tokens = max_tokens
        self._tracer = tracer or NoopTracer()

    async def run(self, prompt: str) -> AgentResult:
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        records: list[ToolCallRecord] = []
        tools = self._registry.to_anthropic_tools()

        with self._tracer.span("agent.run", kind="agent", input=prompt) as span:
            for turn in range(1, self._max_turns + 1):
                params: dict[str, Any] = {
                    "model": self._model,
                    "max_tokens": self._max_tokens,
                    "messages": messages,
                }
                if tools:
                    params["tools"] = tools
                if self._system:
                    params["system"] = self._system
                response = await self._client.messages.create(**params)

                blocks = [_block_to_param(b) for b in response.content]
                messages.append({"role": "assistant", "content": blocks})

                tool_uses = [b for b in blocks if b["type"] == "tool_use"]
                if response.stop_reason != "tool_use" or not tool_uses:
                    text = "".join(b["text"] for b in blocks if b["type"] == "text")
                    span.update(output=text, metadata={"turns": turn, "tool_calls": len(records)})
                    return AgentResult(text, response.stop_reason, turn, records, messages)

                outcomes = await asyncio.gather(*(self._execute(b) for b in tool_uses))
                records.extend(record for record, _ in outcomes)
                messages.append({"role": "user", "content": [block for _, block in outcomes]})

            span.update(level="WARNING", status_message=f"stopped after {self._max_turns} turns")
            last_text = (
                "".join(b["text"] for b in messages[-2]["content"] if b["type"] == "text")
                if len(messages) >= 2
                else ""
            )
            return AgentResult(last_text, "max_turns", self._max_turns, records, messages)

    async def _execute(self, block: dict[str, Any]) -> tuple[ToolCallRecord, dict[str, Any]]:
        name, arguments = block["name"], block["input"]
        try:
            result = await self._registry.call(name, arguments, raise_on_error=False)
            output, is_error = result.text or "(no text output)", result.is_error
        except MCPError as exc:
            # Tell the model what went wrong so it can correct itself instead of crashing the run.
            output, is_error = f"{type(exc).__name__}: {exc}", True
        record = ToolCallRecord(block["id"], name, arguments, output, is_error)
        return record, {
            "type": "tool_result",
            "tool_use_id": block["id"],
            "content": output,
            "is_error": is_error,
        }


def _block_to_param(block: Any) -> dict[str, Any]:
    """Convert an SDK content block (or a plain dict) to a request parameter."""
    get = block.get if isinstance(block, dict) else lambda k, d=None: getattr(block, k, d)
    kind = get("type")
    if kind == "text":
        return {"type": "text", "text": get("text", "")}
    if kind == "tool_use":
        return {"type": "tool_use", "id": get("id"), "name": get("name"), "input": get("input", {})}
    # Other block types (thinking, ...) are not produced because extended thinking is off.
    return {"type": "text", "text": ""}
