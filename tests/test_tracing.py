from __future__ import annotations

import logging
from typing import Any

import pytest

from mcp_orchestrator import LangfuseTracer, NoopTracer, RecordingTracer, tracer_from_env
from mcp_orchestrator import tracing as tracing_module


class FakeObservation:
    def __init__(self, fail_update: bool = False) -> None:
        self.updates: list[dict[str, Any]] = []
        self.fail_update = fail_update

    def update(self, **fields: Any) -> None:
        if self.fail_update:
            raise RuntimeError("update failed")
        self.updates.append(fields)


class FakeManager:
    def __init__(self, observation: FakeObservation, fail_exit: bool = False) -> None:
        self.observation = observation
        self.fail_exit = fail_exit
        self.exit_args: tuple[Any, ...] | None = None

    def __enter__(self) -> FakeObservation:
        return self.observation

    def __exit__(self, *args: Any) -> None:
        self.exit_args = args
        if self.fail_exit:
            raise RuntimeError("exit failed")


class FakeLangfuse:
    def __init__(
        self,
        *,
        fail_open: bool = False,
        fail_update: bool = False,
        fail_exit: bool = False,
        fail_flush: bool = False,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self.managers: list[FakeManager] = []
        self.flushed = 0
        self._fail_open, self._fail_update = fail_open, fail_update
        self._fail_exit, self._fail_flush = fail_exit, fail_flush

    def start_as_current_observation(self, **kwargs: Any) -> FakeManager:
        if self._fail_open:
            raise RuntimeError("cannot open")
        self.calls.append(kwargs)
        manager = FakeManager(FakeObservation(self._fail_update), self._fail_exit)
        self.managers.append(manager)
        return manager

    def flush(self) -> None:
        if self._fail_flush:
            raise RuntimeError("flush failed")
        self.flushed += 1


def test_noop_tracer_does_nothing():
    tracer = NoopTracer()
    with tracer.span("x", input=1) as span:
        span.update(output="y")
    tracer.flush()


def test_recording_tracer_records_updates_and_properties():
    tracer = RecordingTracer()
    with tracer.span("op", kind="agent", input={"a": 1}, metadata={"m": 1}) as span:
        span.update(output="first")
        span.update(output="second", level="WARNING")
    (record,) = tracer.spans
    assert (record.name, record.kind, record.input, record.metadata) == (
        "op",
        "agent",
        {"a": 1},
        {"m": 1},
    )
    assert record.output == "second" and record.level == "WARNING" and record.error is None


def test_recording_tracer_without_updates_has_no_output_or_level():
    tracer = RecordingTracer()
    with tracer.span("op"):
        pass
    assert tracer.spans[0].output is None and tracer.spans[0].level is None


def test_recording_tracer_captures_exceptions_and_reraises():
    tracer = RecordingTracer()
    with pytest.raises(ValueError), tracer.span("op"):
        raise ValueError("bad")
    assert tracer.spans[0].error == "ValueError: bad"


def test_langfuse_tracer_opens_an_observation_with_the_right_arguments():
    client = FakeLangfuse()
    tracer = LangfuseTracer(client)
    with tracer.span(
        "mcp.call_tool.x", kind="tool", input={"a": 1}, metadata={"server": "s"}
    ) as span:
        span.update(output="done", level="DEFAULT")
    assert client.calls == [
        {
            "name": "mcp.call_tool.x",
            "as_type": "tool",
            "input": {"a": 1},
            "metadata": {"server": "s"},
        }
    ]
    manager = client.managers[0]
    assert manager.observation.updates == [{"output": "done", "level": "DEFAULT"}]
    assert manager.exit_args == (None, None, None)


def test_langfuse_tracer_records_exceptions_then_reraises():
    client = FakeLangfuse()
    with pytest.raises(RuntimeError, match="tool blew up"), LangfuseTracer(client).span("op"):
        raise RuntimeError("tool blew up")
    manager = client.managers[0]
    assert manager.observation.updates[-1]["level"] == "ERROR"
    assert "tool blew up" in manager.observation.updates[-1]["status_message"]
    assert manager.exit_args[0] is RuntimeError


def test_langfuse_tracer_never_breaks_the_traced_code_when_langfuse_is_down(caplog):
    client = FakeLangfuse(fail_open=True)
    ran = []
    with caplog.at_level(logging.WARNING), LangfuseTracer(client).span("op") as span:
        span.update(output="ignored")
        ran.append(True)
    assert ran == [True]
    assert "continuing untraced" in caplog.text


def test_langfuse_span_update_failures_are_swallowed(caplog):
    client = FakeLangfuse(fail_update=True)
    with caplog.at_level(logging.WARNING), LangfuseTracer(client).span("op") as span:
        span.update(output="x")
    assert "span update failed" in caplog.text


def test_langfuse_close_failures_are_swallowed(caplog):
    client = FakeLangfuse(fail_exit=True)
    with caplog.at_level(logging.WARNING), LangfuseTracer(client).span("op"):
        pass
    assert "could not close" in caplog.text


def test_langfuse_flush():
    client = FakeLangfuse()
    LangfuseTracer(client).flush()
    assert client.flushed == 1


def test_langfuse_flush_failure_is_swallowed(caplog):
    with caplog.at_level(logging.WARNING):
        LangfuseTracer(FakeLangfuse(fail_flush=True)).flush()
    assert "flush failed" in caplog.text


def test_tracer_from_env_without_keys_is_noop():
    assert isinstance(tracer_from_env({}), NoopTracer)
    assert isinstance(tracer_from_env({"LANGFUSE_PUBLIC_KEY": "pk"}), NoopTracer)


def test_tracer_from_env_with_keys_builds_a_langfuse_tracer(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(tracing_module, "LangfuseTracer", lambda: sentinel)
    env = {"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk"}
    assert tracer_from_env(env) is sentinel


def test_tracer_from_env_falls_back_when_langfuse_is_not_installed(monkeypatch, caplog):
    def missing() -> None:
        raise ImportError("no langfuse")

    monkeypatch.setattr(tracing_module, "LangfuseTracer", missing)
    env = {"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk"}
    with caplog.at_level(logging.WARNING):
        assert isinstance(tracer_from_env(env), NoopTracer)
    assert "not installed" in caplog.text


def test_works_with_the_real_langfuse_sdk_api():
    """Guards against SDK API drift: uses the installed Langfuse client with tracing disabled."""
    langfuse = pytest.importorskip("langfuse")
    client = langfuse.Langfuse(
        public_key="pk-test", secret_key="sk-test", host="http://127.0.0.1:9", tracing_enabled=False
    )
    tracer = LangfuseTracer(client)
    with tracer.span(
        "mcp.call_tool.demo", kind="tool", input={"a": 1}, metadata={"server": "s"}
    ) as span:
        span.update(output="ok", level="DEFAULT", metadata={"duration_ms": 1.0})
    with pytest.raises(ValueError), tracer.span("failing", kind="agent") as span:
        raise ValueError("boom")
    tracer.flush()
