"""Tracing hooks. Every tool call is wrapped in a span; Langfuse is one backend.

Tracing must never break a tool call, so the Langfuse backend swallows its own errors.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol

log = logging.getLogger(__name__)


class Span(Protocol):
    def update(self, **fields: Any) -> None:
        """Attach ``output``, ``metadata``, ``level`` or ``status_message`` to the span."""


class Tracer(Protocol):
    def span(
        self,
        name: str,
        *,
        kind: str = "tool",
        input: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> AbstractContextManager[Span]: ...

    def flush(self) -> None: ...


class NoopSpan:
    def update(self, **fields: Any) -> None:
        return None


class NoopTracer:
    @contextmanager
    def span(self, name: str, **kwargs: Any) -> Iterator[Span]:
        yield NoopSpan()

    def flush(self) -> None:
        return None


@dataclass
class RecordedSpan:
    name: str
    kind: str
    input: Any
    metadata: dict[str, Any]
    updates: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    @property
    def output(self) -> Any:
        for update in reversed(self.updates):
            if "output" in update:
                return update["output"]
        return None

    @property
    def level(self) -> str | None:
        for update in reversed(self.updates):
            if "level" in update:
                return update["level"]
        return None


class _RecordingSpan:
    def __init__(self, record: RecordedSpan) -> None:
        self._record = record

    def update(self, **fields: Any) -> None:
        self._record.updates.append(fields)


class RecordingTracer:
    """Keeps spans in memory. Used by tests and useful for debugging."""

    def __init__(self) -> None:
        self.spans: list[RecordedSpan] = []

    @contextmanager
    def span(
        self,
        name: str,
        *,
        kind: str = "tool",
        input: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[Span]:
        record = RecordedSpan(name=name, kind=kind, input=input, metadata=dict(metadata or {}))
        self.spans.append(record)
        try:
            yield _RecordingSpan(record)
        except BaseException as exc:
            record.error = f"{type(exc).__name__}: {exc}"
            raise

    def flush(self) -> None:
        return None


class _LangfuseSpan:
    def __init__(self, observation: Any) -> None:
        self._observation = observation

    def update(self, **fields: Any) -> None:
        try:
            self._observation.update(**fields)
        except Exception:  # tracing must not break the call
            log.warning("langfuse span update failed", exc_info=True)


class LangfuseTracer:
    """Sends one Langfuse observation per call (Langfuse Python SDK v3, OpenTelemetry-based).

    Credentials come from the standard ``LANGFUSE_PUBLIC_KEY``, ``LANGFUSE_SECRET_KEY`` and
    ``LANGFUSE_HOST`` environment variables unless a client is passed in.
    """

    def __init__(self, client: Any | None = None) -> None:
        if client is None:
            from langfuse import Langfuse  # optional dependency: pip install '.[langfuse]'

            client = Langfuse()
        self._client = client

    @contextmanager
    def span(
        self,
        name: str,
        *,
        kind: str = "tool",
        input: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[Span]:
        try:
            manager = self._client.start_as_current_observation(
                name=name, as_type=kind, input=input, metadata=metadata
            )
            observation = manager.__enter__()
        except Exception:
            log.warning("could not open a langfuse observation; continuing untraced", exc_info=True)
            yield NoopSpan()
            return

        span = _LangfuseSpan(observation)
        try:
            yield span
        except BaseException as exc:
            span.update(level="ERROR", status_message=f"{type(exc).__name__}: {exc}")
            self._close(manager, *sys.exc_info())
            raise
        else:
            self._close(manager, None, None, None)

    @staticmethod
    def _close(manager: Any, *exc_info: Any) -> None:
        try:
            manager.__exit__(*exc_info)
        except Exception:
            log.warning("could not close a langfuse observation", exc_info=True)

    def flush(self) -> None:
        try:
            self._client.flush()
        except Exception:
            log.warning("langfuse flush failed", exc_info=True)


def tracer_from_env(env: Mapping[str, str] | None = None) -> Tracer:
    """A :class:`LangfuseTracer` when Langfuse credentials are set, otherwise a no-op tracer."""
    env = os.environ if env is None else env
    if not (env.get("LANGFUSE_PUBLIC_KEY") and env.get("LANGFUSE_SECRET_KEY")):
        return NoopTracer()
    try:
        return LangfuseTracer()
    except ImportError:
        log.warning("Langfuse keys are set but the 'langfuse' package is not installed")
        return NoopTracer()
