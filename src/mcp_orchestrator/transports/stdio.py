"""stdio transport: launch the server as a subprocess, newline-delimited JSON on its pipes."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

from ..errors import TransportError
from ..jsonrpc import dumps
from .base import Transport

log = logging.getLogger(__name__)

_LINE_LIMIT = 16 * 1024 * 1024  # a single message may be large (e.g. image content)


class StdioTransport(Transport):
    def __init__(
        self,
        command: str,
        args: Sequence[str] = (),
        *,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
        shutdown_timeout: float = 3.0,
    ) -> None:
        self._command = command
        self._args = list(args)
        self._env = {**os.environ, **(env or {})}
        self._cwd = cwd
        self._shutdown_timeout = shutdown_timeout
        self._proc: asyncio.subprocess.Process | None = None
        self._stderr_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        try:
            self._proc = await asyncio.create_subprocess_exec(
                self._command,
                *self._args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._env,
                cwd=self._cwd,
                limit=_LINE_LIMIT,
            )
        except (OSError, ValueError) as exc:
            raise TransportError(f"could not start '{self._command}': {exc}") from exc
        self._stderr_task = asyncio.create_task(self._drain_stderr(self._proc))

    async def _drain_stderr(self, proc: asyncio.subprocess.Process) -> None:
        # Take the process as an argument: close() clears self._proc while this task may still run.
        assert proc.stderr
        while line := await proc.stderr.readline():
            log.debug("[%s stderr] %s", self._command, line.decode(errors="replace").rstrip())

    async def send(self, payload: dict[str, Any]) -> None:
        if not self._proc or not self._proc.stdin or self._proc.stdin.is_closing():
            raise TransportError("stdio transport is not connected")
        try:
            self._proc.stdin.write((dumps(payload) + "\n").encode())
            await self._proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError) as exc:
            raise TransportError(f"server process closed its input: {exc}") from exc

    async def messages(self) -> AsyncIterator[dict[str, Any]]:
        if not self._proc or not self._proc.stdout:
            raise TransportError("stdio transport is not connected")
        stdout = self._proc.stdout
        while True:
            try:
                line = await stdout.readline()
            except (ValueError, asyncio.LimitOverrunError) as exc:
                raise TransportError(f"message exceeded {_LINE_LIMIT} bytes: {exc}") from exc
            if not line:
                return  # EOF: the process exited or closed stdout
            text = line.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            try:
                decoded = json.loads(text)
            except json.JSONDecodeError:
                # Some servers print logs to stdout. Skip anything that is not JSON.
                log.warning("ignoring non-JSON line from server stdout: %.120s", text)
                continue
            if isinstance(decoded, dict):
                yield decoded
            else:
                log.warning("ignoring non-object JSON from server: %.120s", text)

    async def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        if proc.stdin and not proc.stdin.is_closing():
            proc.stdin.close()
        try:
            await asyncio.wait_for(proc.wait(), self._shutdown_timeout)
        except TimeoutError:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), self._shutdown_timeout)
            except TimeoutError:
                proc.kill()
                await proc.wait()
        if self._stderr_task:
            self._stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._stderr_task
