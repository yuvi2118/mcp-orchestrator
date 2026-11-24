"""Server definitions: a pydantic model plus a TOML loader with ``${ENV_VAR}`` expansion."""

from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .errors import ServerConfigError
from .transports import SSETransport, StdioTransport, Transport

_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    transport: Literal["stdio", "sse"]
    # stdio
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: str | None = None
    # sse
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    # shared
    request_timeout: float = Field(default=30.0, gt=0)

    @model_validator(mode="after")
    def _check_transport_fields(self) -> ServerConfig:
        if self.transport == "stdio" and not self.command:
            raise ValueError("stdio servers need 'command'")
        if self.transport == "sse" and not self.url:
            raise ValueError("sse servers need 'url'")
        return self


def build_transport(config: ServerConfig) -> Transport:
    if config.transport == "stdio":
        assert config.command is not None
        return StdioTransport(config.command, config.args, env=config.env, cwd=config.cwd)
    assert config.url is not None
    return SSETransport(config.url, headers=config.headers)


def _expand(value: Any, env: dict[str, str]) -> Any:
    if isinstance(value, str):

        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in env:
                raise ServerConfigError(
                    f"config references undefined environment variable ${{{name}}}"
                )
            return env[name]

        return _ENV_REF.sub(replace, value)
    if isinstance(value, list):
        return [_expand(v, env) for v in value]
    if isinstance(value, dict):
        return {k: _expand(v, env) for k, v in value.items()}
    return value


def load_config(path: str | Path, env: dict[str, str] | None = None) -> list[ServerConfig]:
    """Read ``[[servers]]`` tables from a TOML file."""
    path = Path(path)
    try:
        data = tomllib.loads(path.read_text())
    except OSError as exc:
        raise ServerConfigError(f"cannot read config {path}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ServerConfigError(f"invalid TOML in {path}: {exc}") from exc

    entries = data.get("servers", [])
    if not isinstance(entries, list):
        raise ServerConfigError("'servers' must be an array of tables ([[servers]])")

    expanded = _expand(entries, dict(os.environ if env is None else env))
    configs: list[ServerConfig] = []
    for entry in expanded:
        try:
            configs.append(ServerConfig.model_validate(entry))
        except ValidationError as exc:
            raise ServerConfigError(
                f"invalid server definition {entry.get('name', '?')!r}: {exc}"
            ) from exc

    names = [c.name for c in configs]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise ServerConfigError(f"duplicate server names: {', '.join(sorted(duplicates))}")
    return configs
