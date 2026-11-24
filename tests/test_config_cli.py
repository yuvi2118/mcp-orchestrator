from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from mcp_orchestrator import SSETransport, StdioTransport, build_transport, load_config
from mcp_orchestrator.cli import main
from mcp_orchestrator.config import ServerConfig
from mcp_orchestrator.errors import ServerConfigError

from .conftest import STDIO_SERVER


def write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "servers.toml"
    path.write_text(body)
    return path


# --- ServerConfig ----------------------------------------------------------------------------


def test_stdio_config_needs_a_command():
    with pytest.raises(ValidationError, match="command"):
        ServerConfig(name="x", transport="stdio")


def test_sse_config_needs_a_url():
    with pytest.raises(ValidationError, match="url"):
        ServerConfig(name="x", transport="sse")


@pytest.mark.parametrize("name", ["", "-leading", "has space", "dot.name", "slash/name"])
def test_server_names_are_restricted(name):
    with pytest.raises(ValidationError):
        ServerConfig(name=name, transport="stdio", command="x")


def test_unknown_fields_and_bad_timeouts_are_rejected():
    with pytest.raises(ValidationError):
        ServerConfig(name="x", transport="stdio", command="x", comand="typo")
    with pytest.raises(ValidationError):
        ServerConfig(name="x", transport="stdio", command="x", request_timeout=0)


def test_build_transport_picks_the_right_class():
    assert isinstance(
        build_transport(ServerConfig(name="a", transport="stdio", command="x")), StdioTransport
    )
    assert isinstance(
        build_transport(ServerConfig(name="b", transport="sse", url="http://h/sse")), SSETransport
    )


# --- load_config -----------------------------------------------------------------------------


def test_load_config(tmp_path):
    path = write(
        tmp_path,
        """
[[servers]]
name = "one"
transport = "stdio"
command = "python"
args = ["a.py", "--flag"]
env = { KEY = "value" }
request_timeout = 5

[[servers]]
name = "two"
transport = "sse"
url = "http://localhost:9/sse"
""",
    )
    one, two = load_config(path)
    assert (one.command, one.args, one.env, one.request_timeout) == (
        "python",
        ["a.py", "--flag"],
        {"KEY": "value"},
        5,
    )
    assert two.url == "http://localhost:9/sse"


def test_environment_variables_are_expanded_everywhere(tmp_path):
    path = write(
        tmp_path,
        """
[[servers]]
name = "s"
transport = "sse"
url = "http://${HOST}/sse"
headers = { Authorization = "Bearer ${TOKEN}" }
""",
    )
    (server,) = load_config(path, env={"HOST": "example.org", "TOKEN": "abc"})
    assert server.url == "http://example.org/sse"
    assert server.headers == {"Authorization": "Bearer abc"}


def test_args_in_lists_are_expanded(tmp_path):
    path = write(
        tmp_path, '[[servers]]\nname="s"\ntransport="stdio"\ncommand="x"\nargs=["${DIR}/run.py"]\n'
    )
    assert load_config(path, env={"DIR": "/opt"})[0].args == ["/opt/run.py"]


def test_undefined_environment_variable_is_an_error(tmp_path):
    path = write(tmp_path, '[[servers]]\nname="s"\ntransport="sse"\nurl="http://${NOPE}/sse"\n')
    with pytest.raises(ServerConfigError, match="NOPE"):
        load_config(path, env={})


def test_no_servers_table_means_no_servers(tmp_path):
    assert load_config(write(tmp_path, "# empty\n")) == []


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("servers = 5", "array of tables"),
        ("this is = = not toml", "invalid TOML"),
        (
            '[[servers]]\nname="a"\ntransport="stdio"\ncommand="x"\n[[servers]]\nname="a"\ntransport="stdio"\ncommand="y"',
            "duplicate",
        ),
        ('[[servers]]\nname="a"\ntransport="carrier-pigeon"', "invalid server definition"),
    ],
)
def test_invalid_configs(tmp_path, body, message):
    with pytest.raises(ServerConfigError, match=message):
        load_config(write(tmp_path, body))


def test_missing_file(tmp_path):
    with pytest.raises(ServerConfigError, match="cannot read"):
        load_config(tmp_path / "missing.toml")


# --- command line ----------------------------------------------------------------------------


@pytest.fixture
def config_file(tmp_path) -> str:
    return str(
        write(
            tmp_path,
            '[[servers]]\nname = "mini"\ntransport = "stdio"\n'
            f'command = "{sys.executable}"\nargs = ["{STDIO_SERVER}"]\n'
            '\n[[servers]]\nname = "ghost"\ntransport = "stdio"\ncommand = "/no/such/binary"\n',
        )
    )


def test_cli_list_tools(config_file, capsys):
    assert main(["list-tools", "--config", config_file]) == 0
    captured = capsys.readouterr()
    assert "mini__add\tAdd two integers." in captured.out
    assert "server 'ghost' unavailable" in captured.err


def test_cli_call_prints_the_result(config_file, capsys):
    assert main(["call", "--config", config_file, "mini__add", "--args", '{"a": 20, "b": 22}']) == 0
    assert capsys.readouterr().out.strip() == "42"


def test_cli_call_structured_fallback(config_file, capsys):
    from mcp_orchestrator import cli

    assert cli.main(["call", "--config", config_file, "mini__rich"]) == 0
    assert "hello" in capsys.readouterr().out


def test_cli_reports_mcp_errors_with_exit_code_1(config_file, capsys):
    assert main(["call", "--config", config_file, "mini__add", "--args", '{"a": "x"}']) == 1
    assert "invalid arguments" in capsys.readouterr().err
    assert main(["call", "--config", config_file, "mini__nope"]) == 1
    assert "unknown tool" in capsys.readouterr().err


def test_cli_rejects_bad_json_arguments(config_file, capsys):
    assert main(["call", "--config", config_file, "mini__add", "--args", "{nope"]) == 2
    assert "not valid JSON" in capsys.readouterr().err


def test_cli_requires_a_subcommand():
    with pytest.raises(SystemExit):
        main([])


def test_cli_serve_starts_uvicorn(config_file, monkeypatch):
    import uvicorn

    calls = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, host, port: calls.update(host=host, port=port))
    assert main(["serve", "--config", config_file, "--host", "0.0.0.0", "--port", "9999"]) == 0
    assert calls == {"host": "0.0.0.0", "port": 9999}
