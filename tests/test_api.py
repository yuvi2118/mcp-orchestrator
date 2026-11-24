from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from mcp_orchestrator import ClientSession, ToolRegistry
from mcp_orchestrator.api import create_app

from .conftest import STDIO_SERVER, stdio_config


@pytest.fixture
async def registry(memory_pair) -> AsyncIterator[ToolRegistry]:
    transport, _ = memory_pair
    async with ToolRegistry() as reg:
        session = ClientSession(transport, request_timeout=2.0)
        await session.start()
        await reg.add_session("mini", session)
        yield reg


def make_client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


@pytest.fixture
async def client(registry) -> AsyncIterator[httpx.AsyncClient]:
    async with make_client(create_app(registry)) as c:
        yield c


async def test_healthz(client):
    body = (await client.get("/healthz")).json()
    assert body == {"status": "ok", "servers": {"mini": {"connected": True, "tools": 5}}}


async def test_healthz_reports_degraded_when_a_server_is_down(registry):
    registry.failures["offline"] = "TransportError: refused"
    async with make_client(create_app(registry)) as c:
        body = (await c.get("/healthz")).json()
    assert body["status"] == "degraded"
    assert body["servers"]["offline"]["error"] == "TransportError: refused"


async def test_servers_and_tools(client):
    assert (await client.get("/servers")).json()["mini"]["tools"] == 5
    tools = (await client.get("/tools")).json()
    add = next(t for t in tools if t["name"] == "mini__add")
    assert add["server"] == "mini" and add["input_schema"]["required"] == ["a", "b"]
    assert (await client.get("/tools", params={"server": "other"})).json() == []


async def test_call_tool(client):
    response = await client.post("/tools/mini__add/call", json={"arguments": {"a": 4, "b": 5}})
    body = response.json()
    assert response.status_code == 200
    assert body["text"] == "9" and body["structured_content"] == {"result": 9}
    assert body["content"][0] == {"type": "text", "text": "9"}


async def test_call_unknown_tool_is_404(client):
    response = await client.post("/tools/mini__nope/call", json={})
    assert response.status_code == 404 and response.json()["error"] == "tool_not_found"


async def test_invalid_arguments_are_422_with_details(client):
    response = await client.post("/tools/mini__add/call", json={"arguments": {"a": "x"}})
    body = response.json()
    assert response.status_code == 422 and body["error"] == "invalid_arguments"
    assert any("required property" in d for d in body["detail"])


async def test_tool_failure_is_502(client):
    response = await client.post("/tools/mini__fail/call", json={})
    assert response.status_code == 502
    assert response.json() == {
        "error": "tool_error",
        "tool": "mini__fail",
        "detail": "something went wrong",
    }


async def test_bad_request_body_is_422(client):
    response = await client.post("/tools/mini__add/call", json={"arguments": [1], "timeout": -1})
    assert response.status_code == 422


async def test_timeout_is_504(memory_pair, registry):
    _, server = memory_pair
    server.interceptor = lambda m: m.get("method") == "tools/call"
    async with make_client(create_app(registry)) as c:
        response = await c.post(
            "/tools/mini__echo/call", json={"arguments": {"text": "x"}, "timeout": 0.05}
        )
    assert response.status_code == 504 and response.json()["error"] == "timeout"


async def test_lost_server_connection_is_503(memory_pair, registry):
    _, server = memory_pair
    await server.transport.close()
    await asyncio.sleep(0.05)
    async with make_client(create_app(registry)) as c:
        response = await c.post("/tools/mini__echo/call", json={"arguments": {"text": "x"}})
    assert response.status_code == 503 and response.json()["error"] == "server_unavailable"


async def test_chat_needs_an_api_key(client, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    response = await client.post("/chat", json={"prompt": "hi"})
    assert response.status_code == 503 and "ANTHROPIC_API_KEY" in response.json()["detail"]


async def test_chat_runs_the_agent_loop(registry):
    replies = [
        SimpleNamespace(
            content=[
                SimpleNamespace(type="tool_use", id="t1", name="mini__add", input={"a": 1, "b": 2})
            ],
            stop_reason="tool_use",
        ),
        SimpleNamespace(
            content=[SimpleNamespace(type="text", text="It is 3.")], stop_reason="end_turn"
        ),
    ]

    async def create(**kwargs: Any) -> SimpleNamespace:
        return replies.pop(0)

    fake = SimpleNamespace(messages=SimpleNamespace(create=create))
    async with make_client(create_app(registry, agent_client=fake)) as c:
        response = await c.post("/chat", json={"prompt": "1+2?", "max_turns": 3})
    body = response.json()
    assert body["text"] == "It is 3." and body["turns"] == 2
    assert body["tool_calls"] == [
        {"name": "mini__add", "arguments": {"a": 1, "b": 2}, "output": "3", "is_error": False}
    ]


async def test_chat_validates_input(client):
    assert (await client.post("/chat", json={"prompt": ""})).status_code == 422
    assert (await client.post("/chat", json={"prompt": "x", "max_turns": 99})).status_code == 422


async def test_app_loads_config_and_manages_connections_itself(tmp_path):
    config = tmp_path / "servers.toml"
    config.write_text(
        f'[[servers]]\nname = "mini"\ntransport = "stdio"\ncommand = "{stdio_config().command}"\n'
        f'args = ["{STDIO_SERVER}"]\n\n'
        '[[servers]]\nname = "ghost"\ntransport = "stdio"\ncommand = "/no/such/binary"\n'
    )
    app = create_app(config_path=str(config))
    async with app.router.lifespan_context(app), make_client(app) as c:
        body = (await c.get("/healthz")).json()
        assert body["status"] == "degraded"
        assert body["servers"]["mini"]["connected"] is True
        assert body["servers"]["ghost"]["connected"] is False
        assert (await c.post("/tools/mini__add/call", json={"arguments": {"a": 1, "b": 1}})).json()[
            "text"
        ] == "2"
    assert not app.state.registry.tools()  # closed on shutdown


async def test_app_without_any_config_starts_empty(monkeypatch):
    monkeypatch.delenv("MCP_ORCHESTRATOR_CONFIG", raising=False)
    app = create_app()
    async with app.router.lifespan_context(app), make_client(app) as c:
        assert (await c.get("/tools")).json() == []
        assert (await c.get("/healthz")).json()["status"] == "ok"
