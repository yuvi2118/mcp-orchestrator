"""A minimal, dependency-free MCP server used by the tests.

``handle`` turns one incoming JSON-RPC message into zero or more outgoing messages, so the
same logic backs the stdio subprocess, the HTTP+SSE server and the in-memory fake.
"""

from __future__ import annotations

from typing import Any

PAGE_SIZE = 2
PROTOCOL_VERSION = "2025-03-26"

TOOLS: list[dict[str, Any]] = [
    {
        "name": "echo",
        "description": "Echo the text back.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
    },
    {
        "name": "add",
        "description": "Add two integers.",
        "inputSchema": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    },
    {"name": "fail", "description": "Always reports an error.", "inputSchema": {"type": "object"}},
    {"name": "rich", "description": "Returns mixed content.", "inputSchema": {"type": "object"}},
    {"name": "crash", "description": "Kills the server.", "inputSchema": {"type": "object"}},
]


def _result(request_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name == "echo":
        return {"content": [{"type": "text", "text": arguments["text"]}]}
    if name == "add":
        total = arguments["a"] + arguments["b"]
        return {
            "content": [{"type": "text", "text": str(total)}],
            "structuredContent": {"result": total},
        }
    if name == "fail":
        return {"content": [{"type": "text", "text": "something went wrong"}], "isError": True}
    if name == "rich":
        return {
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "image", "data": "aGk=", "mimeType": "image/png"},
                {"type": "hologram", "payload": 1},
                {"type": "text", "text": "world"},
            ]
        }
    raise KeyError(name)


def handle(message: dict[str, Any]) -> list[dict[str, Any]]:
    method = message.get("method")
    request_id = message.get("id")
    if request_id is None:  # notification: nothing to answer
        return []

    if method == "initialize":
        return [
            _result(
                request_id,
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": True}},
                    "serverInfo": {"name": "mini", "version": "1.0"},
                    "instructions": "test server",
                },
            )
        ]
    if method == "ping":
        return [_result(request_id, {})]
    if method == "tools/list":
        start = int((message.get("params") or {}).get("cursor") or 0)
        page = TOOLS[start : start + PAGE_SIZE]
        result: dict[str, Any] = {"tools": page}
        if start + PAGE_SIZE < len(TOOLS):
            result["nextCursor"] = str(start + PAGE_SIZE)
        return [_result(request_id, result)]
    if method == "tools/call":
        params = message.get("params") or {}
        try:
            return [_result(request_id, _call(params["name"], params.get("arguments") or {}))]
        except KeyError:
            return [_error(request_id, -32602, f"unknown tool {params.get('name')!r}")]
    return [_error(request_id, -32601, f"method '{method}' not found")]
