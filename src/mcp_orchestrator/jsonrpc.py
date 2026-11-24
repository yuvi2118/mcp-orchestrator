"""JSON-RPC 2.0 message models, encoding and parsing.

Only what MCP needs is implemented: requests, notifications and responses.
JSON-RPC batches are rejected because MCP does not use them.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from .errors import ProtocolError

JSONRPC_VERSION = "2.0"

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

RequestId = int | str


class ErrorObject(BaseModel):
    code: int
    message: str
    data: Any | None = None


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid")

    jsonrpc: Literal["2.0"] = JSONRPC_VERSION
    id: RequestId
    method: str
    params: dict[str, Any] | list[Any] | None = None


class Notification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    jsonrpc: Literal["2.0"] = JSONRPC_VERSION
    method: str
    params: dict[str, Any] | list[Any] | None = None


class Response(BaseModel):
    model_config = ConfigDict(extra="forbid")

    jsonrpc: Literal["2.0"] = JSONRPC_VERSION
    id: RequestId | None
    result: Any | None = None
    error: ErrorObject | None = None

    @model_validator(mode="after")
    def _exactly_one_of_result_or_error(self) -> Response:
        has_result = "result" in self.model_fields_set
        has_error = self.error is not None
        if has_result == has_error:
            raise ValueError("a response must contain exactly one of 'result' or 'error'")
        return self


Message = Request | Notification | Response


def make_request(request_id: RequestId, method: str, params: dict[str, Any] | None = None) -> dict:
    return Request(id=request_id, method=method, params=params).model_dump(exclude_none=True)


def make_notification(method: str, params: dict[str, Any] | None = None) -> dict:
    return Notification(method=method, params=params).model_dump(exclude_none=True)


def make_result(request_id: RequestId, result: Any) -> dict:
    return {"jsonrpc": JSONRPC_VERSION, "id": request_id, "result": result}


def make_error(
    request_id: RequestId | None, code: int, message: str, data: Any | None = None
) -> dict:
    error = ErrorObject(code=code, message=message, data=data).model_dump(exclude_none=True)
    return {"jsonrpc": JSONRPC_VERSION, "id": request_id, "error": error}


def dumps(payload: dict[str, Any]) -> str:
    """Compact single-line JSON, as required by the MCP stdio framing."""
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def parse_message(raw: str | bytes | dict[str, Any]) -> Message:
    """Decode and validate one JSON-RPC message.

    Raises :class:`ProtocolError` for malformed JSON, batches, or messages that are
    not a valid request, notification or response.
    """
    if isinstance(raw, dict):
        data: Any = raw
    else:
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ProtocolError(f"invalid JSON: {exc}") from exc

    if isinstance(data, list):
        raise ProtocolError("JSON-RPC batches are not supported")
    if not isinstance(data, dict):
        raise ProtocolError("a JSON-RPC message must be an object")

    try:
        if "method" in data:
            return Request(**data) if "id" in data else Notification(**data)
        if "result" in data or "error" in data:
            return Response(**data)
    except (ValidationError, TypeError) as exc:
        raise ProtocolError(f"invalid JSON-RPC message: {exc}") from exc
    raise ProtocolError("message is not a request, notification or response")
