import json

import pytest

from mcp_orchestrator.errors import ProtocolError
from mcp_orchestrator.jsonrpc import (
    METHOD_NOT_FOUND,
    Notification,
    Request,
    Response,
    dumps,
    make_error,
    make_notification,
    make_request,
    make_result,
    parse_message,
)


def test_make_request_omits_missing_params():
    assert make_request(1, "ping") == {"jsonrpc": "2.0", "id": 1, "method": "ping"}


def test_make_request_with_params():
    assert make_request("a", "tools/call", {"name": "x"})["params"] == {"name": "x"}


def test_make_notification_has_no_id():
    assert "id" not in make_notification("notifications/initialized")


def test_make_result_keeps_empty_result():
    assert make_result(7, {}) == {"jsonrpc": "2.0", "id": 7, "result": {}}


def test_make_error_shape():
    msg = make_error(3, METHOD_NOT_FOUND, "nope", {"x": 1})
    assert msg["error"] == {"code": -32601, "message": "nope", "data": {"x": 1}}


def test_dumps_is_single_line_and_compact():
    text = dumps({"a": 1, "b": "é\nx"})
    assert "\n" not in text and ", " not in text
    assert json.loads(text) == {"a": 1, "b": "é\nx"}


def test_parse_request_notification_response():
    assert isinstance(parse_message({"jsonrpc": "2.0", "id": 1, "method": "m"}), Request)
    assert isinstance(parse_message({"jsonrpc": "2.0", "method": "m"}), Notification)
    assert isinstance(parse_message({"jsonrpc": "2.0", "id": 1, "result": {}}), Response)
    error = parse_message({"jsonrpc": "2.0", "id": 1, "error": {"code": 1, "message": "x"}})
    assert isinstance(error, Response) and error.error.code == 1


def test_parse_accepts_json_text_and_bytes():
    assert isinstance(parse_message('{"jsonrpc":"2.0","method":"m"}'), Notification)
    assert isinstance(parse_message(b'{"jsonrpc":"2.0","method":"m"}'), Notification)


def test_response_with_null_result_is_valid():
    response = parse_message({"jsonrpc": "2.0", "id": 1, "result": None})
    assert isinstance(response, Response) and response.error is None


def test_response_with_null_id_is_valid():
    response = parse_message(
        {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "x"}}
    )
    assert response.id is None


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        b"\xff\xfe",
        "[]",
        "[1, 2]",
        "42",
        '"string"',
        "{}",
        '{"jsonrpc": "1.0", "id": 1, "method": "m"}',
        '{"jsonrpc": "2.0", "id": 1}',
        '{"jsonrpc": "2.0", "id": 1, "result": 1, "error": {"code": 1, "message": "x"}}',
        '{"jsonrpc": "2.0", "id": 1, "method": "m", "extra": true}',
        '{"jsonrpc": "2.0", "id": 1, "method": 5}',
        '{"jsonrpc": "2.0", "id": 1, "error": {"code": "bad"}}',
    ],
)
def test_parse_rejects_invalid_messages(raw):
    with pytest.raises(ProtocolError):
        parse_message(raw)
