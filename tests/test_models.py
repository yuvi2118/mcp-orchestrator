import pytest
from pydantic import ValidationError

from mcp_orchestrator.models import (
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    InitializeResult,
    ListToolsResult,
    ResourceLink,
    TextContent,
    Tool,
    UnknownContent,
)


def test_initialize_result_uses_wire_names():
    result = InitializeResult.model_validate(
        {
            "protocolVersion": "2025-03-26",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "s", "version": "1"},
        }
    )
    assert result.protocol_version == "2025-03-26"
    assert result.server_info.name == "s"
    assert result.instructions is None


def test_initialize_result_requires_server_info():
    with pytest.raises(ValidationError):
        InitializeResult.model_validate({"protocolVersion": "x"})


def test_tool_keeps_unknown_fields():
    tool = Tool.model_validate({"name": "t", "inputSchema": {"type": "object"}, "futureField": 1})
    assert tool.input_schema == {"type": "object"}
    assert tool.model_extra == {"futureField": 1}


def test_tool_requires_input_schema():
    with pytest.raises(ValidationError):
        Tool.model_validate({"name": "t"})


def test_list_tools_result_cursor():
    page = ListToolsResult.model_validate(
        {"tools": [{"name": "t", "inputSchema": {}}], "nextCursor": "abc"}
    )
    assert page.next_cursor == "abc"


def test_call_tool_result_content_types_and_text_property():
    result = CallToolResult.model_validate(
        {
            "content": [
                {"type": "text", "text": "a"},
                {"type": "image", "data": "x", "mimeType": "image/png"},
                {"type": "resource_link", "uri": "file:///x"},
                {"type": "resource", "resource": {"uri": "u"}},
                {"type": "from-the-future", "foo": 1},
                {"type": "text", "text": "b"},
            ]
        }
    )
    kinds = [type(block) for block in result.content]
    assert kinds == [
        TextContent,
        ImageContent,
        ResourceLink,
        EmbeddedResource,
        UnknownContent,
        TextContent,
    ]
    assert result.text == "a\nb"
    assert result.is_error is False


def test_call_tool_result_error_flag_and_structured_content():
    result = CallToolResult.model_validate(
        {"content": [], "isError": True, "structuredContent": {"k": 1}}
    )
    assert result.is_error is True
    assert result.structured_content == {"k": 1}


def test_call_tool_result_rejects_malformed_content():
    with pytest.raises(ValidationError):
        CallToolResult.model_validate({"content": "not a list"})
