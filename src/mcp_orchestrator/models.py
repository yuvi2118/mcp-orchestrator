"""Typed models for the parts of the MCP schema this client uses."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag


class _Model(BaseModel):
    # MCP evolves; unknown fields must not break the client.
    model_config = ConfigDict(extra="allow", populate_by_name=True)


class Implementation(_Model):
    name: str
    version: str = ""
    title: str | None = None


class InitializeResult(_Model):
    protocol_version: str = Field(alias="protocolVersion")
    capabilities: dict[str, Any] = Field(default_factory=dict)
    server_info: Implementation = Field(alias="serverInfo")
    instructions: str | None = None


class Tool(_Model):
    name: str
    title: str | None = None
    description: str | None = None
    input_schema: dict[str, Any] = Field(alias="inputSchema")
    output_schema: dict[str, Any] | None = Field(default=None, alias="outputSchema")
    annotations: dict[str, Any] | None = None


class ListToolsResult(_Model):
    tools: list[Tool]
    next_cursor: str | None = Field(default=None, alias="nextCursor")


class TextContent(_Model):
    type: Literal["text"] = "text"
    text: str


class ImageContent(_Model):
    type: Literal["image"] = "image"
    data: str
    mime_type: str = Field(alias="mimeType")


class AudioContent(_Model):
    type: Literal["audio"] = "audio"
    data: str
    mime_type: str = Field(alias="mimeType")


class ResourceLink(_Model):
    type: Literal["resource_link"] = "resource_link"
    uri: str
    name: str | None = None


class EmbeddedResource(_Model):
    type: Literal["resource"] = "resource"
    resource: dict[str, Any]


class UnknownContent(_Model):
    """Fallback for content types added to the protocol after this client was written."""

    type: str


_KNOWN_CONTENT = {"text", "image", "audio", "resource_link", "resource"}


def _content_tag(value: Any) -> str:
    kind = value.get("type") if isinstance(value, dict) else getattr(value, "type", None)
    return kind if kind in _KNOWN_CONTENT else "unknown"


ContentBlock = Annotated[
    Union[  # noqa: UP007 - Annotated union needs explicit Union with Tag
        Annotated[TextContent, Tag("text")],
        Annotated[ImageContent, Tag("image")],
        Annotated[AudioContent, Tag("audio")],
        Annotated[ResourceLink, Tag("resource_link")],
        Annotated[EmbeddedResource, Tag("resource")],
        Annotated[UnknownContent, Tag("unknown")],
    ],
    Discriminator(_content_tag),
]


class CallToolResult(_Model):
    content: list[ContentBlock] = Field(default_factory=list)
    is_error: bool = Field(default=False, alias="isError")
    structured_content: dict[str, Any] | None = Field(default=None, alias="structuredContent")

    @property
    def text(self) -> str:
        """All text blocks joined with newlines."""
        return "\n".join(b.text for b in self.content if isinstance(b, TextContent))
