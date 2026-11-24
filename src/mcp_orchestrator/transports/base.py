"""Transport interface: moves already-encoded JSON-RPC messages to and from a server."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any


class Transport(ABC):
    """A bidirectional message channel.

    A transport knows nothing about JSON-RPC semantics. It connects, sends one JSON
    object at a time, and yields decoded JSON objects as they arrive.
    """

    @abstractmethod
    async def start(self) -> None:
        """Open the connection. Raises ``TransportError`` if it cannot be opened."""

    @abstractmethod
    async def send(self, payload: dict[str, Any]) -> None:
        """Send one message. Raises ``TransportError`` if the connection is gone."""

    @abstractmethod
    def messages(self) -> AsyncIterator[dict[str, Any]]:
        """Yield incoming messages until the connection closes.

        Ends normally on a clean close and raises ``TransportError`` on failure.
        """

    @abstractmethod
    async def close(self) -> None:
        """Release all resources. Must be safe to call more than once."""
