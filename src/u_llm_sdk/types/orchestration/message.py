"""LLM Types - Inter-LLM Message Types.

Message types for communication between LLM providers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional, Union

from ..config import Provider
from .enums import MessageType


@dataclass
class LLMMessage:
    """Message for inter-LLM communication.

    Attributes:
        message_id: Unique message identifier
        timestamp: When the message was created
        source: Source provider
        target: Target provider or "broadcast" for all
        message_type: Type of message
        payload: Message payload data
        context_reference: Reference to related context/task
        requires_response: Whether a response is expected
    """
    message_id: str
    timestamp: datetime
    source: Provider
    target: Union[Provider, Literal["broadcast"]]
    message_type: MessageType
    payload: dict = field(default_factory=dict)
    context_reference: Optional[str] = None
    requires_response: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        target_value = self.target.value if isinstance(self.target, Provider) else self.target

        return {
            "message_id": self.message_id,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source.value,
            "target": target_value,
            "message_type": self.message_type.value,
            "payload": self.payload,
            "context_reference": self.context_reference,
            "requires_response": self.requires_response,
        }

    @classmethod
    def from_dict(cls, data: dict) -> LLMMessage:
        """Create from dictionary (JSON deserialization)."""
        timestamp_value = data["timestamp"]
        timestamp = (
            datetime.fromisoformat(timestamp_value)
            if isinstance(timestamp_value, str)
            else timestamp_value
        )

        source_value = data["source"]
        source = Provider(source_value) if isinstance(source_value, str) else source_value

        target_value = data["target"]
        if target_value == "broadcast":
            target: Union[Provider, Literal["broadcast"]] = "broadcast"
        else:
            target = Provider(target_value) if isinstance(target_value, str) else target_value

        message_type_value = data["message_type"]
        message_type = (
            MessageType(message_type_value)
            if isinstance(message_type_value, str)
            else message_type_value
        )

        return cls(
            message_id=data["message_id"],
            timestamp=timestamp,
            source=source,
            target=target,
            message_type=message_type,
            payload=data.get("payload", {}),
            context_reference=data.get("context_reference"),
            requires_response=data.get("requires_response", False),
        )
