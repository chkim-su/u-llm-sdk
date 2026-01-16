"""LLM Types - Hook Data Structures.

This module defines the data structures for intervention hooks.
These are the data-only classes shared between U-llm-sdk and MV-rag.

Note: The InterventionHook protocol itself is NOT in this package.
      It lives in U-llm-sdk as it's the consumer-side interface.
      MV-rag implements the protocol via its API endpoints.

Design Philosophy:
    - Data classes are SHARED (this package)
    - Protocol/Interface lives with CONSUMER (U-llm-sdk)
    - Implementation lives with PROVIDER (MV-rag)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional
import uuid

if TYPE_CHECKING:
    from .models import LLMResult


@dataclass
class PreActionContext:
    """Context to inject before the LLM prompt.

    This dataclass contains the context that should be prepended or appended
    to the user's prompt based on DejaVu pattern matching and influence routing.

    Attributes:
        injection_id: Unique ID for this injection (for feedback tracking)
        context_text: Formatted context text to inject
        confidence: Confidence score from influence routing [0.0, 1.0]
        dejavu_group_id: ID of the matched DejaVu group (if any)
        pattern_summary: Brief summary of the detected pattern
        injection_position: Where to inject ("prepend", "append", "system")
        token_count: Estimated token count of context_text
        timestamp: When the injection was created
    """

    injection_id: str
    context_text: str
    confidence: float
    dejavu_group_id: Optional[str] = None
    pattern_summary: Optional[str] = None
    injection_position: str = "prepend"
    token_count: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def create(
        cls,
        context_text: str,
        confidence: float,
        dejavu_group_id: Optional[str] = None,
        pattern_summary: Optional[str] = None,
        injection_position: str = "prepend",
        token_count: int = 0,
    ) -> "PreActionContext":
        """Factory method to create PreActionContext with auto-generated ID."""
        return cls(
            injection_id=str(uuid.uuid4()),
            context_text=context_text,
            confidence=confidence,
            dejavu_group_id=dejavu_group_id,
            pattern_summary=pattern_summary,
            injection_position=injection_position,
            token_count=token_count,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "injection_id": self.injection_id,
            "context_text": self.context_text,
            "confidence": self.confidence,
            "dejavu_group_id": self.dejavu_group_id,
            "pattern_summary": self.pattern_summary,
            "injection_position": self.injection_position,
            "token_count": self.token_count,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PreActionContext":
        """Create from dictionary (JSON deserialization)."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now(timezone.utc)

        return cls(
            injection_id=data["injection_id"],
            context_text=data["context_text"],
            confidence=data["confidence"],
            dejavu_group_id=data.get("dejavu_group_id"),
            pattern_summary=data.get("pattern_summary"),
            injection_position=data.get("injection_position", "prepend"),
            token_count=data.get("token_count", 0),
            timestamp=timestamp,
        )


@dataclass
class PostActionFeedback:
    """Feedback data after LLM action completion.

    This dataclass captures the outcome of an LLM action for feedback collection.
    Used by the intervention hook to update DejaVu groups and influence scores.

    Attributes:
        run_id: Current run/session identifier
        event_id: Unique event identifier
        injection_id: ID of the pre-action injection (if any)
        success: Whether the LLM action was successful
        result_type: Type of result ("text", "file_edit", "command", etc.)
        error_message: Error message if success=False
        duration_ms: Execution duration in milliseconds
        user_feedback: Explicit user feedback ("positive", "negative", None)
        extra: Additional fields for schema-driven logging extension.
               MV-rag schema specifies required fields, U-llm-sdk populates
               available ones dynamically without code changes.
    """

    run_id: str
    event_id: str
    injection_id: Optional[str]
    success: bool
    result_type: str
    error_message: Optional[str] = None
    duration_ms: int = 0
    user_feedback: Optional[str] = None
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_result(
        cls,
        result: "LLMResult",
        run_id: str,
        injection_id: Optional[str] = None,
    ) -> "PostActionFeedback":
        """Create PostActionFeedback from LLMResult."""
        return cls(
            run_id=run_id,
            event_id=str(uuid.uuid4()),
            injection_id=injection_id,
            success=result.success,
            result_type=result.result_type.value,
            error_message=result.error,
            duration_ms=result.duration_ms or 0,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "run_id": self.run_id,
            "event_id": self.event_id,
            "injection_id": self.injection_id,
            "success": self.success,
            "result_type": self.result_type,
            "error_message": self.error_message,
            "duration_ms": self.duration_ms,
            "user_feedback": self.user_feedback,
        }
        # Merge extra fields at top level for MV-rag compatibility
        if self.extra:
            result.update(self.extra)
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "PostActionFeedback":
        """Create from dictionary (JSON deserialization)."""
        # Known fields
        known_fields = {
            "run_id", "event_id", "injection_id", "success",
            "result_type", "error_message", "duration_ms", "user_feedback", "extra"
        }
        # Extract extra fields (anything not in known_fields)
        extra = data.get("extra", {})
        for key, value in data.items():
            if key not in known_fields:
                extra[key] = value

        return cls(
            run_id=data["run_id"],
            event_id=data["event_id"],
            injection_id=data.get("injection_id"),
            success=data["success"],
            result_type=data["result_type"],
            error_message=data.get("error_message"),
            duration_ms=data.get("duration_ms", 0),
            user_feedback=data.get("user_feedback"),
            extra=extra,
        )
