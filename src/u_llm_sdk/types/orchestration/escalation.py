"""LLM Types - Escalation Types.

Escalation request and response types for upward communication
when tasks are unclear or require higher-level guidance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from ..config import Provider
from .task import Task, ClarityAssessment


@dataclass
class EscalationRequest:
    """Request for upward escalation when task is unclear.

    Attributes:
        source_worker: Which worker is escalating
        original_task: The task that needs clarification
        clarity_assessment: The worker's clarity assessment
        specific_questions: Specific questions to answer
        request_type: Type of escalation request
    """
    source_worker: Provider
    original_task: Task
    clarity_assessment: ClarityAssessment
    specific_questions: list[str] = field(default_factory=list)
    request_type: Literal[
        "clarification",
        "scope_definition",
        "permission",
        "guidance"
    ] = "clarification"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "source_worker": self.source_worker.value,
            "original_task": self.original_task.to_dict(),
            "clarity_assessment": self.clarity_assessment.to_dict(),
            "specific_questions": self.specific_questions,
            "request_type": self.request_type,
        }

    @classmethod
    def from_dict(cls, data: dict) -> EscalationRequest:
        """Create from dictionary (JSON deserialization)."""
        source_worker_value = data["source_worker"]
        source_worker = (
            Provider(source_worker_value)
            if isinstance(source_worker_value, str)
            else source_worker_value
        )

        return cls(
            source_worker=source_worker,
            original_task=Task.from_dict(data["original_task"]),
            clarity_assessment=ClarityAssessment.from_dict(data["clarity_assessment"]),
            specific_questions=data.get("specific_questions", []),
            request_type=data.get("request_type", "clarification"),
        )


@dataclass
class EscalationResponse:
    """Response to an escalation request.

    Attributes:
        clarifications: Answers to specific questions (question -> answer)
        refined_task: Refined task with clarifications applied
        additional_context: Additional context provided
        permission_granted: Whether permission was granted (for permission requests)
        guidance: Guidance provided (for guidance requests)
    """
    clarifications: dict = field(default_factory=dict)
    refined_task: Optional[Task] = None
    additional_context: str = ""
    permission_granted: bool = True
    guidance: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "clarifications": self.clarifications,
            "refined_task": self.refined_task.to_dict() if self.refined_task else None,
            "additional_context": self.additional_context,
            "permission_granted": self.permission_granted,
            "guidance": self.guidance,
        }

    @classmethod
    def from_dict(cls, data: dict) -> EscalationResponse:
        """Create from dictionary (JSON deserialization)."""
        refined_task_data = data.get("refined_task")
        refined_task = Task.from_dict(refined_task_data) if refined_task_data else None

        return cls(
            clarifications=data.get("clarifications", {}),
            refined_task=refined_task,
            additional_context=data.get("additional_context", ""),
            permission_granted=data.get("permission_granted", True),
            guidance=data.get("guidance", ""),
        )
