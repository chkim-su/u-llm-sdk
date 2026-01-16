"""LLM Types - Task and Clarity Types.

Task definitions, unclear aspects, and clarity assessments for
worker self-assessment (ClarityGate).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from .enums import ClarityLevel


@dataclass
class Task:
    """Task definition for worker assignment.

    Attributes:
        task_id: Unique task identifier
        objective: What needs to be accomplished
        context: Background information and constraints context
        constraints: List of constraints to follow
        expected_output: Expected output format/content
        clarity_level: Self-assessed clarity (0.0-1.0)
        source: Where this task originated (orchestrator, user, etc.)
    """
    task_id: str
    objective: str
    context: str
    constraints: list[str] = field(default_factory=list)
    expected_output: str = ""
    clarity_level: Optional[float] = None
    source: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "task_id": self.task_id,
            "objective": self.objective,
            "context": self.context,
            "constraints": self.constraints,
            "expected_output": self.expected_output,
            "clarity_level": self.clarity_level,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Task:
        """Create from dictionary (JSON deserialization)."""
        return cls(
            task_id=data["task_id"],
            objective=data["objective"],
            context=data.get("context", ""),
            constraints=data.get("constraints", []),
            expected_output=data.get("expected_output", ""),
            clarity_level=data.get("clarity_level"),
            source=data.get("source", ""),
        )


@dataclass
class UnclearAspect:
    """An unclear aspect of a task that needs clarification.

    Attributes:
        aspect_type: Category of the unclear aspect
        description: What is unclear
        clarification_needed: What clarification is needed
    """
    aspect_type: Literal[
        "knowledge_gap",
        "scope_ambiguity",
        "objective_ambiguity",
        "constraint_missing",
        "context_insufficient"
    ]
    description: str
    clarification_needed: str

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "aspect_type": self.aspect_type,
            "description": self.description,
            "clarification_needed": self.clarification_needed,
        }

    @classmethod
    def from_dict(cls, data: dict) -> UnclearAspect:
        """Create from dictionary (JSON deserialization)."""
        return cls(
            aspect_type=data["aspect_type"],
            description=data["description"],
            clarification_needed=data["clarification_needed"],
        )


@dataclass
class ClarityAssessment:
    """Result of self-assessing task clarity (ClarityGate).

    Attributes:
        level: Overall clarity level
        score: Numeric clarity score (0.0-1.0)
        unclear_aspects: List of unclear aspects identified
        self_questions: Questions the worker has about the task
        recommendation: Recommended action (execute, clarify, escalate)
    """
    level: ClarityLevel
    score: float
    unclear_aspects: list[UnclearAspect] = field(default_factory=list)
    self_questions: list[str] = field(default_factory=list)
    recommendation: Literal["execute", "clarify", "escalate"] = "execute"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "level": self.level.value,
            "score": self.score,
            "unclear_aspects": [a.to_dict() for a in self.unclear_aspects],
            "self_questions": self.self_questions,
            "recommendation": self.recommendation,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ClarityAssessment:
        """Create from dictionary (JSON deserialization)."""
        level_value = data["level"]
        level = ClarityLevel(level_value) if isinstance(level_value, str) else level_value

        unclear_aspects = [
            UnclearAspect.from_dict(a) for a in data.get("unclear_aspects", [])
        ]

        return cls(
            level=level,
            score=data["score"],
            unclear_aspects=unclear_aspects,
            self_questions=data.get("self_questions", []),
            recommendation=data.get("recommendation", "execute"),
        )
