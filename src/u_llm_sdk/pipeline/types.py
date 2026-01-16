"""Pipeline types and data structures.

Defines the workflow template format and execution types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

import yaml


class PhaseType(Enum):
    """Built-in phase types."""

    BRAINSTORM = "brainstorm"
    CLARITY_GATE = "clarity_gate"
    CLARITY_DIALOGUE = "clarity_dialogue"  # Claude asks, Gemini answers with session context
    DELEGATION = "delegation"
    REVIEW = "review"
    CUSTOM = "custom"


class PhaseResult(Enum):
    """Phase execution result."""

    SUCCESS = "success"
    FAILED = "failed"
    NEEDS_INPUT = "needs_input"
    SKIPPED = "skipped"


@dataclass
class PhaseConfig:
    """Configuration for a pipeline phase.

    Attributes:
        name: Phase display name
        type: Phase type (brainstorm, clarity_gate, delegation, review)
        provider: LLM provider to use (claude, gemini, codex)
        config: Phase-specific configuration
        on_success: Next phase on success (default: next in sequence)
        on_failure: Action on failure (abort, skip, retry, ask_user)
        on_needs_input: Action when user input needed
        optional: If True, failure doesn't abort pipeline
        timeout_seconds: Phase timeout
    """

    name: str
    type: PhaseType
    provider: Optional[str] = None
    config: dict[str, Any] = field(default_factory=dict)
    on_success: Optional[str] = None  # next phase name or None for next
    on_failure: str = "abort"  # abort, skip, retry, ask_user
    on_needs_input: str = "ask_user"  # ask_user, abort, skip
    optional: bool = False
    timeout_seconds: int = 600

    @classmethod
    def from_dict(cls, data: dict) -> PhaseConfig:
        """Create from dictionary."""
        phase_type = data.get("type", "custom")
        if isinstance(phase_type, str):
            try:
                phase_type = PhaseType(phase_type)
            except ValueError:
                phase_type = PhaseType.CUSTOM

        return cls(
            name=data.get("name", "unnamed"),
            type=phase_type,
            provider=data.get("provider"),
            config=data.get("config", {}),
            on_success=data.get("on_success"),
            on_failure=data.get("on_failure", "abort"),
            on_needs_input=data.get("on_needs_input", "ask_user"),
            optional=data.get("optional", False),
            timeout_seconds=data.get("timeout_seconds", 600),
        )


@dataclass
class WorkflowTemplate:
    """Workflow template definition.

    Attributes:
        name: Template name
        description: Template description
        version: Template version
        phases: List of phases to execute
        variables: Default variable values
        boundaries: Default boundary constraints
    """

    name: str
    description: str
    version: str
    phases: list[PhaseConfig]
    variables: dict[str, Any] = field(default_factory=dict)
    boundaries: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> WorkflowTemplate:
        """Load template from YAML file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> WorkflowTemplate:
        """Create from dictionary."""
        phases = [
            PhaseConfig.from_dict(p) for p in data.get("phases", [])
        ]

        return cls(
            name=data.get("name", "unnamed"),
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            phases=phases,
            variables=data.get("variables", {}),
            boundaries=data.get("boundaries", {}),
        )

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "phases": [
                {
                    "name": p.name,
                    "type": p.type.value,
                    "provider": p.provider,
                    "config": p.config,
                    "on_success": p.on_success,
                    "on_failure": p.on_failure,
                    "on_needs_input": p.on_needs_input,
                    "optional": p.optional,
                    "timeout_seconds": p.timeout_seconds,
                }
                for p in self.phases
            ],
            "variables": self.variables,
            "boundaries": self.boundaries,
        }


@dataclass
class PhaseOutput:
    """Output from a single phase execution.

    Attributes:
        phase_name: Name of the phase
        result: Execution result
        output: Phase output data
        error: Error message if failed
        questions: Questions for user if needs_input
        duration_ms: Execution time
        cost_usd: Cost if applicable
    """

    phase_name: str
    result: PhaseResult
    output: Any = None
    error: Optional[str] = None
    questions: list[str] = field(default_factory=list)
    duration_ms: int = 0
    cost_usd: float = 0.0


@dataclass
class PipelineResult:
    """Result of pipeline execution.

    Attributes:
        template_name: Name of the workflow template
        success: Whether pipeline completed successfully
        phases: Results from each phase
        final_output: Final aggregated output
        needs_input: Whether user input is needed to continue
        pending_questions: Questions for user
        total_duration_ms: Total execution time
        total_cost_usd: Total cost
        can_resume: Whether pipeline can be resumed
        resume_phase: Phase to resume from
        report_path: Path to the generated report file
    """

    template_name: str
    success: bool = False
    phases: list[PhaseOutput] = field(default_factory=list)
    final_output: Any = None
    needs_input: bool = False
    pending_questions: list[str] = field(default_factory=list)
    total_duration_ms: int = 0
    total_cost_usd: float = 0.0
    can_resume: bool = False
    resume_phase: Optional[str] = None
    report_path: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "template_name": self.template_name,
            "success": self.success,
            "phases": [
                {
                    "name": p.phase_name,
                    "result": p.result.value,
                    "error": p.error,
                    "questions": p.questions,
                    "duration_ms": p.duration_ms,
                    "cost_usd": p.cost_usd,
                }
                for p in self.phases
            ],
            "needs_input": self.needs_input,
            "pending_questions": self.pending_questions,
            "total_duration_ms": self.total_duration_ms,
            "total_cost_usd": self.total_cost_usd,
            "can_resume": self.can_resume,
            "resume_phase": self.resume_phase,
            "report_path": self.report_path,
        }


# NOTE: Execution tracing is now handled by PipelineChronicleAdapter
# which uses llm-types Chronicle records (DecisionRecord, ExecutionRecord)
# See: chronicle_adapter.py


__all__ = [
    "PhaseType",
    "PhaseResult",
    "PhaseConfig",
    "WorkflowTemplate",
    "PhaseOutput",
    "PipelineResult",
]
