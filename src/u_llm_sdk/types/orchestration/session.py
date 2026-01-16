"""LLM Types - Session and State Types.

Session configuration and orchestrator state management types.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import Provider
from .task import Task
from .brainstorm import BrainstormConfig, ConsensusResult
from .escalation import EscalationRequest


@dataclass
class SessionConfig:
    """Configuration for an orchestration session.

    Attributes:
        session_id: Unique session identifier
        orchestrator_provider: Which provider acts as master orchestrator
        brainstorm_config: Configuration for brainstorming
        max_consensus_rounds: Maximum rounds for consensus
        preserve_full_records: Keep full records (no summarization)
    """
    session_id: str
    orchestrator_provider: Provider
    brainstorm_config: BrainstormConfig = field(default_factory=BrainstormConfig)
    max_consensus_rounds: int = 3
    preserve_full_records: bool = True

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "orchestrator_provider": self.orchestrator_provider.value,
            "brainstorm_config": self.brainstorm_config.to_dict(),
            "max_consensus_rounds": self.max_consensus_rounds,
            "preserve_full_records": self.preserve_full_records,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SessionConfig:
        """Create from dictionary (JSON deserialization)."""
        orchestrator_value = data["orchestrator_provider"]
        orchestrator_provider = (
            Provider(orchestrator_value)
            if isinstance(orchestrator_value, str)
            else orchestrator_value
        )

        brainstorm_data = data.get("brainstorm_config", {})
        brainstorm_config = (
            BrainstormConfig.from_dict(brainstorm_data)
            if brainstorm_data
            else BrainstormConfig()
        )

        return cls(
            session_id=data["session_id"],
            orchestrator_provider=orchestrator_provider,
            brainstorm_config=brainstorm_config,
            max_consensus_rounds=data.get("max_consensus_rounds", 3),
            preserve_full_records=data.get("preserve_full_records", True),
        )


@dataclass
class OrchestratorState:
    """Current state of an orchestrator.

    Attributes:
        session_id: Session this state belongs to
        current_provider: Current orchestrator provider
        session_context: Accumulated session context
        active_tasks: Currently active tasks
        pending_escalations: Escalations awaiting response
        consensus_history: History of consensus results
    """
    session_id: str
    current_provider: Provider
    session_context: str
    active_tasks: list[Task] = field(default_factory=list)
    pending_escalations: list[EscalationRequest] = field(default_factory=list)
    consensus_history: list[ConsensusResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "current_provider": self.current_provider.value,
            "session_context": self.session_context,
            "active_tasks": [t.to_dict() for t in self.active_tasks],
            "pending_escalations": [e.to_dict() for e in self.pending_escalations],
            "consensus_history": [c.to_dict() for c in self.consensus_history],
        }

    @classmethod
    def from_dict(cls, data: dict) -> OrchestratorState:
        """Create from dictionary (JSON deserialization)."""
        current_provider_value = data["current_provider"]
        current_provider = (
            Provider(current_provider_value)
            if isinstance(current_provider_value, str)
            else current_provider_value
        )

        active_tasks = [Task.from_dict(t) for t in data.get("active_tasks", [])]
        pending_escalations = [
            EscalationRequest.from_dict(e) for e in data.get("pending_escalations", [])
        ]
        consensus_history = [
            ConsensusResult.from_dict(c) for c in data.get("consensus_history", [])
        ]

        return cls(
            session_id=data["session_id"],
            current_provider=current_provider,
            session_context=data.get("session_context", ""),
            active_tasks=active_tasks,
            pending_escalations=pending_escalations,
            consensus_history=consensus_history,
        )
