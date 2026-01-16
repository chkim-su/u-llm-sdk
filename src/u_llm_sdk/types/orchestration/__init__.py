"""LLM Types - Orchestration Package.

Re-exports all orchestration types for backward compatibility.

Usage:
    from u_llm_sdk.types.orchestration import (
        # Enums
        OrchestratorRole, WorkerRole, ClarityLevel, MessageType, InteractionType,
        OrchestrationMode, DelegationPhase,
        # Task types
        Task, UnclearAspect, ClarityAssessment,
        # Brainstorm types
        BrainstormConfig, ParticipantInput, DiscussionEntry,
        DissentingView, ConsensusEvaluation, ConsensusResult,
        # Discussion types (enhanced brainstorming)
        ParticipantIdentity, SupportRecord, DefenseRecord,
        CriticRecord, FreeCommentRecord, DiscussionResponse, ParticipantContext,
        # Escalation types
        EscalationRequest, EscalationResponse,
        # Delegation types (ORIGINAL_STRICT / SEMI_AUTONOMOUS modes)
        BoundaryConstraints, ConfigurableOptions,
        ClaudeCodeDelegation, DelegationOutcome,
        # Message types
        LLMMessage,
        # Session types
        SessionConfig, OrchestratorState,
    )
"""

# Enums
from .enums import (
    OrchestratorRole,
    WorkerRole,
    ClarityLevel,
    MessageType,
    InteractionType,
)

# Task & Clarity types
from .task import (
    Task,
    UnclearAspect,
    ClarityAssessment,
)

# Brainstorm & Consensus types
from .brainstorm import (
    BrainstormConfig,
    ParticipantInput,
    DiscussionEntry,
    DissentingView,
    ConsensusEvaluation,
    ConsensusResult,
)

# Discussion types (enhanced brainstorming)
from .discussion import (
    ParticipantIdentity,
    SupportRecord,
    DefenseRecord,
    CriticRecord,
    FreeCommentRecord,
    DiscussionResponse,
    ParticipantContext,
)

# Escalation types
from .escalation import (
    EscalationRequest,
    EscalationResponse,
)

# Message types
from .message import (
    LLMMessage,
)

# Session & State types
from .session import (
    SessionConfig,
    OrchestratorState,
)

# Delegation types (ORIGINAL_STRICT / SEMI_AUTONOMOUS modes)
from .delegation import (
    OrchestrationMode,
    DelegationPhase,
    BoundaryConstraints,
    ConfigurableOptions,
    ClaudeCodeDelegation,
    DelegationOutcome,
)

__all__ = [
    # Enums
    "OrchestratorRole",
    "WorkerRole",
    "ClarityLevel",
    "MessageType",
    "InteractionType",
    "OrchestrationMode",
    "DelegationPhase",
    # Task types
    "Task",
    "UnclearAspect",
    "ClarityAssessment",
    # Brainstorm types
    "BrainstormConfig",
    "ParticipantInput",
    "DiscussionEntry",
    "DissentingView",
    "ConsensusEvaluation",
    "ConsensusResult",
    # Discussion types
    "ParticipantIdentity",
    "SupportRecord",
    "DefenseRecord",
    "CriticRecord",
    "FreeCommentRecord",
    "DiscussionResponse",
    "ParticipantContext",
    # Escalation types
    "EscalationRequest",
    "EscalationResponse",
    # Delegation types
    "BoundaryConstraints",
    "ConfigurableOptions",
    "ClaudeCodeDelegation",
    "DelegationOutcome",
    # Message types
    "LLMMessage",
    # Session types
    "SessionConfig",
    "OrchestratorState",
]
