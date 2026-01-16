"""LLM Types - Orchestration Enums.

Enums for multi-LLM orchestration roles, clarity levels, message types,
and interaction types.
"""

from __future__ import annotations

from enum import Enum


class OrchestratorRole(Enum):
    """Orchestrator role in the hierarchy.

    Attributes:
        MASTER: Top-level orchestrator (typically Gemini)
        SUB: Delegated sub-orchestrator (handles specific domains)
    """
    MASTER = "master"
    SUB = "sub"


class WorkerRole(Enum):
    """Worker role specialization.

    Attributes:
        CODE_IMPLEMENTER: Code implementation (Claude's strength)
        DEEP_ANALYZER: Deep analysis and debugging (Codex's strength)
        STRATEGIST: Strategy and design (Gemini's strength)
        DEBUGGER: Debugging specialist (Codex)
        CLARIFIER: Clarification handler (any provider)
    """
    CODE_IMPLEMENTER = "code_implementer"
    DEEP_ANALYZER = "deep_analyzer"
    STRATEGIST = "strategist"
    DEBUGGER = "debugger"
    CLARIFIER = "clarifier"


class ClarityLevel(Enum):
    """Task clarity level assessment.

    Attributes:
        CLEAR: Task is clear, can execute immediately
        NEEDS_CLARIFICATION: Some aspects need clarification
        AMBIGUOUS: Seriously ambiguous, requires escalation
    """
    CLEAR = "clear"
    NEEDS_CLARIFICATION = "needs_clarification"
    AMBIGUOUS = "ambiguous"


class MessageType(Enum):
    """Inter-LLM message types.

    Categorized by purpose:
    - Task Flow: Assignment, results, escalation
    - Discussion: Brainstorming communication
    - Coordination: Clarification and permission
    - Control: System-level operations
    """
    # Task Flow
    TASK_ASSIGNMENT = "task_assignment"
    TASK_RESULT = "task_result"
    TASK_ESCALATION = "task_escalation"

    # Discussion
    OPINION = "opinion"
    REBUTTAL = "rebuttal"
    SUPPORT = "support"
    QUESTION = "question"
    ANSWER = "answer"
    CONSENSUS_PROPOSAL = "consensus_proposal"
    VOTE = "vote"

    # Coordination
    CLARIFICATION_REQUEST = "clarification_request"
    CLARIFICATION_RESPONSE = "clarification_response"
    PERMISSION_REQUEST = "permission_request"
    PERMISSION_RESPONSE = "permission_response"
    STATUS_UPDATE = "status_update"

    # Control
    ORCHESTRATOR_SWITCH = "orchestrator_switch"
    SESSION_STATE = "session_state"


class InteractionType(Enum):
    """Types of interactions in brainstorm discussion.

    Attributes:
        SUPPORT: Supporting another participant's position
        DEFENSE: Defending against criticism
        CRITIC: Criticizing another participant's position
        FREE_COMMENT: Free-form comment
    """
    SUPPORT = "support"
    DEFENSE = "defense"
    CRITIC = "critic"
    FREE_COMMENT = "free_comment"
