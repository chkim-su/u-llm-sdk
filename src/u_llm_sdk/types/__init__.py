"""LLM Types - Shared type definitions for LLM integrations.

This package provides the core types, configurations, and data structures
shared between U-llm-sdk (LLM execution) and MV-rag (experience learning/RAG).

Package Structure:
    - config.py: Configuration enums (Provider, ModelTier, AutoApproval, etc.)
    - models.py: Data models (LLMResult, TokenUsage, FileChange, etc.)
    - hooks.py: Hook data structures (PreActionContext, PostActionFeedback)
    - exceptions.py: Exception classes (UnifiedLLMError and subclasses)
    - schemas/: Domain schema package (DomainSchema, BrainstormSchema, etc.)

Schema Layering Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │  Domain Layer (Variable):                                   │
    │    - BrainstormSchema: Multi-LLM discussion outputs         │
    │    - CodeReviewSchema: Code analysis outputs (future)       │
    │                                                             │
    │  Global Layer (Always Active):                              │
    │    - MV-RAG Observer: Behavior logging (parallel)           │
    │    - Extracts: tokens, duration, success, tool_calls        │
    └─────────────────────────────────────────────────────────────┘

Usage:
    >>> from u_llm_sdk.types import Provider, ModelTier, LLMResult
    >>> from u_llm_sdk.types import PreActionContext, PostActionFeedback
    >>> from u_llm_sdk.types import UnifiedLLMError, ProviderNotFoundError
    >>> from u_llm_sdk.types.schemas import DomainSchema, BrainstormSchema
"""

__version__ = "0.1.0"

# Configuration enums
from .config import (
    Provider,
    AutoApproval,
    SandboxMode,
    ModelTier,
    ReasoningLevel,
    # Type aliases
    ProviderType,
    AutoApprovalType,
    SandboxModeType,
    ModelTierType,
    ReasoningLevelType,
    # Model tier mappings (January 2026)
    MODEL_TIERS,
    API_MODEL_TIERS,
    LEGACY_MODEL_TIERS,
    CURRENT_MODELS,
    TEST_DEV_MODELS,
    DEFAULT_MODELS,
    # Reasoning and approval mappings
    REASONING_MAP,
    CLAUDE_THINKING_TRIGGERS,
    DEPRECATED_THINKING_KEYWORDS,
    APPROVAL_MAP,
    # API and CLI mappings
    API_KEY_ENV_VARS,
    FILE_EDIT_TOOLS,
    FILE_WRITE_TOOLS,
    SHELL_TOOLS,
    CLI_COMMANDS,
    # Model resolution function
    resolve_model,
)

# Data models
from .models import (
    ResultType,
    FileChange,
    CommandRun,
    CodeBlock,
    TokenUsage,
    LLMResult,
    # Type aliases
    FileChanges,
    CommandRuns,
    CodeBlocks,
)

# Hook data structures
from .hooks import (
    PreActionContext,
    PostActionFeedback,
)

# Exceptions
from .exceptions import (
    UnifiedLLMError,
    ProviderNotFoundError,
    ProviderNotAvailableError,
    SessionNotFoundError,
    ExecutionTimeoutError,
    ExecutionCancelledError,
    InvalidConfigError,
    ParseError,
    AuthenticationError,
    RateLimitError,
    ModelNotFoundError,
    ModelNotSpecifiedError,
    RAGConnectionError,
    RAGTimeoutError,
)

# Feature support and validation
from .features import (
    Feature,
    Severity,
    FeatureSupport,
    FeatureValidationResult,
    FEATURE_SUPPORT_MATRIX,
    get_feature_support,
    get_providers_supporting,
    validate_feature,
    validate_config_features,
)

# Orchestration types (Multi-LLM coordination)
from .orchestration import (
    # Enums
    OrchestratorRole,
    WorkerRole,
    ClarityLevel,
    MessageType,
    InteractionType,
    OrchestrationMode,
    DelegationPhase,
    # Task & Clarity
    Task,
    UnclearAspect,
    ClarityAssessment,
    # Brainstorm & Consensus
    BrainstormConfig,
    ParticipantInput,
    DiscussionEntry,
    DissentingView,
    ConsensusEvaluation,
    ConsensusResult,
    # Discussion (Enhanced Brainstorming)
    ParticipantIdentity,
    SupportRecord,
    DefenseRecord,
    CriticRecord,
    FreeCommentRecord,
    DiscussionResponse,
    ParticipantContext,
    # Escalation
    EscalationRequest,
    EscalationResponse,
    # Delegation (ORIGINAL_STRICT / SEMI_AUTONOMOUS modes)
    BoundaryConstraints,
    ConfigurableOptions,
    ClaudeCodeDelegation,
    DelegationOutcome,
    # Inter-LLM Communication
    LLMMessage,
    # Session & State
    SessionConfig,
    OrchestratorState,
)

# Domain Schemas (Schema Layering)
from .schemas import (
    # Core types
    SchemaField,
    ValidationSeverity,
    ValidationResult,
    # Protocol
    DomainSchema,
    # Base implementation
    BaseDomainSchema,
    # Registry
    SchemaRegistry,
    get_schema,
    register_schema,
    list_schemas,
    # Brainstorm schema
    BrainstormSchema,
    BrainstormOutputType,
)

# Evidence (Multi-Provider Code Intelligence)
from .evidence import (
    # Types
    Stage,
    STAGE_TOKEN_BUDGETS,
    # Data classes
    EvidenceSpan,
    EvidenceChunk,
    # Protocol
    EvidenceProvider,
    # Utilities
    filter_chunks_by_provider,
    filter_chunks_by_kind,
    extract_seed_ids,
)

# Chronicle (Decision-Centric Chronicle for Multi-LLM Workflows)
from .chronicle import (
    # ID generation and validation
    RecordType,
    generate_record_id,
    validate_record_id,
    get_record_type,
    extract_event_id,
    # Source reference
    SourceKind,
    SourceReference,
    # Execution outcome
    ExecutionOutcome,
    # Error fingerprint
    ErrorFingerprint,
    # Primary records
    DecisionRecord,
    AmendRecord,
    ExecutionRecord,
    FailureRecord,
    EvidenceRecord,
    # Derived records
    BriefingGenerationParams,
    BriefingRecord,
    InquisitionRecord,
)

__all__ = [
    # Version
    "__version__",
    # Config enums
    "Provider",
    "AutoApproval",
    "SandboxMode",
    "ModelTier",
    "ReasoningLevel",
    "ProviderType",
    "AutoApprovalType",
    "SandboxModeType",
    "ModelTierType",
    "ReasoningLevelType",
    # Model tier mappings
    "MODEL_TIERS",
    "API_MODEL_TIERS",
    "LEGACY_MODEL_TIERS",
    "CURRENT_MODELS",
    "TEST_DEV_MODELS",
    "DEFAULT_MODELS",
    # Reasoning and approval mappings
    "REASONING_MAP",
    "CLAUDE_THINKING_TRIGGERS",
    "DEPRECATED_THINKING_KEYWORDS",
    "APPROVAL_MAP",
    # API and CLI mappings
    "API_KEY_ENV_VARS",
    "FILE_EDIT_TOOLS",
    "FILE_WRITE_TOOLS",
    "SHELL_TOOLS",
    "CLI_COMMANDS",
    # Model resolution function
    "resolve_model",
    # Models
    "ResultType",
    "FileChange",
    "CommandRun",
    "CodeBlock",
    "TokenUsage",
    "LLMResult",
    "FileChanges",
    "CommandRuns",
    "CodeBlocks",
    # Hooks
    "PreActionContext",
    "PostActionFeedback",
    # Exceptions
    "UnifiedLLMError",
    "ProviderNotFoundError",
    "ProviderNotAvailableError",
    "SessionNotFoundError",
    "ExecutionTimeoutError",
    "ExecutionCancelledError",
    "InvalidConfigError",
    "ParseError",
    "AuthenticationError",
    "RateLimitError",
    "ModelNotFoundError",
    "ModelNotSpecifiedError",
    "RAGConnectionError",
    "RAGTimeoutError",
    # Feature support
    "Feature",
    "Severity",
    "FeatureSupport",
    "FeatureValidationResult",
    "FEATURE_SUPPORT_MATRIX",
    "get_feature_support",
    "get_providers_supporting",
    "validate_feature",
    "validate_config_features",
    # Orchestration enums
    "OrchestratorRole",
    "WorkerRole",
    "ClarityLevel",
    "MessageType",
    "InteractionType",
    "OrchestrationMode",
    "DelegationPhase",
    # Orchestration: Task & Clarity
    "Task",
    "UnclearAspect",
    "ClarityAssessment",
    # Orchestration: Brainstorm & Consensus
    "BrainstormConfig",
    "ParticipantInput",
    "DiscussionEntry",
    "DissentingView",
    "ConsensusEvaluation",
    "ConsensusResult",
    # Orchestration: Discussion (Enhanced Brainstorming)
    "ParticipantIdentity",
    "SupportRecord",
    "DefenseRecord",
    "CriticRecord",
    "FreeCommentRecord",
    "DiscussionResponse",
    "ParticipantContext",
    # Orchestration: Escalation
    "EscalationRequest",
    "EscalationResponse",
    # Orchestration: Delegation
    "BoundaryConstraints",
    "ConfigurableOptions",
    "ClaudeCodeDelegation",
    "DelegationOutcome",
    # Orchestration: Inter-LLM Communication
    "LLMMessage",
    # Orchestration: Session & State
    "SessionConfig",
    "OrchestratorState",
    # Domain Schemas (Schema Layering)
    "SchemaField",
    "ValidationSeverity",
    "ValidationResult",
    "DomainSchema",
    "BaseDomainSchema",
    "SchemaRegistry",
    "get_schema",
    "register_schema",
    "list_schemas",
    "BrainstormSchema",
    "BrainstormOutputType",
    # Chronicle: ID generation and validation
    "RecordType",
    "generate_record_id",
    "validate_record_id",
    "get_record_type",
    "extract_event_id",
    # Chronicle: Source reference
    "SourceKind",
    "SourceReference",
    # Chronicle: Execution outcome
    "ExecutionOutcome",
    # Chronicle: Error fingerprint
    "ErrorFingerprint",
    # Chronicle: Primary records
    "DecisionRecord",
    "AmendRecord",
    "ExecutionRecord",
    "FailureRecord",
    "EvidenceRecord",
    # Chronicle: Derived records
    "BriefingGenerationParams",
    "BriefingRecord",
    "InquisitionRecord",
    # Evidence: Types
    "Stage",
    "STAGE_TOKEN_BUDGETS",
    # Evidence: Data classes
    "EvidenceSpan",
    "EvidenceChunk",
    # Evidence: Protocol
    "EvidenceProvider",
    # Evidence: Utilities
    "filter_chunks_by_provider",
    "filter_chunks_by_kind",
    "extract_seed_ids",
]
