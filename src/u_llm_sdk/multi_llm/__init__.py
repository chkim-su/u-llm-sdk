"""Multi-LLM Orchestration Module.

This module provides components for coordinating multiple LLM providers
in complex tasks. The architecture follows a Master-Worker pattern where
Gemini typically serves as the orchestrator, Claude as the code worker,
and Codex as the deep analyzer.

Key Components:
    - ClarityGate: Worker self-assessment before task execution
    - EscalationProtocol: Upward communication when tasks are unclear
    - ConsensusLoop: 3-round majority voting for complex decisions
    - BrainstormModule: Multi-provider discussion and consensus building
    - Orchestrators: GeminiOrchestrator (master), ClaudeOrchestrator, CodexOrchestrator
    - OrchestratorFactory: Creates orchestrators based on provider type
    - HybridOrchestrator: Combines multi-LLM with MergeExecutor
    - EnhancedMasterOrchestrator: Mode-aware orchestrator (ORIGINAL_STRICT / SEMI_AUTONOMOUS)
    - ClaudeCodeExecutor: Executes Claude Code in autonomous delegation mode
    - MultiLLMRAGHook: RAG integration for pattern learning
    - PerformanceOptimizer: Parallel execution and latency tracking
    - OrchestrationLogger: Structured logging and monitoring

Orchestration Modes:
    - ORIGINAL_STRICT: Full master control, mandatory ClarityGate, Claude as worker
    - SEMI_AUTONOMOUS: Design via brainstorm, implementation delegated to Claude Code

Usage:
    >>> from u_llm_sdk.multi_llm import OrchestratorFactory, ClarityGate
    >>> from u_llm_sdk.types import Provider
    >>>
    >>> # Create master orchestrator (Gemini by default)
    >>> factory = OrchestratorFactory(providers)
    >>> master = factory.create_master()
    >>>
    >>> # Process user request
    >>> response = await master.process_request("Build auth system")
    >>> if response.needs_brainstorm:
    ...     consensus = await master.facilitate_brainstorm(response.brainstorm_topic)
    >>>
    >>> # Or use enhanced orchestrator with modes
    >>> from u_llm_sdk.multi_llm import EnhancedMasterOrchestrator
    >>> from u_llm_sdk.types import OrchestrationMode, BoundaryConstraints
    >>>
    >>> orchestrator = EnhancedMasterOrchestrator(providers)
    >>> result = await orchestrator.run(
    ...     "Implement auth",
    ...     cwd="/project",
    ...     mode=OrchestrationMode.SEMI_AUTONOMOUS,
    ...     boundaries=BoundaryConstraints(max_budget_usd=2.0),
    ... )
    >>>
    >>> # Performance tracking
    >>> from u_llm_sdk.multi_llm import PerformanceOptimizer
    >>> optimizer = PerformanceOptimizer()
    >>> async with optimizer.track_latency(Provider.CLAUDE):
    ...     result = await claude.run(prompt)
"""

from .clarity import ClarityGate
from .escalation import EscalationProtocol
from .consensus import ConsensusLoop
from .brainstorm import BrainstormModule
from .orchestrator import (
    # Protocol and base
    MasterOrchestrator,
    BaseOrchestrator,
    # Orchestrator implementations
    GeminiOrchestrator,
    ClaudeOrchestrator,
    CodexOrchestrator,
    # Factory
    OrchestratorFactory,
    # Data classes
    TaskType,
    OrchestratorResponse,
    TaskRouting,
    WorkerResult,
    AggregatedResult,
    # Utilities
    create_default_orchestrator,
    get_specialization_knowledge,
    SPECIALIZATION_KNOWLEDGE,
)
from .rag_integration import (
    # RAG-specific task type (for logging, distinct from routing TaskType)
    TaskType as RAGTaskType,
    # Data classes
    RoutingDecision,
    BrainstormOutcome,
    EscalationOutcome,
    PatternMatch,
    OrchestrationHint,
    # Hooks
    MultiLLMRAGHook,
    NoOpMultiLLMRAGHook,
)
from .migration import (
    # Enums
    ExecutionMode,
    # Results
    HybridExecutionResult,
    # Main classes
    HybridOrchestrator,
    StateRecoveryManager,
    MigrationHelper,
)
from .performance import (
    # Metrics
    MetricType,
    LatencyStats,
    ThroughputStats,
    MemoryStats,
    PerformanceMetrics,
    # Tracking
    LatencyTracker,
    # Execution
    TaskResult,
    ParallelExecutor,
    # Memory
    MemoryManager,
    # Resources
    ResourcePool,
    # Optimization
    PerformanceOptimizer,
    # Batch processing
    BatchProcessor,
)
from .monitoring import (
    # Log levels and events
    LogLevel,
    EventType,
    # Log entries
    LogEntry,
    EventRecord,
    # Logging
    OrchestrationLogger,
    # Debug
    DebugMode,
    debug_trace,
    # Export
    MetricExporter,
    # Events
    EventEmitter,
    # Dashboard
    DashboardData,
    MonitoringDashboard,
)
from .session import (
    # Protocol
    SessionStore,
    # Exceptions
    SessionError,
    SessionExistsError,
    SessionNotFoundError,
    # Store implementations
    InMemorySessionStore,
    FileSessionStore,
    # Manager
    SessionManager,
)
from .context_builder import (
    # State
    DiscussionState,
    # Context building
    DiscussionContextBuilder,
    # Speaker rotation
    SpeakerRotation,
    # ID generation
    IDGenerator,
    # Factory
    create_participant_identity,
)
from .boundary_validation import (
    # Exceptions
    BoundaryViolationError,
    # Trackers
    BudgetTracker,
    FileScopeValidator,
    FileModificationTracker,
    BoundaryValidator,
)
from .claude_executor import (
    # Event parsing
    StreamEvent,
    # Prompt building
    build_delegation_prompt,
    # Executor
    ClaudeCodeExecutor,
    # Availability flag
    CLAUDE_SDK_AVAILABLE,
)

# Re-export claude-only-sdk session utilities if available
if CLAUDE_SDK_AVAILABLE:
    from claude_only_sdk.session import SessionTemplate, get_template_prompt
else:
    SessionTemplate = None  # type: ignore
    get_template_prompt = None  # type: ignore
from .enhanced_orchestrator import (
    # Result types
    EnhancedOrchestratorResult,
    # Main orchestrator
    EnhancedMasterOrchestrator,
)

__all__ = [
    # Core components
    "ClarityGate",
    "EscalationProtocol",
    "ConsensusLoop",
    "BrainstormModule",
    # Protocol and base
    "MasterOrchestrator",
    "BaseOrchestrator",
    # Orchestrator implementations
    "GeminiOrchestrator",
    "ClaudeOrchestrator",
    "CodexOrchestrator",
    # Factory
    "OrchestratorFactory",
    # Data classes (orchestrator)
    "TaskType",
    "OrchestratorResponse",
    "TaskRouting",
    "WorkerResult",
    "AggregatedResult",
    # Utilities
    "create_default_orchestrator",
    "get_specialization_knowledge",
    "SPECIALIZATION_KNOWLEDGE",
    # RAG integration
    "RAGTaskType",
    "RoutingDecision",
    "BrainstormOutcome",
    "EscalationOutcome",
    "PatternMatch",
    "OrchestrationHint",
    "MultiLLMRAGHook",
    "NoOpMultiLLMRAGHook",
    # Migration / Hybrid
    "ExecutionMode",
    "HybridExecutionResult",
    "HybridOrchestrator",
    "StateRecoveryManager",
    "MigrationHelper",
    # Performance (Phase 5)
    "MetricType",
    "LatencyStats",
    "ThroughputStats",
    "MemoryStats",
    "PerformanceMetrics",
    "LatencyTracker",
    "TaskResult",
    "ParallelExecutor",
    "MemoryManager",
    "ResourcePool",
    "PerformanceOptimizer",
    "BatchProcessor",
    # Monitoring (Phase 5)
    "LogLevel",
    "EventType",
    "LogEntry",
    "EventRecord",
    "OrchestrationLogger",
    "DebugMode",
    "debug_trace",
    "MetricExporter",
    "EventEmitter",
    "DashboardData",
    "MonitoringDashboard",
    # Session (Section 07 completion)
    "SessionStore",
    "SessionError",
    "SessionExistsError",
    "SessionNotFoundError",
    "InMemorySessionStore",
    "FileSessionStore",
    "SessionManager",
    # Context Builder (Enhanced Brainstorming)
    "DiscussionState",
    "DiscussionContextBuilder",
    "SpeakerRotation",
    "IDGenerator",
    "create_participant_identity",
    # Boundary Validation (SEMI_AUTONOMOUS)
    "BoundaryViolationError",
    "BudgetTracker",
    "FileScopeValidator",
    "FileModificationTracker",
    "BoundaryValidator",
    # Claude Executor (SEMI_AUTONOMOUS)
    "StreamEvent",
    "build_delegation_prompt",
    "ClaudeCodeExecutor",
    "CLAUDE_SDK_AVAILABLE",
    # claude-only-sdk re-exports (if available)
    "SessionTemplate",
    "get_template_prompt",
    # Enhanced Orchestrator (ORIGINAL_STRICT / SEMI_AUTONOMOUS)
    "EnhancedOrchestratorResult",
    "EnhancedMasterOrchestrator",
]
