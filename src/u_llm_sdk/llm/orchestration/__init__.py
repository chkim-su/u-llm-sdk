"""Multi-provider orchestration.

This module provides the orchestration framework for multi-agent execution:
- types: Execution states and context
- contracts: WorkOrder, Evidence, ReviewReport definitions
- verifier: Result verification system
- checkpoint: Auditable execution checkpoints
- lock_manager: Resource locking for parallel editing
- context_builder: WorkOrder context construction
- delta_compliance: Code change compliance verification
- patch_submission: Patch-based change submission
- evidence_executor: Test/typecheck/lint execution
- agent_executor: Role-based LLM execution
"""

from u_llm_sdk.llm.orchestration.agent_executor import AgentExecutor
from u_llm_sdk.llm.orchestration.checkpoint import (
    Checkpoint,
    attach_checkpoint,
    create_checkpoint,
    preferred_text,
)
from u_llm_sdk.llm.orchestration.context_builder import (
    ContextBuilderConfig,
    ContextItem,
    ContextPack,
    ContextPriority,
    PrioritizedContextItem,
    WorkOrderContextBuilder,
    build_context_for_work_order,
    detect_indexer_from_files,
    extract_files_from_file_set,
    extract_seeds_from_work_order,
)
from u_llm_sdk.llm.orchestration.contracts import (
    ROLES,
    Capability,
    Evidence,
    EvidenceRequirement,
    ExecutionPlan,
    ExpectedDelta,
    ReviewReport,
    RoleSpec,
    SideEffect,
    WorkOrder,
)
from u_llm_sdk.llm.orchestration.delta_compliance import (
    ComplianceResult,
    DeltaAnalyzer,
    DeltaComplianceVerifier,
    DeltaViolation,
    DependencyEdge,
    ExportedSymbol,
    ModuleSnapshot,
    PythonAnalyzer,
    RepoSnapshot,
    SymbolType,
    TypeScriptAnalyzer,
    get_public_exports,
    quick_delta_check,
)
from u_llm_sdk.llm.orchestration.evidence_executor import (
    CommandExecutionResult,
    EvidenceExecutionResult,
    EvidenceExecutor,
    get_evidence_executor,
)
from u_llm_sdk.llm.orchestration.merge_executor import (
    ClarifiedTask,
    MergeExecutor,
    MergeExecutorConfig,
    MergeExecutorResult,
    merge_execute,
    merge_execute_sync,
)
from u_llm_sdk.llm.orchestration.state_machine import StateMachineOrchestrator
from u_llm_sdk.llm.orchestration.file_scope import (
    CreateFilesCollision,
    FileSetExpansion,
    FileSetManager,
    IntersectionResult,
    validate_create_files_disjoint,
    validate_create_vs_file_set,
    validate_work_orders_disjoint,
)
from u_llm_sdk.llm.orchestration.git_integration import (
    BranchInfo,
    GitManager,
    GitResult,
    MergeResult,
    MergeStrategy,
    WorktreeInfo,
    WorktreeSetupResult,
    cleanup_work_order_branch,
    cleanup_work_order_worktree,
    commit_work_order_changes,
    commit_work_order_in_worktree,
    get_worktree_snapshot_info,
    merge_work_order,
    parse_work_order_branch,
    setup_work_order_branch,
    setup_work_order_worktree,
    work_order_branch_name,
    worktree_path_for_work_order,
)
from u_llm_sdk.llm.orchestration.lock_manager import (
    InMemoryLockManager,
    LockAcquisitionResult,
    LockInfo,
    LockManager,
    RedisLockManager,
    compute_layer_locks,
    validate_resource_keys,
)
from u_llm_sdk.llm.orchestration.patch_submission import (
    PatchSubmission,
    PatchSubmissionResult,
    PatchSubmitter,
    apply_patch_to_branch,
    create_patch_from_diff,
    generate_worktree_patch,
)
from u_llm_sdk.llm.orchestration.types import (
    Event,
    EventType,
    ExecutionContext,
    ExecutionState,
    WorkOrderState,
)
from u_llm_sdk.llm.orchestration.verifier import (
    CallbackVerifier,
    CompositeVerifier,
    FileExistsVerifier,
    FileSetComplianceVerifier,
    LLMVerifier,
    VerificationResult,
    Verifier,
    WorkOrderComplianceVerifier,
    normalize_verifier,
)

__all__ = [
    # Types
    "ExecutionState",
    "WorkOrderState",
    "Event",
    "EventType",
    "ExecutionContext",
    # Contracts
    "SideEffect",
    "Capability",
    "RoleSpec",
    "ROLES",
    "ExpectedDelta",
    "EvidenceRequirement",
    "WorkOrder",
    "Evidence",
    "ReviewReport",
    "ExecutionPlan",
    # Verifiers
    "VerificationResult",
    "Verifier",
    "LLMVerifier",
    "CallbackVerifier",
    "FileExistsVerifier",
    "FileSetComplianceVerifier",
    "WorkOrderComplianceVerifier",
    "CompositeVerifier",
    "normalize_verifier",
    # Checkpoint
    "Checkpoint",
    "preferred_text",
    "create_checkpoint",
    "attach_checkpoint",
    # Lock Manager
    "LockInfo",
    "LockAcquisitionResult",
    "LockManager",
    "InMemoryLockManager",
    "RedisLockManager",
    "validate_resource_keys",
    "compute_layer_locks",
    # File Scope
    "FileSetExpansion",
    "IntersectionResult",
    "FileSetManager",
    "CreateFilesCollision",
    "validate_create_files_disjoint",
    "validate_create_vs_file_set",
    "validate_work_orders_disjoint",
    # Git Integration
    "MergeStrategy",
    "GitResult",
    "MergeResult",
    "BranchInfo",
    "WorktreeInfo",
    "WorktreeSetupResult",
    "GitManager",
    "work_order_branch_name",
    "parse_work_order_branch",
    "worktree_path_for_work_order",
    "setup_work_order_branch",
    "commit_work_order_changes",
    "merge_work_order",
    "cleanup_work_order_branch",
    "setup_work_order_worktree",
    "commit_work_order_in_worktree",
    "cleanup_work_order_worktree",
    "get_worktree_snapshot_info",
    # Context Builder
    "ContextBuilderConfig",
    "ContextItem",
    "ContextPack",
    "ContextPriority",
    "PrioritizedContextItem",
    "WorkOrderContextBuilder",
    "build_context_for_work_order",
    "detect_indexer_from_files",
    "extract_files_from_file_set",
    "extract_seeds_from_work_order",
    # Delta Compliance
    "SymbolType",
    "ExportedSymbol",
    "DependencyEdge",
    "ModuleSnapshot",
    "RepoSnapshot",
    "DeltaViolation",
    "ComplianceResult",
    "PythonAnalyzer",
    "TypeScriptAnalyzer",
    "DeltaAnalyzer",
    "DeltaComplianceVerifier",
    "quick_delta_check",
    "get_public_exports",
    # Patch Submission
    "PatchSubmission",
    "PatchSubmissionResult",
    "PatchSubmitter",
    "generate_worktree_patch",
    "create_patch_from_diff",
    "apply_patch_to_branch",
    # Evidence Executor
    "CommandExecutionResult",
    "EvidenceExecutionResult",
    "EvidenceExecutor",
    "get_evidence_executor",
    # Agent Executor
    "AgentExecutor",
    # State Machine
    "StateMachineOrchestrator",
    # Merge Executor
    "ClarifiedTask",
    "MergeExecutor",
    "MergeExecutorConfig",
    "MergeExecutorResult",
    "merge_execute",
    "merge_execute_sync",
]
