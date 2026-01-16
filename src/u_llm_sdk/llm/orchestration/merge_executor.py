"""MergeExecutor - Full Pipeline with Integration Branch Pattern.

Implements a complete pipeline for parallel code editing with:
1. Task clarification
2. WorkOrder decomposition
3. Parallel execution in isolated worktrees
4. Per-branch validation
5. Integration branch merging
6. Final validation and push/PR creation
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from u_llm_sdk.types import ModelTier, Provider

from .agent_executor import AgentExecutor
from .contracts import ROLES, Capability, RoleSpec, SideEffect
from .evidence_executor import EvidenceExecutionResult, get_evidence_executor
from .types import ExecutionContext, WorkOrderState

if TYPE_CHECKING:
    from .git_integration import GitManager


# Optional: WorkOrder context builder for thin context (SCIP-first)
try:
    from .context_builder import (
        ContextBuilderConfig,
        WorkOrderContextBuilder,
        detect_indexer_from_files,
    )

    _HAS_CONTEXT_BUILDER = True
except ImportError:
    _HAS_CONTEXT_BUILDER = False
    WorkOrderContextBuilder = None  # type: ignore
    ContextBuilderConfig = None  # type: ignore
    detect_indexer_from_files = None  # type: ignore


# =============================================================================
# MergeExecutor Data Classes
# =============================================================================


@dataclass
class ClarifiedTask:
    """Result of task clarification phase.

    Transforms vague instructions into executable contracts.

    Attributes:
        original_instruction: Original user instruction
        objective: Clarified objective statement
        acceptance_criteria: Measurable completion criteria
        constraints: Explicit constraints and boundaries
        risks: Identified risks and mitigations
        out_of_scope: Explicitly excluded items
        estimated_work_orders: Expected number of WorkOrders
    """

    original_instruction: str
    objective: str
    acceptance_criteria: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    out_of_scope: list[str] = field(default_factory=list)
    estimated_work_orders: int = 1

    def to_prompt_context(self) -> str:
        """Convert to context string for subsequent phases."""
        lines = [
            f"## Clarified Objective\n{self.objective}",
            "\n## Acceptance Criteria",
            *[f"- {c}" for c in self.acceptance_criteria],
            "\n## Constraints",
            *[f"- {c}" for c in self.constraints],
        ]
        if self.risks:
            lines.append("\n## Identified Risks")
            lines.extend([f"- {r}" for r in self.risks])
        if self.out_of_scope:
            lines.append("\n## Out of Scope")
            lines.extend([f"- {o}" for o in self.out_of_scope])
        return "\n".join(lines)


@dataclass
class MergeExecutorConfig:
    """Configuration for MergeExecutor.

    Attributes:
        base_branch: Branch to base work from (default: main)
        integration_branch: Branch to collect passed merges (default: llm/integration)
        auto_push: Whether to push after final validation
        create_pr: Whether to create PR instead of direct push
        max_parallel: Maximum parallel WorkOrders
        require_all_pass: Require all WorkOrders to pass for push
        cleanup_on_success: Delete worktrees/branches after success
        cleanup_on_failure: Delete worktrees/branches after failure

        planner_provider: Provider for planning (default: CLAUDE)
        editor_provider: Provider for editing (default: GEMINI)
        supervisor_provider: Provider for review (default: CLAUDE)
        clarifier_provider: Provider for clarification (default: CLAUDE)
    """

    base_branch: str = "main"
    integration_branch: str = "llm/integration"
    auto_push: bool = False
    create_pr: bool = False
    max_parallel: int = 3
    require_all_pass: bool = True
    cleanup_on_success: bool = True
    cleanup_on_failure: bool = False

    planner_provider: Provider = Provider.CLAUDE
    editor_provider: Provider = Provider.GEMINI
    supervisor_provider: Provider = Provider.CLAUDE
    clarifier_provider: Provider = Provider.CLAUDE

    # Evidence requirements
    require_tests: bool = True
    require_typecheck: bool = True
    require_lint: bool = True

    # Context builder settings (SCIP-first thin context)
    enable_context_builder: bool = True
    context_max_chars: int = 50000  # ~12.5k tokens
    context_max_symbols: int = 40
    context_indexer: str = "auto"  # "auto", "scip-python", "scip-typescript"
    context_language: str = "auto"  # "auto", "python", "typescript", "javascript"

    # Patch-based submission (worktree -> diff -> integration)
    # When enabled, changes are submitted as patches instead of branch merges
    use_patch_submission: bool = True


@dataclass
class MergeExecutorResult:
    """Result of MergeExecutor run.

    Attributes:
        success: Whether execution completed successfully
        clarified_task: Clarified task from first phase
        execution_context: Full execution context from orchestrator
        integration_branch: Name of integration branch
        integration_commit: Final commit on integration branch
        pushed: Whether changes were pushed
        pr_url: URL of created PR (if create_pr=True)
        final_evidence: Evidence from final validation
        error: Error message if failed
    """

    success: bool
    clarified_task: Optional[ClarifiedTask] = None
    execution_context: Optional[ExecutionContext] = None
    integration_branch: str = ""
    integration_commit: str = ""
    pushed: bool = False
    pr_url: str = ""
    final_evidence: Optional[EvidenceExecutionResult] = None
    error: str = ""

    def to_summary(self) -> str:
        """Generate human-readable summary."""
        if not self.success:
            return f"MergeExecutor failed: {self.error}"

        lines = ["MergeExecutor completed successfully"]

        if self.execution_context:
            summary = self.execution_context.to_summary()
            lines.append(
                f"  WorkOrders: {summary['work_orders_completed']}/"
                f"{summary['work_orders_total']} completed"
            )
            if summary["work_orders_failed"] > 0:
                lines.append(f"  Failed: {summary['work_orders_failed']}")

        lines.append(f"  Integration branch: {self.integration_branch}")

        if self.integration_commit:
            lines.append(f"  Integration commit: {self.integration_commit[:8]}")

        if self.pushed:
            lines.append("  Pushed to remote")

        if self.pr_url:
            lines.append(f"  PR: {self.pr_url}")

        return "\n".join(lines)


# =============================================================================
# MergeExecutor Class
# =============================================================================


class MergeExecutor:
    """Full pipeline executor with integration branch pattern.

    Implements the recipe:
    1. Clarify: Task -> Acceptance criteria + constraints
    2. Split: Create WorkOrders with file_set/locks/expected_delta
    3. Parallel Edit: Execute in isolated worktrees
    4. Validate: Per-branch Evidence + Delta + Guardrails
    5. Merge: Passed branches -> integration branch
    6. Final Validate: Run Evidence on integration branch
    7. Push: Push to remote or create PR

    Key differences from StateMachineOrchestrator:
    - Explicit clarification phase before planning
    - Integration branch pattern (not direct to main)
    - Final validation on merged result
    - Push/PR creation step

    Example:
        >>> config = MergeExecutorConfig(
        ...     integration_branch="llm/feature-auth",
        ...     auto_push=True,
        ... )
        >>> executor = MergeExecutor(config)
        >>> result = await executor.run(
        ...     "Implement user authentication system",
        ...     cwd="/project",
        ... )
        >>> print(result.to_summary())
    """

    def __init__(self, config: Optional[MergeExecutorConfig] = None):
        """Initialize MergeExecutor.

        Args:
            config: Configuration (uses defaults if None)
        """
        self.config = config or MergeExecutorConfig()

        # Initialize agent executor with configured providers
        self.agent_executor = AgentExecutor(
            provider_map={
                "planner": self.config.planner_provider,
                "editor": self.config.editor_provider,
                "supervisor": self.config.supervisor_provider,
                "aggregator": self.config.planner_provider,
                "clarifier": self.config.clarifier_provider,
            },
            tier_map={
                "planner": ModelTier.HIGH,
                "editor": ModelTier.LOW,
                "supervisor": ModelTier.HIGH,
                "aggregator": ModelTier.LOW,
                "clarifier": ModelTier.HIGH,
            },
        )

        # Add clarifier role to ROLES
        self._ensure_clarifier_role()

    def _ensure_clarifier_role(self):
        """Ensure clarifier role exists in ROLES."""
        if "clarifier" not in ROLES:
            ROLES["clarifier"] = RoleSpec(
                name="Task Clarifier",
                purpose="Transform vague instructions into executable contracts. "
                "Clarify objectives, define completion criteria, "
                "and document constraints/risks/scope.",
                capabilities={Capability.FILE_READ, Capability.CODE_SEARCH},
                side_effects=SideEffect.NONE,
                artifacts_produced=["ClarifiedTask"],
                forbidden=["Code modification", "Direct execution", "Scope expansion"],
            )

    async def run(
        self,
        instruction: str,
        cwd: str,
        context: Optional[str] = None,
    ) -> MergeExecutorResult:
        """Run full MergeExecutor pipeline.

        Args:
            instruction: Task instruction from user
            cwd: Working directory (repository root)
            context: Additional context

        Returns:
            MergeExecutorResult with all execution details
        """
        # Import here to avoid circular imports
        from pathlib import Path

        from .git_integration import GitManager
        from .state_machine import StateMachineOrchestrator

        result = MergeExecutorResult(success=False)

        try:
            # Initialize git manager
            git_manager = GitManager(cwd)

            # Phase 0: Fix base commit for reproducibility
            base_commit = git_manager.get_commit_hash(self.config.base_branch)

            # Phase 1: Clarify task
            clarified = await self._clarify_task(instruction, cwd, context)
            result.clarified_task = clarified

            # Phase 2-4: Run StateMachineOrchestrator
            # (Planning, Dispatch, Review, Merge per-WorkOrder)

            # Create context builder if enabled (SCIP-first thin context)
            context_builder = None
            if self.config.enable_context_builder and _HAS_CONTEXT_BUILDER:
                try:
                    # Determine indexer/language (auto-detect or use config)
                    indexer = self.config.context_indexer
                    language = self.config.context_language

                    if indexer == "auto" or language == "auto":
                        # Auto-detect from common file patterns in cwd
                        cwd_path = Path(cwd)
                        patterns = []
                        for ext in [".py", ".ts", ".tsx", ".js", ".jsx"]:
                            if list(cwd_path.glob(f"**/*{ext}"))[:1]:
                                patterns.append(f"*{ext}")
                        if patterns and detect_indexer_from_files:
                            detected_indexer, detected_lang = detect_indexer_from_files(
                                patterns
                            )
                            if indexer == "auto":
                                indexer = detected_indexer
                            if language == "auto":
                                language = detected_lang
                        else:
                            # Fallback defaults
                            if indexer == "auto":
                                indexer = "scip-python"
                            if language == "auto":
                                language = "python"

                    # Context builder creation would go here
                    # (SCIP backend deferred to future milestone)
                    context_builder = None
                except Exception:
                    # Context builder creation failed - proceed without
                    context_builder = None

            orchestrator = StateMachineOrchestrator(
                agent_executor=self.agent_executor,
                max_parallel_editors=self.config.max_parallel,
                context_builder=context_builder,
                use_patch_submission=self.config.use_patch_submission,
            )

            # Override main branch to integration branch for merges
            git_manager.main_branch = self.config.integration_branch

            # Prepare integration branch from base
            self._prepare_integration_branch(git_manager, base_commit)

            # Run orchestrator with clarified context
            ctx = await orchestrator.run(
                objective=clarified.objective,
                cwd=cwd,
                context=clarified.to_prompt_context(),
            )
            result.execution_context = ctx
            result.integration_branch = self.config.integration_branch

            # Check if any WorkOrders succeeded
            merged_count = sum(
                1
                for s in ctx.work_order_states.values()
                if s == WorkOrderState.MERGED
            )

            if merged_count == 0:
                result.error = "No WorkOrders were successfully merged"
                return result

            if self.config.require_all_pass:
                total = len(ctx.work_order_states)
                if merged_count < total:
                    result.error = (
                        f"Only {merged_count}/{total} WorkOrders passed "
                        "(require_all_pass=True)"
                    )
                    return result

            # Phase 5: Final validation on integration branch
            final_evidence = await self._final_validation(cwd)
            result.final_evidence = final_evidence

            # Check final validation
            if not self._check_final_evidence(final_evidence):
                result.error = "Final validation failed on integration branch"
                return result

            # Get integration commit
            result.integration_commit = git_manager.get_commit_hash(
                self.config.integration_branch
            )

            # Phase 6: Push or create PR
            if self.config.auto_push:
                push_result = self._push_integration(git_manager)
                result.pushed = push_result.success
                if not push_result.success:
                    result.error = f"Push failed: {push_result.error}"
                    return result

            if self.config.create_pr:
                pr_url = await self._create_pr(git_manager, clarified, ctx)
                result.pr_url = pr_url

            result.success = True

        except Exception as e:
            result.error = str(e)

        return result

    async def _clarify_task(
        self,
        instruction: str,
        cwd: str,
        context: Optional[str] = None,
    ) -> ClarifiedTask:
        """Phase 1: Clarify task into executable contract.

        Args:
            instruction: Original user instruction
            cwd: Working directory
            context: Additional context

        Returns:
            ClarifiedTask with acceptance criteria and constraints
        """
        clarify_prompt = f"""## Original Instruction
{instruction}

## Task
Transform this instruction into an "executable contract".

### Clarification Checklist
1. **Objective**: Specific goal statement with no ambiguity
2. **Completion Criteria**: Measurable completion conditions (each must be YES/NO)
3. **Constraints**: Explicit prohibitions and scope limits
4. **Risks**: Expected issues and mitigations
5. **Out of Scope**: Explicitly excluded items

## Output Format (JSON)
{{
  "objective": "Clarified objective",
  "acceptance_criteria": [
    "Completion criterion 1 (measurable)",
    "Completion criterion 2"
  ],
  "constraints": [
    "Constraint 1",
    "Constraint 2"
  ],
  "risks": [
    "Risk 1: mitigation",
    "Risk 2: mitigation"
  ],
  "out_of_scope": [
    "Out of scope item 1",
    "Out of scope item 2"
  ],
  "estimated_work_orders": 3
}}

Output ONLY the JSON."""

        result = await self.agent_executor.execute(
            role="clarifier",
            prompt=clarify_prompt,
            context=context,
            cwd=cwd,
            timeout=120.0,
        )

        # Parse clarified task
        return self._parse_clarified_task(instruction, result.text or "")

    def _parse_clarified_task(self, instruction: str, text: str) -> ClarifiedTask:
        """Parse ClarifiedTask from LLM response."""
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                return ClarifiedTask(
                    original_instruction=instruction,
                    objective=data.get("objective", instruction),
                    acceptance_criteria=data.get("acceptance_criteria", []),
                    constraints=data.get("constraints", []),
                    risks=data.get("risks", []),
                    out_of_scope=data.get("out_of_scope", []),
                    estimated_work_orders=data.get("estimated_work_orders", 1),
                )
        except (json.JSONDecodeError, KeyError):
            pass

        # Fallback: use instruction as objective
        return ClarifiedTask(
            original_instruction=instruction,
            objective=instruction,
        )

    def _prepare_integration_branch(
        self,
        git_manager: "GitManager",
        base_commit: str,
    ):
        """Prepare integration branch from base commit.

        Creates fresh integration branch from base_branch.
        Deletes existing integration branch if present.

        Args:
            git_manager: GitManager instance
            base_commit: Commit to base integration branch on
        """
        integration_branch = self.config.integration_branch

        # Delete existing integration branch if it exists
        if git_manager.branch_exists(integration_branch):
            git_manager.delete_branch(integration_branch, force=True)

        # Create new integration branch from base
        git_manager.create_branch(
            integration_branch,
            base_branch=self.config.base_branch,
            checkout=False,
        )

    async def _final_validation(self, cwd: str) -> EvidenceExecutionResult:
        """Phase 5: Run final validation on integration branch.

        Args:
            cwd: Working directory

        Returns:
            EvidenceExecutionResult from running tests/typecheck/lint
        """
        from .contracts import EvidenceRequirement

        evidence_req = EvidenceRequirement(
            tests=["pytest"] if self.config.require_tests else [],
            typecheck=self.config.require_typecheck,
            lint=self.config.require_lint,
        )

        executor = get_evidence_executor()
        return await executor.execute(
            evidence_required=evidence_req,
            cwd=cwd,
        )

    def _check_final_evidence(self, evidence: EvidenceExecutionResult) -> bool:
        """Check if final evidence passes all gates.

        Args:
            evidence: Evidence execution result

        Returns:
            True if all required checks pass
        """
        # Check tests
        if self.config.require_tests:
            if not all(evidence.test_results.values()):
                return False

        # Check typecheck
        if self.config.require_typecheck:
            if not evidence.typecheck_passed:
                return False

        # Check lint
        if self.config.require_lint:
            if not evidence.lint_passed:
                return False

        return True

    def _push_integration(self, git_manager: "GitManager"):
        """Push integration branch to remote.

        Args:
            git_manager: GitManager instance

        Returns:
            GitResult with push status
        """
        return git_manager._run(
            ["push", "-u", "origin", self.config.integration_branch], check=False
        )

    async def _create_pr(
        self,
        git_manager: "GitManager",
        clarified: ClarifiedTask,
        ctx: ExecutionContext,
    ) -> str:
        """Create PR from integration branch.

        Args:
            git_manager: GitManager instance
            clarified: Clarified task for PR description
            ctx: Execution context for summary

        Returns:
            PR URL if created, empty string otherwise
        """
        # Build PR body
        body_lines = [
            "## Summary",
            clarified.objective,
            "",
            "## Acceptance Criteria",
            *[f"- [ ] {c}" for c in clarified.acceptance_criteria],
            "",
            "## Changes",
        ]

        # Add WorkOrder summaries
        for wo_id, state in ctx.work_order_states.items():
            wo = ctx.work_orders.get(wo_id)
            if wo and state == WorkOrderState.MERGED:
                body_lines.append(f"- {wo.objective}")
            elif wo and state == WorkOrderState.REJECTED:
                body_lines.append(f"- {wo.objective} (rejected)")

        body_lines.extend(
            [
                "",
                "---",
                "Generated with U-LLM-SDK MergeExecutor",
            ]
        )

        body = "\n".join(body_lines)

        # Create PR using gh CLI
        result = git_manager._run(
            [
                "gh",
                "pr",
                "create",
                "--base",
                self.config.base_branch,
                "--head",
                self.config.integration_branch,
                "--title",
                clarified.objective[:100],
                "--body",
                body,
            ],
            check=False,
        )

        if result.success:
            # Extract PR URL from output
            return result.output.strip()

        return ""


# =============================================================================
# Convenience Functions
# =============================================================================


async def merge_execute(
    instruction: str,
    cwd: str,
    context: Optional[str] = None,
    auto_push: bool = False,
    create_pr: bool = False,
    max_parallel: int = 3,
    **kwargs,
) -> MergeExecutorResult:
    """Quick MergeExecutor run.

    Args:
        instruction: Task instruction
        cwd: Working directory
        context: Additional context
        auto_push: Whether to push after validation
        create_pr: Whether to create PR
        max_parallel: Max parallel WorkOrders
        **kwargs: Additional MergeExecutorConfig options

    Returns:
        MergeExecutorResult

    Example:
        >>> result = await merge_execute(
        ...     "Add user authentication",
        ...     cwd="/project",
        ...     create_pr=True,
        ... )
        >>> print(result.to_summary())
    """
    config = MergeExecutorConfig(
        auto_push=auto_push,
        create_pr=create_pr,
        max_parallel=max_parallel,
        **kwargs,
    )
    executor = MergeExecutor(config)
    return await executor.run(instruction, cwd, context)


def merge_execute_sync(
    instruction: str,
    cwd: str,
    **kwargs,
) -> MergeExecutorResult:
    """Synchronous version of merge_execute."""
    return asyncio.run(merge_execute(instruction, cwd, **kwargs))


__all__ = [
    "ClarifiedTask",
    "MergeExecutor",
    "MergeExecutorConfig",
    "MergeExecutorResult",
    "merge_execute",
    "merge_execute_sync",
]
