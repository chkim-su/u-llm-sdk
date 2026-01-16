"""State Machine Orchestrator for Contract-Based Parallel Execution.

This is NOT just "parallel coding" - it's a contract-based parallel execution
+ auditable merge pipeline.

Architecture:
    1. Planner: Decomposes objective into WorkOrders (contracts)
    2. Editors: Execute WorkOrders in branch isolation
    3. Supervisor: Reviews Evidence, approves/rejects PRs
    4. Master: Orchestrates via state machine + event-driven transitions

Key Principles:
    - Parallel edits use file_set disjoint + resource_locks
    - Code changes only reach main via PR + Supervisor approval
    - Every action has auditable Evidence

State Machine:
    INIT → PLANNING → DISPATCHING → WAITING_PR → REVIEWING → MERGING → AGGREGATING → COMPLETED
                                         ↓              ↓
                                    REJECTED → REDISPATCHING
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import traceback
import uuid
from typing import TYPE_CHECKING, Optional

from .checkpoint import preferred_text
from .contracts import Evidence, ExecutionPlan, ReviewReport, SideEffect, WorkOrder
from .file_scope import FileSetManager, validate_work_orders_disjoint
from .git_integration import (
    GitManager,
    MergeResult,
    cleanup_work_order_branch,
    cleanup_work_order_worktree,
    commit_work_order_changes,
    commit_work_order_in_worktree,
    merge_work_order,
    setup_work_order_branch,
    setup_work_order_worktree,
)
from .lock_manager import InMemoryLockManager, compute_layer_locks
from .types import Event, EventType, ExecutionContext, ExecutionState, WorkOrderState
from .verifier import WorkOrderComplianceVerifier

if TYPE_CHECKING:
    from .context_builder import WorkOrderContextBuilder

from .agent_executor import AgentExecutor
from .delta_compliance import DeltaAnalyzer
from .evidence_executor import get_evidence_executor
from .patch_submission import PatchSubmitter

# Module-level logger
logger = logging.getLogger(__name__)


class StateMachineOrchestrator:
    """State machine-based orchestrator.

    Key insight: Events are triggers for state transitions.
    Parallel management is event-driven via local EventBus.

    State Transitions:
        INIT → PLANNING: Start planning
        PLANNING → DISPATCHING: Plan complete
        DISPATCHING → WAITING_PR: Work dispatched
        WAITING_PR → REVIEWING: PR opened
        REVIEWING → MERGING: Review passed
        REVIEWING → REJECTED: Review failed
        REJECTED → REDISPATCHING: Retry issued
        MERGING → WAITING_PR: More PRs pending
        MERGING → AGGREGATING: All merged
        AGGREGATING → COMPLETED: Done

    Example:
        >>> orchestrator = StateMachineOrchestrator()
        >>> ctx = await orchestrator.run(
        ...     objective="Implement user authentication",
        ...     cwd="/project",
        ... )
        >>> print(ctx.state)
    """

    def __init__(
        self,
        agent_executor: Optional[AgentExecutor] = None,
        max_parallel_editors: int = 3,
        max_retries: int = 2,
        context_builder: Optional["WorkOrderContextBuilder"] = None,
        use_patch_submission: bool = False,
    ):
        """Initialize orchestrator.

        Args:
            agent_executor: Agent executor instance
            max_parallel_editors: Max concurrent editors
            max_retries: Max retries for failed WorkOrders
            context_builder: Optional WorkOrderContextBuilder for SCIP-based thin context.
                             When provided, builds context pack for each WorkOrder
                             and prepends to editor prompt.
            use_patch_submission: When True, submit changes as patches instead of
                                  branch merges. Provides cleaner audit trail.
        """
        self.executor = agent_executor or AgentExecutor()
        self.max_parallel = max_parallel_editors
        self.max_retries = max_retries
        self.context_builder = context_builder
        self.use_patch_submission = use_patch_submission

        # Retry tracking
        self._retry_counts: dict[str, int] = {}

    async def run(
        self,
        objective: str,
        cwd: str,
        context: Optional[str] = None,
    ) -> ExecutionContext:
        """Run full orchestration pipeline.

        Args:
            objective: What to accomplish
            cwd: Working directory (scope boundary)
            context: Additional context

        Returns:
            ExecutionContext with all results
        """
        ctx = ExecutionContext(objective=objective, cwd=cwd)

        # Initialize resource managers
        ctx.file_set_manager = FileSetManager(cwd)
        ctx.lock_manager = InMemoryLockManager()

        # Initialize git manager (may fail if not a git repo)
        try:
            ctx.git_manager = GitManager(cwd)
        except ValueError:
            # Not a git repo - proceed without git integration
            ctx.git_manager = None

        # Initialize delta analyzer for compliance checking
        try:
            ctx.delta_analyzer = DeltaAnalyzer(cwd)
        except Exception:
            ctx.delta_analyzer = None

        try:
            # Phase 1: Planning
            ctx.state = ExecutionState.PLANNING
            await self._plan(ctx, context)

            # Phase 2: Layer-by-layer execution
            while ctx.state not in {ExecutionState.COMPLETED, ExecutionState.FAILED}:
                await self._process_current_layer(ctx)

            # Phase 3: Aggregation (if completed successfully)
            if ctx.state == ExecutionState.AGGREGATING:
                await self._aggregate(ctx)
                ctx.state = ExecutionState.COMPLETED

        except Exception as e:
            ctx.state = ExecutionState.FAILED
            ctx.push_event(Event(EventType.ERROR, payload={"error": str(e)}))
            logger.error(f"Orchestration failed: {e}", exc_info=True)

        finally:
            # Cleanup: Release any remaining locks
            if ctx.lock_manager and ctx.held_locks:
                for wo_id in list(ctx.held_locks.keys()):
                    await self._release_locks(ctx, wo_id)

            # Cleanup: Remove any remaining worktrees
            if ctx.git_manager and ctx.wo_worktrees:
                cleaned_ids = []
                for wo_id, worktree_path in list(ctx.wo_worktrees.items()):
                    try:
                        branch_name = ctx.wo_branches.get(wo_id, "")
                        await cleanup_work_order_worktree(
                            ctx.git_manager,
                            worktree_path,
                            branch_name,
                            delete_branch=True,
                            force=True,
                        )
                        cleaned_ids.append(wo_id)
                    except Exception as e:
                        logger.warning(f"Failed to cleanup worktree for {wo_id}: {e}")
                for wo_id in cleaned_ids:
                    del ctx.wo_worktrees[wo_id]

            # Prune any stale worktree references
            if ctx.git_manager:
                ctx.git_manager.prune_worktrees()

        ctx.end_time = time.time()
        return ctx

    async def _plan(self, ctx: ExecutionContext, context: Optional[str]):
        """Execute planning phase.

        Planner creates ExecutionPlan with WorkOrders.
        """
        plan_prompt = self._build_plan_prompt(ctx.objective, ctx.cwd)

        result = await self.executor.execute(
            role="planner",
            prompt=plan_prompt,
            context=context,
            cwd=ctx.cwd,
            timeout=120.0,
        )

        # Parse plan from response
        plan = self._parse_plan(ctx.objective, result.text or "")

        # Validate file_set disjoint (using FileSetManager for accurate detection)
        self._validate_file_set_disjoint(plan.work_orders, ctx.file_set_manager)

        # Compute execution layers (using lock_manager's compute_layer_locks)
        layers = compute_layer_locks(plan.work_orders)
        plan.layers = layers

        # Store in context
        ctx.plan = plan
        ctx.work_orders = {wo.id: wo for wo in plan.work_orders}
        ctx.work_order_states = {
            wo.id: WorkOrderState.PENDING for wo in plan.work_orders
        }
        ctx.layers = layers

        ctx.push_event(Event(EventType.PLAN_COMPLETE))
        ctx.state = ExecutionState.DISPATCHING

    async def _process_current_layer(self, ctx: ExecutionContext):
        """Process current layer of WorkOrders."""
        if ctx.current_layer >= len(ctx.layers):
            ctx.push_event(Event(EventType.ALL_MERGED))
            ctx.state = ExecutionState.AGGREGATING
            return

        layer_ids = ctx.layers[ctx.current_layer]
        layer_orders = [ctx.work_orders[wid] for wid in layer_ids]

        # Skip if all in this layer are already done
        if all(
            ctx.work_order_states[wid]
            in {WorkOrderState.MERGED, WorkOrderState.SKIPPED}
            for wid in layer_ids
        ):
            ctx.current_layer += 1
            ctx.push_event(Event(EventType.LAYER_COMPLETE))
            return

        # Dispatch phase
        ctx.state = ExecutionState.DISPATCHING
        await self._dispatch_layer(ctx, layer_orders)

        # Review phase
        ctx.state = ExecutionState.REVIEWING
        await self._review_layer(ctx, layer_ids)

        # Check if layer complete
        all_done = all(
            ctx.work_order_states[wid]
            in {WorkOrderState.MERGED, WorkOrderState.SKIPPED}
            for wid in layer_ids
        )

        if all_done:
            ctx.current_layer += 1
            ctx.push_event(Event(EventType.LAYER_COMPLETE))

    async def _dispatch_layer(
        self, ctx: ExecutionContext, work_orders: list[WorkOrder]
    ):
        """Dispatch WorkOrders to Editors in parallel."""
        semaphore = asyncio.Semaphore(self.max_parallel)

        async def execute_work_order(wo: WorkOrder):
            async with semaphore:
                # Check if already done
                if ctx.work_order_states[wo.id] in {
                    WorkOrderState.MERGED,
                    WorkOrderState.SKIPPED,
                }:
                    return wo.id, ctx.results.get(wo.id)

                # Check dependencies
                for dep_id in wo.dependencies:
                    dep_state = ctx.work_order_states.get(dep_id)
                    if dep_state == WorkOrderState.REJECTED:
                        ctx.work_order_states[wo.id] = WorkOrderState.SKIPPED
                        return wo.id, None

                # Acquire resource locks (if any)
                if wo.resource_locks and ctx.lock_manager:
                    lock_result = await ctx.lock_manager.acquire(
                        wo.resource_locks,
                        holder_id=wo.id,
                        timeout=30.0,
                    )
                    if not lock_result.success:
                        error_msg = (
                            f"Lock acquisition timeout for: {wo.resource_locks}"
                        )
                        logger.warning(f"WorkOrder {wo.id}: {error_msg}")
                        ctx.context_build_errors[wo.id] = error_msg
                        ctx.push_event(
                            Event(
                                EventType.ERROR,
                                wo.id,
                                payload={
                                    "error": error_msg,
                                    "locks_requested": wo.resource_locks,
                                },
                            )
                        )
                        ctx.work_order_states[wo.id] = WorkOrderState.SKIPPED
                        return wo.id, None
                    ctx.held_locks[wo.id] = list(lock_result.acquired_locks)

                ctx.work_order_states[wo.id] = WorkOrderState.IN_PROGRESS

                try:
                    # Set up isolated worktree for this WorkOrder
                    editor_cwd = ctx.cwd
                    worktree_path = None

                    if ctx.git_manager:
                        worktree_result = await setup_work_order_worktree(
                            ctx.git_manager,
                            wo.id,
                            wo.objective,
                        )
                        if worktree_result.success:
                            worktree_path = worktree_result.worktree_path
                            editor_cwd = worktree_path
                            ctx.wo_branches[wo.id] = worktree_result.branch_name
                            ctx.wo_worktrees[wo.id] = worktree_path
                            ctx.wo_base_commits[wo.id] = worktree_result.base_commit
                        else:
                            # Fallback to branch-only
                            success, branch_or_error = await setup_work_order_branch(
                                ctx.git_manager,
                                wo.id,
                                wo.objective,
                            )
                            if success:
                                ctx.wo_branches[wo.id] = branch_or_error

                    # Capture before snapshot for delta compliance
                    if ctx.delta_analyzer and wo.expected_delta:
                        if worktree_path:
                            worktree_analyzer = DeltaAnalyzer(worktree_path)
                            ctx.before_snapshots[wo.id] = (
                                worktree_analyzer.analyze_current()
                            )
                        else:
                            ctx.before_snapshots[wo.id] = (
                                ctx.delta_analyzer.analyze_current()
                            )

                    # Build thin context pack for WorkOrder
                    context_text = ""
                    if self.context_builder:
                        try:
                            commit_sha = ctx.wo_base_commits.get(wo.id)
                            if not commit_sha and ctx.git_manager:
                                commit_sha = ctx.git_manager.get_commit_hash("HEAD")

                            pack = await self.context_builder.build_for_work_order(
                                wo,
                                commit_sha=commit_sha,
                            )

                            pack_hash = self.context_builder.compute_pack_hash(pack)
                            context_text = (
                                f"<!-- ContextPack: {pack_hash} -->\n"
                                f"{pack.to_prompt_text()}\n\n"
                            )
                        except Exception as ctx_err:
                            ctx.context_build_errors[wo.id] = str(ctx_err)
                            context_text = f"<!-- Context build failed: {ctx_err} -->\n"

                    # Execute editor in isolated worktree
                    editor_prompt = f"{context_text}{wo.to_prompt()}"
                    result = await self.executor.execute(
                        role="editor",
                        prompt=editor_prompt,
                        cwd=editor_cwd,
                        timeout=300.0,
                    )

                    # Commit changes in worktree
                    if ctx.git_manager and wo.id in ctx.wo_worktrees:
                        success, commit_or_error = await commit_work_order_in_worktree(
                            ctx.git_manager,
                            ctx.wo_worktrees[wo.id],
                            wo.id,
                            wo.objective[:80],
                        )
                        if success:
                            result.raw = result.raw or {}
                            result.raw["commit"] = commit_or_error
                    elif ctx.git_manager and wo.id in ctx.wo_branches:
                        success, commit_or_error = await commit_work_order_changes(
                            ctx.git_manager,
                            wo.id,
                            wo.objective[:80],
                        )
                        if success:
                            result.raw = result.raw or {}
                            result.raw["commit"] = commit_or_error

                    ctx.work_order_states[wo.id] = WorkOrderState.PR_SUBMITTED
                    return wo.id, result

                except Exception as e:
                    # Release locks on failure
                    if wo.id in ctx.held_locks and ctx.lock_manager:
                        await ctx.lock_manager.release(ctx.held_locks[wo.id], wo.id)
                        del ctx.held_locks[wo.id]
                    raise

        # Execute in parallel
        results = await asyncio.gather(
            *[execute_work_order(wo) for wo in work_orders], return_exceptions=True
        )

        # Store results and collect evidence
        for i, item in enumerate(results):
            if isinstance(item, Exception):
                wo = work_orders[i]
                exc_tb = "".join(
                    traceback.format_exception(type(item), item, item.__traceback__)
                )
                logger.error(f"WorkOrder {wo.id} failed with exception:\n{exc_tb}")

                ctx.context_build_errors[wo.id] = f"Execution failed: {item}"
                ctx.push_event(
                    Event(
                        EventType.ERROR,
                        wo.id,
                        payload={"error": str(item), "traceback": exc_tb[:2000]},
                    )
                )
                ctx.work_order_states[wo.id] = WorkOrderState.SKIPPED
                continue

            wo_id, result = item
            if result is None:
                continue

            ctx.results[wo_id] = result

            # Collect evidence
            evidence = await self._collect_evidence(ctx.work_orders[wo_id], result, ctx)
            ctx.evidences[wo_id] = evidence

        ctx.push_event(Event(EventType.DISPATCH_COMPLETE))

    async def _review_layer(self, ctx: ExecutionContext, work_order_ids: list[str]):
        """Review all WorkOrders in layer."""
        for wo_id in work_order_ids:
            # Skip if already processed
            if ctx.work_order_states[wo_id] in {
                WorkOrderState.MERGED,
                WorkOrderState.SKIPPED,
            }:
                await self._release_locks(ctx, wo_id)
                continue

            if wo_id not in ctx.evidences:
                ctx.work_order_states[wo_id] = WorkOrderState.SKIPPED
                await self._release_locks(ctx, wo_id)
                continue

            # Conduct review
            evidence = ctx.evidences[wo_id]
            review = await self._review(ctx, wo_id, evidence)
            ctx.reviews[wo_id] = review

            if review.verdict == "approved":
                ctx.work_order_states[wo_id] = WorkOrderState.APPROVED

                # Merge
                merge_result = await self._merge(ctx, wo_id)

                if merge_result.success:
                    ctx.work_order_states[wo_id] = WorkOrderState.MERGED
                    wo = ctx.work_orders[wo_id]
                    ctx.merged_branches.append(
                        ctx.wo_branches.get(wo_id) or wo.branch or ""
                    )

                    await self._release_locks(ctx, wo_id)
                    ctx.push_event(
                        Event(
                            EventType.MERGE_COMPLETE,
                            wo_id,
                            payload={"commit": merge_result.merged_commit},
                        )
                    )
                else:
                    ctx.work_order_states[wo_id] = WorkOrderState.REJECTED
                    review.verdict = "rejected"
                    review.rejection_reasons.append(
                        f"Merge failed: {merge_result.error or 'conflicts'}"
                    )
                    if merge_result.conflicts:
                        review.rejection_reasons.append(
                            f"Conflicting files: {', '.join(merge_result.conflicts[:5])}"
                        )

                    await self._release_locks(ctx, wo_id)
                    ctx.push_event(
                        Event(
                            EventType.REVIEW_FAIL,
                            wo_id,
                            payload={"conflicts": merge_result.conflicts},
                        )
                    )
            else:
                ctx.work_order_states[wo_id] = WorkOrderState.REJECTED
                await self._release_locks(ctx, wo_id)

                # Check if retry is possible
                retry_count = self._retry_counts.get(wo_id, 0)
                if retry_count < self.max_retries:
                    self._retry_counts[wo_id] = retry_count + 1

                ctx.push_event(Event(EventType.REVIEW_FAIL, wo_id))

    async def _release_locks(self, ctx: ExecutionContext, wo_id: str):
        """Release locks held by a WorkOrder."""
        if wo_id in ctx.held_locks and ctx.lock_manager:
            await ctx.lock_manager.release(ctx.held_locks[wo_id], wo_id)
            del ctx.held_locks[wo_id]

    async def _review(
        self,
        ctx: ExecutionContext,
        wo_id: str,
        evidence: Evidence,
    ) -> ReviewReport:
        """Supervisor reviews WorkOrder completion."""
        wo = ctx.work_orders[wo_id]

        # Check delta violations first
        if evidence.delta_violations:
            return ReviewReport(
                work_order_id=wo_id,
                verdict="rejected",
                rejection_reasons=[
                    f"[HARD GATE] Delta contract violation: {v}"
                    for v in evidence.delta_violations
                ],
                checks={
                    "delta_compliance": False,
                    "file_set_compliance": len(evidence.file_set_violations) == 0,
                },
            )

        # Check file_set violations
        if evidence.file_set_violations:
            return ReviewReport(
                work_order_id=wo_id,
                verdict="rejected",
                rejection_reasons=[
                    f"[HARD GATE] File scope violation: modified '{f}' outside allowed file_set"
                    for f in evidence.file_set_violations
                ],
                checks={
                    "delta_compliance": True,
                    "file_set_compliance": False,
                },
            )

        # Check CI failures
        ci_failures = []
        failed_tests = [
            cmd for cmd, passed in evidence.test_results.items() if not passed
        ]
        if failed_tests:
            ci_failures.extend(
                [f"[HARD GATE] Test failed: {cmd}" for cmd in failed_tests]
            )

        if not evidence.typecheck_passed:
            ci_failures.append("[HARD GATE] Typecheck failed")

        if not evidence.lint_passed:
            ci_failures.append("[HARD GATE] Lint failed")

        if ci_failures:
            return ReviewReport(
                work_order_id=wo_id,
                verdict="rejected",
                rejection_reasons=ci_failures,
                checks={
                    "delta_compliance": True,
                    "file_set_compliance": True,
                    "tests_passed": len(failed_tests) == 0,
                    "typecheck_passed": evidence.typecheck_passed,
                    "lint_passed": evidence.lint_passed,
                },
            )

        # Run programmatic compliance check
        compliance_verifier = WorkOrderComplianceVerifier(wo)
        compliance_result = await compliance_verifier.verify(
            ctx.results[wo_id], context=wo
        )

        # LLM review for semantic check
        review_prompt = self._build_review_prompt(wo, evidence)

        result = await self.executor.execute(
            role="supervisor",
            prompt=review_prompt,
            cwd=ctx.cwd,
            timeout=120.0,
        )

        review = self._parse_review(wo_id, result.text or "")

        # Combine results
        review.checks["programmatic_compliance"] = compliance_result.passed
        review.checks["delta_compliance"] = True
        review.checks["file_set_compliance"] = True
        review.checks["tests_passed"] = (
            all(evidence.test_results.values()) if evidence.test_results else True
        )
        review.checks["typecheck_passed"] = evidence.typecheck_passed
        review.checks["lint_passed"] = evidence.lint_passed

        if not compliance_result.passed:
            review.rejection_reasons.extend(
                compliance_result.details.get("violations", [])
            )
            review.verdict = "rejected"

        return review

    async def _merge(self, ctx: ExecutionContext, wo_id: str) -> MergeResult:
        """Merge approved WorkOrder branch."""
        wo = ctx.work_orders[wo_id]
        branch_name = ctx.wo_branches.get(wo_id) or wo.branch

        # If no git manager or no branch, skip actual merge
        if not ctx.git_manager or not branch_name:
            return MergeResult(success=True, merged_commit="(no-git)")

        # Get evidence summary for merge message
        evidence = ctx.evidences.get(wo_id)
        evidence_summary = wo.objective
        if evidence:
            evidence_summary = f"{wo.objective} - {len(evidence.files_modified)} files"

        # Use patch-based submission if enabled and worktree exists
        if self.use_patch_submission and wo_id in ctx.wo_worktrees:
            worktree_path = ctx.wo_worktrees[wo_id]
            submitter = PatchSubmitter(ctx.cwd)

            patch_result = await submitter.submit(
                worktree_path=worktree_path,
                work_order=wo,
                integration_branch=ctx.git_manager.main_branch,
            )

            if patch_result.success:
                result = MergeResult(
                    success=True,
                    merged_commit=patch_result.commit_sha,
                )
            else:
                result = MergeResult(
                    success=False,
                    conflicts=patch_result.conflict_files,
                    error=patch_result.error,
                )
        else:
            # Traditional branch merge
            result = await merge_work_order(
                ctx.git_manager,
                wo_id,
                branch_name,
                evidence_summary,
            )

        # Store merge result in evidence
        if evidence:
            evidence.merged_commit = result.merged_commit
            if result.conflicts:
                evidence.merge_conflicts = result.conflicts

        # Cleanup after successful merge
        if result.success and ctx.git_manager:
            if wo_id in ctx.wo_worktrees:
                await cleanup_work_order_worktree(
                    ctx.git_manager,
                    ctx.wo_worktrees[wo_id],
                    branch_name,
                    delete_branch=True,
                    force=True,
                )
                del ctx.wo_worktrees[wo_id]
                if wo_id in ctx.wo_base_commits:
                    del ctx.wo_base_commits[wo_id]
            else:
                await cleanup_work_order_branch(ctx.git_manager, branch_name)

            if wo_id in ctx.wo_branches:
                del ctx.wo_branches[wo_id]

        return result

    async def _aggregate(self, ctx: ExecutionContext):
        """Aggregate all results into final output."""
        completed_summaries = []
        for wo_id, state in ctx.work_order_states.items():
            if state == WorkOrderState.MERGED:
                wo = ctx.work_orders[wo_id]
                result = ctx.results.get(wo_id)
                summary = preferred_text(result) if result else "Completed"
                completed_summaries.append(
                    f"- {wo_id}: {wo.objective}\n  {summary[:200]}"
                )

        aggregate_prompt = f"""## Original Objective
{ctx.objective}

## Completed WorkOrders
{chr(10).join(completed_summaries)}

## Merged Branches
{ctx.merged_branches}

## Task
위 결과들을 종합하여 최종 보고서를 작성하세요.
- 무엇이 완료되었는지
- 변경된 파일들의 요약
- 주의해야 할 사항
"""

        result = await self.executor.execute(
            role="aggregator",
            prompt=aggregate_prompt,
            cwd=ctx.cwd,
            timeout=120.0,
        )

        ctx.final_output = result.text or result.summary
        ctx.push_event(Event(EventType.AGGREGATE_COMPLETE))

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _build_plan_prompt(self, objective: str, cwd: str) -> str:
        """Build planning prompt."""
        return f"""## Objective
{objective}

## Working Directory
{cwd}

## Output Format (JSON)
{{
  "work_orders": [
    {{
      "id": "WO-001",
      "objective": "specific task description",
      "file_set": ["glob/pattern/**"],
      "create_files": ["new/file.py"],
      "constraints": ["constraint1", "constraint2"],
      "resource_locks": ["logical_resource_name"],
      "expected_delta": {{
        "forbid_new_deps": [],
        "allow_symbol_changes": ["symbol1"],
        "forbid_new_public_exports": true,
        "max_files_modified": 10
      }},
      "priority": 1,
      "dependencies": []
    }}
  ]
}}

## Rules
1. file_set MUST be disjoint across all WorkOrders (no overlap)
2. Same resource_lock → cannot be in same layer (no parallel execution)
3. Each WorkOrder must be independently executable
4. dependencies list other WorkOrder IDs that must complete first
5. Lower priority number = higher priority
6. Be specific about allowed files and constraints

## Resource Lock Examples
- "public_api:payments" - Payment API surface
- "shared_types:auth" - Auth type definitions
- "schema:database" - Database schema
- "routing:main" - Main routing config

Output ONLY the JSON, no explanation."""

    def _build_review_prompt(self, wo: WorkOrder, evidence: Evidence) -> str:
        """Build review prompt for Supervisor."""
        return f"""## WorkOrder Contract
{wo.to_prompt()}

## Evidence
{evidence.to_checkpoint()}

## Modified Files
{evidence.files_modified}

## Created Files
{evidence.files_created}

## Review Checklist
Answer YES/NO for each:
1. file_set_compliance: Are ALL changes within allowed file_set?
2. constraint_compliance: Are ALL constraints followed?
3. no_scope_creep: No unauthorized additions or expansions?
4. quality: Is the implementation correct and complete?

## Output Format (JSON)
{{
  "verdict": "approved" or "rejected",
  "checks": {{
    "file_set_compliance": true/false,
    "constraint_compliance": true/false,
    "no_scope_creep": true/false,
    "quality": true/false
  }},
  "rejection_reasons": ["reason1", ...],
  "recommendations": ["optional improvement suggestions"]
}}

Output ONLY the JSON."""

    def _parse_plan(self, objective: str, text: str) -> ExecutionPlan:
        """Parse ExecutionPlan from Planner output."""
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start < 0 or end <= start:
                raise ValueError("No JSON found")
            data = json.loads(text[start:end])

            work_orders = [
                WorkOrder.from_dict(wo_data)
                for wo_data in data.get("work_orders", [])
            ]

            if not work_orders:
                work_orders = [
                    WorkOrder(
                        id="WO-001",
                        objective=objective,
                        file_set=["**/*"],
                    )
                ]

            return ExecutionPlan(
                id=str(uuid.uuid4()),
                objective=objective,
                work_orders=work_orders,
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            return ExecutionPlan(
                id=str(uuid.uuid4()),
                objective=objective,
                work_orders=[
                    WorkOrder(
                        id="WO-001",
                        objective=objective,
                        file_set=["**/*"],
                    )
                ],
                metadata={"parse_error": str(e)},
            )

    def _parse_review(self, wo_id: str, text: str) -> ReviewReport:
        """Parse ReviewReport from Supervisor output."""
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                return ReviewReport.from_dict(
                    {
                        "work_order_id": wo_id,
                        **data,
                    }
                )
        except (json.JSONDecodeError, KeyError):
            pass

        return ReviewReport(
            work_order_id=wo_id,
            verdict="rejected",
            rejection_reasons=["Failed to parse review response"],
        )

    async def _collect_evidence(
        self,
        wo: WorkOrder,
        result,
        ctx: ExecutionContext,
    ) -> Evidence:
        """Collect Evidence from execution result."""
        # Check file_set violations
        file_set_violations = []
        for fc in getattr(result, "files_modified", []):
            path = fc.path if hasattr(fc, "path") else str(fc)
            if not self._file_matches_patterns(path, wo.file_set + wo.create_files):
                file_set_violations.append(path)

        # Check delta compliance
        delta_violations = []
        if ctx.delta_analyzer and wo.expected_delta and wo.id in ctx.before_snapshots:
            before_snapshot = ctx.before_snapshots[wo.id]

            if wo.id in ctx.wo_worktrees:
                worktree_analyzer = DeltaAnalyzer(ctx.wo_worktrees[wo.id])
                after_snapshot = worktree_analyzer.analyze_current()
            else:
                after_snapshot = ctx.delta_analyzer.analyze_current()

            compliance_result = ctx.delta_analyzer.check_compliance(
                before_snapshot,
                after_snapshot,
                wo.expected_delta,
            )

            if not compliance_result.compliant:
                delta_violations = [v.description for v in compliance_result.violations]

        # Execute evidence requirements
        test_results = {}
        typecheck_passed = True
        lint_passed = True

        if wo.evidence_required:
            executor = get_evidence_executor()
            exec_cwd = ctx.wo_worktrees.get(wo.id) or ctx.cwd

            evidence_result = await executor.execute(
                evidence_required=wo.evidence_required,
                cwd=exec_cwd,
            )

            test_results = evidence_result.test_results
            typecheck_passed = evidence_result.typecheck_passed
            lint_passed = evidence_result.lint_passed

        # Create evidence
        files_modified = [
            fc.path if hasattr(fc, "path") else str(fc)
            for fc in getattr(result, "files_modified", [])
        ]
        files_created = [
            fc.path if hasattr(fc, "path") else str(fc)
            for fc in getattr(result, "files_created", [])
        ]

        # Get branch name for evidence
        branch_name = ctx.wo_branches.get(wo.id, "")

        return Evidence(
            work_order_id=wo.id,
            branch=branch_name,
            files_modified=files_modified,
            files_created=files_created,
            file_set_violations=file_set_violations,
            delta_violations=delta_violations,
            test_results=test_results,
            typecheck_passed=typecheck_passed,
            lint_passed=lint_passed,
        )

    def _file_matches_patterns(self, file_path: str, patterns: list[str]) -> bool:
        """Check if file matches any of the patterns."""
        import fnmatch

        for pattern in patterns:
            if fnmatch.fnmatch(file_path, pattern):
                return True
        return False

    def _validate_file_set_disjoint(
        self,
        work_orders: list[WorkOrder],
        file_set_manager: Optional[FileSetManager],
    ):
        """Validate that all WorkOrder file_sets are disjoint."""
        if not file_set_manager:
            return

        is_valid, violations = validate_work_orders_disjoint(
            work_orders,
            file_set_manager.repo_path,
        )

        if not is_valid:
            conflict_info = []
            for v in violations:
                conflict_info.append(
                    f"{v.set_a_pattern} ∩ {v.set_b_pattern}: {v.overlapping_files}"
                )
            logger.warning(
                f"WorkOrder file_sets have overlap: {conflict_info}. "
                "Proceeding with potential conflicts."
            )


__all__ = [
    "StateMachineOrchestrator",
]
