"""Multi-LLM Migration and Integration Module.

This module provides components for integrating Multi-LLM orchestration with
the existing MergeExecutor pipeline, supporting hybrid execution modes and
graceful migration from single-provider to multi-provider workflows.

Key Components:

1. HybridOrchestrator:
   - Combines Multi-LLM coordination with MergeExecutor execution
   - Gemini orchestrates high-level decisions
   - Claude executes via MergeExecutor for code tasks
   - Codex provides deep analysis when needed

2. StateRecoveryManager:
   - Persists orchestration state for recovery
   - Handles session interruption gracefully
   - Supports checkpoint/restore patterns

3. GracefulDegradation:
   - Falls back to single-provider when others unavailable
   - Maintains functionality with reduced capability

Architecture:

    Human Request
         │
         ▼
    ┌─────────────────┐
    │ HybridOrchestrator │ ← Gemini (Master)
    └─────────────────┘
         │
         ├─→ Brainstorm (if complex)
         │       │
         │       └─→ ConsensusLoop
         │
         ├─→ ClarityGate (task clarity)
         │
         └─→ Task Execution
                 │
                 ├─→ MergeExecutor (code tasks)
                 │       └─→ Claude (editor)
                 │
                 └─→ Direct Codex (analysis tasks)

Usage:
    >>> from u_llm_sdk.multi_llm import HybridOrchestrator
    >>>
    >>> orchestrator = HybridOrchestrator(providers, merge_config)
    >>> result = await orchestrator.run("Implement user authentication", cwd="/project")
    >>>
    >>> print(result.to_summary())
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from u_llm_sdk.types import (
    ClarityLevel,
    ConsensusResult,
    Provider,
    Task,
)

if TYPE_CHECKING:
    from u_llm_sdk.types import LLMResult

    from ..llm.orchestration import MergeExecutor, MergeExecutorConfig, MergeExecutorResult
    from ..llm.providers import LLMProvider
    from .orchestrator import GeminiOrchestrator, OrchestratorResponse
    from .rag_integration import MultiLLMRAGHook

logger = logging.getLogger(__name__)


# =============================================================================
# Execution Mode Enum
# =============================================================================


class ExecutionMode(Enum):
    """Execution mode for hybrid orchestration.

    Attributes:
        MULTI_LLM: Full multi-LLM orchestration (Gemini + Claude + Codex)
        MERGE_EXECUTOR_ONLY: Only use MergeExecutor (single provider)
        HYBRID: Combine multi-LLM decisions with MergeExecutor execution
        FALLBACK: Degraded mode when providers unavailable
    """

    MULTI_LLM = "multi_llm"
    MERGE_EXECUTOR_ONLY = "merge_executor_only"
    HYBRID = "hybrid"
    FALLBACK = "fallback"


# =============================================================================
# Hybrid Orchestration Result
# =============================================================================


@dataclass
class HybridExecutionResult:
    """Result of hybrid orchestration execution.

    Attributes:
        success: Whether execution completed successfully
        execution_mode: Mode that was used
        orchestrator_response: Response from multi-llm orchestrator
        merge_result: Result from MergeExecutor (if used)
        analysis_results: Results from Codex analysis tasks
        strategy_results: Results from Gemini strategy tasks
        brainstorm_result: Brainstorm consensus (if held)
        clarity_assessment: Clarity assessment (if performed)
        execution_time_ms: Total execution time
        fallback_reason: Reason if fallback was used
        error: Error message if failed
    """

    success: bool
    execution_mode: ExecutionMode
    orchestrator_response: Optional["OrchestratorResponse"] = None
    merge_result: Optional["MergeExecutorResult"] = None
    analysis_results: Optional[list["LLMResult"]] = None
    strategy_results: Optional[list["LLMResult"]] = None
    brainstorm_result: Optional[ConsensusResult] = None
    clarity_assessment: Optional[dict] = None
    execution_time_ms: int = 0
    fallback_reason: str = ""
    error: str = ""

    def to_summary(self) -> str:
        """Generate human-readable summary."""
        lines = [f"HybridExecution: {'SUCCESS' if self.success else 'FAILED'}"]
        lines.append(f"  Mode: {self.execution_mode.value}")

        if self.brainstorm_result:
            lines.append(f"  Brainstorm: {self.brainstorm_result.final_decision[:50]}...")

        if self.merge_result:
            lines.append(f"  MergeExecutor: {self.merge_result.to_summary()}")

        if self.analysis_results:
            lines.append(f"  Analysis tasks: {len(self.analysis_results)} completed")

        if self.strategy_results:
            lines.append(f"  Strategy tasks: {len(self.strategy_results)} completed")

        if self.execution_time_ms > 0:
            lines.append(f"  Time: {self.execution_time_ms}ms")

        if self.fallback_reason:
            lines.append(f"  Fallback: {self.fallback_reason}")

        if self.error:
            lines.append(f"  Error: {self.error}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "execution_mode": self.execution_mode.value,
            "orchestrator_response": (
                self.orchestrator_response.to_dict()
                if self.orchestrator_response and hasattr(self.orchestrator_response, "to_dict")
                else None
            ),
            "merge_result": (
                {
                    "success": self.merge_result.success,
                    "integration_branch": self.merge_result.integration_branch,
                    "pushed": self.merge_result.pushed,
                    "pr_url": self.merge_result.pr_url,
                    "error": self.merge_result.error,
                }
                if self.merge_result
                else None
            ),
            "analysis_results": (
                [r.to_dict() for r in self.analysis_results]
                if self.analysis_results
                else None
            ),
            "strategy_results": (
                [r.to_dict() for r in self.strategy_results]
                if self.strategy_results
                else None
            ),
            "brainstorm_result": (
                self.brainstorm_result.to_dict() if self.brainstorm_result else None
            ),
            "clarity_assessment": self.clarity_assessment,
            "execution_time_ms": self.execution_time_ms,
            "fallback_reason": self.fallback_reason,
            "error": self.error,
        }


# =============================================================================
# Hybrid Orchestrator
# =============================================================================


class HybridOrchestrator:
    """Hybrid orchestrator combining Multi-LLM and MergeExecutor.

    This orchestrator provides the best of both worlds:
    - Multi-LLM coordination for complex decisions and analysis
    - MergeExecutor for reliable code execution with git integration

    Workflow:
    1. Gemini (master) receives and understands the request
    2. If complex/ambiguous: trigger brainstorming with all providers
    3. ClarityGate ensures task is well-defined before execution
    4. Route execution:
       - Code tasks → MergeExecutor (Claude as editor)
       - Analysis tasks → Direct Codex execution
       - Strategy tasks → Direct Gemini execution
    5. Aggregate results and learn from outcomes

    Example:
        >>> providers = {
        ...     Provider.GEMINI: gemini_provider,
        ...     Provider.CLAUDE: claude_provider,
        ...     Provider.CODEX: codex_provider,
        ... }
        >>>
        >>> orchestrator = HybridOrchestrator(
        ...     providers=providers,
        ...     merge_config=MergeExecutorConfig(create_pr=True),
        ... )
        >>>
        >>> result = await orchestrator.run(
        ...     request="Add OAuth2 authentication to the API",
        ...     cwd="/project",
        ... )
        >>>
        >>> if result.success:
        ...     print(f"PR created: {result.merge_result.pr_url}")
    """

    def __init__(
        self,
        providers: dict[Provider, "LLMProvider"],
        merge_config: Optional["MergeExecutorConfig"] = None,
        *,
        rag_hook: Optional["MultiLLMRAGHook"] = None,
        brainstorm_threshold: float = 0.6,
        state_dir: Optional[Path] = None,
    ):
        """Initialize HybridOrchestrator.

        Args:
            providers: Available LLM providers
            merge_config: Configuration for MergeExecutor
            rag_hook: RAG hook for pattern learning (optional)
            brainstorm_threshold: Ambiguity threshold to trigger brainstorming
            state_dir: Directory for state persistence (optional)
        """
        self.providers = providers
        self._rag_hook = rag_hook
        self._brainstorm_threshold = brainstorm_threshold
        self._state_dir = state_dir

        # Store config for lazy initialization
        self._merge_config = merge_config

        # Lazy-initialized components
        self._gemini_orchestrator: Optional["GeminiOrchestrator"] = None
        self._merge_executor: Optional["MergeExecutor"] = None
        self._recovery_manager: Optional["StateRecoveryManager"] = None

    def _ensure_gemini_orchestrator(self) -> "GeminiOrchestrator":
        """Lazily initialize GeminiOrchestrator."""
        if self._gemini_orchestrator is None:
            from .orchestrator import GeminiOrchestrator

            self._gemini_orchestrator = GeminiOrchestrator(self.providers)

        return self._gemini_orchestrator

    def _ensure_merge_executor(self) -> "MergeExecutor":
        """Lazily initialize MergeExecutor."""
        if self._merge_executor is None:
            from ..llm.orchestration import MergeExecutor, MergeExecutorConfig

            config = self._merge_config or MergeExecutorConfig()
            self._merge_executor = MergeExecutor(config)

        return self._merge_executor

    def _ensure_recovery_manager(self) -> "StateRecoveryManager":
        """Lazily initialize StateRecoveryManager."""
        if self._recovery_manager is None:
            self._recovery_manager = StateRecoveryManager(
                state_dir=self._state_dir or Path.home() / ".cache" / "u-llm-sdk" / "state"
            )

        return self._recovery_manager

    async def run(
        self,
        request: str,
        cwd: str,
        *,
        context: Optional[str] = None,
        mode: ExecutionMode = ExecutionMode.HYBRID,
        session_id: Optional[str] = None,
    ) -> HybridExecutionResult:
        """Run hybrid orchestration.

        Args:
            request: User request/instruction
            cwd: Working directory for code execution
            context: Additional context (optional)
            mode: Execution mode to use
            session_id: Session ID for state recovery

        Returns:
            HybridExecutionResult with execution details
        """
        start_time = datetime.now()
        result = HybridExecutionResult(
            success=False,
            execution_mode=mode,
        )

        try:
            # Check for recovery state
            if session_id:
                recovery = self._ensure_recovery_manager()
                recovered_state = await recovery.load_state(session_id)
                if recovered_state and recovered_state.get("in_progress"):
                    logger.info(f"Resuming session {session_id} from checkpoint")
                    # Could restore partial results here

            # Determine actual mode based on provider availability
            actual_mode = self._determine_execution_mode(mode)
            result.execution_mode = actual_mode

            if actual_mode == ExecutionMode.FALLBACK:
                result.fallback_reason = "Required providers unavailable"
                return await self._run_fallback(request, cwd, context, result)

            if actual_mode == ExecutionMode.MERGE_EXECUTOR_ONLY:
                return await self._run_merge_only(request, cwd, context, result)

            # HYBRID or MULTI_LLM mode
            # Step 1: Master orchestrator processes request
            orchestrator = self._ensure_gemini_orchestrator()
            response = await orchestrator.process_request(request)
            result.orchestrator_response = response

            # Step 2: Handle brainstorming if needed
            if response.needs_brainstorm:
                brainstorm_result = await orchestrator.facilitate_brainstorm(
                    response.brainstorm_topic
                )
                result.brainstorm_result = brainstorm_result

                # Update RAG if available
                if self._rag_hook:
                    from .rag_integration import TaskType

                    await self._rag_hook.save_brainstorm_outcome(
                        topic=response.brainstorm_topic,
                        consensus_result=brainstorm_result,
                        rounds_taken=len(brainstorm_result.full_discussion_log) // 3 + 1
                        if brainstorm_result.full_discussion_log
                        else 1,
                        escalated_to_user=brainstorm_result.escalated_to_user,
                        success=brainstorm_result.success,
                        session_id=session_id,
                    )

            # Step 3: Handle clarification if needed
            if response.needs_clarification:
                result.clarity_assessment = {
                    "needs_clarification": True,
                    "questions": response.clarification_questions,
                }
                result.error = "Clarification needed from user"
                return result

            # Step 4: Route and execute tasks
            if response.tasks:
                for task in response.tasks:
                    routing = await orchestrator.route_task(task)

                    # Save routing decision to RAG
                    if self._rag_hook:
                        from .rag_integration import TaskType

                        task_type = self._map_to_task_type(routing.target_worker)
                        await self._rag_hook.save_routing_decision(
                            request=task.objective,
                            task_type=task_type,
                            assigned_provider=routing.target_worker,
                            routing_reason=routing.instructions,
                            session_id=session_id,
                        )

                    # Execute based on routing
                    if routing.target_worker == Provider.CLAUDE:
                        # Code task → MergeExecutor
                        merge_result = await self._execute_code_task(
                            task, cwd, context
                        )
                        result.merge_result = merge_result

                        if not merge_result.success:
                            result.error = merge_result.error
                            return result

                    elif routing.target_worker == Provider.CODEX:
                        # Analysis task → Direct Codex execution
                        analysis_result = await self._execute_analysis_task(
                            task, context
                        )
                        result.analysis_results = result.analysis_results or []
                        result.analysis_results.append(analysis_result)

                    elif routing.target_worker == Provider.GEMINI:
                        # Strategy task → Direct Gemini execution
                        strategy_result = await self._execute_strategy_task(
                            task, context
                        )
                        result.strategy_results = result.strategy_results or []
                        result.strategy_results.append(strategy_result)

            result.success = True

        except Exception as e:
            logger.error(f"Hybrid orchestration failed: {e}", exc_info=True)
            result.error = str(e)

            # Try fallback
            if mode != ExecutionMode.FALLBACK:
                logger.info("Attempting fallback execution")
                result.fallback_reason = f"Primary failed: {e}"
                return await self._run_fallback(request, cwd, context, result)

        finally:
            end_time = datetime.now()
            result.execution_time_ms = int(
                (end_time - start_time).total_seconds() * 1000
            )

            # Save state for recovery
            if session_id:
                recovery = self._ensure_recovery_manager()
                await recovery.save_state(
                    session_id=session_id,
                    state={
                        "result": result.to_dict(),
                        "in_progress": False,
                    },
                )

        return result

    def _determine_execution_mode(self, requested: ExecutionMode) -> ExecutionMode:
        """Determine actual execution mode based on provider availability."""
        if requested == ExecutionMode.MERGE_EXECUTOR_ONLY:
            # Just need Claude
            if Provider.CLAUDE in self.providers:
                return ExecutionMode.MERGE_EXECUTOR_ONLY
            return ExecutionMode.FALLBACK

        # HYBRID or MULTI_LLM - need at least Gemini and Claude
        has_gemini = Provider.GEMINI in self.providers
        has_claude = Provider.CLAUDE in self.providers
        has_codex = Provider.CODEX in self.providers

        if has_gemini and has_claude:
            # Can do hybrid/multi-llm
            return requested

        if has_claude:
            # Fall back to merge executor only
            logger.warning(
                "Gemini unavailable, falling back to MergeExecutor only mode"
            )
            return ExecutionMode.MERGE_EXECUTOR_ONLY

        # Neither available
        return ExecutionMode.FALLBACK

    async def _run_merge_only(
        self,
        request: str,
        cwd: str,
        context: Optional[str],
        result: HybridExecutionResult,
    ) -> HybridExecutionResult:
        """Run using MergeExecutor only (no multi-LLM coordination)."""
        executor = self._ensure_merge_executor()
        merge_result = await executor.run(request, cwd, context)
        result.merge_result = merge_result
        result.success = merge_result.success
        if not merge_result.success:
            result.error = merge_result.error
        return result

    async def _run_fallback(
        self,
        request: str,
        cwd: str,
        context: Optional[str],
        result: HybridExecutionResult,
    ) -> HybridExecutionResult:
        """Run in fallback mode with reduced functionality."""
        result.execution_mode = ExecutionMode.FALLBACK

        # Try to use any available provider directly
        for provider_enum, provider in self.providers.items():
            try:
                from u_llm_sdk.types import LLMResult

                llm_result = await provider.run(
                    f"Execute this task:\n\n{request}\n\nContext:\n{context or 'None'}"
                )
                result.success = llm_result.success
                if not llm_result.success:
                    result.error = llm_result.error or "Provider execution failed"
                return result

            except Exception as e:
                logger.warning(f"Fallback with {provider_enum} failed: {e}")
                continue

        result.error = "All providers failed in fallback mode"
        return result

    async def _execute_code_task(
        self,
        task: Task,
        cwd: str,
        context: Optional[str],
    ) -> "MergeExecutorResult":
        """Execute a code task using MergeExecutor."""
        executor = self._ensure_merge_executor()

        # Build instruction from task
        instruction = task.objective
        if task.constraints:
            instruction += "\n\nConstraints:\n" + "\n".join(
                f"- {c}" for c in task.constraints
            )

        # Combine context
        full_context = task.context
        if context:
            full_context = f"{context}\n\n{full_context}"

        return await executor.run(instruction, cwd, full_context)

    async def _execute_analysis_task(
        self,
        task: Task,
        context: Optional[str],
    ) -> "LLMResult":
        """Execute an analysis task using Codex (direct execution).

        Analysis tasks don't modify files - they provide deep code analysis,
        debugging insights, or architectural reviews.
        """
        from u_llm_sdk.types import LLMResult

        codex = self.providers.get(Provider.CODEX)
        if not codex:
            # Return empty result if Codex not available
            return LLMResult(
                success=False,
                text="",
                error="Codex provider not available for analysis task",
            )

        # Build analysis prompt
        prompt = f"Analyze the following:\n\n{task.objective}"
        if task.context:
            prompt = f"{task.context}\n\n{prompt}"
        if context:
            prompt = f"{context}\n\n{prompt}"
        if task.constraints:
            prompt += "\n\nConstraints:\n" + "\n".join(f"- {c}" for c in task.constraints)

        return await codex.run(prompt)

    async def _execute_strategy_task(
        self,
        task: Task,
        context: Optional[str],
    ) -> "LLMResult":
        """Execute a strategy task using Gemini (direct execution).

        Strategy tasks don't modify files - they provide architectural
        guidance, design decisions, or high-level planning.
        """
        from u_llm_sdk.types import LLMResult

        gemini = self.providers.get(Provider.GEMINI)
        if not gemini:
            # Return empty result if Gemini not available
            return LLMResult(
                success=False,
                text="",
                error="Gemini provider not available for strategy task",
            )

        # Build strategy prompt
        prompt = f"Provide strategic guidance for:\n\n{task.objective}"
        if task.context:
            prompt = f"{task.context}\n\n{prompt}"
        if context:
            prompt = f"{context}\n\n{prompt}"
        if task.constraints:
            prompt += "\n\nConstraints:\n" + "\n".join(f"- {c}" for c in task.constraints)

        return await gemini.run(prompt)

    def _map_to_task_type(self, provider: Provider) -> "TaskType":
        """Map provider to task type for RAG logging."""
        from .rag_integration import TaskType

        mapping = {
            Provider.CLAUDE: TaskType.CODE_IMPLEMENTATION,
            Provider.CODEX: TaskType.DEEP_ANALYSIS,
            Provider.GEMINI: TaskType.ARCHITECTURE_DESIGN,
        }
        return mapping.get(provider, TaskType.GENERAL)


# =============================================================================
# State Recovery Manager
# =============================================================================


class StateRecoveryManager:
    """Manages orchestration state for recovery.

    Provides checkpoint/restore functionality to handle:
    - Session interruptions
    - Provider failures mid-execution
    - Long-running operations

    State is persisted as JSON files in the state directory.

    Example:
        >>> recovery = StateRecoveryManager(Path("~/.cache/state"))
        >>>
        >>> # Save state
        >>> await recovery.save_state("session-123", {
        ...     "in_progress": True,
        ...     "current_phase": "brainstorming",
        ...     "partial_results": {...},
        ... })
        >>>
        >>> # Later, recover
        >>> state = await recovery.load_state("session-123")
        >>> if state and state.get("in_progress"):
        ...     # Resume from checkpoint
    """

    def __init__(self, state_dir: Path):
        """Initialize StateRecoveryManager.

        Args:
            state_dir: Directory for state files
        """
        self._state_dir = Path(state_dir)
        self._state_dir.mkdir(parents=True, exist_ok=True)

    def _state_path(self, session_id: str) -> Path:
        """Get path for session state file."""
        # Sanitize session_id for filename
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)
        return self._state_dir / f"{safe_id}.json"

    async def save_state(
        self,
        session_id: str,
        state: dict,
    ) -> bool:
        """Save session state.

        Args:
            session_id: Session identifier
            state: State to save

        Returns:
            True if saved successfully
        """
        try:
            state_path = self._state_path(session_id)
            state["_timestamp"] = datetime.now().isoformat()
            state["_session_id"] = session_id

            # Use thread pool for file I/O
            def _write():
                with open(state_path, "w") as f:
                    json.dump(state, f, indent=2, default=str)

            await asyncio.get_event_loop().run_in_executor(None, _write)

            logger.debug(f"Saved state for session {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save state for {session_id}: {e}")
            return False

    async def load_state(self, session_id: str) -> Optional[dict]:
        """Load session state.

        Args:
            session_id: Session identifier

        Returns:
            Saved state or None if not found
        """
        try:
            state_path = self._state_path(session_id)

            if not state_path.exists():
                return None

            def _read():
                with open(state_path) as f:
                    return json.load(f)

            state = await asyncio.get_event_loop().run_in_executor(None, _read)

            logger.debug(f"Loaded state for session {session_id}")
            return state

        except Exception as e:
            logger.warning(f"Failed to load state for {session_id}: {e}")
            return None

    async def delete_state(self, session_id: str) -> bool:
        """Delete session state.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted (or didn't exist)
        """
        try:
            state_path = self._state_path(session_id)

            if state_path.exists():
                def _delete():
                    state_path.unlink()

                await asyncio.get_event_loop().run_in_executor(None, _delete)

            logger.debug(f"Deleted state for session {session_id}")
            return True

        except Exception as e:
            logger.warning(f"Failed to delete state for {session_id}: {e}")
            return False

    async def list_sessions(self) -> list[str]:
        """List all saved session IDs.

        Returns:
            List of session IDs with saved state
        """
        try:
            def _list():
                return [
                    p.stem for p in self._state_dir.glob("*.json")
                ]

            return await asyncio.get_event_loop().run_in_executor(None, _list)

        except Exception as e:
            logger.warning(f"Failed to list sessions: {e}")
            return []

    async def cleanup_old_sessions(
        self,
        max_age_hours: int = 24,
    ) -> int:
        """Clean up old session states.

        Args:
            max_age_hours: Maximum age in hours before cleanup

        Returns:
            Number of sessions cleaned up
        """
        try:
            from datetime import timedelta

            cutoff = datetime.now() - timedelta(hours=max_age_hours)
            count = 0

            for session_id in await self.list_sessions():
                state = await self.load_state(session_id)
                if not state:
                    continue

                timestamp_str = state.get("_timestamp")
                if not timestamp_str:
                    continue

                try:
                    timestamp = datetime.fromisoformat(timestamp_str)
                    if timestamp < cutoff:
                        await self.delete_state(session_id)
                        count += 1
                except (ValueError, TypeError):
                    continue

            logger.info(f"Cleaned up {count} old session states")
            return count

        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")
            return 0


# =============================================================================
# Migration Helper
# =============================================================================


class MigrationHelper:
    """Helper for migrating from single-provider to multi-provider workflows.

    Provides utilities to:
    - Detect if a request would benefit from multi-LLM coordination
    - Wrap existing MergeExecutor calls with orchestration
    - Gradually introduce multi-LLM features

    Example:
        >>> helper = MigrationHelper()
        >>>
        >>> # Check if request would benefit from multi-LLM
        >>> if await helper.should_use_multi_llm(request):
        ...     result = await hybrid_orchestrator.run(request)
        ... else:
        ...     result = await merge_executor.run(request)
    """

    # Keywords that suggest multi-LLM would help
    COMPLEXITY_KEYWORDS = [
        "architecture",
        "design",
        "refactor",
        "optimize",
        "review",
        "analyze",
        "compare",
        "evaluate",
        "strategy",
        "tradeoff",
        "trade-off",
        "decision",
        "approach",
        "best practice",
        "should we",
        "what if",
        "pros and cons",
    ]

    # Keywords that suggest simple code task
    SIMPLE_TASK_KEYWORDS = [
        "fix",
        "add",
        "implement",
        "create",
        "update",
        "change",
        "remove",
        "delete",
        "rename",
        "move",
    ]

    def should_use_multi_llm(
        self,
        request: str,
        *,
        complexity_threshold: float = 0.5,
    ) -> bool:
        """Check if request would benefit from multi-LLM coordination.

        Args:
            request: User request
            complexity_threshold: Threshold for complexity score (0.0-1.0)

        Returns:
            True if multi-LLM would likely help
        """
        request_lower = request.lower()

        # Count complexity indicators
        complexity_score = sum(
            1 for kw in self.COMPLEXITY_KEYWORDS
            if kw in request_lower
        ) / len(self.COMPLEXITY_KEYWORDS)

        # Count simple task indicators
        simple_score = sum(
            1 for kw in self.SIMPLE_TASK_KEYWORDS
            if kw in request_lower
        ) / len(self.SIMPLE_TASK_KEYWORDS)

        # Adjust score based on request length (longer = potentially more complex)
        length_factor = min(len(request) / 500, 1.0)  # Saturate at 500 chars

        final_score = (complexity_score * 0.6 + length_factor * 0.4) - (simple_score * 0.3)

        return final_score >= complexity_threshold

    def extract_subtasks(self, request: str) -> list[str]:
        """Extract potential subtasks from a complex request.

        Args:
            request: User request

        Returns:
            List of potential subtasks
        """
        subtasks = []

        # Look for numbered lists
        import re

        numbered = re.findall(r'\d+\.\s*([^\n]+)', request)
        subtasks.extend(numbered)

        # Look for bullet points
        bullets = re.findall(r'[-*•]\s*([^\n]+)', request)
        subtasks.extend(bullets)

        # Look for "and" separated items
        if not subtasks and " and " in request:
            parts = request.split(" and ")
            if len(parts) <= 4:  # Don't split too much
                subtasks.extend(parts)

        return subtasks

    def suggest_execution_mode(
        self,
        request: str,
        available_providers: set[Provider],
    ) -> ExecutionMode:
        """Suggest best execution mode based on request and availability.

        Args:
            request: User request
            available_providers: Set of available providers

        Returns:
            Suggested ExecutionMode
        """
        has_gemini = Provider.GEMINI in available_providers
        has_claude = Provider.CLAUDE in available_providers
        has_codex = Provider.CODEX in available_providers

        # Check complexity
        is_complex = self.should_use_multi_llm(request)

        if is_complex and has_gemini and has_claude:
            return ExecutionMode.HYBRID

        if has_claude:
            return ExecutionMode.MERGE_EXECUTOR_ONLY

        return ExecutionMode.FALLBACK


__all__ = [
    # Enums
    "ExecutionMode",
    # Results
    "HybridExecutionResult",
    # Main classes
    "HybridOrchestrator",
    "StateRecoveryManager",
    "MigrationHelper",
]
