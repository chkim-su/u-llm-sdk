"""Enhanced Master Orchestrator with Mode Support.

This module provides EnhancedMasterOrchestrator which extends GeminiOrchestrator
with support for ORIGINAL_STRICT and SEMI_AUTONOMOUS execution modes.

Modes:
- ORIGINAL_STRICT: Full master control, ClarityGate mandatory, Claude as worker
- SEMI_AUTONOMOUS: Design via brainstorm, implementation delegated to Claude Code

Usage:
    >>> from u_llm_sdk.multi_llm import EnhancedMasterOrchestrator
    >>> from u_llm_sdk.types import OrchestrationMode
    >>>
    >>> orchestrator = EnhancedMasterOrchestrator(providers)
    >>> result = await orchestrator.run(
    ...     "Implement user auth",
    ...     cwd="/project",
    ...     mode=OrchestrationMode.SEMI_AUTONOMOUS,
    ... )
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

from u_llm_sdk.types import (
    BoundaryConstraints,
    ClarityLevel,
    ClaudeCodeDelegation,
    ConfigurableOptions,
    ConsensusResult,
    DelegationOutcome,
    DelegationPhase,
    OrchestrationMode,
    Provider,
    Task,
)

from .claude_executor import ClaudeCodeExecutor

if TYPE_CHECKING:
    from ..llm.providers import LLMProvider
    from .brainstorm import BrainstormModule, BrainstormResult
    from .clarity import ClarityGate
    from .consensus import ConsensusLoop
    from .escalation import EscalationProtocol
    from .orchestrator import GeminiOrchestrator, OrchestratorResponse, TaskRouting

logger = logging.getLogger(__name__)


# =============================================================================
# Result Types
# =============================================================================


@dataclass
class EnhancedOrchestratorResult:
    """Result from enhanced orchestrator execution.

    Contains mode-specific results and common metrics.

    Attributes:
        mode: The execution mode used
        success: Whether execution succeeded
        summary: Human-readable summary
        error: Error message if failed

        # ORIGINAL_STRICT specific
        orchestrator_response: Response from process_request
        worker_results: Results from worker executions
        clarity_assessments: ClarityGate assessments
        escalations: Any escalations that occurred

        # SEMI_AUTONOMOUS specific
        brainstorm_result: Result from design phase
        delegation_outcome: Result from Claude delegation
        review_result: Result from Codex review (optional)

        # Common metrics
        total_duration_ms: Total execution time
        total_cost_usd: Total cost across all phases
        stream_events: All stream events (for SEMI mode)
    """

    mode: OrchestrationMode
    success: bool = False
    summary: str = ""
    error: str = ""

    # ORIGINAL_STRICT
    orchestrator_response: Optional[Any] = None
    worker_results: list[Any] = field(default_factory=list)
    clarity_assessments: list[dict] = field(default_factory=list)
    escalations: list[dict] = field(default_factory=list)
    aggregated_result: Optional[Any] = None

    # SEMI_AUTONOMOUS
    brainstorm_result: Optional[Any] = None
    delegation_outcome: Optional[DelegationOutcome] = None
    review_result: Optional[dict] = None

    # User interaction
    needs_user_input: bool = False
    user_questions: list[str] = field(default_factory=list)

    # Metrics
    total_duration_ms: int = 0
    total_cost_usd: float = 0.0
    stream_events: list[dict] = field(default_factory=list)

    def add_stream_event(self, event: dict) -> None:
        """Add a stream event (for observability callback)."""
        self.stream_events.append(event)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        result = {
            "mode": self.mode.value,
            "success": self.success,
            "summary": self.summary,
            "error": self.error,
            "needs_user_input": self.needs_user_input,
            "user_questions": self.user_questions,
            "total_duration_ms": self.total_duration_ms,
            "total_cost_usd": self.total_cost_usd,
        }

        if self.delegation_outcome:
            result["delegation_outcome"] = self.delegation_outcome.to_dict()

        if self.review_result:
            result["review_result"] = self.review_result

        return result


# =============================================================================
# Enhanced Master Orchestrator
# =============================================================================


class EnhancedMasterOrchestrator:
    """Mode-aware master orchestrator.

    Supports ORIGINAL_STRICT and SEMI_AUTONOMOUS execution modes.

    ORIGINAL_STRICT:
    - Full master control over all decisions
    - ClarityGate mandatory before every worker execution
    - Claude operates as single-response worker
    - Full audit trail with explicit steps

    SEMI_AUTONOMOUS:
    - Phase 1: Design via Multi-LLM brainstorm
    - Phase 2: ClarityGate self-assessment (Claude decides if clear enough)
    - Phase 3: Implementation delegated to Claude Code (autonomous)
    - Phase 4: Optional Codex review
    - Observable via stream-json events
    """

    def __init__(
        self,
        providers: dict[Provider, LLMProvider],
        *,
        gemini_orchestrator: Optional[GeminiOrchestrator] = None,
        brainstorm_module: Optional[BrainstormModule] = None,
        clarity_gate: Optional[ClarityGate] = None,
        escalation_protocol: Optional[EscalationProtocol] = None,
        default_mode: OrchestrationMode = OrchestrationMode.ORIGINAL_STRICT,
        default_boundaries: Optional[BoundaryConstraints] = None,
        session_id: Optional[str] = None,
    ):
        """Initialize enhanced orchestrator.

        Args:
            providers: Provider instances by type
            gemini_orchestrator: Optional pre-configured GeminiOrchestrator
            brainstorm_module: Optional pre-configured BrainstormModule
            clarity_gate: Optional pre-configured ClarityGate
            escalation_protocol: Optional pre-configured EscalationProtocol
            default_mode: Default execution mode
            default_boundaries: Default boundary constraints for SEMI mode
            session_id: Session ID for tracking
        """
        self.providers = providers
        self.default_mode = default_mode
        self.default_boundaries = default_boundaries or BoundaryConstraints()
        self.session_id = session_id or f"orch-{uuid.uuid4().hex[:8]}"

        # Store or lazy-initialize components
        self._gemini_orchestrator = gemini_orchestrator
        self._brainstorm_module = brainstorm_module
        self._clarity_gate = clarity_gate
        self._escalation_protocol = escalation_protocol

        # Claude executor for SEMI mode
        self._claude_executor: Optional[ClaudeCodeExecutor] = None

    # =========================================================================
    # Main Entry Point
    # =========================================================================

    async def run(
        self,
        request: str,
        cwd: str = "",
        *,
        mode: Optional[OrchestrationMode] = None,
        boundaries: Optional[BoundaryConstraints] = None,
        options: Optional[ConfigurableOptions] = None,
        context: Optional[str] = None,
        session_id: Optional[str] = None,
        on_event: Optional[Callable[[dict], None]] = None,
    ) -> EnhancedOrchestratorResult:
        """Run orchestration in the specified mode.

        Args:
            request: User request
            cwd: Working directory
            mode: Execution mode (defaults to self.default_mode)
            boundaries: Constraints for SEMI_AUTONOMOUS mode
            options: Hints for SEMI_AUTONOMOUS mode
            context: Additional context
            session_id: Session for recovery
            on_event: Callback for stream events (SEMI mode)

        Returns:
            EnhancedOrchestratorResult with mode-specific results
        """
        effective_mode = mode or self.default_mode

        if effective_mode == OrchestrationMode.ORIGINAL_STRICT:
            return await self._run_original_strict(
                request, cwd, context, session_id
            )

        elif effective_mode == OrchestrationMode.SEMI_AUTONOMOUS:
            return await self._run_semi_autonomous(
                request,
                cwd,
                boundaries or self.default_boundaries,
                options or ConfigurableOptions(),
                context,
                session_id,
                on_event,
            )

        else:
            return EnhancedOrchestratorResult(
                mode=effective_mode,
                success=False,
                error=f"Unknown mode: {effective_mode}",
            )

    # =========================================================================
    # ORIGINAL_STRICT Mode
    # =========================================================================

    async def _run_original_strict(
        self,
        request: str,
        cwd: str,
        context: Optional[str],
        session_id: Optional[str],
    ) -> EnhancedOrchestratorResult:
        """Execute in ORIGINAL_STRICT mode.

        Full master control with mandatory ClarityGate.
        """
        result = EnhancedOrchestratorResult(
            mode=OrchestrationMode.ORIGINAL_STRICT,
        )

        try:
            orchestrator = self._get_gemini_orchestrator()

            # Step 1: Process request
            response = await orchestrator.process_request(
                request, context=context, session_id=session_id
            )
            result.orchestrator_response = response

            # Step 2: Handle brainstorm if needed
            if response.needs_brainstorm:
                brainstorm = self._get_brainstorm_module()
                brainstorm_result = await brainstorm.run_session(
                    response.brainstorm_topic,
                    context=context,
                )
                result.brainstorm_result = brainstorm_result

            # Step 3: Handle clarification
            if response.needs_clarification:
                result.needs_user_input = True
                result.user_questions = response.clarification_questions
                result.summary = "Clarification needed before proceeding"
                return result

            # Step 4: Execute tasks with ClarityGate
            clarity_gate = self._get_clarity_gate()
            worker_results = []

            for task in response.tasks:
                # ClarityGate check (mandatory in ORIGINAL_STRICT)
                assessment = await clarity_gate.assess(task)
                result.clarity_assessments.append({
                    "task_id": task.task_id,
                    "level": assessment.level.value,
                    "score": assessment.score,
                    "recommendation": assessment.recommendation,
                })

                if assessment.level == ClarityLevel.AMBIGUOUS:
                    # Must escalate
                    escalation_result = await self._handle_escalation(task, assessment)
                    result.escalations.append(escalation_result)
                    continue

                # Route and execute
                routing = await orchestrator.route_task(task, session_id=session_id)
                worker_result = await self._execute_worker(routing, cwd, context)
                worker_results.append(worker_result)
                result.worker_results.append(worker_result)

            # Step 5: Aggregate
            if worker_results:
                aggregated = await orchestrator.aggregate_results(
                    worker_results, session_id=session_id
                )
                result.aggregated_result = aggregated
                result.success = aggregated.success
                result.summary = aggregated.text if hasattr(aggregated, "text") else str(aggregated)
            else:
                result.success = len(response.tasks) == 0
                result.summary = "No tasks to execute"

        except Exception as e:
            logger.exception(f"ORIGINAL_STRICT execution failed: {e}")
            result.error = str(e)
            result.summary = f"Execution failed: {str(e)}"

        return result

    # =========================================================================
    # SEMI_AUTONOMOUS Mode
    # =========================================================================

    async def _run_semi_autonomous(
        self,
        request: str,
        cwd: str,
        boundaries: BoundaryConstraints,
        options: ConfigurableOptions,
        context: Optional[str],
        session_id: Optional[str],
        on_event: Optional[Callable[[dict], None]],
    ) -> EnhancedOrchestratorResult:
        """Execute in SEMI_AUTONOMOUS mode.

        Phase 1: Design (Multi-LLM Brainstorm)
        Phase 2: ClarityGate (Claude self-assessment)
            - CLEAR: Proceed to implementation
            - NEEDS_CLARIFICATION: Proceed with warning
            - AMBIGUOUS: Return with user_questions
        Phase 3: Implementation (Claude Code Delegation)
        Phase 4: Review (Codex - Optional)
        """
        result = EnhancedOrchestratorResult(
            mode=OrchestrationMode.SEMI_AUTONOMOUS,
        )

        # Event callback that also stores events
        def event_callback(event: dict) -> None:
            result.add_stream_event(event)
            if on_event:
                on_event(event)

        try:
            # Phase 1: Design
            logger.info(f"SEMI_AUTONOMOUS Phase 1: Design brainstorm for '{request[:50]}...'")

            design_topic = f"Design implementation approach for: {request}"
            if context:
                design_topic = f"{design_topic}\n\nContext: {context}"

            brainstorm = self._get_brainstorm_module()
            brainstorm_result = await brainstorm.run_session(design_topic)
            result.brainstorm_result = brainstorm_result

            if not brainstorm_result.consensus.success:
                # Low agreement - need user input
                result.needs_user_input = True
                result.user_questions = brainstorm_result.consensus.user_questions or [
                    "The design team could not reach consensus. Please provide guidance."
                ]
                result.summary = "Design consensus not reached"
                return result

            # Phase 2: ClarityGate - Claude self-assesses before execution
            logger.info(f"SEMI_AUTONOMOUS Phase 2: ClarityGate assessment")

            # Create task for clarity assessment
            delegation_task = Task(
                task_id=f"del-{session_id or uuid.uuid4().hex[:8]}",
                objective=request,
                context=brainstorm_result.consensus.final_decision,
            )

            clarity_gate = self._get_clarity_gate()
            assessment = await clarity_gate.assess(delegation_task)

            result.clarity_assessments.append({
                "task_id": delegation_task.task_id,
                "level": assessment.level.value,
                "score": assessment.score,
                "recommendation": assessment.recommendation,
                "unclear_aspects": [
                    {"type": a.aspect_type.value, "description": a.description}
                    for a in (assessment.unclear_aspects or [])
                ],
            })

            if assessment.level == ClarityLevel.AMBIGUOUS:
                # Too unclear - need user input
                result.needs_user_input = True
                result.user_questions = assessment.self_questions or [
                    "The task is not clear enough. Please provide more details."
                ]
                result.summary = f"Clarity assessment failed: {assessment.recommendation}"
                logger.warning(f"ClarityGate AMBIGUOUS: {assessment.self_questions}")
                return result

            # Log clarity result
            if assessment.level == ClarityLevel.NEEDS_CLARIFICATION:
                logger.info(
                    f"ClarityGate NEEDS_CLARIFICATION (score={assessment.score:.2f}), "
                    f"but proceeding with available context"
                )
            else:
                logger.info(f"ClarityGate CLEAR (score={assessment.score:.2f}), proceeding")

            # Phase 3: Implementation - Claude executes autonomously
            logger.info(f"SEMI_AUTONOMOUS Phase 3: Claude Code delegation")

            delegation = ClaudeCodeDelegation(
                delegation_id=delegation_task.task_id,
                objective=request,
                design_context=brainstorm_result.consensus.final_decision,
                boundaries=boundaries,
                options=options,
                cwd=cwd,
                branch_name=f"delegation/{session_id or 'auto'}",
                session_id=session_id,
            )

            executor = self._get_claude_executor(boundaries)
            outcome = await executor.execute(delegation, on_event=event_callback)
            result.delegation_outcome = outcome

            if not outcome.success:
                result.error = outcome.error
                result.summary = f"Implementation failed: {outcome.error}"
                return result

            # Phase 4: Review (Optional)
            if Provider.CODEX in self.providers and outcome.files_modified:
                logger.info(f"SEMI_AUTONOMOUS Phase 4: Codex review")

                review = await self._codex_review(
                    outcome,
                    brainstorm_result.consensus.final_decision,
                )
                result.review_result = review

            # Success
            result.success = True
            result.summary = outcome.summary
            result.total_cost_usd = outcome.budget_used_usd
            result.total_duration_ms = outcome.duration_ms

        except Exception as e:
            logger.exception(f"SEMI_AUTONOMOUS execution failed: {e}")
            result.error = str(e)
            result.summary = f"Execution failed: {str(e)}"

        return result

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _get_gemini_orchestrator(self) -> GeminiOrchestrator:
        """Get or create Gemini orchestrator."""
        if self._gemini_orchestrator is None:
            from .orchestrator import GeminiOrchestrator

            if Provider.GEMINI not in self.providers:
                raise ValueError("Gemini provider required for orchestration")

            self._gemini_orchestrator = GeminiOrchestrator(
                self.providers,
                session_id=self.session_id,
            )

        return self._gemini_orchestrator

    def _get_brainstorm_module(self) -> BrainstormModule:
        """Get or create brainstorm module."""
        if self._brainstorm_module is None:
            from .brainstorm import BrainstormModule

            self._brainstorm_module = BrainstormModule(self.providers)

        return self._brainstorm_module

    def _get_clarity_gate(self) -> ClarityGate:
        """Get or create clarity gate."""
        if self._clarity_gate is None:
            from .clarity import ClarityGate

            # Use Gemini for clarity assessment
            gemini = self.providers.get(Provider.GEMINI)
            if gemini is None:
                raise ValueError("Gemini provider required for ClarityGate")

            self._clarity_gate = ClarityGate(gemini)

        return self._clarity_gate

    def _get_claude_executor(self, boundaries: BoundaryConstraints) -> ClaudeCodeExecutor:
        """Get or create Claude executor."""
        if Provider.CLAUDE not in self.providers:
            raise ValueError("Claude provider required for SEMI_AUTONOMOUS mode")

        # Always create new executor with current boundaries
        return ClaudeCodeExecutor(
            provider=self.providers[Provider.CLAUDE],
            default_boundaries=boundaries,
        )

    async def _handle_escalation(self, task: Task, assessment: Any) -> dict:
        """Handle task escalation."""
        if self._escalation_protocol is None:
            from .escalation import EscalationProtocol

            self._escalation_protocol = EscalationProtocol(
                self._get_gemini_orchestrator()
            )

        from u_llm_sdk.types import EscalationRequest

        request = EscalationRequest(
            source_worker=Provider.CLAUDE,
            original_task=task,
            clarity_assessment=assessment,
            specific_questions=assessment.self_questions,
            request_type="clarification",
        )

        response = await self._escalation_protocol.escalate(
            request, session_id=self.session_id
        )

        return {
            "task_id": task.task_id,
            "escalation_type": request.request_type,
            "response": response.to_dict() if hasattr(response, "to_dict") else str(response),
        }

    async def _execute_worker(
        self,
        routing: TaskRouting,
        cwd: str,
        context: Optional[str],
    ) -> Any:
        """Execute a worker task."""
        provider = self.providers.get(routing.target_worker)
        if provider is None:
            raise ValueError(f"Provider {routing.target_worker} not available")

        # Build prompt
        prompt = routing.task.objective
        if routing.instructions:
            prompt = f"{prompt}\n\nInstructions: {routing.instructions}"
        if context:
            prompt = f"{prompt}\n\nContext: {context}"

        # Execute
        result = await provider.run(prompt)
        return result

    async def _codex_review(
        self,
        outcome: DelegationOutcome,
        original_design: str,
    ) -> dict:
        """Have Codex review the implementation."""
        codex = self.providers.get(Provider.CODEX)
        if codex is None:
            return {"skipped": True, "reason": "Codex not available"}

        review_prompt = f"""Review this implementation against the original design.

## Original Design Decision
{original_design}

## Implementation Results
- Files Modified: {', '.join(outcome.files_modified) or 'None'}
- Tests Passed: {outcome.tests_passed}
- Typecheck Passed: {outcome.typecheck_passed}
- Budget Used: ${outcome.budget_used_usd:.4f}
- Duration: {outcome.duration_ms}ms

## Review Questions
1. Does the implementation align with the design decision?
2. Are there any deviations that should be flagged?
3. Are there potential issues or improvements?

Provide a brief review summary."""

        try:
            result = await codex.run(review_prompt)
            return {
                "success": result.success,
                "review_text": result.text,
                "approved": result.success,
            }
        except Exception as e:
            logger.warning(f"Codex review failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "approved": True,  # Fail-open: don't block on review failure
            }
