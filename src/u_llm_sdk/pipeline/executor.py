"""Pipeline executor for running workflow templates.

Executes workflow phases using U-llm-sdk components.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

from u_llm_sdk.types import (
    BoundaryConstraints,
    ClaudeCodeDelegation,
    ClarityLevel,
    ConfigurableOptions,
    Provider,
    Task,
)

from .types import (
    PhaseConfig,
    PhaseOutput,
    PhaseResult,
    PhaseType,
    PipelineResult,
    WorkflowTemplate,
)
from .chronicle_adapter import PipelineChronicleAdapter
from .report import PipelineReportWriter

# Scribe imports for editable knowledge state
from ..scribe import (
    ScribeStore,
    ScribeType,
    Clearance,
    PHASE_SECTION_MAP,
    PHASE_UPDATE_MAP,
    should_escalate,
)

if TYPE_CHECKING:
    from ..rag_client import RAGClient

logger = logging.getLogger(__name__)

# Template directory
TEMPLATES_DIR = Path(__file__).parent / "templates"


class PipelineExecutor:
    """Executes workflow pipelines using SDK components.

    Example:
        >>> from u_llm_sdk.pipeline import PipelineExecutor, WorkflowTemplate
        >>>
        >>> # Load template
        >>> template = WorkflowTemplate.from_yaml("templates/semi_autonomous.yaml")
        >>>
        >>> # Create executor with providers
        >>> executor = PipelineExecutor(providers)
        >>>
        >>> # Run pipeline
        >>> result = await executor.run(
        ...     template,
        ...     request="Implement user authentication",
        ...     cwd="/project",
        ... )
    """

    def __init__(
        self,
        providers: dict[Provider, Any],
        *,
        rag_client: Optional["RAGClient"] = None,
        on_phase_start: Optional[Callable[[str], None]] = None,
        on_phase_end: Optional[Callable[[PhaseOutput], None]] = None,
        on_user_input: Optional[Callable[[list[str]], Optional[str]]] = None,
        on_progress: Optional[Callable[[str], None]] = None,
        on_stream: Optional[Callable[[str, str], None]] = None,
        chronicle_enabled: bool = True,
        chronicle_dir: Optional[Union[str, Path]] = None,
        scribe_enabled: bool = True,
        scribe_path: Optional[Union[str, Path]] = None,
        report_enabled: bool = True,
    ):
        """Initialize executor.

        Args:
            providers: LLM provider instances by type
            rag_client: RAGClient for MV-rag server communication (SCIP context, etc.)
            on_phase_start: Callback when phase starts
            on_phase_end: Callback when phase ends
            on_user_input: Callback to get user input (returns answer or None to abort)
            on_progress: Callback for progress updates during phases (e.g., brainstorm steps)
            on_stream: Callback for streaming LLM output (speaker_name, text_chunk)
            chronicle_enabled: Enable Chronicle record generation
            chronicle_dir: Directory to save Chronicle records (default: .chronicle/pipelines)
            scribe_enabled: Enable Scribe editable knowledge state (default: True)
            scribe_path: Path to Scribe SQLite database (default: .chronicle/scribe.db)
            report_enabled: Enable final report generation (default: True)
        """
        self.providers = providers
        self.rag_client = rag_client
        self.on_phase_start = on_phase_start
        self.on_phase_end = on_phase_end
        self.on_user_input = on_user_input
        self.on_progress = on_progress
        self.on_stream = on_stream
        self.chronicle_enabled = chronicle_enabled
        self.chronicle_dir = Path(chronicle_dir) if chronicle_dir else Path(".chronicle/pipelines")
        self.scribe_enabled = scribe_enabled
        self.scribe_path = Path(scribe_path) if scribe_path else Path(".chronicle/scribe.db")
        self.report_enabled = report_enabled

        # Lazy-loaded components
        self._brainstorm_module = None
        self._clarity_gate = None
        self._claude_executor = None
        self._evidence_gate = None

        # Scribe store (editable knowledge state)
        self._scribe_store: Optional[ScribeStore] = None

        # Chronicle adapter (created per-run)
        self._chronicle: Optional[PipelineChronicleAdapter] = None

        # Report writer (created per-run)
        self._report_writer: Optional[PipelineReportWriter] = None

    # =========================================================================
    # Public API
    # =========================================================================

    async def run(
        self,
        template: Union[WorkflowTemplate, str, Path],
        request: str,
        cwd: str = "",
        *,
        variables: Optional[dict[str, Any]] = None,
        boundaries: Optional[BoundaryConstraints] = None,
        context: Optional[str] = None,
        resume_from: Optional[str] = None,
        save_chronicle: bool = True,
    ) -> tuple[PipelineResult, Optional[PipelineChronicleAdapter]]:
        """Execute a workflow pipeline.

        Args:
            template: Workflow template or path to YAML
            request: User request/objective
            cwd: Working directory
            variables: Override template variables
            boundaries: Override boundary constraints
            context: Additional context
            resume_from: Phase name to resume from
            save_chronicle: Whether to save Chronicle records to file

        Returns:
            Tuple of (PipelineResult, PipelineChronicleAdapter or None if disabled)
        """
        # Load template if path
        if isinstance(template, (str, Path)):
            template = self._load_template(template)

        # Merge variables
        merged_vars = {**template.variables, **(variables or {})}

        # Merge boundaries
        boundary_dict = {**template.boundaries, **(boundaries.to_dict() if boundaries else {})}
        effective_boundaries = BoundaryConstraints(**boundary_dict) if boundary_dict else BoundaryConstraints()

        # Initialize result
        result = PipelineResult(template_name=template.name)
        start_time = time.time()

        # Initialize Scribe store (editable knowledge state)
        if self.scribe_enabled:
            self._scribe_store = ScribeStore(self.scribe_path)
            logger.debug(f"Scribe initialized at {self.scribe_path}")

        # Initialize Chronicle adapter
        if self.chronicle_enabled:
            self._chronicle = PipelineChronicleAdapter(cwd=cwd)
            self._chronicle.start_session(template.name, request)

        # Initialize report writer
        if self.report_enabled:
            self._report_writer = PipelineReportWriter(
                template_name=template.name,
                request=request,
                cwd=cwd,
            )

        # Track execution state
        phase_outputs: dict[str, Any] = {}
        should_resume = resume_from is not None
        current_context = context or ""

        try:
            for phase_config in template.phases:
                # Skip until resume point
                if should_resume:
                    if phase_config.name == resume_from:
                        should_resume = False
                    else:
                        continue

                # Check if brainstorm should be skipped (based on prepare phase output)
                if phase_config.name == "design" and phase_config.type == PhaseType.BRAINSTORM:
                    prepare_output = phase_outputs.get("prepare", {})
                    if isinstance(prepare_output, dict) and prepare_output.get("skip_brainstorm", False):
                        skip_reason = prepare_output.get("skip_reason", "low complexity")
                        logger.info(f"Skipping brainstorm phase: {skip_reason}")
                        if self.on_progress:
                            self.on_progress(f"Skipping brainstorm: {skip_reason}")
                        # Record skip decision
                        self._record_decision(
                            question="Should brainstorm be executed?",
                            options=["execute", "skip"],
                            chosen="skip",
                            rationale=skip_reason,
                            phase_name=phase_config.name,
                        )
                        continue

                # Execute phase
                phase_output = await self._execute_phase(
                    phase_config,
                    request=request,
                    cwd=cwd,
                    variables=merged_vars,
                    boundaries=effective_boundaries,
                    context=current_context,
                    previous_outputs=phase_outputs,
                )

                result.phases.append(phase_output)
                phase_outputs[phase_config.name] = phase_output.output

                # Record phase in report
                if self._report_writer:
                    self._report_writer.record_phase(
                        phase_output,
                        phase_type=phase_config.type.value if phase_config.type else "",
                        provider=phase_config.provider if phase_config.provider else None,
                    )

                # Update context with phase output
                if phase_output.output:
                    current_context = self._build_context(current_context, phase_output)

                # Handle result
                if phase_output.result == PhaseResult.NEEDS_INPUT:
                    result.needs_input = True
                    result.pending_questions = phase_output.questions
                    result.can_resume = True
                    result.resume_phase = phase_config.name
                    break

                elif phase_output.result == PhaseResult.FAILED:
                    if phase_config.on_failure == "abort":
                        result.success = False
                        break
                    elif phase_config.on_failure == "skip":
                        continue
                    elif phase_config.on_failure == "ask_user":
                        if self.on_user_input:
                            question = f"Phase '{phase_config.name}' failed: {phase_output.error}. Continue anyway?"
                            answer = self.on_user_input([question, "(yes/no)"])
                            # Record user interaction in report
                            if self._report_writer and answer:
                                self._report_writer.record_user_query(question, answer)
                            if answer and answer.lower() in ("yes", "y"):
                                continue
                        result.success = False
                        break

            else:
                # All phases completed
                result.success = True
                if phase_outputs:
                    result.final_output = list(phase_outputs.values())[-1]

        except Exception as e:
            logger.exception(f"Pipeline execution failed: {e}")
            result.success = False
            result.phases.append(PhaseOutput(
                phase_name="pipeline_error",
                result=PhaseResult.FAILED,
                error=str(e),
            ))

        # Calculate totals
        result.total_duration_ms = int((time.time() - start_time) * 1000)
        result.total_cost_usd = sum(p.cost_usd for p in result.phases)

        # Finalize Chronicle
        chronicle = None
        if self.chronicle_enabled and self._chronicle:
            self._chronicle.end_session(
                success=result.success,
                final_output=str(result.final_output)[:500] if result.final_output else None,
            )

            chronicle = self._chronicle

            # Save Chronicle records
            if save_chronicle:
                try:
                    chronicle.save_to_directory(self.chronicle_dir)
                except Exception as e:
                    logger.warning(f"Failed to save Chronicle: {e}")

            self._chronicle = None

        # Generate final report
        report_path = None
        if self.report_enabled and self._report_writer:
            try:
                report_path = self._report_writer.finalize(result)
                if report_path:
                    result.report_path = str(report_path)
                    logger.info(f"Pipeline report generated: {report_path}")
            except Exception as e:
                logger.warning(f"Failed to generate report: {e}")
            finally:
                self._report_writer = None

        return result, chronicle

    @classmethod
    def list_templates(cls) -> list[dict[str, str]]:
        """List available workflow templates.

        Returns:
            List of template info dicts with name, description, path
        """
        templates = []
        if TEMPLATES_DIR.exists():
            for path in TEMPLATES_DIR.glob("*.yaml"):
                try:
                    template = WorkflowTemplate.from_yaml(path)
                    templates.append({
                        "name": template.name,
                        "description": template.description,
                        "version": template.version,
                        "path": str(path),
                    })
                except Exception as e:
                    logger.warning(f"Failed to load template {path}: {e}")
        return templates

    @classmethod
    def get_template(cls, name: str) -> Optional[WorkflowTemplate]:
        """Get template by name.

        Args:
            name: Template name (without .yaml extension)

        Returns:
            WorkflowTemplate or None if not found
        """
        path = TEMPLATES_DIR / f"{name}.yaml"
        if path.exists():
            return WorkflowTemplate.from_yaml(path)

        # Try exact path
        if Path(name).exists():
            return WorkflowTemplate.from_yaml(name)

        return None

    # =========================================================================
    # Phase Execution
    # =========================================================================

    async def _execute_phase(
        self,
        config: PhaseConfig,
        request: str,
        cwd: str,
        variables: dict[str, Any],
        boundaries: BoundaryConstraints,
        context: str,
        previous_outputs: dict[str, Any],
    ) -> PhaseOutput:
        """Execute a single phase with Scribe context injection.

        Scribe Integration:
        1. Before phase: Get scribe context for this phase (based on PHASE_SECTION_MAP)
        2. Inject context: Prepend scribe knowledge to phase context
        3. After phase: Update scribe based on PHASE_UPDATE_MAP
        4. On failure: Escalate clearance for retry
        """
        if self.on_phase_start:
            self.on_phase_start(config.name)

        start_time = time.time()

        # Get Scribe context for this phase
        scribe_context = ""
        scribe_clearance = Clearance.DEFAULT
        if self._scribe_store:
            scribe_context = self._scribe_store.get_for_phase(config.name)
            if scribe_context:
                logger.debug(
                    f"Scribe injected for phase '{config.name}': "
                    f"{len(scribe_context)} chars"
                )

        # Prepend scribe context to phase context
        enhanced_context = context
        if scribe_context:
            enhanced_context = f"{scribe_context}\n---\n\n{context}"

        try:
            if config.type == PhaseType.BRAINSTORM:
                output = await self._run_brainstorm(config, request, enhanced_context)
            elif config.type == PhaseType.CLARITY_GATE:
                output = await self._run_clarity_gate(config, request, enhanced_context)
            elif config.type == PhaseType.CLARITY_DIALOGUE:
                output = await self._run_clarity_dialogue(config, request, enhanced_context, previous_outputs)
            elif config.type == PhaseType.DELEGATION:
                output = await self._run_delegation(
                    config, request, cwd, boundaries, enhanced_context,
                    previous_outputs=previous_outputs,
                    scribe_store=self._scribe_store,
                )
            elif config.type == PhaseType.REVIEW:
                output = await self._run_review(config, previous_outputs, enhanced_context)
            elif config.type == PhaseType.CUSTOM:
                output = await self._run_custom(config, request, cwd, enhanced_context, previous_outputs)
            else:
                output = PhaseOutput(
                    phase_name=config.name,
                    result=PhaseResult.FAILED,
                    error=f"Unknown phase type: {config.type}",
                )

            output.duration_ms = int((time.time() - start_time) * 1000)

            # Update Scribe based on phase output (PHASE_UPDATE_MAP)
            if self._scribe_store and output.result == PhaseResult.SUCCESS:
                await self._update_scribe_from_phase(
                    config.name, output, previous_outputs
                )

        except asyncio.TimeoutError:
            output = PhaseOutput(
                phase_name=config.name,
                result=PhaseResult.FAILED,
                error=f"Phase timed out after {config.timeout_seconds}s",
                duration_ms=int((time.time() - start_time) * 1000),
            )
        except Exception as e:
            logger.exception(f"Phase {config.name} failed: {e}")
            output = PhaseOutput(
                phase_name=config.name,
                result=PhaseResult.FAILED,
                error=str(e),
                duration_ms=int((time.time() - start_time) * 1000),
            )

        # Record execution to Chronicle
        if self.chronicle_enabled and self._chronicle:
            exit_code = 0 if output.result == PhaseResult.SUCCESS else 1
            output_summary = ""
            if output.output:
                output_summary = str(output.output)[:300]
            elif output.error:
                output_summary = f"Error: {output.error}"

            self._chronicle.record_execution(
                tool_name=f"Phase:{config.name}",
                input_args={"type": config.type.value, "provider": config.provider},
                exit_code=exit_code,
                duration_ms=output.duration_ms,
                output_summary=output_summary,
            )

        if self.on_phase_end:
            self.on_phase_end(output)

        return output

    def _record_decision(
        self,
        question: str,
        options: list[str],
        chosen: str,
        rationale: str,
        *,
        phase_name: Optional[str] = None,
        provider: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> None:
        """Record a decision to Chronicle."""
        if not self.chronicle_enabled or not self._chronicle:
            return

        self._chronicle.record_decision(
            question=question,
            options=options,
            chosen=chosen,
            rationale=rationale,
            phase_name=phase_name,
            provider=provider,
            confidence=confidence,
        )

    async def _run_brainstorm(
        self,
        config: PhaseConfig,
        request: str,
        context: str,
    ) -> PhaseOutput:
        """Run brainstorm phase."""
        from ..multi_llm import BrainstormModule

        if self._brainstorm_module is None:
            self._brainstorm_module = BrainstormModule(
                self.providers,
                on_progress=self.on_progress,
                on_stream=self.on_stream,
            )

        topic = config.config.get("topic_template", "Design approach for: {request}")
        topic = topic.format(request=request, context=context)

        self._record_decision(
            question=f"How to approach: {topic[:100]}",
            options=["brainstorm", "skip"],
            chosen="brainstorm",
            rationale=f"Starting multi-LLM brainstorm session",
            phase_name=config.name,
        )

        result = await self._brainstorm_module.run_session(topic, context=context)

        # Record discussion summary as a decision
        if result.rounds:
            participants = set()
            for round_data in result.rounds:
                for entry in round_data.discussion_log:
                    speaker = entry.speaker.value if hasattr(entry.speaker, 'value') else str(entry.speaker)
                    participants.add(speaker)

            self._record_decision(
                question=f"Brainstorm discussion outcome for: {topic[:80]}",
                options=["consensus", "no_consensus", "escalate"],
                chosen="consensus" if result.consensus.success else "no_consensus",
                rationale=f"Participants: {', '.join(participants)}. "
                          f"Rounds: {len(result.rounds)}. "
                          f"Key points from discussion.",
                phase_name=config.name,
                provider=", ".join(participants),
            )

        if not result.consensus.success:
            self._record_decision(
                question="How to handle brainstorm failure?",
                options=["ask_user", "abort", "retry"],
                chosen="ask_user",
                rationale="Brainstorm failed to reach consensus, need user input",
                phase_name=config.name,
            )
            return PhaseOutput(
                phase_name=config.name,
                result=PhaseResult.NEEDS_INPUT,
                questions=result.consensus.user_questions or [
                    "The design team could not reach consensus."
                ],
            )

        self._record_decision(
            question="Accept brainstorm consensus?",
            options=["accept", "reject", "modify"],
            chosen="accept",
            rationale=f"Consensus reached: {result.consensus.final_decision[:200]}",
            phase_name=config.name,
        )
        return PhaseOutput(
            phase_name=config.name,
            result=PhaseResult.SUCCESS,
            output=result.consensus.final_decision,
        )

    async def _run_clarity_gate(
        self,
        config: PhaseConfig,
        request: str,
        context: str,
    ) -> PhaseOutput:
        """Run clarity gate phase."""
        from ..multi_llm import ClarityGate

        if self._clarity_gate is None:
            claude = self.providers.get(Provider.CLAUDE)
            if not claude:
                self._record_decision(
                    question="Can we run ClarityGate?",
                    options=["run", "skip", "fail"],
                    chosen="fail",
                    rationale="Claude provider not available for ClarityGate",
                    phase_name=config.name,
                )
                return PhaseOutput(
                    phase_name=config.name,
                    result=PhaseResult.FAILED,
                    error="Claude provider not available for ClarityGate",
                )
            self._clarity_gate = ClarityGate(claude)

        task = Task(
            task_id=f"clarity-{config.name}",
            objective=request,
            context=context,
        )

        self._record_decision(
            question=f"Is task clear? {request[:80]}",
            options=["assess", "assume_clear"],
            chosen="assess",
            rationale="Running ClarityGate assessment",
            phase_name=config.name,
            provider="claude",
        )
        assessment = await self._clarity_gate.assess(task)

        # Record the clarity assessment result
        self._record_decision(
            question=f"Clarity assessment result for: {request[:60]}",
            options=["CLEAR", "NEEDS_CLARIFICATION", "AMBIGUOUS"],
            chosen=assessment.level.value,
            rationale=f"Score: {assessment.score:.2f}. Recommendation: {assessment.recommendation}",
            phase_name=config.name,
            provider="claude",
            confidence=assessment.score,
        )

        if assessment.level == ClarityLevel.AMBIGUOUS:
            self._record_decision(
                question="How to handle ambiguous task?",
                options=["ask_user", "proceed_anyway", "abort"],
                chosen="ask_user",
                rationale=f"Task is ambiguous (score: {assessment.score:.2f}). "
                          f"Questions: {assessment.self_questions}",
                phase_name=config.name,
            )
            return PhaseOutput(
                phase_name=config.name,
                result=PhaseResult.NEEDS_INPUT,
                output=assessment,
                questions=assessment.self_questions or ["Task is too ambiguous."],
            )
        return PhaseOutput(
            phase_name=config.name,
            result=PhaseResult.SUCCESS,
            output={
                "level": assessment.level.value,
                "score": assessment.score,
                "recommendation": assessment.recommendation,
            },
        )

    async def _run_clarity_dialogue(
        self,
        config: PhaseConfig,
        request: str,
        context: str,
        previous_outputs: dict[str, Any],
    ) -> PhaseOutput:
        """Run clarity dialogue phase (Claude questions → Gemini answers).

        This implements the atlas-auto clarity check protocol:
        1. Claude identifies ambiguities and generates questions
        2. Gemini answers with session context from previous brainstorm
        3. Loop until clarity is achieved or max_rounds reached
        """
        claude = self.providers.get(Provider.CLAUDE)
        gemini = self.providers.get(Provider.GEMINI)

        if not claude or not gemini:
            missing = []
            if not claude:
                missing.append("Claude")
            if not gemini:
                missing.append("Gemini")
            return PhaseOutput(
                phase_name=config.name,
                result=PhaseResult.FAILED,
                error=f"Required providers not available: {', '.join(missing)}",
            )

        max_rounds = config.config.get("max_rounds", 3)
        required_outputs = config.config.get("required_outputs", [
            "term_definitions",
            "scope_boundaries",
            "exception_handling",
            "success_criteria",
        ])

        # Build session context from previous phases (especially brainstorm)
        session_context = context
        if "design" in previous_outputs:
            design_output = previous_outputs["design"]
            if isinstance(design_output, str):
                session_context = f"[Previous Design Discussion]\n{design_output}\n\n[Current Task]\n{request}"
            else:
                session_context = f"[Previous Design Discussion]\n{str(design_output)}\n\n[Current Task]\n{request}"

        clarity_results = {key: None for key in required_outputs}
        dialogue_history = []

        for round_num in range(1, max_rounds + 1):
            # Step 1: Claude identifies ambiguities and asks questions
            claude_prompt = f"""You are analyzing a task for ambiguities before implementation.

Task: {request}

Context from previous discussion:
{session_context}

Already clarified:
{[k for k, v in clarity_results.items() if v is not None]}

Still need clarification:
{[k for k, v in clarity_results.items() if v is None]}

Previous dialogue:
{chr(10).join(dialogue_history) if dialogue_history else "None yet"}

Generate specific clarifying questions for the unclarified items.
If everything is clear, respond with "CLARITY_ACHIEVED" and a summary.
Format your questions clearly, one per line."""

            self._record_decision(
                question=f"Clarity dialogue round {round_num}: What needs clarification?",
                options=["ask_questions", "clarity_achieved"],
                chosen="ask_questions" if round_num == 1 else "continue",
                rationale=f"Round {round_num}/{max_rounds} of clarity dialogue",
                phase_name=config.name,
                provider="claude",
            )

            claude_result = await claude.run(claude_prompt)
            claude_response = claude_result.text if hasattr(claude_result, "text") else str(claude_result)

            dialogue_history.append(f"[Claude Round {round_num}]: {claude_response}")

            # Check if clarity achieved
            if "CLARITY_ACHIEVED" in claude_response.upper():
                self._record_decision(
                    question="Has clarity been achieved?",
                    options=["yes", "no", "partial"],
                    chosen="yes",
                    rationale=f"Claude confirmed clarity after {round_num} rounds",
                    phase_name=config.name,
                    provider="claude",
                    confidence=0.9,
                )
                return PhaseOutput(
                    phase_name=config.name,
                    result=PhaseResult.SUCCESS,
                    output={
                        "rounds_taken": round_num,
                        "dialogue": dialogue_history,
                        "clarity_results": clarity_results,
                        "summary": claude_response,
                    },
                )

            # Step 2: Gemini answers with session context (session resume)
            gemini_prompt = f"""You are continuing a design discussion to clarify implementation details.

IMPORTANT: You are resuming from the previous brainstorm session. Consider all prior context.

Original Task: {request}

Session Context (from brainstorm):
{session_context}

Claude's clarifying questions:
{claude_response}

Provide clear, definitive answers to each question.
Focus on:
- Term definitions: What exactly does each term mean?
- Scope boundaries: What is in/out of scope?
- Exception handling: How should edge cases be handled?
- Success criteria: How do we know it's done correctly?

Be decisive. Avoid "it depends" - make concrete recommendations."""

            gemini_result = await gemini.run(gemini_prompt)
            gemini_response = gemini_result.text if hasattr(gemini_result, "text") else str(gemini_result)

            dialogue_history.append(f"[Gemini Round {round_num}]: {gemini_response}")

            self._record_decision(
                question=f"Gemini's clarification round {round_num}",
                options=["accept", "need_more", "reject"],
                chosen="accept",
                rationale=f"Gemini provided clarifications: {gemini_response[:200]}",
                phase_name=config.name,
                provider="gemini",
            )

            # Update session context with the dialogue
            session_context += f"\n\n[Clarification Round {round_num}]\nQ: {claude_response[:500]}\nA: {gemini_response[:500]}"

            # Parse Gemini's response to update clarity results
            response_lower = gemini_response.lower()
            for key in required_outputs:
                if key.replace("_", " ") in response_lower or key in response_lower:
                    clarity_results[key] = f"Round {round_num}: Addressed"

        # Max rounds reached
        unresolved = [k for k, v in clarity_results.items() if v is None]
        if unresolved:
            self._record_decision(
                question="Max clarity rounds reached - how to proceed?",
                options=["ask_user", "proceed_anyway", "abort"],
                chosen="ask_user",
                rationale=f"Unresolved after {max_rounds} rounds: {unresolved}",
                phase_name=config.name,
            )
            return PhaseOutput(
                phase_name=config.name,
                result=PhaseResult.NEEDS_INPUT,
                output={
                    "rounds_taken": max_rounds,
                    "dialogue": dialogue_history,
                    "clarity_results": clarity_results,
                },
                questions=[
                    f"Clarity check incomplete after {max_rounds} rounds.",
                    f"Unresolved items: {', '.join(unresolved)}",
                    "Please provide guidance on these items.",
                ],
            )

        return PhaseOutput(
            phase_name=config.name,
            result=PhaseResult.SUCCESS,
            output={
                "rounds_taken": max_rounds,
                "dialogue": dialogue_history,
                "clarity_results": clarity_results,
            },
        )

    async def _run_delegation(
        self,
        config: PhaseConfig,
        request: str,
        cwd: str,
        boundaries: BoundaryConstraints,
        context: str,
        *,
        previous_outputs: Optional[dict] = None,
        scribe_store: Optional[ScribeStore] = None,
    ) -> PhaseOutput:
        """Run delegation phase with evidence-based context injection.

        This phase:
        1. Gets evidence context via EvidenceGate (DETAILS/DEEP_DIVE based on task type)
        2. Injects evidence into the delegation context
        3. On failure, escalates to deeper context and retries
        4. Uses scribe_digest in cache key for invalidation
        """
        from ..multi_llm import ClaudeCodeExecutor, CLAUDE_SDK_AVAILABLE
        from .evidence_gate import EvidenceGate, get_delegation_evidence_context

        claude = self.providers.get(Provider.CLAUDE)
        if not claude:
            self._record_decision(
                question="Can we run delegation?",
                options=["run", "skip", "fail"],
                chosen="fail",
                rationale="Claude provider not available for delegation",
                phase_name=config.name,
            )
            return PhaseOutput(
                phase_name=config.name,
                result=PhaseResult.FAILED,
                error="Claude provider not available for delegation",
            )

        if self._claude_executor is None:
            self._claude_executor = ClaudeCodeExecutor(claude, cwd=cwd)

        # Initialize evidence gate if RAGClient available
        if self._evidence_gate is None and self.rag_client is not None:
            self._evidence_gate = EvidenceGate(self.rag_client, enforce_requirements=False)

        # Get task type and system prompt from previous preparation phase
        task_type = "unknown"
        generated_system_prompt: Optional[str] = None

        # Check for prepare phase output (may be named "prepare" or "preparation")
        prep_output = None
        if previous_outputs:
            prep_output = previous_outputs.get("prepare") or previous_outputs.get("preparation")

        # Extract plugin chain for suggested_plugins
        plugin_chain: Optional[str] = None

        if prep_output and isinstance(prep_output, dict):
            task_type = prep_output.get("task_type", "unknown")
            generated_system_prompt = prep_output.get("system_prompt")
            plugin_chain = prep_output.get("plugin_chain")

            if generated_system_prompt:
                template_name = prep_output.get("system_template", "custom")
                logger.info(
                    f"Using system prompt from prepare phase (task_type={task_type}, "
                    f"template={template_name}, prompt_len={len(generated_system_prompt)})"
                )
                self._record_decision(
                    question="Which system prompt to inject?",
                    options=["custom_from_prepare", "session_template", "none"],
                    chosen="custom_from_prepare",
                    rationale=f"Using generated system prompt: template={template_name}, "
                              f"task_type={task_type}, length={len(generated_system_prompt)} chars",
                    phase_name=config.name,
                    provider="claude",
                )

            if plugin_chain:
                logger.info(f"Using plugin from prepare phase: {plugin_chain}")

        # =================================================================
        # Context Gathering (Active + Passive)
        # =================================================================

        # Phase A: Active Context Scout (Gemini-driven)
        # Asks Gemini what files/symbols are needed, then fetches from RAG
        scout_context = ""
        if config.config.get("enable_scout", True):  # Enabled by default
            scout_context = await self._gather_context_with_scout(
                request=request,
                cwd=cwd,
                task_type=task_type,
            )
            if scout_context:
                logger.info(f"Scout gathered {len(scout_context)} chars of context")

        # Phase B: Passive Evidence Gate (rule-based)
        # Auto-promotes based on task type (INDEX → DETAILS → DEEP_DIVE)
        evidence_context = ""

        # Build session_id with scribe_digest for cache invalidation
        scribe_digest = "none"
        if scribe_store:
            scribe_digest = scribe_store.get_digest(config.name)
        session_id = f"del-{config.name}-scribe:{scribe_digest}"

        if self._evidence_gate is not None:
            evidence = await self._evidence_gate.get_context_for_task(
                task_type=task_type,
                query=request,
                cwd=cwd,
                session_id=session_id,
            )

            if evidence.context_text:
                evidence_context = get_delegation_evidence_context(evidence, task_type)
                self._record_decision(
                    question="Evidence stage for delegation?",
                    options=["INDEX", "DETAILS", "DEEP_DIVE", "NONE"],
                    chosen=evidence.stage.value,
                    rationale=f"Retrieved {evidence.chunk_count} chunks for {task_type}",
                    phase_name=config.name,
                )

            if not evidence.sufficient and evidence.warning:
                logger.warning(f"Evidence gate warning: {evidence.warning}")

        # Build enhanced context with Scout + Evidence
        # Priority: Scout (active, targeted) > Evidence (passive, broad) > Base context
        context_parts = []
        if scout_context:
            context_parts.append(scout_context)
        if evidence_context:
            context_parts.append(evidence_context)
        if context:
            context_parts.append(context)

        enhanced_context = "\n\n---\n\n".join(context_parts) if context_parts else context

        # Get template if specified
        session_template = None
        template_name = config.config.get("session_template", "default")
        if CLAUDE_SDK_AVAILABLE and config.config.get("session_template"):
            from ..multi_llm import SessionTemplate
            try:
                session_template = SessionTemplate(template_name)
                self._record_decision(
                    question="Which session template to use?",
                    options=["default", template_name, "custom"],
                    chosen=template_name,
                    rationale=f"Using configured session template: {template_name}",
                    phase_name=config.name,
                    provider="claude",
                )
            except ValueError:
                logger.warning(f"Unknown session template: {template_name}")
                self._record_decision(
                    question="Which session template to use?",
                    options=["default", template_name],
                    chosen="default",
                    rationale=f"Unknown session template: {template_name}, falling back to default",
                    phase_name=config.name,
                    provider="claude",
                )

        # Build configurable options with suggested plugins from prepare phase
        options = ConfigurableOptions(
            suggested_plugins=[plugin_chain] if plugin_chain else [],
        )

        delegation = ClaudeCodeDelegation(
            delegation_id=session_id,
            objective=request,
            design_context=enhanced_context,
            boundaries=boundaries,
            options=options,
            cwd=cwd,
        )

        self._record_decision(
            question=f"How to implement: {request[:80]}",
            options=["delegate_to_claude", "manual", "skip"],
            chosen="delegate_to_claude",
            rationale=f"Delegating implementation to Claude with boundaries: {boundaries}",
            phase_name=config.name,
            provider="claude",
        )

        # Execute delegation with retry on failure
        max_retries = config.config.get("max_retries", 1)
        attempt = 0
        outcome = None

        while attempt <= max_retries:
            attempt += 1

            outcome = await self._claude_executor.execute(
                delegation,
                session_template=session_template,
                system_prompt=generated_system_prompt,  # From prepare phase
            )

            if outcome.success:
                break

            # On failure, try escalating evidence
            if attempt <= max_retries and self._evidence_gate is not None:
                self._record_decision(
                    question="How to handle delegation failure?",
                    options=["retry_with_escalation", "abort"],
                    chosen="retry_with_escalation",
                    rationale=f"Attempt {attempt}/{max_retries+1} failed: {outcome.error}. "
                              f"Escalating evidence context.",
                    phase_name=config.name,
                    provider="claude",
                    confidence=0.5,
                )

                # Escalate to deeper context
                escalated = await self._evidence_gate.escalate_on_failure(
                    error=outcome.error or "Execution failed",
                    cwd=cwd,
                    session_id=session_id,
                    task_type=task_type,
                )

                if escalated.context_text:
                    escalated_context = get_delegation_evidence_context(escalated, task_type)
                    enhanced_context = f"{escalated_context}\n\n---\n\n{context}"
                    delegation.design_context = enhanced_context

                    logger.info(
                        f"Retry with escalated context: {escalated.stage.value} "
                        f"({escalated.chunk_count} chunks)"
                    )

        # Final outcome
        files_modified = getattr(outcome, 'files_modified', [])

        if not outcome.success:
            self._record_decision(
                question="How to handle delegation failure?",
                options=["retry", "escalate", "abort"],
                chosen="abort",
                rationale=f"Delegation failed after {attempt} attempts: {outcome.error}",
                phase_name=config.name,
                provider="claude",
                confidence=0.3,
            )
            return PhaseOutput(
                phase_name=config.name,
                result=PhaseResult.FAILED,
                error=outcome.error,
                output=outcome,
                cost_usd=outcome.budget_used_usd,
            )

        self._record_decision(
            question="Accept delegation result?",
            options=["accept", "reject", "modify"],
            chosen="accept",
            rationale=f"Delegation succeeded. Modified {len(files_modified)} files: "
                      f"{files_modified[:5] if files_modified else 'None'}",
            phase_name=config.name,
            provider="claude",
            confidence=0.8,
        )

        # Cleanup evidence gate session
        if self._evidence_gate is not None:
            self._evidence_gate.reset_session(session_id)

        return PhaseOutput(
            phase_name=config.name,
            result=PhaseResult.SUCCESS,
            output=outcome,
            cost_usd=outcome.budget_used_usd,
        )

    async def _run_review(
        self,
        config: PhaseConfig,
        previous_outputs: dict[str, Any],
        context: str,
    ) -> PhaseOutput:
        """Run review phase with Review Contract support.

        Uses provider from config (default: gemini for analysis tasks).
        Falls back to available providers if specified one is unavailable.
        """
        # Get review provider from config (prefer gemini for analysis)
        review_provider_name = config.provider or "gemini"
        provider_map = {
            "gemini": Provider.GEMINI,
            "claude": Provider.CLAUDE,
            "codex": Provider.CODEX,
        }

        review_provider_enum = provider_map.get(review_provider_name, Provider.GEMINI)
        reviewer = self.providers.get(review_provider_enum)

        # Fallback chain: specified -> gemini -> claude
        if not reviewer:
            for fallback in [Provider.GEMINI, Provider.CLAUDE]:
                reviewer = self.providers.get(fallback)
                if reviewer:
                    logger.info(f"Review provider {review_provider_name} unavailable, using {fallback.value}")
                    review_provider_name = fallback.value
                    break

        if not reviewer:
            self._record_decision(
                question="Can we run code review?",
                options=["run", "skip"],
                chosen="skip",
                rationale="No review-capable provider available, skipping review",
                phase_name=config.name,
            )
            return PhaseOutput(
                phase_name=config.name,
                result=PhaseResult.SKIPPED,
                output="No review-capable provider available, skipping review",
            )

        # Get review type and contract
        review_type = config.config.get("review_type", "result")  # plan or result
        review_contract_name = config.config.get("review_contract")
        aspects = config.config.get("aspects", ["code_quality", "security"])
        max_iterations = config.config.get("max_iterations", 3)

        # Load Review Contract if specified
        review_contract = ""
        if review_contract_name:
            contract_path = Path(__file__).parent.parent / "prompts" / f"{review_contract_name}.md"
            if contract_path.exists():
                review_contract = contract_path.read_text(encoding="utf-8")
                self._record_decision(
                    question="Load Review Contract?",
                    options=["load", "skip"],
                    chosen="load",
                    rationale=f"Loaded review contract: {review_contract_name}",
                    phase_name=config.name,
                    provider=review_provider_name,
                )
            else:
                logger.warning(f"Review contract not found: {contract_path}")

        # Determine what to review based on review_type
        if review_type == "plan":
            # Review the plan from previous phase
            plan_output = previous_outputs.get("plan", "")
            if not plan_output:
                return PhaseOutput(
                    phase_name=config.name,
                    result=PhaseResult.SKIPPED,
                    output="No plan available to review",
                )
            # Extract actual plan content from DelegationOutcome or other types
            review_target = self._extract_review_content(plan_output, "plan")
            if not review_target or len(review_target.strip()) < 50:
                logger.warning(f"Plan content too short ({len(review_target)} chars), may lack detail")
            target_description = "implementation plan"
        else:
            # Review the result from execution phase
            delegation_output = None
            for key in ["execute", "implementation", "delegation", "fix", "result_fix"]:
                if key in previous_outputs:
                    delegation_output = previous_outputs[key]
                    break

            if not delegation_output:
                self._record_decision(
                    question="What to review?",
                    options=["review_output", "skip"],
                    chosen="skip",
                    rationale="No delegation output available to review",
                    phase_name=config.name,
                )
                return PhaseOutput(
                    phase_name=config.name,
                    result=PhaseResult.SKIPPED,
                    output="No delegation output to review",
                )

            # Extract actual result content from DelegationOutcome
            review_target = self._extract_review_content(delegation_output, "result")
            files = getattr(delegation_output, "files_modified", [])
            target_description = f"{len(files)} modified files" if files else "implementation result"

        self._record_decision(
            question=f"Review {target_description}?",
            options=["run_review", "skip", "partial_review"],
            chosen="run_review",
            rationale=f"Running {review_type} review with aspects: {aspects}",
            phase_name=config.name,
            provider=review_provider_name,
        )

        # Build review prompt with contract
        review_prompt = f"""[REVIEW CONTRACT APPLIED]

{review_contract if review_contract else ''}

---

## Review Task

Review Type: {review_type.upper()} REVIEW
Aspects to check: {', '.join(aspects)}

Context:
{context}

Target:
{review_target}

---

Provide your review in the following YAML format:

```yaml
{review_type}_review:
  verdict: approved | needs_revision | rejected
  critical_issues:
    - category: [missing_work | conflict | hallucination | placeholder | hardcoding | no_verification]
      description: "문제 설명"
      location: "위치"
      severity: critical | major | minor
  suggestions: []
  iteration: 1
```

Remember:
- Only flag issues from the Critical Risk Rules
- Do NOT suggest over-engineering or scope expansion
- Be specific with locations and evidence
"""
        result = await reviewer.run(review_prompt)
        review_text = result.text if hasattr(result, "text") else str(result)

        # Parse verdict from response
        # Empty or very short response should not be auto-approved
        if not review_text or len(review_text.strip()) < 20:
            verdict = "needs_revision"
            logger.warning(f"Review response too short ({len(review_text)} chars), treating as needs_revision")
        elif "needs_revision" in review_text.lower():
            verdict = "needs_revision"
        elif "rejected" in review_text.lower():
            verdict = "rejected"
        elif "approved" in review_text.lower():
            verdict = "approved"
        else:
            # No explicit verdict found - require explicit approval
            verdict = "needs_revision"
            logger.warning("No explicit verdict in review response, treating as needs_revision")

        # Determine result based on verdict
        if verdict == "approved":
            self._record_decision(
                question=f"{review_type.title()} review verdict?",
                options=["approved", "needs_revision", "rejected"],
                chosen="approved",
                rationale=f"{review_provider_name.title()} approved the {review_type}",
                phase_name=config.name,
                provider=review_provider_name,
                confidence=0.85,
            )
            return PhaseOutput(
                phase_name=config.name,
                result=PhaseResult.SUCCESS,
                output={
                    "verdict": verdict,
                    "review": review_text,
                    "review_type": review_type,
                },
            )
        else:
            self._record_decision(
                question=f"{review_type.title()} review verdict?",
                options=["approved", "needs_revision", "rejected"],
                chosen=verdict,
                rationale=f"{review_provider_name.title()} found issues: {review_text[:200]}",
                phase_name=config.name,
                provider=review_provider_name,
                confidence=0.75,
            )
            # Return FAILED to trigger fix phase
            return PhaseOutput(
                phase_name=config.name,
                result=PhaseResult.FAILED,
                error=f"Review verdict: {verdict}",
                output={
                    "verdict": verdict,
                    "review": review_text,
                    "review_type": review_type,
                    "issues": review_text,
                },
            )

    async def _run_custom(
        self,
        config: PhaseConfig,
        request: str,
        cwd: str,
        context: str,
        previous_outputs: dict[str, Any],
    ) -> PhaseOutput:
        """Run custom phase handler.

        Supports built-in custom handlers:
        - task_preparation: Classify task and configure templates/plugins
        """
        handler_name = config.config.get("handler", "")

        if handler_name == "task_preparation":
            return await self._run_task_preparation(config, request, cwd)
        else:
            return PhaseOutput(
                phase_name=config.name,
                result=PhaseResult.FAILED,
                error=f"Unknown custom handler: {handler_name}",
            )

    async def _run_task_preparation(
        self,
        config: PhaseConfig,
        request: str,
        cwd: str,
    ) -> PhaseOutput:
        """Run task preparation phase.

        1. Detect task type from request
        2. Analyze complexity to decide if brainstorm needed
        3. Select appropriate system prompt template
        4. Configure plugin chain
        5. Extract codebase context (if enabled)

        Returns output dict with configuration for subsequent phases.
        """
        from .prompt_generator import (
            detect_task_type,
            get_task_config,
            SystemPromptGenerator,
        )

        # Detect task type
        task_type = detect_task_type(request)
        task_config = get_task_config(task_type)

        logger.info(f"Task type detected: {task_type} -> template: {task_config.template_name}")

        self._record_decision(
            question="What type of task is this?",
            options=["new_feature", "refactoring", "integration", "project_creation", "bug_fix", "unknown"],
            chosen=task_type,
            rationale=f"Detected from request keywords",
            phase_name=config.name,
            confidence=0.75,
        )

        # Analyze complexity to decide if brainstorm is needed
        skip_brainstorm, skip_reason = self._analyze_complexity(request, task_type, cwd)

        self._record_decision(
            question="Is brainstorming needed for this task?",
            options=["needed", "skip"],
            chosen="skip" if skip_brainstorm else "needed",
            rationale=skip_reason if skip_brainstorm else "Task complexity requires multi-LLM discussion",
            phase_name=config.name,
        )

        # Extract codebase context if requested
        codebase_context = ""
        project_conventions = ""

        if config.config.get("extract_codebase_context", True):
            # Pass request as query for SCIP relevance filtering
            codebase_context = await self._extract_codebase_context(cwd, query=request)
            project_conventions = await self._extract_project_conventions(cwd)

        # Generate system prompt
        generator = SystemPromptGenerator()
        system_prompt = generator.generate(
            task_type,
            codebase_context=codebase_context,
            project_conventions=project_conventions,
        )

        # Return configuration for subsequent phases
        output = {
            "task_type": task_type,
            "system_template": task_config.template_name,
            "plugin_chain": task_config.plugins[0] if task_config.plugins else "feature-dev:feature-dev",
            "codebase_context": codebase_context,
            "project_conventions": project_conventions,
            "system_prompt": system_prompt,
            "review_focus": task_config.review_focus,
            "behavior_critical": task_config.behavior_critical,
            # Brainstorm skip decision
            "skip_brainstorm": skip_brainstorm,
            "skip_reason": skip_reason,
        }

        return PhaseOutput(
            phase_name=config.name,
            result=PhaseResult.SUCCESS,
            output=output,
        )

    def _analyze_complexity(
        self,
        request: str,
        task_type: str,
        cwd: str,
    ) -> tuple[bool, str]:
        """Analyze task complexity to decide if brainstorming is needed.

        Returns:
            (skip_brainstorm, skip_reason): Whether to skip and why

        Complexity factors:
        - Query length and keyword analysis
        - Task type inherent complexity
        - Explicit file scope in request
        """
        # Simple tasks that don't need brainstorming
        SIMPLE_KEYWORDS = {
            "fix typo", "rename", "add logging", "add comment",
            "update version", "fix import", "remove unused",
            "오타 수정", "이름 변경", "로깅 추가", "주석 추가",
        }

        # Complex tasks that always need brainstorming
        COMPLEX_KEYWORDS = {
            "implement", "refactor", "integrate", "migrate",
            "architect", "design", "restructure", "overhaul",
            "구현", "리팩토링", "통합", "마이그레이션", "설계",
        }

        request_lower = request.lower()

        # Check for explicit simple task keywords
        for keyword in SIMPLE_KEYWORDS:
            if keyword in request_lower:
                return True, f"Simple task detected: '{keyword}'"

        # Check for explicit complex task keywords
        for keyword in COMPLEX_KEYWORDS:
            if keyword in request_lower:
                return False, f"Complex task detected: '{keyword}'"

        # Task type based heuristics
        if task_type == "bug_fix":
            # Bug fixes are usually focused
            if len(request) < 100:
                return True, "Short bug fix request"

        if task_type in ("new_feature", "refactoring", "integration"):
            # These usually benefit from brainstorming
            return False, f"Task type '{task_type}' typically requires design discussion"

        if task_type == "project_creation":
            # New projects need architectural discussion
            return False, "Project creation requires architectural planning"

        # Default: check query length as proxy for complexity
        if len(request) < 80:
            return True, "Short request, likely simple task"

        # Default: do brainstorm for safety
        return False, "Default: brainstorming for thorough analysis"

    async def _extract_codebase_context(self, cwd: str, query: str = "") -> str:
        """Extract codebase context for template injection.

        Uses SCIP-based code intelligence to gather project structure,
        key symbols, and entry points. Falls back to basic project markers
        if SCIP is not available or RAGClient is not configured.

        Args:
            cwd: Current working directory (project root)
            query: Task description for relevance-based filtering

        Returns:
            Formatted codebase context string for template injection

        Note:
            SCIP-based context is fetched through RAGClient HTTP API,
            not through direct import (U-llm-sdk must not import from mv_rag).
            See: U-llm-sdk/AGENTS.md for architecture rules.
        """
        if not cwd:
            return "(No working directory specified)"

        cwd_path = Path(cwd)
        if not cwd_path.exists():
            return f"(Working directory not found: {cwd})"

        # Try SCIP-based context via RAGClient HTTP
        if self.rag_client is not None:
            try:
                context = await self.rag_client.get_codebase_context(
                    cwd=cwd,
                    query=query or "repository overview",
                    stage="INDEX",
                )
                if context:
                    logger.debug(f"Retrieved SCIP codebase context ({len(context)} chars)")
                    return context
            except Exception as e:
                logger.warning(f"RAGClient codebase context failed: {e}, falling back to basic")

        # Fallback to basic project markers
        return self._extract_basic_context(cwd_path)

    def _extract_basic_context(self, cwd_path: Path) -> str:
        """Extract basic project context using file markers.

        Fallback method when SCIP is not available.
        """
        context_parts = []

        # Check for common project files
        project_markers = {
            "pyproject.toml": "Python (pyproject.toml)",
            "setup.py": "Python (setup.py)",
            "package.json": "Node.js/TypeScript",
            "Cargo.toml": "Rust",
            "go.mod": "Go",
            "pom.xml": "Java/Maven",
            "build.gradle": "Java/Gradle",
        }

        for marker, description in project_markers.items():
            if (cwd_path / marker).exists():
                context_parts.append(f"Project type: {description}")
                break

        # Check for src directory structure
        if (cwd_path / "src").is_dir():
            context_parts.append("Has src/ directory structure")

        # Check for tests
        for test_dir in ["tests", "test", "__tests__"]:
            if (cwd_path / test_dir).is_dir():
                context_parts.append(f"Has {test_dir}/ directory")
                break

        return "; ".join(context_parts) if context_parts else "(Standard project structure)"

    async def _extract_project_conventions(self, cwd: str) -> str:
        """Extract project conventions from CLAUDE.md or similar files."""
        if not cwd:
            return ""

        cwd_path = Path(cwd)

        # Look for convention files
        convention_files = ["CLAUDE.md", ".claude.md", "CONTRIBUTING.md"]

        for filename in convention_files:
            filepath = cwd_path / filename
            if filepath.exists():
                try:
                    content = filepath.read_text(encoding="utf-8")
                    # Return first 1000 chars as summary
                    return content[:1000]
                except Exception:
                    pass

        return "(No project conventions file found)"

    # =========================================================================
    # Context Scout (Active RAG Gathering)
    # =========================================================================

    async def _gather_context_with_scout(
        self,
        request: str,
        cwd: str,
        task_type: str = "unknown",
    ) -> str:
        """Gather context using Gemini as a scout to identify needed files.

        This implements "Active Querying" - instead of rule-based context injection,
        we use an LLM to analyze the request and determine what files/symbols
        are actually needed, then fetch them from RAG.

        Args:
            request: The user's task request
            cwd: Working directory
            task_type: Detected task type for context

        Returns:
            Gathered context string with relevant file contents
        """
        gemini = self.providers.get(Provider.GEMINI)
        if not gemini or not self.rag_client:
            logger.debug("Context Scout skipped: Gemini or RAGClient not available")
            return ""

        try:
            # Step 1: Ask Gemini to analyze what files/symbols are needed
            scout_prompt = f"""You are a code scout. Analyze this task and identify what files or code symbols need to be examined.

Task: {request}
Task Type: {task_type}
Working Directory: {cwd}

Respond in JSON format ONLY (no markdown, no explanation):
{{
  "files": ["path/to/file1.py", "path/to/file2.ts"],
  "symbols": ["ClassName", "function_name", "ModuleName"],
  "search_queries": ["authentication handler", "database connection"]
}}

Rules:
- List specific file paths if mentioned or implied in the task
- List class/function names that are likely relevant
- List search queries for finding related code
- Maximum 5 items per category
- If unsure, return empty arrays"""

            scout_result = await gemini.run(scout_prompt)
            scout_text = scout_result.text if hasattr(scout_result, "text") else str(scout_result)

            # Step 2: Parse the scout's response
            import json
            try:
                # Clean up response (remove markdown code blocks if present)
                clean_text = scout_text.strip()
                if clean_text.startswith("```"):
                    clean_text = clean_text.split("```")[1]
                    if clean_text.startswith("json"):
                        clean_text = clean_text[4:]
                scout_data = json.loads(clean_text)
            except json.JSONDecodeError:
                logger.warning(f"Scout response not valid JSON: {scout_text[:200]}")
                return ""

            files = scout_data.get("files", [])[:5]
            symbols = scout_data.get("symbols", [])[:5]
            queries = scout_data.get("search_queries", [])[:3]

            if not files and not symbols and not queries:
                logger.debug("Scout found no specific targets")
                return ""

            # Step 3: Fetch context from RAG
            context_parts = []

            # Fetch specific files
            for file_path in files:
                try:
                    content = await self.rag_client.get_file_content(
                        file_path=file_path,
                        cwd=cwd,
                    )
                    if content and len(content) > 50:
                        context_parts.append(f"### File: {file_path}\n```\n{content[:3000]}\n```")
                except Exception as e:
                    logger.debug(f"Scout: Could not fetch {file_path}: {e}")

            # Search for symbols
            for symbol in symbols:
                try:
                    results = await self.rag_client.search_symbols(
                        query=symbol,
                        cwd=cwd,
                        limit=2,
                    )
                    if results:
                        for result in results:
                            loc = result.get("location", "unknown")
                            code = result.get("code", "")[:1500]
                            if code:
                                context_parts.append(f"### Symbol: {symbol} ({loc})\n```\n{code}\n```")
                except Exception as e:
                    logger.debug(f"Scout: Could not search symbol {symbol}: {e}")

            # Execute search queries
            for query in queries:
                try:
                    results = await self.rag_client.semantic_search(
                        query=query,
                        cwd=cwd,
                        limit=2,
                    )
                    if results:
                        for result in results:
                            file_path = result.get("file", "unknown")
                            snippet = result.get("content", "")[:1000]
                            if snippet:
                                context_parts.append(f"### Search: \"{query}\" (in {file_path})\n```\n{snippet}\n```")
                except Exception as e:
                    logger.debug(f"Scout: Could not search '{query}': {e}")

            if context_parts:
                gathered = "\n\n".join(context_parts)
                logger.info(
                    f"Context Scout gathered {len(context_parts)} items "
                    f"({len(gathered)} chars) for task"
                )
                self._record_decision(
                    question="What context did Scout gather?",
                    options=["files", "symbols", "searches", "none"],
                    chosen=f"{len(files)} files, {len(symbols)} symbols, {len(queries)} searches",
                    rationale=f"Scout identified and fetched {len(context_parts)} relevant code sections",
                    phase_name="context_scout",
                    provider="gemini",
                )
                return f"## Scout-Gathered Context\n\n{gathered}"

            return ""

        except Exception as e:
            logger.warning(f"Context Scout failed: {e}")
            return ""

    # =========================================================================
    # Helpers
    # =========================================================================

    def _load_template(self, path: Union[str, Path]) -> WorkflowTemplate:
        """Load template from path or name."""
        path = Path(path)

        # Check templates directory first
        if not path.exists():
            template_path = TEMPLATES_DIR / f"{path.stem}.yaml"
            if template_path.exists():
                path = template_path

        return WorkflowTemplate.from_yaml(path)

    def _build_context(self, current: str, phase_output: PhaseOutput) -> str:
        """Build context from phase output."""
        output_str = ""
        if phase_output.output:
            if isinstance(phase_output.output, str):
                output_str = phase_output.output
            elif hasattr(phase_output.output, "to_dict"):
                output_str = str(phase_output.output.to_dict())
            else:
                output_str = str(phase_output.output)

        if current:
            return f"{current}\n\n[{phase_output.phase_name}]:\n{output_str}"
        return output_str

    def _extract_review_content(self, output: Any, review_type: str) -> str:
        """Extract actual content from phase output for review.

        This method extracts meaningful content from various output types,
        specifically designed to provide reviewers (like Codex) with
        the actual text/code to review instead of metadata objects.

        Args:
            output: Phase output (DelegationOutcome, dict, str, etc.)
            review_type: Type of review ("plan" or "result")

        Returns:
            Formatted string with actual content for review
        """
        from u_llm_sdk.types import DelegationOutcome

        # Handle DelegationOutcome objects (from delegation phases)
        if isinstance(output, DelegationOutcome):
            parts = []

            # Primary content: the summary/output text from Claude
            if output.summary and output.summary != "No output":
                parts.append("## Implementation Summary")
                parts.append(output.summary)
                parts.append("")

            # Files modified with details
            if output.files_modified:
                parts.append("## Files Modified")
                for file_path in output.files_modified:
                    parts.append(f"- {file_path}")
                parts.append("")

            # Commands executed
            if output.commands_run:
                parts.append("## Commands Executed")
                for cmd in output.commands_run[:10]:  # Limit to 10
                    parts.append(f"- `{cmd}`")
                parts.append("")

            # Execution metadata
            parts.append("## Execution Metadata")
            parts.append(f"- Success: {output.success}")
            parts.append(f"- Duration: {output.duration_ms}ms")
            parts.append(f"- Budget used: ${output.budget_used_usd:.4f}")
            if output.tests_passed is not None:
                parts.append(f"- Tests passed: {output.tests_passed}")
            if output.typecheck_passed is not None:
                parts.append(f"- Typecheck passed: {output.typecheck_passed}")

            # If we still have nothing substantial, try raw events
            if not parts or (len(parts) < 3 and output.raw_events):
                # Extract text content from raw events
                text_content = self._extract_text_from_events(output.raw_events)
                if text_content:
                    parts.insert(0, "## Raw Output Content")
                    parts.insert(1, text_content)
                    parts.insert(2, "")

            return "\n".join(parts) if parts else "(No substantial content found)"

        # Handle dict outputs (from clarity_check, prepare, etc.)
        if isinstance(output, dict):
            parts = []

            # Common dict fields to extract
            for key in ["plan", "summary", "design", "content", "result", "dialogue"]:
                if key in output:
                    value = output[key]
                    if isinstance(value, str) and len(value) > 20:
                        parts.append(f"## {key.title()}")
                        parts.append(value)
                        parts.append("")
                    elif isinstance(value, list):
                        parts.append(f"## {key.title()}")
                        for item in value[:20]:  # Limit items
                            parts.append(f"- {str(item)[:500]}")
                        parts.append("")

            # If specific fields not found, dump the whole dict
            if not parts:
                import json
                try:
                    parts.append(json.dumps(output, indent=2, ensure_ascii=False, default=str))
                except Exception:
                    parts.append(str(output))

            return "\n".join(parts)

        # Handle string outputs directly
        if isinstance(output, str):
            return output

        # Fallback: convert to string
        return str(output)

    def _extract_text_from_events(self, events: list) -> str:
        """Extract text content from raw stream events.

        Used as fallback when DelegationOutcome.summary is empty.
        """
        text_parts = []

        for event in events:
            if not isinstance(event, dict):
                continue

            event_type = event.get("type", "")

            # Extract text from message events
            if event_type in ("assistant", "message"):
                content = event.get("message", {}).get("content", [])
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text = block.get("text", "")
                            if text and len(text) > 20:
                                text_parts.append(text)

            # Extract from result events
            if event_type == "result":
                result_text = event.get("result", "")
                if isinstance(result_text, str) and len(result_text) > 20:
                    text_parts.append(result_text)

        # Deduplicate and join
        seen = set()
        unique_parts = []
        for part in text_parts:
            if part not in seen:
                seen.add(part)
                unique_parts.append(part)

        return "\n\n---\n\n".join(unique_parts[:10])  # Limit to 10 sections

    # =========================================================================
    # Scribe Update Helpers
    # =========================================================================

    async def _update_scribe_from_phase(
        self,
        phase_name: str,
        phase_output: PhaseOutput,
        previous_outputs: dict[str, Any],
    ) -> None:
        """Update Scribe based on phase output.

        Uses PHASE_UPDATE_MAP to determine which sections this phase should update.

        Args:
            phase_name: Name of the completed phase
            phase_output: Output from the phase
            previous_outputs: Outputs from previous phases
        """
        if not self._scribe_store:
            return

        update_sections = PHASE_UPDATE_MAP.get(phase_name, [])
        if not update_sections:
            return

        output = phase_output.output
        if not output:
            return

        updated_by = f"phase:{phase_name}"

        # Route to section-specific updaters
        for section_name in update_sections:
            try:
                if section_name == "repo_map":
                    await self._update_scribe_repo_map(output, updated_by)
                elif section_name == "conventions":
                    await self._update_scribe_conventions(output, updated_by)
                elif section_name == "constraints":
                    await self._update_scribe_constraints(output, updated_by)
                elif section_name == "plan":
                    await self._update_scribe_plan(output, updated_by)
                elif section_name == "risks":
                    await self._update_scribe_risks(output, updated_by)
                elif section_name == "failures":
                    await self._update_scribe_failures(output, updated_by)
            except Exception as e:
                logger.warning(f"Failed to update scribe section '{section_name}': {e}")

    async def _update_scribe_repo_map(self, output: Any, updated_by: str) -> None:
        """Update repo_map section from prepare phase output."""
        if not isinstance(output, dict):
            return

        codebase_context = output.get("codebase_context", "")
        if not codebase_context or codebase_context.startswith("("):
            return  # Skip placeholder or error messages

        self._scribe_store.upsert_section(
            ScribeType.REPO_MAP,
            "repo:overview",
            public_summary="Repository structure and key entry points.",
            public_notice="Auto-extracted during preparation phase.",
            sealed_payload={"full_context": codebase_context},
            provenance=[f"phase:{updated_by}"],
            updated_by=updated_by,
        )

    async def _update_scribe_conventions(self, output: Any, updated_by: str) -> None:
        """Update conventions section from prepare phase output."""
        if not isinstance(output, dict):
            return

        conventions = output.get("project_conventions", "")
        if not conventions or conventions.startswith("("):
            return

        self._scribe_store.upsert_section(
            ScribeType.CONVENTION,
            "conv:project",
            public_summary="Project-specific coding conventions.",
            public_notice="Extracted from CLAUDE.md or similar.",
            sealed_payload={"full_conventions": conventions},
            provenance=[f"phase:{updated_by}"],
            updated_by=updated_by,
        )

    async def _update_scribe_constraints(self, output: Any, updated_by: str) -> None:
        """Update constraints section from clarity_check phase output."""
        if not isinstance(output, dict):
            return

        # Extract clarity results
        clarity_results = output.get("clarity_results", {})
        if not clarity_results:
            return

        # Create constraint items from clarity results
        for key, value in clarity_results.items():
            if value:  # Only add if resolved
                item_id = f"constraint:{key}"
                self._scribe_store.upsert_section(
                    ScribeType.CONSTRAINT,
                    item_id,
                    public_summary=f"Clarified: {key.replace('_', ' ')}",
                    public_notice=f"Resolved in clarity check: {value}",
                    sealed_payload={"clarification": value},
                    provenance=[f"phase:{updated_by}"],
                    updated_by=updated_by,
                )

    async def _update_scribe_plan(self, output: Any, updated_by: str) -> None:
        """Update plan section from plan phase output."""
        if isinstance(output, str):
            # Plan is a string description
            self._scribe_store.upsert_section(
                ScribeType.PLAN_NODE,
                "plan:main",
                public_summary="Implementation plan for current task.",
                public_notice="Generated by planning phase.",
                sealed_payload={"full_plan": output},
                provenance=[f"phase:{updated_by}"],
                updated_by=updated_by,
            )
        elif isinstance(output, dict):
            plan_content = output.get("plan", output.get("summary", str(output)))
            self._scribe_store.upsert_section(
                ScribeType.PLAN_NODE,
                "plan:main",
                public_summary="Implementation plan for current task.",
                public_notice="Generated by planning phase.",
                sealed_payload={"plan_data": output},
                provenance=[f"phase:{updated_by}"],
                updated_by=updated_by,
            )

    async def _update_scribe_risks(self, output: Any, updated_by: str) -> None:
        """Update risks section from review phase output."""
        if not isinstance(output, dict):
            return

        # Extract review findings
        verdict = output.get("verdict", "")
        issues = output.get("issues", output.get("review", ""))

        if verdict in ("needs_revision", "rejected"):
            self._scribe_store.upsert_section(
                ScribeType.REVIEW_FINDING,
                f"review:{updated_by}",
                public_summary=f"Review verdict: {verdict}",
                public_notice=f"Issues found during {output.get('review_type', 'code')} review.",
                sealed_payload={"verdict": verdict, "issues": issues},
                provenance=[f"phase:{updated_by}"],
                updated_by=updated_by,
            )

    async def _update_scribe_failures(self, output: Any, updated_by: str) -> None:
        """Update failures section from fix phase output."""
        if not isinstance(output, dict):
            return

        # Record failure signature if there was an error
        error = output.get("error", "")
        if error:
            self._scribe_store.upsert_section(
                ScribeType.FAILURE_SIGNATURE,
                f"failure:{updated_by}:{hash(error) % 10000}",
                public_summary=f"Failure during {updated_by}.",
                public_notice=f"Error: {error[:100]}...",
                sealed_payload={"full_error": error, "phase": updated_by},
                provenance=[f"phase:{updated_by}"],
                updated_by=updated_by,
            )


__all__ = [
    "PipelineExecutor",
    "PipelineChronicleAdapter",
    "TEMPLATES_DIR",
]
