"""Chronicle Adapter for Pipeline Execution.

Converts pipeline execution events to Chronicle records.
Uses llm-types Chronicle types directly, stores locally as JSON.

This replaces the custom ExecutionTrace/PhaseTrace/LLMThought types
with the established Chronicle record system.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from u_llm_sdk.types.chronicle import (
    DecisionRecord,
    ExecutionRecord,
    ExecutionOutcome,
    SourceReference,
    SourceKind,
    RecordType,
    generate_record_id,
)

logger = logging.getLogger(__name__)


class PipelineChronicleAdapter:
    """Adapter that records pipeline execution as Chronicle records.

    Converts pipeline phase executions to proper Chronicle records:
    - Phase decisions → DecisionRecord
    - Phase executions → ExecutionRecord

    Usage:
        >>> adapter = PipelineChronicleAdapter(session_id="pipeline-001")
        >>> adapter.start_session("semi_autonomous", "Implement auth")
        >>>
        >>> # Record a decision
        >>> adapter.record_decision(
        ...     question="Which auth method to use?",
        ...     options=["JWT", "Session"],
        ...     chosen="JWT",
        ...     rationale="Stateless for API",
        ... )
        >>>
        >>> # Record an execution
        >>> adapter.record_execution(
        ...     tool_name="ClarityGate",
        ...     input_args={"task": "..."},
        ...     exit_code=0,
        ...     duration_ms=150,
        ...     output_summary="Task is CLEAR (score: 0.85)",
        ... )
        >>>
        >>> # Save all records
        >>> adapter.save_to_directory(".chronicle/pipelines")
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        *,
        cwd: Optional[str] = None,
    ):
        """Initialize the adapter.

        Args:
            session_id: Session identifier (generated if not provided)
            cwd: Working directory for source references
        """
        self.session_id = session_id or f"pipe_{uuid.uuid4().hex[:8]}"
        self.cwd = cwd or "."
        self.logical_step = 0

        # Stored records
        self._decisions: list[DecisionRecord] = []
        self._executions: list[ExecutionRecord] = []

        # Current active decision (for linking executions)
        self._active_decision_id: Optional[str] = None

        # Session metadata
        self._template_name: Optional[str] = None
        self._user_query: Optional[str] = None
        self._started_at: Optional[datetime] = None

    def start_session(
        self,
        template_name: str,
        user_query: str,
    ) -> str:
        """Start a new pipeline session.

        Creates a bootstrap decision for the session.

        Args:
            template_name: Name of the workflow template
            user_query: Original user request

        Returns:
            Session ID
        """
        self._template_name = template_name
        self._user_query = user_query
        self._started_at = datetime.now()
        self.logical_step = 0

        # Create bootstrap decision
        bootstrap = DecisionRecord.create(
            session_id=self.session_id,
            logical_step=self._next_step(),
            question=f"Execute pipeline '{template_name}' for: {user_query[:100]}",
            options_considered=["execute", "abort"],
            chosen_option="execute",
            rationale=f"User requested pipeline execution: {user_query[:200]}",
            sources=[
                SourceReference(
                    kind=SourceKind.EXTERNAL,
                    location=f"user_query:{self.session_id}",
                    snapshot_hash=self._hash_content(user_query),
                    description=f"Original user query",
                )
            ],
            participants=["user", "pipeline_executor"],
        )
        self._decisions.append(bootstrap)
        self._active_decision_id = bootstrap.record_id

        logger.debug(f"Pipeline session started: {self.session_id}")
        return self.session_id

    def record_decision(
        self,
        question: str,
        options: list[str],
        chosen: str,
        rationale: str,
        *,
        phase_name: Optional[str] = None,
        provider: Optional[str] = None,
        confidence: Optional[float] = None,
        sources: Optional[list[SourceReference]] = None,
        caused_by: Optional[str] = None,
    ) -> str:
        """Record a decision made during pipeline execution.

        Args:
            question: What needed deciding?
            options: Options considered
            chosen: Chosen option
            rationale: Why (LLM reasoning goes here)
            phase_name: Pipeline phase name
            provider: LLM provider that made decision
            confidence: Confidence score (0.0-1.0)
            sources: Source references
            caused_by: ID of triggering record

        Returns:
            Decision record ID
        """
        # Build sources
        if sources is None:
            sources = []
            if self._user_query:
                sources.append(SourceReference(
                    kind=SourceKind.EXTERNAL,
                    location=f"user_query:{self.session_id}",
                    snapshot_hash=self._hash_content(self._user_query),
                ))

        # Add confidence to rationale if provided
        full_rationale = rationale
        if confidence is not None:
            full_rationale = f"[Confidence: {confidence:.0%}] {rationale}"

        # Build participants
        participants = []
        if provider:
            participants.append(provider)

        decision = DecisionRecord.create(
            session_id=self.session_id,
            logical_step=self._next_step(),
            question=question,
            options_considered=options,
            chosen_option=chosen,
            rationale=full_rationale,
            sources=sources,
            boundary_constraints={"phase": phase_name} if phase_name else {},
            participants=participants,
            caused_by=caused_by or self._active_decision_id,
        )
        self._decisions.append(decision)
        self._active_decision_id = decision.record_id

        logger.debug(f"Decision recorded: {question[:50]} -> {chosen}")
        return decision.record_id

    def record_execution(
        self,
        tool_name: str,
        input_args: dict[str, Any],
        exit_code: int,
        duration_ms: int,
        output_summary: str,
        *,
        outcome: Optional[ExecutionOutcome] = None,
        decision_id: Optional[str] = None,
    ) -> str:
        """Record a tool/phase execution.

        Args:
            tool_name: Name of tool or phase executed
            input_args: Input arguments
            exit_code: Exit code (0 = success)
            duration_ms: Duration in milliseconds
            output_summary: Summary of output
            outcome: Explicit outcome (auto-derived if not provided)
            decision_id: Link to decision (uses active if not provided)

        Returns:
            Execution record ID
        """
        # Auto-derive outcome
        if outcome is None:
            outcome = ExecutionOutcome.from_exit_code(exit_code)

        # Use active decision if not specified
        if decision_id is None:
            decision_id = self._active_decision_id
            if decision_id is None:
                # Create implicit decision
                decision_id = self.record_decision(
                    question=f"Execute {tool_name}?",
                    options=["execute", "skip"],
                    chosen="execute",
                    rationale="Required by pipeline flow",
                )

        execution = ExecutionRecord.create(
            event_id=uuid.uuid4().hex[:8],
            session_id=self.session_id,
            logical_step=self._next_step(),
            decision_id=decision_id,
            tool_name=tool_name,
            input_args=input_args,
            exit_code=exit_code,
            duration_ms=duration_ms,
            output_summary=output_summary,
            outcome=outcome,
        )
        self._executions.append(execution)

        logger.debug(f"Execution recorded: {tool_name} -> {outcome.value}")
        return execution.record_id

    def end_session(
        self,
        success: bool,
        final_output: Optional[str] = None,
    ) -> None:
        """End the pipeline session.

        Records final decision about session outcome.

        Args:
            success: Whether pipeline succeeded
            final_output: Final output summary
        """
        outcome = "completed" if success else "failed"
        self.record_decision(
            question="Pipeline execution outcome",
            options=["completed", "failed", "needs_input"],
            chosen=outcome,
            rationale=final_output or f"Pipeline {outcome}",
        )
        logger.debug(f"Pipeline session ended: {self.session_id} -> {outcome}")

    def save_to_directory(self, directory: str | Path) -> Path:
        """Save all records to a directory as JSON.

        Args:
            directory: Target directory

        Returns:
            Path to saved session file
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self._template_name or 'pipeline'}_{timestamp}_{self.session_id}.json"
        filepath = directory / filename

        data = {
            "session_id": self.session_id,
            "template_name": self._template_name,
            "user_query": self._user_query,
            "started_at": self._started_at.isoformat() if self._started_at else None,
            "ended_at": datetime.now().isoformat(),
            "decisions": [d.to_dict() for d in self._decisions],
            "executions": [e.to_dict() for e in self._executions],
            "summary": {
                "total_decisions": len(self._decisions),
                "total_executions": len(self._executions),
                "success_count": sum(
                    1 for e in self._executions
                    if e.outcome == ExecutionOutcome.SUCCESS
                ),
                "failure_count": sum(
                    1 for e in self._executions
                    if e.outcome in (ExecutionOutcome.FAILURE, ExecutionOutcome.ERROR)
                ),
            },
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Chronicle records saved: {filepath}")
        return filepath

    def to_markdown(self) -> str:
        """Export session as readable markdown.

        Returns:
            Markdown string
        """
        lines = [
            f"# Pipeline Execution Chronicle",
            f"",
            f"**Session ID:** `{self.session_id}`",
            f"**Template:** {self._template_name}",
            f"**Started:** {self._started_at.isoformat() if self._started_at else 'N/A'}",
            f"",
            f"## User Query",
            f"",
            f"> {self._user_query}",
            f"",
            f"---",
            f"",
            f"## Decisions ({len(self._decisions)})",
            f"",
        ]

        for i, dec in enumerate(self._decisions, 1):
            lines.extend([
                f"### {i}. {dec.question[:80]}",
                f"",
                f"**Chosen:** {dec.chosen_option}",
                f"",
                f"**Options:** {', '.join(dec.options_considered)}",
                f"",
                f"**Rationale:**",
                f"{dec.rationale}",
                f"",
            ])

        lines.extend([
            f"---",
            f"",
            f"## Executions ({len(self._executions)})",
            f"",
        ])

        for i, exe in enumerate(self._executions, 1):
            status = "✓" if exe.outcome == ExecutionOutcome.SUCCESS else "✗"
            lines.extend([
                f"### {i}. {status} {exe.tool_name}",
                f"",
                f"**Outcome:** {exe.outcome.value} | **Duration:** {exe.duration_ms}ms",
                f"",
                f"**Summary:** {exe.output_summary}",
                f"",
            ])

        lines.extend([
            f"---",
            f"",
            f"*Generated by Multivers SDK Pipeline Executor*",
        ])

        return "\n".join(lines)

    def _next_step(self) -> int:
        """Get and increment logical step counter."""
        self.logical_step += 1
        return self.logical_step

    def _hash_content(self, content: str) -> str:
        """Generate content hash for snapshot."""
        import hashlib
        return hashlib.sha256(content.encode()).hexdigest()[:16]


__all__ = ["PipelineChronicleAdapter"]
