"""Pipeline execution report generator.

Generates human-readable markdown reports for completed pipeline executions.
This is separate from Chronicle (which is for learning) - reports are for users.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .types import PhaseOutput, PhaseResult, PipelineResult

logger = logging.getLogger(__name__)


@dataclass
class PhaseReportEntry:
    """Report entry for a single phase."""

    name: str
    phase_type: str
    provider: Optional[str]
    result: PhaseResult
    duration_ms: int = 0
    cost_usd: float = 0.0
    summary: str = ""
    error: Optional[str] = None
    user_interaction: Optional[str] = None  # If user was queried
    key_outputs: list[str] = field(default_factory=list)


@dataclass
class PipelineReport:
    """Complete pipeline execution report."""

    # Metadata
    template_name: str
    request: str
    started_at: datetime
    ended_at: datetime
    cwd: str = ""

    # Summary
    success: bool = False
    total_phases: int = 0
    completed_phases: int = 0
    failed_phases: int = 0
    total_duration_ms: int = 0
    total_cost_usd: float = 0.0

    # Phase details
    phases: list[PhaseReportEntry] = field(default_factory=list)

    # User interactions
    user_queries: list[dict[str, str]] = field(default_factory=list)

    # Final result
    final_output_summary: str = ""
    recommendations: list[str] = field(default_factory=list)


class PipelineReportWriter:
    """Generates and writes pipeline reports.

    Collects data during pipeline execution and generates
    a human-readable markdown report at completion.

    Example:
        >>> writer = PipelineReportWriter(template_name, request, cwd)
        >>> writer.record_phase(phase_output)
        >>> writer.record_user_query("What is X?", "X is Y")
        >>> report_path = writer.finalize(result)
    """

    def __init__(
        self,
        template_name: str,
        request: str,
        cwd: str = "",
        *,
        report_dir: Optional[str] = None,
    ):
        """Initialize report writer.

        Args:
            template_name: Name of the pipeline template
            request: User's original request
            cwd: Working directory
            report_dir: Directory for reports (default: .atlas/reports)
        """
        self.template_name = template_name
        self.request = request
        self.cwd = cwd
        self.report_dir = Path(report_dir) if report_dir else Path(cwd or ".") / ".atlas" / "reports"

        self.started_at = datetime.now()
        self.phases: list[PhaseReportEntry] = []
        self.user_queries: list[dict[str, str]] = []

    def record_phase(
        self,
        output: PhaseOutput,
        *,
        phase_type: str = "",
        provider: Optional[str] = None,
        key_outputs: Optional[list[str]] = None,
    ) -> None:
        """Record a completed phase.

        Args:
            output: Phase output
            phase_type: Type of phase (brainstorm, review, etc.)
            provider: Provider used
            key_outputs: Key outputs to highlight
        """
        # Extract summary from output
        summary = ""
        if output.output:
            if isinstance(output.output, str):
                summary = output.output[:500]
            elif isinstance(output.output, dict):
                summary = str(output.output)[:500]

        entry = PhaseReportEntry(
            name=output.phase_name,
            phase_type=phase_type,
            provider=provider,
            result=output.result,
            duration_ms=output.duration_ms,
            cost_usd=output.cost_usd,
            summary=summary,
            error=output.error,
            key_outputs=key_outputs or [],
        )

        self.phases.append(entry)

    def record_user_query(self, question: str, answer: str) -> None:
        """Record a user interaction.

        Args:
            question: Question asked to user
            answer: User's answer
        """
        self.user_queries.append({
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().isoformat(),
        })

        # Also update the last phase if it was waiting for input
        if self.phases and self.phases[-1].result == PhaseResult.NEEDS_INPUT:
            self.phases[-1].user_interaction = f"Q: {question}\nA: {answer}"

    def finalize(
        self,
        result: PipelineResult,
        *,
        recommendations: Optional[list[str]] = None,
    ) -> Optional[Path]:
        """Generate and write the final report.

        Args:
            result: Final pipeline result
            recommendations: Optional recommendations to include

        Returns:
            Path to generated report file, or None on failure
        """
        ended_at = datetime.now()

        # Build report data
        report = PipelineReport(
            template_name=self.template_name,
            request=self.request,
            started_at=self.started_at,
            ended_at=ended_at,
            cwd=self.cwd,
            success=result.success,
            total_phases=len(self.phases),
            completed_phases=sum(1 for p in self.phases if p.result == PhaseResult.SUCCESS),
            failed_phases=sum(1 for p in self.phases if p.result == PhaseResult.FAILED),
            total_duration_ms=result.total_duration_ms,
            total_cost_usd=result.total_cost_usd,
            phases=self.phases,
            user_queries=self.user_queries,
            final_output_summary=self._summarize_output(result.final_output),
            recommendations=recommendations or [],
        )

        # Generate markdown
        markdown = self._generate_markdown(report)

        # Write to file
        try:
            self.report_dir.mkdir(parents=True, exist_ok=True)

            # Filename: atlas-{template}-{timestamp}.md
            timestamp = ended_at.strftime("%Y%m%d-%H%M%S")
            filename = f"atlas-{self.template_name}-{timestamp}.md"
            report_path = self.report_dir / filename

            report_path.write_text(markdown, encoding="utf-8")
            logger.info(f"Pipeline report saved: {report_path}")

            return report_path

        except Exception as e:
            logger.error(f"Failed to write report: {e}")
            return None

    def _summarize_output(self, output: Any) -> str:
        """Summarize the final output."""
        if output is None:
            return "(No output)"

        if isinstance(output, str):
            return output[:1000]

        if isinstance(output, dict):
            # Extract key fields
            if "summary" in output:
                return str(output["summary"])[:1000]
            if "result" in output:
                return str(output["result"])[:1000]

        return str(output)[:1000]

    def _generate_markdown(self, report: PipelineReport) -> str:
        """Generate markdown report content."""
        lines = []

        # Header
        lines.append(f"# Atlas Pipeline Report: {report.template_name}")
        lines.append("")
        lines.append(f"**Generated:** {report.ended_at.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Status banner
        status_icon = "✅" if report.success else "❌"
        lines.append(f"## {status_icon} Status: {'SUCCESS' if report.success else 'FAILED'}")
        lines.append("")

        # Summary table
        lines.append("## Summary")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Request | {report.request[:100]}{'...' if len(report.request) > 100 else ''} |")
        lines.append(f"| Template | `{report.template_name}` |")
        lines.append(f"| Duration | {report.total_duration_ms / 1000:.1f}s |")
        lines.append(f"| Cost | ${report.total_cost_usd:.4f} |")
        lines.append(f"| Phases | {report.completed_phases}/{report.total_phases} completed |")
        if report.cwd:
            lines.append(f"| Working Dir | `{report.cwd}` |")
        lines.append("")

        # Phase timeline
        lines.append("## Phase Execution")
        lines.append("")

        for i, phase in enumerate(report.phases, 1):
            # Phase header with status icon
            if phase.result == PhaseResult.SUCCESS:
                icon = "✅"
            elif phase.result == PhaseResult.FAILED:
                icon = "❌"
            elif phase.result == PhaseResult.NEEDS_INPUT:
                icon = "⏸️"
            elif phase.result == PhaseResult.SKIPPED:
                icon = "⏭️"
            else:
                icon = "⚪"

            lines.append(f"### {i}. {icon} {phase.name}")
            lines.append("")

            # Phase info
            phase_info = []
            if phase.phase_type:
                phase_info.append(f"Type: `{phase.phase_type}`")
            if phase.provider:
                phase_info.append(f"Provider: `{phase.provider}`")
            phase_info.append(f"Duration: {phase.duration_ms}ms")
            if phase.cost_usd > 0:
                phase_info.append(f"Cost: ${phase.cost_usd:.4f}")

            lines.append(" | ".join(phase_info))
            lines.append("")

            # Summary
            if phase.summary:
                lines.append("**Output:**")
                lines.append("```")
                lines.append(phase.summary[:500])
                lines.append("```")
                lines.append("")

            # Error
            if phase.error:
                lines.append(f"**Error:** {phase.error}")
                lines.append("")

            # User interaction
            if phase.user_interaction:
                lines.append("**User Interaction:**")
                lines.append("```")
                lines.append(phase.user_interaction)
                lines.append("```")
                lines.append("")

            # Key outputs
            if phase.key_outputs:
                lines.append("**Key Outputs:**")
                for output in phase.key_outputs:
                    lines.append(f"- {output}")
                lines.append("")

        # User queries section (if any)
        if report.user_queries:
            lines.append("## User Interactions")
            lines.append("")
            for i, query in enumerate(report.user_queries, 1):
                lines.append(f"### Query {i}")
                lines.append(f"**Q:** {query['question']}")
                lines.append(f"**A:** {query['answer']}")
                lines.append("")

        # Final output
        if report.final_output_summary:
            lines.append("## Final Output")
            lines.append("")
            lines.append("```")
            lines.append(report.final_output_summary)
            lines.append("```")
            lines.append("")

        # Recommendations
        if report.recommendations:
            lines.append("## Recommendations")
            lines.append("")
            for rec in report.recommendations:
                lines.append(f"- {rec}")
            lines.append("")

        # Footer
        lines.append("---")
        lines.append(f"*Report generated by Atlas Pipeline v2.0*")
        lines.append(f"*Execution: {report.started_at.strftime('%H:%M:%S')} → {report.ended_at.strftime('%H:%M:%S')}*")

        return "\n".join(lines)


__all__ = [
    "PhaseReportEntry",
    "PipelineReport",
    "PipelineReportWriter",
]
