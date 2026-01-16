"""Evidence Gate for Auto-Promotion to DETAILS/DEEP_DIVE.

This module provides the EvidenceGate that automatically escalates
context retrieval from INDEX to DETAILS/DEEP_DIVE based on task type.

Architecture:
    The gate operates at TWO levels:
    1. Prompt-level: Injects context BEFORE Claude starts thinking
    2. Retry-level: On execution failure, escalates to deeper context

Design Philosophy:
    - "No speculation without evidence" - file modifications require context
    - Progressive disclosure: INDEX → DETAILS → DEEP_DIVE
    - Graceful fallback: If context unavailable, warn but don't block

Usage:
    >>> from u_llm_sdk.pipeline import EvidenceGate
    >>>
    >>> gate = EvidenceGate(rag_client)
    >>> context = await gate.get_context_for_task(
    ...     task_type="new_feature",
    ...     query="implement authentication",
    ...     cwd="/project",
    ... )
    >>>
    >>> # On failure, escalate
    >>> context = await gate.escalate_on_failure(
    ...     error="TypeError: missing attribute",
    ...     current_stage="DETAILS",
    ...     cwd="/project",
    ... )
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..rag_client import RAGClient

logger = logging.getLogger(__name__)


class EvidenceStage(str, Enum):
    """Evidence disclosure stages."""

    NONE = "NONE"      # No context needed (read-only operations)
    INDEX = "INDEX"    # High-level overview (~800 tokens)
    DETAILS = "DETAILS"  # Expanded context (~2500 tokens)
    DEEP_DIVE = "DEEP_DIVE"  # Full source excerpts (~4500 tokens)


@dataclass
class StagePolicy:
    """Policy for determining evidence stage based on task type.

    Attributes:
        initial_stage: Starting stage for this task type
        on_failure_stage: Stage to escalate to on execution failure
        require_evidence: Whether to block if no evidence available
        min_chunks: Minimum evidence chunks required (0 = no minimum)
    """

    initial_stage: EvidenceStage
    on_failure_stage: EvidenceStage
    require_evidence: bool = True
    min_chunks: int = 0


# Task type → Stage policy mapping
TASK_POLICIES: dict[str, StagePolicy] = {
    # New feature: Start with DETAILS (need to understand existing patterns)
    "new_feature": StagePolicy(
        initial_stage=EvidenceStage.DETAILS,
        on_failure_stage=EvidenceStage.DEEP_DIVE,
        require_evidence=True,
        min_chunks=3,
    ),

    # Bug fix: Start with DEEP_DIVE (need full context around the bug)
    "bug_fix": StagePolicy(
        initial_stage=EvidenceStage.DEEP_DIVE,
        on_failure_stage=EvidenceStage.DEEP_DIVE,
        require_evidence=True,
        min_chunks=5,
    ),

    # Refactoring: Start with DETAILS (need to understand structure)
    "refactoring": StagePolicy(
        initial_stage=EvidenceStage.DETAILS,
        on_failure_stage=EvidenceStage.DEEP_DIVE,
        require_evidence=True,
        min_chunks=3,
    ),

    # Integration: Start with INDEX (breadth over depth)
    "integration": StagePolicy(
        initial_stage=EvidenceStage.INDEX,
        on_failure_stage=EvidenceStage.DETAILS,
        require_evidence=False,
        min_chunks=0,
    ),

    # Project creation: No evidence needed (greenfield)
    "project_creation": StagePolicy(
        initial_stage=EvidenceStage.NONE,
        on_failure_stage=EvidenceStage.NONE,
        require_evidence=False,
        min_chunks=0,
    ),

    # Default: INDEX with optional evidence
    "unknown": StagePolicy(
        initial_stage=EvidenceStage.INDEX,
        on_failure_stage=EvidenceStage.DETAILS,
        require_evidence=False,
        min_chunks=0,
    ),
}


@dataclass
class EvidenceResult:
    """Result of evidence retrieval.

    Attributes:
        context_text: Formatted context for prompt injection
        stage: Stage that was retrieved
        chunk_count: Number of evidence chunks retrieved
        seed_ids: IDs of evidence chunks (for escalation)
        sufficient: Whether evidence meets policy requirements
        warning: Warning message if evidence is insufficient
    """

    context_text: Optional[str]
    stage: EvidenceStage
    chunk_count: int = 0
    seed_ids: list[str] = field(default_factory=list)
    sufficient: bool = True
    warning: Optional[str] = None

    @classmethod
    def empty(cls, stage: EvidenceStage, warning: str) -> "EvidenceResult":
        """Create an empty result with warning."""
        return cls(
            context_text=None,
            stage=stage,
            chunk_count=0,
            seed_ids=[],
            sufficient=False,
            warning=warning,
        )


class EvidenceGate:
    """Gate for evidence-based context injection.

    This gate ensures that LLM actions have sufficient codebase context
    before proceeding. It implements progressive disclosure:

    1. Initial context based on task type policy
    2. Escalation on failure (retry with deeper context)
    3. Minimum evidence requirements enforcement
    """

    def __init__(
        self,
        rag_client: Optional["RAGClient"],
        *,
        policies: Optional[dict[str, StagePolicy]] = None,
        enforce_requirements: bool = True,
    ):
        """Initialize evidence gate.

        Args:
            rag_client: RAGClient for MV-rag communication (None = disabled)
            policies: Custom task type → policy mapping
            enforce_requirements: If True, raise on insufficient evidence
        """
        self.rag_client = rag_client
        self.policies = {**TASK_POLICIES, **(policies or {})}
        self.enforce_requirements = enforce_requirements

        # Track escalation state per session
        self._current_stage: dict[str, EvidenceStage] = {}
        self._seed_ids: dict[str, list[str]] = {}

    def get_policy(self, task_type: str) -> StagePolicy:
        """Get policy for task type."""
        return self.policies.get(task_type, self.policies["unknown"])

    async def get_context_for_task(
        self,
        task_type: str,
        query: str,
        cwd: str,
        *,
        session_id: Optional[str] = None,
        override_stage: Optional[EvidenceStage] = None,
    ) -> EvidenceResult:
        """Get initial context for a task.

        Args:
            task_type: Type of task (new_feature, bug_fix, etc.)
            query: Task description/objective
            cwd: Working directory
            session_id: Session ID for state tracking
            override_stage: Force specific stage (ignores policy)

        Returns:
            EvidenceResult with context and metadata
        """
        policy = self.get_policy(task_type)
        stage = override_stage or policy.initial_stage

        # Track state
        if session_id:
            self._current_stage[session_id] = stage

        if stage == EvidenceStage.NONE:
            return EvidenceResult(
                context_text=None,
                stage=stage,
                sufficient=True,
            )

        # Fetch evidence
        return await self._fetch_evidence(
            stage=stage,
            query=query,
            cwd=cwd,
            policy=policy,
            session_id=session_id,
        )

    async def escalate_on_failure(
        self,
        error: str,
        cwd: str,
        *,
        session_id: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> EvidenceResult:
        """Escalate to deeper context after execution failure.

        This is called when Claude's execution fails. It:
        1. Determines next escalation stage
        2. Fetches deeper context using previous seed_ids
        3. Includes error context in the query

        Args:
            error: Error message from failed execution
            cwd: Working directory
            session_id: Session ID for state tracking
            task_type: Task type for policy lookup

        Returns:
            EvidenceResult with escalated context
        """
        # Get current state
        current = self._current_stage.get(session_id or "", EvidenceStage.INDEX)
        seeds = self._seed_ids.get(session_id or "", [])

        # Determine next stage
        policy = self.get_policy(task_type or "unknown")
        next_stage = policy.on_failure_stage

        # Can't escalate beyond DEEP_DIVE
        if current == EvidenceStage.DEEP_DIVE:
            logger.warning("Already at DEEP_DIVE, cannot escalate further")
            next_stage = EvidenceStage.DEEP_DIVE

        # Build error-focused query
        error_query = self._extract_error_symbols(error)
        query = f"Debug error: {error_query}" if error_query else "Debug execution failure"

        logger.info(
            f"Escalating from {current.value} to {next_stage.value} "
            f"after failure: {error[:100]}"
        )

        # Fetch deeper context
        return await self._fetch_evidence(
            stage=next_stage,
            query=query,
            cwd=cwd,
            policy=policy,
            session_id=session_id,
            seed_ids=seeds,
        )

    async def _fetch_evidence(
        self,
        stage: EvidenceStage,
        query: str,
        cwd: str,
        policy: StagePolicy,
        session_id: Optional[str] = None,
        seed_ids: Optional[list[str]] = None,
    ) -> EvidenceResult:
        """Fetch evidence from RAGClient."""
        if self.rag_client is None:
            warning = "RAGClient not configured, no evidence available"
            logger.debug(warning)
            return EvidenceResult.empty(stage, warning)

        try:
            context = await self.rag_client.get_codebase_context(
                cwd=cwd,
                query=query,
                stage=stage.value,
                seed_ids=seed_ids,
            )

            if not context:
                warning = f"No evidence returned for stage {stage.value}"
                if policy.require_evidence and self.enforce_requirements:
                    logger.warning(warning)
                    return EvidenceResult.empty(stage, warning)
                return EvidenceResult(
                    context_text=None,
                    stage=stage,
                    sufficient=not policy.require_evidence,
                    warning=warning,
                )

            # Parse chunk count from context (look for chunk markers)
            chunk_count = self._count_chunks(context)

            # Check minimum requirement
            if chunk_count < policy.min_chunks:
                warning = (
                    f"Insufficient evidence: {chunk_count} chunks "
                    f"(minimum: {policy.min_chunks})"
                )
                if self.enforce_requirements and policy.require_evidence:
                    logger.warning(warning)
                    return EvidenceResult(
                        context_text=context,
                        stage=stage,
                        chunk_count=chunk_count,
                        sufficient=False,
                        warning=warning,
                    )

            # Update state
            if session_id:
                self._current_stage[session_id] = stage
                # Extract seed IDs from context (for future escalation)
                extracted_ids = self._extract_seed_ids(context)
                self._seed_ids[session_id] = extracted_ids

            logger.info(
                f"Evidence retrieved: stage={stage.value}, "
                f"chunks={chunk_count}, query={query[:50]}"
            )

            return EvidenceResult(
                context_text=context,
                stage=stage,
                chunk_count=chunk_count,
                seed_ids=self._seed_ids.get(session_id or "", []),
                sufficient=True,
            )

        except Exception as e:
            warning = f"Evidence fetch failed: {e}"
            logger.error(warning, exc_info=True)
            return EvidenceResult.empty(stage, warning)

    def _count_chunks(self, context: str) -> int:
        """Count evidence chunks in context text."""
        # Look for chunk markers like "- **" or numbered items
        markers = re.findall(r'^-\s+\*\*', context, re.MULTILINE)
        return len(markers) if markers else 1  # At least 1 if context exists

    def _extract_seed_ids(self, context: str) -> list[str]:
        """Extract seed IDs from context for escalation.

        Looks for patterns like [scip:symbol:...] or file paths.
        """
        # Match evidence IDs like scip:symbol:xxx or forensics:event:xxx
        pattern = r'\[([a-z]+:[a-z_]+:[^\]]+)\]'
        matches = re.findall(pattern, context)
        return list(set(matches))[:10]  # Limit to 10 seeds

    def _extract_error_symbols(self, error: str) -> str:
        """Extract relevant symbols from error message.

        Looks for:
        - Python class/function names
        - File paths
        - Line numbers
        """
        symbols = []

        # Match class/function names (CamelCase or snake_case)
        names = re.findall(r'\b([A-Z][a-zA-Z0-9]+|[a-z_][a-z0-9_]+)\b', error)
        symbols.extend(n for n in names if len(n) > 3 and n not in (
            "Error", "Exception", "None", "True", "False", "self"
        ))

        # Match file paths
        paths = re.findall(r'(?:File\s+)?["\']?([/\w.-]+\.py)', error)
        symbols.extend(paths)

        # Deduplicate and limit
        unique = list(dict.fromkeys(symbols))[:5]
        return " ".join(unique)

    def reset_session(self, session_id: str) -> None:
        """Reset escalation state for a session."""
        self._current_stage.pop(session_id, None)
        self._seed_ids.pop(session_id, None)


def get_delegation_evidence_context(
    evidence: EvidenceResult,
    task_type: str,
) -> str:
    """Format evidence for delegation prompt injection.

    This builds the context section to inject into the delegation prompt.

    Args:
        evidence: Evidence result from gate
        task_type: Task type for tailored instructions

    Returns:
        Formatted context string for prompt injection
    """
    if not evidence.context_text:
        if evidence.warning:
            return f"⚠️ {evidence.warning}\nProceeding with limited codebase context."
        return ""

    sections = []

    # Header based on stage
    stage_headers = {
        EvidenceStage.INDEX: "## Codebase Overview (INDEX)",
        EvidenceStage.DETAILS: "## Relevant Code Context (DETAILS)",
        EvidenceStage.DEEP_DIVE: "## Detailed Source Context (DEEP_DIVE)",
    }
    sections.append(stage_headers.get(evidence.stage, "## Codebase Context"))
    sections.append("")

    # Add evidence content
    sections.append(evidence.context_text)
    sections.append("")

    # Add task-specific guidance
    if task_type == "new_feature":
        sections.append("**Note**: Review existing patterns above before implementing.")
    elif task_type == "bug_fix":
        sections.append("**Note**: Trace the error path through the code above.")
    elif task_type == "refactoring":
        sections.append("**Note**: Understand the current structure before modifying.")

    # Warning if insufficient
    if not evidence.sufficient and evidence.warning:
        sections.append("")
        sections.append(f"⚠️ Warning: {evidence.warning}")

    return "\n".join(sections)


__all__ = [
    "EvidenceGate",
    "EvidenceStage",
    "EvidenceResult",
    "StagePolicy",
    "TASK_POLICIES",
    "get_delegation_evidence_context",
]
