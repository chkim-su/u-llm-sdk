"""ClarityGate - Worker self-assessment before task execution.

ClarityGate is the first line of defense against executing unclear tasks.
When a worker (typically Claude) receives a task, it should first assess
whether the task is clear enough to execute autonomously.

The assessment evaluates:
1. Objective clarity - Is the goal well-defined?
2. Scope clarity - Are boundaries clear?
3. Knowledge sufficiency - Do I have enough context?
4. Constraint completeness - Are constraints explicit?
5. Context adequacy - Is there enough background?

Based on the assessment, the gate recommends:
- "execute": Task is clear, proceed with autonomous execution
- "clarify": Some aspects need clarification, but can work on it
- "escalate": Task is too ambiguous, must query orchestrator
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Optional, Protocol

from u_llm_sdk.types import (
    ClarityAssessment,
    ClarityLevel,
    LLMResult,
    Task,
    UnclearAspect,
)

from .utils import extract_json

if TYPE_CHECKING:
    from u_llm_sdk.llm.providers.base import BaseProvider

logger = logging.getLogger(__name__)


# Thresholds for clarity assessment
CLARITY_THRESHOLDS = {
    "clear": 0.8,      # >= 0.8 = CLEAR
    "ambiguous": 0.4,  # < 0.4 = AMBIGUOUS, else NEEDS_CLARIFICATION
}

# System prompt for clarity assessment
CLARITY_ASSESSMENT_PROMPT = """You are a task clarity assessor. Your job is to evaluate whether a given task is clear enough to execute.

Evaluate the task on these 5 dimensions (each 0.0 - 1.0):
1. objective_clarity: Is the goal well-defined?
2. scope_clarity: Are the boundaries of the task clear?
3. knowledge_sufficiency: Do you have enough information to proceed?
4. constraint_completeness: Are constraints and requirements explicit?
5. context_adequacy: Is there enough background context?

For any dimension scoring below 0.7, identify the specific unclear aspect.

Respond in this exact JSON format:
{
  "scores": {
    "objective_clarity": 0.0-1.0,
    "scope_clarity": 0.0-1.0,
    "knowledge_sufficiency": 0.0-1.0,
    "constraint_completeness": 0.0-1.0,
    "context_adequacy": 0.0-1.0
  },
  "unclear_aspects": [
    {
      "aspect_type": "knowledge_gap|scope_ambiguity|objective_ambiguity|constraint_missing|context_insufficient",
      "description": "What is unclear",
      "clarification_needed": "What clarification would help"
    }
  ],
  "self_questions": ["Questions you would ask to clarify"],
  "overall_assessment": "Brief summary of clarity status"
}"""


class LLMProvider(Protocol):
    """Protocol for LLM provider used by ClarityGate."""

    async def run(
        self,
        prompt: str,
        *,
        session_id: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> LLMResult: ...


class ClarityGate:
    """Assesses task clarity before execution.

    ClarityGate evaluates whether a task is clear enough for autonomous
    execution. It uses an LLM to assess 5 dimensions of clarity and
    returns a structured assessment with recommendations.

    Example:
        >>> gate = ClarityGate(claude_provider)
        >>> task = Task(
        ...     task_id="t1",
        ...     objective="Implement user authentication",
        ...     context="FastAPI backend project",
        ... )
        >>> assessment = await gate.assess(task)
        >>> if assessment.level == ClarityLevel.CLEAR:
        ...     # Proceed with execution
        ... elif assessment.recommendation == "escalate":
        ...     # Query orchestrator for clarification

    Attributes:
        provider: LLM provider for running assessments
        clear_threshold: Score threshold for CLEAR level (default: 0.8)
        ambiguous_threshold: Score threshold for AMBIGUOUS level (default: 0.4)
    """

    def __init__(
        self,
        provider: LLMProvider,
        *,
        clear_threshold: float = CLARITY_THRESHOLDS["clear"],
        ambiguous_threshold: float = CLARITY_THRESHOLDS["ambiguous"],
    ):
        """Initialize ClarityGate.

        Args:
            provider: LLM provider for running assessments
            clear_threshold: Score >= this is CLEAR (default: 0.8)
            ambiguous_threshold: Score < this is AMBIGUOUS (default: 0.4)
        """
        self.provider = provider
        self.clear_threshold = clear_threshold
        self.ambiguous_threshold = ambiguous_threshold

    async def assess(self, task: Task) -> ClarityAssessment:
        """Assess task clarity.

        Evaluates the task using an LLM and returns a structured
        assessment with clarity level, score, and recommendations.

        Args:
            task: The task to assess

        Returns:
            ClarityAssessment with level, score, and recommendation
        """
        prompt = self._build_assessment_prompt(task)

        try:
            result = await self.provider.run(prompt)
            return self._parse_assessment(result.text, task)
        except Exception as e:
            logger.warning(f"Clarity assessment failed: {e}, defaulting to NEEDS_CLARIFICATION")
            return self._default_assessment(str(e))

    def _build_assessment_prompt(self, task: Task) -> str:
        """Build the assessment prompt for the LLM."""
        task_description = f"""
Task ID: {task.task_id}
Objective: {task.objective}
Context: {task.context}
Constraints: {', '.join(task.constraints) if task.constraints else 'None specified'}
Expected Output: {task.expected_output or 'Not specified'}
Source: {task.source or 'Unknown'}
"""
        return f"{CLARITY_ASSESSMENT_PROMPT}\n\n---\n\nTask to assess:\n{task_description}"

    def _parse_assessment(self, response_text: str, task: Task) -> ClarityAssessment:
        """Parse LLM response into ClarityAssessment."""
        try:
            # Try to extract JSON from response
            json_str = extract_json(response_text)
            data = json.loads(json_str)

            # Calculate overall score
            scores = data.get("scores", {})
            overall_score = self._calculate_overall_score(scores)

            # Parse unclear aspects
            unclear_aspects = [
                UnclearAspect(
                    aspect_type=asp.get("aspect_type", "context_insufficient"),
                    description=asp.get("description", ""),
                    clarification_needed=asp.get("clarification_needed", ""),
                )
                for asp in data.get("unclear_aspects", [])
            ]

            # Determine clarity level and recommendation
            level = self._determine_level(overall_score)
            recommendation = self._determine_recommendation(level, unclear_aspects)

            return ClarityAssessment(
                level=level,
                score=overall_score,
                unclear_aspects=unclear_aspects,
                self_questions=data.get("self_questions", []),
                recommendation=recommendation,
            )

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse clarity assessment: {e}")
            return self._default_assessment(f"Parse error: {e}")

    def _calculate_overall_score(self, scores: dict) -> float:
        """Calculate overall clarity score from dimension scores.

        Uses weighted average where objective and scope are weighted higher.
        """
        weights = {
            "objective_clarity": 0.25,
            "scope_clarity": 0.25,
            "knowledge_sufficiency": 0.20,
            "constraint_completeness": 0.15,
            "context_adequacy": 0.15,
        }

        total_weight = 0.0
        weighted_sum = 0.0

        for dim, weight in weights.items():
            if dim in scores:
                weighted_sum += scores[dim] * weight
                total_weight += weight

        if total_weight == 0:
            return 0.5  # Default to middle

        return weighted_sum / total_weight

    def _determine_level(self, score: float) -> ClarityLevel:
        """Determine clarity level from score."""
        if score >= self.clear_threshold:
            return ClarityLevel.CLEAR
        elif score < self.ambiguous_threshold:
            return ClarityLevel.AMBIGUOUS
        else:
            return ClarityLevel.NEEDS_CLARIFICATION

    def _determine_recommendation(
        self,
        level: ClarityLevel,
        unclear_aspects: list[UnclearAspect],
    ) -> str:
        """Determine recommendation based on level and aspects."""
        if level == ClarityLevel.CLEAR:
            return "execute"
        elif level == ClarityLevel.AMBIGUOUS:
            return "escalate"
        else:
            # NEEDS_CLARIFICATION: check if aspects are workable
            critical_types = {"objective_ambiguity", "scope_ambiguity"}
            has_critical = any(
                asp.aspect_type in critical_types for asp in unclear_aspects
            )
            return "escalate" if has_critical else "clarify"

    def _default_assessment(self, error_msg: str) -> ClarityAssessment:
        """Return default assessment when parsing fails."""
        return ClarityAssessment(
            level=ClarityLevel.NEEDS_CLARIFICATION,
            score=0.5,
            unclear_aspects=[
                UnclearAspect(
                    aspect_type="context_insufficient",
                    description=f"Assessment failed: {error_msg}",
                    clarification_needed="Manual review recommended",
                )
            ],
            self_questions=["Could not automatically assess clarity"],
            recommendation="clarify",
        )


# Convenience function for quick assessment
async def assess_task_clarity(
    task: Task,
    provider: LLMProvider,
    *,
    clear_threshold: float = CLARITY_THRESHOLDS["clear"],
) -> ClarityAssessment:
    """Quick function to assess task clarity.

    Args:
        task: Task to assess
        provider: LLM provider
        clear_threshold: Threshold for CLEAR level

    Returns:
        ClarityAssessment with recommendation
    """
    gate = ClarityGate(provider, clear_threshold=clear_threshold)
    return await gate.assess(task)
