"""ConsensusLoop - 3-round majority voting for complex decisions.

When multiple LLMs need to reach a decision, ConsensusLoop facilitates
a structured voting process with up to 3 rounds of discussion.

Key Features:
- Maximum 3 rounds to prevent endless discussion
- 2/3 majority (0.67) threshold for consensus
- Low agreement (<0.4) triggers user escalation
- Full discussion records preserved (no summarization)

The loop evaluates agreement by analyzing:
- Position similarity across providers
- Strength of supporting arguments
- Presence of blocking concerns
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Optional, Protocol

from u_llm_sdk.types import (
    BrainstormConfig,
    ConsensusEvaluation,
    ConsensusResult,
    DiscussionEntry,
    DissentingView,
    ParticipantInput,
    Provider,
    LLMResult,
)

from .utils import extract_json

if TYPE_CHECKING:
    from u_llm_sdk.llm.providers.base import BaseProvider

logger = logging.getLogger(__name__)


# Default configuration
DEFAULT_MAX_ROUNDS = 3
DEFAULT_CONSENSUS_THRESHOLD = 0.67  # 2/3 majority
DEFAULT_LOW_AGREEMENT_THRESHOLD = 0.40


class LLMProvider(Protocol):
    """Protocol for LLM provider used in consensus."""

    async def run(
        self,
        prompt: str,
        *,
        session_id: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> LLMResult: ...


# System prompt for gathering opinions
OPINION_PROMPT = """You are participating in a multi-LLM consensus process.

Topic for discussion:
{topic}

{previous_context}

Please provide your opinion in this JSON format:
{{
  "position": "Your stance (support/oppose/neutral with specific approach)",
  "analysis": "Your detailed analysis of the topic",
  "supporting_evidence": ["Evidence point 1", "Evidence point 2"],
  "concerns": ["Concern 1 if any", "Concern 2 if any"],
  "proposed_approach": "Your recommended approach",
  "vote": "support" or "oppose" or "abstain"
}}"""


class ConsensusLoop:
    """Runs a consensus-building loop among multiple LLM providers.

    ConsensusLoop facilitates structured decision-making by running
    up to 3 rounds of opinion gathering and evaluation. It calculates
    agreement levels and determines whether to proceed, continue
    discussion, or escalate to the user.

    Example:
        >>> providers = {
        ...     Provider.GEMINI: gemini_provider,
        ...     Provider.CLAUDE: claude_provider,
        ...     Provider.CODEX: codex_provider,
        ... }
        >>> loop = ConsensusLoop(providers)
        >>> result = await loop.run("Should we use microservices architecture?")
        >>> if result.success:
        ...     print(f"Decision: {result.final_decision}")
        ... else:
        ...     print(f"Questions for user: {result.user_questions}")

    Attributes:
        providers: Dict mapping Provider enum to provider instances
        config: BrainstormConfig with consensus parameters
    """

    def __init__(
        self,
        providers: dict[Provider, LLMProvider],
        config: Optional[BrainstormConfig] = None,
    ):
        """Initialize ConsensusLoop.

        Args:
            providers: Dict of Provider -> LLMProvider instances
            config: BrainstormConfig with consensus parameters
        """
        self.providers = providers
        self.config = config or BrainstormConfig()

    async def run(
        self,
        topic: str,
        *,
        initial_context: Optional[str] = None,
    ) -> ConsensusResult:
        """Run the consensus loop.

        Executes up to max_rounds of opinion gathering, evaluates
        agreement after each round, and returns the final result.

        Args:
            topic: The topic/question for consensus
            initial_context: Optional context from previous discussion

        Returns:
            ConsensusResult with final decision and full discussion log
        """
        all_entries: list[DiscussionEntry] = []
        previous_context = initial_context or ""
        final_inputs: dict[Provider, ParticipantInput] = {}

        for round_num in range(1, self.config.max_rounds + 1):
            logger.info(f"Consensus round {round_num}/{self.config.max_rounds}")

            # Gather opinions from all providers
            inputs = await self._gather_opinions(topic, previous_context, round_num)
            final_inputs = inputs

            # Log entries for this round
            for provider, inp in inputs.items():
                entry = DiscussionEntry(
                    timestamp=datetime.now(),
                    speaker=provider,
                    message_type="opinion",
                    content=inp.analysis,
                    references=[],
                )
                all_entries.append(entry)

            # Evaluate consensus
            evaluation = self._evaluate_consensus(inputs)

            # Check if we can conclude
            if evaluation.agreement_category == "high":
                return self._finalize_consensus(
                    inputs, evaluation, all_entries, escalated=False
                )

            # Check if we should escalate to user
            if (
                evaluation.agreement_level < self.config.low_agreement_threshold
                and round_num == self.config.max_rounds
            ):
                return self._escalate_to_user(inputs, evaluation, all_entries)

            # Build context for next round
            if round_num < self.config.max_rounds:
                previous_context = self._build_feedback_context(inputs, evaluation)

        # Max rounds reached: use majority or escalate
        final_eval = self._evaluate_consensus(final_inputs)

        if final_eval.agreement_level < self.config.low_agreement_threshold:
            return self._escalate_to_user(final_inputs, final_eval, all_entries)
        else:
            return self._finalize_with_majority(final_inputs, final_eval, all_entries)

    async def _gather_opinions(
        self,
        topic: str,
        previous_context: str,
        round_num: int,
    ) -> dict[Provider, ParticipantInput]:
        """Gather opinions from all providers."""
        import asyncio
        import json

        context_section = ""
        if previous_context:
            context_section = f"\n\nContext from previous rounds:\n{previous_context}"

        prompt = OPINION_PROMPT.format(
            topic=topic,
            previous_context=context_section,
        )

        async def get_opinion(provider: Provider) -> tuple[Provider, ParticipantInput]:
            try:
                llm = self.providers[provider]
                result = await llm.run(prompt)
                data = self._parse_opinion(result.text)
                return provider, ParticipantInput(
                    provider=provider,
                    analysis=data.get("analysis", result.text),
                    position=data.get("position", "unknown"),
                    supporting_evidence=data.get("supporting_evidence", []),
                    concerns=data.get("concerns", []),
                    proposed_approach=data.get("proposed_approach", ""),
                )
            except Exception as e:
                logger.warning(f"Failed to get opinion from {provider}: {e}")
                return provider, ParticipantInput(
                    provider=provider,
                    analysis=f"Error: {e}",
                    position="abstain",
                    supporting_evidence=[],
                    concerns=[str(e)],
                    proposed_approach="",
                )

        # Gather opinions in parallel
        tasks = [get_opinion(p) for p in self.providers.keys()]
        results = await asyncio.gather(*tasks)

        return dict(results)

    def _parse_opinion(self, text: str) -> dict:
        """Parse opinion from LLM response."""
        import json

        try:
            # Extract JSON
            json_str = extract_json(text)
            return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            return {"analysis": text, "position": "unclear"}

    def _evaluate_consensus(
        self,
        inputs: dict[Provider, ParticipantInput],
    ) -> ConsensusEvaluation:
        """Evaluate the level of consensus among inputs.

        This method analyzes positions to determine agreement level.
        Agreement is calculated by checking position similarity.

        Args:
            inputs: Dict of Provider -> ParticipantInput

        Returns:
            ConsensusEvaluation with agreement level and recommendation
        """
        if not inputs:
            return ConsensusEvaluation(
                agreement_level=0.0,
                agreement_category="low",
                majority_position="No inputs",
                dissenting_views=[],
                recommendation="another_round",
            )

        # Extract positions and normalize
        positions = {}
        for provider, inp in inputs.items():
            pos = inp.position.lower().strip()
            # Normalize to support/oppose/neutral
            if any(word in pos for word in ["support", "agree", "yes", "favor"]):
                positions[provider] = "support"
            elif any(word in pos for word in ["oppose", "disagree", "no", "against"]):
                positions[provider] = "oppose"
            else:
                positions[provider] = "neutral"

        # Count positions
        position_counts: dict[str, int] = {}
        for pos in positions.values():
            position_counts[pos] = position_counts.get(pos, 0) + 1

        # Find majority
        total = len(positions)
        majority_pos = max(position_counts.keys(), key=lambda p: position_counts[p])
        majority_count = position_counts[majority_pos]

        # Calculate agreement level
        agreement_level = majority_count / total if total > 0 else 0.0

        # Categorize agreement
        if agreement_level >= self.config.consensus_threshold:
            category = "high"
        elif agreement_level >= self.config.low_agreement_threshold:
            category = "medium"
        else:
            category = "low"

        # Identify dissenters
        dissenting_views = []
        for provider, pos in positions.items():
            if pos != majority_pos:
                inp = inputs[provider]
                dissenting_views.append(DissentingView(
                    provider=provider,
                    position=inp.position,
                    reasoning=inp.analysis[:200] + "..." if len(inp.analysis) > 200 else inp.analysis,
                ))

        # Determine recommendation
        if category == "high":
            recommendation = "proceed"
        elif category == "low":
            recommendation = "escalate_to_user"
        else:
            recommendation = "another_round"

        return ConsensusEvaluation(
            agreement_level=agreement_level,
            agreement_category=category,
            majority_position=majority_pos,
            dissenting_views=dissenting_views,
            recommendation=recommendation,
        )

    def _build_feedback_context(
        self,
        inputs: dict[Provider, ParticipantInput],
        evaluation: ConsensusEvaluation,
    ) -> str:
        """Build context for next round including feedback."""
        lines = [
            f"Previous round results:",
            f"- Agreement level: {evaluation.agreement_level:.2%}",
            f"- Majority position: {evaluation.majority_position}",
            "",
            "Positions from each provider:",
        ]

        for provider, inp in inputs.items():
            lines.append(f"\n{provider.value.upper()}:")
            lines.append(f"  Position: {inp.position}")
            if inp.concerns:
                lines.append(f"  Concerns: {', '.join(inp.concerns)}")

        if evaluation.dissenting_views:
            lines.append("\nDissenting views to consider:")
            for dv in evaluation.dissenting_views:
                lines.append(f"  - {dv.provider.value}: {dv.position}")

        return "\n".join(lines)

    def _finalize_consensus(
        self,
        inputs: dict[Provider, ParticipantInput],
        evaluation: ConsensusEvaluation,
        all_entries: list[DiscussionEntry],
        escalated: bool,
    ) -> ConsensusResult:
        """Finalize a successful consensus."""
        vote_breakdown = {p.value: inp.position for p, inp in inputs.items()}

        # Build summary from majority position
        summary_lines = [
            f"Consensus reached with {evaluation.agreement_level:.0%} agreement.",
            f"Decision: {evaluation.majority_position}",
        ]

        if evaluation.dissenting_views:
            summary_lines.append(f"Note: {len(evaluation.dissenting_views)} dissenting view(s) recorded.")

        return ConsensusResult(
            success=True,
            final_decision=evaluation.majority_position,
            vote_breakdown=vote_breakdown,
            discussion_summary="\n".join(summary_lines),
            full_discussion_log=all_entries,
            escalated_to_user=escalated,
            user_questions=[],
        )

    def _finalize_with_majority(
        self,
        inputs: dict[Provider, ParticipantInput],
        evaluation: ConsensusEvaluation,
        all_entries: list[DiscussionEntry],
    ) -> ConsensusResult:
        """Finalize with majority decision after max rounds."""
        return self._finalize_consensus(inputs, evaluation, all_entries, escalated=False)

    def _escalate_to_user(
        self,
        inputs: dict[Provider, ParticipantInput],
        evaluation: ConsensusEvaluation,
        all_entries: list[DiscussionEntry],
    ) -> ConsensusResult:
        """Escalate to user due to low agreement."""
        vote_breakdown = {p.value: inp.position for p, inp in inputs.items()}

        # Generate questions for user
        questions = self._generate_user_questions(inputs, evaluation)

        return ConsensusResult(
            success=False,
            final_decision="",
            vote_breakdown=vote_breakdown,
            discussion_summary=f"Agreement level ({evaluation.agreement_level:.0%}) below threshold. User input requested.",
            full_discussion_log=all_entries,
            escalated_to_user=True,
            user_questions=questions,
        )

    def _generate_user_questions(
        self,
        inputs: dict[Provider, ParticipantInput],
        evaluation: ConsensusEvaluation,
    ) -> list[str]:
        """Generate questions to ask the user for clarification."""
        questions = []

        # Add main decision question
        positions = set(inp.position for inp in inputs.values())
        if len(positions) > 1:
            questions.append(
                f"The LLMs have different opinions. Please choose your preference among: {', '.join(positions)}"
            )

        # Add questions based on concerns
        all_concerns = []
        for inp in inputs.values():
            all_concerns.extend(inp.concerns)

        unique_concerns = list(set(all_concerns))[:3]  # Top 3 unique concerns
        for concern in unique_concerns:
            questions.append(f"How should we address this concern: {concern}?")

        return questions if questions else ["Please provide guidance on the preferred approach."]
