"""BrainstormModule - Multi-provider discussion and consensus building.

Enhanced brainstorming with:
- Combined rounds: (Preparation + Discussion) = 1 Round
- Sequential discussion with speaker rotation
- Personalized context for each participant
- Early consensus detection

Round Structure:
    Round N = Preparation Phase (parallel) + Discussion Phase (sequential)

    Preparation Phase:
        - Each provider independently analyzes the topic
        - Gemini: Strategic perspective, human-centric view
        - Claude: Technical analysis, implementation focus
        - Codex: Theoretical analysis, risk perspective

    Discussion Phase (sequential with rotation):
        - Round 1: Gemini → Claude → Codex
        - Round 2: Claude → Codex → Gemini
        - Round 3: Codex → Gemini → Claude
        - Each speaker sees personalized context (attacks to defend, etc.)

Key Principle: PRESERVE FULL RECORDS (no summarization allowed)
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Callable, Optional, Protocol

from u_llm_sdk.types import (
    BrainstormConfig,
    ConsensusResult,
    DiscussionEntry,
    ParticipantInput,
    Provider,
    LLMResult,
    # Enhanced discussion types
    ParticipantIdentity,
    DiscussionResponse,
    SupportRecord,
    DefenseRecord,
    CriticRecord,
    FreeCommentRecord,
)

from .consensus import ConsensusLoop
from .context_builder import (
    DiscussionState,
    DiscussionContextBuilder,
    SpeakerRotation,
    IDGenerator,
    create_participant_identity,
)

if TYPE_CHECKING:
    from u_llm_sdk.llm.providers.base import BaseProvider

logger = logging.getLogger(__name__)


class LLMProvider(Protocol):
    """Protocol for LLM provider."""

    async def run(
        self,
        prompt: str,
        *,
        session_id: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> LLMResult: ...


@dataclass
class BrainstormRound:
    """Record of a single brainstorming round.

    In the new structure, each round contains both preparation and discussion.

    Attributes:
        round_number: Which round (1, 2, or 3)
        round_type: Type of round ("preparation", "discussion", or "final")
        preparation_inputs: Inputs from preparation phase (parallel)
        discussion_responses: Responses from discussion phase (sequential)
        discussion_log: Full discussion entries (no summarization)
        speaker_order: Order of speakers in this round
        consensus_check: Whether consensus was reached this round
        started_at: When this round started
        ended_at: When this round ended
    """
    round_number: int
    round_type: str = ""
    preparation_inputs: dict[str, ParticipantInput] = field(default_factory=dict)
    discussion_responses: list[DiscussionResponse] = field(default_factory=list)
    discussion_log: list[DiscussionEntry] = field(default_factory=list)
    speaker_order: list[str] = field(default_factory=list)
    consensus_check: bool = False
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None


@dataclass
class BrainstormResult:
    """Result of a complete brainstorming session.

    Attributes:
        topic: The topic that was discussed
        rounds: All round records (full history)
        consensus: Final consensus result
        total_duration_ms: Total time spent
        all_discussion_entries: Flat list of all discussion entries
        early_consensus: Whether consensus was reached before max rounds
        final_round: Which round consensus was reached (or max)
    """
    topic: str
    rounds: list[BrainstormRound]
    consensus: ConsensusResult
    total_duration_ms: int = 0
    all_discussion_entries: list[DiscussionEntry] = field(default_factory=list)
    early_consensus: bool = False
    final_round: int = 0


# Provider-specific prompts for preparation phase
PREPARATION_PROMPTS = {
    Provider.GEMINI: """You are {identity}, analyzing this topic from a STRATEGIC and HUMAN-CENTRIC perspective.

Topic: {topic}

Focus on:
- User impact and experience
- Strategic implications
- Communication aspects
- Long-term sustainability

Provide your analysis in JSON format:
{{
  "statement_id": "{statement_id}",
  "analysis": "Your strategic analysis",
  "position": "Your stance",
  "supporting_evidence": ["Evidence 1", "Evidence 2"],
  "concerns": ["Concern 1"],
  "proposed_approach": "Your recommended approach"
}}""",

    Provider.CLAUDE: """You are {identity}, analyzing this topic from a TECHNICAL and IMPLEMENTATION perspective.

Topic: {topic}

Focus on:
- Technical feasibility
- Implementation details
- Code quality considerations
- Best practices

Provide your analysis in JSON format:
{{
  "statement_id": "{statement_id}",
  "analysis": "Your technical analysis",
  "position": "Your stance",
  "supporting_evidence": ["Evidence 1", "Evidence 2"],
  "concerns": ["Concern 1"],
  "proposed_approach": "Your recommended approach"
}}""",

    Provider.CODEX: """You are {identity}, analyzing this topic from a THEORETICAL and RISK perspective.

Topic: {topic}

Focus on:
- Theoretical soundness
- Potential risks and edge cases
- Academic/research considerations
- Hidden dangers

Provide your analysis in JSON format:
{{
  "statement_id": "{statement_id}",
  "analysis": "Your theoretical analysis",
  "position": "Your stance",
  "supporting_evidence": ["Evidence 1", "Evidence 2"],
  "concerns": ["Concern 1"],
  "proposed_approach": "Your recommended approach"
}}""",
}


# Discussion phase prompt (sequential)
DISCUSSION_PROMPT = """Round {round_number} Discussion - Your Turn

You are {identity}.

Topic: {topic}

{context}

=== ALL PARTICIPANTS' PREPARATION ===
{all_preparations}

=== PREVIOUS DISCUSSION IN THIS ROUND ===
{previous_discussion}

Now respond to the discussion. You MUST use the structured JSON format below.
Consider:
- Defend against attacks targeting you (if any)
- Support positions you agree with
- Criticize positions you disagree with
- Add free comments if needed

CRITICAL: Use exact participant IDs when referencing others.

Response format (JSON):
{{
  "speaker_id": "{speaker_id}",
  "supports": [
    {{
      "support_id": "sup-XXX",
      "target_id": "<participant_id you support>",
      "target_statement_id": "<their statement_id>",
      "reason": "Why you support this"
    }}
  ],
  "defenses": [
    {{
      "defense_id": "def-XXX",
      "attack_id": "<attack_id you're defending against>",
      "attacker_id": "<who attacked you>",
      "rebuttal": "Your defense/rebuttal"
    }}
  ],
  "critics": [
    {{
      "attack_id": "atk-XXX",
      "target_id": "<participant_id you criticize>",
      "target_statement_id": "<their statement_id>",
      "criticism": "Your criticism with reasoning"
    }}
  ],
  "free_comment": {{
    "comment_id": "fc-XXX",
    "content": "Any additional thoughts",
    "mentions": ["<participant_ids you mention>"]
  }},
  "consensus_reached": false,
  "round_number": {round_number}
}}

Note: Empty arrays [] are fine if you have nothing for that category."""


class BrainstormModule:
    """Orchestrates brainstorming sessions with sequential discussion.

    Key Features:
    - (Preparation + Discussion) = 1 Round
    - Sequential discussion with speaker rotation
    - Personalized context for each participant
    - Early consensus detection

    Example:
        >>> providers = {
        ...     Provider.GEMINI: gemini_provider,
        ...     Provider.CLAUDE: claude_provider,
        ...     Provider.CODEX: codex_provider,
        ... }
        >>> module = BrainstormModule(providers)
        >>> result = await module.run_session("Should we use microservices?")
        >>> if result.consensus.success:
        ...     print(f"Decision: {result.consensus.final_decision}")
        >>> if result.early_consensus:
        ...     print(f"Reached consensus in round {result.final_round}")

    Attributes:
        providers: Dict of Provider -> LLMProvider
        config: BrainstormConfig with session parameters
    """

    def __init__(
        self,
        providers: dict[Provider, LLMProvider],
        config: Optional[BrainstormConfig] = None,
        *,
        model_names: Optional[dict[Provider, str]] = None,
        on_progress: Optional[Callable[[str], None]] = None,
        on_stream: Optional[Callable[[str, str], None]] = None,
    ):
        """Initialize BrainstormModule.

        Args:
            providers: Dict mapping Provider to provider instances
            config: Optional BrainstormConfig for customization
            model_names: Optional model name mapping for identity creation
            on_progress: Optional callback for progress updates (receives status messages)
            on_stream: Optional callback for streaming output (speaker_name, text_chunk)
        """
        self.providers = providers
        self.config = config or BrainstormConfig()
        self._consensus_loop = ConsensusLoop(providers, config)
        self._on_progress = on_progress
        self._on_stream = on_stream

        # Model names for identity creation
        # Priority: explicit model_names > provider.config.get_model() > fallback
        self._model_names = model_names or self._extract_model_names_from_providers()

        # Session state
        self._state: Optional[DiscussionState] = None
        self._id_gen: Optional[IDGenerator] = None
        self._rotation: Optional[SpeakerRotation] = None
        self._identities: dict[Provider, ParticipantIdentity] = {}

    def _emit_progress(self, message: str) -> None:
        """Emit progress message to callback if available."""
        logger.info(message)
        if self._on_progress:
            try:
                self._on_progress(message)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")

    async def _run_with_stream(
        self,
        provider: Provider,
        prompt: str,
        speaker_name: str,
    ) -> str:
        """Run LLM with streaming output.

        Uses provider.stream() if on_stream callback is set,
        otherwise falls back to provider.run().

        Args:
            provider: Provider enum
            prompt: Prompt to send
            speaker_name: Display name for streaming output

        Returns:
            Complete response text
        """
        llm = self.providers[provider]

        # If no stream callback, use regular run
        if not self._on_stream:
            result = await llm.run(prompt)
            return result.text

        # Use streaming
        collected_text = []
        try:
            async for event in llm.stream(prompt):
                # Extract text from event
                # Event format varies by provider but typically has 'text' or 'content'
                chunk = ""
                if isinstance(event, dict):
                    chunk = event.get("text", "") or event.get("content", "")
                    # Handle nested content structure
                    if not chunk and "message" in event:
                        msg = event["message"]
                        if isinstance(msg, dict):
                            chunk = msg.get("text", "") or msg.get("content", "")
                elif isinstance(event, str):
                    chunk = event

                if chunk:
                    collected_text.append(chunk)
                    try:
                        self._on_stream(speaker_name, chunk)
                    except Exception as e:
                        logger.warning(f"Stream callback failed: {e}")

        except Exception as e:
            logger.warning(f"Streaming failed for {provider}, falling back to run(): {e}")
            result = await llm.run(prompt)
            return result.text

        return "".join(collected_text)

    def _extract_model_names_from_providers(self) -> dict[Provider, str]:
        """Extract model names from provider configs.

        Falls back to reasonable defaults if config doesn't specify model.
        """
        fallbacks = {
            Provider.GEMINI: "gemini",
            Provider.CLAUDE: "claude",
            Provider.CODEX: "codex",
        }
        model_names = {}
        for provider, instance in self.providers.items():
            try:
                # Get model from provider's config
                model = instance.config.get_model(require_explicit=False)
                if model:
                    model_names[provider] = model
                else:
                    model_names[provider] = fallbacks.get(provider, provider.value)
            except Exception:
                model_names[provider] = fallbacks.get(provider, provider.value)
        return model_names

    def _init_session(self, topic: str) -> None:
        """Initialize session state."""
        self._state = DiscussionState(topic=topic)
        self._id_gen = IDGenerator()

        # Create identities for all participants
        self._identities = {}
        participant_ids = []

        for idx, provider in enumerate(self.providers.keys(), start=1):
            model = self._model_names.get(provider, "default")
            identity = create_participant_identity(provider, model, idx)
            self._identities[provider] = identity
            self._state.register_participant(identity)
            participant_ids.append(identity.full_id)

        # Initialize rotation
        self._rotation = SpeakerRotation(participant_ids)

    async def run_session(
        self,
        topic: str,
        *,
        context: Optional[str] = None,
    ) -> BrainstormResult:
        """Run a complete brainstorming session.

        Executes rounds until consensus or max_rounds reached.
        Each round = Preparation (parallel) + Discussion (sequential).

        Args:
            topic: The topic to brainstorm about
            context: Optional additional context

        Returns:
            BrainstormResult with full history and consensus
        """
        start_time = datetime.now()
        self._init_session(topic)

        rounds: list[BrainstormRound] = []
        all_entries: list[DiscussionEntry] = []
        early_consensus = False
        final_round = 0

        for round_num in range(1, self.config.max_rounds + 1):
            self._emit_progress(f"Round {round_num}/{self.config.max_rounds} starting...")
            self._rotation.current_round = round_num

            # Execute round (prep + discussion)
            round_result = await self._execute_round(topic, round_num, context)
            rounds.append(round_result)
            all_entries.extend(round_result.discussion_log)

            final_round = round_num

            # Check for early consensus
            if round_result.consensus_check:
                self._emit_progress(f"Early consensus reached in round {round_num}")
                early_consensus = True
                break

        # Build final consensus
        consensus = await self._build_consensus(rounds[-1], all_entries)

        end_time = datetime.now()
        duration_ms = int((end_time - start_time).total_seconds() * 1000)

        return BrainstormResult(
            topic=topic,
            rounds=rounds,
            consensus=consensus,
            total_duration_ms=duration_ms,
            all_discussion_entries=all_entries,
            early_consensus=early_consensus,
            final_round=final_round,
        )

    async def _execute_round(
        self,
        topic: str,
        round_num: int,
        context: Optional[str],
    ) -> BrainstormRound:
        """Execute a single round (preparation + discussion)."""
        round_start = datetime.now()
        speaker_order = self._rotation.get_current_order()

        # Phase 1: Preparation (parallel)
        providers_str = ", ".join(p.value for p in self.providers.keys())
        self._emit_progress(f"  Preparation: {providers_str} (parallel)...")
        prep_inputs = await self._preparation_phase(topic, round_num, context)

        # Phase 2: Discussion (sequential with rotation)
        # Convert full_id to display_name for cleaner progress output
        display_names = [
            self._state.participants[fid].display_name
            if fid in self._state.participants else fid
            for fid in speaker_order[:3]
        ]
        self._emit_progress(f"  Discussion: {' → '.join(display_names)}...")
        logger.debug(f"Full speaker order: {speaker_order}")

        discussion_responses, discussion_log = await self._discussion_phase(
            topic, round_num, prep_inputs, speaker_order
        )

        # Check consensus from responses
        consensus_votes = sum(1 for r in discussion_responses if r.consensus_reached)
        consensus_check = consensus_votes >= len(discussion_responses) * self.config.consensus_threshold

        # Determine round type based on round number
        round_type_map = {1: "preparation", 2: "discussion", 3: "final"}
        round_type = round_type_map.get(round_num, f"round_{round_num}")

        return BrainstormRound(
            round_number=round_num,
            round_type=round_type,
            preparation_inputs=prep_inputs,
            discussion_responses=discussion_responses,
            discussion_log=discussion_log,
            speaker_order=speaker_order,
            consensus_check=consensus_check,
            started_at=round_start,
            ended_at=datetime.now(),
        )

    async def _preparation_phase(
        self,
        topic: str,
        round_num: int,
        context: Optional[str],
    ) -> dict[str, ParticipantInput]:
        """Execute preparation phase (parallel)."""
        inputs: dict[str, ParticipantInput] = {}

        async def prepare(provider: Provider) -> tuple[str, ParticipantInput]:
            identity = self._identities[provider]
            statement_id = self._id_gen.next_statement_id(identity.full_id)

            prompt_template = PREPARATION_PROMPTS.get(
                provider,
                PREPARATION_PROMPTS[Provider.CLAUDE]
            )
            prompt = prompt_template.format(
                identity=identity.display_name,
                topic=topic,
                statement_id=statement_id,
            )

            if context:
                prompt = f"{prompt}\n\nAdditional context:\n{context}"

            try:
                llm = self.providers[provider]
                result = await llm.run(prompt)
                data = self._parse_json_response(result.text)

                return identity.full_id, ParticipantInput(
                    provider=provider,
                    analysis=data.get("analysis", result.text),
                    position=data.get("position", "analyzing"),
                    supporting_evidence=data.get("supporting_evidence", []),
                    concerns=data.get("concerns", []),
                    proposed_approach=data.get("proposed_approach", ""),
                )
            except Exception as e:
                logger.warning(f"Preparation failed for {provider}: {e}")
                return identity.full_id, ParticipantInput(
                    provider=provider,
                    analysis=f"Error during preparation: {e}",
                    position="error",
                )

        # Execute in parallel
        tasks = [prepare(p) for p in self.providers.keys()]
        results = await asyncio.gather(*tasks)

        for full_id, inp in results:
            inputs[full_id] = inp

        return inputs

    async def _discussion_phase(
        self,
        topic: str,
        round_num: int,
        prep_inputs: dict[str, ParticipantInput],
        speaker_order: list[str],
    ) -> tuple[list[DiscussionResponse], list[DiscussionEntry]]:
        """Execute discussion phase (sequential with rotation)."""
        responses: list[DiscussionResponse] = []
        entries: list[DiscussionEntry] = []
        previous_discussion: list[str] = []

        # Format all preparations for prompt
        all_preparations = self._format_preparations(prep_inputs)

        # Sequential execution in rotation order
        for speaker_id in speaker_order:
            identity = self._state.participants[speaker_id]
            provider = identity.provider

            # Build personalized context
            builder = DiscussionContextBuilder(self._state)
            try:
                context = builder.build_context_for(speaker_id)
                context_str = builder.format_context_prompt(context)
            except ValueError:
                context_str = "=== First time speaking, no attacks to defend ==="

            # Build prompt
            prompt = DISCUSSION_PROMPT.format(
                round_number=round_num,
                identity=identity.display_name,
                topic=topic,
                context=context_str,
                all_preparations=all_preparations,
                previous_discussion="\n".join(previous_discussion) if previous_discussion else "(You speak first in this round)",
                speaker_id=speaker_id,
            )

            try:
                # Use streaming if callback is set
                response_text = await self._run_with_stream(
                    provider, prompt, identity.display_name
                )
                response = self._parse_discussion_response(response_text, speaker_id, round_num)

                # Add to state for next speaker's context
                self._state.add_response(response)
                responses.append(response)

                # Create discussion entry
                entry = DiscussionEntry(
                    timestamp=datetime.now(),
                    speaker=provider,
                    message_type="opinion",
                    content=self._format_response_content(response),
                )
                entries.append(entry)

                # Add to previous discussion for next speaker
                previous_discussion.append(
                    f"[{identity.display_name}]: {self._format_response_summary(response)}"
                )

            except Exception as e:
                logger.warning(f"Discussion failed for {speaker_id}: {e}")
                # Create minimal response on error
                response = DiscussionResponse(
                    speaker_id=speaker_id,
                    round_number=round_num,
                    timestamp=datetime.now(),
                )
                responses.append(response)

                entry = DiscussionEntry(
                    timestamp=datetime.now(),
                    speaker=provider,
                    message_type="opinion",
                    content=f"Error: {e}",
                )
                entries.append(entry)

        return responses, entries

    def _format_preparations(self, prep_inputs: dict[str, ParticipantInput]) -> str:
        """Format all preparations for discussion prompt."""
        lines = []
        for full_id, inp in prep_inputs.items():
            identity = self._state.participants.get(full_id)
            name = identity.display_name if identity else full_id
            lines.append(f"\n--- {name} ({full_id}) ---")
            lines.append(f"Position: {inp.position}")
            lines.append(f"Analysis: {inp.analysis}")
            if inp.concerns:
                lines.append(f"Concerns: {', '.join(inp.concerns)}")
            if inp.proposed_approach:
                lines.append(f"Approach: {inp.proposed_approach}")
        return "\n".join(lines)

    def _parse_discussion_response(
        self,
        text: str,
        speaker_id: str,
        round_num: int,
    ) -> DiscussionResponse:
        """Parse LLM response into DiscussionResponse."""
        data = self._parse_json_response(text)

        # Parse supports
        supports = []
        for s in data.get("supports", []):
            supports.append(SupportRecord(
                support_id=s.get("support_id", self._id_gen.next_support_id()),
                target_id=s.get("target_id", ""),
                target_statement_id=s.get("target_statement_id", ""),
                reason=s.get("reason", ""),
                timestamp=datetime.now(),
            ))

        # Parse defenses
        defenses = []
        for d in data.get("defenses", []):
            defenses.append(DefenseRecord(
                defense_id=d.get("defense_id", self._id_gen.next_defense_id()),
                attack_id=d.get("attack_id", ""),
                attacker_id=d.get("attacker_id", ""),
                rebuttal=d.get("rebuttal", ""),
                timestamp=datetime.now(),
            ))

        # Parse critics
        critics = []
        for c in data.get("critics", []):
            critics.append(CriticRecord(
                attack_id=c.get("attack_id", self._id_gen.next_attack_id()),
                target_id=c.get("target_id", ""),
                target_statement_id=c.get("target_statement_id", ""),
                criticism=c.get("criticism", ""),
                timestamp=datetime.now(),
            ))

        # Parse free comment
        free_comment = None
        fc_data = data.get("free_comment")
        if fc_data and fc_data.get("content"):
            free_comment = FreeCommentRecord(
                comment_id=fc_data.get("comment_id", self._id_gen.next_comment_id()),
                content=fc_data.get("content", ""),
                mentions=fc_data.get("mentions", []),
                timestamp=datetime.now(),
            )

        return DiscussionResponse(
            speaker_id=speaker_id,
            supports=supports,
            defenses=defenses,
            critics=critics,
            free_comment=free_comment,
            consensus_reached=data.get("consensus_reached", False),
            round_number=round_num,
            timestamp=datetime.now(),
        )

    def _format_response_content(self, response: DiscussionResponse) -> str:
        """Format response for discussion log (full content, no summarization)."""
        lines = []

        if response.defenses:
            lines.append("DEFENSES:")
            for d in response.defenses:
                lines.append(f"  [{d.defense_id}] Against {d.attack_id}: {d.rebuttal}")

        if response.supports:
            lines.append("SUPPORTS:")
            for s in response.supports:
                lines.append(f"  [{s.support_id}] For {s.target_id}: {s.reason}")

        if response.critics:
            lines.append("CRITICISMS:")
            for c in response.critics:
                lines.append(f"  [{c.attack_id}] To {c.target_id}: {c.criticism}")

        if response.free_comment:
            lines.append(f"COMMENT: {response.free_comment.content}")

        if response.consensus_reached:
            lines.append("** SIGNALS CONSENSUS REACHED **")

        return "\n".join(lines) if lines else "(No substantive response)"

    def _format_response_summary(self, response: DiscussionResponse) -> str:
        """Format brief summary for next speaker's context."""
        parts = []
        if response.defenses:
            parts.append(f"defended {len(response.defenses)} attacks")
        if response.supports:
            parts.append(f"supported {len(response.supports)} positions")
        if response.critics:
            parts.append(f"criticized {len(response.critics)} positions")
        if response.free_comment:
            parts.append("added comment")
        if response.consensus_reached:
            parts.append("signals consensus")

        return ", ".join(parts) if parts else "no interaction"

    async def _build_consensus(
        self,
        final_round: BrainstormRound,
        all_entries: list[DiscussionEntry],
    ) -> ConsensusResult:
        """Build final consensus from the last round."""
        # Convert to legacy format for ConsensusLoop compatibility
        legacy_inputs: dict[Provider, ParticipantInput] = {}
        for full_id, inp in final_round.preparation_inputs.items():
            identity = self._state.participants.get(full_id)
            if identity:
                legacy_inputs[identity.provider] = inp

        # Use ConsensusLoop's evaluation logic
        evaluation = self._consensus_loop._evaluate_consensus(legacy_inputs)

        vote_breakdown = {
            full_id: inp.position
            for full_id, inp in final_round.preparation_inputs.items()
        }

        # Check if consensus was reached
        success = evaluation.agreement_level >= self.config.consensus_threshold

        if not success and evaluation.agreement_level < self.config.low_agreement_threshold:
            user_questions = self._generate_user_questions(final_round)
            return ConsensusResult(
                success=False,
                final_decision="",
                vote_breakdown=vote_breakdown,
                discussion_summary=f"Low agreement ({evaluation.agreement_level:.0%}). User input needed.",
                full_discussion_log=all_entries,
                escalated_to_user=True,
                user_questions=user_questions,
            )

        return ConsensusResult(
            success=success,
            final_decision=evaluation.majority_position,
            vote_breakdown=vote_breakdown,
            discussion_summary=self._build_summary(evaluation, final_round),
            full_discussion_log=all_entries,
            escalated_to_user=False,
            user_questions=[],
        )

    def _build_summary(self, evaluation, final_round: BrainstormRound) -> str:
        """Build summary for successful consensus."""
        lines = [
            f"Consensus reached: {evaluation.majority_position}",
            f"Agreement level: {evaluation.agreement_level:.0%}",
            "",
            "Final positions:",
        ]

        for full_id, inp in final_round.preparation_inputs.items():
            lines.append(f"  - {full_id}: {inp.position}")

        if evaluation.dissenting_views:
            lines.append("")
            lines.append("Dissenting views recorded for future reference.")

        return "\n".join(lines)

    def _generate_user_questions(self, final_round: BrainstormRound) -> list[str]:
        """Generate questions for user when consensus fails."""
        questions = ["The LLMs could not reach consensus. Please provide guidance."]

        all_concerns = []
        for inp in final_round.preparation_inputs.values():
            all_concerns.extend(inp.concerns)

        unique_concerns = list(set(all_concerns))[:3]
        for concern in unique_concerns:
            questions.append(f"How should we address: {concern}?")

        return questions

    def _parse_json_response(self, text: str) -> dict:
        """Parse JSON from LLM response."""
        text = text.strip()

        # Handle code blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()

        # Find JSON object
        if "{" in text:
            start = text.find("{")
            depth = 0
            for i, c in enumerate(text[start:], start):
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        text = text[start:i+1]
                        break

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"analysis": text}
