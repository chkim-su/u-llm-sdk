"""DiscussionContextBuilder - Build personalized context for brainstorm participants.

This module transforms the global discussion state into participant-specific
context, enabling each participant to understand:
- Attacks they need to defend against
- Support they've received
- Their own interaction history
- Mentions in free comments

The builder is called before each participant's turn in sequential discussion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from u_llm_sdk.types import (
    Provider,
    ParticipantIdentity,
    ParticipantContext,
    DiscussionResponse,
    SupportRecord,
    DefenseRecord,
    CriticRecord,
    FreeCommentRecord,
)


@dataclass
class DiscussionState:
    """Global state of the discussion across all rounds.

    Attributes:
        responses: All responses from all participants across rounds
        participants: Registry of all participants by full_id
        current_round: Current round number
        topic: Discussion topic
    """
    responses: list[DiscussionResponse] = field(default_factory=list)
    participants: dict[str, ParticipantIdentity] = field(default_factory=dict)
    current_round: int = 1
    topic: str = ""

    def add_response(self, response: DiscussionResponse) -> None:
        """Add a response to the discussion state."""
        self.responses.append(response)

    def register_participant(self, identity: ParticipantIdentity) -> None:
        """Register a participant in the discussion."""
        self.participants[identity.full_id] = identity

    def get_all_critics(self) -> list[CriticRecord]:
        """Get all criticism records from all responses."""
        critics = []
        for response in self.responses:
            critics.extend(response.critics)
        return critics

    def get_all_supports(self) -> list[SupportRecord]:
        """Get all support records from all responses."""
        supports = []
        for response in self.responses:
            supports.extend(response.supports)
        return supports

    def get_all_defenses(self) -> list[DefenseRecord]:
        """Get all defense records from all responses."""
        defenses = []
        for response in self.responses:
            defenses.extend(response.defenses)
        return defenses

    def get_all_free_comments(self) -> list[FreeCommentRecord]:
        """Get all free comment records from all responses."""
        comments = []
        for response in self.responses:
            if response.free_comment:
                comments.append(response.free_comment)
        return comments


class DiscussionContextBuilder:
    """Builds personalized context for each participant.

    The builder analyzes the global discussion state and creates
    a ParticipantContext tailored to each participant's perspective.

    Example:
        >>> builder = DiscussionContextBuilder(state)
        >>> context = builder.build_context_for(participant_id)
        >>> # context.undefended_attacks contains attacks needing defense
        >>> # context.supports_received contains support statements
    """

    def __init__(self, state: DiscussionState):
        """Initialize with discussion state.

        Args:
            state: The global discussion state
        """
        self.state = state
        self._defense_map: dict[str, DefenseRecord] = {}
        self._build_defense_map()

    def _build_defense_map(self) -> None:
        """Build a map of attack_id -> defense for quick lookup."""
        for defense in self.state.get_all_defenses():
            self._defense_map[defense.attack_id] = defense

    def build_context_for(self, participant_id: str) -> ParticipantContext:
        """Build personalized context for a specific participant.

        Args:
            participant_id: Full ID of the participant (e.g., "gemini-2.5-pro-001")

        Returns:
            ParticipantContext with all relevant interaction data
        """
        identity = self.state.participants.get(participant_id)
        if not identity:
            raise ValueError(f"Unknown participant: {participant_id}")

        # Categorize attacks on this participant
        undefended_attacks: list[CriticRecord] = []
        defended_attacks: list[CriticRecord] = []

        for critic in self.state.get_all_critics():
            if critic.target_id == participant_id:
                if critic.attack_id in self._defense_map:
                    defended_attacks.append(critic)
                else:
                    undefended_attacks.append(critic)

        # Collect supports received
        supports_received = [
            support for support in self.state.get_all_supports()
            if support.target_id == participant_id
        ]

        # Collect mentions in free comments
        mentions_received = [
            comment for comment in self.state.get_all_free_comments()
            if participant_id in comment.mentions
        ]

        # Collect participant's own actions
        my_attacks: list[CriticRecord] = []
        my_supports: list[SupportRecord] = []
        my_defenses: list[DefenseRecord] = []

        for response in self.state.responses:
            if response.speaker_id == participant_id:
                my_attacks.extend(response.critics)
                my_supports.extend(response.supports)
                my_defenses.extend(response.defenses)

        return ParticipantContext(
            identity=identity,
            undefended_attacks=undefended_attacks,
            defended_attacks=defended_attacks,
            supports_received=supports_received,
            mentions_received=mentions_received,
            my_attacks=my_attacks,
            my_supports=my_supports,
            my_defenses=my_defenses,
        )

    def build_all_contexts(self) -> dict[str, ParticipantContext]:
        """Build context for all registered participants.

        Returns:
            Dict mapping participant_id to their context
        """
        return {
            pid: self.build_context_for(pid)
            for pid in self.state.participants.keys()
        }

    def format_context_prompt(
        self,
        context: ParticipantContext,
        include_history: bool = True,
    ) -> str:
        """Format context as a prompt string for LLM consumption.

        Args:
            context: The participant context
            include_history: Whether to include own interaction history

        Returns:
            Formatted string suitable for LLM prompt
        """
        lines = [
            f"=== Your Identity ===",
            f"You are: {context.identity.display_name}",
            f"Full ID: {context.identity.full_id}",
            "",
        ]

        # Undefended attacks (priority)
        if context.undefended_attacks:
            lines.append("=== ATTACKS REQUIRING YOUR DEFENSE ===")
            for attack in context.undefended_attacks:
                lines.append(f"[{attack.attack_id}] From {attack.attacker_id if hasattr(attack, 'attacker_id') else 'unknown'}:")
                lines.append(f"  Target statement: {attack.target_statement_id}")
                lines.append(f"  Criticism: {attack.criticism}")
                lines.append("")

        # Supports received
        if context.supports_received:
            lines.append("=== SUPPORT YOU'VE RECEIVED ===")
            for support in context.supports_received:
                lines.append(f"[{support.support_id}] From {support.target_id}:")
                lines.append(f"  Reason: {support.reason}")
                lines.append("")

        # Mentions
        if context.mentions_received:
            lines.append("=== MENTIONS IN FREE COMMENTS ===")
            for comment in context.mentions_received:
                lines.append(f"[{comment.comment_id}]: {comment.content}")
                lines.append("")

        # Own history (optional)
        if include_history:
            if context.my_attacks or context.my_supports or context.my_defenses:
                lines.append("=== YOUR INTERACTION HISTORY ===")

                if context.my_attacks:
                    lines.append(f"Attacks made: {len(context.my_attacks)}")
                    for atk in context.my_attacks[-3:]:  # Last 3
                        lines.append(f"  - [{atk.attack_id}] vs {atk.target_id}: {atk.criticism[:50]}...")

                if context.my_defenses:
                    lines.append(f"Defenses made: {len(context.my_defenses)}")

                if context.my_supports:
                    lines.append(f"Supports given: {len(context.my_supports)}")

                lines.append("")

        return "\n".join(lines)


@dataclass
class SpeakerRotation:
    """Manages speaker rotation across rounds.

    Round 1: G → C → X
    Round 2: C → X → G
    Round 3: X → G → C

    Attributes:
        participants: List of participant IDs in base order
        current_round: Current round number
    """
    participants: list[str]
    current_round: int = 1

    def get_order_for_round(self, round_num: int) -> list[str]:
        """Get speaker order for a specific round.

        Args:
            round_num: Round number (1-based)

        Returns:
            List of participant IDs in speaking order
        """
        if not self.participants:
            return []

        # Rotate by (round_num - 1) positions
        offset = (round_num - 1) % len(self.participants)
        return self.participants[offset:] + self.participants[:offset]

    def get_current_order(self) -> list[str]:
        """Get speaker order for current round."""
        return self.get_order_for_round(self.current_round)

    def advance_round(self) -> None:
        """Move to next round."""
        self.current_round += 1


class IDGenerator:
    """Generates unique IDs for discussion elements."""

    def __init__(self):
        self._counters: dict[str, int] = {
            "atk": 0,
            "def": 0,
            "sup": 0,
            "fc": 0,
            "stmt": 0,
        }

    def next_attack_id(self) -> str:
        """Generate next attack ID."""
        self._counters["atk"] += 1
        return f"atk-{self._counters['atk']:03d}"

    def next_defense_id(self) -> str:
        """Generate next defense ID."""
        self._counters["def"] += 1
        return f"def-{self._counters['def']:03d}"

    def next_support_id(self) -> str:
        """Generate next support ID."""
        self._counters["sup"] += 1
        return f"sup-{self._counters['sup']:03d}"

    def next_comment_id(self) -> str:
        """Generate next free comment ID."""
        self._counters["fc"] += 1
        return f"fc-{self._counters['fc']:03d}"

    def next_statement_id(self, speaker_id: str) -> str:
        """Generate next statement ID for a speaker."""
        self._counters["stmt"] += 1
        return f"{speaker_id}-stmt-{self._counters['stmt']:03d}"


def create_participant_identity(
    provider: Provider,
    model: str,
    participant_num: int,
) -> ParticipantIdentity:
    """Factory function to create ParticipantIdentity.

    Args:
        provider: Provider enum (GEMINI, CLAUDE, CODEX)
        model: Model name (e.g., "2.5-pro", "opus-4")
        participant_num: Unique number for this session

    Returns:
        ParticipantIdentity with formatted ID
    """
    return ParticipantIdentity(
        provider=provider,
        model=model,
        participant_id=f"{participant_num:03d}",
    )
