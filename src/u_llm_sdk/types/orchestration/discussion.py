"""LLM Types - Discussion Types for Enhanced Brainstorming.

Participant identity, interaction records, and context types for
structured brainstorm discussions with precise targeting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from ..config import Provider


@dataclass
class ParticipantIdentity:
    """Unique identity for a brainstorm participant.

    Format: {provider}-{model}-{participant_id}
    Example: gemini-2.5-pro-001, claude-opus-4-002

    Attributes:
        provider: Provider type (GEMINI, CLAUDE, CODEX)
        model: Model name (e.g., "2.5-pro", "opus-4")
        participant_id: Session-unique ID (e.g., "001", "002")
    """
    provider: Provider
    model: str
    participant_id: str

    @property
    def full_id(self) -> str:
        """Full identifier: gemini-2.5-pro-001"""
        return f"{self.provider.value}-{self.model}-{self.participant_id}"

    @property
    def display_name(self) -> str:
        """Display name: Gemini (001)"""
        return f"{self.provider.value.capitalize()} ({self.participant_id})"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "provider": self.provider.value,
            "model": self.model,
            "participant_id": self.participant_id,
            "full_id": self.full_id,
            "display_name": self.display_name,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ParticipantIdentity:
        """Create from dictionary (JSON deserialization)."""
        provider_value = data["provider"]
        provider = Provider(provider_value) if isinstance(provider_value, str) else provider_value

        return cls(
            provider=provider,
            model=data["model"],
            participant_id=data["participant_id"],
        )


@dataclass
class SupportRecord:
    """Record of a support statement.

    Attributes:
        support_id: Unique ID for this support (e.g., "sup-001")
        target_id: Full ID of participant being supported
        target_statement_id: ID of the statement being supported
        reason: Reason for support
        timestamp: When this support was made
    """
    support_id: str
    target_id: str
    target_statement_id: str
    reason: str
    timestamp: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "support_id": self.support_id,
            "target_id": self.target_id,
            "target_statement_id": self.target_statement_id,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SupportRecord:
        """Create from dictionary (JSON deserialization)."""
        timestamp_value = data.get("timestamp")
        timestamp = (
            datetime.fromisoformat(timestamp_value)
            if timestamp_value and isinstance(timestamp_value, str)
            else None
        )

        return cls(
            support_id=data["support_id"],
            target_id=data["target_id"],
            target_statement_id=data["target_statement_id"],
            reason=data["reason"],
            timestamp=timestamp,
        )


@dataclass
class DefenseRecord:
    """Record of a defense statement against criticism.

    Attributes:
        defense_id: Unique ID for this defense (e.g., "def-001")
        attack_id: ID of the attack being defended against
        attacker_id: Full ID of the attacker
        rebuttal: The defense/rebuttal content
        timestamp: When this defense was made
    """
    defense_id: str
    attack_id: str
    attacker_id: str
    rebuttal: str
    timestamp: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "defense_id": self.defense_id,
            "attack_id": self.attack_id,
            "attacker_id": self.attacker_id,
            "rebuttal": self.rebuttal,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> DefenseRecord:
        """Create from dictionary (JSON deserialization)."""
        timestamp_value = data.get("timestamp")
        timestamp = (
            datetime.fromisoformat(timestamp_value)
            if timestamp_value and isinstance(timestamp_value, str)
            else None
        )

        return cls(
            defense_id=data["defense_id"],
            attack_id=data["attack_id"],
            attacker_id=data["attacker_id"],
            rebuttal=data["rebuttal"],
            timestamp=timestamp,
        )


@dataclass
class CriticRecord:
    """Record of a criticism/attack statement.

    Attributes:
        attack_id: Unique ID for this attack (e.g., "atk-001")
        target_id: Full ID of participant being criticized
        target_statement_id: ID of the statement being criticized
        criticism: The criticism content
        timestamp: When this criticism was made
    """
    attack_id: str
    target_id: str
    target_statement_id: str
    criticism: str
    timestamp: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "attack_id": self.attack_id,
            "target_id": self.target_id,
            "target_statement_id": self.target_statement_id,
            "criticism": self.criticism,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CriticRecord:
        """Create from dictionary (JSON deserialization)."""
        timestamp_value = data.get("timestamp")
        timestamp = (
            datetime.fromisoformat(timestamp_value)
            if timestamp_value and isinstance(timestamp_value, str)
            else None
        )

        return cls(
            attack_id=data["attack_id"],
            target_id=data["target_id"],
            target_statement_id=data["target_statement_id"],
            criticism=data["criticism"],
            timestamp=timestamp,
        )


@dataclass
class FreeCommentRecord:
    """Record of a free-form comment.

    Attributes:
        comment_id: Unique ID for this comment (e.g., "fc-001")
        content: The comment content
        mentions: List of participant IDs mentioned
        timestamp: When this comment was made
    """
    comment_id: str
    content: str
    mentions: list[str] = field(default_factory=list)
    timestamp: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "comment_id": self.comment_id,
            "content": self.content,
            "mentions": self.mentions,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> FreeCommentRecord:
        """Create from dictionary (JSON deserialization)."""
        timestamp_value = data.get("timestamp")
        timestamp = (
            datetime.fromisoformat(timestamp_value)
            if timestamp_value and isinstance(timestamp_value, str)
            else None
        )

        return cls(
            comment_id=data["comment_id"],
            content=data["content"],
            mentions=data.get("mentions", []),
            timestamp=timestamp,
        )


@dataclass
class DiscussionResponse:
    """Structured response from a participant in discussion.

    This is the standardized JSON format each participant must follow
    during the discussion phase of brainstorming.

    Attributes:
        speaker_id: Full ID of the speaker
        supports: List of support statements
        defenses: List of defense statements against attacks
        critics: List of criticism statements
        free_comment: Optional free-form comment
        consensus_reached: Self-assessment of whether consensus is reached
        round_number: Which round this response is from
        timestamp: When this response was made
    """
    speaker_id: str
    supports: list[SupportRecord] = field(default_factory=list)
    defenses: list[DefenseRecord] = field(default_factory=list)
    critics: list[CriticRecord] = field(default_factory=list)
    free_comment: Optional[FreeCommentRecord] = None
    consensus_reached: bool = False
    round_number: int = 1
    timestamp: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "speaker_id": self.speaker_id,
            "supports": [s.to_dict() for s in self.supports],
            "defenses": [d.to_dict() for d in self.defenses],
            "critics": [c.to_dict() for c in self.critics],
            "free_comment": self.free_comment.to_dict() if self.free_comment else None,
            "consensus_reached": self.consensus_reached,
            "round_number": self.round_number,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> DiscussionResponse:
        """Create from dictionary (JSON deserialization)."""
        supports = [SupportRecord.from_dict(s) for s in data.get("supports", [])]
        defenses = [DefenseRecord.from_dict(d) for d in data.get("defenses", [])]
        critics = [CriticRecord.from_dict(c) for c in data.get("critics", [])]

        free_comment_data = data.get("free_comment")
        free_comment = (
            FreeCommentRecord.from_dict(free_comment_data)
            if free_comment_data
            else None
        )

        timestamp_value = data.get("timestamp")
        timestamp = (
            datetime.fromisoformat(timestamp_value)
            if timestamp_value and isinstance(timestamp_value, str)
            else None
        )

        return cls(
            speaker_id=data["speaker_id"],
            supports=supports,
            defenses=defenses,
            critics=critics,
            free_comment=free_comment,
            consensus_reached=data.get("consensus_reached", False),
            round_number=data.get("round_number", 1),
            timestamp=timestamp,
        )


@dataclass
class ParticipantContext:
    """Contextual information for a participant during discussion.

    This is built by DiscussionContextBuilder to provide each participant
    with a personalized view of the discussion state.

    Attributes:
        identity: The participant's identity
        undefended_attacks: Attacks on this participant not yet defended
        defended_attacks: Attacks already defended (avoid duplicate defense)
        supports_received: Support statements received
        mentions_received: Free comments mentioning this participant
        my_attacks: Attacks this participant has made
        my_supports: Supports this participant has given
        my_defenses: Defenses this participant has made
    """
    identity: ParticipantIdentity
    undefended_attacks: list[CriticRecord] = field(default_factory=list)
    defended_attacks: list[CriticRecord] = field(default_factory=list)
    supports_received: list[SupportRecord] = field(default_factory=list)
    mentions_received: list[FreeCommentRecord] = field(default_factory=list)
    my_attacks: list[CriticRecord] = field(default_factory=list)
    my_supports: list[SupportRecord] = field(default_factory=list)
    my_defenses: list[DefenseRecord] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "identity": self.identity.to_dict(),
            "undefended_attacks": [a.to_dict() for a in self.undefended_attacks],
            "defended_attacks": [a.to_dict() for a in self.defended_attacks],
            "supports_received": [s.to_dict() for s in self.supports_received],
            "mentions_received": [m.to_dict() for m in self.mentions_received],
            "my_attacks": [a.to_dict() for a in self.my_attacks],
            "my_supports": [s.to_dict() for s in self.my_supports],
            "my_defenses": [d.to_dict() for d in self.my_defenses],
        }

    @classmethod
    def from_dict(cls, data: dict) -> ParticipantContext:
        """Create from dictionary (JSON deserialization)."""
        return cls(
            identity=ParticipantIdentity.from_dict(data["identity"]),
            undefended_attacks=[CriticRecord.from_dict(a) for a in data.get("undefended_attacks", [])],
            defended_attacks=[CriticRecord.from_dict(a) for a in data.get("defended_attacks", [])],
            supports_received=[SupportRecord.from_dict(s) for s in data.get("supports_received", [])],
            mentions_received=[FreeCommentRecord.from_dict(m) for m in data.get("mentions_received", [])],
            my_attacks=[CriticRecord.from_dict(a) for a in data.get("my_attacks", [])],
            my_supports=[SupportRecord.from_dict(s) for s in data.get("my_supports", [])],
            my_defenses=[DefenseRecord.from_dict(d) for d in data.get("my_defenses", [])],
        )
