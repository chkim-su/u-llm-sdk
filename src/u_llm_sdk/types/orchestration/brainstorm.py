"""LLM Types - Brainstorm and Consensus Types.

Configuration and data structures for brainstorming sessions
and consensus building.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from ..config import Provider


@dataclass
class BrainstormConfig:
    """Configuration for brainstorming sessions.

    Attributes:
        max_rounds: Maximum discussion rounds (default: 3)
        consensus_method: How to determine consensus
        consensus_threshold: Required agreement level (default: 0.67 = 2/3)
        low_agreement_threshold: Below this, escalate to user
        preserve_full_discussion: Keep entire discussion (no summarization)
    """
    max_rounds: int = 3
    consensus_method: Literal["majority", "unanimous"] = "majority"
    consensus_threshold: float = 0.67
    low_agreement_threshold: float = 0.4
    preserve_full_discussion: bool = True

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "max_rounds": self.max_rounds,
            "consensus_method": self.consensus_method,
            "consensus_threshold": self.consensus_threshold,
            "low_agreement_threshold": self.low_agreement_threshold,
            "preserve_full_discussion": self.preserve_full_discussion,
        }

    @classmethod
    def from_dict(cls, data: dict) -> BrainstormConfig:
        """Create from dictionary (JSON deserialization)."""
        return cls(
            max_rounds=data.get("max_rounds", 3),
            consensus_method=data.get("consensus_method", "majority"),
            consensus_threshold=data.get("consensus_threshold", 0.67),
            low_agreement_threshold=data.get("low_agreement_threshold", 0.4),
            preserve_full_discussion=data.get("preserve_full_discussion", True),
        )


@dataclass
class ParticipantInput:
    """Input from a brainstorm participant.

    Attributes:
        provider: Which LLM provider provided this input
        analysis: The participant's analysis
        position: Their position/stance
        supporting_evidence: Evidence supporting their position
        concerns: Any concerns raised
        proposed_approach: Their proposed approach
    """
    provider: Provider
    analysis: str
    position: str
    supporting_evidence: list[str] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)
    proposed_approach: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "provider": self.provider.value,
            "analysis": self.analysis,
            "position": self.position,
            "supporting_evidence": self.supporting_evidence,
            "concerns": self.concerns,
            "proposed_approach": self.proposed_approach,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ParticipantInput:
        """Create from dictionary (JSON deserialization)."""
        provider_value = data["provider"]
        provider = Provider(provider_value) if isinstance(provider_value, str) else provider_value

        return cls(
            provider=provider,
            analysis=data["analysis"],
            position=data["position"],
            supporting_evidence=data.get("supporting_evidence", []),
            concerns=data.get("concerns", []),
            proposed_approach=data.get("proposed_approach", ""),
        )


@dataclass
class DiscussionEntry:
    """A single entry in the discussion log.

    Attributes:
        timestamp: When this entry was created
        speaker: Which provider spoke
        message_type: Type of message (opinion, rebuttal, etc.)
        content: The message content
        references: References to other entries or sources
    """
    timestamp: datetime
    speaker: Provider
    message_type: Literal["opinion", "rebuttal", "support", "question", "answer"]
    content: str
    references: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "speaker": self.speaker.value,
            "message_type": self.message_type,
            "content": self.content,
            "references": self.references,
        }

    @classmethod
    def from_dict(cls, data: dict) -> DiscussionEntry:
        """Create from dictionary (JSON deserialization)."""
        timestamp_value = data["timestamp"]
        timestamp = (
            datetime.fromisoformat(timestamp_value)
            if isinstance(timestamp_value, str)
            else timestamp_value
        )

        speaker_value = data["speaker"]
        speaker = Provider(speaker_value) if isinstance(speaker_value, str) else speaker_value

        return cls(
            timestamp=timestamp,
            speaker=speaker,
            message_type=data["message_type"],
            content=data["content"],
            references=data.get("references", []),
        )


@dataclass
class DissentingView:
    """A dissenting view from a participant.

    Attributes:
        provider: Which provider dissents
        position: Their dissenting position
        reasoning: Why they dissent
    """
    provider: Provider
    position: str
    reasoning: str

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "provider": self.provider.value,
            "position": self.position,
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, data: dict) -> DissentingView:
        """Create from dictionary (JSON deserialization)."""
        provider_value = data["provider"]
        provider = Provider(provider_value) if isinstance(provider_value, str) else provider_value

        return cls(
            provider=provider,
            position=data["position"],
            reasoning=data["reasoning"],
        )


@dataclass
class ConsensusEvaluation:
    """Evaluation of consensus level in a discussion.

    Attributes:
        agreement_level: Numeric agreement level (0.0-1.0)
        agreement_category: Categorical assessment (high/medium/low)
        majority_position: The majority position
        dissenting_views: List of dissenting views
        recommendation: Recommended action
    """
    agreement_level: float
    agreement_category: Literal["high", "medium", "low"]
    majority_position: str
    dissenting_views: list[DissentingView] = field(default_factory=list)
    recommendation: Literal["proceed", "escalate_to_user", "another_round"] = "proceed"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "agreement_level": self.agreement_level,
            "agreement_category": self.agreement_category,
            "majority_position": self.majority_position,
            "dissenting_views": [d.to_dict() for d in self.dissenting_views],
            "recommendation": self.recommendation,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ConsensusEvaluation:
        """Create from dictionary (JSON deserialization)."""
        dissenting_views = [
            DissentingView.from_dict(d) for d in data.get("dissenting_views", [])
        ]

        return cls(
            agreement_level=data["agreement_level"],
            agreement_category=data["agreement_category"],
            majority_position=data["majority_position"],
            dissenting_views=dissenting_views,
            recommendation=data.get("recommendation", "proceed"),
        )


@dataclass
class ConsensusResult:
    """Final result of a consensus process.

    Attributes:
        success: Whether consensus was reached
        final_decision: The final decision made
        vote_breakdown: How each provider voted
        discussion_summary: Master orchestrator's summary
        full_discussion_log: Complete discussion (preserved, not summarized)
        escalated_to_user: Whether escalation to user occurred
        user_questions: Questions posed to user (if escalated)
    """
    success: bool
    final_decision: str
    vote_breakdown: dict
    discussion_summary: str
    full_discussion_log: list[DiscussionEntry] = field(default_factory=list)
    escalated_to_user: bool = False
    user_questions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "final_decision": self.final_decision,
            "vote_breakdown": self.vote_breakdown,
            "discussion_summary": self.discussion_summary,
            "full_discussion_log": [e.to_dict() for e in self.full_discussion_log],
            "escalated_to_user": self.escalated_to_user,
            "user_questions": self.user_questions,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ConsensusResult:
        """Create from dictionary (JSON deserialization)."""
        full_discussion_log = [
            DiscussionEntry.from_dict(e) for e in data.get("full_discussion_log", [])
        ]

        return cls(
            success=data["success"],
            final_decision=data["final_decision"],
            vote_breakdown=data.get("vote_breakdown", {}),
            discussion_summary=data.get("discussion_summary", ""),
            full_discussion_log=full_discussion_log,
            escalated_to_user=data.get("escalated_to_user", False),
            user_questions=data.get("user_questions", []),
        )
