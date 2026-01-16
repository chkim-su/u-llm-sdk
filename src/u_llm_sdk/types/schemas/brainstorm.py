"""Brainstorm Schema - Domain schema for multi-LLM brainstorming outputs.

This schema defines the expected output structure for brainstorming sessions.
It validates LLM responses against the DiscussionResponse format defined in
llm_types.orchestration.discussion.

Integration with MV-RAG:
    BrainstormSchema defines WHAT output structure is expected.
    MV-RAG observes HOW each LLM behaved (tokens, duration, tool calls).
    These are independent layers - MV-RAG works regardless of domain schema.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from .base import (
    BaseDomainSchema,
    SchemaField,
    ValidationResult,
    ValidationSeverity,
    register_schema,
)


class BrainstormOutputType(Enum):
    """Types of brainstorm output structures.

    Different phases of brainstorming have different output requirements.
    """

    PREPARATION = "preparation"  # Initial analysis and position
    DISCUSSION = "discussion"    # Structured response with interactions
    CONSENSUS = "consensus"      # Final consensus evaluation


class BrainstormSchema(BaseDomainSchema):
    """Domain schema for multi-LLM brainstorming.

    This schema validates LLM outputs during brainstorming sessions.
    It supports different output types based on the brainstorm phase:

        - PREPARATION: Initial position and analysis
        - DISCUSSION: Structured interactions (support/defend/criticize)
        - CONSENSUS: Final consensus evaluation

    Usage:
        >>> from u_llm_sdk.types.schemas import BrainstormSchema
        >>> schema = BrainstormSchema(output_type=BrainstormOutputType.DISCUSSION)
        >>> result = schema.validate(llm_output)
        >>> if result.valid:
        ...     # Use result.coerced_data
    """

    _name = "brainstorm"
    _version = "1.0.0"

    def __init__(
        self,
        output_type: BrainstormOutputType = BrainstormOutputType.DISCUSSION,
    ):
        """Initialize brainstorm schema.

        Args:
            output_type: Type of brainstorm output expected
        """
        self._output_type = output_type
        super().__init__()

    @property
    def output_type(self) -> BrainstormOutputType:
        """Current output type for this schema instance."""
        return self._output_type

    def _define_fields(self) -> list[SchemaField]:
        """Define schema fields based on output type."""
        if self._output_type == BrainstormOutputType.PREPARATION:
            return self._preparation_fields()
        elif self._output_type == BrainstormOutputType.DISCUSSION:
            return self._discussion_fields()
        elif self._output_type == BrainstormOutputType.CONSENSUS:
            return self._consensus_fields()
        return []

    def _preparation_fields(self) -> list[SchemaField]:
        """Fields for preparation phase output."""
        return [
            SchemaField(
                name="analysis",
                type_hint="str",
                required=True,
                description="Analysis of the topic from this participant's perspective",
            ),
            SchemaField(
                name="position",
                type_hint="str",
                required=True,
                description="Initial position/stance on the topic",
            ),
            SchemaField(
                name="key_points",
                type_hint="list[str]",
                required=True,
                description="Key points supporting the position",
            ),
            SchemaField(
                name="concerns",
                type_hint="list[str]",
                required=False,
                description="Concerns or potential counterarguments",
                default=[],
            ),
            SchemaField(
                name="evidence",
                type_hint="list[str]",
                required=False,
                description="Evidence or references supporting position",
                default=[],
            ),
        ]

    def _discussion_fields(self) -> list[SchemaField]:
        """Fields for discussion phase output (DiscussionResponse format)."""
        return [
            SchemaField(
                name="speaker_id",
                type_hint="str",
                required=True,
                description="Full ID of the speaker (e.g., 'gemini-2.5-pro-001')",
            ),
            SchemaField(
                name="supports",
                type_hint="list[dict]",
                required=False,
                description="List of support records (support_id, target_id, reason)",
                default=[],
            ),
            SchemaField(
                name="defenses",
                type_hint="list[dict]",
                required=False,
                description="List of defense records against attacks",
                default=[],
            ),
            SchemaField(
                name="critics",
                type_hint="list[dict]",
                required=False,
                description="List of criticism records",
                default=[],
            ),
            SchemaField(
                name="free_comment",
                type_hint="dict",
                required=False,
                description="Optional free-form comment",
                default=None,
            ),
            SchemaField(
                name="consensus_reached",
                type_hint="bool",
                required=False,
                description="Self-assessment of whether consensus is reached",
                default=False,
            ),
        ]

    def _consensus_fields(self) -> list[SchemaField]:
        """Fields for consensus evaluation output."""
        return [
            SchemaField(
                name="consensus_level",
                type_hint="float",
                required=True,
                description="Consensus level (0.0-1.0)",
            ),
            SchemaField(
                name="final_decision",
                type_hint="str",
                required=True,
                description="Final consensus decision/recommendation",
            ),
            SchemaField(
                name="supporting_participants",
                type_hint="list[str]",
                required=True,
                description="List of participant IDs supporting the decision",
            ),
            SchemaField(
                name="dissenting_views",
                type_hint="list[dict]",
                required=False,
                description="List of dissenting views (participant_id, argument)",
                default=[],
            ),
            SchemaField(
                name="key_agreements",
                type_hint="list[str]",
                required=False,
                description="Key points all participants agreed on",
                default=[],
            ),
            SchemaField(
                name="unresolved_issues",
                type_hint="list[str]",
                required=False,
                description="Issues that remained unresolved",
                default=[],
            ),
        ]

    def validate(self, data: dict) -> ValidationResult:
        """Validate brainstorm output.

        Extends base validation with brainstorm-specific checks:
            - Speaker ID format validation (DISCUSSION)
            - Consensus level range check (CONSENSUS)
            - Interaction record structure validation

        Args:
            data: Raw output data to validate

        Returns:
            ValidationResult with validation status
        """
        # Run base validation first
        result = super().validate(data)

        # Brainstorm-specific validation
        if self._output_type == BrainstormOutputType.DISCUSSION:
            self._validate_discussion_specific(data, result)
        elif self._output_type == BrainstormOutputType.CONSENSUS:
            self._validate_consensus_specific(data, result)

        return result

    def _validate_discussion_specific(
        self,
        data: dict,
        result: ValidationResult,
    ) -> None:
        """Discussion-specific validation.

        Args:
            data: Raw data
            result: ValidationResult to update
        """
        # Validate speaker_id format
        speaker_id = data.get("speaker_id", "")
        if speaker_id and not self._is_valid_speaker_id(speaker_id):
            result.add_issue(
                ValidationSeverity.WARNING,
                "speaker_id",
                f"speaker_id '{speaker_id}' doesn't match expected format "
                "(provider-model-id)",
            )

        # Validate interaction records structure
        for field_name in ["supports", "defenses", "critics"]:
            records = data.get(field_name, [])
            if records and isinstance(records, list):
                for i, record in enumerate(records):
                    if not isinstance(record, dict):
                        result.add_issue(
                            ValidationSeverity.WARNING,
                            field_name,
                            f"{field_name}[{i}] should be a dict, got {type(record).__name__}",
                        )

    def _validate_consensus_specific(
        self,
        data: dict,
        result: ValidationResult,
    ) -> None:
        """Consensus-specific validation.

        Args:
            data: Raw data
            result: ValidationResult to update
        """
        # Validate consensus_level range
        consensus_level = data.get("consensus_level")
        if consensus_level is not None:
            if not isinstance(consensus_level, (int, float)):
                result.add_issue(
                    ValidationSeverity.ERROR,
                    "consensus_level",
                    f"consensus_level must be a number, got {type(consensus_level).__name__}",
                )
            elif not 0.0 <= consensus_level <= 1.0:
                result.add_issue(
                    ValidationSeverity.WARNING,
                    "consensus_level",
                    f"consensus_level {consensus_level} outside expected range [0.0, 1.0]",
                )

    def _is_valid_speaker_id(self, speaker_id: str) -> bool:
        """Check if speaker_id matches expected format.

        Expected format: {provider}-{model}-{id}
        Examples: gemini-2.5-pro-001, claude-opus-4-002
        """
        parts = speaker_id.split("-")
        return len(parts) >= 3

    def get_prompt_guidance(self) -> Optional[str]:
        """Generate prompt guidance specific to output type."""
        if self._output_type == BrainstormOutputType.DISCUSSION:
            return self._discussion_prompt_guidance()
        return super().get_prompt_guidance()

    def _discussion_prompt_guidance(self) -> str:
        """Specific guidance for discussion output."""
        return """Please format your response as JSON with the following structure:
```json
{
  "speaker_id": "<your-full-id>",  // e.g., "gemini-2.5-pro-001"
  "supports": [
    {
      "support_id": "<unique-id>",  // e.g., "sup-001"
      "target_id": "<participant-id>",  // Who you're supporting
      "target_statement_id": "<stmt-id>",  // Which statement
      "reason": "<why you support>"
    }
  ],
  "defenses": [
    {
      "defense_id": "<unique-id>",  // e.g., "def-001"
      "attack_id": "<atk-id>",  // Which attack you're defending against
      "attacker_id": "<who-attacked>",
      "rebuttal": "<your defense>"
    }
  ],
  "critics": [
    {
      "attack_id": "<unique-id>",  // e.g., "atk-001"
      "target_id": "<who-you-criticize>",
      "target_statement_id": "<stmt-id>",
      "criticism": "<your criticism>"
    }
  ],
  "free_comment": {  // Optional
    "comment_id": "<fc-001>",
    "content": "<general observation>",
    "mentions": ["<participant-ids>"]
  },
  "consensus_reached": false  // Your assessment
}
```

Important:
- Use precise IDs when referencing other participants or statements
- Only include non-empty arrays (omit empty supports/defenses/critics)
- Set consensus_reached to true if you believe consensus has been reached"""

    def to_dict(self) -> dict:
        """Convert schema definition to dictionary."""
        base = super().to_dict()
        base["output_type"] = self._output_type.value
        return base

    @classmethod
    def from_dict(cls, data: dict) -> "BrainstormSchema":
        """Create schema instance from dictionary."""
        output_type = BrainstormOutputType(
            data.get("output_type", BrainstormOutputType.DISCUSSION.value)
        )
        return cls(output_type=output_type)

    def with_output_type(self, output_type: BrainstormOutputType) -> "BrainstormSchema":
        """Create new instance with different output type.

        Useful for switching between phases within a session.

        Args:
            output_type: New output type

        Returns:
            New BrainstormSchema instance
        """
        return BrainstormSchema(output_type=output_type)


# Auto-register on import
try:
    register_schema(BrainstormSchema())
except ValueError:
    # Already registered (e.g., during reload)
    pass
