"""Chronicle ID generation and validation.

This module defines:
- RecordType enum with ID prefixes
- ID generation functions
- ID validation functions

ID Canonicalization Rule:
    ExecutionRecord.record_id is derived from TimeKeeper event_id:
    record_id = f"exec_{event_id}"

    This ensures:
    1. RawVault blobs are retrievable via record_id transformation
    2. Unified Query Axiom 1 is satisfied
    3. No collision between record types via prefixes
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Optional


class RecordType(Enum):
    """Chronicle record types with their ID prefixes.

    Each record type has a unique prefix to prevent ID collisions
    and enable type identification from ID alone.
    """

    DECISION = "dec"
    EXECUTION = "exec"
    FAILURE = "fail"
    EVIDENCE = "evid"
    BRIEFING = "brf"
    INQUISITION = "inq"
    AMEND = "amend"

    @property
    def prefix(self) -> str:
        """Get the ID prefix for this record type."""
        return f"{self.value}_"


# Mapping from prefix to RecordType for validation
_PREFIX_TO_TYPE: dict[str, RecordType] = {
    rt.prefix: rt for rt in RecordType
}


def generate_record_id(record_type: RecordType, event_id: Optional[str] = None) -> str:
    """Generate a new record ID with the appropriate prefix.

    Args:
        record_type: The type of record to generate an ID for
        event_id: Optional event ID to use (for ExecutionRecord canonicalization).
                  If not provided, generates a new UUID.

    Returns:
        A prefixed record ID (e.g., "dec_abc123", "exec_event456")

    Examples:
        >>> generate_record_id(RecordType.DECISION)
        'dec_550e8400-e29b-41d4-a716-446655440000'

        >>> generate_record_id(RecordType.EXECUTION, "event123")
        'exec_event123'
    """
    if event_id is not None:
        # Use provided event_id (for TimeKeeper canonicalization)
        return f"{record_type.prefix}{event_id}"
    else:
        # Generate new UUID
        return f"{record_type.prefix}{uuid.uuid4()}"


def validate_record_id(record_id: str, expected_type: Optional[RecordType] = None) -> bool:
    """Validate a record ID format and optionally check its type.

    Args:
        record_id: The record ID to validate
        expected_type: If provided, also verify the ID matches this type

    Returns:
        True if the ID is valid (and matches expected_type if provided)

    Examples:
        >>> validate_record_id("dec_abc123")
        True

        >>> validate_record_id("dec_abc123", RecordType.DECISION)
        True

        >>> validate_record_id("dec_abc123", RecordType.EXECUTION)
        False

        >>> validate_record_id("invalid")
        False
    """
    if not record_id or not isinstance(record_id, str):
        return False

    # Check if any valid prefix matches
    for prefix, record_type in _PREFIX_TO_TYPE.items():
        if record_id.startswith(prefix):
            # Has valid prefix, check if there's content after prefix
            if len(record_id) <= len(prefix):
                return False

            # If expected_type specified, verify match
            if expected_type is not None and record_type != expected_type:
                return False

            return True

    return False


def get_record_type(record_id: str) -> Optional[RecordType]:
    """Extract the record type from a record ID.

    Args:
        record_id: The record ID to analyze

    Returns:
        The RecordType if valid, None otherwise

    Examples:
        >>> get_record_type("dec_abc123")
        RecordType.DECISION

        >>> get_record_type("exec_event456")
        RecordType.EXECUTION

        >>> get_record_type("invalid")
        None
    """
    if not record_id or not isinstance(record_id, str):
        return None

    for prefix, record_type in _PREFIX_TO_TYPE.items():
        if record_id.startswith(prefix):
            return record_type

    return None


def extract_event_id(execution_record_id: str) -> Optional[str]:
    """Extract the original event_id from an ExecutionRecord ID.

    This reverses the canonicalization: exec_{event_id} -> event_id

    Args:
        execution_record_id: An ExecutionRecord ID (must start with "exec_")

    Returns:
        The original event_id, or None if not a valid ExecutionRecord ID

    Examples:
        >>> extract_event_id("exec_event123")
        'event123'

        >>> extract_event_id("dec_abc123")
        None
    """
    prefix = RecordType.EXECUTION.prefix
    if execution_record_id and execution_record_id.startswith(prefix):
        return execution_record_id[len(prefix):]
    return None
