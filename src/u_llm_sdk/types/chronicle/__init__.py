"""Chronicle - Decision-Centric Chronicle for Local, Multi-LLM Workflows.

Chronicle is a local, structured "chronicle" (실록) storage layer that records
*why* a decision was made, *what* actions were executed, and *what* outcomes
happened—so that later analysis can reconstruct causal chains and reuse experience.

Record Types:
    Primary Records (causal backbone):
        - DecisionRecord: Audit-grade strict immutability (frozen)
        - ExecutionRecord: Pragmatic immutability
        - FailureRecord: Core immutable, resolution append-once
        - EvidenceRecord: Fully immutable
        - AmendRecord: Corrections/supplements to DecisionRecord

    Derived Records (regenerable views):
        - BriefingRecord: Deterministic, NO LLM calls
        - InquisitionRecord: Offline-only, LLM-assisted allowed

Supporting Types:
    - RecordType: Enum with ID prefixes
    - SourceReference: Snapshot semantics for decision context
    - SourceKind: Type of source being referenced
    - ErrorFingerprint: Structured key for failure similarity matching
    - BriefingGenerationParams: Parameters for briefing reproducibility

ID Prefix Rules:
    - dec_  : DecisionRecord
    - exec_ : ExecutionRecord
    - fail_ : FailureRecord
    - evid_ : EvidenceRecord
    - brf_  : BriefingRecord
    - inq_  : InquisitionRecord
    - amend_: AmendRecord

Unified Query Axioms:
    1. All results resolve to record_id
    2. Decision is the root of causal chains
    3. Cross-session similarity routes through DejaVu linkage
    4. Success guidance is indexed via FailureRecord.resolution

Usage:
    >>> from u_llm_sdk.types.chronicle import (
    ...     DecisionRecord,
    ...     ExecutionRecord,
    ...     FailureRecord,
    ...     RecordType,
    ...     generate_record_id,
    ... )
    >>>
    >>> # Create a decision
    >>> decision = DecisionRecord.create(
    ...     session_id="sess_123",
    ...     logical_step=1,
    ...     question="Which authentication method?",
    ...     options_considered=["JWT", "Session", "OAuth"],
    ...     chosen_option="JWT",
    ...     rationale="Stateless, better for API",
    ...     sources=[SourceReference.from_git("auth.py", "abc123")],
    ... )
"""

# ID generation and validation
from .ids import (
    RecordType,
    generate_record_id,
    validate_record_id,
    get_record_type,
    extract_event_id,
)

# Source reference
from .source import (
    SourceKind,
    SourceReference,
)

# Primary records
from .records import (
    # Execution outcome enum
    ExecutionOutcome,
    # Error fingerprint
    ErrorFingerprint,
    # Primary record types
    DecisionRecord,
    AmendRecord,
    ExecutionRecord,
    FailureRecord,
    EvidenceRecord,
)

# Derived records
from .derived import (
    BriefingGenerationParams,
    BriefingRecord,
    InquisitionRecord,
)

__all__ = [
    # ID generation and validation
    "RecordType",
    "generate_record_id",
    "validate_record_id",
    "get_record_type",
    "extract_event_id",
    # Source reference
    "SourceKind",
    "SourceReference",
    # Execution outcome
    "ExecutionOutcome",
    # Error fingerprint
    "ErrorFingerprint",
    # Primary records
    "DecisionRecord",
    "AmendRecord",
    "ExecutionRecord",
    "FailureRecord",
    "EvidenceRecord",
    # Derived records
    "BriefingGenerationParams",
    "BriefingRecord",
    "InquisitionRecord",
]
