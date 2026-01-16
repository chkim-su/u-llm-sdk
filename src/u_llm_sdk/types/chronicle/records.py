"""Chronicle Primary Records - The causal backbone of Chronicle.

Primary Records are the source of truth for causal chains:
- DecisionRecord: Audit-grade strict immutability (frozen dataclass)
- ExecutionRecord: Pragmatic immutability (core immutable, linkage mutable)
- FailureRecord: Core immutable, resolution append-once, clustering mutable
- EvidenceRecord: Fully immutable
- AmendRecord: Corrections/supplements to DecisionRecord (immutable)

Field Mutability Categories:
- Core (immutable): Never change after creation
- Append-once: Set exactly once later (e.g., resolution)
- Mutable: Background processes may update (e.g., clustering IDs)

Decision-Centric Audit Boundary:
    DecisionRecord is the root of all causal chains. Every ExecutionRecord
    traces back to a DecisionRecord via decision_id. This is Axiom 2.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from enum import Enum

from .ids import RecordType, generate_record_id
from .source import SourceReference


# =============================================================================
# ExecutionOutcome (aligned with MV-rag Outcome enum)
# =============================================================================


class ExecutionOutcome(str, Enum):
    """Outcome classification for executions.

    Aligned with MV-rag's Outcome enum for seamless integration.
    This is the semantic classification beyond just exit_code.

    Values:
        SUCCESS: Completed successfully (exit_code == 0)
        PARTIAL: Partially completed (soft limit exceeded but continued)
        FAILURE: Failed (expected failure path, exit_code != 0)
        ERROR: Unexpected error (exception during execution)
        TIMEOUT: Timed out
        ABORT: User/system/policy aborted (hard block)
    """

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"
    ERROR = "error"
    TIMEOUT = "timeout"
    ABORT = "abort"

    @classmethod
    def from_exit_code(cls, exit_code: int) -> "ExecutionOutcome":
        """Derive basic outcome from exit code.

        For richer classification (PARTIAL, ERROR, TIMEOUT, ABORT),
        use the explicit constructor.
        """
        return cls.SUCCESS if exit_code == 0 else cls.FAILURE


# =============================================================================
# ErrorFingerprint (for FailureRecord similarity matching)
# =============================================================================


@dataclass
class ErrorFingerprint:
    """Structured key for similarity matching without full-text search.

    Enables ExperienceInjector to find similar failures efficiently
    even though Chronicle doesn't support full-text search.

    Attributes:
        error_type: Error class name (e.g., "TypeError", "ConnectionError")
        error_code: Error code if present (e.g., "ECONNREFUSED", "404")
        normalized_message: Error message with variables removed
        stack_top_3: Top 3 function names from stack trace
        affected_file: Primary file involved (if identifiable)

    Normalization Rules for `normalized_message`:
        1. Replace file paths with <PATH>
        2. Replace line numbers with <LINE>
        3. Replace timestamps with <TIMESTAMP>
        4. Replace UUIDs/hashes with <ID>
        5. Preserve error type prefix (e.g., "TypeError:")
    """

    error_type: str
    error_code: Optional[str]
    normalized_message: str
    stack_top_3: list[str]
    affected_file: Optional[str]

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "error_type": self.error_type,
            "error_code": self.error_code,
            "normalized_message": self.normalized_message,
            "stack_top_3": self.stack_top_3,
            "affected_file": self.affected_file,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ErrorFingerprint:
        """Deserialize from dictionary."""
        return cls(
            error_type=data["error_type"],
            error_code=data.get("error_code"),
            normalized_message=data["normalized_message"],
            stack_top_3=data.get("stack_top_3", []),
            affected_file=data.get("affected_file"),
        )

    @classmethod
    def normalize_message(cls, raw_message: str) -> str:
        """Normalize an error message by replacing variable parts.

        Applies the 5 normalization rules (order matters for correctness):
        1. Timestamps -> <TIMESTAMP> (must be before line numbers)
        2. UUIDs/hashes -> <ID>
        3. File paths -> <PATH>
        4. Line numbers -> <LINE>
        5. Preserve error type prefix

        Args:
            raw_message: The original error message

        Returns:
            Normalized message suitable for similarity matching
        """
        msg = raw_message

        # Rule 1: Replace timestamps FIRST (before line numbers can match parts)
        # ISO format: 2024-01-15T10:30:00Z or 2024-01-15T10:30:00.123+09:00
        msg = re.sub(
            r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?',
            '<TIMESTAMP>',
            msg
        )
        # Common format: 01/15/2024 10:30:00
        msg = re.sub(r'\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2}', '<TIMESTAMP>', msg)

        # Rule 2: Replace UUIDs and hex hashes
        msg = re.sub(
            r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b',
            '<ID>',
            msg
        )
        # Only replace long hex strings (12+ chars) to avoid false positives
        msg = re.sub(r'\b[0-9a-fA-F]{12,}\b', '<ID>', msg)

        # Rule 3: Replace file paths (Unix and Windows)
        # Matches /path/to/file.py or C:\path\to\file.py
        msg = re.sub(r'[A-Za-z]?:?[\\/][\w\-./\\]+\.\w+', '<PATH>', msg)

        # Rule 4: Replace line numbers (e.g., "line 42", ":42:", ":42,")
        msg = re.sub(r'\bline\s+\d+', 'line <LINE>', msg, flags=re.IGNORECASE)
        msg = re.sub(r':(\d+)(?=[:,\s]|$)', ':<LINE>', msg)

        # Rule 5: Preserve error type prefix (already preserved by above rules)

        return msg


# =============================================================================
# DecisionRecord (Audit-grade, frozen)
# =============================================================================


@dataclass(frozen=True)
class DecisionRecord:
    """Captures WHY something was decided. Audit-grade strict immutability.

    This is the root of causal chains (Axiom 2). Every ExecutionRecord
    traces back to a DecisionRecord.

    Scope Rule:
        A DecisionRecord is created when INTENT changes, not when each
        action executes. Multiple tool calls under the same intent share
        the same decision_id.

    Intent Identity:
        Two actions share the same intent iff ALL remain unchanged:
        1. question - the problem being solved
        2. boundary_constraints - constraints/limitations defining scope
        3. target_artifacts - the set of files/resources being modified

    Decision Triggers:
        - Explicit consensus (brainstorm completion, voting)
        - User direction (confirms plan, selects approach)
        - Irreversible action approval (file modification, external API call)
        - Proceed-without-asking moments that change state

    Attributes:
        record_id: Unique identifier (prefix: "dec_")
        session_id: Session this decision belongs to
        created_at: When the decision was made
        logical_step: Ordering within session

        question: What needed deciding?
        options_considered: What options were evaluated?
        chosen_option: What was chosen?
        rationale: Why?

        boundary_constraints: Constraints defining scope (for intent identity)
        target_artifacts: Files/resources being modified (for intent identity)

        sources: Context at decision time (MANDATORY, min 1, with snapshot_hash)
        participants: Who made this decision (provider IDs, "user", etc.)

        caused_by: ID of triggering record (DecisionRecord or FailureRecord)
    """

    record_id: str
    session_id: str
    created_at: datetime
    logical_step: int

    question: str
    options_considered: tuple[str, ...]  # Immutable list
    chosen_option: str
    rationale: str

    boundary_constraints: dict  # From BoundaryConstraints.to_dict()
    target_artifacts: tuple[str, ...]  # Immutable list

    sources: tuple[SourceReference, ...]  # MANDATORY, min 1
    participants: tuple[str, ...] = ()

    caused_by: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate constraints."""
        # Validate record_id prefix
        if not self.record_id.startswith(RecordType.DECISION.prefix):
            raise ValueError(
                f"DecisionRecord ID must start with '{RecordType.DECISION.prefix}', "
                f"got: {self.record_id}"
            )

        # Validate sources is not empty
        if not self.sources:
            raise ValueError("DecisionRecord must have at least 1 SourceReference")

        # Validate all sources have snapshot_hash
        for src in self.sources:
            if not src.has_snapshot():
                raise ValueError(
                    f"DecisionRecord sources must have snapshot_hash. "
                    f"Source at '{src.location}' is missing snapshot_hash."
                )

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "record_id": self.record_id,
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "logical_step": self.logical_step,
            "question": self.question,
            "options_considered": list(self.options_considered),
            "chosen_option": self.chosen_option,
            "rationale": self.rationale,
            "boundary_constraints": self.boundary_constraints,
            "target_artifacts": list(self.target_artifacts),
            "sources": [s.to_dict() for s in self.sources],
            "participants": list(self.participants),
            "caused_by": self.caused_by,
        }

    @classmethod
    def from_dict(cls, data: dict) -> DecisionRecord:
        """Deserialize from dictionary."""
        return cls(
            record_id=data["record_id"],
            session_id=data["session_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            logical_step=data["logical_step"],
            question=data["question"],
            options_considered=tuple(data["options_considered"]),
            chosen_option=data["chosen_option"],
            rationale=data["rationale"],
            boundary_constraints=data["boundary_constraints"],
            target_artifacts=tuple(data["target_artifacts"]),
            sources=tuple(SourceReference.from_dict(s) for s in data["sources"]),
            participants=tuple(data.get("participants", [])),
            caused_by=data.get("caused_by"),
        )

    @classmethod
    def create(
        cls,
        session_id: str,
        logical_step: int,
        question: str,
        options_considered: list[str],
        chosen_option: str,
        rationale: str,
        sources: list[SourceReference],
        boundary_constraints: Optional[dict] = None,
        target_artifacts: Optional[list[str]] = None,
        participants: Optional[list[str]] = None,
        caused_by: Optional[str] = None,
    ) -> DecisionRecord:
        """Factory method to create a new DecisionRecord.

        Args:
            session_id: Session this decision belongs to
            logical_step: Ordering within session
            question: What needed deciding?
            options_considered: What options were evaluated?
            chosen_option: What was chosen?
            rationale: Why?
            sources: Context at decision time (min 1, with snapshot_hash)
            boundary_constraints: Constraints defining scope
            target_artifacts: Files/resources being modified
            participants: Who made this decision
            caused_by: ID of triggering record

        Returns:
            A new DecisionRecord with generated record_id
        """
        return cls(
            record_id=generate_record_id(RecordType.DECISION),
            session_id=session_id,
            created_at=datetime.now(),
            logical_step=logical_step,
            question=question,
            options_considered=tuple(options_considered),
            chosen_option=chosen_option,
            rationale=rationale,
            boundary_constraints=boundary_constraints or {},
            target_artifacts=tuple(target_artifacts or []),
            sources=tuple(sources),
            participants=tuple(participants or []),
            caused_by=caused_by,
        )


# =============================================================================
# AmendRecord (Corrections/supplements to DecisionRecord)
# =============================================================================


@dataclass(frozen=True)
class AmendRecord:
    """Append-only addendum to a DecisionRecord. Never modifies original.

    Use for corrections and supplements only. Direction changes (reversals)
    require a NEW DecisionRecord, not an AmendRecord.

    Amendment Types:
        - correction: Fixes factual error (typo, wrong file name).
                      Does NOT change the decision outcome.
        - supplement: Adds later-discovered evidence that SUPPORTS
                      the original decision.

    Attributes:
        record_id: Unique identifier (prefix: "amend_")
        amends: ID of the DecisionRecord being amended
        amendment_type: "correction" or "supplement"
        content: The amendment itself
        created_at: When the amendment was made
    """

    record_id: str
    amends: str  # DecisionRecord ID
    amendment_type: str  # "correction" | "supplement"
    content: str
    created_at: datetime

    def __post_init__(self) -> None:
        """Validate constraints."""
        if not self.record_id.startswith(RecordType.AMEND.prefix):
            raise ValueError(
                f"AmendRecord ID must start with '{RecordType.AMEND.prefix}'"
            )
        if self.amendment_type not in ("correction", "supplement"):
            raise ValueError(
                f"amendment_type must be 'correction' or 'supplement', "
                f"got: {self.amendment_type}"
            )
        if not self.amends.startswith(RecordType.DECISION.prefix):
            raise ValueError(
                f"AmendRecord.amends must reference a DecisionRecord ID, "
                f"got: {self.amends}"
            )

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "record_id": self.record_id,
            "amends": self.amends,
            "amendment_type": self.amendment_type,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> AmendRecord:
        """Deserialize from dictionary."""
        return cls(
            record_id=data["record_id"],
            amends=data["amends"],
            amendment_type=data["amendment_type"],
            content=data["content"],
            created_at=datetime.fromisoformat(data["created_at"]),
        )

    @classmethod
    def create_correction(
        cls,
        decision_id: str,
        content: str,
    ) -> AmendRecord:
        """Create a correction amendment."""
        return cls(
            record_id=generate_record_id(RecordType.AMEND),
            amends=decision_id,
            amendment_type="correction",
            content=content,
            created_at=datetime.now(),
        )

    @classmethod
    def create_supplement(
        cls,
        decision_id: str,
        content: str,
    ) -> AmendRecord:
        """Create a supplement amendment."""
        return cls(
            record_id=generate_record_id(RecordType.AMEND),
            amends=decision_id,
            amendment_type="supplement",
            content=content,
            created_at=datetime.now(),
        )


# =============================================================================
# ExecutionRecord (Pragmatic immutability)
# =============================================================================


@dataclass
class ExecutionRecord:
    """Captures WHAT was executed and its outcome.

    Pragmatic immutability:
    - Core fields (immutable): identity, tool_name, input_args, exit_code,
                               output_summary, decision_id
    - Mutable fields: parent_execution_id (can be linked later)

    ID Canonicalization:
        record_id is derived from TimeKeeper event_id:
        record_id = f"exec_{event_id}"

    Attributes:
        record_id: Unique identifier (prefix: "exec_", derived from event_id)
        session_id: Session this execution belongs to
        created_at: When execution started
        logical_step: Ordering within session

        decision_id: Link to DecisionRecord that triggered this execution
        tool_name: Name of tool executed (e.g., "Bash", "Edit", "Read")
        input_args: Arguments passed to tool
        exit_code: Tool exit code (0 = success)
        outcome: Semantic classification (SUCCESS, PARTIAL, FAILURE, ERROR, TIMEOUT, ABORT)
        duration_ms: Execution time in milliseconds
        output_summary: Summary of output (NOT full output - that's in RawVault)

        parent_execution_id: For nested executions (mutable)
    """

    record_id: str
    session_id: str
    created_at: datetime
    logical_step: int

    decision_id: str
    tool_name: str
    input_args: dict
    exit_code: int
    outcome: ExecutionOutcome
    duration_ms: int
    output_summary: str

    parent_execution_id: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate constraints."""
        if not self.record_id.startswith(RecordType.EXECUTION.prefix):
            raise ValueError(
                f"ExecutionRecord ID must start with '{RecordType.EXECUTION.prefix}'"
            )

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "record_id": self.record_id,
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "logical_step": self.logical_step,
            "decision_id": self.decision_id,
            "tool_name": self.tool_name,
            "input_args": self.input_args,
            "exit_code": self.exit_code,
            "outcome": self.outcome.value,
            "duration_ms": self.duration_ms,
            "output_summary": self.output_summary,
            "parent_execution_id": self.parent_execution_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ExecutionRecord:
        """Deserialize from dictionary."""
        # Handle outcome - derive from exit_code if not present (backward compat)
        outcome_value = data.get("outcome")
        if outcome_value:
            outcome = ExecutionOutcome(outcome_value)
        else:
            outcome = ExecutionOutcome.from_exit_code(data["exit_code"])

        return cls(
            record_id=data["record_id"],
            session_id=data["session_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            logical_step=data["logical_step"],
            decision_id=data["decision_id"],
            tool_name=data["tool_name"],
            input_args=data["input_args"],
            exit_code=data["exit_code"],
            outcome=outcome,
            duration_ms=data["duration_ms"],
            output_summary=data["output_summary"],
            parent_execution_id=data.get("parent_execution_id"),
        )

    @classmethod
    def create(
        cls,
        event_id: str,
        session_id: str,
        logical_step: int,
        decision_id: str,
        tool_name: str,
        input_args: dict,
        exit_code: int,
        duration_ms: int,
        output_summary: str,
        outcome: Optional[ExecutionOutcome] = None,
        parent_execution_id: Optional[str] = None,
    ) -> ExecutionRecord:
        """Factory method to create a new ExecutionRecord.

        Args:
            event_id: TimeKeeper event ID (for canonicalization)
            session_id: Session this execution belongs to
            logical_step: Ordering within session
            decision_id: Link to DecisionRecord
            tool_name: Name of tool executed
            input_args: Arguments passed to tool
            exit_code: Tool exit code
            duration_ms: Execution time in milliseconds
            output_summary: Summary of output
            outcome: Semantic outcome (auto-derived from exit_code if not provided)
            parent_execution_id: For nested executions

        Returns:
            A new ExecutionRecord with canonicalized record_id
        """
        # Auto-derive outcome from exit_code if not explicitly provided
        if outcome is None:
            outcome = ExecutionOutcome.from_exit_code(exit_code)

        return cls(
            record_id=generate_record_id(RecordType.EXECUTION, event_id),
            session_id=session_id,
            created_at=datetime.now(),
            logical_step=logical_step,
            decision_id=decision_id,
            tool_name=tool_name,
            input_args=input_args,
            exit_code=exit_code,
            outcome=outcome,
            duration_ms=duration_ms,
            output_summary=output_summary,
            parent_execution_id=parent_execution_id,
        )

    @property
    def succeeded(self) -> bool:
        """Check if execution succeeded."""
        return self.outcome == ExecutionOutcome.SUCCESS


# =============================================================================
# FailureRecord (Core immutable, resolution append-once, clustering mutable)
# =============================================================================


@dataclass
class FailureRecord:
    """Captures WHAT went wrong and (optionally) WHAT fixed it.

    Field Mutability:
        - Core (immutable): record_id, session_id, created_at, logical_step,
                            execution_id, symptom, cause, error_fingerprint
        - Append-once: resolution, resolution_execution_id
        - Mutable: dejavu_group_id, similarity_score

    Attributes:
        record_id: Unique identifier (prefix: "fail_")
        session_id: Session this failure belongs to
        created_at: When failure was recorded
        logical_step: Ordering within session

        execution_id: Which execution failed
        symptom: What failed? (human-readable)
        cause: Why? (if known)
        error_fingerprint: Structured key for similarity matching

        resolution: How was it fixed? (append-once)
        resolution_execution_id: Which execution fixed it? (append-once)

        dejavu_group_id: DejaVu cluster ID (mutable, background analysis)
        similarity_score: Similarity to cluster centroid (mutable)
    """

    record_id: str
    session_id: str
    created_at: datetime
    logical_step: int

    execution_id: str
    symptom: str
    cause: Optional[str]
    error_fingerprint: ErrorFingerprint

    # Append-once fields
    resolution: Optional[str] = None
    resolution_execution_id: Optional[str] = None

    # Mutable fields (background analysis)
    dejavu_group_id: Optional[str] = None
    similarity_score: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate constraints."""
        if not self.record_id.startswith(RecordType.FAILURE.prefix):
            raise ValueError(
                f"FailureRecord ID must start with '{RecordType.FAILURE.prefix}'"
            )

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "record_id": self.record_id,
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "logical_step": self.logical_step,
            "execution_id": self.execution_id,
            "symptom": self.symptom,
            "cause": self.cause,
            "error_fingerprint": self.error_fingerprint.to_dict(),
            "resolution": self.resolution,
            "resolution_execution_id": self.resolution_execution_id,
            "dejavu_group_id": self.dejavu_group_id,
            "similarity_score": self.similarity_score,
        }

    @classmethod
    def from_dict(cls, data: dict) -> FailureRecord:
        """Deserialize from dictionary."""
        return cls(
            record_id=data["record_id"],
            session_id=data["session_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            logical_step=data["logical_step"],
            execution_id=data["execution_id"],
            symptom=data["symptom"],
            cause=data.get("cause"),
            error_fingerprint=ErrorFingerprint.from_dict(data["error_fingerprint"]),
            resolution=data.get("resolution"),
            resolution_execution_id=data.get("resolution_execution_id"),
            dejavu_group_id=data.get("dejavu_group_id"),
            similarity_score=data.get("similarity_score"),
        )

    @classmethod
    def create(
        cls,
        session_id: str,
        logical_step: int,
        execution_id: str,
        symptom: str,
        error_fingerprint: ErrorFingerprint,
        cause: Optional[str] = None,
    ) -> FailureRecord:
        """Factory method to create a new FailureRecord.

        Args:
            session_id: Session this failure belongs to
            logical_step: Ordering within session
            execution_id: Which execution failed
            symptom: What failed?
            error_fingerprint: Structured key for similarity matching
            cause: Why? (if known)

        Returns:
            A new FailureRecord with generated record_id
        """
        return cls(
            record_id=generate_record_id(RecordType.FAILURE),
            session_id=session_id,
            created_at=datetime.now(),
            logical_step=logical_step,
            execution_id=execution_id,
            symptom=symptom,
            cause=cause,
            error_fingerprint=error_fingerprint,
        )

    def set_resolution(
        self,
        resolution: str,
        resolution_execution_id: Optional[str] = None,
    ) -> None:
        """Set the resolution (append-once).

        Args:
            resolution: How was it fixed?
            resolution_execution_id: Which execution fixed it?

        Raises:
            ValueError: If resolution is already set
        """
        if self.resolution is not None:
            raise ValueError(
                "FailureRecord.resolution is append-once and already set. "
                "Create a new FailureRecord if a different resolution is needed."
            )
        self.resolution = resolution
        self.resolution_execution_id = resolution_execution_id

    @property
    def is_resolved(self) -> bool:
        """Check if this failure has been resolved."""
        return self.resolution is not None


# =============================================================================
# EvidenceRecord (Fully immutable)
# =============================================================================


@dataclass(frozen=True)
class EvidenceRecord:
    """Immutable pointer to raw artifacts (logs, diffs, traces).

    EvidenceRecord points to data stored in RawVault (filesystem / blob storage).
    It does NOT store the actual content, only a reference.

    Attributes:
        record_id: Unique identifier (prefix: "evid_")
        session_id: Session this evidence belongs to
        created_at: When evidence was recorded
        logical_step: Ordering within session

        kind: Type of evidence ("stdout", "stderr", "patch", "trace", etc.)
        raw_ref: RawVault pointer (path / blob id)
        related_record_id: Which record this evidence relates to
                           (DecisionRecord, ExecutionRecord, or FailureRecord)
    """

    record_id: str
    session_id: str
    created_at: datetime
    logical_step: int

    kind: str
    raw_ref: str
    related_record_id: str

    def __post_init__(self) -> None:
        """Validate constraints."""
        if not self.record_id.startswith(RecordType.EVIDENCE.prefix):
            raise ValueError(
                f"EvidenceRecord ID must start with '{RecordType.EVIDENCE.prefix}'"
            )

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "record_id": self.record_id,
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "logical_step": self.logical_step,
            "kind": self.kind,
            "raw_ref": self.raw_ref,
            "related_record_id": self.related_record_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> EvidenceRecord:
        """Deserialize from dictionary."""
        return cls(
            record_id=data["record_id"],
            session_id=data["session_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            logical_step=data["logical_step"],
            kind=data["kind"],
            raw_ref=data["raw_ref"],
            related_record_id=data["related_record_id"],
        )

    @classmethod
    def create(
        cls,
        session_id: str,
        logical_step: int,
        kind: str,
        raw_ref: str,
        related_record_id: str,
    ) -> EvidenceRecord:
        """Factory method to create a new EvidenceRecord.

        Args:
            session_id: Session this evidence belongs to
            logical_step: Ordering within session
            kind: Type of evidence
            raw_ref: RawVault pointer
            related_record_id: Which record this evidence relates to

        Returns:
            A new EvidenceRecord with generated record_id
        """
        return cls(
            record_id=generate_record_id(RecordType.EVIDENCE),
            session_id=session_id,
            created_at=datetime.now(),
            logical_step=logical_step,
            kind=kind,
            raw_ref=raw_ref,
            related_record_id=related_record_id,
        )
