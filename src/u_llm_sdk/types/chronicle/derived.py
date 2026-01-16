"""Chronicle Derived Records - Regenerable views and offline verification.

Derived Records are views that can be rebuilt from Primary records:
- BriefingRecord: Deterministic, regenerable "injection view" (NO LLM)
- InquisitionRecord: Offline-only selective freeze (LLM-assisted allowed)

Key Distinction:
    BriefingRecord: Pure deterministic (template filling, sorting, extraction)
    InquisitionRecord: May use LLM, but only offline. Verbatim output stored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .ids import RecordType, generate_record_id


# =============================================================================
# BriefingGenerationParams (for reproducibility)
# =============================================================================


@dataclass
class BriefingGenerationParams:
    """Parameters needed to reproduce a briefing from its inputs.

    Even though BriefingRecord is derived (regenerable), we store these
    params to answer "what did the LLM actually see at that moment?"
    without re-running the generation with potentially different algorithms.

    Attributes:
        derived_from: Primary record IDs used as input
        injector_versions: Version of each injector used
        token_budget: Token budget per injector
        generation_timestamp: When the briefing was generated
        filter_thresholds: Thresholds used for filtering (similarity, recency, etc.)
    """

    derived_from: list[str]
    injector_versions: dict[str, str]
    token_budget: dict[str, int]
    generation_timestamp: datetime
    filter_thresholds: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "derived_from": self.derived_from,
            "injector_versions": self.injector_versions,
            "token_budget": self.token_budget,
            "generation_timestamp": self.generation_timestamp.isoformat(),
            "filter_thresholds": self.filter_thresholds,
        }

    @classmethod
    def from_dict(cls, data: dict) -> BriefingGenerationParams:
        """Deserialize from dictionary."""
        return cls(
            derived_from=data["derived_from"],
            injector_versions=data["injector_versions"],
            token_budget=data["token_budget"],
            generation_timestamp=datetime.fromisoformat(data["generation_timestamp"]),
            filter_thresholds=data.get("filter_thresholds", {}),
        )


# =============================================================================
# BriefingRecord (Deterministic, NO LLM)
# =============================================================================


@dataclass
class BriefingRecord:
    """Deterministic, regenerable "injection view" for context handoff.

    IMPORTANT: BriefingRecord generation is PURE DETERMINISTIC.
    - Template filling
    - Sorting
    - Extraction
    - NO LLM calls
    - NO stochasticity

    Storage Semantics:
        `summary` is stored as a PERFORMANCE CACHE, not audit evidence.
        It may be discarded and regenerated at any time from
        `derived_from` + `generation_params`.

        For audit-grade LLM context, use InquisitionRecord.

    Attributes:
        record_id: Unique identifier (prefix: "brf_")
        session_id: Session this briefing belongs to
        created_at: When the briefing was generated
        logical_step: Ordering within session

        derived_from: Primary record IDs used as input
        generation_params: Parameters for reproducibility

        summary: The briefing text (CACHED, regenerable)
        key_decisions: IDs of relevant DecisionRecords
    """

    record_id: str
    session_id: str
    created_at: datetime
    logical_step: int

    derived_from: list[str]
    generation_params: BriefingGenerationParams

    summary: str
    key_decisions: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate constraints."""
        if not self.record_id.startswith(RecordType.BRIEFING.prefix):
            raise ValueError(
                f"BriefingRecord ID must start with '{RecordType.BRIEFING.prefix}'"
            )

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "record_id": self.record_id,
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "logical_step": self.logical_step,
            "derived_from": self.derived_from,
            "generation_params": self.generation_params.to_dict(),
            "summary": self.summary,
            "key_decisions": self.key_decisions,
        }

    @classmethod
    def from_dict(cls, data: dict) -> BriefingRecord:
        """Deserialize from dictionary."""
        return cls(
            record_id=data["record_id"],
            session_id=data["session_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            logical_step=data["logical_step"],
            derived_from=data["derived_from"],
            generation_params=BriefingGenerationParams.from_dict(
                data["generation_params"]
            ),
            summary=data["summary"],
            key_decisions=data.get("key_decisions", []),
        )

    @classmethod
    def create(
        cls,
        session_id: str,
        logical_step: int,
        derived_from: list[str],
        generation_params: BriefingGenerationParams,
        summary: str,
        key_decisions: Optional[list[str]] = None,
    ) -> BriefingRecord:
        """Factory method to create a new BriefingRecord.

        Args:
            session_id: Session this briefing belongs to
            logical_step: Ordering within session
            derived_from: Primary record IDs used as input
            generation_params: Parameters for reproducibility
            summary: The briefing text
            key_decisions: IDs of relevant DecisionRecords

        Returns:
            A new BriefingRecord with generated record_id
        """
        return cls(
            record_id=generate_record_id(RecordType.BRIEFING),
            session_id=session_id,
            created_at=datetime.now(),
            logical_step=logical_step,
            derived_from=derived_from,
            generation_params=generation_params,
            summary=summary,
            key_decisions=key_decisions or [],
        )


# =============================================================================
# InquisitionRecord (Offline-only, LLM-assisted allowed)
# =============================================================================


@dataclass(frozen=True)
class InquisitionRecord:
    """Offline verification / interrogation output. NEVER on hot path.

    InquisitionRecord is an EVIDENCE APPENDIX to a Decision, not a causal root.
    All causal chains still terminate at DecisionRecord (Axiom 2).
    `related_decision_id` links verification results back to the Decision.

    May use LLM assistance (local models or any executor), but always OFFLINE.
    Output is stored verbatim because it is explicitly "selective freeze".

    Enforcement (how "offline-only" is guaranteed):
        InquisitionRecord creation is gated by `InquisitorSession`, a context
        manager that:
        1. Must be explicitly instantiated (not callable from hot-path code)
        2. Logs warning if execution_context != "offline" (soft enforcement)
        3. Optionally refuses execution in "hot-path mode" (strict enforcement)

        This is a POLICY BOUNDARY, not a hard technical lock.
        Intentional bypass is possible but visible in logs and audit trail.

    Attributes:
        record_id: Unique identifier (prefix: "inq_")
        session_id: Session this inquisition belongs to
        created_at: When the inquisition was performed

        related_decision_id: Evidence appendix to this Decision
        derived_from: Record IDs examined during inquisition

        method: "llm_verification", "rule_based", "hybrid"
        model_id: LLM model used (if any)
        prompt_hash: Hash of prompt for reproducibility (if LLM used)

        input_context_ref: RawVault ref (what was shown to verifier)
        llm_output: Stored VERBATIM (selective freeze)

        verdict: "PASS" | "FAIL" | "NEEDS_MORE_EVIDENCE"
        claims: Structured claims from verification
        evidence_refs: Record IDs / raw_refs supporting claims
    """

    record_id: str
    session_id: str
    created_at: datetime

    related_decision_id: str
    derived_from: tuple[str, ...]

    method: str  # "llm_verification" | "rule_based" | "hybrid"
    model_id: Optional[str]
    prompt_hash: Optional[str]

    input_context_ref: Optional[str]  # RawVault ref
    llm_output: str  # Stored verbatim

    verdict: str  # "PASS" | "FAIL" | "NEEDS_MORE_EVIDENCE"
    claims: tuple[str, ...]
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate constraints."""
        if not self.record_id.startswith(RecordType.INQUISITION.prefix):
            raise ValueError(
                f"InquisitionRecord ID must start with '{RecordType.INQUISITION.prefix}'"
            )
        if self.verdict not in ("PASS", "FAIL", "NEEDS_MORE_EVIDENCE"):
            raise ValueError(
                f"InquisitionRecord.verdict must be 'PASS', 'FAIL', or "
                f"'NEEDS_MORE_EVIDENCE', got: {self.verdict}"
            )
        if self.method not in ("llm_verification", "rule_based", "hybrid"):
            raise ValueError(
                f"InquisitionRecord.method must be 'llm_verification', "
                f"'rule_based', or 'hybrid', got: {self.method}"
            )

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "record_id": self.record_id,
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "related_decision_id": self.related_decision_id,
            "derived_from": list(self.derived_from),
            "method": self.method,
            "model_id": self.model_id,
            "prompt_hash": self.prompt_hash,
            "input_context_ref": self.input_context_ref,
            "llm_output": self.llm_output,
            "verdict": self.verdict,
            "claims": list(self.claims),
            "evidence_refs": list(self.evidence_refs),
        }

    @classmethod
    def from_dict(cls, data: dict) -> InquisitionRecord:
        """Deserialize from dictionary."""
        return cls(
            record_id=data["record_id"],
            session_id=data["session_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            related_decision_id=data["related_decision_id"],
            derived_from=tuple(data["derived_from"]),
            method=data["method"],
            model_id=data.get("model_id"),
            prompt_hash=data.get("prompt_hash"),
            input_context_ref=data.get("input_context_ref"),
            llm_output=data["llm_output"],
            verdict=data["verdict"],
            claims=tuple(data.get("claims", [])),
            evidence_refs=tuple(data.get("evidence_refs", [])),
        )

    @classmethod
    def create(
        cls,
        session_id: str,
        related_decision_id: str,
        derived_from: list[str],
        method: str,
        llm_output: str,
        verdict: str,
        claims: Optional[list[str]] = None,
        evidence_refs: Optional[list[str]] = None,
        model_id: Optional[str] = None,
        prompt_hash: Optional[str] = None,
        input_context_ref: Optional[str] = None,
    ) -> InquisitionRecord:
        """Factory method to create a new InquisitionRecord.

        Args:
            session_id: Session this inquisition belongs to
            related_decision_id: Decision being verified
            derived_from: Record IDs examined
            method: Verification method
            llm_output: Output to store verbatim
            verdict: Verification result
            claims: Structured claims
            evidence_refs: Supporting references
            model_id: LLM model used (if any)
            prompt_hash: Hash of prompt (if LLM used)
            input_context_ref: RawVault ref for input context

        Returns:
            A new InquisitionRecord with generated record_id
        """
        return cls(
            record_id=generate_record_id(RecordType.INQUISITION),
            session_id=session_id,
            created_at=datetime.now(),
            related_decision_id=related_decision_id,
            derived_from=tuple(derived_from),
            method=method,
            model_id=model_id,
            prompt_hash=prompt_hash,
            input_context_ref=input_context_ref,
            llm_output=llm_output,
            verdict=verdict,
            claims=tuple(claims or []),
            evidence_refs=tuple(evidence_refs or []),
        )
