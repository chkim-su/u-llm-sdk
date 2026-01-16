"""Scribe type definitions.

This module defines the core data structures for the Scribe system,
including the two-layer visibility model (public vs sealed).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


# =============================================================================
# Enums
# =============================================================================


class ScribeType(str, Enum):
    """Types of knowledge items the Scribe manages.

    Each type corresponds to a "section" in the injected context.
    """

    REPO_MAP = "repo_map"
    """Project structure overview (INDEX level)."""

    CONVENTION = "convention"
    """Coding rules, naming conventions, patterns."""

    DECISION = "decision"
    """Confirmed decisions with rationale."""

    CONSTRAINT = "constraint"
    """Scope, boundaries, compatibility requirements."""

    PLAN_NODE = "plan_node"
    """Implementation plan units."""

    RISK = "risk"
    """Identified risks and mitigations."""

    REVIEW_FINDING = "review_finding"
    """Review results and verdicts."""

    FAILURE_SIGNATURE = "failure_sig"
    """Error patterns and resolution hints."""


class ScribeStatus(str, Enum):
    """Lifecycle status of a Scribe item."""

    ACTIVE = "active"
    """Currently valid and should be injected."""

    RETRACTED = "retracted"
    """Invalidated/deleted - notice visible, content sealed."""

    SUPERSEDED = "superseded"
    """Replaced by newer version - notice visible, content sealed."""


class Clearance(str, Enum):
    """Information clearance level for injection.

    Controls what parts of ScribeItems are visible.
    """

    DEFAULT = "default"
    """Public layer only: public_notice + public_summary.
    Sealed content is NEVER included.
    """

    ESCALATED = "escalated"
    """Public layer + partial sealed content.
    Triggered by: failure, conflict detection, review needs_revision.
    """

    AUDIT = "audit"
    """Full history including all sealed content.
    For human review or offline analysis (TimeWeaver).
    """


class ChangeReason(str, Enum):
    """Reason for status change (retract/supersede)."""

    OBSOLETE = "obsolete"
    """Information is outdated."""

    INCORRECT = "incorrect"
    """Information was wrong."""

    CONFLICT = "conflict"
    """Conflicts with other information."""

    SCOPE_CHANGE = "scope_change"
    """Scope/requirements changed."""

    SUPERSEDED_BY_NEW = "superseded_by_new"
    """Replaced by updated version."""

    SECURITY = "security"
    """Security-related removal."""

    USER_REQUEST = "user_request"
    """User explicitly requested change."""


# =============================================================================
# Core Data Structure
# =============================================================================


@dataclass
class ScribeItem:
    """A single knowledge item in the Scribe.

    This is the atomic unit of the Scribe's editable state.
    Items have two visibility layers:

    1. Public Layer (always visible in DEFAULT clearance):
       - public_notice: Change facts + reason + links
       - public_summary: Brief description of what this is about

    2. Sealed Layer (only visible in ESCALATED/AUDIT clearance):
       - sealed_payload: Full content, past versions, sensitive details

    Example:
        >>> item = ScribeItem(
        ...     id="conv:auth-pattern",
        ...     type=ScribeType.CONVENTION,
        ...     status=ScribeStatus.ACTIVE,
        ...     public_notice="Authentication uses JWT with refresh tokens.",
        ...     public_summary="JWT auth pattern for all API endpoints.",
        ...     sealed_payload={"full_example": "...", "exceptions": [...]},
        ...     provenance=["chronicle:exec:123", "scip:symbol:AuthService"],
        ... )
    """

    # === Identity (immutable after creation) ===
    id: str
    """Globally unique identifier.
    Format: {type_prefix}:{name}:{optional_qualifier}
    Examples: conv:auth-pattern, risk:sql-injection, plan:step-3
    """

    type: ScribeType
    """What kind of knowledge this represents."""

    # === Lifecycle ===
    status: ScribeStatus = ScribeStatus.ACTIVE
    """Current status - determines visibility behavior."""

    version: int = 1
    """Version number, incremented on supersede."""

    created_at: datetime = field(default_factory=datetime.now)
    """When first created."""

    updated_at: datetime = field(default_factory=datetime.now)
    """When last modified."""

    updated_by: str = ""
    """Which phase/agent made the last change.
    Examples: 'phase:prepare', 'agent:codex', 'user:manual'
    """

    # === Public Layer (always visible) ===
    public_notice: str = ""
    """Change notice - always shown even if item is retracted/superseded.

    For ACTIVE items: Brief description of what this is.
    For RETRACTED: "❌ Retracted: {reason}. See {replacement} instead."
    For SUPERSEDED: "📝 Updated: {reason}. New version: {new_id}"

    This is NEVER empty. Even active items should have a notice.
    """

    public_summary: str = ""
    """1-2 sentence summary of what this item is about.

    Should provide enough context that the model understands
    the domain/scope without needing sealed details.

    Example: "Authentication pattern for API endpoints using JWT."
    """

    # === Sealed Layer (only visible on escalation) ===
    sealed_payload: dict[str, Any] = field(default_factory=dict)
    """Full content, examples, detailed explanations.

    Structure varies by ScribeType. Examples:
    - CONVENTION: {"rules": [...], "examples": [...], "exceptions": [...]}
    - PLAN_NODE: {"steps": [...], "dependencies": [...], "verification": ...}
    - FAILURE_SIGNATURE: {"stack_trace": ..., "root_cause": ..., "fix_hints": [...]}

    NEVER exposed in DEFAULT clearance.
    """

    # === Provenance (links to immutable sources) ===
    provenance: list[str] = field(default_factory=list)
    """Links to Chronicle records, SCIP symbols, evidence chunks.

    Format: {source_type}:{source_id}
    Examples:
    - 'chronicle:exec:abc123' - Chronicle ExecutionRecord
    - 'chronicle:decision:def456' - Chronicle DecisionRecord
    - 'scip:symbol:AuthService' - SCIP symbol
    - 'diff:abc123' - Git diff hash
    - 'evidence:chunk:xyz' - EvidenceChunk ID
    """

    # === Change Tracking ===
    supersedes: Optional[str] = None
    """ID of the item this supersedes (for SUPERSEDED status)."""

    superseded_by: Optional[str] = None
    """ID of the item that superseded this (set when superseded)."""

    retract_reason: Optional[ChangeReason] = None
    """Why this was retracted (for RETRACTED status)."""

    def to_public_text(self) -> str:
        """Format for DEFAULT clearance injection."""
        if self.status == ScribeStatus.ACTIVE:
            return f"• {self.public_summary}"
        elif self.status == ScribeStatus.RETRACTED:
            return f"• ~~{self.public_summary}~~ [RETRACTED: {self.public_notice}]"
        else:  # SUPERSEDED
            return f"• ~~{self.public_summary}~~ [UPDATED: {self.public_notice}]"

    def to_escalated_text(self) -> str:
        """Format for ESCALATED clearance injection."""
        base = self.to_public_text()
        if self.sealed_payload:
            # Include key sealed info for debugging/verification
            sealed_preview = str(self.sealed_payload)[:500]
            return f"{base}\n  [SEALED]: {sealed_preview}..."
        return base

    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage."""
        return {
            "id": self.id,
            "type": self.type.value,
            "status": self.status.value,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "updated_by": self.updated_by,
            "public_notice": self.public_notice,
            "public_summary": self.public_summary,
            "sealed_payload": self.sealed_payload,
            "provenance": self.provenance,
            "supersedes": self.supersedes,
            "superseded_by": self.superseded_by,
            "retract_reason": self.retract_reason.value if self.retract_reason else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScribeItem":
        """Deserialize from storage."""
        return cls(
            id=data["id"],
            type=ScribeType(data["type"]),
            status=ScribeStatus(data["status"]),
            version=data.get("version", 1),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            updated_by=data.get("updated_by", ""),
            public_notice=data.get("public_notice", ""),
            public_summary=data.get("public_summary", ""),
            sealed_payload=data.get("sealed_payload", {}),
            provenance=data.get("provenance", []),
            supersedes=data.get("supersedes"),
            superseded_by=data.get("superseded_by"),
            retract_reason=ChangeReason(data["retract_reason"]) if data.get("retract_reason") else None,
        )


# =============================================================================
# Section Grouping
# =============================================================================


@dataclass
class ScribeSection:
    """A section of the Scribe for injection.

    Sections group related ScribeTypes for phase-specific injection.
    """

    name: str
    """Section name for display."""

    types: list[ScribeType]
    """ScribeTypes included in this section."""

    header: str
    """Markdown header for injection."""


# Pre-defined sections for injection
SECTIONS = {
    "repo_map": ScribeSection(
        name="Repository Map",
        types=[ScribeType.REPO_MAP],
        header="## 📁 Repository Structure",
    ),
    "conventions": ScribeSection(
        name="Conventions",
        types=[ScribeType.CONVENTION],
        header="## 📋 Project Conventions",
    ),
    "constraints": ScribeSection(
        name="Current Constraints",
        types=[ScribeType.CONSTRAINT, ScribeType.DECISION],
        header="## 🎯 Current Constraints & Decisions",
    ),
    "plan": ScribeSection(
        name="Active Plan",
        types=[ScribeType.PLAN_NODE],
        header="## 📝 Implementation Plan",
    ),
    "risks": ScribeSection(
        name="Known Risks",
        types=[ScribeType.RISK, ScribeType.REVIEW_FINDING],
        header="## ⚠️ Known Risks & Review Findings",
    ),
    "failures": ScribeSection(
        name="Recent Failures",
        types=[ScribeType.FAILURE_SIGNATURE],
        header="## 💥 Recent Failures & Fix Hints",
    ),
}


# =============================================================================
# Phase → Section Mapping
# =============================================================================


# Which sections each phase receives (injection)
PHASE_SECTION_MAP: dict[str, list[str]] = {
    "prepare": [],  # No injection, only updates
    "design": ["repo_map"],
    "clarity_check": ["repo_map", "conventions"],
    "plan": ["repo_map", "conventions", "constraints"],
    "plan_review": ["conventions", "plan"],
    "plan_fix": ["conventions", "plan", "risks", "failures"],
    "execute": ["repo_map", "conventions", "constraints", "plan", "risks", "failures"],
    "result_review": ["conventions", "plan", "risks"],
    "result_fix": ["conventions", "plan", "risks", "failures"],
}

# Default clearance level for each phase
PHASE_CLEARANCE_MAP: dict[str, Clearance] = {
    "prepare": Clearance.DEFAULT,
    "design": Clearance.DEFAULT,
    "clarity_check": Clearance.DEFAULT,
    "plan": Clearance.DEFAULT,
    "plan_review": Clearance.DEFAULT,  # Can escalate on conflict
    "plan_fix": Clearance.ESCALATED,   # Fix phases get sealed content
    "execute": Clearance.DEFAULT,       # Can escalate on failure
    "result_review": Clearance.DEFAULT, # Can escalate on conflict
    "result_fix": Clearance.ESCALATED,  # Fix phases get sealed content
}

# Which sections each phase updates (after completion)
PHASE_UPDATE_MAP: dict[str, list[str]] = {
    "prepare": ["repo_map", "conventions"],
    "design": [],  # Brainstorm doesn't update scribe directly
    "clarity_check": ["constraints"],
    "plan": ["plan"],
    "plan_review": ["risks"],
    "plan_fix": ["plan", "risks"],
    "execute": ["plan"],  # Status updates
    "result_review": ["risks"],
    "result_fix": ["plan", "risks", "failures"],
}


__all__ = [
    "ScribeType",
    "ScribeStatus",
    "Clearance",
    "ChangeReason",
    "ScribeItem",
    "ScribeSection",
    "SECTIONS",
    "PHASE_SECTION_MAP",
    "PHASE_CLEARANCE_MAP",
    "PHASE_UPDATE_MAP",
]
