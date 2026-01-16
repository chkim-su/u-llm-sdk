"""Scribe - Editable Knowledge State Manager.

The Scribe maintains a "current effective knowledge state" that can be
updated/superseded/retracted, unlike Chronicle which is append-only.

Key Concepts:
    - Chronicle (RawVault): Immutable, append-only audit log
    - Scribe: Mutable, editable "current truth" for injection

Architecture:
    ScribeItem contains two visibility layers:
    - Public Layer: public_notice + public_summary (always visible)
    - Sealed Layer: sealed_payload (only visible on escalation)

    This ensures "deleted/changed items are known but not exposed in detail"
    unless explicitly escalated.

Usage:
    >>> from u_llm_sdk.scribe import ScribeStore, Clearance
    >>>
    >>> store = ScribeStore(db_path)
    >>> context = store.get_for_phase("plan", Clearance.DEFAULT)
    >>> # Later, on failure:
    >>> context = store.get_for_phase("execute", Clearance.ESCALATED)
"""

from .types import (
    ScribeType,
    ScribeStatus,
    Clearance,
    ChangeReason,
    ScribeItem,
    ScribeSection,
    SECTIONS,
    PHASE_SECTION_MAP,
    PHASE_CLEARANCE_MAP,
    PHASE_UPDATE_MAP,
)

from .store_sqlite import ScribeStore

from .policy import (
    get_sections_for_phase,
    get_clearance_for_phase,
    should_escalate,
    format_for_injection,
    format_change_notice,
    ScribeContext,
)

__all__ = [
    # Types
    "ScribeType",
    "ScribeStatus",
    "Clearance",
    "ChangeReason",
    "ScribeItem",
    "ScribeSection",
    # Maps
    "SECTIONS",
    "PHASE_SECTION_MAP",
    "PHASE_CLEARANCE_MAP",
    "PHASE_UPDATE_MAP",
    # Store
    "ScribeStore",
    # Policy
    "get_sections_for_phase",
    "get_clearance_for_phase",
    "should_escalate",
    "format_for_injection",
    "format_change_notice",
    "ScribeContext",
]
