"""Scribe Policy - Phase rules and escalation triggers.

This module defines:
1. Which sections each phase gets
2. Default clearance per phase
3. Conditions for escalation (unsealing)
4. Injection text formatting
"""

from __future__ import annotations

import re
from typing import Optional

from .types import (
    Clearance,
    PHASE_CLEARANCE_MAP,
    PHASE_SECTION_MAP,
    SECTIONS,
    ScribeItem,
)


# =============================================================================
# Phase Rules
# =============================================================================


def get_sections_for_phase(phase_name: str) -> list[str]:
    """Get section names that should be injected for a phase.

    Args:
        phase_name: Name of the pipeline phase

    Returns:
        List of section names (keys in SECTIONS)
    """
    return PHASE_SECTION_MAP.get(phase_name, [])


def get_clearance_for_phase(phase_name: str) -> Clearance:
    """Get default clearance level for a phase.

    Args:
        phase_name: Name of the pipeline phase

    Returns:
        Default Clearance level
    """
    return PHASE_CLEARANCE_MAP.get(phase_name, Clearance.DEFAULT)


# =============================================================================
# Escalation Triggers
# =============================================================================


def should_escalate(
    phase_name: str,
    *,
    error_message: Optional[str] = None,
    review_verdict: Optional[str] = None,
    conflict_detected: bool = False,
    retry_count: int = 0,
) -> Clearance:
    """Determine if clearance should be escalated.

    Escalation from DEFAULT to ESCALATED happens when:
    1. Failure/error occurs (need to see sealed failure signatures)
    2. Review verdict is needs_revision or rejected
    3. Conflict is detected (need to compare with superseded items)
    4. Retry count exceeds threshold

    Args:
        phase_name: Current phase name
        error_message: Error message if execution failed
        review_verdict: Codex review verdict (approved/needs_revision/rejected)
        conflict_detected: Whether a conflict was detected
        retry_count: Number of retries so far

    Returns:
        Clearance level to use (may be escalated from default)
    """
    base_clearance = get_clearance_for_phase(phase_name)

    # Already escalated phases stay escalated
    if base_clearance == Clearance.ESCALATED:
        return Clearance.ESCALATED

    # Error triggers escalation
    if error_message:
        # Only escalate for substantial errors
        if _is_substantial_error(error_message):
            return Clearance.ESCALATED

    # Review needs_revision or rejected triggers escalation
    if review_verdict and review_verdict.lower() in ("needs_revision", "rejected"):
        return Clearance.ESCALATED

    # Conflict detection triggers escalation
    if conflict_detected:
        return Clearance.ESCALATED

    # Multiple retries trigger escalation
    if retry_count >= 2:
        return Clearance.ESCALATED

    return base_clearance


def _is_substantial_error(error_message: str) -> bool:
    """Check if error is substantial enough to warrant escalation.

    Filters out trivial errors like typos or lint warnings.
    """
    # Patterns that indicate substantial errors
    substantial_patterns = [
        r"TypeError",
        r"AttributeError",
        r"ImportError",
        r"ModuleNotFoundError",
        r"NameError",
        r"KeyError",
        r"IndexError",
        r"RuntimeError",
        r"ValueError",
        r"AssertionError",
        r"ConnectionError",
        r"TimeoutError",
        r"failed",
        r"error:",
        r"exception",
        r"traceback",
        r"undefined",
        r"not found",
        r"missing",
        r"conflict",
        r"incompatible",
    ]

    error_lower = error_message.lower()
    for pattern in substantial_patterns:
        if re.search(pattern, error_lower, re.IGNORECASE):
            return True

    # Length heuristic: longer errors are usually more substantial
    return len(error_message) > 200


# =============================================================================
# Injection Formatting
# =============================================================================


def format_for_injection(
    sections: dict[str, list[ScribeItem]],
    clearance: Clearance,
    *,
    include_empty_sections: bool = False,
) -> str:
    """Format scribe sections for injection into LLM context.

    Args:
        sections: Dict mapping section_name → list of ScribeItems
        clearance: Visibility level
        include_empty_sections: Whether to include headers for empty sections

    Returns:
        Formatted markdown text
    """
    lines: list[str] = []

    for section_name, items in sections.items():
        section = SECTIONS.get(section_name)
        if not section:
            continue

        if not items and not include_empty_sections:
            continue

        lines.append(section.header)
        lines.append("")

        if not items:
            lines.append("_(No items in this section)_")
            lines.append("")
            continue

        # Group by status for clarity
        active_items = [i for i in items if i.status.value == "active"]
        changed_items = [i for i in items if i.status.value != "active"]

        # Active items first
        for item in active_items:
            if clearance == Clearance.DEFAULT:
                lines.append(item.to_public_text())
            else:
                lines.append(item.to_escalated_text())

        # Then changed items (with notice only in DEFAULT)
        if changed_items:
            lines.append("")
            lines.append("**Changes:**")
            for item in changed_items:
                lines.append(f"  {item.to_public_text()}")

        lines.append("")

    return "\n".join(lines)


def format_change_notice(
    item: ScribeItem,
    change_type: str,
    reason: str,
) -> str:
    """Format a change notice for an item.

    Used when updating public_notice field.

    Args:
        item: The ScribeItem being changed
        change_type: Type of change (retracted, superseded, updated)
        reason: Human-readable reason

    Returns:
        Formatted notice string
    """
    if change_type == "retracted":
        return f"❌ Retracted: {reason}"
    elif change_type == "superseded":
        return f"📝 Superseded: {reason}"
    elif change_type == "updated":
        return f"🔄 Updated: {reason}"
    else:
        return f"ℹ️ Changed: {reason}"


# =============================================================================
# Injection Context Builder
# =============================================================================


class ScribeContext:
    """Builder for injection context with scribe digest.

    This is used to build the context for a phase and track
    the digest for cache invalidation.
    """

    def __init__(
        self,
        phase_name: str,
        text: str,
        digest: str,
        clearance: Clearance,
        section_names: list[str],
    ):
        self.phase_name = phase_name
        self.text = text
        self.digest = digest
        self.clearance = clearance
        self.section_names = section_names

    @property
    def is_empty(self) -> bool:
        """Check if context has any content."""
        return not self.text or self.text.strip() == ""

    @property
    def cache_key(self) -> str:
        """Generate cache key component for EvidenceGate.

        Format: scribe:{phase}:{clearance}:{digest}
        """
        return f"scribe:{self.phase_name}:{self.clearance.value}:{self.digest}"

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return f"ScribeContext(phase={self.phase_name}, clearance={self.clearance.value}, digest={self.digest[:8]})"


__all__ = [
    "get_sections_for_phase",
    "get_clearance_for_phase",
    "should_escalate",
    "format_for_injection",
    "format_change_notice",
    "ScribeContext",
]
