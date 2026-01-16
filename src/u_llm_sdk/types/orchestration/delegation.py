"""LLM Types - Delegation Types for Orchestration Modes.

This module defines types for the orchestration mode system:
- ORIGINAL_STRICT: Full master control with mandatory ClarityGate
- SEMI_AUTONOMOUS: Design/review via Multi-LLM, implementation via Claude delegation
- FULL_AUTONOMOUS: Future - placeholder only

Key types:
- OrchestrationMode: Autonomy level selection
- BoundaryConstraints: LOCKED limits that cannot be exceeded
- ConfigurableOptions: HINTS that can be adjusted
- ClaudeCodeDelegation: Input schema for delegation
- DelegationOutcome: Output schema with results and metrics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from typing import Optional


# =============================================================================
# Enums
# =============================================================================


class OrchestrationMode(Enum):
    """Orchestration mode - autonomy level philosophy.

    This is separate from ExecutionMode (in migration.py) which determines
    provider combination strategy. OrchestrationMode determines how much
    autonomy to give to individual providers.

    Attributes:
        ORIGINAL_STRICT: Full master control, ClarityGate mandatory,
                         Claude as single-response worker
        SEMI_AUTONOMOUS: Design via Multi-LLM brainstorm, implementation
                         delegated to Claude Code, optional Codex review
    """

    ORIGINAL_STRICT = "original_strict"
    SEMI_AUTONOMOUS = "semi_autonomous"


class DelegationPhase(Enum):
    """Current phase of delegation execution.

    Used to track progress through SEMI_AUTONOMOUS workflow phases.

    Attributes:
        DESIGN: Multi-LLM brainstorm phase
        IMPLEMENTATION: Claude autonomous execution phase
        REVIEW: Codex verification phase (optional)
        COMPLETED: Successfully finished
        FAILED: Error state
    """

    DESIGN = "design"
    IMPLEMENTATION = "implementation"
    REVIEW = "review"
    COMPLETED = "completed"
    FAILED = "failed"


# =============================================================================
# Boundary Constraints (LOCKED)
# =============================================================================


@dataclass
class BoundaryConstraints:
    """Hard limits that CANNOT be overridden by Claude Code.

    These constraints are LOCKED - Claude must operate within them.
    Violation terminates the delegation immediately.

    Attributes:
        max_budget_usd: Maximum spend in USD (default: 1.0)
        max_timeout_seconds: Maximum execution time in seconds (default: 600)
        file_scope: Glob patterns of files Claude CAN modify (empty = all)
        forbidden_paths: Paths Claude must NEVER touch
        max_files_modified: Maximum number of files to modify (default: 20)
        require_tests: Must run tests before completion (default: True)
        require_typecheck: Must pass type checking (default: True)
        allow_shell_commands: Whether Bash tool is permitted (default: True)
        allow_web_access: Whether WebSearch/WebFetch is permitted (default: False)
    """

    max_budget_usd: float = 1.0
    max_timeout_seconds: int = 600  # 10 minutes
    file_scope: list[str] = field(default_factory=list)
    forbidden_paths: list[str] = field(default_factory=list)
    max_files_modified: int = 20
    require_tests: bool = True
    require_typecheck: bool = True
    allow_shell_commands: bool = True
    allow_web_access: bool = False

    @property
    def max_timeout(self) -> timedelta:
        """Get timeout as timedelta."""
        return timedelta(seconds=self.max_timeout_seconds)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "max_budget_usd": self.max_budget_usd,
            "max_timeout_seconds": self.max_timeout_seconds,
            "file_scope": self.file_scope,
            "forbidden_paths": self.forbidden_paths,
            "max_files_modified": self.max_files_modified,
            "require_tests": self.require_tests,
            "require_typecheck": self.require_typecheck,
            "allow_shell_commands": self.allow_shell_commands,
            "allow_web_access": self.allow_web_access,
        }

    @classmethod
    def from_dict(cls, data: dict) -> BoundaryConstraints:
        """Deserialize from dictionary."""
        return cls(
            max_budget_usd=data.get("max_budget_usd", 1.0),
            max_timeout_seconds=data.get("max_timeout_seconds", 600),
            file_scope=data.get("file_scope", []),
            forbidden_paths=data.get("forbidden_paths", []),
            max_files_modified=data.get("max_files_modified", 20),
            require_tests=data.get("require_tests", True),
            require_typecheck=data.get("require_typecheck", True),
            allow_shell_commands=data.get("allow_shell_commands", True),
            allow_web_access=data.get("allow_web_access", False),
        )


# =============================================================================
# Configurable Options (HINTS)
# =============================================================================


@dataclass
class ConfigurableOptions:
    """Soft hints that Claude MAY consider but can override.

    These options are HINTS - Claude can adjust based on context.
    They influence behavior but don't restrict it.

    Attributes:
        suggested_plugins: Plugins Claude might find useful
        suggested_tools: Tools to prefer (e.g., ["Glob", "Grep", "Read"])
        suggested_approach: Recommended implementation approach (free text)
        prefer_incremental: Suggest incremental commits (default: True)
        code_style_hints: Code style preferences (e.g., {"indent": "4 spaces"})
        agent_model_preferences: Agent type to model mapping hints
    """

    suggested_plugins: list[str] = field(default_factory=list)
    suggested_tools: list[str] = field(default_factory=list)
    suggested_approach: str = ""
    prefer_incremental: bool = True
    code_style_hints: dict[str, str] = field(default_factory=dict)
    agent_model_preferences: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "suggested_plugins": self.suggested_plugins,
            "suggested_tools": self.suggested_tools,
            "suggested_approach": self.suggested_approach,
            "prefer_incremental": self.prefer_incremental,
            "code_style_hints": self.code_style_hints,
            "agent_model_preferences": self.agent_model_preferences,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ConfigurableOptions:
        """Deserialize from dictionary."""
        return cls(
            suggested_plugins=data.get("suggested_plugins", []),
            suggested_tools=data.get("suggested_tools", []),
            suggested_approach=data.get("suggested_approach", ""),
            prefer_incremental=data.get("prefer_incremental", True),
            code_style_hints=data.get("code_style_hints", {}),
            agent_model_preferences=data.get("agent_model_preferences", {}),
        )


# =============================================================================
# Claude Code Delegation (Input Schema)
# =============================================================================


@dataclass
class ClaudeCodeDelegation:
    """Input schema for delegating implementation to Claude Code.

    This is the "work order" given to Claude Code for autonomous execution
    in SEMI_AUTONOMOUS mode.

    Attributes:
        delegation_id: Unique identifier for this delegation
        objective: Clear statement of what to accomplish
        design_context: Output from design phase (brainstorm consensus)
        boundaries: Hard constraints (LOCKED)
        options: Soft hints (configurable)
        cwd: Working directory for execution
        branch_name: Git branch for work (isolation)
        session_id: Optional session to resume
    """

    delegation_id: str
    objective: str
    design_context: str = ""
    boundaries: BoundaryConstraints = field(default_factory=BoundaryConstraints)
    options: ConfigurableOptions = field(default_factory=ConfigurableOptions)
    cwd: str = ""
    branch_name: str = ""
    session_id: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "delegation_id": self.delegation_id,
            "objective": self.objective,
            "design_context": self.design_context,
            "boundaries": self.boundaries.to_dict(),
            "options": self.options.to_dict(),
            "cwd": self.cwd,
            "branch_name": self.branch_name,
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ClaudeCodeDelegation:
        """Deserialize from dictionary."""
        return cls(
            delegation_id=data["delegation_id"],
            objective=data["objective"],
            design_context=data.get("design_context", ""),
            boundaries=BoundaryConstraints.from_dict(data.get("boundaries", {})),
            options=ConfigurableOptions.from_dict(data.get("options", {})),
            cwd=data.get("cwd", ""),
            branch_name=data.get("branch_name", ""),
            session_id=data.get("session_id"),
        )


# =============================================================================
# Delegation Outcome (Output Schema)
# =============================================================================


@dataclass
class DelegationOutcome:
    """Output schema capturing delegation execution results.

    Contains execution results, metrics, and audit trail for learning.

    Attributes:
        delegation_id: Matches input delegation_id
        phase: Final phase reached
        success: Whether delegation completed successfully
        summary: Human-readable summary of what was done
        files_modified: List of files that were changed
        commands_run: List of shell commands executed
        tests_passed: Whether tests passed (if required)
        typecheck_passed: Whether type checking passed (if required)
        budget_used_usd: Actual spend
        duration_ms: Total execution time in milliseconds
        total_turns: Number of Claude turns used
        session_id: Claude session ID for resume
        error: Error message if failed
        boundary_violations: Any constraint violations detected
        raw_events: Raw stream-json events for audit (not serialized)
    """

    delegation_id: str
    phase: DelegationPhase
    success: bool
    summary: str = ""
    files_modified: list[str] = field(default_factory=list)
    commands_run: list[str] = field(default_factory=list)
    tests_passed: Optional[bool] = None
    typecheck_passed: Optional[bool] = None
    budget_used_usd: float = 0.0
    duration_ms: int = 0
    total_turns: int = 0
    session_id: Optional[str] = None
    error: str = ""
    boundary_violations: list[str] = field(default_factory=list)
    raw_events: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to dictionary.

        Note: raw_events is excluded (too large for serialization).
        """
        return {
            "delegation_id": self.delegation_id,
            "phase": self.phase.value,
            "success": self.success,
            "summary": self.summary,
            "files_modified": self.files_modified,
            "commands_run": self.commands_run,
            "tests_passed": self.tests_passed,
            "typecheck_passed": self.typecheck_passed,
            "budget_used_usd": self.budget_used_usd,
            "duration_ms": self.duration_ms,
            "total_turns": self.total_turns,
            "session_id": self.session_id,
            "error": self.error,
            "boundary_violations": self.boundary_violations,
            # raw_events excluded - too large
        }

    @classmethod
    def from_dict(cls, data: dict) -> DelegationOutcome:
        """Deserialize from dictionary."""
        return cls(
            delegation_id=data["delegation_id"],
            phase=DelegationPhase(data["phase"]),
            success=data["success"],
            summary=data.get("summary", ""),
            files_modified=data.get("files_modified", []),
            commands_run=data.get("commands_run", []),
            tests_passed=data.get("tests_passed"),
            typecheck_passed=data.get("typecheck_passed"),
            budget_used_usd=data.get("budget_used_usd", 0.0),
            duration_ms=data.get("duration_ms", 0),
            total_turns=data.get("total_turns", 0),
            session_id=data.get("session_id"),
            error=data.get("error", ""),
            boundary_violations=data.get("boundary_violations", []),
            raw_events=data.get("raw_events", []),
        )

    @classmethod
    def failed(
        cls,
        delegation_id: str,
        error: str,
        phase: DelegationPhase = DelegationPhase.FAILED,
        violations: Optional[list[str]] = None,
    ) -> DelegationOutcome:
        """Create a failed outcome.

        Args:
            delegation_id: The delegation identifier
            error: Error message
            phase: Phase where failure occurred
            violations: List of boundary violations if any

        Returns:
            DelegationOutcome with success=False
        """
        return cls(
            delegation_id=delegation_id,
            phase=phase,
            success=False,
            error=error,
            boundary_violations=violations or [],
        )
