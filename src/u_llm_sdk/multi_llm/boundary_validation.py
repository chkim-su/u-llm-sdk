"""Boundary Validation for Claude Code Delegation.

This module provides validation components for enforcing BoundaryConstraints
during SEMI_AUTONOMOUS mode execution.

Components:
- BoundaryViolationError: Exception for constraint violations
- BudgetTracker: Tracks and enforces budget limits
- FileScopeValidator: Validates file paths against scope constraints
"""

from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from u_llm_sdk.types import BoundaryConstraints

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class BoundaryViolationError(Exception):
    """Raised when a boundary constraint is violated.

    This exception indicates that Claude has attempted an action that
    exceeds the LOCKED boundary constraints. Delegation should be
    terminated immediately when this is raised.

    Attributes:
        constraint: Name of the violated constraint
        limit: The limit value that was exceeded
        actual: The actual value that violated the limit
        message: Human-readable description
    """

    def __init__(
        self,
        message: str,
        constraint: str = "",
        limit: Optional[float] = None,
        actual: Optional[float] = None,
    ):
        super().__init__(message)
        self.constraint = constraint
        self.limit = limit
        self.actual = actual
        self.message = message

    def __str__(self) -> str:
        if self.limit is not None and self.actual is not None:
            return f"{self.message} (constraint={self.constraint}, limit={self.limit}, actual={self.actual})"
        return self.message


# =============================================================================
# Budget Tracker
# =============================================================================


@dataclass
class BudgetTracker:
    """Tracks and enforces budget limits during delegation.

    Monitors cumulative cost and raises BoundaryViolationError when
    the budget limit is exceeded.

    Attributes:
        max_budget_usd: Maximum allowed spend
        total_spent: Current cumulative spend
        cost_history: List of (turn, cost) tuples for audit
    """

    max_budget_usd: float
    total_spent: float = 0.0
    cost_history: list[tuple[int, float]] = field(default_factory=list)
    _turn_counter: int = 0

    def record_cost(self, cost_usd: float) -> bool:
        """Record a cost and check if within budget.

        Args:
            cost_usd: Cost in USD for this turn

        Returns:
            True if still within budget, False if at limit

        Raises:
            BoundaryViolationError: If budget is exceeded
        """
        self._turn_counter += 1
        self.cost_history.append((self._turn_counter, cost_usd))

        new_total = self.total_spent + cost_usd

        if new_total > self.max_budget_usd:
            raise BoundaryViolationError(
                f"Budget exceeded: ${new_total:.4f} > ${self.max_budget_usd:.2f}",
                constraint="max_budget_usd",
                limit=self.max_budget_usd,
                actual=new_total,
            )

        self.total_spent = new_total
        return True

    def remaining_budget(self) -> float:
        """Get remaining budget.

        Returns:
            Remaining budget in USD
        """
        return max(0.0, self.max_budget_usd - self.total_spent)

    def is_at_limit(self) -> bool:
        """Check if at or over budget limit.

        Returns:
            True if no budget remaining
        """
        return self.total_spent >= self.max_budget_usd

    def get_summary(self) -> dict:
        """Get budget summary for audit.

        Returns:
            Dictionary with budget tracking information
        """
        return {
            "max_budget_usd": self.max_budget_usd,
            "total_spent_usd": self.total_spent,
            "remaining_usd": self.remaining_budget(),
            "total_turns": self._turn_counter,
            "cost_history": self.cost_history,
        }


# =============================================================================
# File Scope Validator
# =============================================================================


@dataclass
class FileScopeValidator:
    """Validates file paths against scope constraints.

    Ensures that file operations only target allowed paths and
    avoid forbidden paths.

    Attributes:
        allowed_patterns: Glob patterns for allowed files (empty = all allowed)
        forbidden_patterns: Glob patterns for forbidden files
    """

    allowed_patterns: list[str] = field(default_factory=list)
    forbidden_patterns: list[str] = field(default_factory=list)

    @classmethod
    def from_constraints(cls, constraints: BoundaryConstraints) -> FileScopeValidator:
        """Create validator from BoundaryConstraints.

        Args:
            constraints: The boundary constraints

        Returns:
            FileScopeValidator instance
        """
        return cls(
            allowed_patterns=constraints.file_scope,
            forbidden_patterns=constraints.forbidden_paths,
        )

    def is_allowed(self, file_path: str) -> bool:
        """Check if a file path is allowed.

        A file is allowed if:
        1. It matches at least one allowed pattern (or allowed_patterns is empty)
        2. It does NOT match any forbidden pattern

        Args:
            file_path: The file path to check

        Returns:
            True if the file is allowed, False otherwise
        """
        # Normalize path
        normalized = str(Path(file_path))

        # Check forbidden patterns first (highest priority)
        for pattern in self.forbidden_patterns:
            if self._matches_pattern(normalized, pattern):
                logger.debug(f"File {normalized} blocked by forbidden pattern: {pattern}")
                return False

        # If no allowed patterns, all non-forbidden files are allowed
        if not self.allowed_patterns:
            return True

        # Check allowed patterns
        for pattern in self.allowed_patterns:
            if self._matches_pattern(normalized, pattern):
                return True

        logger.debug(f"File {normalized} not in allowed patterns: {self.allowed_patterns}")
        return False

    def validate(self, file_path: str) -> None:
        """Validate a file path, raising exception if not allowed.

        Args:
            file_path: The file path to validate

        Raises:
            BoundaryViolationError: If file is not allowed
        """
        if not self.is_allowed(file_path):
            # Determine which constraint was violated
            normalized = str(Path(file_path))

            for pattern in self.forbidden_patterns:
                if self._matches_pattern(normalized, pattern):
                    raise BoundaryViolationError(
                        f"File modification blocked: {file_path} matches forbidden pattern {pattern}",
                        constraint="forbidden_paths",
                    )

            raise BoundaryViolationError(
                f"File modification blocked: {file_path} not in allowed scope {self.allowed_patterns}",
                constraint="file_scope",
            )

    def _matches_pattern(self, path: str, pattern: str) -> bool:
        """Check if path matches a glob pattern.

        Args:
            path: The file path
            pattern: The glob pattern

        Returns:
            True if matches
        """
        # Handle both forward and backslashes
        path = path.replace("\\", "/")
        pattern = pattern.replace("\\", "/")

        # fnmatch for glob matching
        if fnmatch.fnmatch(path, pattern):
            return True

        # Also try matching against just the filename
        filename = Path(path).name
        if fnmatch.fnmatch(filename, pattern):
            return True

        # Try with ** expansion (recursive match)
        if "**" in pattern:
            # Convert ** to work with fnmatch
            # src/**/*.py should match src/foo/bar.py
            parts = pattern.split("**")
            if len(parts) == 2:
                prefix, suffix = parts
                prefix = prefix.rstrip("/")
                suffix = suffix.lstrip("/")

                # Check if path starts with prefix and ends with suffix pattern
                if prefix and not path.startswith(prefix):
                    return False
                if suffix and not fnmatch.fnmatch(path, f"*{suffix}"):
                    return False
                if prefix and suffix:
                    return True
                if prefix and path.startswith(prefix):
                    return True
                if suffix:
                    return fnmatch.fnmatch(path, f"*{suffix}")

        return False


# =============================================================================
# File Modification Tracker
# =============================================================================


@dataclass
class FileModificationTracker:
    """Tracks file modifications and enforces limits.

    Attributes:
        max_files: Maximum number of files that can be modified
        modified_files: Set of modified file paths
    """

    max_files: int
    modified_files: set[str] = field(default_factory=set)

    def record_modification(self, file_path: str) -> None:
        """Record a file modification.

        Args:
            file_path: The file that was modified

        Raises:
            BoundaryViolationError: If max files limit exceeded
        """
        normalized = str(Path(file_path))

        # If already tracked, no limit check needed
        if normalized in self.modified_files:
            return

        # Check limit before adding
        if len(self.modified_files) >= self.max_files:
            raise BoundaryViolationError(
                f"Maximum file modification limit exceeded: {self.max_files} files",
                constraint="max_files_modified",
                limit=float(self.max_files),
                actual=float(len(self.modified_files) + 1),
            )

        self.modified_files.add(normalized)

    def get_modified_count(self) -> int:
        """Get count of modified files.

        Returns:
            Number of unique files modified
        """
        return len(self.modified_files)

    def get_modified_files(self) -> list[str]:
        """Get list of modified files.

        Returns:
            List of modified file paths
        """
        return list(self.modified_files)


# =============================================================================
# Boundary Validator (Combined)
# =============================================================================


@dataclass
class BoundaryValidator:
    """Combined boundary validator for delegation execution.

    Combines budget tracking, file scope validation, and modification
    limits into a single validator.

    Attributes:
        constraints: The boundary constraints
        budget_tracker: Budget tracking component
        file_scope_validator: File scope validation component
        file_modification_tracker: File modification tracking component
    """

    constraints: BoundaryConstraints
    budget_tracker: BudgetTracker = field(init=False)
    file_scope_validator: FileScopeValidator = field(init=False)
    file_modification_tracker: FileModificationTracker = field(init=False)

    def __post_init__(self):
        """Initialize sub-validators from constraints."""
        self.budget_tracker = BudgetTracker(self.constraints.max_budget_usd)
        self.file_scope_validator = FileScopeValidator.from_constraints(self.constraints)
        self.file_modification_tracker = FileModificationTracker(
            self.constraints.max_files_modified
        )

    def validate_file_operation(self, file_path: str) -> None:
        """Validate a file operation.

        Args:
            file_path: The file being operated on

        Raises:
            BoundaryViolationError: If operation violates constraints
        """
        self.file_scope_validator.validate(file_path)
        self.file_modification_tracker.record_modification(file_path)

    def validate_shell_command(self, command: str) -> None:
        """Validate a shell command.

        Args:
            command: The shell command to execute

        Raises:
            BoundaryViolationError: If shell commands not allowed
        """
        if not self.constraints.allow_shell_commands:
            raise BoundaryViolationError(
                f"Shell command execution not allowed: {command[:50]}...",
                constraint="allow_shell_commands",
            )

    def validate_web_access(self, url: str) -> None:
        """Validate web access.

        Args:
            url: The URL being accessed

        Raises:
            BoundaryViolationError: If web access not allowed
        """
        if not self.constraints.allow_web_access:
            raise BoundaryViolationError(
                f"Web access not allowed: {url[:50]}...",
                constraint="allow_web_access",
            )

    def record_cost(self, cost_usd: float) -> None:
        """Record a cost.

        Args:
            cost_usd: Cost in USD

        Raises:
            BoundaryViolationError: If budget exceeded
        """
        self.budget_tracker.record_cost(cost_usd)

    def get_summary(self) -> dict:
        """Get validation summary for audit.

        Returns:
            Dictionary with all validation state
        """
        return {
            "budget": self.budget_tracker.get_summary(),
            "files_modified": self.file_modification_tracker.get_modified_files(),
            "files_modified_count": self.file_modification_tracker.get_modified_count(),
            "constraints": self.constraints.to_dict(),
        }
