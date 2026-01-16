"""File Scope Management for Parallel Editing.

Provides:
1. Glob Intersection Detection - Ensures file_set disjoint
2. File Set Expansion - Converts glob patterns to actual file sets
3. Scope Validation - Validates WorkOrder file scopes

Key Insight:
    Glob intersection cannot be solved by string comparison alone.
    `**/*.py` vs `src/**` - are they disjoint? Depends on actual files.

    Solution: Expand globs against actual tracked files, then set intersection.
    This is O(n) where n = tracked files, but guarantees correctness.
"""

from __future__ import annotations

import fnmatch
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from .contracts import WorkOrder


@dataclass
class FileSetExpansion:
    """Result of expanding a glob pattern to actual files.

    Attributes:
        pattern: Original glob pattern
        files: Set of matched file paths
        base_path: Base directory used for expansion
    """

    pattern: str
    files: Set[str]
    base_path: str


@dataclass
class IntersectionResult:
    """Result of checking intersection between two file sets.

    Attributes:
        has_intersection: Whether sets overlap
        overlapping_files: Set of files in both sets
        set_a_pattern: First pattern
        set_b_pattern: Second pattern
    """

    has_intersection: bool
    overlapping_files: Set[str]
    set_a_pattern: str
    set_b_pattern: str

    def to_dict(self) -> dict:
        return {
            "has_intersection": self.has_intersection,
            "overlapping_files": list(self.overlapping_files),
            "set_a_pattern": self.set_a_pattern,
            "set_b_pattern": self.set_b_pattern,
        }


class FileSetManager:
    """Manages file set operations for parallel editing.

    Uses git ls-files as the source of truth for tracked files.
    Caches expansions for performance.

    Example:
        >>> manager = FileSetManager("/path/to/repo")
        >>> expansion = manager.expand_pattern("src/**/*.py")
        >>> print(expansion.files)  # {'src/main.py', 'src/utils.py', ...}
    """

    def __init__(
        self,
        repo_path: str,
        cache_enabled: bool = True,
    ):
        """Initialize file set manager.

        Args:
            repo_path: Path to git repository root
            cache_enabled: Whether to cache expansions
        """
        self.repo_path = Path(repo_path).resolve()
        self.cache_enabled = cache_enabled
        self._tracked_files_cache: Optional[Tuple[str, Set[str]]] = None
        self._expansion_cache: Dict[str, Set[str]] = {}

    def get_tracked_files(self, refresh: bool = False) -> Set[str]:
        """Get all tracked files in the repository.

        Uses `git ls-files` for accuracy. Results are cached
        with the current HEAD commit as cache key.

        Args:
            refresh: Force refresh of cache

        Returns:
            Set of tracked file paths (relative to repo root)
        """
        current_commit = self._get_current_commit()

        # Check cache
        if not refresh and self._tracked_files_cache:
            cached_commit, cached_files = self._tracked_files_cache
            if cached_commit == current_commit:
                return cached_files.copy()

        # Fetch from git
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=str(self.repo_path),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to get tracked files: {result.stderr}")

        files = set(result.stdout.strip().split("\n")) if result.stdout.strip() else set()

        # Cache result
        if self.cache_enabled:
            self._tracked_files_cache = (current_commit, files)

        return files

    def expand_pattern(self, pattern: str, refresh: bool = False) -> FileSetExpansion:
        """Expand a glob pattern to actual file set.

        Args:
            pattern: Glob pattern (e.g., "src/**/*.py", "tests/*")
            refresh: Force refresh of tracked files cache

        Returns:
            FileSetExpansion with matched files

        Example:
            >>> manager.expand_pattern("src/**/*.py")
            FileSetExpansion(pattern='src/**/*.py', files={'src/main.py', ...}, ...)
        """
        # Check expansion cache
        cache_key = f"{self._get_current_commit()}:{pattern}"
        if self.cache_enabled and cache_key in self._expansion_cache:
            return FileSetExpansion(
                pattern=pattern,
                files=self._expansion_cache[cache_key].copy(),
                base_path=str(self.repo_path),
            )

        tracked_files = self.get_tracked_files(refresh=refresh)
        matched_files = self._match_pattern(pattern, tracked_files)

        # Cache result
        if self.cache_enabled:
            self._expansion_cache[cache_key] = matched_files

        return FileSetExpansion(
            pattern=pattern,
            files=matched_files,
            base_path=str(self.repo_path),
        )

    def expand_patterns(self, patterns: List[str]) -> Set[str]:
        """Expand multiple patterns and union the results.

        Args:
            patterns: List of glob patterns

        Returns:
            Union of all matched files
        """
        all_files: Set[str] = set()
        for pattern in patterns:
            expansion = self.expand_pattern(pattern)
            all_files.update(expansion.files)
        return all_files

    def check_intersection(
        self,
        patterns_a: List[str],
        patterns_b: List[str],
    ) -> IntersectionResult:
        """Check if two pattern sets have file intersection.

        This is the KEY function for validating file_set disjoint.

        Args:
            patterns_a: First set of glob patterns
            patterns_b: Second set of glob patterns

        Returns:
            IntersectionResult with overlap details

        Example:
            >>> result = manager.check_intersection(
            ...     ["src/payments/**"],
            ...     ["src/auth/**", "src/payments/utils.py"]
            ... )
            >>> if result.has_intersection:
            ...     print(f"Overlap: {result.overlapping_files}")
        """
        files_a = self.expand_patterns(patterns_a)
        files_b = self.expand_patterns(patterns_b)

        overlapping = files_a & files_b

        return IntersectionResult(
            has_intersection=len(overlapping) > 0,
            overlapping_files=overlapping,
            set_a_pattern=", ".join(patterns_a),
            set_b_pattern=", ".join(patterns_b),
        )

    def validate_disjoint(
        self,
        work_order_file_sets: Dict[str, List[str]],
    ) -> Tuple[bool, List[IntersectionResult]]:
        """Validate that all WorkOrder file_sets are disjoint.

        This should be called by Planner before dispatching WorkOrders.

        Args:
            work_order_file_sets: Dict of {work_order_id: file_set patterns}

        Returns:
            Tuple of (is_valid, list of violations)

        Example:
            >>> file_sets = {
            ...     "WO-001": ["src/payments/**"],
            ...     "WO-002": ["src/auth/**"],
            ...     "WO-003": ["src/payments/utils.py"],  # Overlaps with WO-001!
            ... }
            >>> is_valid, violations = manager.validate_disjoint(file_sets)
            >>> if not is_valid:
            ...     for v in violations:
            ...         print(f"{v.set_a_pattern} overlaps with {v.set_b_pattern}")
        """
        violations: List[IntersectionResult] = []
        wo_ids = list(work_order_file_sets.keys())

        # Check all pairs
        for i in range(len(wo_ids)):
            for j in range(i + 1, len(wo_ids)):
                wo_a, wo_b = wo_ids[i], wo_ids[j]
                patterns_a = work_order_file_sets[wo_a]
                patterns_b = work_order_file_sets[wo_b]

                result = self.check_intersection(patterns_a, patterns_b)
                if result.has_intersection:
                    # Update pattern names to include WO IDs
                    result.set_a_pattern = f"{wo_a}: {result.set_a_pattern}"
                    result.set_b_pattern = f"{wo_b}: {result.set_b_pattern}"
                    violations.append(result)

        return len(violations) == 0, violations

    def _match_pattern(self, pattern: str, files: Set[str]) -> Set[str]:
        """Match a glob pattern against a set of files.

        Handles:
        - Simple wildcards: *.py, test_*
        - Directory wildcards: **/*.py
        - Path patterns: src/utils/*.py

        Args:
            pattern: Glob pattern
            files: Set of file paths to match against

        Returns:
            Set of matching files
        """
        matched: Set[str] = set()

        # Normalize pattern
        pattern = pattern.rstrip("/")

        for file_path in files:
            if self._matches(pattern, file_path):
                matched.add(file_path)

        return matched

    def _matches(self, pattern: str, file_path: str) -> bool:
        """Check if a file path matches a glob pattern.

        Supports:
        - * matches any characters except /
        - ** matches any characters including /
        - ? matches single character

        Args:
            pattern: Glob pattern
            file_path: File path to check

        Returns:
            True if matches
        """
        # Handle ** patterns (recursive)
        if "**" in pattern:
            # Convert ** to regex-like matching
            # "src/**/*.py" should match "src/a/b/c.py"
            parts = pattern.split("**")

            if len(parts) == 2:
                prefix, suffix = parts
                prefix = prefix.rstrip("/")
                suffix = suffix.lstrip("/")

                # Check prefix
                if prefix:
                    if not file_path.startswith(prefix):
                        return False
                    remaining = file_path[len(prefix) :].lstrip("/")
                else:
                    remaining = file_path

                # Check suffix
                if suffix:
                    return (
                        fnmatch.fnmatch(remaining, f"*{suffix}")
                        or fnmatch.fnmatch(remaining, f"*/{suffix}")
                        or fnmatch.fnmatch(remaining, suffix)
                    )
                else:
                    return True
            else:
                # Multiple ** - complex case, fall back to segment matching
                return self._match_complex_pattern(pattern, file_path)

        # Simple pattern - use fnmatch
        return fnmatch.fnmatch(file_path, pattern)

    def _match_complex_pattern(self, pattern: str, file_path: str) -> bool:
        """Match complex patterns with multiple ** segments.

        This is a fallback for patterns like "a/**/b/**/c.py"
        """
        # Split by ** and try to match segments
        segments = pattern.split("**")
        segments = [s.strip("/") for s in segments if s.strip("/")]

        if not segments:
            return True

        # First segment must match start
        if segments[0] and not file_path.startswith(segments[0]):
            return False

        # Last segment must match end
        if segments[-1] and not fnmatch.fnmatch(file_path, f"*{segments[-1]}"):
            return False

        # Middle segments must appear in order
        pos = 0
        for segment in segments:
            if not segment:
                continue
            idx = file_path.find(segment, pos)
            if idx == -1:
                return False
            pos = idx + len(segment)

        return True

    def _get_current_commit(self) -> str:
        """Get current HEAD commit hash for cache invalidation."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            # Not a git repo or no commits
            return "no-commit"
        return "no-commit"

    def clear_cache(self):
        """Clear all caches."""
        self._tracked_files_cache = None
        self._expansion_cache.clear()


@dataclass
class CreateFilesCollision:
    """Result of create_files collision detection.

    Attributes:
        wo_a_id: First WorkOrder ID
        wo_b_id: Second WorkOrder ID
        colliding_paths: Paths that both WorkOrders want to create
    """

    wo_a_id: str
    wo_b_id: str
    colliding_paths: Set[str]

    def to_dict(self) -> dict:
        return {
            "wo_a_id": self.wo_a_id,
            "wo_b_id": self.wo_b_id,
            "colliding_paths": list(self.colliding_paths),
        }


def validate_create_files_disjoint(
    work_orders: List["WorkOrder"],
) -> Tuple[bool, List[CreateFilesCollision]]:
    """Validate that create_files don't collide across WorkOrders.

    This is the P1 enhancement to detect create_files collision.
    The existing disjoint validation uses git ls-files which doesn't
    include new files that don't exist yet.

    Args:
        work_orders: List of WorkOrder objects

    Returns:
        Tuple of (is_valid, collisions)

    Example:
        >>> is_valid, collisions = validate_create_files_disjoint([wo1, wo2])
        >>> if not is_valid:
        ...     for c in collisions:
        ...         print(f"{c.wo_a_id} and {c.wo_b_id} both create {c.colliding_paths}")
    """
    collisions = []

    # Build map of create_files per WorkOrder
    create_sets: Dict[str, Set[str]] = {}
    for wo in work_orders:
        # Normalize paths (handle glob patterns in create_files)
        paths = set()
        for path in wo.create_files:
            # If it's a glob pattern, treat as literal path for collision
            # (we can't know what files will actually be created)
            paths.add(path)
        create_sets[wo.id] = paths

    # O(n²) pairwise check for collisions
    wo_ids = list(create_sets.keys())
    for i, id_a in enumerate(wo_ids):
        for id_b in wo_ids[i + 1 :]:
            set_a = create_sets[id_a]
            set_b = create_sets[id_b]

            # Direct path collision
            intersection = set_a & set_b
            if intersection:
                collisions.append(
                    CreateFilesCollision(
                        wo_a_id=id_a,
                        wo_b_id=id_b,
                        colliding_paths=intersection,
                    )
                )

    return len(collisions) == 0, collisions


def validate_create_vs_file_set(
    work_orders: List["WorkOrder"],
) -> Tuple[bool, List[dict]]:
    """Validate that create_files don't overlap with other WorkOrders' file_sets.

    This catches the case where WO-A creates a file that WO-B's file_set
    would also include (ownership conflict).

    Args:
        work_orders: List of WorkOrder objects

    Returns:
        Tuple of (is_valid, conflicts)
    """
    conflicts = []

    for wo_a in work_orders:
        for create_path in wo_a.create_files:
            for wo_b in work_orders:
                if wo_a.id == wo_b.id:
                    continue

                # Check if create_path matches any pattern in wo_b's file_set
                for pattern in wo_b.file_set:
                    if _path_matches_pattern(create_path, pattern):
                        conflicts.append(
                            {
                                "creator_id": wo_a.id,
                                "create_path": create_path,
                                "owner_id": wo_b.id,
                                "owner_pattern": pattern,
                                "description": f"{wo_a.id} creates '{create_path}' which matches {wo_b.id}'s file_set pattern '{pattern}'",
                            }
                        )

    return len(conflicts) == 0, conflicts


def _path_matches_pattern(path: str, pattern: str) -> bool:
    """Check if a specific path matches a glob pattern.

    Args:
        path: File path
        pattern: Glob pattern

    Returns:
        True if path matches pattern
    """
    # Handle ** patterns
    if "**" in pattern:
        parts = pattern.split("**")
        if len(parts) == 2:
            prefix, suffix = parts
            prefix = prefix.rstrip("/")
            suffix = suffix.lstrip("/")

            if prefix and not path.startswith(prefix):
                return False

            remaining = path[len(prefix) :].lstrip("/") if prefix else path

            if suffix:
                return fnmatch.fnmatch(remaining, f"*{suffix}") or fnmatch.fnmatch(
                    remaining, suffix
                )
            return True

    return fnmatch.fnmatch(path, pattern)


def validate_work_orders_disjoint(
    work_orders: List["WorkOrder"],
    repo_path: str,
) -> Tuple[bool, List[IntersectionResult]]:
    """Convenience function to validate WorkOrder file_sets are disjoint.

    This now includes create_files collision detection (P1 enhancement).

    Args:
        work_orders: List of WorkOrder objects
        repo_path: Path to git repository

    Returns:
        Tuple of (is_valid, violations)

    Example:
        >>> from u_llm_sdk.llm.orchestration import WorkOrder
        >>> work_orders = [wo1, wo2, wo3]
        >>> is_valid, violations = validate_work_orders_disjoint(work_orders, "/repo")
    """
    manager = FileSetManager(repo_path)

    # 1. Check file_set disjoint (tracked files)
    file_sets = {wo.id: wo.file_set + wo.create_files for wo in work_orders}
    is_valid, violations = manager.validate_disjoint(file_sets)

    # 2. P1 Enhancement: Check create_files collision (new files)
    # This catches collisions that file_set check misses because
    # new files aren't in git ls-files yet
    create_valid, create_collisions = validate_create_files_disjoint(work_orders)

    if not create_valid:
        for collision in create_collisions:
            # Convert to IntersectionResult format
            violations.append(
                IntersectionResult(
                    has_intersection=True,
                    overlapping_files=collision.colliding_paths,
                    set_a_pattern=f"{collision.wo_a_id}:create_files",
                    set_b_pattern=f"{collision.wo_b_id}:create_files",
                )
            )
        is_valid = False

    # 3. P1 Enhancement: Check create_files vs file_set ownership conflict
    cross_valid, cross_conflicts = validate_create_vs_file_set(work_orders)

    if not cross_valid:
        for conflict in cross_conflicts:
            violations.append(
                IntersectionResult(
                    has_intersection=True,
                    overlapping_files={conflict["create_path"]},
                    set_a_pattern=f"{conflict['creator_id']}:create_files",
                    set_b_pattern=f"{conflict['owner_id']}:file_set[{conflict['owner_pattern']}]",
                )
            )
        is_valid = False

    return is_valid, violations


__all__ = [
    "FileSetExpansion",
    "IntersectionResult",
    "FileSetManager",
    "CreateFilesCollision",
    "validate_create_files_disjoint",
    "validate_create_vs_file_set",
    "validate_work_orders_disjoint",
]
