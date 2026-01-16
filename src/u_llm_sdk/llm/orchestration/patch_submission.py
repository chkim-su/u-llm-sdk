"""Patch-based Submission for WorkOrder changes.

Implements worktree → diff → integration pattern:
    1. Generate patch from worktree changes
    2. Validate patch (conflicts, file_set compliance)
    3. Apply patch to integration branch

Benefits over direct branch merge:
    - Cleaner audit trail (each patch is atomic)
    - Better conflict detection before commit
    - Easier rollback (unapply patch)
    - Deterministic: same changes = same patch

Usage:
    >>> submitter = PatchSubmitter(git_manager)
    >>> result = await submitter.submit(
    ...     worktree_path="/path/to/worktree",
    ...     work_order=wo,
    ...     integration_branch="llm/integration",
    ... )
    >>> if result.success:
    ...     print(f"Applied: {result.commit_sha}")
"""

from __future__ import annotations

import asyncio
import fnmatch
import hashlib
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Set, Union

from .contracts import WorkOrder


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class PatchSubmission:
    """Represents a patch to be submitted.

    Attributes:
        work_order_id: ID of the WorkOrder this patch is for
        patch_content: The unified diff content
        patch_hash: SHA256 hash of the patch (for dedup/audit)
        files_changed: List of files affected
        lines_added: Number of lines added
        lines_deleted: Number of lines deleted
        generated_at: When the patch was generated
        base_commit: Commit the patch is based on
    """

    work_order_id: str
    patch_content: str
    patch_hash: str = ""
    files_changed: list[str] = field(default_factory=list)
    lines_added: int = 0
    lines_deleted: int = 0
    generated_at: str = ""
    base_commit: str = ""

    def __post_init__(self):
        if not self.patch_hash:
            self.patch_hash = hashlib.sha256(self.patch_content.encode()).hexdigest()[
                :16
            ]
        if not self.generated_at:
            self.generated_at = datetime.now(timezone.utc).isoformat()


@dataclass
class PatchSubmissionResult:
    """Result of applying a patch.

    Attributes:
        success: Whether the patch was applied successfully
        commit_sha: Commit SHA if successful
        patch_hash: Hash of the applied patch
        files_modified: Files that were modified
        conflict_files: Files with conflicts (if any)
        error: Error message if failed
    """

    success: bool
    commit_sha: str = ""
    patch_hash: str = ""
    files_modified: list[str] = field(default_factory=list)
    conflict_files: list[str] = field(default_factory=list)
    error: str = ""


# =============================================================================
# Patch Generation
# =============================================================================


def generate_worktree_patch(
    worktree_path: Union[str, Path],
    base_commit: Optional[str] = None,
    staged_only: bool = False,
) -> PatchSubmission:
    """Generate a patch from worktree changes.

    Args:
        worktree_path: Path to the worktree
        base_commit: Base commit to diff against (default: HEAD)
        staged_only: Only include staged changes

    Returns:
        PatchSubmission with the diff content

    Example:
        >>> patch = generate_worktree_patch("/path/to/worktree")
        >>> print(patch.patch_hash)
        "a1b2c3d4e5f6"
    """
    worktree_path = Path(worktree_path)

    # Build git diff command
    cmd = ["git", "-C", str(worktree_path), "diff"]
    if staged_only:
        cmd.append("--cached")
    if base_commit:
        cmd.append(base_commit)
    cmd.extend(["--unified=3", "--no-color"])

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=30, check=False
    )

    if result.returncode != 0:
        # Try unstaged diff
        cmd = ["git", "-C", str(worktree_path), "diff", "--unified=3", "--no-color"]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30, check=False
        )

    patch_content = result.stdout

    # Parse diff stats
    files_changed = []
    lines_added = 0
    lines_deleted = 0

    for line in patch_content.split("\n"):
        if line.startswith("diff --git"):
            # Extract filename: diff --git a/file.py b/file.py
            parts = line.split()
            if len(parts) >= 4:
                filename = parts[2].lstrip("a/")
                files_changed.append(filename)
        elif line.startswith("+") and not line.startswith("+++"):
            lines_added += 1
        elif line.startswith("-") and not line.startswith("---"):
            lines_deleted += 1

    # Get base commit
    if not base_commit:
        base_result = subprocess.run(
            ["git", "-C", str(worktree_path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        base_commit = base_result.stdout.strip() if base_result.returncode == 0 else ""

    return PatchSubmission(
        work_order_id="",  # Set by caller
        patch_content=patch_content,
        files_changed=files_changed,
        lines_added=lines_added,
        lines_deleted=lines_deleted,
        base_commit=base_commit,
    )


def create_patch_from_diff(
    diff_content: str,
    work_order_id: str = "",
) -> PatchSubmission:
    """Create a PatchSubmission from raw diff content.

    Args:
        diff_content: Raw unified diff content
        work_order_id: Optional WorkOrder ID

    Returns:
        PatchSubmission instance
    """
    files_changed = []
    lines_added = 0
    lines_deleted = 0

    for line in diff_content.split("\n"):
        if line.startswith("diff --git"):
            parts = line.split()
            if len(parts) >= 4:
                filename = parts[2].lstrip("a/")
                files_changed.append(filename)
        elif line.startswith("+") and not line.startswith("+++"):
            lines_added += 1
        elif line.startswith("-") and not line.startswith("---"):
            lines_deleted += 1

    return PatchSubmission(
        work_order_id=work_order_id,
        patch_content=diff_content,
        files_changed=files_changed,
        lines_added=lines_added,
        lines_deleted=lines_deleted,
    )


# =============================================================================
# Patch Application
# =============================================================================


def apply_patch_to_branch(
    repo_path: Union[str, Path],
    patch: PatchSubmission,
    target_branch: str,
    commit_message: Optional[str] = None,
    dry_run: bool = False,
) -> PatchSubmissionResult:
    """Apply a patch to a target branch.

    Args:
        repo_path: Path to the repository
        patch: PatchSubmission to apply
        target_branch: Branch to apply the patch to
        commit_message: Commit message (default: auto-generated)
        dry_run: If True, check for conflicts without applying

    Returns:
        PatchSubmissionResult with success/failure info
    """
    repo_path = Path(repo_path)

    if not patch.patch_content.strip():
        # Empty patch = no changes to apply = not a "success" (nothing was done)
        return PatchSubmissionResult(
            success=False,
            patch_hash=patch.patch_hash,
            files_modified=[],
            error="NO_CHANGES: Empty patch - nothing to apply",
        )

    # Save patch to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
        f.write(patch.patch_content)
        patch_file = f.name

    # Save current branch for recovery
    original_branch = None
    try:
        branch_result = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if branch_result.returncode == 0:
            original_branch = branch_result.stdout.strip()
    except Exception:
        pass  # Best effort

    def _restore_branch():
        """Restore original branch on failure."""
        if original_branch and original_branch != target_branch:
            subprocess.run(
                ["git", "-C", str(repo_path), "checkout", original_branch],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )

    # Track files that existed before patch for cleanup
    existed_before: Dict[str, bool] = {}
    for file_path in dict.fromkeys(patch.files_changed):  # stable dedup
        try:
            existed_before[file_path] = (repo_path / file_path).exists()
        except Exception:
            existed_before[file_path] = True  # conservative: don't delete

    def _remove_new_files_best_effort():
        """Remove files introduced by this patch (best effort)."""
        for file_path, existed in existed_before.items():
            if existed:
                continue
            full_path = repo_path / file_path
            try:
                # Only remove if not tracked
                tracked = (
                    subprocess.run(
                        [
                            "git",
                            "-C",
                            str(repo_path),
                            "ls-files",
                            "--error-unmatch",
                            file_path,
                        ],
                        capture_output=True,
                        text=True,
                        timeout=10,
                        check=False,
                    ).returncode
                    == 0
                )
                if tracked:
                    continue
                if full_path.is_file() or full_path.is_symlink():
                    full_path.unlink()
            except Exception:
                pass

    def _rollback_working_tree():
        """Rollback changes on target branch (best effort)."""
        subprocess.run(
            ["git", "-C", str(repo_path), "reset", "--hard", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        _remove_new_files_best_effort()

    try:
        # Checkout target branch
        checkout_result = subprocess.run(
            ["git", "-C", str(repo_path), "checkout", target_branch],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        if checkout_result.returncode != 0:
            _restore_branch()
            return PatchSubmissionResult(
                success=False,
                patch_hash=patch.patch_hash,
                error=f"Failed to checkout {target_branch}: {checkout_result.stderr}",
            )

        # Check if patch applies cleanly
        check_result = subprocess.run(
            ["git", "-C", str(repo_path), "apply", "--check", patch_file],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        if check_result.returncode != 0:
            # Detect conflict files
            conflict_files = []
            for line in check_result.stderr.split("\n"):
                if "error:" in line and ":" in line:
                    # Extract filename from error message
                    parts = line.split(":")
                    if len(parts) >= 2:
                        conflict_files.append(parts[1].strip())

            _restore_branch()
            return PatchSubmissionResult(
                success=False,
                patch_hash=patch.patch_hash,
                conflict_files=conflict_files,
                error=f"Patch conflicts: {check_result.stderr}",
            )

        if dry_run:
            _restore_branch()
            return PatchSubmissionResult(
                success=True,
                patch_hash=patch.patch_hash,
                files_modified=patch.files_changed,
            )

        # Apply the patch
        apply_result = subprocess.run(
            ["git", "-C", str(repo_path), "apply", patch_file],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        if apply_result.returncode != 0:
            # Apply can leave partial changes; rollback to keep repo clean
            _rollback_working_tree()
            _restore_branch()
            return PatchSubmissionResult(
                success=False,
                patch_hash=patch.patch_hash,
                error=f"Failed to apply patch: {apply_result.stderr}",
            )

        # Stage changes
        add_result = subprocess.run(
            ["git", "-C", str(repo_path), "add", "-A"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        if add_result.returncode != 0:
            # Staging failed - try to reset and restore
            subprocess.run(
                ["git", "-C", str(repo_path), "reset", "--hard", "HEAD"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            _restore_branch()
            return PatchSubmissionResult(
                success=False,
                patch_hash=patch.patch_hash,
                error=f"Failed to stage changes: {add_result.stderr}",
            )

        # Commit
        if not commit_message:
            commit_message = (
                f"Apply patch {patch.patch_hash}\n\n"
                f"WorkOrder: {patch.work_order_id}\n"
                f"Files: {len(patch.files_changed)}\n"
                f"+{patch.lines_added}/-{patch.lines_deleted}"
            )

        commit_result = subprocess.run(
            ["git", "-C", str(repo_path), "commit", "-m", commit_message],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        if commit_result.returncode != 0:
            # Commit failed - rollback
            _rollback_working_tree()
            _restore_branch()
            return PatchSubmissionResult(
                success=False,
                patch_hash=patch.patch_hash,
                error=f"Failed to commit: {commit_result.stderr or commit_result.stdout}",
            )

        # Get commit SHA
        sha_result = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

        commit_sha = sha_result.stdout.strip() if sha_result.returncode == 0 else ""

        return PatchSubmissionResult(
            success=True,
            commit_sha=commit_sha,
            patch_hash=patch.patch_hash,
            files_modified=patch.files_changed,
        )

    finally:
        # Cleanup temp file
        try:
            os.unlink(patch_file)
        except OSError:
            pass


# =============================================================================
# PatchSubmitter (async wrapper)
# =============================================================================


class PatchSubmitter:
    """Async wrapper for patch submission operations.

    Handles the full flow:
    1. Generate patch from worktree
    2. Validate patch against WorkOrder constraints
    3. Apply patch to integration branch

    Example:
        >>> submitter = PatchSubmitter(git_manager)
        >>> result = await submitter.submit(
        ...     worktree_path="/path/to/worktree",
        ...     work_order=wo,
        ...     integration_branch="llm/integration",
        ... )
    """

    def __init__(self, repo_path: Union[str, Path]):
        """Initialize PatchSubmitter.

        Args:
            repo_path: Path to the main repository
        """
        self.repo_path = Path(repo_path)

    async def submit(
        self,
        worktree_path: Union[str, Path],
        work_order: WorkOrder,
        integration_branch: str,
        dry_run: bool = False,
    ) -> PatchSubmissionResult:
        """Submit changes from worktree as a patch.

        Args:
            worktree_path: Path to the worktree with changes
            work_order: WorkOrder for validation
            integration_branch: Target branch for the patch
            dry_run: If True, validate without applying

        Returns:
            PatchSubmissionResult with success/failure info
        """
        # Generate patch (runs in thread to avoid blocking)
        loop = asyncio.get_event_loop()
        patch = await loop.run_in_executor(
            None,
            lambda: generate_worktree_patch(worktree_path),
        )
        patch.work_order_id = work_order.id

        # Validate files are in file_set
        if work_order.file_set:
            expanded_file_set = self._expand_file_set(work_order.file_set)
            for file in patch.files_changed:
                if not self._file_matches_set(file, expanded_file_set):
                    return PatchSubmissionResult(
                        success=False,
                        patch_hash=patch.patch_hash,
                        error=f"File {file} not in WorkOrder file_set",
                    )

        # Apply patch
        result = await loop.run_in_executor(
            None,
            lambda: apply_patch_to_branch(
                self.repo_path,
                patch,
                integration_branch,
                commit_message=(
                    f"[{work_order.id}] {work_order.objective[:50]}\n\n"
                    f"Patch: {patch.patch_hash}\n"
                    f"Files: {', '.join(patch.files_changed[:5])}"
                    + (
                        f" (+{len(patch.files_changed) - 5} more)"
                        if len(patch.files_changed) > 5
                        else ""
                    )
                ),
                dry_run=dry_run,
            ),
        )

        return result

    def _expand_file_set(self, file_set: list[str]) -> Set[str]:
        """Expand glob patterns in file_set.

        Uses git ls-files for efficiency instead of filesystem glob.
        Falls back to glob for non-git repos or if git fails.
        """
        expanded: Set[str] = set()

        # Try git ls-files first (much faster for large repos)
        try:
            result = subprocess.run(
                ["git", "-C", str(self.repo_path), "ls-files"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if result.returncode == 0:
                git_files = set(result.stdout.strip().split("\n"))
                for pattern in file_set:
                    if "**" in pattern or "*" in pattern:
                        # Match git files against pattern
                        for f in git_files:
                            if fnmatch.fnmatch(f, pattern):
                                expanded.add(f)
                    else:
                        expanded.add(pattern)
                return expanded
        except Exception:
            pass  # Fall back to glob

        # Fallback: filesystem glob (slower for large repos)
        for pattern in file_set:
            if "**" in pattern or "*" in pattern:
                for path in self.repo_path.glob(pattern):
                    if path.is_file():
                        expanded.add(str(path.relative_to(self.repo_path)))
            else:
                expanded.add(pattern)
        return expanded

    def _file_matches_set(self, file: str, file_set: Set[str]) -> bool:
        """Check if a file matches the file_set."""
        if file in file_set:
            return True

        # Check glob patterns
        for pattern in file_set:
            if fnmatch.fnmatch(file, pattern):
                return True

        return False

    async def generate_patch(
        self,
        worktree_path: Union[str, Path],
        work_order_id: str = "",
    ) -> PatchSubmission:
        """Generate a patch from worktree without applying.

        Args:
            worktree_path: Path to the worktree
            work_order_id: Optional WorkOrder ID to attach

        Returns:
            PatchSubmission with the diff
        """
        loop = asyncio.get_event_loop()
        patch = await loop.run_in_executor(
            None,
            lambda: generate_worktree_patch(worktree_path),
        )
        patch.work_order_id = work_order_id
        return patch


__all__ = [
    # Data structures
    "PatchSubmission",
    "PatchSubmissionResult",
    # Functions
    "generate_worktree_patch",
    "create_patch_from_diff",
    "apply_patch_to_branch",
    # Class
    "PatchSubmitter",
]
