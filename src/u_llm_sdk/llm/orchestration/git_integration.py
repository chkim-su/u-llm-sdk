"""Git Integration for Contract-Based Parallel Execution.

Provides:
1. Branch Management - Create/switch/delete branches for WorkOrders
2. Merge Operations - Merge with conflict detection
3. Conflict Resolution - Detect and report merge conflicts
4. PR-like Workflow - Simulate PR flow without GitHub API

Key Design:
    - Every WorkOrder executes in its own branch
    - Merges are --no-ff for audit trail
    - Conflicts trigger REJECTED state
    - All operations use subprocess.run (not git library)

Merge Path:
    main ─┬─ WO-001/branch ─ [execute] ─ [review] ─ merge ─┬─ main
          ├─ WO-002/branch ─ [execute] ─ [review] ─ merge ─┤
          └─ WO-003/branch ─ [execute] ─ [review] ─ merge ─┘

Conflict Handling:
    If merge conflicts occur:
    1. Abort merge
    2. Mark WorkOrder as REJECTED
    3. Include conflict info in rejection
    4. Let orchestrator handle retry or skip
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple


class MergeStrategy(Enum):
    """Git merge strategy."""

    NO_FF = "no-ff"  # Always create merge commit (recommended for audit)
    FF_ONLY = "ff-only"  # Only fast-forward
    SQUASH = "squash"  # Squash commits


@dataclass
class GitResult:
    """Result of a git operation.

    Attributes:
        success: Whether operation succeeded
        output: stdout from git
        error: stderr from git (if failed)
        return_code: Process return code
    """

    success: bool
    output: str = ""
    error: str = ""
    return_code: int = 0

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "return_code": self.return_code,
        }


@dataclass
class MergeResult:
    """Result of a merge operation.

    Attributes:
        success: Whether merge succeeded
        merged_commit: The merge commit hash (if successful)
        conflicts: List of conflicting files (if failed)
        error: Error message (if failed)
    """

    success: bool
    merged_commit: Optional[str] = None
    conflicts: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "merged_commit": self.merged_commit,
            "conflicts": self.conflicts,
            "error": self.error,
        }


@dataclass
class BranchInfo:
    """Information about a git branch.

    Attributes:
        name: Branch name
        commit: HEAD commit hash
        behind_main: Commits behind main branch
        ahead_main: Commits ahead of main branch
    """

    name: str
    commit: str
    behind_main: int = 0
    ahead_main: int = 0


@dataclass
class WorktreeInfo:
    """Information about a git worktree.

    Attributes:
        path: Absolute path to worktree directory
        branch: Branch name checked out in worktree
        commit: HEAD commit hash in worktree
        is_main: Whether this is the main worktree
        is_bare: Whether this is a bare worktree
        locked: Whether worktree is locked
    """

    path: str
    branch: str
    commit: str
    is_main: bool = False
    is_bare: bool = False
    locked: bool = False

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "branch": self.branch,
            "commit": self.commit,
            "is_main": self.is_main,
            "is_bare": self.is_bare,
            "locked": self.locked,
        }


class GitManager:
    """Git operations manager for parallel execution.

    Handles branch creation, merging, and conflict detection.
    All operations are atomic - failures leave repository in clean state.

    Example:
        >>> manager = GitManager("/path/to/repo")
        >>> manager.create_branch("WO-001-feature")
        >>> # ... execute work ...
        >>> result = manager.merge_to_main("WO-001-feature", "WO-001: Add feature")
        >>> if not result.success:
        ...     print(f"Conflicts: {result.conflicts}")
    """

    def __init__(
        self,
        repo_path: str,
        main_branch: str = "main",
    ):
        """Initialize git manager.

        Args:
            repo_path: Path to git repository root
            main_branch: Name of main branch (default: main)
        """
        self.repo_path = Path(repo_path).resolve()
        self.main_branch = main_branch

        # Verify it's a git repo
        if not (self.repo_path / ".git").exists():
            raise ValueError(f"Not a git repository: {repo_path}")

    def _run(
        self,
        args: List[str],
        check: bool = True,
        capture_output: bool = True,
    ) -> GitResult:
        """Run a git command.

        Args:
            args: Git command arguments (without 'git')
            check: Whether to raise on failure
            capture_output: Whether to capture stdout/stderr

        Returns:
            GitResult with output and status
        """
        cmd = ["git"] + args

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.repo_path),
                capture_output=capture_output,
                text=True,
            )
            success = result.returncode == 0

            if check and not success:
                raise subprocess.CalledProcessError(
                    result.returncode, cmd, result.stdout, result.stderr
                )

            return GitResult(
                success=success,
                output=result.stdout.strip() if result.stdout else "",
                error=result.stderr.strip() if result.stderr else "",
                return_code=result.returncode,
            )

        except subprocess.CalledProcessError as e:
            return GitResult(
                success=False,
                output=e.stdout if e.stdout else "",
                error=e.stderr if e.stderr else str(e),
                return_code=e.returncode,
            )

    def get_current_branch(self) -> str:
        """Get current branch name."""
        result = self._run(["branch", "--show-current"])
        return result.output

    def get_main_branch(self) -> str:
        """Get main branch name (detect if not explicitly set)."""
        # Check if configured main branch exists
        result = self._run(["branch", "--list", self.main_branch], check=False)
        if result.output:
            return self.main_branch

        # Try common alternatives
        for branch in ["main", "master"]:
            result = self._run(["branch", "--list", branch], check=False)
            if result.output:
                self.main_branch = branch
                return branch

        return self.main_branch

    def create_branch(
        self,
        branch_name: str,
        base_branch: Optional[str] = None,
        checkout: bool = True,
    ) -> GitResult:
        """Create a new branch for WorkOrder execution.

        Args:
            branch_name: Name for the new branch
            base_branch: Branch to base from (default: main)
            checkout: Whether to checkout the new branch

        Returns:
            GitResult with operation status
        """
        base = base_branch or self.get_main_branch()

        # Create branch from base
        result = self._run(["branch", branch_name, base], check=False)
        if not result.success:
            return result

        # Checkout if requested
        if checkout:
            result = self._run(["checkout", branch_name], check=False)

        return result

    def checkout(self, branch_name: str) -> GitResult:
        """Checkout a branch.

        Args:
            branch_name: Branch to checkout

        Returns:
            GitResult with operation status
        """
        return self._run(["checkout", branch_name], check=False)

    def delete_branch(
        self,
        branch_name: str,
        force: bool = False,
    ) -> GitResult:
        """Delete a branch.

        Args:
            branch_name: Branch to delete
            force: Force delete even if not merged

        Returns:
            GitResult with operation status
        """
        flag = "-D" if force else "-d"
        return self._run(["branch", flag, branch_name], check=False)

    def commit(
        self,
        message: str,
        add_all: bool = True,
    ) -> GitResult:
        """Create a commit.

        Args:
            message: Commit message
            add_all: Whether to add all changes first

        Returns:
            GitResult with operation status
        """
        if add_all:
            # Add all changes
            result = self._run(["add", "-A"], check=False)
            if not result.success:
                return result

        # Check if there are staged changes
        result = self._run(["diff", "--staged", "--quiet"], check=False)
        if result.success:
            # No staged changes
            return GitResult(
                success=True,
                output="Nothing to commit",
            )

        # Commit
        return self._run(["commit", "-m", message], check=False)

    def merge_to_main(
        self,
        branch_name: str,
        message: Optional[str] = None,
        strategy: MergeStrategy = MergeStrategy.NO_FF,
    ) -> MergeResult:
        """Merge a branch into main.

        This is the KEY function for the parallel execution flow.
        Handles conflicts by aborting and reporting.

        Args:
            branch_name: Branch to merge
            message: Merge commit message
            strategy: Merge strategy

        Returns:
            MergeResult with success status and conflict info
        """
        main_branch = self.get_main_branch()
        current_branch = self.get_current_branch()

        # Ensure we're on main
        if current_branch != main_branch:
            checkout_result = self.checkout(main_branch)
            if not checkout_result.success:
                return MergeResult(
                    success=False,
                    error=f"Failed to checkout {main_branch}: {checkout_result.error}",
                )

        # Build merge command
        merge_cmd = ["merge"]

        if strategy == MergeStrategy.NO_FF:
            merge_cmd.append("--no-ff")
        elif strategy == MergeStrategy.FF_ONLY:
            merge_cmd.append("--ff-only")
        elif strategy == MergeStrategy.SQUASH:
            merge_cmd.append("--squash")

        if message:
            merge_cmd.extend(["-m", message])

        merge_cmd.append(branch_name)

        # Attempt merge
        result = self._run(merge_cmd, check=False)

        if result.success:
            # Get merge commit hash
            commit_result = self._run(["rev-parse", "HEAD"])
            return MergeResult(
                success=True,
                merged_commit=commit_result.output,
            )

        # Check for merge conflicts
        if "CONFLICT" in result.output or "CONFLICT" in result.error:
            # Get list of conflicting files
            conflicts = self._get_conflict_files()

            # Abort merge to restore clean state
            self._run(["merge", "--abort"], check=False)

            return MergeResult(
                success=False,
                conflicts=conflicts,
                error="Merge conflicts detected",
            )

        # Other error
        return MergeResult(
            success=False,
            error=result.error or result.output,
        )

    def _get_conflict_files(self) -> List[str]:
        """Get list of files with merge conflicts."""
        result = self._run(["diff", "--name-only", "--diff-filter=U"], check=False)
        if result.output:
            return result.output.split("\n")
        return []

    def get_diff(
        self,
        base_ref: Optional[str] = None,
        head_ref: Optional[str] = None,
    ) -> str:
        """Get diff between two refs.

        Args:
            base_ref: Base reference (default: main)
            head_ref: Head reference (default: HEAD)

        Returns:
            Diff output as string
        """
        base = base_ref or self.get_main_branch()
        head = head_ref or "HEAD"

        result = self._run(["diff", f"{base}...{head}"], check=False)
        return result.output

    def get_changed_files(
        self,
        base_ref: Optional[str] = None,
        head_ref: Optional[str] = None,
    ) -> List[str]:
        """Get list of changed files between refs.

        Args:
            base_ref: Base reference (default: main)
            head_ref: Head reference (default: HEAD)

        Returns:
            List of changed file paths
        """
        base = base_ref or self.get_main_branch()
        head = head_ref or "HEAD"

        result = self._run(["diff", "--name-only", f"{base}...{head}"], check=False)
        if result.output:
            return result.output.split("\n")
        return []

    def stash(self, message: Optional[str] = None) -> GitResult:
        """Stash current changes.

        Args:
            message: Stash message

        Returns:
            GitResult with operation status
        """
        cmd = ["stash", "push"]
        if message:
            cmd.extend(["-m", message])
        return self._run(cmd, check=False)

    def stash_pop(self) -> GitResult:
        """Pop latest stash.

        Returns:
            GitResult with operation status
        """
        return self._run(["stash", "pop"], check=False)

    def has_uncommitted_changes(self) -> bool:
        """Check if there are uncommitted changes."""
        result = self._run(["status", "--porcelain"], check=False)
        return bool(result.output)

    def get_commit_hash(self, ref: str = "HEAD") -> str:
        """Get commit hash for a ref.

        Args:
            ref: Git reference (default: HEAD)

        Returns:
            Commit hash
        """
        result = self._run(["rev-parse", ref], check=False)
        return result.output

    def branch_exists(self, branch_name: str) -> bool:
        """Check if a branch exists.

        Args:
            branch_name: Branch name to check

        Returns:
            True if branch exists
        """
        result = self._run(["branch", "--list", branch_name], check=False)
        return bool(result.output)

    def get_branch_info(self, branch_name: str) -> Optional[BranchInfo]:
        """Get information about a branch.

        Args:
            branch_name: Branch to get info for

        Returns:
            BranchInfo or None if branch doesn't exist
        """
        if not self.branch_exists(branch_name):
            return None

        # Get commit hash
        commit = self._run(["rev-parse", branch_name], check=False).output

        # Get ahead/behind counts
        main = self.get_main_branch()
        result = self._run(
            ["rev-list", "--left-right", "--count", f"{main}...{branch_name}"],
            check=False,
        )

        behind, ahead = 0, 0
        if result.output:
            parts = result.output.split()
            if len(parts) == 2:
                behind, ahead = int(parts[0]), int(parts[1])

        return BranchInfo(
            name=branch_name,
            commit=commit,
            behind_main=behind,
            ahead_main=ahead,
        )

    # =========================================================================
    # Worktree Operations (for parallel execution isolation)
    # =========================================================================

    def create_worktree(
        self,
        worktree_path: str,
        branch_name: str,
        base_branch: Optional[str] = None,
        create_branch: bool = True,
    ) -> Tuple[bool, str]:
        """Create a new worktree for isolated parallel execution.

        Each WorkOrder executes in its own worktree to prevent:
        - checkout/index conflicts between parallel executions
        - delta snapshot pollution from other WorkOrders
        - git state race conditions

        Args:
            worktree_path: Path where worktree will be created
            branch_name: Branch name for the worktree
            base_branch: Base branch to create from (default: main)
            create_branch: Create new branch if it doesn't exist

        Returns:
            Tuple of (success, worktree_path or error message)
        """
        worktree_abs = Path(worktree_path).resolve()
        base = base_branch or self.get_main_branch()

        # Build worktree add command
        cmd = ["worktree", "add"]

        if create_branch:
            # Create new branch from base: git worktree add -b <branch> <path> <base>
            cmd.extend(["-b", branch_name, str(worktree_abs), base])
        else:
            # Use existing branch: git worktree add <path> <branch>
            cmd.extend([str(worktree_abs), branch_name])

        result = self._run(cmd, check=False)

        if result.success:
            return True, str(worktree_abs)
        else:
            return False, result.error or result.output

    def remove_worktree(
        self,
        worktree_path: str,
        force: bool = False,
    ) -> GitResult:
        """Remove a worktree.

        Args:
            worktree_path: Path to worktree to remove
            force: Force removal even if worktree has uncommitted changes

        Returns:
            GitResult with operation status
        """
        worktree_abs = Path(worktree_path).resolve()

        cmd = ["worktree", "remove"]
        if force:
            cmd.append("--force")
        cmd.append(str(worktree_abs))

        result = self._run(cmd, check=False)

        # Also clean up the directory if it still exists and force is True
        if force and worktree_abs.exists():
            try:
                shutil.rmtree(str(worktree_abs))
            except OSError:
                pass

        return result

    def list_worktrees(self) -> List[WorktreeInfo]:
        """List all worktrees.

        Returns:
            List of WorktreeInfo for each worktree
        """
        result = self._run(["worktree", "list", "--porcelain"], check=False)

        if not result.output:
            return []

        worktrees = []
        current_wt: dict = {}

        for line in result.output.split("\n"):
            line = line.strip()
            if not line:
                if current_wt.get("path"):
                    worktrees.append(
                        WorktreeInfo(
                            path=current_wt.get("path", ""),
                            branch=current_wt.get("branch", "").replace(
                                "refs/heads/", ""
                            ),
                            commit=current_wt.get("HEAD", ""),
                            is_main=current_wt.get("is_main", False),
                            is_bare=current_wt.get("bare", False),
                            locked=current_wt.get("locked", False),
                        )
                    )
                current_wt = {}
                continue

            if line.startswith("worktree "):
                current_wt["path"] = line[9:]
                # Main worktree is the repo_path
                if Path(current_wt["path"]).resolve() == self.repo_path:
                    current_wt["is_main"] = True
            elif line.startswith("HEAD "):
                current_wt["HEAD"] = line[5:]
            elif line.startswith("branch "):
                current_wt["branch"] = line[7:]
            elif line == "bare":
                current_wt["bare"] = True
            elif line == "locked":
                current_wt["locked"] = True

        # Handle last worktree if no trailing newline
        if current_wt.get("path"):
            worktrees.append(
                WorktreeInfo(
                    path=current_wt.get("path", ""),
                    branch=current_wt.get("branch", "").replace("refs/heads/", ""),
                    commit=current_wt.get("HEAD", ""),
                    is_main=current_wt.get("is_main", False),
                    is_bare=current_wt.get("bare", False),
                    locked=current_wt.get("locked", False),
                )
            )

        return worktrees

    def prune_worktrees(self) -> GitResult:
        """Prune stale worktree references.

        Removes worktree entries where the directory no longer exists.

        Returns:
            GitResult with operation status
        """
        return self._run(["worktree", "prune"], check=False)

    def run_in_worktree(
        self,
        worktree_path: str,
        args: List[str],
        check: bool = False,
    ) -> GitResult:
        """Run a git command in a specific worktree.

        Args:
            worktree_path: Path to worktree
            args: Git command arguments (without 'git')
            check: Whether to raise on failure

        Returns:
            GitResult with output and status
        """
        cmd = ["git"] + args

        try:
            result = subprocess.run(
                cmd,
                cwd=str(worktree_path),
                capture_output=True,
                text=True,
            )
            success = result.returncode == 0

            if check and not success:
                raise subprocess.CalledProcessError(
                    result.returncode, cmd, result.stdout, result.stderr
                )

            return GitResult(
                success=success,
                output=result.stdout.strip() if result.stdout else "",
                error=result.stderr.strip() if result.stderr else "",
                return_code=result.returncode,
            )

        except subprocess.CalledProcessError as e:
            return GitResult(
                success=False,
                output=e.stdout if e.stdout else "",
                error=e.stderr if e.stderr else str(e),
                return_code=e.returncode,
            )

    def commit_in_worktree(
        self,
        worktree_path: str,
        message: str,
        add_all: bool = True,
    ) -> GitResult:
        """Create a commit in a specific worktree.

        Args:
            worktree_path: Path to worktree
            message: Commit message
            add_all: Whether to add all changes first

        Returns:
            GitResult with operation status
        """
        if add_all:
            result = self.run_in_worktree(worktree_path, ["add", "-A"])
            if not result.success:
                return result

        # Check if there are staged changes
        result = self.run_in_worktree(worktree_path, ["diff", "--staged", "--quiet"])
        if result.success:
            return GitResult(success=True, output="Nothing to commit")

        return self.run_in_worktree(worktree_path, ["commit", "-m", message])

    def get_worktree_commit(self, worktree_path: str) -> str:
        """Get HEAD commit hash for a worktree.

        Args:
            worktree_path: Path to worktree

        Returns:
            Commit hash
        """
        result = self.run_in_worktree(worktree_path, ["rev-parse", "HEAD"])
        return result.output


# =============================================================================
# WorkOrder Branch Naming Convention
# =============================================================================


def work_order_branch_name(wo_id: str, suffix: str = "") -> str:
    """Generate branch name for a WorkOrder.

    Convention: wo/{wo_id}/{suffix}

    Args:
        wo_id: WorkOrder ID (e.g., "WO-001")
        suffix: Optional suffix (e.g., "feature", "fix")

    Returns:
        Branch name (e.g., "wo/WO-001/feature")

    Example:
        >>> work_order_branch_name("WO-001", "payments")
        'wo/WO-001/payments'
    """
    parts = ["wo", wo_id]
    if suffix:
        parts.append(suffix)
    return "/".join(parts)


def parse_work_order_branch(branch_name: str) -> Optional[str]:
    """Extract WorkOrder ID from branch name.

    Args:
        branch_name: Branch name (e.g., "wo/WO-001/feature")

    Returns:
        WorkOrder ID or None if not a WorkOrder branch

    Example:
        >>> parse_work_order_branch("wo/WO-001/feature")
        'WO-001'
    """
    if not branch_name.startswith("wo/"):
        return None

    parts = branch_name.split("/")
    if len(parts) >= 2:
        return parts[1]
    return None


# =============================================================================
# Integration with StateMachineOrchestrator
# =============================================================================


async def setup_work_order_branch(
    git_manager: GitManager,
    wo_id: str,
    objective: str,
) -> Tuple[bool, str]:
    """Set up branch for WorkOrder execution.

    Called by orchestrator before executing WorkOrder.

    Args:
        git_manager: GitManager instance
        wo_id: WorkOrder ID
        objective: WorkOrder objective (for branch suffix)

    Returns:
        Tuple of (success, branch_name or error)
    """
    # Generate branch name
    suffix = objective.lower().replace(" ", "-")[:30]
    branch_name = work_order_branch_name(wo_id, suffix)

    # Create branch
    result = git_manager.create_branch(branch_name, checkout=True)

    if result.success:
        return True, branch_name
    else:
        return False, result.error


async def commit_work_order_changes(
    git_manager: GitManager,
    wo_id: str,
    message: str,
) -> Tuple[bool, str]:
    """Commit changes for a WorkOrder.

    Called by orchestrator after WorkOrder execution.

    Args:
        git_manager: GitManager instance
        wo_id: WorkOrder ID
        message: Commit message

    Returns:
        Tuple of (success, commit_hash or error)
    """
    result = git_manager.commit(f"[{wo_id}] {message}")

    if result.success:
        commit_hash = git_manager.get_commit_hash()
        return True, commit_hash
    else:
        return False, result.error


async def merge_work_order(
    git_manager: GitManager,
    wo_id: str,
    branch_name: str,
    evidence_summary: str,
) -> MergeResult:
    """Merge WorkOrder branch into main.

    Called by orchestrator after Supervisor approval.

    Args:
        git_manager: GitManager instance
        wo_id: WorkOrder ID
        branch_name: Branch to merge
        evidence_summary: Summary for merge commit message

    Returns:
        MergeResult with success status and conflict info
    """
    message = f"Merge {wo_id}: {evidence_summary[:100]}"

    return git_manager.merge_to_main(
        branch_name,
        message=message,
        strategy=MergeStrategy.NO_FF,
    )


async def cleanup_work_order_branch(
    git_manager: GitManager,
    branch_name: str,
    force: bool = False,
) -> bool:
    """Delete WorkOrder branch after merge.

    Args:
        git_manager: GitManager instance
        branch_name: Branch to delete
        force: Force delete even if not merged

    Returns:
        True if deleted successfully
    """
    result = git_manager.delete_branch(branch_name, force=force)
    return result.success


# =============================================================================
# Worktree-Based WorkOrder Isolation (P0 Safety Enhancement)
# =============================================================================


@dataclass
class WorktreeSetupResult:
    """Result of setting up a worktree for a WorkOrder.

    Attributes:
        success: Whether setup succeeded
        worktree_path: Path to the worktree directory (if success)
        branch_name: Name of the branch in worktree (if success)
        base_commit: Commit hash the branch was based on
        error: Error message (if failed)
    """

    success: bool
    worktree_path: str = ""
    branch_name: str = ""
    base_commit: str = ""
    error: str = ""


def worktree_path_for_work_order(
    repo_path: str,
    wo_id: str,
) -> str:
    """Generate worktree path for a WorkOrder.

    Worktrees are created in .worktrees/ directory next to .git/

    Args:
        repo_path: Path to main repository
        wo_id: WorkOrder ID

    Returns:
        Absolute path for worktree directory
    """
    repo = Path(repo_path).resolve()
    worktrees_dir = repo / ".worktrees"
    return str(worktrees_dir / wo_id)


async def setup_work_order_worktree(
    git_manager: GitManager,
    wo_id: str,
    objective: str,
) -> WorktreeSetupResult:
    """Set up isolated worktree for WorkOrder execution.

    This is the P0 safety enhancement for parallel editing.
    Each WorkOrder executes in its own worktree, ensuring:
    - No checkout/index conflicts
    - Clean delta snapshots per WorkOrder
    - Full git isolation

    Args:
        git_manager: GitManager instance
        wo_id: WorkOrder ID
        objective: WorkOrder objective (for branch naming)

    Returns:
        WorktreeSetupResult with path and branch info

    Example:
        >>> result = await setup_work_order_worktree(git_manager, "WO-001", "Add auth")
        >>> if result.success:
        ...     # Editor executes in result.worktree_path
        ...     editor.run(cwd=result.worktree_path)
    """
    # Generate branch name
    suffix = objective.lower().replace(" ", "-")[:30]
    branch_name = work_order_branch_name(wo_id, suffix)

    # Generate worktree path
    worktree_path = worktree_path_for_work_order(
        str(git_manager.repo_path),
        wo_id,
    )

    # Get base commit before creating worktree
    base_commit = git_manager.get_commit_hash(git_manager.get_main_branch())

    # Create worktree with new branch
    success, path_or_error = git_manager.create_worktree(
        worktree_path,
        branch_name,
        base_branch=git_manager.get_main_branch(),
        create_branch=True,
    )

    if success:
        return WorktreeSetupResult(
            success=True,
            worktree_path=path_or_error,
            branch_name=branch_name,
            base_commit=base_commit,
        )
    else:
        return WorktreeSetupResult(
            success=False,
            error=path_or_error,
        )


async def commit_work_order_in_worktree(
    git_manager: GitManager,
    worktree_path: str,
    wo_id: str,
    message: str,
) -> Tuple[bool, str]:
    """Commit changes in a WorkOrder's worktree.

    Args:
        git_manager: GitManager instance
        worktree_path: Path to WorkOrder's worktree
        wo_id: WorkOrder ID
        message: Commit message

    Returns:
        Tuple of (success, commit_hash or error)
    """
    result = git_manager.commit_in_worktree(
        worktree_path,
        f"[{wo_id}] {message}",
    )

    if result.success:
        commit_hash = git_manager.get_worktree_commit(worktree_path)
        return True, commit_hash
    else:
        return False, result.error


async def cleanup_work_order_worktree(
    git_manager: GitManager,
    worktree_path: str,
    branch_name: str,
    delete_branch: bool = True,
    force: bool = False,
) -> bool:
    """Clean up WorkOrder worktree after merge.

    Args:
        git_manager: GitManager instance
        worktree_path: Path to worktree to remove
        branch_name: Branch name to delete
        delete_branch: Whether to also delete the branch
        force: Force removal even if uncommitted changes

    Returns:
        True if cleanup succeeded
    """
    # Remove worktree first
    result = git_manager.remove_worktree(worktree_path, force=force)

    if not result.success and not force:
        return False

    # Delete branch if requested
    if delete_branch and branch_name:
        git_manager.delete_branch(branch_name, force=True)

    # Prune stale worktree references
    git_manager.prune_worktrees()

    return True


async def get_worktree_snapshot_info(
    git_manager: GitManager,
    worktree_path: str,
) -> dict:
    """Get snapshot information from a worktree for delta compliance.

    This captures the commit state for reproducible delta analysis.

    Args:
        git_manager: GitManager instance
        worktree_path: Path to worktree

    Returns:
        Dict with commit hash and branch info
    """
    commit = git_manager.get_worktree_commit(worktree_path)
    branch_result = git_manager.run_in_worktree(
        worktree_path,
        ["branch", "--show-current"],
    )

    return {
        "commit": commit,
        "branch": branch_result.output,
        "worktree_path": worktree_path,
    }


__all__ = [
    # Enums
    "MergeStrategy",
    # Data classes
    "GitResult",
    "MergeResult",
    "BranchInfo",
    "WorktreeInfo",
    "WorktreeSetupResult",
    # Main class
    "GitManager",
    # Branch naming
    "work_order_branch_name",
    "parse_work_order_branch",
    "worktree_path_for_work_order",
    # Orchestrator integration
    "setup_work_order_branch",
    "commit_work_order_changes",
    "merge_work_order",
    "cleanup_work_order_branch",
    # Worktree operations
    "setup_work_order_worktree",
    "commit_work_order_in_worktree",
    "cleanup_work_order_worktree",
    "get_worktree_snapshot_info",
]
