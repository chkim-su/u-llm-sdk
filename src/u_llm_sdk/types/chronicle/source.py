"""Chronicle SourceReference - Snapshot semantics for decision context.

SourceReference captures "what did the decider actually see?" at decision time.

Snapshot Policy:
    - DecisionRecord: snapshot_hash is MANDATORY (audit-grade)
    - Other records: snapshot_hash is optional

Snapshot Options (lightweight, local-first):
    - "git:<commit_hash>" - Git commit hash (preferred when available)
    - "excerpt:sha256:<hash>" - Hash of referenced line ranges only
    - "file_hash:sha256:<hash>" - Full file content hash

Known Limitation:
    Excerpt hash captures referenced lines, not full semantic context.
    If surrounding code (imports, decorators, class hierarchy) materially
    affects the decision, expand `location` to include context lines.

    Responsibility: The decision-maker (human or orchestrator) determines
    materiality; Chronicle does not auto-expand.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class SourceKind(Enum):
    """Kind of source being referenced."""

    GIT = "git"  # Git commit reference
    FILE_EXCERPT = "file_excerpt"  # Specific lines from a file
    FILE_HASH = "file_hash"  # Full file content hash
    RECORD = "record"  # Reference to another Chronicle record
    EXTERNAL = "external"  # External resource (URL, API response, etc.)


@dataclass
class SourceReference:
    """Reference to a source with snapshot semantics.

    Captures "what did the decider see at that moment?" for reproducibility.

    Attributes:
        kind: Type of source (git, file_excerpt, file_hash, record, external)
        location: Path + line ranges if excerpt (e.g., "src/main.py:42-58")
        snapshot_hash: Content fingerprint at reference time
                       Format: "git:<hash>", "excerpt:sha256:<hash>", etc.
                       MANDATORY for DecisionRecord sources.
        description: Optional human-readable description

    Examples:
        # Git commit reference (preferred)
        SourceReference(
            kind=SourceKind.GIT,
            location="src/config.py",
            snapshot_hash="git:a1b2c3d4e5f6"
        )

        # File excerpt (for dirty working tree)
        SourceReference(
            kind=SourceKind.FILE_EXCERPT,
            location="src/config.py:42-48",
            snapshot_hash="excerpt:sha256:e3b0c44298fc..."
        )

        # Reference to another record
        SourceReference(
            kind=SourceKind.RECORD,
            location="dec_abc123",
            snapshot_hash=None  # Records are immutable, no hash needed
        )
    """

    kind: SourceKind
    location: str
    snapshot_hash: Optional[str] = None
    description: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        result = {
            "kind": self.kind.value,
            "location": self.location,
        }
        if self.snapshot_hash is not None:
            result["snapshot_hash"] = self.snapshot_hash
        if self.description is not None:
            result["description"] = self.description
        return result

    @classmethod
    def from_dict(cls, data: dict) -> SourceReference:
        """Deserialize from dictionary."""
        return cls(
            kind=SourceKind(data["kind"]),
            location=data["location"],
            snapshot_hash=data.get("snapshot_hash"),
            description=data.get("description"),
        )

    def has_snapshot(self) -> bool:
        """Check if this reference has a snapshot hash."""
        return self.snapshot_hash is not None and len(self.snapshot_hash) > 0

    @classmethod
    def from_git(
        cls,
        file_path: str,
        commit_hash: str,
        description: Optional[str] = None,
    ) -> SourceReference:
        """Create a git-based source reference.

        Args:
            file_path: Path to the file
            commit_hash: Git commit hash
            description: Optional description

        Returns:
            SourceReference with git snapshot
        """
        return cls(
            kind=SourceKind.GIT,
            location=file_path,
            snapshot_hash=f"git:{commit_hash}",
            description=description,
        )

    @classmethod
    def from_excerpt(
        cls,
        file_path: str,
        start_line: int,
        end_line: int,
        content_hash: str,
        description: Optional[str] = None,
    ) -> SourceReference:
        """Create an excerpt-based source reference.

        Use this when the file has uncommitted changes and you need to
        capture the exact content that was seen.

        Args:
            file_path: Path to the file
            start_line: Starting line number (1-indexed)
            end_line: Ending line number (inclusive)
            content_hash: SHA256 hash of the excerpt content
            description: Optional description

        Returns:
            SourceReference with excerpt snapshot
        """
        return cls(
            kind=SourceKind.FILE_EXCERPT,
            location=f"{file_path}:{start_line}-{end_line}",
            snapshot_hash=f"excerpt:sha256:{content_hash}",
            description=description,
        )

    @classmethod
    def from_record(
        cls,
        record_id: str,
        description: Optional[str] = None,
    ) -> SourceReference:
        """Create a reference to another Chronicle record.

        Args:
            record_id: The record ID being referenced
            description: Optional description

        Returns:
            SourceReference pointing to a record
        """
        return cls(
            kind=SourceKind.RECORD,
            location=record_id,
            snapshot_hash=None,  # Records are immutable
            description=description,
        )

    @classmethod
    def from_file(
        cls,
        file_path: str,
        content_hash: Optional[str] = None,
        description: Optional[str] = None,
    ) -> SourceReference:
        """Create a file-based source reference.

        This is a convenience factory for files without git tracking.
        For git-tracked files, prefer from_git().

        Args:
            file_path: Path to the file
            content_hash: SHA256 hash of file content (optional but recommended)
            description: Optional description

        Returns:
            SourceReference with file hash snapshot
        """
        return cls(
            kind=SourceKind.FILE_HASH,
            location=file_path,
            snapshot_hash=f"file_hash:sha256:{content_hash}" if content_hash else None,
            description=description,
        )

    @classmethod
    def from_external(
        cls,
        url: str,
        content_hash: Optional[str] = None,
        description: Optional[str] = None,
    ) -> SourceReference:
        """Create an external resource reference (URL, API, etc.).

        Args:
            url: URL or identifier of the external resource
            content_hash: Optional hash of the content at access time
            description: Optional description

        Returns:
            SourceReference for external resource
        """
        return cls(
            kind=SourceKind.EXTERNAL,
            location=url,
            snapshot_hash=content_hash,
            description=description,
        )

    def __post_init__(self):
        """Validate SourceReference after initialization."""
        # Check for common mistakes
        if hasattr(self, '_validated'):
            return
        self._validated = True

        # Warn about deprecated/non-existent parameters
        # (This runs during dataclass initialization if extra args were somehow passed)

    @staticmethod
    def _validate_kind(kind_str: str) -> SourceKind:
        """Validate and convert kind string to SourceKind enum.

        Args:
            kind_str: String representation of SourceKind

        Returns:
            SourceKind enum value

        Raises:
            ValueError: If kind_str is not a valid SourceKind
        """
        valid_kinds = [k.value for k in SourceKind]
        if kind_str not in valid_kinds:
            raise ValueError(
                f"Invalid SourceKind: '{kind_str}'. "
                f"Valid options: {valid_kinds}. "
                f"Note: 'GIT_FILE' does not exist, use 'git' (SourceKind.GIT) instead."
            )
        return SourceKind(kind_str)

    # =========================================================================
    # Auto-hash factory methods (for convenience)
    # =========================================================================

    @classmethod
    def from_file_auto(
        cls,
        file_path: str,
        project_root: Optional[str] = None,
        description: Optional[str] = None,
    ) -> SourceReference:
        """Create a file reference with auto-computed hash.

        Reads the file and computes SHA256 hash automatically.
        For DecisionRecord sources, use this instead of from_file().

        Args:
            file_path: Path to the file (relative to project_root or absolute)
            project_root: Optional project root directory
            description: Optional description

        Returns:
            SourceReference with computed file hash

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        import hashlib
        from pathlib import Path

        # Resolve path
        if project_root:
            full_path = Path(project_root) / file_path
        else:
            full_path = Path(file_path)

        if not full_path.exists():
            # Try as absolute path
            if not Path(file_path).exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            full_path = Path(file_path)

        content = full_path.read_text(encoding="utf-8", errors="replace")
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

        return cls(
            kind=SourceKind.FILE_HASH,
            location=file_path,
            snapshot_hash=f"file_hash:sha256:{content_hash}",
            description=description,
        )

    @classmethod
    def from_excerpt_auto(
        cls,
        file_path: str,
        start_line: int,
        end_line: int,
        project_root: Optional[str] = None,
        description: Optional[str] = None,
    ) -> SourceReference:
        """Create an excerpt reference with auto-computed hash.

        Reads the specified lines and computes SHA256 hash automatically.
        For DecisionRecord sources, use this instead of from_excerpt().

        Args:
            file_path: Path to the file
            start_line: Starting line number (1-indexed)
            end_line: Ending line number (inclusive)
            project_root: Optional project root directory
            description: Optional description

        Returns:
            SourceReference with computed excerpt hash

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        import hashlib
        from pathlib import Path

        # Resolve path
        if project_root:
            full_path = Path(project_root) / file_path
        else:
            full_path = Path(file_path)

        if not full_path.exists():
            if not Path(file_path).exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            full_path = Path(file_path)

        content = full_path.read_text(encoding="utf-8", errors="replace")
        lines = content.split("\n")

        # Extract excerpt (1-indexed)
        excerpt_lines = lines[start_line - 1:end_line]
        excerpt = "\n".join(excerpt_lines)
        content_hash = hashlib.sha256(excerpt.encode()).hexdigest()[:16]

        return cls(
            kind=SourceKind.FILE_EXCERPT,
            location=f"{file_path}:{start_line}-{end_line}",
            snapshot_hash=f"excerpt:sha256:{content_hash}",
            description=description,
        )

    @classmethod
    def from_git_auto(
        cls,
        file_path: str,
        project_root: Optional[str] = None,
        description: Optional[str] = None,
    ) -> SourceReference:
        """Create a git reference with auto-detected commit hash.

        Gets the current HEAD commit hash automatically.

        Args:
            file_path: Path to the file
            project_root: Git repository root (default: current directory)
            description: Optional description

        Returns:
            SourceReference with git commit hash

        Raises:
            RuntimeError: If not in a git repository or git command fails
        """
        import subprocess
        from pathlib import Path

        cwd = Path(project_root) if project_root else Path.cwd()

        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Git command failed: {result.stderr}")

            commit_hash = result.stdout.strip()[:12]  # Short hash

            return cls(
                kind=SourceKind.GIT,
                location=file_path,
                snapshot_hash=f"git:{commit_hash}",
                description=description,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Git command timed out")
        except FileNotFoundError:
            raise RuntimeError("Git is not installed or not in PATH")
