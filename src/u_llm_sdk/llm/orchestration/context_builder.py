"""WorkOrder-based Context Builder.

Provides context construction for WorkOrder execution.
This module extracts relevant context from WorkOrder specifications
to help LLM understand the task scope.

NOTE: Full SCIP (Source Code Intelligence Protocol) integration
is planned for a future milestone. Currently provides:
- Contract context extraction
- Seed extraction from WorkOrder text
- File-based context building (fallback mode)

Design Philosophy:
    1. Budget-aware - max_chars/max_symbols enforce limits
    2. Deterministic - Same WorkOrder = same context structure
    3. Priority ordering - Contract > Definition > Excerpts

Usage:
    >>> builder = WorkOrderContextBuilder(repo_root=Path("/project"))
    >>> pack = await builder.build_for_work_order(work_order)
    >>> print(pack.to_prompt_text())
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .contracts import WorkOrder


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class ContextBuilderConfig:
    """Configuration for WorkOrderContextBuilder.

    Budget Allocation:
        - 50% of max_chars: Symbol definitions / file content
        - 25% of max_chars: File excerpts
        - 25% of max_chars: Contract context, hover docs, references

    Attributes:
        max_chars: Maximum total characters (default: 50000 = ~12.5k tokens)
        max_symbols: Maximum symbol definitions to include (default: 40)
        max_references_per_symbol: Max references per symbol (default: 10)
        excerpt_context_lines: Lines around each target (default: 3)
        include_hover: Include hover docs (default: True)
        include_references: Include references (default: True, capped)
        chars_per_token: Estimation ratio (default: 4.0)
    """

    max_chars: int = 50000
    max_symbols: int = 40
    max_references_per_symbol: int = 10
    excerpt_context_lines: int = 3
    include_hover: bool = True
    include_references: bool = True
    chars_per_token: float = 4.0


# =============================================================================
# Context Item and Pack
# =============================================================================


@dataclass
class ContextItem:
    """A single context item for LLM consumption.

    Attributes:
        source: Source of context (user, file, search, scip)
        path: File path if applicable
        content: The actual context content
        relevance_score: Relevance score (0-1)
        token_count: Estimated token count
        metadata: Additional metadata
    """

    source: str
    path: str
    content: str
    relevance_score: float = 1.0
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def excerpt_hash(self) -> str:
        """Generate hash for deduplication."""
        return hashlib.md5(
            f"{self.path}:{self.content[:100]}".encode()
        ).hexdigest()[:8]


@dataclass
class ContextPack:
    """Collection of context items for LLM consumption.

    Attributes:
        items: List of context items
        budget_tokens: Maximum tokens allowed
        truncated: Whether pack was truncated due to budget
        metadata: Pack metadata (commit_sha, etc.)
    """

    items: List[ContextItem] = field(default_factory=list)
    budget_tokens: int = 12500
    truncated: bool = False
    metadata: Optional[Dict[str, Any]] = None

    def add(self, item: ContextItem) -> bool:
        """Add item if not duplicate.

        Returns:
            True if added, False if duplicate
        """
        existing_hashes = {i.excerpt_hash for i in self.items}
        if item.excerpt_hash in existing_hashes:
            return False
        self.items.append(item)
        return True

    def to_prompt_text(self) -> str:
        """Convert pack to prompt text."""
        parts = []
        for item in self.items:
            parts.append(item.content)
        return "\n\n---\n\n".join(parts)

    def total_chars(self) -> int:
        """Get total character count."""
        return sum(len(item.content) for item in self.items)

    def total_tokens(self) -> int:
        """Get estimated total tokens."""
        return sum(item.token_count for item in self.items)


# =============================================================================
# Language Detection
# =============================================================================

# Extension → (indexer, language) mapping
_LANGUAGE_MAP: dict[str, tuple[str, str]] = {
    ".py": ("scip-python", "python"),
    ".pyi": ("scip-python", "python"),
    ".ts": ("scip-typescript", "typescript"),
    ".tsx": ("scip-typescript", "typescript"),
    ".js": ("scip-typescript", "javascript"),
    ".jsx": ("scip-typescript", "javascript"),
}


def detect_indexer_from_files(
    file_patterns: list[str],
) -> tuple[str, str]:
    """Detect SCIP indexer and language from file patterns.

    Args:
        file_patterns: List of file paths or glob patterns

    Returns:
        Tuple of (indexer, language). Defaults to ("scip-python", "python")
        if no supported extension is found.

    Example:
        >>> detect_indexer_from_files(["src/**/*.ts", "lib/*.tsx"])
        ("scip-typescript", "typescript")
    """
    ext_counts: dict[str, int] = {}
    for pattern in file_patterns:
        for ext in _LANGUAGE_MAP:
            if pattern.endswith(ext) or f"*{ext}" in pattern:
                ext_counts[ext] = ext_counts.get(ext, 0) + 1

    if not ext_counts:
        return ("scip-python", "python")

    most_common_ext = max(ext_counts, key=lambda e: ext_counts[e])
    return _LANGUAGE_MAP[most_common_ext]


# =============================================================================
# Seed Extraction from WorkOrder
# =============================================================================


def extract_seeds_from_work_order(wo: WorkOrder) -> list[str]:
    """Extract seed symbol candidates from WorkOrder.

    Looks for potential symbol names in:
    - objective
    - constraints
    - file_set patterns (extracts filenames as module hints)

    Args:
        wo: WorkOrder to extract seeds from

    Returns:
        List of potential symbol names/patterns
    """
    seeds: list[str] = []

    # Extract from objective - look for CamelCase, snake_case, dot.notation
    text = f"{wo.objective} {' '.join(wo.constraints)}"

    # CamelCase symbols: UserAuth, MyClass, etc. (multi-part)
    camel_case = re.findall(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b", text)
    seeds.extend(camel_case)

    # Single CamelCase: User, Model, Config (single capital word, 3+ chars)
    single_camel = re.findall(r"\b[A-Z][a-z]{2,}\b", text)
    seeds.extend(single_camel)

    # Short all-caps acronyms
    acronyms = re.findall(
        r"\b(?:IO|DB|URL|API|SDK|CLI|GUI|HTTP|JSON|XML|SQL|CSS|HTML)\b", text
    )
    seeds.extend(acronyms)

    # snake_case symbols: user_auth, my_function, etc.
    snake_case = re.findall(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b", text)
    seeds.extend(snake_case)

    # dot.notation: module.Class.method
    dot_notation = re.findall(r"\b\w+(?:\.\w+)+\b", text)
    seeds.extend(dot_notation)

    # Extract module names from file_set patterns
    for pattern in wo.file_set:
        if "/" in pattern:
            parts = pattern.replace("**/", "").replace("*.py", "").replace("*.ts", "")
            parts = parts.strip("/").replace("/", ".")
            if parts and not parts.startswith("*"):
                seeds.append(parts)

    # Deduplicate while preserving order
    seen = set()
    unique_seeds = []
    for seed in seeds:
        if seed not in seen and len(seed) > 2:
            seen.add(seed)
            unique_seeds.append(seed)

    return unique_seeds


def extract_files_from_file_set(
    wo: WorkOrder,
    repo_root: Path,
) -> list[str]:
    """Expand file_set globs to actual file paths.

    Args:
        wo: WorkOrder with file_set patterns
        repo_root: Repository root for glob expansion

    Returns:
        List of relative file paths
    """
    files: list[str] = []

    for pattern in wo.file_set:
        if "**" in pattern:
            # Recursive glob
            for path in repo_root.rglob(pattern.replace("**/", "")):
                if path.is_file():
                    files.append(str(path.relative_to(repo_root)))
        elif "*" in pattern:
            # Simple glob
            for path in repo_root.glob(pattern):
                if path.is_file():
                    files.append(str(path.relative_to(repo_root)))
        else:
            # Direct path
            path = repo_root / pattern
            if path.exists() and path.is_file():
                files.append(pattern)

    # Also add create_files
    files.extend(wo.create_files)

    # Deterministic dedup: preserve order, then sort for reproducibility
    seen = set()
    unique_files = []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique_files.append(f)
    return sorted(unique_files)[:50]


# =============================================================================
# Context Item Priority
# =============================================================================


class ContextPriority:
    """Priority levels for context items (lower = higher priority)."""

    CONTRACT = 0  # Acceptance criteria, constraints
    DEFINITION = 10  # Symbol definitions
    HOVER = 20  # Hover docs / signatures
    REFERENCE = 30  # References (capped)
    EXCERPT = 40  # Additional file excerpts


@dataclass
class PrioritizedContextItem:
    """Context item with priority for deterministic ordering."""

    priority: int
    item: ContextItem
    sort_key: str = ""

    def __post_init__(self):
        if not self.sort_key:
            self.sort_key = f"{self.item.path}:{self.item.excerpt_hash}"


# =============================================================================
# WorkOrderContextBuilder
# =============================================================================


class WorkOrderContextBuilder:
    """Builds context packs for WorkOrders.

    Currently uses file-based context building.
    Full SCIP integration is planned for a future milestone.

    Example:
        >>> builder = WorkOrderContextBuilder(repo_root=Path("/project"))
        >>> pack = await builder.build_for_work_order(work_order)
    """

    def __init__(
        self,
        repo_root: Path,
        config: Optional[ContextBuilderConfig] = None,
    ):
        """Initialize WorkOrderContextBuilder.

        Args:
            repo_root: Repository root path
            config: Configuration options
        """
        self.repo_root = Path(repo_root)
        self.config = config or ContextBuilderConfig()
        self._char_budget = self.config.max_chars
        self._chars_used = 0

    async def build_for_work_order(
        self,
        wo: WorkOrder,
        commit_sha: Optional[str] = None,
    ) -> ContextPack:
        """Build context pack for a WorkOrder.

        Args:
            wo: WorkOrder to build context for
            commit_sha: Specific commit (for metadata only)

        Returns:
            ContextPack with prioritized context
        """
        # Reset budget tracking
        self._chars_used = 0

        # Collect prioritized items
        items: list[PrioritizedContextItem] = []

        # Priority 0: Contract context (acceptance criteria, constraints)
        contract_item = self._build_contract_context(wo)
        if contract_item:
            items.append(
                PrioritizedContextItem(
                    priority=ContextPriority.CONTRACT,
                    item=contract_item,
                    sort_key="000_contract",
                )
            )

        # Extract seeds and files
        seeds = extract_seeds_from_work_order(wo)
        files = extract_files_from_file_set(wo, self.repo_root)

        # Build file-based context (fallback mode)
        fallback_items = self._build_fallback_excerpts(files)
        items.extend(fallback_items)

        # Search-based context
        search_items = self._build_search_based_context(seeds, files)
        items.extend(search_items)

        # Sort by priority, then by sort_key for determinism
        items.sort(key=lambda x: (x.priority, x.sort_key))

        # Build pack with budget
        pack = ContextPack(
            budget_tokens=int(self.config.max_chars / self.config.chars_per_token),
            metadata={"commit_sha": commit_sha, "work_order_id": wo.id},
        )

        for pitem in items:
            if self._chars_used + len(pitem.item.content) > self._char_budget:
                pack.truncated = True
                break

            if pack.add(pitem.item):
                self._chars_used += len(pitem.item.content)

        return pack

    def _build_contract_context(self, wo: WorkOrder) -> Optional[ContextItem]:
        """Build contract context item from WorkOrder."""
        lines = [
            f"## WorkOrder: {wo.id}",
            f"### Objective\n{wo.objective}",
        ]

        if wo.constraints:
            lines.append("### Constraints")
            lines.extend([f"- {c}" for c in wo.constraints])

        if wo.expected_delta:
            lines.append("### Expected Delta")
            delta = wo.expected_delta
            if delta.forbid_new_deps:
                lines.append(f"- Forbid new deps: {delta.forbid_new_deps}")
            if delta.allow_symbol_changes:
                lines.append(f"- Allow symbol changes: {delta.allow_symbol_changes}")
            if delta.forbid_new_public_exports:
                lines.append("- Forbid new public exports: True")
            if delta.max_files_modified:
                lines.append(f"- Max files modified: {delta.max_files_modified}")

        if wo.file_set:
            lines.append("### Allowed Files")
            lines.extend([f"- {f}" for f in wo.file_set[:10]])
            if len(wo.file_set) > 10:
                lines.append(f"- ... and {len(wo.file_set) - 10} more")

        if wo.evidence_required:
            lines.append("### Evidence Required")
            if wo.evidence_required.tests:
                lines.append(f"- Tests: {wo.evidence_required.tests}")
            if wo.evidence_required.typecheck:
                lines.append("- Typecheck: True")
            if wo.evidence_required.lint:
                lines.append("- Lint: True")

        content = "\n".join(lines)

        return ContextItem(
            source="user",
            path="",
            content=content,
            relevance_score=1.0,
            token_count=int(len(content) / self.config.chars_per_token),
            metadata={"work_order_id": wo.id},
        )

    def _build_fallback_excerpts(
        self,
        files: list[str],
    ) -> list[PrioritizedContextItem]:
        """Build file excerpts as context."""
        items: list[PrioritizedContextItem] = []
        max_lines_per_file = 100

        for file_path in files[:15]:
            full_path = self.repo_root / file_path
            if not full_path.exists():
                continue

            try:
                with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                    content_lines = f.readlines()

                if file_path.endswith(".py"):
                    excerpt = self._extract_python_definitions(
                        content_lines, max_lines_per_file
                    )
                else:
                    excerpt = "".join(content_lines[:max_lines_per_file])

                content = f"### File: {file_path}\n```\n{excerpt}\n```"
                if len(content_lines) > max_lines_per_file:
                    content += f"\n... ({len(content_lines) - max_lines_per_file} more lines)"

                items.append(
                    PrioritizedContextItem(
                        priority=ContextPriority.DEFINITION,
                        item=ContextItem(
                            source="file",
                            path=file_path,
                            content=content,
                            relevance_score=0.7,
                            token_count=int(len(content) / self.config.chars_per_token),
                            metadata={"total_lines": len(content_lines)},
                        ),
                        sort_key=f"010_{file_path}",
                    )
                )

            except (OSError, IOError):
                continue

        return items

    def _extract_python_definitions(
        self,
        lines: list[str],
        max_lines: int,
    ) -> str:
        """Extract class and function definitions from Python code."""
        result_lines = []
        in_def = False
        def_lines: list[str] = []
        def_indent = 0

        for i, line in enumerate(lines):
            if len(result_lines) >= max_lines:
                break

            stripped = line.lstrip()
            current_indent = len(line) - len(stripped)

            if stripped.startswith(("class ", "def ", "async def ")):
                if def_lines:
                    result_lines.extend(def_lines[:20])
                    if len(def_lines) > 20:
                        result_lines.append("        # ... (truncated)\n")

                in_def = True
                def_lines = [line]
                def_indent = current_indent

            elif in_def:
                if stripped and current_indent <= def_indent and not stripped.startswith(
                    "#"
                ):
                    result_lines.extend(def_lines[:20])
                    if len(def_lines) > 20:
                        result_lines.append("        # ... (truncated)\n")
                    def_lines = []
                    in_def = False

                    if stripped.startswith(("class ", "def ", "async def ")):
                        in_def = True
                        def_lines = [line]
                        def_indent = current_indent
                else:
                    def_lines.append(line)
            else:
                if i < 30:
                    result_lines.append(line)

        if def_lines:
            result_lines.extend(def_lines[:20])

        return "".join(result_lines)

    def _build_search_based_context(
        self,
        seeds: list[str],
        files: list[str],
    ) -> list[PrioritizedContextItem]:
        """Build context by searching for seed terms in files."""
        items: list[PrioritizedContextItem] = []
        seen_matches: set[str] = set()
        search_seeds = seeds[:10]

        for file_path in files[:10]:
            full_path = self.repo_root / file_path
            if not full_path.exists():
                continue

            try:
                with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()

                matches: list[tuple[int, str, str]] = []

                for line_num, line in enumerate(lines, 1):
                    for seed in search_seeds:
                        if seed.lower() in line.lower():
                            match_key = f"{file_path}:{line_num}"
                            if match_key not in seen_matches:
                                seen_matches.add(match_key)
                                matches.append((line_num, seed, line.rstrip()))

                if matches:
                    content_parts = [f"### Matches in {file_path}"]

                    for line_num, seed, _ in matches[:5]:
                        start = max(0, line_num - 3)
                        end = min(len(lines), line_num + 2)
                        context_lines = lines[start:end]

                        content_parts.append(
                            f"\n**Match for '{seed}' at line {line_num}:**"
                        )
                        content_parts.append(f"```\n{''.join(context_lines)}```")

                    content = "\n".join(content_parts)

                    items.append(
                        PrioritizedContextItem(
                            priority=ContextPriority.REFERENCE,
                            item=ContextItem(
                                source="search",
                                path=file_path,
                                content=content,
                                relevance_score=0.5,
                                token_count=int(
                                    len(content) / self.config.chars_per_token
                                ),
                                metadata={
                                    "seeds_matched": [m[1] for m in matches[:5]]
                                },
                            ),
                            sort_key=f"030_{file_path}",
                        )
                    )

            except (OSError, IOError):
                continue

        return items

    def compute_pack_hash(self, pack: ContextPack) -> str:
        """Compute deterministic hash of pack contents."""
        sorted_hashes = sorted(item.excerpt_hash for item in pack.items)
        combined = "|".join(sorted_hashes)

        if pack.metadata:
            commit_sha = pack.metadata.get("commit_sha", "")
            if commit_sha:
                combined = f"{commit_sha}|{combined}"

        return hashlib.sha256(combined.encode()).hexdigest()[:16]


# =============================================================================
# Convenience Functions
# =============================================================================


async def build_context_for_work_order(
    wo: WorkOrder,
    repo_root: Path,
    config: Optional[ContextBuilderConfig] = None,
) -> ContextPack:
    """Build context pack for a WorkOrder.

    Args:
        wo: WorkOrder to build context for
        repo_root: Repository root
        config: Builder configuration

    Returns:
        ContextPack with context

    Example:
        >>> pack = await build_context_for_work_order(
        ...     work_order,
        ...     Path("/project"),
        ... )
        >>> print(pack.to_prompt_text())
    """
    builder = WorkOrderContextBuilder(
        repo_root=repo_root,
        config=config,
    )

    return await builder.build_for_work_order(wo)


__all__ = [
    # Configuration
    "ContextBuilderConfig",
    # Context items
    "ContextItem",
    "ContextPack",
    "ContextPriority",
    "PrioritizedContextItem",
    # Builder
    "WorkOrderContextBuilder",
    # Utility functions
    "detect_indexer_from_files",
    "extract_seeds_from_work_order",
    "extract_files_from_file_set",
    "build_context_for_work_order",
]
