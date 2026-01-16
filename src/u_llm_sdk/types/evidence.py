"""Evidence types for multi-provider code intelligence injection.

This module defines the standard unit for injectable evidence and the protocol
that evidence providers must implement.

Architecture:
    - EvidenceChunk: Standard unit of evidence with provenance
    - EvidenceProvider: Protocol for evidence sources (SCIP, Forensics, GraphCode)
    - Stage: Progressive disclosure stages (INDEX, DETAILS, DEEP_DIVE)

ID Namespace Convention:
    All evidence IDs must follow the format: "{provider}:{kind}:{local_id}"
    Examples:
        - "scip:symbol:pkg.mod.MyClass"
        - "forensics:event:exec_abc123"
        - "graphcode:node:fn_hash_xyz"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

# Progressive disclosure stages with token budgets:
#   INDEX: ~800 tokens - Repository map, entry points, key modules
#   DETAILS: ~2500 tokens - 1-hop expansion, neighbors, related symbols
#   DEEP_DIVE: ~4500 tokens - Full source excerpts, execution traces
Stage = Literal["INDEX", "DETAILS", "DEEP_DIVE"]

# Token budgets per stage (reference values)
STAGE_TOKEN_BUDGETS: dict[Stage, int] = {
    "INDEX": 800,
    "DETAILS": 2500,
    "DEEP_DIVE": 4500,
}


@dataclass(frozen=True)
class EvidenceSpan:
    """Source location span for an evidence chunk.

    Attributes:
        file_path: Relative path from project root
        start_line: 1-based start line (None if file-level)
        end_line: 1-based end line (None if file-level)
    """

    file_path: str
    start_line: int | None = None
    end_line: int | None = None

    def __str__(self) -> str:
        if self.start_line is None:
            return self.file_path
        if self.end_line is None or self.end_line == self.start_line:
            return f"{self.file_path}:{self.start_line}"
        return f"{self.file_path}:{self.start_line}-{self.end_line}"


@dataclass(frozen=True)
class EvidenceChunk:
    """Standard unit of injectable evidence with provenance.

    Every chunk must have:
    - Unique ID with provider namespace (e.g., "scip:symbol:pkg.mod.fn")
    - Provider name for policy decisions
    - Kind for filtering (symbol, call_edge, excerpt, test, error_trace, etc.)
    - Title and preview for ranking/display

    Attributes:
        id: Namespaced ID in format "{provider}:{kind}:{local_id}"
        provider: Evidence source name (scip, forensics, graphcode)
        kind: Evidence type (symbol, call_edge, excerpt, test, error_trace)
        title: Human-readable title for display
        preview: Short preview text for ranking (max ~160 chars)
        content: Full content (may be None until DEEP_DIVE stage)
        span: Source location if applicable
        score: Provider-internal relevance score (0.0 if not scored)
        meta: Additional provider-specific metadata
    """

    id: str
    provider: str
    kind: str
    title: str
    preview: str
    content: str | None = None
    span: EvidenceSpan | None = None
    score: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate ID format."""
        # ID must be "{provider}:{kind}:{local_id}" or at minimum "{provider}:{local_id}"
        parts = self.id.split(":", 2)
        if len(parts) < 2:
            raise ValueError(
                f"Evidence ID must be namespaced: '{self.id}'. "
                f"Expected format: '{{provider}}:{{kind}}:{{local_id}}' or '{{provider}}:{{local_id}}'"
            )
        if parts[0] != self.provider:
            raise ValueError(
                f"Evidence ID namespace '{parts[0]}' does not match provider '{self.provider}'"
            )

    @property
    def local_id(self) -> str:
        """Extract local ID without provider namespace."""
        parts = self.id.split(":", 2)
        return parts[-1] if len(parts) > 1 else self.id

    def with_content(self, content: str) -> EvidenceChunk:
        """Create a new chunk with content filled (for DEEP_DIVE upgrade)."""
        return EvidenceChunk(
            id=self.id,
            provider=self.provider,
            kind=self.kind,
            title=self.title,
            preview=self.preview,
            content=content,
            span=self.span,
            score=self.score,
            meta=self.meta,
        )


@runtime_checkable
class EvidenceProvider(Protocol):
    """Protocol for evidence sources.

    Providers supply evidence at three progressive stages:
    1. INDEX: High-level repository map (entry points, key modules)
    2. DETAILS: Expanded context (neighbors, related symbols)
    3. DEEP_DIVE: Full source excerpts and execution traces

    Implementation notes:
    - All methods are synchronous for MV-rag pipeline compatibility
    - Wrap with asyncio.to_thread() if async context needed
    - Return empty list if stage not supported (not an error)

    ID conventions:
    - All returned chunk IDs must start with "{provider_name}:"
    - Use consistent kind values: symbol, call_edge, excerpt, test, error_trace
    """

    @property
    def name(self) -> str:
        """Provider name used in evidence ID namespace."""
        ...

    def query_index(
        self,
        query: str,
        limit: int = 30,
        preview_chars: int = 160,
    ) -> list[EvidenceChunk]:
        """Query for INDEX stage evidence (~800 tokens).

        Returns high-level repository structure:
        - Entry points and main modules
        - Key classes and functions
        - Package organization

        Args:
            query: Natural language query or task description
            limit: Maximum number of chunks to return
            preview_chars: Maximum preview length per chunk

        Returns:
            List of EvidenceChunks without full content
        """
        ...

    def query_details(
        self,
        seed_ids: list[str],
        query: str,
        neighbor_limit: int = 80,
    ) -> list[EvidenceChunk]:
        """Query for DETAILS stage evidence (~2500 tokens).

        Expands from seed IDs to related context:
        - Callers and callees (1-hop)
        - Related tests
        - Type definitions
        - Import relationships

        Args:
            seed_ids: Namespaced IDs from previous stage to expand
            query: Natural language query for relevance ranking
            neighbor_limit: Maximum neighbors per seed

        Returns:
            List of EvidenceChunks with expanded context
        """
        ...

    def query_deep_dive(
        self,
        seed_ids: list[str],
        query: str,
        max_bytes: int = 2048,
    ) -> list[EvidenceChunk]:
        """Query for DEEP_DIVE stage evidence (~4500 tokens).

        Returns full source excerpts:
        - Complete function/method bodies
        - Execution traces
        - Error context with stack frames

        Args:
            seed_ids: Namespaced IDs from previous stage
            query: Natural language query for relevance ranking
            max_bytes: Maximum bytes per content field

        Returns:
            List of EvidenceChunks with full content populated
        """
        ...


def filter_chunks_by_provider(
    chunks: list[EvidenceChunk],
    provider: str,
) -> list[EvidenceChunk]:
    """Filter chunks by provider name."""
    return [c for c in chunks if c.provider == provider]


def filter_chunks_by_kind(
    chunks: list[EvidenceChunk],
    kinds: set[str],
) -> list[EvidenceChunk]:
    """Filter chunks by kind."""
    return [c for c in chunks if c.kind in kinds]


def extract_seed_ids(
    chunks: list[EvidenceChunk],
    provider: str | None = None,
) -> list[str]:
    """Extract IDs from chunks for use as seeds in next stage.

    Args:
        chunks: Chunks to extract IDs from
        provider: If specified, only extract IDs from this provider

    Returns:
        List of namespaced IDs
    """
    if provider:
        return [c.id for c in chunks if c.provider == provider]
    return [c.id for c in chunks]


__all__ = [
    # Types
    "Stage",
    "STAGE_TOKEN_BUDGETS",
    # Data classes
    "EvidenceSpan",
    "EvidenceChunk",
    # Protocol
    "EvidenceProvider",
    # Utilities
    "filter_chunks_by_provider",
    "filter_chunks_by_kind",
    "extract_seed_ids",
]
