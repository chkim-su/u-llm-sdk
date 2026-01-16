"""Chronicle Integration for U-llm-sdk.

This module provides integration between U-llm-sdk's InterventionHook
system and MV-rag's Chronicle storage.

Components:
    ChronicleAdapter: InterventionHook implementation that records
                      ExecutionRecord and FailureRecord to Chronicle.

Design Philosophy:
    - Chronicle is OPTIONAL: Providers work without Chronicle (backward compatible)
    - Composable: ChronicleAdapter can wrap existing hooks (like RAGClient)
    - Minimal overhead: Uses local storage (SQLite), no network latency
    - Deterministic IDs: run_id maps to exec_{run_id} for canonicalization

Usage:
    >>> from u_llm_sdk.chronicle import ChronicleAdapter
    >>> from mv_rag.chronicle import ChronicleStore
    >>>
    >>> store = ChronicleStore("/path/to/chronicle.db")
    >>> adapter = ChronicleAdapter(store)
    >>>
    >>> # Use with LLM
    >>> config = LLMConfig(provider=Provider.CLAUDE)
    >>> async with LLM(config, intervention_hook=adapter) as llm:
    ...     result = await llm.run("Fix the bug")
"""

from .adapter import ChronicleAdapter, ChronicleConfig

__all__ = [
    "ChronicleAdapter",
    "ChronicleConfig",
]
