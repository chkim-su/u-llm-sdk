"""LLM module containing client, providers, orchestration, and parsers.

This module provides:
- LLM: Async LLM client
- LLMSync: Synchronous LLM client
- Provider implementations: ClaudeProvider, CodexProvider, GeminiProvider
- Unified JSON parsers: UnifiedResponse, parse_claude_json, parse_gemini_json, etc.
"""

from u_llm_sdk.llm.client import LLM, LLMSync, create_llm
from u_llm_sdk.llm.parsers import (
    UnifiedResponse,
    parse_claude_json,
    parse_claude_stream_json,
    parse_codex_jsonl,
    parse_gemini_json,
)
from u_llm_sdk.llm.providers import (
    BaseProvider,
    ClaudeProvider,
    CodexProvider,
    GeminiProvider,
    InterventionHook,
    NoOpHook,
)

__all__ = [
    # Client
    "LLM",
    "LLMSync",
    "create_llm",
    # Providers
    "BaseProvider",
    "ClaudeProvider",
    "CodexProvider",
    "GeminiProvider",
    # Hooks
    "InterventionHook",
    "NoOpHook",
    # Parsers
    "UnifiedResponse",
    "parse_claude_json",
    "parse_gemini_json",
    "parse_codex_jsonl",
    "parse_claude_stream_json",
]
