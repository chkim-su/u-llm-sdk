"""LLM provider implementations.

This module provides:
- BaseProvider: Abstract base class for all CLI providers
- ClaudeProvider: Claude CLI provider
- CodexProvider: Codex CLI provider
- GeminiProvider: Gemini CLI provider
- GeminiAPIProvider: Gemini API provider (direct API access)
- InterventionHook: Protocol for PGSDK intervention hooks
- NoOpHook: No-operation hook for testing

CLI vs API Providers:
    CLI providers (ClaudeProvider, CodexProvider, GeminiProvider) wrap their
    respective CLI tools and execute them as subprocesses.

    API providers (GeminiAPIProvider) call the API directly without subprocess.
    Currently only Gemini has an API provider.

Model Naming:
    - GeminiProvider (CLI): Uses MODEL_TIERS (gemini-3-pro)
    - GeminiAPIProvider (API): Uses API_MODEL_TIERS (gemini-3-pro-preview)
"""

from u_llm_sdk.llm.providers.base import BaseProvider
from u_llm_sdk.llm.providers.claude import ClaudeProvider
from u_llm_sdk.llm.providers.codex import CodexProvider
from u_llm_sdk.llm.providers.gemini import GeminiProvider
from u_llm_sdk.llm.providers.gemini_api import GeminiAPIProvider
from u_llm_sdk.llm.providers.hooks import InterventionHook, NoOpHook

__all__ = [
    "BaseProvider",
    "ClaudeProvider",
    "CodexProvider",
    "GeminiProvider",
    "GeminiAPIProvider",
    "InterventionHook",
    "NoOpHook",
]
