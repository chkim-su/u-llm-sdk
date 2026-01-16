"""Advanced SDK module for unified multi-provider orchestration.

This module provides advanced features for LLM orchestration that work
across all supported providers (Claude, Codex, Gemini).

Core Components:
    UnifiedAdvanced: Main async client with agent orchestration
    UnifiedAdvancedSync: Synchronous wrapper for non-async contexts
    AdvancedConfig: Provider-aware configuration
    AgentDefinition: Specialized agent definitions with optional provider override

Example:
    >>> from u_llm_sdk.advanced import UnifiedAdvanced, AdvancedConfig, AgentDefinition
    >>> from u_llm_sdk.types import Provider, ModelTier
    >>>
    >>> # Basic usage
    >>> async with UnifiedAdvanced(provider=Provider.CLAUDE) as client:
    ...     result = await client.run("Hello world")
    >>>
    >>> # With agent
    >>> planner = AgentDefinition(
    ...     name="planner",
    ...     description="Task planning",
    ...     system_prompt="You are a planning expert.",
    ...     tier=ModelTier.HIGH,
    ... )
    >>> async with UnifiedAdvanced() as client:
    ...     result = await client.run_with_agent("Plan feature", planner)
    >>>
    >>> # Agent with different provider
    >>> codex_agent = AgentDefinition(
    ...     name="analyzer",
    ...     description="Code analysis",
    ...     system_prompt="Analyze code.",
    ...     provider=Provider.CODEX,  # Override client's provider
    ... )
"""

from u_llm_sdk.advanced.config import (
    AdvancedConfig,
    AgentDefinition,
)
from u_llm_sdk.advanced.client import (
    UnifiedAdvanced,
    UnifiedAdvancedSync,
)

__all__ = [
    # Client
    "UnifiedAdvanced",
    "UnifiedAdvancedSync",
    # Config
    "AdvancedConfig",
    "AgentDefinition",
]
