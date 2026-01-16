"""Unified LLM SDK with multi-provider support.

This package provides a unified interface for multiple LLM providers
(Claude, Codex, Gemini) with identical output schemas.

Package Dependencies:
    - llm-types: Required. Shared types (auto-installed with U-llm-sdk)
    - MV-rag: Optional. Experience learning server (auto-detected if installed)

Basic Usage (Async):
    >>> from u_llm_sdk import LLM, LLMConfig, Provider
    >>> async with LLM(LLMConfig(provider=Provider.CLAUDE)) as llm:
    ...     result = await llm.run("Hello!")

Basic Usage (Sync):
    >>> from u_llm_sdk import LLMSync, LLMConfig, Provider
    >>> with LLMSync(LLMConfig(provider=Provider.CLAUDE)) as llm:
    ...     result = llm.run("Hello!")

Auto Provider Selection:
    >>> async with LLM.auto() as llm:
    ...     result = await llm.run("Hello!")  # Uses first available provider

MV-rag Auto-Detection (default enabled):
    MV-rag is automatically used if:
    1. MV-rag package is installed (`pip install mv-rag`)
    2. MV-rag server is running (`python -m uvicorn mv_rag.api.server:app`)

    To disable auto-detection:
    >>> async with LLM(config, auto_rag=False) as llm:
    ...     result = await llm.run("Hello!")  # No RAG integration

    To check MV-rag status:
    >>> from u_llm_sdk.rag_client import is_mv_rag_installed, is_mv_rag_running
    >>> print(is_mv_rag_installed())  # True if package installed
    >>> print(is_mv_rag_running())    # True if server responding

Presets:
    >>> from u_llm_sdk import SAFE_CONFIG, AUTO_CONFIG
    >>> config = SAFE_CONFIG.with_provider(Provider.CODEX)

Test/Dev Mode:
    >>> from u_llm_sdk import TEST_DEV_CLAUDE_CONFIG
"""

__version__ = "0.2.0"

# Re-export from llm-types for convenience
from u_llm_sdk.types import (
    # Enums
    AutoApproval,
    ModelTier,
    Provider,
    ReasoningLevel,
    SandboxMode,
    # Data models
    CodeBlock,
    CommandRun,
    FileChange,
    LLMResult,
    ResultType,
    TokenUsage,
    # Hook data
    PostActionFeedback,
    PreActionContext,
    # Exceptions
    AuthenticationError,
    ExecutionCancelledError,
    ExecutionTimeoutError,
    InvalidConfigError,
    ModelNotFoundError,
    ModelNotSpecifiedError,
    ParseError,
    ProviderNotAvailableError,
    ProviderNotFoundError,
    RAGConnectionError,
    RAGTimeoutError,
    RateLimitError,
    SessionNotFoundError,
    UnifiedLLMError,
)

# Configuration
from u_llm_sdk.config import (
    AUTO_CONFIG,
    CLAUDE_CONFIG,
    CODEX_CONFIG,
    GEMINI_CONFIG,
    SAFE_CONFIG,
    TEST_DEV_CLAUDE_CONFIG,
    TEST_DEV_CODEX_CONFIG,
    TEST_DEV_GEMINI_CONFIG,
    LLMConfig,
    create_test_dev_config,
    get_effective_reasoning,
)

# LLM Client
from u_llm_sdk.llm.client import LLM, LLMSync, create_llm

# Providers
from u_llm_sdk.llm.providers import (
    BaseProvider,
    ClaudeProvider,
    CodexProvider,
    GeminiProvider,
    GeminiAPIProvider,
    InterventionHook,
    NoOpHook,
)

# RAG Client
from u_llm_sdk.rag_client import (
    PreActionCache,
    RAGClient,
    RAGClientConfig,
    # Auto-detection utilities
    is_mv_rag_installed,
    is_mv_rag_running,
    is_mv_rag_available,
    create_rag_client_if_available,
)

# Quick utils
from u_llm_sdk.core.utils import (
    auto_run,
    auto_run_sync,
    multi_provider_run,
    parallel_run,
    parallel_run_sync,
    quick_run,
    quick_run_sync,
    quick_text,
    quick_text_sync,
    structured_run,
    structured_run_sync,
    template_run,
    template_run_sync,
)

# Session management (unified for all providers)
from u_llm_sdk.session import (
    # Base/Factory
    BaseSessionManager,
    get_session_manager,
    inject_system_prompt,
    # Provider implementations
    ClaudeSessionManager,
    CodexSessionManager,
    GeminiSessionManager,
    # Message
    SessionMessage,
    resolve_prompt,
    # Templates
    SessionTemplate,
    get_template_prompt,
    list_templates,
    create_custom_template,
)

# Advanced client (unified for all providers)
from u_llm_sdk.advanced import (
    UnifiedAdvanced,
    UnifiedAdvancedSync,
    AdvancedConfig,
    AgentDefinition,
)

# Parsers
from u_llm_sdk.llm.parsers import (
    UnifiedResponse,
    parse_claude_json,
    parse_claude_stream_json,
    parse_codex_jsonl,
    parse_gemini_json,
)

__all__ = [
    # Version
    "__version__",
    # Client
    "LLM",
    "LLMSync",
    "create_llm",
    # Providers
    "BaseProvider",
    "ClaudeProvider",
    "CodexProvider",
    "GeminiProvider",
    "GeminiAPIProvider",
    # Enums (from llm-types)
    "Provider",
    "AutoApproval",
    "SandboxMode",
    "ModelTier",
    "ReasoningLevel",
    "ResultType",
    # Data models (from llm-types)
    "LLMResult",
    "TokenUsage",
    "FileChange",
    "CommandRun",
    "CodeBlock",
    "PreActionContext",
    "PostActionFeedback",
    # Exceptions (from llm-types)
    "UnifiedLLMError",
    "ProviderNotFoundError",
    "ProviderNotAvailableError",
    "InvalidConfigError",
    "SessionNotFoundError",
    "ExecutionCancelledError",
    "ExecutionTimeoutError",
    "ParseError",
    "AuthenticationError",
    "RateLimitError",
    "ModelNotSpecifiedError",
    "ModelNotFoundError",
    "RAGConnectionError",
    "RAGTimeoutError",
    # Config
    "LLMConfig",
    "get_effective_reasoning",
    "create_test_dev_config",
    # Preset configs
    "SAFE_CONFIG",
    "AUTO_CONFIG",
    "CLAUDE_CONFIG",
    "CODEX_CONFIG",
    "GEMINI_CONFIG",
    "TEST_DEV_CLAUDE_CONFIG",
    "TEST_DEV_CODEX_CONFIG",
    "TEST_DEV_GEMINI_CONFIG",
    # RAG Client
    "RAGClient",
    "RAGClientConfig",
    "PreActionCache",
    # MV-rag auto-detection
    "is_mv_rag_installed",
    "is_mv_rag_running",
    "is_mv_rag_available",
    "create_rag_client_if_available",
    # Hooks
    "InterventionHook",
    "NoOpHook",
    # Quick utils (async)
    "quick_run",
    "quick_text",
    "auto_run",
    "parallel_run",
    "multi_provider_run",
    "structured_run",
    "template_run",
    # Quick utils (sync)
    "quick_run_sync",
    "quick_text_sync",
    "auto_run_sync",
    "parallel_run_sync",
    "structured_run_sync",
    "template_run_sync",
    # Session management
    "BaseSessionManager",
    "get_session_manager",
    "inject_system_prompt",
    "ClaudeSessionManager",
    "CodexSessionManager",
    "GeminiSessionManager",
    "SessionMessage",
    "resolve_prompt",
    "SessionTemplate",
    "get_template_prompt",
    "list_templates",
    "create_custom_template",
    # Advanced client
    "UnifiedAdvanced",
    "UnifiedAdvancedSync",
    "AdvancedConfig",
    "AgentDefinition",
    # Parsers
    "UnifiedResponse",
    "parse_claude_json",
    "parse_gemini_json",
    "parse_codex_jsonl",
    "parse_claude_stream_json",
]
