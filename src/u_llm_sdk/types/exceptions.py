"""LLM Types - Exception Classes.

This module defines all exceptions used by the unified LLM SDK.
All exceptions inherit from UnifiedLLMError for easy catching.

Usage:
    try:
        result = await llm.run("Do something")
    except UnifiedLLMError as e:
        print(f"SDK error: {e}")
"""

from __future__ import annotations


class UnifiedLLMError(Exception):
    """Base exception for all unified LLM SDK errors.

    All exceptions in this SDK inherit from this class,
    making it easy to catch all SDK-related errors.
    """
    pass


class ProviderNotFoundError(UnifiedLLMError):
    """Raised when the specified provider CLI is not installed."""

    def __init__(self, provider: str, path: str = ""):
        self.provider = provider
        self.path = path
        msg = f"{provider} CLI not found"
        if path:
            msg += f" at {path}"
        msg += f". Please install {provider} CLI first."
        super().__init__(msg)


class ProviderNotAvailableError(UnifiedLLMError):
    """Raised when the provider is not available (not configured, no API key, etc.)."""

    def __init__(self, provider: str, reason: str = ""):
        self.provider = provider
        self.reason = reason
        msg = f"{provider} is not available"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


class SessionNotFoundError(UnifiedLLMError):
    """Raised when trying to resume a non-existent session."""

    def __init__(self, session_id: str = ""):
        self.session_id = session_id
        msg = "Session not found"
        if session_id:
            msg += f": {session_id}"
        super().__init__(msg)


class ExecutionTimeoutError(UnifiedLLMError):
    """Raised when execution exceeds the timeout limit."""

    def __init__(self, timeout: float, provider: str = ""):
        self.timeout = timeout
        self.provider = provider
        msg = f"Execution timed out after {timeout}s"
        if provider:
            msg += f" ({provider})"
        super().__init__(msg)


class ExecutionCancelledError(UnifiedLLMError):
    """Raised when execution is cancelled by user."""

    def __init__(self, reason: str = "Operation cancelled by user"):
        self.reason = reason
        super().__init__(reason)


class InvalidConfigError(UnifiedLLMError):
    """Raised when configuration is invalid."""

    def __init__(self, message: str):
        super().__init__(f"Invalid configuration: {message}")


class ParseError(UnifiedLLMError):
    """Raised when unable to parse provider output.

    This usually indicates an unexpected output format from the CLI.
    """

    def __init__(self, provider: str, message: str = ""):
        self.provider = provider
        msg = f"Failed to parse {provider} output"
        if message:
            msg += f": {message}"
        super().__init__(msg)


class AuthenticationError(UnifiedLLMError):
    """Raised when authentication fails."""

    def __init__(self, provider: str, message: str = ""):
        self.provider = provider
        msg = f"Authentication failed for {provider}"
        if message:
            msg += f": {message}"
        super().__init__(msg)


class RateLimitError(UnifiedLLMError):
    """Raised when rate limit is exceeded."""

    def __init__(self, provider: str, retry_after: float = 0):
        self.provider = provider
        self.retry_after = retry_after
        msg = f"Rate limit exceeded for {provider}"
        if retry_after > 0:
            msg += f". Retry after {retry_after}s"
        super().__init__(msg)


class ModelNotFoundError(UnifiedLLMError):
    """Raised when the specified model is not available."""

    def __init__(self, model: str, provider: str = ""):
        self.model = model
        self.provider = provider
        msg = f"Model not found: {model}"
        if provider:
            msg += f" (provider: {provider})"
        super().__init__(msg)


class ModelNotSpecifiedError(UnifiedLLMError):
    """Raised when model/tier is not specified and is required.

    The SDK requires explicit model or tier specification.
    Use tier=ModelTier.HIGH or tier=ModelTier.LOW for convenience,
    or specify model directly with model="model-name".
    """

    def __init__(
        self,
        provider: str,
        available_tiers: list[str] | None = None,
        available_models: list[str] | None = None,
    ):
        self.provider = provider
        self.available_tiers = available_tiers or []
        self.available_models = available_models or []

        msg = f"Model or tier must be explicitly specified for {provider}.\n"
        msg += "\nOptions:\n"
        msg += "  1. Use tier (recommended):\n"
        msg += "     - tier=ModelTier.HIGH  (best performance, higher cost)\n"
        msg += "     - tier=ModelTier.LOW   (fast/cheap, good for simple tasks)\n"
        if self.available_models:
            msg += f"\n  2. Use specific model:\n"
            msg += f"     - model=\"{self.available_models[0]}\"" if self.available_models else ""
            for m in self.available_models[1:4]:
                msg += f", \"{m}\""
            if len(self.available_models) > 4:
                msg += f" ... (+{len(self.available_models) - 4} more)"

        super().__init__(msg)


# RAG-specific exceptions
class RAGConnectionError(UnifiedLLMError):
    """Raised when connection to MV-rag service fails."""

    def __init__(self, url: str, reason: str = ""):
        self.url = url
        self.reason = reason
        msg = f"Failed to connect to RAG service at {url}"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


class RAGTimeoutError(UnifiedLLMError):
    """Raised when RAG service request times out."""

    def __init__(self, timeout_ms: int, endpoint: str = ""):
        self.timeout_ms = timeout_ms
        self.endpoint = endpoint
        msg = f"RAG service request timed out after {timeout_ms}ms"
        if endpoint:
            msg += f" ({endpoint})"
        super().__init__(msg)
