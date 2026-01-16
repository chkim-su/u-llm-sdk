"""High-level LLM Client with async and sync interfaces.

Provides a unified interface for all LLM providers.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncIterator, Optional, Union

from u_llm_sdk.types import LLMResult, Provider

from u_llm_sdk.config import LLMConfig
from u_llm_sdk.core.discovery import available_providers
from u_llm_sdk.llm.providers.base import BaseProvider
from u_llm_sdk.llm.providers.claude import ClaudeProvider
from u_llm_sdk.llm.providers.codex import CodexProvider
from u_llm_sdk.llm.providers.gemini import GeminiProvider
from u_llm_sdk.rag_client import create_rag_client_if_available

if TYPE_CHECKING:
    from u_llm_sdk.llm.providers.hooks import InterventionHook

logger = logging.getLogger(__name__)

# Provider class mapping
PROVIDER_CLASSES: dict[Provider, type[BaseProvider]] = {
    Provider.CLAUDE: ClaudeProvider,
    Provider.CODEX: CodexProvider,
    Provider.GEMINI: GeminiProvider,
}


class LLM:
    """Async LLM client with context manager support.

    Provides a unified async interface for all LLM providers.

    Basic Usage:
        >>> from u_llm_sdk import LLM, LLMConfig, Provider
        >>> config = LLMConfig(provider=Provider.CLAUDE)
        >>> async with LLM(config) as llm:
        ...     result = await llm.run("Hello, world!")
        ...     print(result.text)

    Auto RAG Integration (default):
        >>> async with LLM(config) as llm:
        ...     # Automatically uses MV-rag if installed and server is running
        ...     result = await llm.run("Hello!")

    Disable Auto RAG:
        >>> async with LLM(config, auto_rag=False) as llm:
        ...     result = await llm.run("Hello!")  # No RAG integration

    Session Resume:
        >>> async with LLM(config).resume("session-123") as llm:
        ...     result = await llm.run("Continue...")

    Auto Provider Selection:
        >>> async with LLM.auto() as llm:
        ...     result = await llm.run("Hello!")  # Uses first available provider
    """

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        intervention_hook: Optional["InterventionHook"] = None,
        auto_rag: bool = True,
    ):
        """Initialize LLM client.

        Args:
            config: LLM configuration (uses defaults if None)
            intervention_hook: Optional hook for context injection/feedback.
                              If None and auto_rag=True, will auto-detect MV-rag.
            auto_rag: If True, automatically detect and use MV-rag if available.
                     Ignored if intervention_hook is explicitly provided.
        """
        self._config = config or LLMConfig()
        self._auto_rag = auto_rag
        self._owns_rag_client = False  # Track if we created the RAGClient

        # Auto-detect MV-rag if no hook provided and auto_rag enabled
        if intervention_hook is None and auto_rag:
            intervention_hook = create_rag_client_if_available()
            if intervention_hook is not None:
                self._owns_rag_client = True
                logger.info("Auto-detected MV-rag, RAG integration enabled")

        self._intervention_hook = intervention_hook
        self._provider: Optional[BaseProvider] = None
        self._session_id: Optional[str] = None

    @classmethod
    def auto(
        cls,
        config: Optional[LLMConfig] = None,
        intervention_hook: Optional["InterventionHook"] = None,
        auto_rag: bool = True,
    ) -> "LLM":
        """Create client with auto-selected provider.

        Selects the first available provider from the priority list:
        Claude > Codex > Gemini

        Args:
            config: Base configuration (provider will be overwritten)
            intervention_hook: Optional hook for context injection/feedback
            auto_rag: If True, automatically detect and use MV-rag if available

        Returns:
            LLM instance with available provider

        Raises:
            ProviderNotAvailableError: If no providers are available
        """
        from u_llm_sdk.types import ProviderNotAvailableError

        available = available_providers()
        if not available:
            raise ProviderNotAvailableError(
                "No LLM providers available. Install claude, codex, or gemini CLI."
            )

        # Priority: Claude > Codex > Gemini
        priority = [Provider.CLAUDE, Provider.CODEX, Provider.GEMINI]
        selected = None
        for provider in priority:
            if provider in available:
                selected = provider
                break

        if selected is None:
            selected = available[0]

        base_config = config or LLMConfig()
        final_config = base_config.with_provider(selected)

        return cls(config=final_config, intervention_hook=intervention_hook, auto_rag=auto_rag)

    def resume(self, session_id: str) -> "LLM":
        """Set session ID for conversation continuity.

        Args:
            session_id: Session ID from previous execution

        Returns:
            Self for method chaining
        """
        self._session_id = session_id
        return self

    @property
    def session_id(self) -> Optional[str]:
        """Get current session ID."""
        if self._provider:
            return self._provider.session_id
        return self._session_id

    @property
    def provider(self) -> Optional[BaseProvider]:
        """Get the underlying provider instance."""
        return self._provider

    @property
    def config(self) -> LLMConfig:
        """Get the configuration."""
        return self._config

    async def __aenter__(self) -> "LLM":
        """Enter async context manager."""
        self._provider = self._create_provider()
        if self._session_id:
            self._provider.resume(self._session_id)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context manager."""
        # Clean up auto-created RAGClient
        if self._owns_rag_client and self._intervention_hook is not None:
            try:
                await self._intervention_hook.close()
            except Exception as e:
                logger.warning(f"Error closing RAGClient: {e}")

        self._provider = None

    def _create_provider(self) -> BaseProvider:
        """Create provider instance based on config."""
        provider_class = PROVIDER_CLASSES.get(self._config.provider)
        if provider_class is None:
            from u_llm_sdk.types import ProviderNotFoundError

            raise ProviderNotFoundError(
                self._config.provider.value,
                f"Unknown provider: {self._config.provider.value}",
            )

        return provider_class(
            config=self._config,
            intervention_hook=self._intervention_hook,
        )

    def _ensure_provider(self) -> BaseProvider:
        """Ensure provider is initialized."""
        if self._provider is None:
            self._provider = self._create_provider()
            if self._session_id:
                self._provider.resume(self._session_id)
        return self._provider

    async def run(
        self,
        prompt: str,
        *,
        session_id: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> LLMResult:
        """Execute a prompt and return the result.

        Args:
            prompt: Prompt to execute
            session_id: Session ID to resume (overrides instance session)
            timeout: Timeout in seconds (overrides config)

        Returns:
            LLMResult with unified schema
        """
        provider = self._ensure_provider()
        return await provider.run(prompt, session_id=session_id, timeout=timeout)

    async def stream(
        self,
        prompt: str,
        *,
        session_id: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream execution results.

        Args:
            prompt: Prompt to execute
            session_id: Session ID to resume
            timeout: Timeout in seconds (overrides config)

        Yields:
            Event dictionaries from CLI output
        """
        provider = self._ensure_provider()
        async for event in provider.stream(prompt, session_id=session_id, timeout=timeout):
            yield event

    async def parallel_run(
        self,
        prompts: list[str],
        *,
        timeout: Optional[float] = None,
    ) -> list[LLMResult]:
        """Execute multiple prompts in parallel.

        Args:
            prompts: List of prompts to execute
            timeout: Timeout per execution

        Returns:
            List of LLMResults in same order as prompts
        """
        provider = self._ensure_provider()
        return await provider.parallel_run(prompts, timeout=timeout)


class LLMSync:
    """Synchronous wrapper around LLM.

    Provides a synchronous interface for environments where async is not convenient.

    Basic Usage:
        >>> from u_llm_sdk import LLMSync, LLMConfig, Provider
        >>> config = LLMConfig(provider=Provider.CLAUDE)
        >>> with LLMSync(config) as llm:
        ...     result = llm.run("Hello, world!")
        ...     print(result.text)

    Auto RAG Integration (default):
        >>> with LLMSync(config) as llm:
        ...     # Automatically uses MV-rag if installed and server is running
        ...     result = llm.run("Hello!")

    Session Resume:
        >>> with LLMSync(config).resume("session-123") as llm:
        ...     result = llm.run("Continue...")

    Auto Provider Selection:
        >>> with LLMSync.auto() as llm:
        ...     result = llm.run("Hello!")  # Uses first available provider
    """

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        intervention_hook: Optional["InterventionHook"] = None,
        auto_rag: bool = True,
    ):
        """Initialize sync LLM client.

        Args:
            config: LLM configuration (uses defaults if None)
            intervention_hook: Optional hook for context injection/feedback.
                              If None and auto_rag=True, will auto-detect MV-rag.
            auto_rag: If True, automatically detect and use MV-rag if available.
        """
        self._llm = LLM(config=config, intervention_hook=intervention_hook, auto_rag=auto_rag)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    @classmethod
    def auto(
        cls,
        config: Optional[LLMConfig] = None,
        intervention_hook: Optional["InterventionHook"] = None,
        auto_rag: bool = True,
    ) -> "LLMSync":
        """Create client with auto-selected provider.

        Args:
            config: Base configuration (provider will be overwritten)
            intervention_hook: Optional hook for context injection/feedback
            auto_rag: If True, automatically detect and use MV-rag if available

        Returns:
            LLMSync instance with available provider
        """
        llm = LLM.auto(config=config, intervention_hook=intervention_hook, auto_rag=auto_rag)
        instance = cls.__new__(cls)
        instance._llm = llm
        instance._loop = None
        return instance

    def resume(self, session_id: str) -> "LLMSync":
        """Set session ID for conversation continuity.

        Args:
            session_id: Session ID from previous execution

        Returns:
            Self for method chaining
        """
        self._llm.resume(session_id)
        return self

    @property
    def session_id(self) -> Optional[str]:
        """Get current session ID."""
        return self._llm.session_id

    @property
    def provider(self) -> Optional[BaseProvider]:
        """Get the underlying provider instance."""
        return self._llm.provider

    @property
    def config(self) -> LLMConfig:
        """Get the configuration."""
        return self._llm.config

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create event loop."""
        if self._loop is None or self._loop.is_closed():
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
        return self._loop

    def _run_sync(self, coro):
        """Run coroutine synchronously."""
        loop = self._get_loop()
        try:
            if loop.is_running():
                # If we're in an async context, use nest_asyncio pattern
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, coro)
                    return future.result()
            else:
                return loop.run_until_complete(coro)
        except RuntimeError:
            # Fallback to asyncio.run
            return asyncio.run(coro)

    def __enter__(self) -> "LLMSync":
        """Enter context manager."""
        self._run_sync(self._llm.__aenter__())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager."""
        self._run_sync(self._llm.__aexit__(exc_type, exc_val, exc_tb))

    def run(
        self,
        prompt: str,
        *,
        session_id: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> LLMResult:
        """Execute a prompt and return the result.

        Args:
            prompt: Prompt to execute
            session_id: Session ID to resume (overrides instance session)
            timeout: Timeout in seconds (overrides config)

        Returns:
            LLMResult with unified schema
        """
        return self._run_sync(
            self._llm.run(prompt, session_id=session_id, timeout=timeout)
        )

    def parallel_run(
        self,
        prompts: list[str],
        *,
        timeout: Optional[float] = None,
    ) -> list[LLMResult]:
        """Execute multiple prompts in parallel.

        Args:
            prompts: List of prompts to execute
            timeout: Timeout per execution

        Returns:
            List of LLMResults in same order as prompts
        """
        return self._run_sync(self._llm.parallel_run(prompts, timeout=timeout))


@asynccontextmanager
async def create_llm(
    config: Optional[LLMConfig] = None,
    intervention_hook: Optional["InterventionHook"] = None,
    auto_select: bool = False,
) -> AsyncIterator[LLM]:
    """Create LLM instance as async context manager.

    Helper function for creating LLM instances.

    Args:
        config: LLM configuration
        intervention_hook: Optional hook for context injection/feedback
        auto_select: If True, auto-select available provider

    Yields:
        Configured LLM instance
    """
    if auto_select:
        llm = LLM.auto(config=config, intervention_hook=intervention_hook)
    else:
        llm = LLM(config=config, intervention_hook=intervention_hook)

    async with llm:
        yield llm


__all__ = [
    "LLM",
    "LLMSync",
    "create_llm",
    "PROVIDER_CLASSES",
]
