"""Unified Advanced Client for multi-provider orchestration.

This module provides the UnifiedAdvanced client that supports
all providers (Claude, Codex, Gemini) with advanced orchestration features.
"""

from __future__ import annotations

import asyncio
import logging
from types import TracebackType
from typing import TYPE_CHECKING, Any, AsyncIterator, Optional

from u_llm_sdk.types import LLMResult, Provider

from u_llm_sdk.advanced.config import AdvancedConfig, AgentDefinition
from u_llm_sdk.config import LLMConfig
from u_llm_sdk.llm.client import LLM, LLMSync
from u_llm_sdk.session.base import get_session_manager, inject_system_prompt

if TYPE_CHECKING:
    from u_llm_sdk.llm.providers import InterventionHook
    from u_llm_sdk.session import BaseSessionManager, SessionTemplate

logger = logging.getLogger(__name__)


class UnifiedAdvanced:
    """Advanced client supporting all providers.

    Provides sophisticated orchestration features including:
    - Agent-based execution with custom system prompts
    - Parallel agent execution with concurrency control
    - Session management for context continuity
    - Template-based personas

    Example:
        >>> async with UnifiedAdvanced(provider=Provider.CLAUDE) as client:
        ...     result = await client.run("Hello!")
        >>>
        >>> # With agent
        >>> async with UnifiedAdvanced() as client:
        ...     result = await client.run_with_agent("Plan feature", planner_agent)
        >>>
        >>> # Multi-provider workflow
        >>> planner = AgentDefinition(name="planner", provider=Provider.GEMINI, ...)
        >>> executor = AgentDefinition(name="executor", provider=Provider.CLAUDE, ...)
        >>> async with UnifiedAdvanced() as client:
        ...     results = await client.run_with_agents("Implement auth", [planner, executor])
    """

    def __init__(
        self,
        provider: Optional[Provider] = None,
        config: Optional[AdvancedConfig] = None,
        intervention_hook: Optional["InterventionHook"] = None,
    ):
        """Initialize UnifiedAdvanced client.

        Args:
            provider: Default provider (overridden by config if provided)
            config: Full configuration (optional)
            intervention_hook: Optional hook for MV-rag integration
        """
        if config:
            self.config = config
        else:
            self.config = AdvancedConfig(
                provider=provider or Provider.CLAUDE,
            )

        if intervention_hook:
            self.config.intervention_hook = intervention_hook

        self._llm: Optional[LLM] = None
        self._session_manager: Optional["BaseSessionManager"] = None

    @property
    def provider(self) -> Provider:
        """Get the default provider."""
        return self.config.provider

    async def __aenter__(self) -> "UnifiedAdvanced":
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Async context manager exit."""
        await self.close()

    async def close(self) -> None:
        """Close the client and release resources."""
        if self._llm:
            await self._llm.__aexit__(None, None, None)
            self._llm = None

    async def _get_llm(
        self,
        config: Optional[LLMConfig] = None,
    ) -> LLM:
        """Get or create LLM instance.

        Args:
            config: Optional config override

        Returns:
            LLM instance
        """
        effective_config = config or self.config.to_llm_config()
        llm = LLM(
            config=effective_config,
            intervention_hook=self.config.intervention_hook,
        )
        await llm.__aenter__()
        return llm

    def get_session_manager(self, project_path: Optional[str] = None) -> "BaseSessionManager":
        """Get session manager for current provider.

        Args:
            project_path: Project path (defaults to cwd)

        Returns:
            Provider-specific session manager
        """
        import os

        path = project_path or self.config.cwd or os.getcwd()
        return get_session_manager(self.provider, path)

    # =========================================================================
    # Basic Execution
    # =========================================================================

    async def run(
        self,
        prompt: str,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Execute a prompt with the default provider.

        Args:
            prompt: Prompt to execute
            timeout: Timeout override
            **kwargs: Additional options

        Returns:
            LLMResult with execution results
        """
        llm = await self._get_llm()
        try:
            return await llm.run(prompt, timeout=timeout, **kwargs)
        finally:
            await llm.__aexit__(None, None, None)

    async def run_with_system_prompt(
        self,
        prompt: str,
        system_prompt: str,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Execute with a custom system prompt.

        Uses provider-appropriate injection method:
        - Claude: Native --system-prompt flag
        - Codex/Gemini: Prepend to prompt

        Args:
            prompt: User prompt
            system_prompt: System context to inject
            timeout: Timeout override
            **kwargs: Additional options

        Returns:
            LLMResult with execution results
        """
        effective_prompt, extra_config = inject_system_prompt(
            self.provider, prompt, system_prompt
        )

        config = self.config.to_llm_config()
        if extra_config.get("system_prompt"):
            config = LLMConfig(
                provider=config.provider,
                model=config.model,
                tier=config.tier,
                auto_approval=config.auto_approval,
                sandbox=config.sandbox,
                timeout=config.timeout,
                cwd=config.cwd,
                system_prompt=extra_config.get("system_prompt"),
                session_id=config.session_id,
                reasoning_level=config.reasoning_level,
                api_key=config.api_key,
                env_file=config.env_file,
                strict_env_security=config.strict_env_security,
                intervention_hook=config.intervention_hook,
                provider_options=config.provider_options,
            )

        llm = await self._get_llm(config)
        try:
            return await llm.run(effective_prompt, timeout=timeout, **kwargs)
        finally:
            await llm.__aexit__(None, None, None)

    async def run_with_template(
        self,
        prompt: str,
        template: "SessionTemplate",
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Execute with a predefined template.

        Args:
            prompt: User prompt
            template: SessionTemplate enum value
            timeout: Timeout override
            **kwargs: Additional options

        Returns:
            LLMResult with execution results
        """
        from u_llm_sdk.session import get_template_prompt

        system_prompt = get_template_prompt(template)
        return await self.run_with_system_prompt(
            prompt, system_prompt, timeout=timeout, **kwargs
        )

    # =========================================================================
    # Agent-Based Execution
    # =========================================================================

    async def run_with_agent(
        self,
        prompt: str,
        agent: AgentDefinition,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Execute with a specific agent.

        The agent may specify its own provider, which overrides the client's default.

        Args:
            prompt: Prompt to execute
            agent: Agent definition
            timeout: Timeout override
            **kwargs: Additional options

        Returns:
            LLMResult with execution results
        """
        agent_config = self.config.with_agent(agent)
        effective_timeout = timeout or self.config.get_agent_timeout()

        llm = await self._get_llm(agent_config)
        try:
            return await llm.run(prompt, timeout=effective_timeout, **kwargs)
        finally:
            await llm.__aexit__(None, None, None)

    async def run_with_agents(
        self,
        objective: str,
        agents: list[AgentDefinition],
        sequential: bool = True,
        pass_context: bool = True,
    ) -> list[LLMResult]:
        """Execute objective with multiple agents.

        Args:
            objective: The objective to achieve
            agents: List of agents to execute
            sequential: If True, execute sequentially; if False, in parallel
            pass_context: If True, pass previous results as context (sequential only)

        Returns:
            List of results from each agent
        """
        if sequential:
            return await self._run_agents_sequential(objective, agents, pass_context)
        else:
            return await self._run_agents_parallel(objective, agents)

    async def _run_agents_sequential(
        self,
        objective: str,
        agents: list[AgentDefinition],
        pass_context: bool,
    ) -> list[LLMResult]:
        """Run agents sequentially with optional context passing."""
        results: list[LLMResult] = []
        context = ""

        for agent in agents:
            if pass_context and context:
                prompt = f"Previous context:\n{context}\n\nObjective: {objective}"
            else:
                prompt = objective

            result = await self.run_with_agent(prompt, agent)
            results.append(result)

            if result.success:
                context = result.text

        return results

    async def _run_agents_parallel(
        self,
        objective: str,
        agents: list[AgentDefinition],
    ) -> list[LLMResult]:
        """Run agents in parallel with concurrency control."""
        semaphore = asyncio.Semaphore(self.config.max_parallel_agents)

        async def run_with_semaphore(agent: AgentDefinition) -> LLMResult:
            async with semaphore:
                return await self.run_with_agent(objective, agent)

        tasks = [run_with_semaphore(agent) for agent in agents]
        return await asyncio.gather(*tasks)

    async def parallel_agents(
        self,
        tasks: list[str],
        agent: AgentDefinition,
    ) -> list[LLMResult]:
        """Execute multiple tasks with the same agent in parallel.

        Args:
            tasks: List of task prompts
            agent: Agent to use for all tasks

        Returns:
            List of results for each task
        """
        semaphore = asyncio.Semaphore(self.config.max_parallel_agents)

        async def run_task(task: str) -> LLMResult:
            async with semaphore:
                return await self.run_with_agent(task, agent)

        task_futures = [run_task(task) for task in tasks]
        return await asyncio.gather(*task_futures)

    # =========================================================================
    # Streaming
    # =========================================================================

    async def stream(
        self,
        prompt: str,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream execution results.

        Args:
            prompt: Prompt to execute
            timeout: Timeout override
            **kwargs: Additional options

        Yields:
            Stream events from the provider
        """
        llm = await self._get_llm()
        try:
            async for event in llm.stream(prompt, timeout=timeout, **kwargs):
                yield event
        finally:
            await llm.__aexit__(None, None, None)

    async def stream_with_agent(
        self,
        prompt: str,
        agent: AgentDefinition,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream execution with a specific agent.

        Args:
            prompt: Prompt to execute
            agent: Agent definition
            timeout: Timeout override
            **kwargs: Additional options

        Yields:
            Stream events from the provider
        """
        agent_config = self.config.with_agent(agent)
        effective_timeout = timeout or self.config.get_agent_timeout()

        llm = await self._get_llm(agent_config)
        try:
            async for event in llm.stream(prompt, timeout=effective_timeout, **kwargs):
                yield event
        finally:
            await llm.__aexit__(None, None, None)


class UnifiedAdvancedSync:
    """Synchronous wrapper for UnifiedAdvanced.

    Provides the same functionality as UnifiedAdvanced but with
    synchronous methods for non-async contexts.

    Example:
        >>> with UnifiedAdvancedSync(provider=Provider.CLAUDE) as client:
        ...     result = client.run("Hello!")
    """

    def __init__(
        self,
        provider: Optional[Provider] = None,
        config: Optional[AdvancedConfig] = None,
        intervention_hook: Optional["InterventionHook"] = None,
    ):
        """Initialize synchronous client.

        Args:
            provider: Default provider
            config: Full configuration
            intervention_hook: Optional hook for MV-rag integration
        """
        if config:
            self.config = config
        else:
            self.config = AdvancedConfig(
                provider=provider or Provider.CLAUDE,
            )

        if intervention_hook:
            self.config.intervention_hook = intervention_hook

    @property
    def provider(self) -> Provider:
        """Get the default provider."""
        return self.config.provider

    def __enter__(self) -> "UnifiedAdvancedSync":
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Context manager exit."""
        pass

    def get_session_manager(self, project_path: Optional[str] = None) -> "BaseSessionManager":
        """Get session manager for current provider."""
        import os

        path = project_path or self.config.cwd or os.getcwd()
        return get_session_manager(self.provider, path)

    def run(
        self,
        prompt: str,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Execute a prompt synchronously."""
        config = self.config.to_llm_config()

        with LLMSync(config=config, intervention_hook=self.config.intervention_hook) as llm:
            return llm.run(prompt, timeout=timeout, **kwargs)

    def run_with_system_prompt(
        self,
        prompt: str,
        system_prompt: str,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Execute with a custom system prompt synchronously."""
        effective_prompt, extra_config = inject_system_prompt(
            self.provider, prompt, system_prompt
        )

        config = self.config.to_llm_config()
        if extra_config.get("system_prompt"):
            config = LLMConfig(
                provider=config.provider,
                model=config.model,
                tier=config.tier,
                auto_approval=config.auto_approval,
                sandbox=config.sandbox,
                timeout=config.timeout,
                cwd=config.cwd,
                system_prompt=extra_config.get("system_prompt"),
                session_id=config.session_id,
                reasoning_level=config.reasoning_level,
                api_key=config.api_key,
                env_file=config.env_file,
                strict_env_security=config.strict_env_security,
                intervention_hook=config.intervention_hook,
                provider_options=config.provider_options,
            )

        with LLMSync(config=config, intervention_hook=self.config.intervention_hook) as llm:
            return llm.run(effective_prompt, timeout=timeout, **kwargs)

    def run_with_template(
        self,
        prompt: str,
        template: "SessionTemplate",
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Execute with a predefined template synchronously."""
        from u_llm_sdk.session import get_template_prompt

        system_prompt = get_template_prompt(template)
        return self.run_with_system_prompt(
            prompt, system_prompt, timeout=timeout, **kwargs
        )

    def run_with_agent(
        self,
        prompt: str,
        agent: AgentDefinition,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Execute with a specific agent synchronously."""
        agent_config = self.config.with_agent(agent)
        effective_timeout = timeout or self.config.get_agent_timeout()

        with LLMSync(config=agent_config, intervention_hook=self.config.intervention_hook) as llm:
            return llm.run(prompt, timeout=effective_timeout, **kwargs)

    def quick_run(
        self,
        prompt: str,
        timeout: Optional[float] = None,
    ) -> LLMResult:
        """Quick one-shot execution without context manager.

        Args:
            prompt: Prompt to execute
            timeout: Timeout override

        Returns:
            LLMResult with execution results
        """
        return self.run(prompt, timeout=timeout)


__all__ = [
    "UnifiedAdvanced",
    "UnifiedAdvancedSync",
]
