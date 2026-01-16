"""Quick utility functions for LLM execution.

Provides convenient one-shot functions for common LLM tasks.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Optional

from u_llm_sdk.types import LLMResult, Provider

from u_llm_sdk.config import LLMConfig
from u_llm_sdk.llm.client import LLM, LLMSync

if TYPE_CHECKING:
    from u_llm_sdk.llm.providers.hooks import InterventionHook
    from u_llm_sdk.session.templates import SessionTemplate

logger = logging.getLogger(__name__)


# =============================================================================
# Async Functions
# =============================================================================


async def quick_run(
    prompt: str,
    *,
    provider: Provider = Provider.CLAUDE,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
    config: Optional[LLMConfig] = None,
    intervention_hook: Optional["InterventionHook"] = None,
) -> LLMResult:
    """Execute a single prompt quickly.

    Convenience function for one-shot LLM execution.

    Args:
        prompt: Prompt to execute
        provider: Provider to use (default: CLAUDE)
        model: Model to use (optional)
        timeout: Timeout in seconds
        config: Full config (overrides provider/model)
        intervention_hook: Optional hook for context injection

    Returns:
        LLMResult with execution results

    Example:
        >>> result = await quick_run("What is 2+2?")
        >>> print(result.text)
    """
    effective_config = config or LLMConfig(provider=provider, model=model)

    async with LLM(config=effective_config, intervention_hook=intervention_hook) as llm:
        return await llm.run(prompt, timeout=timeout)


async def quick_text(
    prompt: str,
    *,
    provider: Provider = Provider.CLAUDE,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
    config: Optional[LLMConfig] = None,
) -> str:
    """Execute a prompt and return just the text response.

    Args:
        prompt: Prompt to execute
        provider: Provider to use (default: CLAUDE)
        model: Model to use (optional)
        timeout: Timeout in seconds
        config: Full config (overrides provider/model)

    Returns:
        Text response from the LLM

    Example:
        >>> answer = await quick_text("What is 2+2?")
        >>> print(answer)  # "4"
    """
    result = await quick_run(
        prompt, provider=provider, model=model, timeout=timeout, config=config
    )
    return result.text


async def auto_run(
    prompt: str,
    *,
    timeout: Optional[float] = None,
    config: Optional[LLMConfig] = None,
    intervention_hook: Optional["InterventionHook"] = None,
) -> LLMResult:
    """Execute a prompt with auto-selected provider.

    Automatically selects the first available provider.
    Priority: Claude > Codex > Gemini

    Args:
        prompt: Prompt to execute
        timeout: Timeout in seconds
        config: Base config (provider will be auto-selected)
        intervention_hook: Optional hook for context injection

    Returns:
        LLMResult with execution results

    Example:
        >>> result = await auto_run("Hello!")
        >>> print(result.provider)  # First available provider
    """
    async with LLM.auto(config=config, intervention_hook=intervention_hook) as llm:
        return await llm.run(prompt, timeout=timeout)


async def parallel_run(
    prompts: list[str],
    *,
    provider: Provider = Provider.CLAUDE,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
    config: Optional[LLMConfig] = None,
    intervention_hook: Optional["InterventionHook"] = None,
) -> list[LLMResult]:
    """Execute multiple prompts in parallel.

    Args:
        prompts: List of prompts to execute
        provider: Provider to use (default: CLAUDE)
        model: Model to use (optional)
        timeout: Timeout per execution
        config: Full config (overrides provider/model)
        intervention_hook: Optional hook for context injection

    Returns:
        List of LLMResults in same order as prompts

    Example:
        >>> results = await parallel_run(["Q1?", "Q2?", "Q3?"])
        >>> for r in results:
        ...     print(r.text)
    """
    effective_config = config or LLMConfig(provider=provider, model=model)

    async with LLM(config=effective_config, intervention_hook=intervention_hook) as llm:
        return await llm.parallel_run(prompts, timeout=timeout)


async def multi_provider_run(
    prompt: str,
    *,
    providers: Optional[list[Provider]] = None,
    timeout: Optional[float] = None,
    return_first: bool = False,
) -> dict[Provider, LLMResult] | LLMResult:
    """Execute prompt across multiple providers.

    Args:
        prompt: Prompt to execute
        providers: List of providers to use (default: all available)
        timeout: Timeout per execution
        return_first: If True, return first successful result only

    Returns:
        Dict mapping provider to result, or single result if return_first=True

    Example:
        >>> results = await multi_provider_run("Hello!")
        >>> for provider, result in results.items():
        ...     print(f"{provider}: {result.text[:50]}")
    """
    from u_llm_sdk.core.discovery import available_providers

    target_providers = providers or available_providers()

    if not target_providers:
        from u_llm_sdk.types import ProviderNotAvailableError

        raise ProviderNotAvailableError("No providers available")

    async def run_for_provider(p: Provider) -> tuple[Provider, LLMResult]:
        config = LLMConfig(provider=p)
        async with LLM(config=config) as llm:
            result = await llm.run(prompt, timeout=timeout)
            return (p, result)

    tasks = [asyncio.create_task(run_for_provider(p)) for p in target_providers]

    if return_first:
        # Return first successful result
        for coro in asyncio.as_completed(tasks):
            provider, result = await coro
            if result.success:
                # Cancel remaining tasks
                for t in tasks:
                    if not t.done():
                        t.cancel()
                return result
        # All failed, return last result
        return result
    else:
        # Return all results
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        return {
            p: r if isinstance(r, LLMResult) else LLMResult(
                success=False,
                result_type="error",
                provider=p.value,
                model="",
                text="",
                summary="Execution failed",
                error=str(r),
            )
            for p, r in results_list
            if not isinstance(r, Exception)
        }


async def structured_run(
    query: str,
    *,
    context: Optional[str] = None,
    verification: Optional[str] = None,
    provider: Provider = Provider.CLAUDE,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
    config: Optional[LLMConfig] = None,
) -> LLMResult:
    """Execute a structured prompt with context and verification.

    Combines context, query, and verification into a structured prompt.

    Args:
        query: Main query to answer
        context: Optional context/background information
        verification: Optional verification question for self-check
        provider: Provider to use (default: CLAUDE)
        model: Model to use (optional)
        timeout: Timeout in seconds
        config: Full config (overrides provider/model)

    Returns:
        LLMResult with execution results

    Example:
        >>> result = await structured_run(
        ...     query="What is the capital of France?",
        ...     context="We are discussing European geography.",
        ...     verification="Verify this is a city name."
        ... )
    """
    # Build structured prompt
    parts = []

    if context:
        parts.append(f"<context>\n{context}\n</context>")

    parts.append(f"<query>\n{query}\n</query>")

    if verification:
        parts.append(
            f"<verification>\nAfter answering, verify: {verification}\n</verification>"
        )

    structured_prompt = "\n\n".join(parts)

    return await quick_run(
        structured_prompt,
        provider=provider,
        model=model,
        timeout=timeout,
        config=config,
    )


# =============================================================================
# Sync Functions
# =============================================================================


def quick_run_sync(
    prompt: str,
    *,
    provider: Provider = Provider.CLAUDE,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
    config: Optional[LLMConfig] = None,
    intervention_hook: Optional["InterventionHook"] = None,
) -> LLMResult:
    """Synchronous version of quick_run.

    Args:
        prompt: Prompt to execute
        provider: Provider to use (default: CLAUDE)
        model: Model to use (optional)
        timeout: Timeout in seconds
        config: Full config (overrides provider/model)
        intervention_hook: Optional hook for context injection

    Returns:
        LLMResult with execution results
    """
    effective_config = config or LLMConfig(provider=provider, model=model)

    with LLMSync(config=effective_config, intervention_hook=intervention_hook) as llm:
        return llm.run(prompt, timeout=timeout)


def quick_text_sync(
    prompt: str,
    *,
    provider: Provider = Provider.CLAUDE,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
    config: Optional[LLMConfig] = None,
) -> str:
    """Synchronous version of quick_text.

    Args:
        prompt: Prompt to execute
        provider: Provider to use (default: CLAUDE)
        model: Model to use (optional)
        timeout: Timeout in seconds
        config: Full config (overrides provider/model)

    Returns:
        Text response from the LLM
    """
    result = quick_run_sync(
        prompt, provider=provider, model=model, timeout=timeout, config=config
    )
    return result.text


def auto_run_sync(
    prompt: str,
    *,
    timeout: Optional[float] = None,
    config: Optional[LLMConfig] = None,
    intervention_hook: Optional["InterventionHook"] = None,
) -> LLMResult:
    """Synchronous version of auto_run.

    Args:
        prompt: Prompt to execute
        timeout: Timeout in seconds
        config: Base config (provider will be auto-selected)
        intervention_hook: Optional hook for context injection

    Returns:
        LLMResult with execution results
    """
    with LLMSync.auto(config=config, intervention_hook=intervention_hook) as llm:
        return llm.run(prompt, timeout=timeout)


def parallel_run_sync(
    prompts: list[str],
    *,
    provider: Provider = Provider.CLAUDE,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
    config: Optional[LLMConfig] = None,
    intervention_hook: Optional["InterventionHook"] = None,
) -> list[LLMResult]:
    """Synchronous version of parallel_run.

    Args:
        prompts: List of prompts to execute
        provider: Provider to use (default: CLAUDE)
        model: Model to use (optional)
        timeout: Timeout per execution
        config: Full config (overrides provider/model)
        intervention_hook: Optional hook for context injection

    Returns:
        List of LLMResults in same order as prompts
    """
    effective_config = config or LLMConfig(provider=provider, model=model)

    with LLMSync(config=effective_config, intervention_hook=intervention_hook) as llm:
        return llm.parallel_run(prompts, timeout=timeout)


def structured_run_sync(
    query: str,
    *,
    context: Optional[str] = None,
    verification: Optional[str] = None,
    provider: Provider = Provider.CLAUDE,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
    config: Optional[LLMConfig] = None,
) -> LLMResult:
    """Synchronous version of structured_run.

    Args:
        query: Main query to answer
        context: Optional context/background information
        verification: Optional verification question for self-check
        provider: Provider to use (default: CLAUDE)
        model: Model to use (optional)
        timeout: Timeout in seconds
        config: Full config (overrides provider/model)

    Returns:
        LLMResult with execution results
    """
    # Build structured prompt
    parts = []

    if context:
        parts.append(f"<context>\n{context}\n</context>")

    parts.append(f"<query>\n{query}\n</query>")

    if verification:
        parts.append(
            f"<verification>\nAfter answering, verify: {verification}\n</verification>"
        )

    structured_prompt = "\n\n".join(parts)

    return quick_run_sync(
        structured_prompt,
        provider=provider,
        model=model,
        timeout=timeout,
        config=config,
    )


# =============================================================================
# Template-aware Functions
# =============================================================================


async def template_run(
    prompt: str,
    template: "SessionTemplate",
    *,
    provider: Provider = Provider.CLAUDE,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
    config: Optional[LLMConfig] = None,
    intervention_hook: Optional["InterventionHook"] = None,
) -> LLMResult:
    """Execute a prompt with a predefined template.

    Uses the session template system to inject specialized system prompts.

    Args:
        prompt: User prompt to execute
        template: SessionTemplate enum value
        provider: Provider to use (default: CLAUDE)
        model: Model to use (optional)
        timeout: Timeout in seconds
        config: Full config (overrides provider/model)
        intervention_hook: Optional hook for context injection

    Returns:
        LLMResult with execution results

    Example:
        >>> from u_llm_sdk.session import SessionTemplate
        >>> result = await template_run(
        ...     "Review auth.py for security issues",
        ...     SessionTemplate.SECURITY_ANALYST,
        ...     provider=Provider.CLAUDE,
        ... )
    """
    from u_llm_sdk.session import get_template_prompt, inject_system_prompt

    system_prompt = get_template_prompt(template)

    # Inject system prompt using provider-appropriate method
    effective_prompt, extra_config = inject_system_prompt(
        provider, prompt, system_prompt
    )

    # Build config
    if config:
        effective_config = LLMConfig(
            provider=config.provider,
            model=model or config.model,
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
            provider_options={**config.provider_options, **extra_config},
        )
    else:
        effective_config = LLMConfig(
            provider=provider,
            model=model,
            system_prompt=extra_config.get("system_prompt"),
            provider_options=extra_config,
        )

    async with LLM(config=effective_config, intervention_hook=intervention_hook) as llm:
        return await llm.run(effective_prompt, timeout=timeout)


def template_run_sync(
    prompt: str,
    template: "SessionTemplate",
    *,
    provider: Provider = Provider.CLAUDE,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
    config: Optional[LLMConfig] = None,
    intervention_hook: Optional["InterventionHook"] = None,
) -> LLMResult:
    """Synchronous version of template_run.

    Args:
        prompt: User prompt to execute
        template: SessionTemplate enum value
        provider: Provider to use (default: CLAUDE)
        model: Model to use (optional)
        timeout: Timeout in seconds
        config: Full config (overrides provider/model)
        intervention_hook: Optional hook for context injection

    Returns:
        LLMResult with execution results

    Example:
        >>> from u_llm_sdk.session import SessionTemplate
        >>> result = template_run_sync(
        ...     "Review auth.py",
        ...     SessionTemplate.CODE_REVIEWER,
        ... )
    """
    from u_llm_sdk.session import get_template_prompt, inject_system_prompt

    system_prompt = get_template_prompt(template)

    # Inject system prompt using provider-appropriate method
    effective_prompt, extra_config = inject_system_prompt(
        provider, prompt, system_prompt
    )

    # Build config
    if config:
        effective_config = LLMConfig(
            provider=config.provider,
            model=model or config.model,
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
            provider_options={**config.provider_options, **extra_config},
        )
    else:
        effective_config = LLMConfig(
            provider=provider,
            model=model,
            system_prompt=extra_config.get("system_prompt"),
            provider_options=extra_config,
        )

    with LLMSync(config=effective_config, intervention_hook=intervention_hook) as llm:
        return llm.run(effective_prompt, timeout=timeout)


__all__ = [
    # Async functions
    "quick_run",
    "quick_text",
    "auto_run",
    "parallel_run",
    "multi_provider_run",
    "structured_run",
    "template_run",
    # Sync functions
    "quick_run_sync",
    "quick_text_sync",
    "auto_run_sync",
    "parallel_run_sync",
    "structured_run_sync",
    "template_run_sync",
]
