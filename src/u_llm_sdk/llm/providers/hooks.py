"""Intervention Hooks for Provider Integration.

This module defines the protocol for PGSDK intervention hooks.
The data classes (PreActionContext, PostActionFeedback) live in llm-types,
while this protocol lives here with the consumer (U-llm-sdk).

Design Philosophy:
    - Hooks are OPTIONAL: Providers work without hooks (backward compatible)
    - Pre-action: Injects context BEFORE the prompt is sent to LLM
    - Post-action: Collects feedback AFTER the result is received
    - Minimal latency: Hooks should complete within 100ms (P95 SLO)

Architecture:
    - Data classes → llm-types (shared between U-llm-sdk and MV-rag)
    - Protocol → U-llm-sdk (this module, consumer-side interface)
    - Implementation → MV-rag (TimeCorrector via API endpoints)
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Optional, Protocol

if TYPE_CHECKING:
    from u_llm_sdk.types import LLMResult, PreActionContext


class InterventionHook(Protocol):
    """Protocol for PGSDK intervention hooks.

    Providers call these hooks before and after LLM actions to enable
    context injection and feedback collection. The hooks are OPTIONAL -
    providers work normally without them.

    The TimeCorrector implements this protocol to bridge providers with
    the PGSDK integration layer (DejaVu, Influence, Context Injection).

    Example Usage:
        ```python
        # In provider __init__
        def __init__(self, config, intervention_hook=None):
            self._hook = intervention_hook

        # In provider run()
        async def run(self, prompt, ...):
            pre_ctx = None
            if self._hook:
                pre_ctx = await self._hook.on_pre_action(prompt, ...)
                if pre_ctx:
                    prompt = self._apply_injection(prompt, pre_ctx)

            result = await self._execute(prompt, ...)

            if self._hook:
                await self._hook.on_post_action(result, pre_ctx)

            return result
        ```

    SLO Requirements:
        - on_pre_action: P95 latency < 100ms
        - on_post_action: P95 latency < 50ms (async, non-blocking)
    """

    @abstractmethod
    async def on_pre_action(
        self,
        prompt: str,
        provider: str,
        model: Optional[str] = None,
        session_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Optional[PreActionContext]:
        """Called before LLM action to potentially inject context.

        This hook is called BEFORE the prompt is sent to the LLM. It uses
        PGSDK components (DejaVu, Influence, Context Injection) to determine
        if context should be injected based on similar past failures.

        Args:
            prompt: The user's prompt (before any injection)
            provider: Provider name ("claude", "gemini", "codex")
            model: Model name (e.g., "claude-opus-4", "gemini-2.0-flash")
            session_id: Session ID for conversation continuity
            run_id: Current run ID for forensics tracking

        Returns:
            PreActionContext if context should be injected, None otherwise.
            The provider is responsible for applying the injection to the prompt.

        Note:
            Returning None means "proceed without injection" - this is the
            normal case when no relevant patterns are detected.
        """
        ...

    @abstractmethod
    async def on_post_action(
        self,
        result: "LLMResult",
        pre_action_context: Optional[PreActionContext],
        run_id: Optional[str] = None,
    ) -> None:
        """Called after LLM action to record feedback.

        This hook is called AFTER the LLM result is received. It records
        feedback to update DejaVu groups and influence scores based on
        whether the action succeeded or failed.

        Args:
            result: The LLMResult from the provider
            pre_action_context: The PreActionContext if injection occurred
            run_id: Current run ID for forensics tracking

        Note:
            This method should be non-blocking. Any heavy processing
            should be done asynchronously to avoid blocking the response.
        """
        ...


class NoOpHook:
    """No-operation hook for testing or disabled PGSDK.

    This hook does nothing - useful for testing providers without PGSDK
    or when PGSDK is explicitly disabled.
    """

    async def on_pre_action(
        self,
        prompt: str,
        provider: str,
        model: Optional[str] = None,
        session_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Optional[PreActionContext]:
        """Always returns None (no injection)."""
        return None

    async def on_post_action(
        self,
        result: "LLMResult",
        pre_action_context: Optional[PreActionContext],
        run_id: Optional[str] = None,
    ) -> None:
        """Does nothing."""
        pass
