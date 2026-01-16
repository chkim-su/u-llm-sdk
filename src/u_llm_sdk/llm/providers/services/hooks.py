"""Hook Manager Service.

Extracts intervention hook orchestration from BaseProvider (SRP compliance).
Handles pre-action and post-action hook invocations with fail-open semantics.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from u_llm_sdk.types import LLMResult, PreActionContext

if TYPE_CHECKING:
    from u_llm_sdk.llm.providers.hooks import InterventionHook

logger = logging.getLogger(__name__)


class HookManager:
    """Manages intervention hook lifecycle.

    Design principles:
    - Fail-open: Hook failures never block LLM execution
    - Pre-action: Returns context injection (prompt augmentation)
    - Post-action: Fire-and-forget feedback recording

    The hook protocol allows external systems (like RAG) to:
    1. Inject context before LLM calls (pre-action)
    2. Record feedback after LLM calls (post-action)
    """

    def __init__(
        self,
        hook: Optional[InterventionHook],
        provider_name: str,
    ) -> None:
        """Initialize hook manager.

        Args:
            hook: Optional intervention hook (None for no-op behavior)
            provider_name: Provider name for logging context
        """
        self._hook = hook
        self._provider_name = provider_name

    @property
    def has_hook(self) -> bool:
        """Check if a hook is configured."""
        return self._hook is not None

    async def call_pre_action(
        self,
        prompt: str,
        model: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Optional[PreActionContext]:
        """Call pre-action hook if configured.

        Args:
            prompt: Original prompt
            model: Model name (if available)
            session_id: Session ID (if available)

        Returns:
            PreActionContext if hook returned injection, None otherwise
        """
        if not self._hook:
            return None

        try:
            return await self._hook.on_pre_action(
                prompt=prompt,
                provider=self._provider_name,
                model=model,
                session_id=session_id,
            )
        except Exception:
            # Fail silently - don't block the request
            logger.exception("Intervention hook on_pre_action failed")
            return None

    async def call_post_action(
        self,
        result: LLMResult,
        pre_action_context: Optional[PreActionContext],
    ) -> None:
        """Call post-action hook if configured.

        Args:
            result: LLMResult from provider execution
            pre_action_context: PreActionContext if injection occurred
        """
        if not self._hook:
            return

        try:
            await self._hook.on_post_action(
                result=result,
                pre_action_context=pre_action_context,
            )
        except Exception:
            # Fail silently - don't affect the result
            logger.exception("Intervention hook on_post_action failed")


__all__ = ["HookManager"]
