"""Role-Based Agent Executor.

Executes LLM prompts with role-specific constraints and configurations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

from u_llm_sdk.types import AutoApproval, LLMResult, ModelTier, Provider

from .contracts import ROLES, SideEffect

if TYPE_CHECKING:
    pass


class AgentExecutor:
    """Role-based agent executor.

    Executes prompts with role-specific constraints and configurations.
    Each role (planner, editor, supervisor, etc.) has different:
    - Provider (Claude, Gemini, etc.)
    - Model tier (HIGH, LOW)
    - Side effects (READ, WRITE, NONE)

    Example:
        >>> executor = AgentExecutor()
        >>> result = await executor.execute(
        ...     role="planner",
        ...     prompt="Create a plan for implementing user authentication",
        ...     cwd="/path/to/repo",
        ... )
    """

    def __init__(
        self,
        provider_map: Optional[Dict[str, Provider]] = None,
        tier_map: Optional[Dict[str, ModelTier]] = None,
    ):
        """Initialize agent executor.

        Args:
            provider_map: Role -> Provider mapping
            tier_map: Role -> ModelTier mapping
        """
        self.provider_map = provider_map or {
            "planner": Provider.CLAUDE,
            "editor": Provider.GEMINI,
            "supervisor": Provider.CLAUDE,
            "aggregator": Provider.CLAUDE,
            "code_searcher": Provider.GEMINI,
            "web_searcher": Provider.GEMINI,
        }
        self.tier_map = tier_map or {
            "planner": ModelTier.HIGH,
            "editor": ModelTier.LOW,
            "supervisor": ModelTier.HIGH,
            "aggregator": ModelTier.LOW,
        }

    async def execute(
        self,
        role: str,
        prompt: str,
        context: Optional[str] = None,
        cwd: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> LLMResult:
        """Execute prompt with role constraints.

        Args:
            role: Role name (must exist in ROLES)
            prompt: Prompt to execute
            context: Optional context to prepend
            cwd: Working directory
            timeout: Execution timeout

        Returns:
            LLMResult from execution

        Raises:
            ValueError: If role is not found in ROLES
        """
        from u_llm_sdk.core.utils import quick_run

        spec = ROLES.get(role)
        if not spec:
            raise ValueError(f"Unknown role: {role}")

        # Build system prompt from role spec
        system_prompt = spec.to_system_prompt()

        # Determine auto_approval based on side effects
        if spec.side_effects == SideEffect.WRITE:
            auto_approval = AutoApproval.FULL
        elif spec.side_effects == SideEffect.READ:
            auto_approval = AutoApproval.EDITS_ONLY
        else:
            auto_approval = AutoApproval.NONE

        # Combine context and prompt
        full_prompt = f"{context}\n\n{prompt}" if context else prompt

        return await quick_run(
            full_prompt,
            provider=self.provider_map.get(role, Provider.CLAUDE),
            tier=self.tier_map.get(role, ModelTier.LOW),
            system_prompt=system_prompt,
            cwd=cwd,
            auto_approval=auto_approval,
            timeout=timeout,
        )


__all__ = [
    "AgentExecutor",
]
