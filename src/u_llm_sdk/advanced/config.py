"""Configuration for Advanced SDK.

This module provides provider-agnostic configuration for advanced
LLM orchestration features like agent definitions and parallel execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from u_llm_sdk.types import (
    AutoApproval,
    ModelTier,
    Provider,
    ReasoningLevel,
    SandboxMode,
)

if TYPE_CHECKING:
    from u_llm_sdk.config import LLMConfig
    from u_llm_sdk.llm.providers import InterventionHook


@dataclass
class AgentDefinition:
    """Definition of a specialized agent.

    Agents are specialized configurations for specific roles
    (e.g., planner, executor, reviewer) with custom prompts and constraints.

    Unlike ClaudeAdvancedConfig, agents can optionally override the
    provider, enabling workflows where different agents use different
    LLM providers.

    Attributes:
        name: Agent identifier (e.g., "planner", "executor")
        description: Human-readable description of agent's role
        system_prompt: System prompt defining agent behavior
        provider: Optional provider override (None uses client's provider)
        model: Model to use (None for config default)
        tier: Model tier for automatic selection
        reasoning_level: Reasoning intensity for this agent
        allowed_tools: List of allowed tool names
        disallowed_tools: List of disallowed tool names
        max_turns: Maximum conversation turns for this agent
        temperature: Temperature setting (if provider supports it)
        additional_options: Extra provider-specific options

    Example:
        >>> planner = AgentDefinition(
        ...     name="planner",
        ...     description="Breaks down tasks into subtasks",
        ...     system_prompt="You are a planning specialist...",
        ...     tier=ModelTier.HIGH,
        ...     reasoning_level=ReasoningLevel.XHIGH,
        ...     allowed_tools=["Read", "Grep", "Glob"],
        ... )
        >>>
        >>> # Agent with different provider
        >>> codex_analyzer = AgentDefinition(
        ...     name="analyzer",
        ...     description="Analyzes code patterns",
        ...     system_prompt="You are a code analyst...",
        ...     provider=Provider.CODEX,  # Override client's provider
        ... )
    """

    name: str
    description: str
    system_prompt: str
    provider: Optional[Provider] = None  # Override client's provider
    model: Optional[str] = None
    tier: Optional[ModelTier] = None
    reasoning_level: Optional[ReasoningLevel] = None
    allowed_tools: Optional[list[str]] = None
    disallowed_tools: Optional[list[str]] = None
    max_turns: Optional[int] = None
    temperature: Optional[float] = None
    additional_options: dict[str, Any] = field(default_factory=dict)

    def to_provider_options(self) -> dict[str, Any]:
        """Convert agent definition to provider_options dict.

        Returns:
            Dict suitable for LLMConfig.provider_options
        """
        options: dict[str, Any] = {}

        if self.allowed_tools is not None:
            options["allowed_tools"] = self.allowed_tools
        if self.disallowed_tools is not None:
            options["disallowed_tools"] = self.disallowed_tools
        if self.max_turns is not None:
            options["max_turns"] = self.max_turns
        if self.temperature is not None:
            options["temperature"] = self.temperature

        # Merge additional options
        options.update(self.additional_options)

        return options

    def to_llm_config(self, base_config: Optional["LLMConfig"] = None) -> "LLMConfig":
        """Convert agent definition to LLMConfig.

        Args:
            base_config: Base configuration to extend (uses defaults if None)

        Returns:
            LLMConfig with agent-specific settings
        """
        from u_llm_sdk.config import LLMConfig

        if base_config is None:
            base_config = LLMConfig(provider=self.provider or Provider.CLAUDE)

        # Determine provider - agent's provider takes precedence
        effective_provider = self.provider or base_config.provider

        return LLMConfig(
            provider=effective_provider,
            model=self.model or base_config.model,
            tier=self.tier or base_config.tier,
            auto_approval=base_config.auto_approval,
            sandbox=base_config.sandbox,
            timeout=base_config.timeout,
            cwd=base_config.cwd,
            system_prompt=self.system_prompt,
            session_id=base_config.session_id,
            reasoning_level=self.reasoning_level or base_config.reasoning_level,
            api_key=base_config.api_key,
            env_file=base_config.env_file,
            strict_env_security=base_config.strict_env_security,
            intervention_hook=base_config.intervention_hook,
            provider_options={
                **base_config.provider_options,
                **self.to_provider_options(),
            },
        )


@dataclass
class AdvancedConfig:
    """Configuration for advanced SDK features.

    Provider-aware configuration for sophisticated LLM orchestration
    including agent definitions, parallel execution, and task workflows.

    Attributes:
        provider: Default provider for client
        model: Model name (None for tier-based selection)
        tier: Model tier for automatic selection
        auto_approval: Approval mode for actions
        sandbox: Sandbox mode
        timeout: Timeout in seconds
        cwd: Working directory
        reasoning_level: Default reasoning level for agents
        api_key: API key (optional, usually from env)
        env_file: Path to .env file
        strict_env_security: Block on permissive .env permissions
        max_parallel_agents: Maximum concurrent agent executions
        agent_timeout_multiplier: Timeout multiplier for agents
        enable_task_tools: Enable Task tool execution helpers
        provider_options: Additional provider-specific options
        intervention_hook: Optional hook for MV-rag integration

    Example:
        >>> config = AdvancedConfig(
        ...     provider=Provider.CLAUDE,
        ...     tier=ModelTier.HIGH,
        ...     auto_approval=AutoApproval.EDITS_ONLY,
        ...     max_parallel_agents=3,
        ... )
    """

    # Core LLM settings
    provider: Provider = Provider.CLAUDE
    model: Optional[str] = None
    tier: Optional[ModelTier] = None
    auto_approval: AutoApproval = AutoApproval.EDITS_ONLY
    sandbox: SandboxMode = SandboxMode.NONE
    timeout: float = 1200.0  # 20 minutes
    cwd: Optional[str] = None
    reasoning_level: Optional[ReasoningLevel] = None
    api_key: Optional[str] = None
    env_file: Optional[str] = None
    strict_env_security: bool = False

    # Advanced features
    max_parallel_agents: int = 3
    agent_timeout_multiplier: float = 1.5
    enable_task_tools: bool = True

    # Provider-specific options
    provider_options: dict[str, Any] = field(default_factory=dict)

    # Hook support (optional, for MV-rag integration)
    intervention_hook: Optional["InterventionHook"] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.max_parallel_agents < 1:
            raise ValueError(
                f"max_parallel_agents must be >= 1, got {self.max_parallel_agents}"
            )

        if self.agent_timeout_multiplier < 1.0:
            raise ValueError(
                f"agent_timeout_multiplier must be >= 1.0, got {self.agent_timeout_multiplier}"
            )

    def to_llm_config(self) -> "LLMConfig":
        """Convert to base LLMConfig.

        Returns:
            LLMConfig with advanced settings
        """
        from u_llm_sdk.config import LLMConfig

        return LLMConfig(
            provider=self.provider,
            model=self.model,
            tier=self.tier,
            auto_approval=self.auto_approval,
            sandbox=self.sandbox,
            timeout=self.timeout,
            cwd=self.cwd,
            system_prompt=None,  # Set per-agent
            session_id=None,
            reasoning_level=self.reasoning_level,
            api_key=self.api_key,
            env_file=self.env_file,
            strict_env_security=self.strict_env_security,
            intervention_hook=self.intervention_hook,
            provider_options=self.provider_options,
        )

    def with_agent(self, agent: AgentDefinition) -> "LLMConfig":
        """Create LLMConfig for a specific agent.

        Args:
            agent: Agent definition to configure

        Returns:
            LLMConfig with agent-specific settings
        """
        base_config = self.to_llm_config()
        return agent.to_llm_config(base_config)

    def get_agent_timeout(self) -> float:
        """Get timeout for agent execution.

        Returns:
            Timeout in seconds for agent tasks
        """
        return self.timeout * self.agent_timeout_multiplier


__all__ = [
    "AgentDefinition",
    "AdvancedConfig",
]
