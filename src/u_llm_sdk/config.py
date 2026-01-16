"""U-llm-sdk Configuration.

This module defines the LLMConfig class and preset configurations.
All enums and mappings are imported from llm-types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from u_llm_sdk.types import (
    APPROVAL_MAP,
    CURRENT_MODELS,
    MODEL_TIERS,
    REASONING_MAP,
    TEST_DEV_MODELS,
    AutoApproval,
    ModelTier,
    Provider,
    ReasoningLevel,
    SandboxMode,
    resolve_model,
    # Feature validation
    Feature,
    FeatureValidationResult,
    validate_config_features,
    # Domain schemas
    DomainSchema,
)

if TYPE_CHECKING:
    from u_llm_sdk.llm.providers.hooks import InterventionHook


@dataclass
class LLMConfig:
    """Unified configuration for all LLM providers.

    This configuration works identically across all providers.
    Provider-specific options can be passed via provider_options.

    Attributes:
        provider: LLM provider to use
        model: Model name (None for provider default)
        tier: Model tier for automatic model selection
        auto_approval: Approval mode for actions
        sandbox: Sandbox mode (Codex/Gemini only)
        timeout: Timeout in seconds
        cwd: Working directory
        system_prompt: System prompt (provider support varies)
        append_system_prompt: Additional system prompt to append
        session_id: Session ID to resume
        continue_session: Whether to continue an existing session
        reasoning_level: Reasoning/thinking intensity
        api_key: API key (optional, usually from env)
        env_file: Path to .env file for credentials
        strict_env_security: Block on permissive .env permissions
        allowed_tools: List of allowed tools (None = all allowed)
        disallowed_tools: List of disallowed tools (None = none disallowed)
        web_search: Enable web search tools (provider-specific):
            - Claude: WebSearch, WebFetch tools
            - Gemini: GoogleSearch (built-in, enabled by default when available)
            - Codex: NOT SUPPORTED in exec mode
        intervention_hook: Hook for pre/post action callbacks
        provider_options: Provider-specific options dict
        domain_schema: Domain-specific output schema (e.g., BrainstormSchema)
            - Defines expected output structure for validation
            - Independent of MV-RAG logging (Global Layer)
            - Fail-Open: validation failures log warnings, don't crash

    Provider-specific options:
        Claude:
            - allowed_tools: List of allowed tools
            - disallowed_tools: List of disallowed tools
            - mcp_config: Path to MCP config file
            - setting_sources: List like ["user", "project"]
            - max_turns: Maximum conversation turns

        Codex:
            - skip_git_check: Allow running outside git repo
            - add_dirs: Additional writable directories
            - search: Enable web search
            - images: List of image paths to attach

        Gemini:
            - temperature: Temperature (0.0-1.0)
            - top_p: Top-p sampling
            - top_k: Top-k sampling
            - allowed_tools: List of allowed tools (space-separated)
            - extensions: List of extensions (-e flag, multiple allowed)
            - include_directories: List of additional directories
            - allowed_mcp_server_names: List of allowed MCP server names
            - sandbox: Boolean flag to enable sandbox mode

    Example:
        >>> config = LLMConfig(
        ...     provider=Provider.CLAUDE,
        ...     tier=ModelTier.HIGH,
        ...     auto_approval=AutoApproval.EDITS_ONLY,
        ...     timeout=1200.0,
        ... )

        >>> config = LLMConfig(
        ...     provider=Provider.CODEX,
        ...     auto_approval=AutoApproval.FULL,
        ...     provider_options={"skip_git_check": True},
        ... )

        >>> # With domain schema for structured output
        >>> from u_llm_sdk.types import BrainstormSchema
        >>> config = LLMConfig(
        ...     provider=Provider.CLAUDE,
        ...     domain_schema=BrainstormSchema(),
        ... )
    """
    provider: Provider = Provider.CLAUDE
    model: Optional[str] = None
    tier: Optional[ModelTier] = None
    auto_approval: AutoApproval = AutoApproval.EDITS_ONLY
    sandbox: SandboxMode = SandboxMode.NONE
    timeout: float = 1200.0  # 20 minutes default
    cwd: Optional[str] = None
    system_prompt: Optional[str] = None
    append_system_prompt: Optional[str] = None
    session_id: Optional[str] = None
    continue_session: bool = False
    reasoning_level: Optional[ReasoningLevel] = None
    api_key: Optional[str] = None
    env_file: Optional[str] = None
    strict_env_security: bool = False
    allowed_tools: Optional[list[str]] = None
    disallowed_tools: Optional[list[str]] = None
    web_search: bool = False  # Enable web search tools (Claude: WebSearch/WebFetch, Gemini: GoogleSearch)
    intervention_hook: Optional["InterventionHook"] = None
    provider_options: dict[str, Any] = field(default_factory=dict)
    domain_schema: Optional[DomainSchema] = None

    def get_model(self, require_explicit: bool = True) -> str:
        """Get model name with automatic tier-based routing.

        Args:
            require_explicit: If True, raises error when no model/tier specified

        Returns:
            Resolved model name for the provider

        Raises:
            ModelNotSpecifiedError: If require_explicit=True and no model/tier given

        Priority:
            1. If tier is specified, return that tier's model
            2. If model is specified and is a current model, use it
            3. If model is a legacy model, route to appropriate tier
            4. If model is unknown, pass through
            5. If nothing specified and require_explicit=True, raise error
        """
        return resolve_model(self.provider, self.model, self.tier, require_explicit)

    def get_approval_args(self) -> dict[str, Any]:
        """Get provider-specific approval arguments."""
        return APPROVAL_MAP.get((self.provider, self.auto_approval), {})

    def get_reasoning_args(self) -> dict[str, Any]:
        """Get provider-specific reasoning arguments."""
        effective_reasoning = get_effective_reasoning(self)
        return REASONING_MAP.get((self.provider, effective_reasoning), {})

    def validate_for_provider(
        self,
        strict: bool = False,
        log_warnings: bool = True,
    ) -> list[FeatureValidationResult]:
        """Validate configuration against provider capabilities.

        Returns structured validation results for unsupported or caveated features.
        This is designed to be LLM-friendly - the results are JSON-serializable
        and contain actionable suggestions.

        Args:
            strict: If True, unsupported features return ERROR severity
            log_warnings: If True, log warnings for unsupported features

        Returns:
            List of FeatureValidationResult for any issues found.
            Empty list means all requested features are fully supported.

        Example:
            >>> config = LLMConfig(
            ...     provider=Provider.CODEX,
            ...     auto_approval=AutoApproval.EDITS_ONLY,  # Not supported!
            ... )
            >>> results = config.validate_for_provider()
            >>> for r in results:
            ...     print(r.to_dict())  # LLM can parse this
            {
                "feature": "auto_approval_edits_only",
                "provider": "codex",
                "severity": "warning",
                "supported": false,
                "message": "Feature 'auto_approval_edits_only' is NOT supported by codex CLI...",
                "suggestion": "Use one of these providers instead: claude, gemini | ...",
                "supported_providers": ["claude", "gemini"],
                ...
            }
        """
        import logging

        logger = logging.getLogger(__name__)

        # Get effective reasoning level
        effective_reasoning = get_effective_reasoning(self)

        results = validate_config_features(
            provider=self.provider.value,
            auto_approval=self.auto_approval.value if self.auto_approval else None,
            sandbox=self.sandbox.value if self.sandbox else None,
            reasoning_level=effective_reasoning.value if effective_reasoning else None,
            strict=strict,
        )

        if log_warnings and results:
            for result in results:
                log_msg = result.to_log_message()
                if result.severity.value == "error":
                    logger.error(log_msg)
                else:
                    logger.warning(log_msg)

        return results

    def with_provider(self, provider: Provider) -> LLMConfig:
        """Return a new config with different provider."""
        return LLMConfig(
            provider=provider,
            model=self.model,
            tier=self.tier,
            auto_approval=self.auto_approval,
            sandbox=self.sandbox,
            timeout=self.timeout,
            cwd=self.cwd,
            system_prompt=self.system_prompt,
            append_system_prompt=self.append_system_prompt,
            session_id=self.session_id,
            continue_session=self.continue_session,
            reasoning_level=self.reasoning_level,
            api_key=self.api_key,
            env_file=self.env_file,
            strict_env_security=self.strict_env_security,
            allowed_tools=self.allowed_tools,
            disallowed_tools=self.disallowed_tools,
            web_search=self.web_search,
            intervention_hook=self.intervention_hook,
            provider_options=self.provider_options.copy(),
            domain_schema=self.domain_schema,
        )

    def with_model(self, model: str) -> LLMConfig:
        """Return a new config with different model."""
        return LLMConfig(
            provider=self.provider,
            model=model,
            tier=self.tier,
            auto_approval=self.auto_approval,
            sandbox=self.sandbox,
            timeout=self.timeout,
            cwd=self.cwd,
            system_prompt=self.system_prompt,
            append_system_prompt=self.append_system_prompt,
            session_id=self.session_id,
            continue_session=self.continue_session,
            reasoning_level=self.reasoning_level,
            api_key=self.api_key,
            env_file=self.env_file,
            strict_env_security=self.strict_env_security,
            allowed_tools=self.allowed_tools,
            disallowed_tools=self.disallowed_tools,
            web_search=self.web_search,
            intervention_hook=self.intervention_hook,
            provider_options=self.provider_options.copy(),
            domain_schema=self.domain_schema,
        )

    def with_tier(self, tier: ModelTier) -> LLMConfig:
        """Return a new config with different tier."""
        return LLMConfig(
            provider=self.provider,
            model=None,  # Clear model when setting tier
            tier=tier,
            auto_approval=self.auto_approval,
            sandbox=self.sandbox,
            timeout=self.timeout,
            cwd=self.cwd,
            system_prompt=self.system_prompt,
            append_system_prompt=self.append_system_prompt,
            session_id=self.session_id,
            continue_session=self.continue_session,
            reasoning_level=self.reasoning_level,
            api_key=self.api_key,
            env_file=self.env_file,
            strict_env_security=self.strict_env_security,
            allowed_tools=self.allowed_tools,
            disallowed_tools=self.disallowed_tools,
            web_search=self.web_search,
            intervention_hook=self.intervention_hook,
            provider_options=self.provider_options.copy(),
            domain_schema=self.domain_schema,
        )

    def with_reasoning(self, level: ReasoningLevel) -> LLMConfig:
        """Return a new config with different reasoning level."""
        return LLMConfig(
            provider=self.provider,
            model=self.model,
            tier=self.tier,
            auto_approval=self.auto_approval,
            sandbox=self.sandbox,
            timeout=self.timeout,
            cwd=self.cwd,
            system_prompt=self.system_prompt,
            append_system_prompt=self.append_system_prompt,
            session_id=self.session_id,
            continue_session=self.continue_session,
            reasoning_level=level,
            api_key=self.api_key,
            env_file=self.env_file,
            strict_env_security=self.strict_env_security,
            allowed_tools=self.allowed_tools,
            disallowed_tools=self.disallowed_tools,
            web_search=self.web_search,
            intervention_hook=self.intervention_hook,
            provider_options=self.provider_options.copy(),
            domain_schema=self.domain_schema,
        )

    def with_hook(self, hook: Optional["InterventionHook"]) -> LLMConfig:
        """Return a new config with different intervention hook."""
        return LLMConfig(
            provider=self.provider,
            model=self.model,
            tier=self.tier,
            auto_approval=self.auto_approval,
            sandbox=self.sandbox,
            timeout=self.timeout,
            cwd=self.cwd,
            system_prompt=self.system_prompt,
            append_system_prompt=self.append_system_prompt,
            session_id=self.session_id,
            continue_session=self.continue_session,
            reasoning_level=self.reasoning_level,
            api_key=self.api_key,
            env_file=self.env_file,
            strict_env_security=self.strict_env_security,
            allowed_tools=self.allowed_tools,
            disallowed_tools=self.disallowed_tools,
            web_search=self.web_search,
            intervention_hook=hook,
            provider_options=self.provider_options.copy(),
            domain_schema=self.domain_schema,
        )

    def with_continue_session(self, continue_session: bool = True) -> LLMConfig:
        """Return a new config with session continuation enabled/disabled."""
        return LLMConfig(
            provider=self.provider,
            model=self.model,
            tier=self.tier,
            auto_approval=self.auto_approval,
            sandbox=self.sandbox,
            timeout=self.timeout,
            cwd=self.cwd,
            system_prompt=self.system_prompt,
            append_system_prompt=self.append_system_prompt,
            session_id=self.session_id,
            continue_session=continue_session,
            reasoning_level=self.reasoning_level,
            api_key=self.api_key,
            env_file=self.env_file,
            strict_env_security=self.strict_env_security,
            allowed_tools=self.allowed_tools,
            disallowed_tools=self.disallowed_tools,
            web_search=self.web_search,
            intervention_hook=self.intervention_hook,
            provider_options=self.provider_options.copy(),
            domain_schema=self.domain_schema,
        )

    def with_schema(self, schema: Optional[DomainSchema]) -> LLMConfig:
        """Return a new config with different domain schema.

        Args:
            schema: Domain schema for structured output validation
                    (e.g., BrainstormSchema, CodeReviewSchema)

        Returns:
            New LLMConfig with the specified domain schema

        Example:
            >>> from u_llm_sdk.types import BrainstormSchema, BrainstormOutputType
            >>> config = LLMConfig(provider=Provider.CLAUDE)
            >>> # Use preparation phase schema
            >>> prep_config = config.with_schema(
            ...     BrainstormSchema(output_type=BrainstormOutputType.PREPARATION)
            ... )
            >>> # Switch to discussion phase
            >>> disc_config = config.with_schema(
            ...     BrainstormSchema(output_type=BrainstormOutputType.DISCUSSION)
            ... )
        """
        return LLMConfig(
            provider=self.provider,
            model=self.model,
            tier=self.tier,
            auto_approval=self.auto_approval,
            sandbox=self.sandbox,
            timeout=self.timeout,
            cwd=self.cwd,
            system_prompt=self.system_prompt,
            append_system_prompt=self.append_system_prompt,
            session_id=self.session_id,
            continue_session=self.continue_session,
            reasoning_level=self.reasoning_level,
            api_key=self.api_key,
            env_file=self.env_file,
            strict_env_security=self.strict_env_security,
            allowed_tools=self.allowed_tools,
            disallowed_tools=self.disallowed_tools,
            web_search=self.web_search,
            intervention_hook=self.intervention_hook,
            provider_options=self.provider_options.copy(),
            domain_schema=schema,
        )

    def with_web_search(self, enabled: bool = True) -> LLMConfig:
        """Return a new config with web search enabled/disabled.

        Enables provider-specific web search tools:
        - Claude: WebSearch (query-based), WebFetch (URL-based)
        - Gemini: GoogleSearch (built-in)
        - Codex: NOT SUPPORTED (exec mode limitation)

        Args:
            enabled: Whether to enable web search tools

        Returns:
            New LLMConfig with web_search setting

        Example:
            >>> config = LLMConfig(provider=Provider.CLAUDE)
            >>> research_config = config.with_web_search()
            >>> # Now Claude will have access to WebSearch and WebFetch tools
        """
        return LLMConfig(
            provider=self.provider,
            model=self.model,
            tier=self.tier,
            auto_approval=self.auto_approval,
            sandbox=self.sandbox,
            timeout=self.timeout,
            cwd=self.cwd,
            system_prompt=self.system_prompt,
            append_system_prompt=self.append_system_prompt,
            session_id=self.session_id,
            continue_session=self.continue_session,
            reasoning_level=self.reasoning_level,
            api_key=self.api_key,
            env_file=self.env_file,
            strict_env_security=self.strict_env_security,
            allowed_tools=self.allowed_tools,
            disallowed_tools=self.disallowed_tools,
            web_search=enabled,
            intervention_hook=self.intervention_hook,
            provider_options=self.provider_options.copy(),
            domain_schema=self.domain_schema,
        )


def get_effective_reasoning(config: LLMConfig) -> ReasoningLevel:
    """Get effective reasoning level based on config.

    Combines tier and explicit reasoning_level to determine final level.

    Args:
        config: LLM configuration

    Returns:
        Effective reasoning level to use

    Logic:
        1. If reasoning_level is explicitly set, use it
        2. If tier is HIGH, automatically use XHIGH reasoning
        3. If tier is LOW (Codex only), use LOW reasoning
        4. Otherwise, use MEDIUM (default)

    Example:
        >>> config = LLMConfig(tier=ModelTier.HIGH)
        >>> get_effective_reasoning(config)
        ReasoningLevel.XHIGH  # HIGH tier auto-enables max reasoning
    """
    # Explicit reasoning level takes precedence
    if config.reasoning_level is not None:
        return config.reasoning_level

    # HIGH tier automatically uses XHIGH reasoning
    if config.tier == ModelTier.HIGH:
        return ReasoningLevel.XHIGH

    # LOW tier for Codex uses LOW reasoning
    if config.tier == ModelTier.LOW and config.provider == Provider.CODEX:
        return ReasoningLevel.LOW

    # Default to MEDIUM
    return ReasoningLevel.MEDIUM


# =============================================================================
# Preset Configurations
# =============================================================================

# Safety-first: require approval, read-only
SAFE_CONFIG = LLMConfig(
    auto_approval=AutoApproval.NONE,
    sandbox=SandboxMode.READ_ONLY,
)

# Autonomous: auto-approve, workspace write
AUTO_CONFIG = LLMConfig(
    auto_approval=AutoApproval.FULL,
    sandbox=SandboxMode.WORKSPACE_WRITE,
)

# Provider-specific defaults
CLAUDE_CONFIG = LLMConfig(provider=Provider.CLAUDE)
CODEX_CONFIG = LLMConfig(provider=Provider.CODEX)
GEMINI_CONFIG = LLMConfig(provider=Provider.GEMINI)


# =============================================================================
# Test/Dev Mode Configurations
# =============================================================================

def create_test_dev_config(provider: Provider) -> LLMConfig:
    """Create a test/dev configuration for the given provider.

    Uses low-cost, fast models optimized for testing:
    - Claude: haiku
    - Gemini: gemini-2.5-flash-lite
    - Codex: gpt-5.2 with low reasoning

    Args:
        provider: LLM provider to use

    Returns:
        LLMConfig configured for test/dev mode
    """
    # Use LOW reasoning for Codex test/dev
    reasoning = ReasoningLevel.LOW if provider == Provider.CODEX else ReasoningLevel.NONE
    provider_options: dict[str, Any] = (
        {"skip_git_check": True} if provider == Provider.CODEX else {}
    )
    return LLMConfig(
        provider=provider,
        model=TEST_DEV_MODELS[provider],
        reasoning_level=reasoning,
        auto_approval=AutoApproval.EDITS_ONLY,
        provider_options=provider_options,
    )


# Pre-built test/dev configs for each provider
TEST_DEV_CLAUDE_CONFIG = create_test_dev_config(Provider.CLAUDE)
TEST_DEV_CODEX_CONFIG = create_test_dev_config(Provider.CODEX)
TEST_DEV_GEMINI_CONFIG = create_test_dev_config(Provider.GEMINI)


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Main config class
    "LLMConfig",
    # Helper function
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
    # Re-export from llm-types for convenience
    "Feature",
    "FeatureValidationResult",
    "DomainSchema",
]
