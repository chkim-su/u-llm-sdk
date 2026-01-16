"""Tests for U-llm-sdk configuration."""

import pytest

from u_llm_sdk import (
    AUTO_CONFIG,
    CLAUDE_CONFIG,
    CODEX_CONFIG,
    GEMINI_CONFIG,
    SAFE_CONFIG,
    TEST_DEV_CLAUDE_CONFIG,
    TEST_DEV_CODEX_CONFIG,
    TEST_DEV_GEMINI_CONFIG,
    AutoApproval,
    LLMConfig,
    ModelNotSpecifiedError,
    ModelTier,
    Provider,
    ReasoningLevel,
    SandboxMode,
    create_test_dev_config,
    get_effective_reasoning,
)


class TestLLMConfig:
    """Test LLMConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = LLMConfig()
        assert config.provider == Provider.CLAUDE
        assert config.model is None
        assert config.tier is None
        assert config.auto_approval == AutoApproval.EDITS_ONLY
        assert config.sandbox == SandboxMode.NONE
        assert config.timeout == 1200.0
        assert config.intervention_hook is None

    def test_get_model_with_tier(self):
        """Test model resolution with tier."""
        config = LLMConfig(provider=Provider.CLAUDE, tier=ModelTier.HIGH)
        assert config.get_model() == "opus"

        config = LLMConfig(provider=Provider.CLAUDE, tier=ModelTier.LOW)
        assert config.get_model() == "haiku"

        config = LLMConfig(provider=Provider.CODEX, tier=ModelTier.HIGH)
        assert config.get_model() == "gpt-5.2"

        config = LLMConfig(provider=Provider.GEMINI, tier=ModelTier.HIGH)
        assert config.get_model() == "gemini-3-pro-preview"

    def test_get_model_with_explicit_model(self):
        """Test model resolution with explicit model."""
        config = LLMConfig(provider=Provider.CLAUDE, model="opus")
        assert config.get_model() == "opus"

    def test_get_model_no_model_no_tier_raises(self):
        """Test that no model/tier raises ModelNotSpecifiedError."""
        config = LLMConfig(provider=Provider.CLAUDE)
        with pytest.raises(ModelNotSpecifiedError):
            config.get_model(require_explicit=True)

    def test_get_model_no_model_no_tier_fallback(self):
        """Test that no model/tier returns empty when not required."""
        config = LLMConfig(provider=Provider.CLAUDE)
        assert config.get_model(require_explicit=False) == ""

    def test_get_approval_args_claude(self):
        """Test approval args for Claude."""
        config = LLMConfig(provider=Provider.CLAUDE, auto_approval=AutoApproval.NONE)
        assert config.get_approval_args() == {"permission_mode": "default"}

        config = LLMConfig(provider=Provider.CLAUDE, auto_approval=AutoApproval.EDITS_ONLY)
        assert config.get_approval_args() == {"permission_mode": "acceptEdits"}

        config = LLMConfig(provider=Provider.CLAUDE, auto_approval=AutoApproval.FULL)
        assert config.get_approval_args() == {"permission_mode": "bypassPermissions"}

    def test_get_approval_args_codex(self):
        """Test approval args for Codex.

        NOTE: In codex exec subcommand, -a flag is NOT available.
        Only --full-auto and --dangerously-bypass-approvals-and-sandbox work.
        AutoApproval.NONE and EDITS_ONLY return empty dict (default behavior).
        """
        config = LLMConfig(provider=Provider.CODEX, auto_approval=AutoApproval.NONE)
        assert config.get_approval_args() == {}  # No -a flag in codex exec

        config = LLMConfig(provider=Provider.CODEX, auto_approval=AutoApproval.EDITS_ONLY)
        assert config.get_approval_args() == {}  # No direct equivalent

        config = LLMConfig(provider=Provider.CODEX, auto_approval=AutoApproval.FULL)
        assert config.get_approval_args() == {"full_auto": True}

    def test_get_reasoning_args(self):
        """Test reasoning args."""
        config = LLMConfig(provider=Provider.CLAUDE, reasoning_level=ReasoningLevel.XHIGH)
        args = config.get_reasoning_args()
        assert args == {"thinking_prefix": "ultrathink: "}

        # NOTE: --model-reasoning-effort flag is NOT available in codex exec CLI
        config = LLMConfig(provider=Provider.CODEX, reasoning_level=ReasoningLevel.HIGH)
        args = config.get_reasoning_args()
        assert args == {}  # Codex exec doesn't support reasoning flags

    def test_with_provider(self):
        """Test with_provider creates new config."""
        original = LLMConfig(provider=Provider.CLAUDE, tier=ModelTier.HIGH)
        new = original.with_provider(Provider.CODEX)

        assert original.provider == Provider.CLAUDE
        assert new.provider == Provider.CODEX
        assert new.tier == ModelTier.HIGH

    def test_with_model(self):
        """Test with_model creates new config."""
        original = LLMConfig(provider=Provider.CLAUDE)
        new = original.with_model("opus")

        assert original.model is None
        assert new.model == "opus"

    def test_with_tier(self):
        """Test with_tier creates new config and clears model."""
        original = LLMConfig(provider=Provider.CLAUDE, model="opus")
        new = original.with_tier(ModelTier.LOW)

        assert original.model == "opus"
        assert new.model is None
        assert new.tier == ModelTier.LOW

    def test_with_reasoning(self):
        """Test with_reasoning creates new config."""
        original = LLMConfig(provider=Provider.CLAUDE)
        new = original.with_reasoning(ReasoningLevel.HIGH)

        assert original.reasoning_level is None
        assert new.reasoning_level == ReasoningLevel.HIGH


class TestEffectiveReasoning:
    """Test get_effective_reasoning function."""

    def test_explicit_reasoning_takes_precedence(self):
        """Explicit reasoning_level should override tier."""
        config = LLMConfig(
            tier=ModelTier.HIGH,
            reasoning_level=ReasoningLevel.LOW,
        )
        assert get_effective_reasoning(config) == ReasoningLevel.LOW

    def test_high_tier_uses_xhigh(self):
        """HIGH tier should use XHIGH reasoning."""
        config = LLMConfig(tier=ModelTier.HIGH)
        assert get_effective_reasoning(config) == ReasoningLevel.XHIGH

    def test_low_tier_codex_uses_low(self):
        """LOW tier for Codex should use LOW reasoning."""
        config = LLMConfig(provider=Provider.CODEX, tier=ModelTier.LOW)
        assert get_effective_reasoning(config) == ReasoningLevel.LOW

    def test_low_tier_claude_uses_medium(self):
        """LOW tier for Claude should use MEDIUM reasoning."""
        config = LLMConfig(provider=Provider.CLAUDE, tier=ModelTier.LOW)
        assert get_effective_reasoning(config) == ReasoningLevel.MEDIUM

    def test_default_uses_medium(self):
        """Default should use MEDIUM reasoning."""
        config = LLMConfig()
        assert get_effective_reasoning(config) == ReasoningLevel.MEDIUM


class TestPresetConfigs:
    """Test preset configurations."""

    def test_safe_config(self):
        """Test SAFE_CONFIG preset."""
        assert SAFE_CONFIG.auto_approval == AutoApproval.NONE
        assert SAFE_CONFIG.sandbox == SandboxMode.READ_ONLY

    def test_auto_config(self):
        """Test AUTO_CONFIG preset."""
        assert AUTO_CONFIG.auto_approval == AutoApproval.FULL
        assert AUTO_CONFIG.sandbox == SandboxMode.WORKSPACE_WRITE

    def test_provider_configs(self):
        """Test provider-specific presets."""
        assert CLAUDE_CONFIG.provider == Provider.CLAUDE
        assert CODEX_CONFIG.provider == Provider.CODEX
        assert GEMINI_CONFIG.provider == Provider.GEMINI


class TestTestDevConfigs:
    """Test test/dev mode configurations."""

    def test_test_dev_claude_config(self):
        """Test TEST_DEV_CLAUDE_CONFIG."""
        assert TEST_DEV_CLAUDE_CONFIG.provider == Provider.CLAUDE
        assert TEST_DEV_CLAUDE_CONFIG.model == "haiku"
        assert TEST_DEV_CLAUDE_CONFIG.reasoning_level == ReasoningLevel.NONE

    def test_test_dev_codex_config(self):
        """Test TEST_DEV_CODEX_CONFIG."""
        assert TEST_DEV_CODEX_CONFIG.provider == Provider.CODEX
        assert TEST_DEV_CODEX_CONFIG.model == "gpt-5.2"
        assert TEST_DEV_CODEX_CONFIG.reasoning_level == ReasoningLevel.LOW
        assert TEST_DEV_CODEX_CONFIG.provider_options.get("skip_git_check") is True

    def test_test_dev_gemini_config(self):
        """Test TEST_DEV_GEMINI_CONFIG."""
        assert TEST_DEV_GEMINI_CONFIG.provider == Provider.GEMINI
        assert TEST_DEV_GEMINI_CONFIG.model == "gemini-2.5-flash-lite"
        assert TEST_DEV_GEMINI_CONFIG.reasoning_level == ReasoningLevel.NONE

    def test_create_test_dev_config(self):
        """Test create_test_dev_config factory."""
        config = create_test_dev_config(Provider.CLAUDE)
        assert config.provider == Provider.CLAUDE
        assert config.model == "haiku"


class TestConfigChaining:
    """Test config method chaining."""

    def test_chain_with_methods(self):
        """Test chaining multiple with_* methods."""
        config = (
            SAFE_CONFIG
            .with_provider(Provider.CODEX)
            .with_tier(ModelTier.HIGH)
            .with_reasoning(ReasoningLevel.HIGH)
        )

        assert config.provider == Provider.CODEX
        assert config.tier == ModelTier.HIGH
        assert config.reasoning_level == ReasoningLevel.HIGH
        # Original preset properties preserved
        assert config.auto_approval == AutoApproval.NONE

    def test_original_unchanged(self):
        """Test that original config is unchanged after chaining."""
        original = SAFE_CONFIG
        _ = original.with_provider(Provider.CODEX)

        assert original.provider == Provider.CLAUDE
