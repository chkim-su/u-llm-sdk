"""LLM Types - Configuration Enums and Model Mappings.

This module defines all configuration enums and model mappings used by
both U-llm-sdk and MV-rag. These are shared types that both packages depend on.

Model Configuration (December 2025):
- Defines current models per provider (MODEL_TIERS)
- Supports legacy model routing (LEGACY_MODEL_TIERS)
- Provides resolve_model() for automatic tier-based routing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Provider(Enum):
    """Available LLM providers.

    Attributes:
        CLAUDE: Anthropic Claude CLI
        CODEX: OpenAI Codex CLI
        GEMINI: Google Gemini CLI
    """
    CLAUDE = "claude"
    CODEX = "codex"
    GEMINI = "gemini"


class AutoApproval(Enum):
    """Unified approval mode for all providers.

    This maps to provider-specific settings internally:
    - NONE: Require approval for all actions
    - EDITS_ONLY: Auto-approve file edits only
    - FULL: Auto-approve all actions (use with caution!)

    Attributes:
        NONE: Require approval for all actions
        EDITS_ONLY: Auto-approve file edits
        FULL: Auto-approve all actions (dangerous!)
    """
    NONE = "none"
    EDITS_ONLY = "edits"
    FULL = "full"


class SandboxMode(Enum):
    """Sandbox mode for command execution (Codex/Gemini).

    Attributes:
        NONE: No sandbox (Claude default)
        READ_ONLY: Read-only access
        WORKSPACE_WRITE: Write to workspace only
        FULL_ACCESS: Full system access (dangerous!)
    """
    NONE = "none"
    READ_ONLY = "read-only"
    WORKSPACE_WRITE = "workspace-write"
    FULL_ACCESS = "full-access"


class ModelTier(Enum):
    """Model performance tier for automatic model routing.

    Use this to select models by performance level instead of specific names.
    Legacy or unknown model names will be automatically routed to the
    appropriate tier's current model.

    Attributes:
        HIGH: Best performance (complex reasoning, difficult tasks)
        LOW: Fast/cheap (simple tasks, quick responses)
    """
    HIGH = "high"
    LOW = "low"


class ReasoningLevel(Enum):
    """Reasoning/thinking intensity level.

    Controls how much reasoning effort the model applies:
    - Claude: Maps to extended thinking with budget control
    - Codex: Maps to model_reasoning_effort setting
    - Gemini: Limited CLI support (API-level only)

    Attributes:
        NONE: No reasoning (fastest)
        LOW: Minimal reasoning (LOW tier auto-applies this for Codex)
        MEDIUM: Basic reasoning (default)
        HIGH: Deep reasoning
        XHIGH: Maximum reasoning (expensive, HIGH tier auto-applies this)
    """
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"


# Type alias for convenience
ProviderType = Provider
AutoApprovalType = AutoApproval
SandboxModeType = SandboxMode
ModelTierType = ModelTier
ReasoningLevelType = ReasoningLevel


# =============================================================================
# Model Tier Mappings (January 2026)
# =============================================================================

# Model tier mappings - current models per provider
# NOTE: These are CLI model names. For API-direct calls, use API_MODEL_TIERS below.
MODEL_TIERS: dict[Provider, dict[ModelTier, str]] = {
    Provider.CLAUDE: {
        ModelTier.HIGH: "opus",   # Claude Opus
        ModelTier.LOW: "haiku",   # Claude Haiku
    },
    Provider.CODEX: {
        ModelTier.HIGH: "gpt-5.2",  # GPT-5.2 (with xhigh thinking via reasoning level)
        ModelTier.LOW: "gpt-5.2",   # GPT-5.2 (with low thinking via reasoning level)
    },
    Provider.GEMINI: {
        ModelTier.HIGH: "gemini-3-pro",          # Gemini 3 Pro (CLI)
        ModelTier.LOW: "gemini-3-flash-preview", # Gemini 3 Flash Preview
    },
}


# API model tier mappings - for direct API calls (e.g., GeminiAPIProvider)
# NOTE: Only Gemini has different model names between CLI and API.
API_MODEL_TIERS: dict[Provider, dict[ModelTier, str]] = {
    Provider.GEMINI: {
        ModelTier.HIGH: "gemini-3-pro-preview",  # Gemini 3 Pro Preview (API)
        ModelTier.LOW: "gemini-3-flash-preview", # Gemini 3 Flash Preview
    },
}


# Legacy model → tier mapping (for automatic routing)
# When a legacy model name is used, resolve_model() routes to the current model.
LEGACY_MODEL_TIERS: dict[str, ModelTier] = {
    # Claude legacy models
    "claude-3-opus": ModelTier.HIGH,
    "claude-3-sonnet": ModelTier.HIGH,
    "claude-3-haiku": ModelTier.LOW,
    "claude-2": ModelTier.HIGH,
    "claude-instant": ModelTier.LOW,

    # OpenAI legacy models
    "gpt-4": ModelTier.HIGH,
    "gpt-4-turbo": ModelTier.HIGH,
    "gpt-4o": ModelTier.HIGH,
    "gpt-3.5-turbo": ModelTier.LOW,
    "o1-preview": ModelTier.HIGH,
    "o1-mini": ModelTier.LOW,

    # Gemini legacy models
    "gemini-pro": ModelTier.HIGH,
    "gemini-1.5-pro": ModelTier.HIGH,
    "gemini-1.5-flash": ModelTier.LOW,
    "gemini-2.5-pro": ModelTier.HIGH,           # Route to gemini-3-pro
    "gemini-2.5-flash-lite": ModelTier.LOW,     # Route to gemini-3-flash-preview
    "gemini-3-pro-preview": ModelTier.HIGH,     # API name → CLI route
}


# Current models (no routing needed) - January 2026
CURRENT_MODELS: dict[Provider, set[str]] = {
    Provider.CLAUDE: {
        "opus", "haiku",                                   # Aliases
        "claude-opus-4-5-20251101",                        # Full names
        "claude-haiku-4-5-20251001",
    },
    Provider.CODEX: {
        "gpt-5.2",                                          # Current production model
        "gpt-5.1-codex-max",                               # High-end model
        "o1", "o3", "o4-mini",                              # Legacy but still supported
    },
    Provider.GEMINI: {
        "gemini-3-pro",                                     # Latest CLI model (HIGH)
        "gemini-3-pro-preview",                             # API model (HIGH)
        "gemini-3-flash-preview",                           # Flash preview (LOW)
        "gemini-2.5-pro", "gemini-2.5-flash-lite",         # Legacy but supported
        "gemini-2.0-flash",                                 # Legacy but supported
    },
}


# Test/Dev Mode - Low-cost models for testing
TEST_DEV_MODELS: dict[Provider, str] = {
    Provider.CLAUDE: "haiku",
    Provider.GEMINI: "gemini-3-flash-preview",  # Lightweight flash model
    Provider.CODEX: "gpt-5.2",  # Same model as production, but with low reasoning
}


# Default models per provider (empty = require explicit selection)
DEFAULT_MODELS: dict[Provider, str] = {
    Provider.CLAUDE: "",  # Require explicit selection
    Provider.CODEX: "",   # Require explicit selection
    Provider.GEMINI: "",  # Require explicit selection
}


# =============================================================================
# Reasoning Level Mappings
# =============================================================================

# Reasoning level mapping per provider
REASONING_MAP: dict[tuple[Provider, ReasoningLevel], dict] = {
    # Claude: ONLY "ultrathink" triggers extended thinking (allocates up to 31,999 tokens)
    # NOTE: "think" and "think hard" are NOT triggers - they are just plain text
    # This was verified against Claude CLI v2.0.0+ behavior (2026-01-08)
    (Provider.CLAUDE, ReasoningLevel.NONE): {"thinking_prefix": ""},
    (Provider.CLAUDE, ReasoningLevel.LOW): {"thinking_prefix": ""},
    (Provider.CLAUDE, ReasoningLevel.MEDIUM): {"thinking_prefix": ""},  # DEPRECATED: "think" removed
    (Provider.CLAUDE, ReasoningLevel.HIGH): {"thinking_prefix": ""},    # DEPRECATED: "think hard" removed
    (Provider.CLAUDE, ReasoningLevel.XHIGH): {"thinking_prefix": "ultrathink: "},

    # Codex: --model-reasoning-effort flag NOT available in codex exec CLI
    # Reasoning is controlled via model selection or config file, not CLI flags
    (Provider.CODEX, ReasoningLevel.NONE): {},
    (Provider.CODEX, ReasoningLevel.LOW): {},
    (Provider.CODEX, ReasoningLevel.MEDIUM): {},
    (Provider.CODEX, ReasoningLevel.HIGH): {},
    (Provider.CODEX, ReasoningLevel.XHIGH): {},

    # Gemini: CLI control not available, pass through
    (Provider.GEMINI, ReasoningLevel.NONE): {},
    (Provider.GEMINI, ReasoningLevel.LOW): {},
    (Provider.GEMINI, ReasoningLevel.MEDIUM): {},
    (Provider.GEMINI, ReasoningLevel.HIGH): {},
    (Provider.GEMINI, ReasoningLevel.XHIGH): {},
}

# Claude thinking trigger words for -p mode
# IMPORTANT: Only "ultrathink" actually triggers extended thinking in Claude CLI v2.0.0+
# "think" and "think hard" were deprecated as they do NOT allocate thinking tokens
CLAUDE_THINKING_TRIGGERS: dict[ReasoningLevel, str] = {
    ReasoningLevel.NONE: "",
    ReasoningLevel.LOW: "",
    ReasoningLevel.MEDIUM: "",    # DEPRECATED: "think" does not trigger thinking
    ReasoningLevel.HIGH: "",      # DEPRECATED: "think hard" does not trigger thinking
    ReasoningLevel.XHIGH: "ultrathink",  # Only working trigger (max 31,999 tokens)
}

# Deprecated thinking keywords (for reference/migration)
DEPRECATED_THINKING_KEYWORDS: dict[ReasoningLevel, str] = {
    ReasoningLevel.MEDIUM: "think",       # Was used before v2.0.0
    ReasoningLevel.HIGH: "think hard",    # Was used before v2.0.0
}


# =============================================================================
# Provider-specific Approval Mapping
# =============================================================================

APPROVAL_MAP: dict[tuple[Provider, AutoApproval], dict] = {
    # Claude mappings (--permission-mode flag)
    (Provider.CLAUDE, AutoApproval.NONE): {"permission_mode": "default"},
    (Provider.CLAUDE, AutoApproval.EDITS_ONLY): {"permission_mode": "acceptEdits"},
    (Provider.CLAUDE, AutoApproval.FULL): {"permission_mode": "bypassPermissions"},

    # Codex mappings (codex exec subcommand - NO -a flag available!)
    # Only --full-auto and --dangerously-bypass-approvals-and-sandbox are supported
    (Provider.CODEX, AutoApproval.NONE): {},  # Default interactive behavior
    (Provider.CODEX, AutoApproval.EDITS_ONLY): {},  # No direct equivalent in codex exec
    (Provider.CODEX, AutoApproval.FULL): {"full_auto": True},  # --full-auto

    # Gemini mappings (--approval-mode and -y flags)
    (Provider.GEMINI, AutoApproval.NONE): {"approval_mode": "default"},
    (Provider.GEMINI, AutoApproval.EDITS_ONLY): {"approval_mode": "auto_edit"},
    (Provider.GEMINI, AutoApproval.FULL): {"yolo": True},
}


# =============================================================================
# API and CLI Mappings
# =============================================================================

# API key environment variables per provider
API_KEY_ENV_VARS: dict[Provider, str] = {
    Provider.CLAUDE: "ANTHROPIC_API_KEY",
    Provider.CODEX: "OPENAI_API_KEY",
    Provider.GEMINI: "GOOGLE_API_KEY",
}


# Tool name mappings per provider (for output parsing)
FILE_EDIT_TOOLS: dict[Provider, set[str]] = {
    Provider.CLAUDE: {"Edit", "Write"},
    Provider.CODEX: {"write_file", "edit_file", "create_file"},
    Provider.GEMINI: {"write_file", "edit_file", "Write", "Edit", "WriteFile", "EditFile"},
}

FILE_WRITE_TOOLS: dict[Provider, set[str]] = {
    Provider.CLAUDE: {"Write"},
    Provider.CODEX: {"write_file", "create_file"},
    Provider.GEMINI: {"write_file", "Write", "WriteFile"},
}

SHELL_TOOLS: dict[Provider, set[str]] = {
    Provider.CLAUDE: {"Bash"},
    Provider.CODEX: {"shell", "shell_command"},
    Provider.GEMINI: {"shell", "run_command", "Bash", "Shell", "RunCommand"},
}


# CLI command structure per provider
CLI_COMMANDS: dict[Provider, dict[str, str]] = {
    Provider.CLAUDE: {
        "executable": "claude",
        "prompt_flag": "-p",
        "model_flag": "--model",
        "output_format_flag": "--output-format",
        "output_format_value": "stream-json",
        "resume_flag": "--resume",
        "permission_flag": "--permission-mode",
        "system_prompt_flag": "--system-prompt",
        "verbose_flag": "--verbose",
    },
    Provider.CODEX: {
        "executable": "codex",
        "subcommand": "exec",
        "model_flag": "-m",
        "json_flag": "--json",
        "resume_subcommand": "resume",
        # NOTE: -a/--ask-for-approval NOT available in "codex exec" subcommand!
        # Only main "codex" command supports -a flag.
        "full_auto_flag": "--full-auto",
        "bypass_flag": "--dangerously-bypass-approvals-and-sandbox",
        "sandbox_flag": "-s",
        "cwd_flag": "-C",
        "skip_git_flag": "--skip-git-repo-check",
        "config_flag": "-c",
    },
    Provider.GEMINI: {
        "executable": "gemini",
        "model_flag": "-m",
        "output_flag": "-o",
        "output_value": "stream-json",
        "resume_flag": "--resume",
        "yolo_flag": "-y",
        "approval_flag": "--approval-mode",
        "sandbox_flag": "-s",
    },
}


# =============================================================================
# Model Resolution Functions
# =============================================================================

def resolve_model(
    provider: Provider,
    model: Optional[str] = None,
    tier: Optional[ModelTier] = None,
    require_explicit: bool = True,
) -> str:
    """Resolve model name with automatic tier-based routing.

    Routes legacy or unknown model names to current models based on
    their performance tier.

    Args:
        provider: The LLM provider
        model: Explicit model name (may be legacy)
        tier: Explicit tier selection (overrides model)
        require_explicit: If True, raises error when no model/tier specified

    Returns:
        Resolved model name for the provider

    Raises:
        ModelNotSpecifiedError: If require_explicit=True and no model/tier given

    Priority:
        1. If tier is specified, return that tier's model
        2. If model is a current model, use it directly
        3. If model is a legacy model, route to appropriate tier
        4. If model is unknown, assume it's a valid model name (pass-through)
        5. If nothing specified and require_explicit=True, raise error
        6. If nothing specified and require_explicit=False, use provider default

    Example:
        >>> resolve_model(Provider.CODEX, model="gpt-4")
        'gpt-5.2'  # gpt-4 was HIGH tier → routes to current HIGH
        >>> resolve_model(Provider.CLAUDE, tier=ModelTier.HIGH)
        'opus'
        >>> resolve_model(Provider.CLAUDE)  # Raises ModelNotSpecifiedError!
    """
    # Tier explicitly specified
    if tier is not None:
        return MODEL_TIERS[provider][tier]

    # Model explicitly specified
    if model:
        # Current model - use directly
        if model in CURRENT_MODELS.get(provider, set()):
            return model

        # Legacy model - route to appropriate tier
        if model in LEGACY_MODEL_TIERS:
            inferred_tier = LEGACY_MODEL_TIERS[model]
            return MODEL_TIERS[provider][inferred_tier]

        # Unknown model - pass through (assume user knows what they're doing)
        return model

    # No model or tier specified
    if require_explicit:
        # Import here to avoid circular import
        from .exceptions import ModelNotSpecifiedError

        available_models = list(CURRENT_MODELS.get(provider, set()))
        raise ModelNotSpecifiedError(
            provider=provider.value,
            available_tiers=["HIGH", "LOW"],
            available_models=available_models,
        )

    # Fallback to default (only when require_explicit=False)
    return DEFAULT_MODELS.get(provider, "")
