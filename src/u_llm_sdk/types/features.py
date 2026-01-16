"""Feature Support Matrix and Validation.

This module defines which features are supported by each provider's CLI,
and provides validation functions with LLM-friendly error messages.

Design Principles:
    - Structured data: All results are JSON-serializable
    - Actionable info: Alternative providers are always suggested
    - Severity levels: ERROR (blocks execution) vs WARNING (proceeds with degraded behavior)
    - LLM-friendly: Clear, parseable messages suitable for agent consumption
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Feature(Enum):
    """SDK features that may have varying provider support.

    Each feature maps to a specific configuration option in LLMConfig.
    """

    # Approval modes
    AUTO_APPROVAL_NONE = "auto_approval_none"
    AUTO_APPROVAL_EDITS_ONLY = "auto_approval_edits_only"
    AUTO_APPROVAL_FULL = "auto_approval_full"

    # Sandbox modes
    SANDBOX_READ_ONLY = "sandbox_read_only"
    SANDBOX_WORKSPACE_WRITE = "sandbox_workspace_write"
    SANDBOX_FULL_ACCESS = "sandbox_full_access"

    # Reasoning levels
    REASONING_CONTROL = "reasoning_control"

    # Session management
    SESSION_RESUME = "session_resume"

    # Advanced features
    STRUCTURED_OUTPUT = "structured_output"
    TOOL_CONTROL = "tool_control"
    SYSTEM_PROMPT = "system_prompt"
    IMAGE_INPUT = "image_input"
    WEB_SEARCH = "web_search"
    MCP_CONFIG = "mcp_config"
    COST_LIMIT = "cost_limit"


class Severity(Enum):
    """Validation result severity."""

    ERROR = "error"  # Blocks execution - feature will fail
    WARNING = "warning"  # Proceeds but feature is degraded/ignored


@dataclass
class FeatureSupport:
    """Support status for a feature on a specific provider.

    Attributes:
        supported: Whether the feature is supported
        cli_flag: The CLI flag used (if supported)
        notes: Additional implementation notes
        alternative_behavior: What happens if used despite no support
    """

    supported: bool
    cli_flag: Optional[str] = None
    notes: Optional[str] = None
    alternative_behavior: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "supported": self.supported,
            "cli_flag": self.cli_flag,
            "notes": self.notes,
            "alternative_behavior": self.alternative_behavior,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FeatureSupport":
        """Create from dictionary."""
        return cls(
            supported=data.get("supported", False),
            cli_flag=data.get("cli_flag"),
            notes=data.get("notes"),
            alternative_behavior=data.get("alternative_behavior"),
        )


@dataclass
class FeatureValidationResult:
    """Result of validating a feature configuration.

    This is designed to be LLM-friendly with structured, actionable information.

    Attributes:
        feature: The feature being validated
        provider: The provider being used
        severity: ERROR or WARNING
        supported: Whether the feature is supported
        message: Human-readable description
        suggestion: Actionable suggestion for the user/LLM
        supported_providers: List of providers that support this feature
        cli_flag_used: The CLI flag that would be used (if any)
        fallback_behavior: What will happen instead
        code: Machine-readable error code for programmatic handling
    """

    feature: Feature
    provider: str
    severity: Severity
    supported: bool
    message: str
    suggestion: str
    supported_providers: list[str] = field(default_factory=list)
    cli_flag_used: Optional[str] = None
    fallback_behavior: Optional[str] = None
    code: str = ""

    def __post_init__(self) -> None:
        """Generate error code if not provided."""
        if not self.code:
            self.code = f"{self.severity.value.upper()}_{self.feature.value.upper()}_{self.provider.upper()}"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization.

        LLM agents can parse this structured format for decision making.
        """
        return {
            "feature": self.feature.value,
            "provider": self.provider,
            "severity": self.severity.value,
            "supported": self.supported,
            "message": self.message,
            "suggestion": self.suggestion,
            "supported_providers": self.supported_providers,
            "cli_flag_used": self.cli_flag_used,
            "fallback_behavior": self.fallback_behavior,
            "code": self.code,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FeatureValidationResult":
        """Create from dictionary."""
        return cls(
            feature=Feature(data["feature"]),
            provider=data["provider"],
            severity=Severity(data["severity"]),
            supported=data.get("supported", False),
            message=data["message"],
            suggestion=data["suggestion"],
            supported_providers=data.get("supported_providers", []),
            cli_flag_used=data.get("cli_flag_used"),
            fallback_behavior=data.get("fallback_behavior"),
            code=data.get("code", ""),
        )

    def to_log_message(self) -> str:
        """Format as a log message with full context."""
        lines = [
            f"[{self.severity.value.upper()}] {self.code}",
            f"  Feature: {self.feature.value}",
            f"  Provider: {self.provider}",
            f"  Message: {self.message}",
            f"  Suggestion: {self.suggestion}",
        ]
        if self.supported_providers:
            lines.append(f"  Supported by: {', '.join(self.supported_providers)}")
        if self.fallback_behavior:
            lines.append(f"  Fallback: {self.fallback_behavior}")
        return "\n".join(lines)


# =============================================================================
# Feature Support Matrix
# =============================================================================

# Import Provider here to avoid circular import at module level
# This is used only in the matrix definition

FEATURE_SUPPORT_MATRIX: dict[str, dict[Feature, FeatureSupport]] = {
    "claude": {
        Feature.AUTO_APPROVAL_NONE: FeatureSupport(
            supported=True,
            cli_flag="--permission-mode default",
        ),
        Feature.AUTO_APPROVAL_EDITS_ONLY: FeatureSupport(
            supported=True,
            cli_flag="--permission-mode acceptEdits",
        ),
        Feature.AUTO_APPROVAL_FULL: FeatureSupport(
            supported=True,
            cli_flag="--permission-mode bypassPermissions",
        ),
        Feature.SANDBOX_READ_ONLY: FeatureSupport(
            supported=False,
            notes="Claude CLI does not have sandbox modes",
            alternative_behavior="No sandboxing applied",
        ),
        Feature.SANDBOX_WORKSPACE_WRITE: FeatureSupport(
            supported=False,
            notes="Claude CLI does not have sandbox modes",
            alternative_behavior="No sandboxing applied",
        ),
        Feature.SANDBOX_FULL_ACCESS: FeatureSupport(
            supported=False,
            notes="Claude CLI does not have sandbox modes",
            alternative_behavior="No sandboxing applied (default behavior)",
        ),
        Feature.REASONING_CONTROL: FeatureSupport(
            supported=True,
            cli_flag="Prompt prefix (think:, think hard:, ultrathink:)",
            notes="Uses prompt prefix triggers, not CLI flags",
        ),
        Feature.SESSION_RESUME: FeatureSupport(
            supported=True,
            cli_flag="--resume <session_id>",
        ),
        Feature.STRUCTURED_OUTPUT: FeatureSupport(
            supported=True,
            cli_flag="--json-schema <schema>",
            notes="Requires -p flag",
        ),
        Feature.TOOL_CONTROL: FeatureSupport(
            supported=True,
            cli_flag="--allowed-tools, --disallowed-tools",
        ),
        Feature.SYSTEM_PROMPT: FeatureSupport(
            supported=True,
            cli_flag="--system-prompt",
        ),
        Feature.IMAGE_INPUT: FeatureSupport(
            supported=False,
            notes="Claude CLI does not support image input",
        ),
        Feature.WEB_SEARCH: FeatureSupport(
            supported=False,
            notes="Claude CLI does not have built-in web search",
        ),
        Feature.MCP_CONFIG: FeatureSupport(
            supported=True,
            cli_flag="--mcp-config <path>",
        ),
        Feature.COST_LIMIT: FeatureSupport(
            supported=True,
            cli_flag="--max-budget-usd <amount>",
            notes="Requires -p flag",
        ),
    },
    "codex": {
        Feature.AUTO_APPROVAL_NONE: FeatureSupport(
            supported=True,
            notes="Default interactive behavior (no flag needed)",
            alternative_behavior="Prompts for approval interactively",
        ),
        Feature.AUTO_APPROVAL_EDITS_ONLY: FeatureSupport(
            supported=False,
            notes="codex exec does NOT support -a/--ask-for-approval flag",
            alternative_behavior="Falls back to default behavior (interactive prompts)",
        ),
        Feature.AUTO_APPROVAL_FULL: FeatureSupport(
            supported=True,
            cli_flag="--full-auto",
            notes="Enables automatic approval with workspace-write sandbox",
        ),
        Feature.SANDBOX_READ_ONLY: FeatureSupport(
            supported=True,
            cli_flag="-s read-only",
        ),
        Feature.SANDBOX_WORKSPACE_WRITE: FeatureSupport(
            supported=True,
            cli_flag="-s workspace-write",
        ),
        Feature.SANDBOX_FULL_ACCESS: FeatureSupport(
            supported=True,
            cli_flag="-s danger-full-access",
            notes="Use with caution!",
        ),
        Feature.REASONING_CONTROL: FeatureSupport(
            supported=False,
            notes="--model-reasoning-effort is NOT available in codex exec",
            alternative_behavior="Reasoning level ignored; use model selection instead",
        ),
        Feature.SESSION_RESUME: FeatureSupport(
            supported=True,
            cli_flag="codex exec resume <session_id>",
            notes="Uses subcommand, not flag; limited options available",
        ),
        Feature.STRUCTURED_OUTPUT: FeatureSupport(
            supported=True,
            cli_flag="--output-schema <schema>",
        ),
        Feature.TOOL_CONTROL: FeatureSupport(
            supported=False,
            notes="Codex CLI does not support tool allow/deny lists",
        ),
        Feature.SYSTEM_PROMPT: FeatureSupport(
            supported=False,
            notes="Codex CLI does not support system prompts",
            alternative_behavior="Include system context in the prompt itself",
        ),
        Feature.IMAGE_INPUT: FeatureSupport(
            supported=True,
            cli_flag="-i, --image <path>",
            notes="Supported in codex exec mode (0.71.0+)",
        ),
        Feature.WEB_SEARCH: FeatureSupport(
            supported=False,
            notes="--search flag does NOT exist in codex exec CLI",
            alternative_behavior="Web search disabled; option ignored with warning",
        ),
        Feature.MCP_CONFIG: FeatureSupport(
            supported=False,
            notes="Codex CLI does not support MCP",
        ),
        Feature.COST_LIMIT: FeatureSupport(
            supported=False,
            notes="Codex CLI does not support cost limits",
        ),
    },
    "gemini": {
        Feature.AUTO_APPROVAL_NONE: FeatureSupport(
            supported=True,
            cli_flag="--approval-mode default",
        ),
        Feature.AUTO_APPROVAL_EDITS_ONLY: FeatureSupport(
            supported=True,
            cli_flag="--approval-mode auto_edit",
        ),
        Feature.AUTO_APPROVAL_FULL: FeatureSupport(
            supported=True,
            cli_flag="-y (--yolo)",
        ),
        Feature.SANDBOX_READ_ONLY: FeatureSupport(
            supported=False,
            notes="Gemini -s is a boolean flag, not mode-based",
            alternative_behavior="Use -s to enable sandboxing (single mode only)",
        ),
        Feature.SANDBOX_WORKSPACE_WRITE: FeatureSupport(
            supported=True,
            cli_flag="-s",
            notes="Boolean flag - enables sandbox (single mode)",
        ),
        Feature.SANDBOX_FULL_ACCESS: FeatureSupport(
            supported=False,
            notes="Gemini sandbox is boolean, no full-access mode",
            alternative_behavior="Omit -s flag for full access",
        ),
        Feature.REASONING_CONTROL: FeatureSupport(
            supported=False,
            notes="Gemini CLI does not support reasoning level control",
            alternative_behavior="Reasoning level ignored",
        ),
        Feature.SESSION_RESUME: FeatureSupport(
            supported=True,
            cli_flag="--resume <index|latest>",
            notes="Uses numeric index or 'latest', not session ID",
        ),
        Feature.STRUCTURED_OUTPUT: FeatureSupport(
            supported=False,
            notes="Gemini CLI does not support structured output schemas",
        ),
        Feature.TOOL_CONTROL: FeatureSupport(
            supported=True,
            cli_flag="--allowed-tools",
        ),
        Feature.SYSTEM_PROMPT: FeatureSupport(
            supported=False,
            notes="Gemini CLI does not support system prompts directly",
            alternative_behavior="Include system context in the prompt itself",
        ),
        Feature.IMAGE_INPUT: FeatureSupport(
            supported=False,
            notes="Gemini CLI does not support image input",
        ),
        Feature.WEB_SEARCH: FeatureSupport(
            supported=False,
            notes="Gemini CLI does not have built-in web search",
        ),
        Feature.MCP_CONFIG: FeatureSupport(
            supported=False,
            notes="Gemini CLI does not support MCP",
        ),
        Feature.COST_LIMIT: FeatureSupport(
            supported=False,
            notes="Gemini CLI does not support cost limits",
        ),
    },
}


# =============================================================================
# Validation Functions
# =============================================================================


def get_feature_support(provider: str, feature: Feature) -> FeatureSupport:
    """Get support status for a feature on a provider.

    Args:
        provider: Provider name (claude, codex, gemini)
        feature: Feature to check

    Returns:
        FeatureSupport object with details
    """
    provider_lower = provider.lower()
    if provider_lower not in FEATURE_SUPPORT_MATRIX:
        return FeatureSupport(
            supported=False,
            notes=f"Unknown provider: {provider}",
        )

    return FEATURE_SUPPORT_MATRIX[provider_lower].get(
        feature,
        FeatureSupport(supported=False, notes="Feature not defined for this provider"),
    )


def get_providers_supporting(feature: Feature) -> list[str]:
    """Get list of providers that support a feature.

    Args:
        feature: Feature to check

    Returns:
        List of provider names that support the feature
    """
    return [
        provider
        for provider, features in FEATURE_SUPPORT_MATRIX.items()
        if features.get(feature, FeatureSupport(supported=False)).supported
    ]


def validate_feature(
    provider: str,
    feature: Feature,
    strict: bool = False,
) -> Optional[FeatureValidationResult]:
    """Validate if a feature is supported for a provider.

    Args:
        provider: Provider name
        feature: Feature to validate
        strict: If True, unsupported features return ERROR; otherwise WARNING

    Returns:
        FeatureValidationResult if feature is unsupported or has caveats,
        None if fully supported with no issues

    Example:
        >>> result = validate_feature("codex", Feature.AUTO_APPROVAL_EDITS_ONLY)
        >>> if result:
        ...     print(result.to_dict())  # LLM can parse this
        ...     logger.warning(result.to_log_message())
    """
    support = get_feature_support(provider, feature)
    supported_providers = get_providers_supporting(feature)

    if support.supported and not support.notes:
        # Fully supported, no issues
        return None

    if support.supported:
        # Supported but has notes/caveats
        return FeatureValidationResult(
            feature=feature,
            provider=provider,
            severity=Severity.WARNING,
            supported=True,
            message=f"Feature '{feature.value}' is supported with caveats: {support.notes}",
            suggestion="Review the implementation notes for any limitations.",
            supported_providers=supported_providers,
            cli_flag_used=support.cli_flag,
        )

    # Not supported
    severity = Severity.ERROR if strict else Severity.WARNING
    suggestion = _build_suggestion(feature, supported_providers, support)

    return FeatureValidationResult(
        feature=feature,
        provider=provider,
        severity=severity,
        supported=False,
        message=f"Feature '{feature.value}' is NOT supported by {provider} CLI. {support.notes or ''}".strip(),
        suggestion=suggestion,
        supported_providers=supported_providers,
        cli_flag_used=None,
        fallback_behavior=support.alternative_behavior,
    )


def _build_suggestion(
    feature: Feature,
    supported_providers: list[str],
    support: FeatureSupport,
) -> str:
    """Build actionable suggestion for unsupported feature."""
    parts = []

    if supported_providers:
        parts.append(f"Use one of these providers instead: {', '.join(supported_providers)}")

    if support.alternative_behavior:
        parts.append(f"Current behavior: {support.alternative_behavior}")

    if not parts:
        parts.append("This feature is not available in any provider CLI.")

    return " | ".join(parts)


def validate_config_features(
    provider: str,
    auto_approval: Optional[str] = None,
    sandbox: Optional[str] = None,
    reasoning_level: Optional[str] = None,
    strict: bool = False,
) -> list[FeatureValidationResult]:
    """Validate multiple features from a config at once.

    Args:
        provider: Provider name
        auto_approval: AutoApproval value (none, edits, full)
        sandbox: SandboxMode value (none, read-only, workspace-write, full-access)
        reasoning_level: ReasoningLevel value (none, low, medium, high, xhigh)
        strict: If True, unsupported features return ERROR

    Returns:
        List of validation results for unsupported/caveated features
    """
    results = []

    # Check auto_approval
    if auto_approval:
        feature_map = {
            "none": Feature.AUTO_APPROVAL_NONE,
            "edits": Feature.AUTO_APPROVAL_EDITS_ONLY,
            "full": Feature.AUTO_APPROVAL_FULL,
        }
        if auto_approval in feature_map:
            result = validate_feature(provider, feature_map[auto_approval], strict)
            if result:
                results.append(result)

    # Check sandbox
    if sandbox and sandbox != "none":
        feature_map = {
            "read-only": Feature.SANDBOX_READ_ONLY,
            "workspace-write": Feature.SANDBOX_WORKSPACE_WRITE,
            "full-access": Feature.SANDBOX_FULL_ACCESS,
        }
        if sandbox in feature_map:
            result = validate_feature(provider, feature_map[sandbox], strict)
            if result:
                results.append(result)

    # Check reasoning
    if reasoning_level and reasoning_level not in ("none", "medium"):
        result = validate_feature(provider, Feature.REASONING_CONTROL, strict)
        if result:
            results.append(result)

    return results


__all__ = [
    "Feature",
    "Severity",
    "FeatureSupport",
    "FeatureValidationResult",
    "FEATURE_SUPPORT_MATRIX",
    "get_feature_support",
    "get_providers_supporting",
    "validate_feature",
    "validate_config_features",
]
