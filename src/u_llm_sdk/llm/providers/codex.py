"""Codex Provider Implementation.

Wraps OpenAI Codex CLI for unified SDK access.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from u_llm_sdk.types import LLMResult, Provider

from u_llm_sdk.llm.providers.base import BaseProvider
from u_llm_sdk.llm.providers.parsing import CodexEventParser

logger = logging.getLogger(__name__)


class CodexProvider(BaseProvider):
    """Codex CLI Provider.

    Wraps the OpenAI Codex CLI (codex) for unified SDK access.

    CLI Usage:
        codex exec "prompt" --json -m <model> -s <sandbox> [options]

    Session Resume (different subcommand):
        codex resume <session_id>

    Approval Mode Mapping:
        NOTE: codex exec does NOT support -a/--ask-for-approval flag!
        - AutoApproval.NONE → Default interactive behavior (no flag)
        - AutoApproval.EDITS_ONLY → NOT SUPPORTED (warning logged, fallback to default)
        - AutoApproval.FULL → --full-auto (automatic approval)

    Sandbox Mode Mapping:
        - SandboxMode.READ_ONLY → -s read-only
        - SandboxMode.WORKSPACE_WRITE → -s workspace-write
        - SandboxMode.FULL_ACCESS → -s danger-full-access

    Special Features (via provider_options):
        - images: List of image file paths (-i flag, works in exec mode)
        - search: Enable web search (--search flag)
        - full_auto: Full auto mode (--full-auto flag)
        - output_schema: JSON schema file path (--output-schema flag)
        - add_dirs: List of additional directories (--add-dir flag)
        - features_enable: List of features to enable (--enable flag)
        - features_disable: List of features to disable (--disable flag)
        - config_overrides: Dict of config overrides (-c flag)
        - skip_git_repo_check: Skip git repository check
        - reasoning_effort: Reasoning effort level ("low", "medium", "high", "xhigh")
          Maps to -c model_reasoning_effort="<value>"
        - temperature: Temperature setting (0.0-2.0)
          Maps to -c temperature=<value>

    MCP Server Support:
        - Codex supports MCP servers via `mcp_servers.*` config in ~/.codex/config.toml
        - Use `codex mcp list` to see configured servers
        - NOT exposed as exec flags; configure via config file or -c flag

    Important Notes:
        - Resume returns plain text (limited parsing)
        - JSON output is JSONL format (line by line JSON events)
    """

    PROVIDER = Provider.CODEX
    CLI_NAME = "codex"

    def _build_args(
        self,
        prompt: str,
        session_id: Optional[str] = None,
    ) -> list[str]:
        """Build Codex CLI arguments.

        Supports CLI features (codex exec 0.71.0+):
        - codex exec subcommand for non-interactive mode
        - JSON output via --json flag (JSONL format)
        - Approval mode: --full-auto ONLY (-a flag does NOT exist)
        - Sandbox modes: -s read-only/workspace-write/danger-full-access
        - Image inputs: -i <path>
        - Web search: --search flag (enables native web_search tool)
        - Full auto: --full-auto
        - Output schema: --output-schema <path>
        - Additional directories: --add-dir <path>
        - Working directory: -C <path>
        - Feature flags: --enable/--disable
        - Config overrides: -c <key>=<value>
        """
        args = [self.cli_executable]

        # Session resume uses different subcommand
        effective_session = self._get_effective_session_id(session_id)
        if effective_session:
            args.extend(["resume", effective_session])
            return args  # Resume has limited options

        # Normal execution: codex exec "prompt"
        args.extend(["exec", prompt])

        # Output format - always use JSON for parsing (JSONL format)
        args.append("--json")

        # Model
        try:
            model = self.config.get_model()
            if model:
                args.extend(["-m", model])
        except Exception:
            pass  # ModelNotSpecifiedError - proceed without model flag

        # Approval mode (--full-auto flag only)
        # NOTE: codex exec does NOT support -a/--ask-for-approval flag!
        # AutoApproval.NONE → No flags (default interactive behavior)
        # AutoApproval.EDITS_ONLY → Not supported (warning logged, falls back to default)
        # AutoApproval.FULL → --full-auto (automatic approval)
        from u_llm_sdk.types import AutoApproval, SandboxMode

        approval_mode = self.config.auto_approval
        if approval_mode == AutoApproval.FULL:
            # --full-auto is the ONLY approval flag that exists in codex exec
            args.append("--full-auto")
        elif approval_mode == AutoApproval.EDITS_ONLY:
            # Not supported - log warning and fall back to default
            logger.warning(
                "AutoApproval.EDITS_ONLY is not supported by codex exec CLI. "
                "Falling back to default interactive behavior. "
                "Use AutoApproval.FULL with --full-auto for automatic approval."
            )
        # AutoApproval.NONE uses default behavior (no flag needed)

        # Sandbox mode (-s flag)
        # SandboxMode.READ_ONLY → -s read-only
        # SandboxMode.WORKSPACE → -s workspace-write
        # SandboxMode.FULL_ACCESS → -s danger-full-access
        sandbox_mode = self.config.sandbox
        sandbox_mapping = {
            SandboxMode.READ_ONLY: "read-only",
            SandboxMode.WORKSPACE_WRITE: "workspace-write",
            SandboxMode.FULL_ACCESS: "danger-full-access",
        }
        if sandbox_mode in sandbox_mapping:
            args.extend(["-s", sandbox_mapping[sandbox_mode]])

        # Working directory (-C flag)
        if self.config.cwd:
            args.extend(["-C", self.config.cwd])

        # Provider-specific options
        opts = self.config.provider_options

        # Image inputs (-i flag, can be repeated)
        if "images" in opts:
            images = opts["images"]
            if isinstance(images, list):
                for image_path in images:
                    args.extend(["-i", str(image_path)])

        # Web search (--search flag)
        # Verified: --search flag EXISTS in codex exec CLI (v0.71.0+)
        # "Enable web search (off by default). When enabled, uses native web_search tool"
        if opts.get("search") or self.config.web_search:
            args.append("--search")

        # Full auto (--full-auto flag) from provider_options
        # This provides an alternative to AutoApproval.FULL
        if opts.get("full_auto"):
            if "--full-auto" not in args:  # Avoid duplicate flags
                args.append("--full-auto")

        # Output schema (--output-schema flag)
        if "output_schema" in opts:
            args.extend(["--output-schema", str(opts["output_schema"])])

        # Additional directories (--add-dir flag, can be repeated)
        if "add_dirs" in opts:
            add_dirs = opts["add_dirs"]
            if isinstance(add_dirs, list):
                for dir_path in add_dirs:
                    args.extend(["--add-dir", str(dir_path)])

        # Feature enable flags (--enable flag, can be repeated)
        if "features_enable" in opts:
            features = opts["features_enable"]
            if isinstance(features, list):
                for feature in features:
                    args.extend(["--enable", str(feature)])

        # Feature disable flags (--disable flag, can be repeated)
        if "features_disable" in opts:
            features = opts["features_disable"]
            if isinstance(features, list):
                for feature in features:
                    args.extend(["--disable", str(feature)])

        # Config overrides (-c flag, can be repeated)
        if "config_overrides" in opts:
            config_overrides = opts["config_overrides"]
            if isinstance(config_overrides, dict):
                for key, value in config_overrides.items():
                    args.extend(["-c", f"{key}={value}"])

        # Skip git repo check (--skip-git-repo-check flag)
        if opts.get("skip_git_repo_check"):
            args.append("--skip-git-repo-check")

        # Reasoning effort (-c model_reasoning_effort="<value>")
        # Valid values: "low", "medium", "high", "xhigh"
        if "reasoning_effort" in opts:
            effort = opts["reasoning_effort"]
            if effort in ("low", "medium", "high", "xhigh"):
                args.extend(["-c", f'model_reasoning_effort="{effort}"'])

        # Temperature (-c temperature=<value>)
        if "temperature" in opts:
            temp = opts["temperature"]
            args.extend(["-c", f"temperature={temp}"])

        return args

    # run() and stream() are inherited from BaseProvider (Template Method pattern)

    def _parse_output(
        self,
        stdout: str,
        stderr: str,
        success: bool,
        duration_ms: int,
        **kwargs: Any,
    ) -> LLMResult:
        """Parse Codex CLI JSON output into LLMResult."""
        parser = CodexEventParser(self.config, self.provider_name)
        session_id_ref: list[Optional[str]] = [self._session_id]

        result = parser.parse(
            stdout=stdout,
            stderr=stderr,
            success=success,
            duration_ms=duration_ms,
            session_id_ref=session_id_ref,
            **kwargs,
        )

        # Update session_id from parser
        if session_id_ref[0]:
            self._session_id = session_id_ref[0]

        return result


__all__ = ["CodexProvider"]
