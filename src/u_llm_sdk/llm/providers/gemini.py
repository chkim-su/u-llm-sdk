"""Gemini Provider Implementation.

Wraps Google Gemini CLI for unified SDK access.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from u_llm_sdk.types import LLMResult, Provider, SandboxMode

from u_llm_sdk.llm.providers.base import BaseProvider
from u_llm_sdk.llm.providers.parsing import GeminiEventParser

logger = logging.getLogger(__name__)


class GeminiProvider(BaseProvider):
    """Gemini CLI Provider.

    Wraps the Google Gemini CLI (gemini) for unified SDK access.

    CLI Usage:
        gemini "prompt" -m <model> -o stream-json --approval-mode <mode>

    Session Resume:
        gemini --resume <session_id>
        gemini --resume latest

    Web Search (via config.web_search=True):
        Gemini has GoogleSearch built-in and it works automatically.
        When web_search=True, GoogleSearch is explicitly added to allowed_tools
        to ensure it's available even when other tool restrictions are applied.

    Special Features (via provider_options):
        - temperature: Temperature setting (0.0-1.0) - NOTE: May be settings-based only
        - top_p: Top-p sampling - NOTE: May be settings-based only
        - top_k: Top-k sampling - NOTE: May be settings-based only
        - allowed_tools: List of allowed tools (space-separated CLI args)
        - extensions: List of extensions to enable (-e flag, multiple allowed)
        - include_directories: List of additional directories to include
        - allowed_mcp_server_names: List of allowed MCP server names

    Important Notes:
        - Gemini CLI outputs stream-json by default with -o flag
        - Reasoning/thinking levels have limited CLI support
        - Uses -y flag for YOLO (full auto) mode
        - Sandbox mode (-s) is a boolean flag (any non-NONE enables it)
        - Non-interactive mode uses positional argument for prompt (not -p flag)
        - Temperature/top_p/top_k: NOT CLI flags, use ~/.gemini/settings.json

    Image Input:
        Gemini CLI supports images via in-prompt references (NOT CLI flags):
        - @path/to/image.png - Reference image file in prompt
        Example: "Describe this: @./design/mockup.png"

    System Prompt:
        Gemini CLI uses file-based system prompts (NOT CLI flag):
        - .gemini/system.md in project directory
        - GEMINI.md in project root
    """

    PROVIDER = Provider.GEMINI
    CLI_NAME = "gemini"

    def _build_args(
        self,
        prompt: str,
        session_id: Optional[str] = None,
    ) -> list[str]:
        """Build Gemini CLI arguments."""
        args = [self.cli_executable]

        # Session resume uses --resume flag
        effective_session = self._get_effective_session_id(session_id)
        if effective_session:
            args.extend(["--resume", effective_session])
            return args  # Resume has limited options

        # Normal execution: gemini "prompt"
        args.append(prompt)

        # Output format - use stream-json for parsing
        args.extend(["-o", "stream-json"])

        # Model
        try:
            model = self.config.get_model()
            if model:
                args.extend(["-m", model])
        except Exception:
            pass  # ModelNotSpecifiedError - proceed without model flag

        # Approval mode
        approval_args = self.config.get_approval_args()
        if "yolo" in approval_args and approval_args["yolo"]:
            args.append("-y")
        elif "approval_mode" in approval_args:
            mode = approval_args["approval_mode"]
            if mode != "default":  # Don't pass default mode
                args.extend(["--approval-mode", mode])

        # Sandbox mode (Gemini uses -s as BOOLEAN flag, not mode-based like Codex)
        # Any non-NONE sandbox mode enables the sandbox
        if self.config.sandbox != SandboxMode.NONE:
            args.append("-s")  # Boolean flag, no value needed

        # Provider-specific options
        opts = self.config.provider_options

        # Temperature/top_p/top_k: NOT CLI flags (verified 2026-01-08)
        # Use ~/.gemini/settings.json instead
        if "temperature" in opts or "top_p" in opts or "top_k" in opts:
            logger.warning(
                "Gemini CLI does not support --temperature, --top-p, --top-k flags. "
                "Use ~/.gemini/settings.json for sampling parameters. "
                "These options will be ignored."
            )

        # Allowed tools (space-separated list)
        # Collect tools from provider_options
        # - list[str]: passed as separate CLI args (nargs-style)
        # - str: treated as already-formatted single argument (do not split)
        allowed_tools: list[str] = []
        allowed_tools_raw: Optional[str] = None
        if "allowed_tools" in opts:
            tools = opts["allowed_tools"]
            if isinstance(tools, list):
                allowed_tools.extend(tools)
            elif isinstance(tools, str):
                allowed_tools_raw = tools.strip()

        # Add GoogleSearch if web_search is enabled
        if self.config.web_search:
            if allowed_tools_raw is not None:
                raw_parts = allowed_tools_raw.split() if allowed_tools_raw else []
                if "GoogleSearch" not in raw_parts:
                    allowed_tools_raw = (
                        f"{allowed_tools_raw} GoogleSearch".strip()
                        if allowed_tools_raw
                        else "GoogleSearch"
                    )
            else:
                if "GoogleSearch" not in allowed_tools:
                    allowed_tools.append("GoogleSearch")

        if allowed_tools_raw is not None:
            if allowed_tools_raw:
                args.extend(["--allowed-tools", allowed_tools_raw])
        elif allowed_tools:
            args.extend(["--allowed-tools"] + allowed_tools)

        # Extensions (-e flag, can be specified multiple times)
        if "extensions" in opts:
            extensions = opts["extensions"]
            if isinstance(extensions, list):
                for ext in extensions:
                    args.extend(["-e", ext])
            elif isinstance(extensions, str):
                args.extend(["-e", extensions])

        # Include directories
        if "include_directories" in opts:
            dirs = opts["include_directories"]
            if isinstance(dirs, list):
                args.extend(["--include-directories"] + dirs)
            elif isinstance(dirs, str):
                args.extend(["--include-directories", dirs])

        # MCP server filter (allowed MCP server names)
        if "allowed_mcp_server_names" in opts:
            servers = opts["allowed_mcp_server_names"]
            if isinstance(servers, list):
                args.extend(["--allowed-mcp-server-names"] + servers)
            elif isinstance(servers, str):
                args.extend(["--allowed-mcp-server-names", servers])

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
        """Parse Gemini CLI stream-json output into LLMResult."""
        parser = GeminiEventParser(self.config, self.provider_name)
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


__all__ = ["GeminiProvider"]
