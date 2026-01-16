"""Claude Provider Implementation.

Wraps Claude CLI for unified SDK access.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from u_llm_sdk.types import LLMResult, Provider

from u_llm_sdk.llm.providers.base import BaseProvider
from u_llm_sdk.llm.providers.parsing import ClaudeEventParser

logger = logging.getLogger(__name__)


class ClaudeProvider(BaseProvider):
    """Claude CLI Provider.

    Wraps the Claude CLI (claude) for unified SDK access.

    CLI Usage:
        claude -p "prompt" --output-format stream-json --permission-mode acceptEdits

    Session Resume:
        claude -p "prompt" --resume <session_id>

    Web Search (via config.web_search=True):
        Automatically adds WebSearch and WebFetch tools to --allowed-tools.
        - WebSearch: Query-based web search with optional domain filtering
        - WebFetch: Fetch and analyze content from specific URLs

    Special Features (via provider_options):
        - continue_session: Continue most recent session (bool)
        - fork_session: Fork session ID (string)
        - allowed_tools: List of allowed tools
        - disallowed_tools: List of disallowed tools
        - mcp_config: Path to MCP configuration file
        - strict_mcp_config: Strict MCP config validation (bool)
        - plugin_dirs: List of plugin directories
        - setting_sources: List like ["user", "project"]
        - max_turns: Maximum conversation turns (int)
        - max_budget_usd: Maximum budget in USD (float)
        - agents: Agent definitions (dict)
        - agent: Agent name to use (string)
        - json_schema: JSON schema for structured output (dict)
        - system_prompt: Override config.system_prompt (string)
        - append_system_prompt: Append to base system prompt (string)

    Image Input:
        Claude CLI supports images via in-prompt references (NOT CLI flags):
        - @path/to/image.png - Reference image file in prompt
        - Ctrl+V paste - Paste from clipboard (interactive mode)
        - Drag-drop - Drop image into terminal (interactive mode)
        Example: "Analyze this image: @./screenshot.png"

    Extended Thinking (verified 2026-01-08):
        Only "ultrathink" triggers extended thinking in Claude CLI v2.0.0+.
        - XHIGH (ultrathink): Allocates up to 31,999 thinking tokens
        - MEDIUM/HIGH: DEPRECATED - "think"/"think hard" are plain text, NOT triggers
    """

    PROVIDER = Provider.CLAUDE
    CLI_NAME = "claude"

    def _build_args(
        self,
        prompt: str,
        session_id: Optional[str] = None,
    ) -> list[str]:
        """Build Claude CLI arguments."""
        args = [self.cli_executable]

        # Apply thinking prefix if reasoning is enabled
        effective_prompt = self._apply_thinking_prefix(prompt)

        # Prompt mode
        args.extend(["-p", effective_prompt])

        # Session resume or continuation
        effective_session = self._get_effective_session_id(session_id)
        if effective_session:
            args.extend(["--resume", effective_session])

        # Provider-specific options
        opts = self.config.provider_options

        # Continue session (when no explicit session_id)
        if not effective_session and self.config.provider_options.get("continue_session"):
            args.append("--continue")

        # Fork session
        if opts.get("fork_session"):
            args.extend(["--fork-session", opts["fork_session"]])

        # Model
        try:
            model = self.config.get_model()
            if model:
                args.extend(["--model", model])
        except Exception:
            pass  # ModelNotSpecifiedError - proceed without model flag

        # Permission mode (approval)
        approval_args = self.config.get_approval_args()
        if "permission_mode" in approval_args:
            args.extend(["--permission-mode", approval_args["permission_mode"]])

        # Output format - always use stream-json for parsing
        args.extend(["--output-format", "stream-json"])
        args.append("--verbose")

        # Setting sources
        if "setting_sources" in opts:
            args.extend(["--setting-sources", ",".join(opts["setting_sources"])])

        # Allowed tools (from config or provider_options)
        # Start with explicitly allowed tools from provider_options
        allowed_tools = list(self.config.provider_options.get("allowed_tools") or [])

        # Add web search tools if web_search is enabled
        if self.config.web_search:
            web_tools = ["WebSearch", "WebFetch"]
            for tool in web_tools:
                if tool not in allowed_tools:
                    allowed_tools.append(tool)

        if allowed_tools:
            args.extend(["--allowed-tools", ",".join(allowed_tools)])

        # Disallowed tools (from config or provider_options)
        disallowed_tools = self.config.provider_options.get("disallowed_tools")
        if disallowed_tools:
            args.extend(["--disallowed-tools", ",".join(disallowed_tools)])

        # MCP config
        if "mcp_config" in opts:
            args.extend(["--mcp-config", opts["mcp_config"]])

        # Strict MCP config
        if opts.get("strict_mcp_config"):
            args.append("--strict-mcp-config")

        # Plugin directories
        if "plugin_dirs" in opts:
            for plugin_dir in opts["plugin_dirs"]:
                args.extend(["--plugin-dir", plugin_dir])

        # Max turns
        if "max_turns" in opts:
            args.extend(["--max-turns", str(opts["max_turns"])])

        # Max budget USD
        if "max_budget_usd" in opts:
            args.extend(["--max-budget-usd", str(opts["max_budget_usd"])])

        # Agent definitions (JSON dict)
        if "agents" in opts:
            agents_json = json.dumps(opts["agents"])
            args.extend(["--agents", agents_json])

        # Agent name (string)
        if "agent" in opts:
            args.extend(["--agent", opts["agent"]])

        # JSON Schema for structured output
        if "json_schema" in opts:
            schema_json = json.dumps(opts["json_schema"])
            args.extend(["--json-schema", schema_json])

        # System prompt (from config or provider_options)
        # provider_options takes precedence over config.system_prompt
        system_prompt = opts.get("system_prompt") or self.config.system_prompt
        if system_prompt:
            args.extend(["--system-prompt", system_prompt])

        # Append system prompt (additive to base system prompt)
        append_system_prompt = opts.get("append_system_prompt")
        if append_system_prompt:
            args.extend(["--append-system-prompt", append_system_prompt])

        return args

    def _apply_thinking_prefix(self, prompt: str) -> str:
        """Apply thinking prefix to prompt based on reasoning level.

        Claude CLI in -p mode only supports "ultrathink" as a thinking trigger:
        - "ultrathink" - maximum thinking budget (31,999 tokens)

        NOTE: "think" and "think hard" do NOT trigger extended thinking in
        Claude CLI v2.0.0+. They are treated as plain text.
        Only XHIGH reasoning level (ultrathink) activates thinking tokens.

        Args:
            prompt: Original prompt

        Returns:
            Prompt with "ultrathink: " prefix if XHIGH reasoning is enabled
        """
        reasoning_args = self.config.get_reasoning_args()
        thinking_prefix = reasoning_args.get("thinking_prefix", "")

        # Only apply ultrathink prefix (the only working trigger)
        if thinking_prefix and not prompt.lower().startswith("ultrathink"):
            return f"{thinking_prefix}{prompt}"

        return prompt

    # run() and stream() are inherited from BaseProvider (Template Method pattern)

    def _parse_output(
        self,
        stdout: str,
        stderr: str,
        success: bool,
        duration_ms: int,
        **kwargs: Any,
    ) -> LLMResult:
        """Parse Claude CLI stream-json output into LLMResult."""
        parser = ClaudeEventParser(self.config, self.provider_name)
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


__all__ = ["ClaudeProvider"]
