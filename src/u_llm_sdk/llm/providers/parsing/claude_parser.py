"""Claude Event Parser.

Handles Claude CLI stream-json output format.
"""

from __future__ import annotations

from typing import Any

from u_llm_sdk.types import Provider, TokenUsage

from .base_parser import BaseEventParser, ParseContext


class ClaudeEventParser(BaseEventParser):
    """Parser for Claude CLI stream-json output.

    Claude-specific event types:
    - "system": Contains session_id
    - "assistant": Contains message with content blocks (text, thinking)
    - "tool_use": Contains tool name and input
    - "result": Contains usage stats and structured_output
    """

    PROVIDER = Provider.CLAUDE

    def _process_event(self, data: dict[str, Any], ctx: ParseContext) -> None:
        """Process Claude-specific events."""
        event_type = data.get("type", "")

        # Extract session ID (Claude: in "system" event)
        if event_type == "system":
            self._extract_session_id(data, ctx)

        # Extract assistant message content
        if event_type == "assistant":
            self._extract_assistant_content(data, ctx)

        # Extract tool use
        if event_type == "tool_use":
            self._extract_tool_use(data, ctx, tool_name_keys=("name",))

        # Extract usage stats and structured_output
        if event_type == "result":
            self._extract_result(data, ctx)

    def _extract_assistant_content(
        self, data: dict[str, Any], ctx: ParseContext
    ) -> None:
        """Extract text and thinking from assistant message."""
        message = data.get("message", {})
        content = message.get("content", [])

        for block in content:
            block_type = block.get("type", "")
            if block_type == "text":
                text = block.get("text", "")
                if text:
                    ctx.text_parts.append(text)
            elif block_type == "thinking":
                ctx.thinking_text = block.get("thinking", "")

    def _extract_result(self, data: dict[str, Any], ctx: ParseContext) -> None:
        """Extract usage stats and structured_output from result event."""
        # Token usage
        if "usage" in data:
            usage = data["usage"]
            ctx.token_usage = TokenUsage(
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                cached_tokens=usage.get("cache_read_input_tokens", 0),
                reasoning_tokens=0,
                total_tokens=usage.get("input_tokens", 0)
                + usage.get("output_tokens", 0),
            )

        # Structured output (JSON schema response)
        if "structured_output" in data:
            ctx.structured_output = data["structured_output"]

    def _get_diff_content(self, tool_input: dict[str, Any]) -> str:
        """Claude uses 'new_string' for edit diffs."""
        return tool_input.get("new_string", "")

    def _handle_non_json_line(self, line: str, ctx: ParseContext) -> None:
        """Claude: skip non-JSON lines (strict JSON output)."""
        # Claude stream-json output should always be valid JSON
        # Non-JSON lines are likely errors or noise
        pass


__all__ = ["ClaudeEventParser"]
