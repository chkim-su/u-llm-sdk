"""Gemini Event Parser.

Handles Gemini CLI stream-json output format.
"""

from __future__ import annotations

from typing import Any

from u_llm_sdk.types import Provider, TokenUsage

from .base_parser import BaseEventParser, ParseContext


class GeminiEventParser(BaseEventParser):
    """Parser for Gemini CLI stream-json output.

    Gemini-specific event types:
    - "assistant" / "message" / "text": Contains text content
    - "tool_use" / "function_call" / "tool_call": Contains tool invocations
    - "result" / "done" / "usage": Contains usage stats

    Gemini uses "stats" field (not "usage") for token counts.
    """

    PROVIDER = Provider.GEMINI

    def _process_event(self, data: dict[str, Any], ctx: ParseContext) -> None:
        """Process Gemini-specific events."""
        event_type = data.get("type", "")

        # Extract session ID (Gemini: direct field)
        self._extract_session_id(data, ctx)

        # Extract text content
        if event_type in ("assistant", "message", "text"):
            self._extract_text_content(data, ctx)

        # Extract tool use
        if event_type in ("tool_use", "function_call", "tool_call"):
            self._extract_tool_use(
                data,
                ctx,
                tool_name_keys=("name", "function", "tool"),
                tool_input_keys=("input", "arguments", "args"),
            )

        # Extract usage stats
        if event_type in ("result", "done", "usage"):
            self._extract_usage(data, ctx)

    def _extract_text_content(self, data: dict[str, Any], ctx: ParseContext) -> None:
        """Extract text from message events (Gemini format variations)."""
        message = data.get("message", data)
        content = message.get("content", message.get("text", ""))

        if isinstance(content, str):
            if content:
                ctx.text_parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    if text:
                        ctx.text_parts.append(text)

    def _extract_usage(self, data: dict[str, Any], ctx: ParseContext) -> None:
        """Extract token usage from result/done/usage events.

        Gemini CLI uses "stats" field, not "usage".
        Also handles multiple field name variations.
        """
        # Try both "stats" (Gemini CLI) and "usage" (other formats)
        usage = data.get("stats", data.get("usage", data))
        if not usage or not isinstance(usage, dict):
            return

        # Handle different field names across formats
        input_tokens = usage.get(
            "input_tokens", usage.get("prompt_tokens", usage.get("promptTokenCount", 0))
        )
        output_tokens = usage.get(
            "output_tokens",
            usage.get("completion_tokens", usage.get("candidatesTokenCount", 0)),
        )
        total_tokens = usage.get("total_tokens", input_tokens + output_tokens)
        cached_tokens = usage.get(
            "cached_tokens", usage.get("cached", usage.get("cachedContentTokenCount", 0))
        )

        ctx.token_usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            reasoning_tokens=0,  # Gemini doesn't expose reasoning tokens
            total_tokens=total_tokens,
        )

    def _get_diff_content(self, tool_input: dict[str, Any]) -> str:
        """Gemini uses 'content' or 'new_content' for file content."""
        return tool_input.get("content", tool_input.get("new_content", ""))


__all__ = ["GeminiEventParser"]
