"""Codex Event Parser.

Handles Codex CLI JSON (JSONL) output format.
"""

from __future__ import annotations

from typing import Any, Optional

from u_llm_sdk.types import LLMResult, Provider, ResultType, TokenUsage

from .base_parser import BaseEventParser, ParseContext


class CodexEventParser(BaseEventParser):
    """Parser for Codex CLI JSONL output.

    Codex-specific event types:
    - "message" / "assistant": Contains text content
    - "item.completed": Codex CLI 0.71.0+ format for agent messages
    - "tool_use" / "function_call": Contains tool invocations
    - "result" / "done": Contains usage stats

    Special handling:
    - Resume mode returns plain text (not JSONL)
    """

    PROVIDER = Provider.CODEX

    def parse(
        self,
        stdout: str,
        stderr: str,
        success: bool,
        duration_ms: int,
        session_id_ref: list[Optional[str]],
        **kwargs: Any,
    ) -> LLMResult:
        """Parse Codex output with special handling for resume mode."""
        # Resume mode returns plain text
        is_resume = kwargs.get("session_id") is not None
        if is_resume:
            return self._build_resume_result(stdout, stderr, success, duration_ms)

        # Normal JSONL parsing
        return super().parse(
            stdout, stderr, success, duration_ms, session_id_ref, **kwargs
        )

    def _build_resume_result(
        self, stdout: str, stderr: str, success: bool, duration_ms: int
    ) -> LLMResult:
        """Build result for resume mode (plain text output)."""
        effective_model = self.config.get_model(require_explicit=False)
        return LLMResult(
            success=success,
            result_type=ResultType.TEXT,
            provider=self.provider_name,
            model=effective_model,
            text=stdout.strip(),
            summary="Resumed session",
            duration_ms=duration_ms,
            error=stderr if not success and stderr else None,
        )

    def _process_event(self, data: dict[str, Any], ctx: ParseContext) -> None:
        """Process Codex-specific events."""
        event_type = data.get("type", "")

        # Extract session ID (Codex: direct field)
        self._extract_session_id(data, ctx)

        # Extract text content
        if event_type in ("message", "assistant"):
            self._extract_text_content(data, ctx)

        # Handle item.completed events (Codex CLI 0.71.0+ format)
        if event_type == "item.completed":
            self._extract_item_completed(data, ctx)

        # Extract tool use
        if event_type in ("tool_use", "function_call"):
            self._extract_tool_use(
                data, ctx, tool_name_keys=("name", "function"), tool_input_keys=("input", "arguments")
            )

        # Extract usage stats
        if event_type in ("result", "done"):
            self._extract_usage(data, ctx)

    def _extract_text_content(self, data: dict[str, Any], ctx: ParseContext) -> None:
        """Extract text from message/assistant events."""
        content = data.get("content", data.get("text", ""))

        if isinstance(content, str):
            if content:
                ctx.text_parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    if text:
                        ctx.text_parts.append(text)

    def _extract_item_completed(self, data: dict[str, Any], ctx: ParseContext) -> None:
        """Extract text from item.completed events (Codex 0.71.0+)."""
        item = data.get("item", {})
        if item.get("type") == "agent_message":
            text = item.get("text", "")
            if text:
                ctx.text_parts.append(text)

    def _extract_usage(self, data: dict[str, Any], ctx: ParseContext) -> None:
        """Extract token usage from result/done events."""
        usage = data.get("usage", {})
        if not usage:
            return

        input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
        output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
        total_tokens = usage.get("total_tokens", 0)

        # Calculate total if not provided
        if total_tokens == 0:
            total_tokens = input_tokens + output_tokens

        ctx.token_usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=usage.get("cached_tokens", 0),
            reasoning_tokens=usage.get("reasoning_tokens", 0),
            total_tokens=total_tokens,
        )

    def _get_diff_content(self, tool_input: dict[str, Any]) -> str:
        """Codex uses 'content' for file content."""
        return tool_input.get("content", "")


__all__ = ["CodexEventParser"]
