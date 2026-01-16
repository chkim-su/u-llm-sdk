"""Unified JSON output parsers for all LLM providers.

This module provides unified parsing of JSON output from Claude, Gemini, and Codex CLIs.
All parsers return a UnifiedResponse dataclass with normalized fields across providers.

Supported Formats:
    - Claude: --output-format stream-json --verbose
    - Gemini: -o json (single JSON output)
    - Codex: --json (JSONL events)
    - Claude: --output-format stream-json (stream events)

Usage:
    >>> # Parse Claude single JSON output
    >>> data = json.loads(claude_json_output)
    >>> response = parse_claude_json(data)
    >>> print(f"Result: {response.result_text}")
    >>> print(f"Tokens: {response.input_tokens} + {response.output_tokens}")

    >>> # Parse Gemini JSON output
    >>> data = json.loads(gemini_json_output)
    >>> response = parse_gemini_json(data)

    >>> # Parse Codex JSONL events
    >>> lines = [json.loads(line) for line in codex_jsonl_output.splitlines()]
    >>> response = parse_codex_jsonl(lines)

    >>> # Parse Claude stream events
    >>> lines = [json.loads(line) for line in stream_output.splitlines()]
    >>> response = parse_claude_stream_json(lines)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class UnifiedResponse:
    """Unified response structure for all LLM providers.

    This dataclass normalizes the output from Claude, Gemini, and Codex CLIs
    into a consistent schema for downstream processing.

    Attributes:
        result_text: The main text response from the LLM
        session_id: Session/thread identifier for conversation continuity
        success: Whether the execution was successful
        duration_ms: Execution duration in milliseconds
        input_tokens: Number of input tokens consumed
        output_tokens: Number of output tokens generated
        total_tokens: Total tokens (input + output)
        cached_tokens: Number of cached/reused input tokens (default: 0)
        cost_usd: Estimated cost in USD (None if not available)
        structured_output: Optional structured data (tool calls, etc.)
        raw_response: The raw parsed JSON response for debugging
    """

    result_text: str
    session_id: str
    success: bool
    duration_ms: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_tokens: int = 0
    cost_usd: Optional[float] = None
    structured_output: Optional[dict[str, Any]] = None
    raw_response: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation of the response
        """
        return {
            "result_text": self.result_text,
            "session_id": self.session_id,
            "success": self.success,
            "duration_ms": self.duration_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cached_tokens,
            "cost_usd": self.cost_usd,
            "structured_output": self.structured_output,
            "raw_response": self.raw_response,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UnifiedResponse:
        """Create UnifiedResponse from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            UnifiedResponse instance
        """
        return cls(
            result_text=data["result_text"],
            session_id=data["session_id"],
            success=data["success"],
            duration_ms=data["duration_ms"],
            input_tokens=data["input_tokens"],
            output_tokens=data["output_tokens"],
            total_tokens=data["total_tokens"],
            cached_tokens=data.get("cached_tokens", 0),
            cost_usd=data.get("cost_usd"),
            structured_output=data.get("structured_output"),
            raw_response=data.get("raw_response", {}),
        )


def parse_claude_json(data: dict[str, Any]) -> UnifiedResponse:
    """Parse Claude CLI JSON output (result event).

    Claude CLI with --output-format stream-json outputs events including
    a final "result" event with execution summary and token usage.

    Expected structure:
        {
            "type": "result",
            "subtype": "success" | "error",
            "result": "text result",
            "session_id": "...",
            "is_error": false,
            "duration_ms": 3648,
            "usage": {
                "input_tokens": 123,
                "output_tokens": 456,
                "cache_read_input_tokens": 789
            },
            "total_cost_usd": 0.089716,
            "structured_output": {...}  // Optional
        }

    Args:
        data: Parsed JSON dict from Claude result event

    Returns:
        UnifiedResponse with normalized fields
    """
    # Extract basic fields
    result_text = data.get("result", "")
    session_id = data.get("session_id", "")
    is_error = data.get("is_error", False)
    success = not is_error
    duration_ms = data.get("duration_ms", 0)

    # Extract token usage
    usage = data.get("usage", {})
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    cached_tokens = usage.get("cache_read_input_tokens", 0)
    total_tokens = input_tokens + output_tokens

    # Extract cost
    cost_usd = data.get("total_cost_usd")

    # Extract structured output (e.g., tool calls)
    structured_output = data.get("structured_output")

    return UnifiedResponse(
        result_text=result_text,
        session_id=session_id,
        success=success,
        duration_ms=duration_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_tokens=cached_tokens,
        cost_usd=cost_usd,
        structured_output=structured_output,
        raw_response=data,
    )


def parse_gemini_json(data: dict[str, Any]) -> UnifiedResponse:
    """Parse Gemini CLI JSON output (single JSON format).

    Gemini CLI with -o json outputs a single JSON object with response and stats.

    Expected structure:
        {
            "response": "Four.",
            "stats": {
                "models": {
                    "gemini-2.5-flash-lite": {
                        "tokens": {
                            "prompt": 3535,
                            "candidates": 62,
                            "total": 3753,
                            "cached": 0
                        }
                    }
                }
            }
        }

    Note: Gemini JSON format does not include session_id or duration_ms.
          These must be tracked externally or extracted from stream events.

    Args:
        data: Parsed JSON dict from Gemini output

    Returns:
        UnifiedResponse with normalized fields
    """
    # Extract response text
    result_text = data.get("response", "")

    # Session ID not available in single JSON format
    session_id = data.get("session_id", "")

    # Success assumed if no error field
    success = not data.get("error", False)

    # Duration not available in single JSON format
    duration_ms = 0

    # Extract token usage from stats.models
    stats = data.get("stats", {})
    models = stats.get("models", {})

    input_tokens = 0
    output_tokens = 0
    cached_tokens = 0
    total_tokens = 0

    # Aggregate tokens from all models
    for model_name, model_stats in models.items():
        tokens = model_stats.get("tokens", {})
        input_tokens += tokens.get("prompt", 0)
        output_tokens += tokens.get("candidates", 0)
        cached_tokens += tokens.get("cached", 0)
        total_tokens += tokens.get("total", 0)

    return UnifiedResponse(
        result_text=result_text,
        session_id=session_id,
        success=success,
        duration_ms=duration_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_tokens=cached_tokens,
        cost_usd=None,  # Gemini doesn't provide cost in CLI output
        structured_output=None,
        raw_response=data,
    )


def parse_codex_jsonl(lines: list[dict[str, Any]]) -> UnifiedResponse:
    """Parse Codex CLI JSONL events.

    Codex CLI with --json outputs newline-delimited JSON events.

    Expected events:
        {"type": "thread.started", "thread_id": "..."}
        {"type": "turn.started"}
        {"type": "item.completed", "item": {"type": "agent_message", "text": "..."}}
        {"type": "turn.completed", "usage": {"input_tokens": 123, "output_tokens": 456}}

    Args:
        lines: List of parsed JSON event dicts

    Returns:
        UnifiedResponse with normalized fields
    """
    result_text = ""
    session_id = ""
    success = True
    duration_ms = 0
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0

    for event in lines:
        event_type = event.get("type", "")

        # Extract thread ID (session ID)
        if event_type == "thread.started":
            session_id = event.get("thread_id", "")

        # Extract text from agent messages
        elif event_type == "item.completed":
            item = event.get("item", {})
            item_type = item.get("type", "")
            if item_type == "agent_message":
                text = item.get("text", "")
                if text:
                    if result_text:
                        result_text += "\n"
                    result_text += text

        # Extract token usage from turn completion
        elif event_type == "turn.completed":
            usage = event.get("usage", {})
            turn_input = usage.get("input_tokens", 0)
            turn_output = usage.get("output_tokens", 0)
            turn_total = usage.get("total_tokens", 0)

            input_tokens += turn_input
            output_tokens += turn_output

            # If total_tokens provided, use it; otherwise compute it
            if turn_total > 0:
                total_tokens += turn_total
            else:
                total_tokens += turn_input + turn_output

        # Check for errors
        elif event_type == "error":
            success = False
            error_message = event.get("message", "Unknown error")
            if result_text:
                result_text += "\n"
            result_text += f"Error: {error_message}"

    return UnifiedResponse(
        result_text=result_text,
        session_id=session_id,
        success=success,
        duration_ms=duration_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_tokens=0,  # Codex doesn't provide cache info in CLI output
        cost_usd=None,  # Codex doesn't provide cost in CLI output
        structured_output=None,
        raw_response={"events": lines},
    )


def parse_claude_stream_json(lines: list[dict[str, Any]]) -> UnifiedResponse:
    """Parse Claude CLI stream JSON events.

    Claude CLI with --output-format stream-json outputs multiple JSON events:
    - system (subtype=init): Session initialization
    - assistant: LLM responses with content and usage
    - tool_use: Tool execution events
    - result: Final execution summary

    This parser aggregates all events into a unified response.

    Expected events:
        {"type": "system", "subtype": "init", "session_id": "...", "model": "..."}
        {"type": "assistant", "message": {"content": [...], "usage": {...}}}
        {"type": "tool_use", "name": "Edit", "input": {...}}
        {"type": "result", "result": "...", "duration_ms": 3648, "usage": {...}}

    Args:
        lines: List of parsed JSON event dicts

    Returns:
        UnifiedResponse with normalized fields
    """
    result_text = ""
    session_id = ""
    success = True
    duration_ms = 0
    input_tokens = 0
    output_tokens = 0
    cached_tokens = 0
    total_tokens = 0
    cost_usd = None
    structured_output = None
    tool_uses = []

    for event in lines:
        event_type = event.get("type", "")

        # Extract session ID from system init
        if event_type == "system":
            subtype = event.get("subtype", "")
            if subtype == "init":
                session_id = event.get("session_id", "")

        # Extract text and usage from assistant messages
        elif event_type == "assistant":
            message = event.get("message", {})
            content = message.get("content", [])

            # Aggregate text content
            for content_block in content:
                if content_block.get("type") == "text":
                    text = content_block.get("text", "")
                    if text:
                        if result_text:
                            result_text += "\n"
                        result_text += text

            # Aggregate usage
            usage = message.get("usage", {})
            input_tokens += usage.get("input_tokens", 0)
            output_tokens += usage.get("output_tokens", 0)
            cached_tokens += usage.get("cache_read_input_tokens", 0)

        # Collect tool uses
        elif event_type == "tool_use":
            tool_uses.append({
                "name": event.get("name", ""),
                "input": event.get("input", {}),
            })

        # Extract final result
        elif event_type == "result":
            # Use result text if available (overrides aggregated text)
            if "result" in event:
                result_text = event.get("result", result_text)

            # Update success status
            is_error = event.get("is_error", False)
            success = not is_error

            # Update duration
            duration_ms = event.get("duration_ms", duration_ms)

            # Update usage (final usage supersedes aggregated)
            usage = event.get("usage", {})
            if usage:
                input_tokens = usage.get("input_tokens", input_tokens)
                output_tokens = usage.get("output_tokens", output_tokens)
                cached_tokens = usage.get("cache_read_input_tokens", cached_tokens)

            # Update cost
            cost_usd = event.get("total_cost_usd", cost_usd)

            # Update structured output
            structured_output = event.get("structured_output", structured_output)

    # Compute total tokens
    total_tokens = input_tokens + output_tokens

    # Add tool uses to structured output if any
    if tool_uses and not structured_output:
        structured_output = {"tool_uses": tool_uses}
    elif tool_uses and structured_output:
        structured_output["tool_uses"] = tool_uses

    return UnifiedResponse(
        result_text=result_text,
        session_id=session_id,
        success=success,
        duration_ms=duration_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_tokens=cached_tokens,
        cost_usd=cost_usd,
        structured_output=structured_output,
        raw_response={"events": lines},
    )


__all__ = [
    "UnifiedResponse",
    "parse_claude_json",
    "parse_gemini_json",
    "parse_codex_jsonl",
    "parse_claude_stream_json",
]
