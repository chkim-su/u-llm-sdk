"""Base Event Parser - Template Method pattern for output parsing.

Extracts common parsing logic shared across all providers.
Each provider subclass implements provider-specific hooks.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from u_llm_sdk.types import (
    FILE_EDIT_TOOLS,
    FILE_WRITE_TOOLS,
    SHELL_TOOLS,
    CodeBlock,
    CommandRun,
    FileChange,
    LLMResult,
    Provider,
    ResultType,
    TokenUsage,
)


@dataclass
class ParseContext:
    """Accumulated parsing state during event processing."""

    events: list[dict[str, Any]] = field(default_factory=list)
    text_parts: list[str] = field(default_factory=list)
    files_modified: list[FileChange] = field(default_factory=list)
    commands_run: list[CommandRun] = field(default_factory=list)
    code_blocks: list[CodeBlock] = field(default_factory=list)
    token_usage: Optional[TokenUsage] = None
    session_id: Optional[str] = None

    # Provider-specific extensions (Claude-only)
    thinking_text: Optional[str] = None
    structured_output: Optional[dict[str, Any]] = None


class BaseEventParser(ABC):
    """Abstract base parser using Template Method pattern.

    Common flow:
    1. parse_jsonl() - iterate lines and dispatch
    2. process_event() - handle single event (template method)
    3. build_result() - construct final LLMResult

    Subclasses implement hooks for provider-specific behavior.
    """

    PROVIDER: Provider

    def __init__(self, config: Any, provider_name: str):
        """Initialize parser with config reference.

        Args:
            config: LLMConfig instance for model resolution
            provider_name: Provider name string for result
        """
        self.config = config
        self.provider_name = provider_name

    def parse(
        self,
        stdout: str,
        stderr: str,
        success: bool,
        duration_ms: int,
        session_id_ref: list[Optional[str]],
        **kwargs: Any,
    ) -> LLMResult:
        """Parse CLI output into LLMResult (Template Method).

        Args:
            stdout: Standard output from CLI
            stderr: Standard error from CLI
            success: Whether CLI exited successfully
            duration_ms: Execution duration in milliseconds
            session_id_ref: Mutable reference to update session_id
            **kwargs: Provider-specific arguments

        Returns:
            Parsed LLMResult
        """
        ctx = ParseContext()

        # Parse JSONL output
        self._parse_jsonl(stdout, ctx)

        # Update session_id reference if found
        if ctx.session_id:
            session_id_ref[0] = ctx.session_id

        # Determine result type
        result_type = self._determine_result_type(
            ctx.text_parts, ctx.files_modified, ctx.commands_run, success
        )

        # Build summary
        summary = self._build_summary(
            result_type, ctx.text_parts, ctx.files_modified, ctx.commands_run
        )

        # Build final result
        return self._build_result(
            ctx=ctx,
            success=success,
            result_type=result_type,
            summary=summary,
            duration_ms=duration_ms,
            stderr=stderr,
        )

    def _parse_jsonl(self, stdout: str, ctx: ParseContext) -> None:
        """Parse JSONL output line by line."""
        for line in stdout.strip().split("\n"):
            if not line:
                continue
            try:
                data = json.loads(line)
                ctx.events.append(data)
                self._process_event(data, ctx)
            except json.JSONDecodeError:
                # Non-JSON line - handle as plain text
                self._handle_non_json_line(line, ctx)

    @abstractmethod
    def _process_event(self, data: dict[str, Any], ctx: ParseContext) -> None:
        """Process a single parsed event (provider-specific).

        Subclasses implement to handle their specific event formats.
        """
        pass

    def _handle_non_json_line(self, line: str, ctx: ParseContext) -> None:
        """Handle non-JSON lines (override if needed)."""
        if line.strip():
            ctx.text_parts.append(line.strip())

    # --- Common extraction helpers ---

    def _extract_session_id(self, data: dict[str, Any], ctx: ParseContext) -> bool:
        """Extract session_id if present. Returns True if found."""
        if "session_id" in data:
            ctx.session_id = data["session_id"]
            return True
        return False

    def _extract_tool_use(
        self,
        data: dict[str, Any],
        ctx: ParseContext,
        tool_name_keys: tuple[str, ...] = ("name", "function", "tool"),
        tool_input_keys: tuple[str, ...] = ("input", "arguments", "args"),
    ) -> None:
        """Extract tool use from event data."""
        # Get tool name from various possible keys
        tool_name = ""
        for key in tool_name_keys:
            if key in data:
                tool_name = data[key]
                break

        if not tool_name:
            return

        # Get tool input from various possible keys
        tool_input: dict[str, Any] = {}
        for key in tool_input_keys:
            if key in data:
                raw_input = data[key]
                if isinstance(raw_input, str):
                    try:
                        tool_input = json.loads(raw_input)
                    except json.JSONDecodeError:
                        tool_input = {}
                elif isinstance(raw_input, dict):
                    tool_input = raw_input
                break

        # Handle file operations
        file_edit_tools = FILE_EDIT_TOOLS.get(self.PROVIDER, set())
        file_write_tools = FILE_WRITE_TOOLS.get(self.PROVIDER, set())

        if tool_name in file_edit_tools:
            file_path = self._get_file_path(tool_input)
            if file_path:
                is_write = tool_name in file_write_tools
                action = "created" if is_write else "modified"
                diff = self._get_diff_content(tool_input)
                ctx.files_modified.append(
                    FileChange(path=file_path, action=action, diff=diff)
                )

        # Handle shell commands
        shell_tools = SHELL_TOOLS.get(self.PROVIDER, set())
        if tool_name in shell_tools:
            command = self._get_command(tool_input)
            if command:
                ctx.commands_run.append(
                    CommandRun(command=command, exit_code=0, output="")
                )

    def _get_file_path(self, tool_input: dict[str, Any]) -> str:
        """Extract file path from tool input (provider may override)."""
        return tool_input.get(
            "file_path", tool_input.get("path", tool_input.get("filename", ""))
        )

    def _get_diff_content(self, tool_input: dict[str, Any]) -> str:
        """Extract diff/content from tool input (provider may override)."""
        return tool_input.get(
            "content", tool_input.get("new_content", tool_input.get("new_string", ""))
        )

    def _get_command(self, tool_input: dict[str, Any]) -> str:
        """Extract command from tool input (provider may override)."""
        return tool_input.get(
            "command", tool_input.get("cmd", tool_input.get("script", ""))
        )

    # --- Result building helpers ---

    def _determine_result_type(
        self,
        text_parts: list[str],
        files_modified: list[FileChange],
        commands_run: list[CommandRun],
        success: bool,
    ) -> ResultType:
        """Determine the result type based on outputs."""
        if not success:
            return ResultType.ERROR

        has_text = any(text_parts)
        has_files = bool(files_modified)
        has_commands = bool(commands_run)

        if has_files and has_text:
            return ResultType.MIXED
        if has_files:
            return ResultType.FILE_EDIT
        if has_commands:
            return ResultType.COMMAND
        return ResultType.TEXT

    def _build_summary(
        self,
        result_type: ResultType,
        text_parts: list[str],
        files_modified: list[FileChange],
        commands_run: list[CommandRun],
    ) -> str:
        """Build a short summary string."""
        parts: list[str] = []

        if files_modified:
            file_names = [f.path.split("/")[-1] for f in files_modified]
            parts.append(
                f"{len(files_modified)} files modified: {', '.join(file_names[:3])}"
            )
            if len(file_names) > 3:
                parts[-1] += f" (+{len(file_names) - 3} more)"

        if commands_run:
            parts.append(f"{len(commands_run)} commands executed")

        if text_parts and not parts:
            full_text = " ".join(text_parts)
            if len(full_text) > 100:
                parts.append(full_text[:100] + "...")
            else:
                parts.append(full_text)

        return "; ".join(parts) if parts else "Completed"

    def _build_result(
        self,
        ctx: ParseContext,
        success: bool,
        result_type: ResultType,
        summary: str,
        duration_ms: int,
        stderr: str,
    ) -> LLMResult:
        """Build the final LLMResult (override for provider-specific fields)."""
        effective_model = self.config.get_model(require_explicit=False)

        return LLMResult(
            success=success,
            result_type=result_type,
            provider=self.provider_name,
            model=effective_model,
            text="\n".join(ctx.text_parts),
            summary=summary,
            files_modified=ctx.files_modified,
            commands_run=ctx.commands_run,
            code_blocks=ctx.code_blocks,
            session_id=ctx.session_id,
            duration_ms=duration_ms,
            token_usage=ctx.token_usage,
            thinking=ctx.thinking_text,
            structured_output=ctx.structured_output,
            error=stderr if not success and stderr else None,
            raw=ctx.events,
        )


__all__ = ["BaseEventParser", "ParseContext"]
