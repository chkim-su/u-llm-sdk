"""LLM Types - Data Models.

This module defines the unified output schema for all LLM providers.
All providers return the same LLMResult structure for consistent handling.

For LLM Users:
    1. Check result.success first
    2. Check result.result_type to understand what kind of response this is
    3. Check result.summary for a quick overview (always exists)
    4. Check result.text for text response (may be empty for file edits!)
    5. Check result.files_modified, result.commands_run for action details

Serialization:
    All dataclasses provide to_dict() and from_dict() methods for JSON
    serialization/deserialization. This is required for HTTP communication
    between U-llm-sdk and MV-rag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any


class ResultType(Enum):
    """Result type - LLM can determine response type by checking this field.

    Attributes:
        TEXT: Text response (explanation, analysis, etc.)
        CODE: Code generation/modification
        FILE_EDIT: Files were modified (text may be empty!)
        COMMAND: Commands were executed
        ERROR: Error occurred
        MIXED: Multiple types (text + file edits, etc.)
    """
    TEXT = "text"
    CODE = "code"
    FILE_EDIT = "file_edit"
    COMMAND = "command"
    ERROR = "error"
    MIXED = "mixed"


@dataclass
class FileChange:
    """File change information.

    Attributes:
        path: File path (relative or absolute)
        action: Action type - "created", "modified", "deleted"
        diff: Diff content if available
    """
    path: str
    action: str  # "created", "modified", "deleted"
    diff: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "path": self.path,
            "action": self.action,
            "diff": self.diff,
        }

    @classmethod
    def from_dict(cls, data: dict) -> FileChange:
        """Create from dictionary (JSON deserialization)."""
        return cls(
            path=data["path"],
            action=data["action"],
            diff=data.get("diff"),
        )


@dataclass
class CommandRun:
    """Command execution information.

    Attributes:
        command: Command that was executed
        exit_code: Exit code (0 = success)
        output: Command output (stdout/stderr)
    """
    command: str
    exit_code: int
    output: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "command": self.command,
            "exit_code": self.exit_code,
            "output": self.output,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CommandRun:
        """Create from dictionary (JSON deserialization)."""
        return cls(
            command=data["command"],
            exit_code=data["exit_code"],
            output=data.get("output", ""),
        )


@dataclass
class CodeBlock:
    """Generated code block.

    Attributes:
        language: Programming language
        code: Code content
        filename: Target filename if specified
    """
    language: str
    code: str
    filename: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "language": self.language,
            "code": self.code,
            "filename": self.filename,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CodeBlock:
        """Create from dictionary (JSON deserialization)."""
        return cls(
            language=data["language"],
            code=data["code"],
            filename=data.get("filename"),
        )


@dataclass
class TokenUsage:
    """Token usage information.

    Attributes:
        input_tokens: Input tokens consumed
        output_tokens: Output tokens generated
        cached_tokens: Cached tokens used (if applicable)
        reasoning_tokens: Reasoning tokens (for models like o1/o3)
        total_tokens: Total tokens
    """
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_tokens": self.cached_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TokenUsage:
        """Create from dictionary (JSON deserialization)."""
        return cls(
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            cached_tokens=data.get("cached_tokens", 0),
            reasoning_tokens=data.get("reasoning_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
        )


@dataclass
class LLMResult:
    """Unified result schema for all providers.

    This is the main result class returned by all providers.
    The schema is identical regardless of which provider (Claude/Codex/Gemini) was used.

    For LLM Users - Field Reference:
        - success: Check this first! True if operation succeeded
        - result_type: What kind of result (TEXT, FILE_EDIT, COMMAND, etc.)
        - summary: Quick summary (always exists, even for file edits)
        - text: Text response (may be empty for file-only operations!)
        - files_modified: List of files that were changed
        - commands_run: List of commands that were executed
        - structured_output: Structured JSON output (when using --json-schema)
        - error: Error message if success=False

    Attributes:
        success: Whether the operation succeeded
        result_type: Type of result (TEXT, FILE_EDIT, COMMAND, ERROR, MIXED)
        provider: Provider name (claude, codex, gemini)
        model: Model name used
        text: Text response (empty string if no text)
        summary: Brief summary (always present)
        files_modified: List of file changes
        commands_run: List of executed commands
        code_blocks: List of generated code blocks
        session_id: Session ID for continuation
        duration_ms: Execution duration in milliseconds
        token_usage: Token usage statistics
        error: Error message if failed
        thinking: Claude extended thinking content (when enabled)
        structured_output: Structured JSON output (when using --json-schema)
        raw: Raw response for debugging
    """
    # === Required fields (always present) ===
    success: bool
    result_type: ResultType
    provider: str
    model: str

    # === Response content ===
    text: str = ""
    summary: str = ""

    # === Action details (when applicable) ===
    files_modified: list[FileChange] = field(default_factory=list)
    commands_run: list[CommandRun] = field(default_factory=list)
    code_blocks: list[CodeBlock] = field(default_factory=list)

    # === Metadata ===
    session_id: Optional[str] = None
    duration_ms: int = 0
    token_usage: Optional[TokenUsage] = None
    error: Optional[str] = None

    # === Thinking/Reasoning (when enabled) ===
    thinking: Optional[str] = None  # Claude extended thinking content

    # === Structured Output (when using --json-schema) ===
    structured_output: Optional[dict[str, Any]] = None  # Structured JSON output

    # === Raw response (for debugging) ===
    raw: Optional[Any] = None

    @property
    def has_text(self) -> bool:
        """Check if result has text content."""
        return bool(self.text)

    @property
    def has_file_changes(self) -> bool:
        """Check if result has file changes."""
        return bool(self.files_modified)

    @property
    def has_commands(self) -> bool:
        """Check if result has command executions."""
        return bool(self.commands_run)

    def __str__(self) -> str:
        """Human-readable string representation."""
        if self.success:
            return f"LLMResult({self.result_type.value}, {self.provider}/{self.model}): {self.summary}"
        else:
            return f"LLMResult(ERROR, {self.provider}/{self.model}): {self.error}"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization.

        Handles nested objects (FileChange, CommandRun, CodeBlock, TokenUsage)
        and enums (ResultType) properly.
        """
        return {
            "success": self.success,
            "result_type": self.result_type.value,
            "provider": self.provider,
            "model": self.model,
            "text": self.text,
            "summary": self.summary,
            "files_modified": [f.to_dict() for f in self.files_modified],
            "commands_run": [c.to_dict() for c in self.commands_run],
            "code_blocks": [b.to_dict() for b in self.code_blocks],
            "session_id": self.session_id,
            "duration_ms": self.duration_ms,
            "token_usage": self.token_usage.to_dict() if self.token_usage else None,
            "error": self.error,
            "thinking": self.thinking,
            "structured_output": self.structured_output,
            # Note: raw is excluded from serialization (debugging only)
        }

    @classmethod
    def from_dict(cls, data: dict) -> LLMResult:
        """Create from dictionary (JSON deserialization).

        Handles nested objects and enums properly.
        """
        # Parse result_type enum
        result_type_value = data.get("result_type", "text")
        if isinstance(result_type_value, ResultType):
            result_type = result_type_value
        else:
            result_type = ResultType(result_type_value)

        # Parse nested objects
        files_modified = [
            FileChange.from_dict(f) for f in data.get("files_modified", [])
        ]
        commands_run = [
            CommandRun.from_dict(c) for c in data.get("commands_run", [])
        ]
        code_blocks = [
            CodeBlock.from_dict(b) for b in data.get("code_blocks", [])
        ]
        token_usage_data = data.get("token_usage")
        token_usage = (
            TokenUsage.from_dict(token_usage_data)
            if token_usage_data
            else None
        )

        return cls(
            success=data["success"],
            result_type=result_type,
            provider=data["provider"],
            model=data["model"],
            text=data.get("text", ""),
            summary=data.get("summary", ""),
            files_modified=files_modified,
            commands_run=commands_run,
            code_blocks=code_blocks,
            session_id=data.get("session_id"),
            duration_ms=data.get("duration_ms", 0),
            token_usage=token_usage,
            error=data.get("error"),
            thinking=data.get("thinking"),
            structured_output=data.get("structured_output"),
            raw=data.get("raw"),
        )


# Type aliases for convenience
FileChanges = list[FileChange]
CommandRuns = list[CommandRun]
CodeBlocks = list[CodeBlock]
