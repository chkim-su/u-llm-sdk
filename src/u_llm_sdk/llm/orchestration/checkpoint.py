"""Checkpoint System for Auditable Execution.

Provides execution evidence even when LLM produces no text output.
Checkpoint is "what happened", not "what the model said happened".

Key insight: LLMs often produce side effects (file edits, commands)
without text output. Checkpoint guarantees there's always a
human-readable record of what occurred.
"""

from dataclasses import dataclass, field
from typing import Optional

from u_llm_sdk.types import LLMResult


@dataclass
class Checkpoint:
    """Execution checkpoint - audit log for humans.

    Even if result.text is empty, checkpoint provides evidence
    of what happened during execution.

    Attributes:
        success: Whether execution succeeded
        result_type: Type of result (TEXT, FILE_EDIT, COMMAND, etc.)
        summary: Brief summary of what happened
        files_changed: List of file changes (action:path format)
        commands_executed: List of commands that were run
        duration_ms: Execution duration in milliseconds
        error: Error message if failed
        provider: Provider that was used
        model: Model that was used
    """

    success: bool
    result_type: str
    summary: str
    files_changed: list[str] = field(default_factory=list)
    commands_executed: list[str] = field(default_factory=list)
    duration_ms: int = 0
    error: Optional[str] = None
    provider: str = ""
    model: str = ""

    def to_text(self) -> str:
        """Convert to human-readable text.

        This is the key function - it guarantees meaningful output
        even when result.text is empty.
        """
        status = "✓" if self.success else "✗"
        lines = [f"[{status} {self.result_type}] {self.summary}"]

        if self.files_changed:
            lines.append(f"  Files: {', '.join(self.files_changed[:5])}")
            if len(self.files_changed) > 5:
                lines.append(f"    ... and {len(self.files_changed) - 5} more")

        if self.commands_executed:
            lines.append(f"  Commands: {', '.join(self.commands_executed[:3])}")
            if len(self.commands_executed) > 3:
                lines.append(f"    ... and {len(self.commands_executed) - 3} more")

        if self.duration_ms:
            lines.append(f"  Duration: {self.duration_ms}ms")

        if self.error:
            lines.append(f"  Error: {self.error}")

        if self.provider and self.model:
            lines.append(f"  Provider: {self.provider}/{self.model}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "result_type": self.result_type,
            "summary": self.summary,
            "files_changed": self.files_changed,
            "commands_executed": self.commands_executed,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "provider": self.provider,
            "model": self.model,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Checkpoint":
        """Create Checkpoint from dictionary."""
        return cls(
            success=data["success"],
            result_type=data["result_type"],
            summary=data["summary"],
            files_changed=data.get("files_changed", []),
            commands_executed=data.get("commands_executed", []),
            duration_ms=data.get("duration_ms", 0),
            error=data.get("error"),
            provider=data.get("provider", ""),
            model=data.get("model", ""),
        )

    @classmethod
    def from_result(cls, result: LLMResult) -> "Checkpoint":
        """Create Checkpoint from LLMResult.

        Extracts all relevant information from LLMResult
        to create a comprehensive checkpoint.
        """
        # Format file changes as "action:path"
        files_changed = []
        for fc in result.files_modified:
            files_changed.append(f"{fc.action}:{fc.path}")

        # Format commands
        commands = []
        for cmd in result.commands_run:
            status = "ok" if cmd.exit_code == 0 else f"exit={cmd.exit_code}"
            commands.append(f"{cmd.command} ({status})")

        # Generate summary if not present
        summary = result.summary
        if not summary:
            if result.success:
                if files_changed:
                    summary = f"Modified {len(files_changed)} file(s)"
                elif commands:
                    summary = f"Executed {len(commands)} command(s)"
                elif result.text:
                    summary = (
                        result.text[:100] + "..."
                        if len(result.text) > 100
                        else result.text
                    )
                else:
                    summary = "Completed successfully"
            else:
                summary = result.error or "Failed"

        return cls(
            success=result.success,
            result_type=result.result_type.value,
            summary=summary,
            files_changed=files_changed,
            commands_executed=commands,
            duration_ms=result.duration_ms,
            error=result.error,
            provider=result.provider,
            model=result.model,
        )


def preferred_text(result: LLMResult) -> str:
    """Get meaningful text output from LLMResult.

    ALWAYS returns something meaningful:
    1. If result.text exists and is non-empty, return it
    2. Otherwise, return checkpoint.to_text()

    This is the key function for the "작업 확인자" requirement:
    - Users always get feedback about what happened
    - No silent failures or empty responses

    Args:
        result: LLMResult from any provider

    Returns:
        Human-readable text describing result or actions taken

    Example:
        >>> result = await quick_run("Fix the bug")
        >>> print(preferred_text(result))
        # If text response: prints the text
        # If file edit only: "[✓ file_edit] Modified 2 file(s)..."
    """
    # If there's meaningful text, return it
    if result.text and result.text.strip():
        return result.text

    # Otherwise, create checkpoint and return its text
    checkpoint = Checkpoint.from_result(result)
    return checkpoint.to_text()


def create_checkpoint(result: LLMResult) -> Checkpoint:
    """Create Checkpoint from LLMResult.

    Convenience function that wraps Checkpoint.from_result().

    Args:
        result: LLMResult to create checkpoint from

    Returns:
        Checkpoint instance
    """
    return Checkpoint.from_result(result)


def attach_checkpoint(result: LLMResult) -> LLMResult:
    """Attach checkpoint to result.raw.

    Modifies result in-place to include checkpoint data
    in result.raw["checkpoint"].

    Args:
        result: LLMResult to attach checkpoint to

    Returns:
        Same LLMResult with checkpoint attached
    """
    checkpoint = Checkpoint.from_result(result)

    if result.raw is None:
        result.raw = {}

    result.raw["checkpoint"] = checkpoint.to_dict()

    return result


__all__ = [
    "Checkpoint",
    "preferred_text",
    "create_checkpoint",
    "attach_checkpoint",
]
