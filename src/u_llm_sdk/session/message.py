"""Session message dataclass for all providers.

This module provides the SessionMessage dataclass that supports
both inline strings and file-based prompts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union


@dataclass
class SessionMessage:
    """A message in a session conversation.

    Supports both inline text and file-based prompts. File paths
    are resolved relative to the current working directory.

    Attributes:
        role: The message role ("user" or "assistant")
        content: Either a string or path to a file containing the content

    Example:
        >>> # Inline content
        >>> msg = SessionMessage(role="user", content="Hello!")
        >>>
        >>> # File-based content
        >>> msg = SessionMessage(role="user", content="prompts/greeting.md")
        >>> text = msg.resolve()  # Reads file if exists
    """

    role: Literal["user", "assistant"]
    content: str  # Either inline text or file path

    def resolve(self, base_path: Optional[str] = None) -> str:
        """Resolve the content, reading from file if it's a path.

        Args:
            base_path: Base path for resolving relative file paths

        Returns:
            The resolved content string
        """
        return resolve_prompt(self.content, base_path)

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary format.

        Returns:
            Dict with role and content keys
        """
        return {
            "role": self.role,
            "content": self.resolve(),
        }


def resolve_prompt(
    content: Union[str, Path],
    base_path: Optional[str] = None,
) -> str:
    """Resolve prompt content from string or file.

    If content looks like a file path and the file exists,
    reads and returns the file contents. Otherwise returns
    the content as-is.

    Args:
        content: String content or path to a file
        base_path: Base path for resolving relative file paths

    Returns:
        Resolved prompt string

    Example:
        >>> # Direct string
        >>> text = resolve_prompt("Hello world!")
        >>> print(text)  # "Hello world!"
        >>>
        >>> # File path
        >>> text = resolve_prompt("prompts/system.md")
        >>> print(text)  # Contents of prompts/system.md
    """
    if isinstance(content, Path):
        content = str(content)

    # Check if it looks like a file path
    if _looks_like_file_path(content):
        if base_path:
            full_path = Path(base_path) / content
        else:
            full_path = Path(content)

        if full_path.exists() and full_path.is_file():
            return full_path.read_text(encoding="utf-8")

    return content


def _looks_like_file_path(content: str) -> bool:
    """Check if content looks like a file path.

    Heuristic: contains path separator or common file extensions.

    Args:
        content: String to check

    Returns:
        True if content looks like a file path
    """
    # Common file extensions for prompts
    file_extensions = {".md", ".txt", ".prompt", ".template"}

    # Check for path separators or file extensions
    if "/" in content or "\\" in content:
        return True

    # Check for file extensions
    for ext in file_extensions:
        if content.lower().endswith(ext):
            return True

    return False


__all__ = [
    "SessionMessage",
    "resolve_prompt",
]
