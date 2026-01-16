"""Event Parser module for LLM provider output parsing.

Uses Template Method pattern to extract common JSONL parsing logic
while allowing provider-specific customization.
"""

from .base_parser import BaseEventParser, ParseContext
from .claude_parser import ClaudeEventParser
from .codex_parser import CodexEventParser
from .gemini_parser import GeminiEventParser

__all__ = [
    "BaseEventParser",
    "ParseContext",
    "ClaudeEventParser",
    "CodexEventParser",
    "GeminiEventParser",
]
