"""Shared utilities for Multi-LLM Orchestration.

This module provides common utility functions used across multiple
orchestration components to avoid code duplication.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable


def sanitize_json_string(text: str) -> str:
    """Sanitize a string for JSON parsing.

    Handles common issues in LLM-generated JSON:
    - Control characters in strings
    - Unescaped newlines
    - Tab characters

    Args:
        text: Raw JSON string

    Returns:
        Sanitized JSON string safe for parsing
    """
    # Remove control characters except for valid JSON whitespace
    # JSON allows: space (0x20), tab (0x09), newline (0x0A), carriage return (0x0D)
    # But only outside of strings - inside strings they must be escaped

    # Strategy: Find strings and escape control chars inside them
    result = []
    in_string = False
    escape_next = False
    i = 0

    while i < len(text):
        char = text[i]

        if escape_next:
            result.append(char)
            escape_next = False
            i += 1
            continue

        if char == '\\' and in_string:
            escape_next = True
            result.append(char)
            i += 1
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            result.append(char)
            i += 1
            continue

        if in_string:
            # Inside a string - escape control characters
            if char == '\n':
                result.append('\\n')
            elif char == '\r':
                result.append('\\r')
            elif char == '\t':
                result.append('\\t')
            elif ord(char) < 32:
                # Other control characters - escape as unicode
                result.append(f'\\u{ord(char):04x}')
            else:
                result.append(char)
        else:
            # Outside string - keep as is (JSON parser handles whitespace)
            result.append(char)

        i += 1

    return ''.join(result)


def extract_json(text: str) -> str:
    """Extract JSON from LLM response text.

    Handles various formats:
    - ```json ... ``` code blocks
    - ``` ... ``` generic code blocks
    - Raw JSON objects

    Also sanitizes the JSON to handle:
    - Control characters that break parsing
    - Unescaped newlines in strings

    Args:
        text: Raw LLM response text

    Returns:
        Extracted and sanitized JSON string, or original text if no JSON found
    """
    text = text.strip()

    extracted = None

    # Try ```json block first
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end > start:
            extracted = text[start:end].strip()

    # Try generic ``` block
    if extracted is None and "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end > start:
            content = text[start:end].strip()
            # Skip language identifier if present
            if content.startswith(('python', 'javascript', 'json')):
                first_newline = content.find('\n')
                if first_newline > 0:
                    content = content[first_newline + 1:].strip()
            extracted = content

    # Try to find raw JSON object by brace matching
    if extracted is None and "{" in text:
        start = text.find("{")
        depth = 0
        for i, c in enumerate(text[start:], start):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    extracted = text[start : i + 1]
                    break

    if extracted is None:
        return text

    # Sanitize the extracted JSON to handle control characters
    return sanitize_json_string(extracted)


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM provider implementations.

    This minimal protocol defines the interface that orchestration
    components expect from LLM providers.
    """

    async def run(self, prompt: str, **kwargs) -> "LLMResult":
        """Run a prompt and return the result."""
        ...


# Type alias for forward reference
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from u_llm_sdk.types import LLMResult
