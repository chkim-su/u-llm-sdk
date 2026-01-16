"""Base session management for all providers.

This module provides the abstract base class for provider-specific
session managers and the factory function for getting the appropriate
implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from u_llm_sdk.types import Provider

    from u_llm_sdk.session.message import SessionMessage


class BaseSessionManager(ABC):
    """Abstract base class for file-based session management.

    Each provider stores sessions differently:
    - Claude: ~/.claude/projects/<path-key>/<uuid>.jsonl
    - Codex: ~/.codex/sessions/YYYY/MM/DD/<uuid>.jsonl
    - Gemini: ~/.gemini/tmp/<project-hash>/chats/<uuid>.json

    Implementations handle session file creation and reading in
    provider-specific formats.
    """

    def __init__(self, project_path: str):
        """Initialize session manager for a project.

        Args:
            project_path: Absolute path to the project directory
        """
        self.project_path = project_path

    @abstractmethod
    def create_from_system_prompt(
        self,
        system_prompt: str,
        assistant_acknowledgment: str = "Understood.",
    ) -> str:
        """Create a new session with an injected system prompt.

        Creates a virtual session file with a pre-seeded conversation
        that injects the system prompt as context.

        Args:
            system_prompt: The system context to inject
            assistant_acknowledgment: Assistant's acknowledgment response

        Returns:
            Session ID for resuming with CLI
        """
        pass

    @abstractmethod
    def create_session(self, messages: list["SessionMessage"]) -> str:
        """Create a new session with message history.

        Args:
            messages: List of messages to pre-seed the session

        Returns:
            Session ID for resuming with CLI
        """
        pass

    @abstractmethod
    def read_session(self, session_id: str) -> list[dict[str, Any]]:
        """Read messages from an existing session.

        Args:
            session_id: The session ID to read

        Returns:
            List of message dictionaries from the session

        Raises:
            SessionNotFoundError: If session doesn't exist
        """
        pass

    @abstractmethod
    def list_sessions(self) -> list[str]:
        """List all available session IDs.

        Returns:
            List of session IDs
        """
        pass

    @abstractmethod
    def get_session_path(self, session_id: str) -> str:
        """Get the file path for a session.

        Args:
            session_id: The session ID

        Returns:
            Absolute path to the session file
        """
        pass

    @abstractmethod
    def delete_session(self, session_id: str) -> bool:
        """Delete a session file.

        Args:
            session_id: The session ID to delete

        Returns:
            True if deleted, False if not found
        """
        pass


def get_session_manager(provider: "Provider", project_path: str) -> BaseSessionManager:
    """Factory function to get provider-specific session manager.

    Args:
        provider: The LLM provider
        project_path: Absolute path to the project directory

    Returns:
        Provider-specific session manager instance

    Example:
        >>> from u_llm_sdk.types import Provider
        >>> manager = get_session_manager(Provider.CLAUDE, "/my/project")
        >>> session_id = manager.create_from_system_prompt("You are a security expert.")
    """
    from u_llm_sdk.types import Provider

    from u_llm_sdk.session.claude import ClaudeSessionManager
    from u_llm_sdk.session.codex import CodexSessionManager
    from u_llm_sdk.session.gemini import GeminiSessionManager

    if provider == Provider.CLAUDE:
        return ClaudeSessionManager(project_path)
    elif provider == Provider.CODEX:
        return CodexSessionManager(project_path)
    elif provider == Provider.GEMINI:
        return GeminiSessionManager(project_path)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def inject_system_prompt(
    provider: "Provider",
    prompt: str,
    system_prompt: str,
) -> tuple[str, dict[str, Any]]:
    """Inject system prompt using provider-appropriate method.

    Different providers handle system prompts differently:
    - Claude: Uses native --system-prompt flag
    - Codex/Gemini: Prepend to the user prompt

    Args:
        provider: The LLM provider
        prompt: The user's prompt
        system_prompt: The system prompt to inject

    Returns:
        Tuple of (effective_prompt, extra_config_dict)
        - For Claude: (original_prompt, {"system_prompt": system_prompt})
        - For others: (prepended_prompt, {})

    Example:
        >>> from u_llm_sdk.types import Provider
        >>> prompt, config = inject_system_prompt(
        ...     Provider.CODEX,
        ...     "Sort this list",
        ...     "You are a Python expert.",
        ... )
        >>> print(prompt)
        You are a Python expert.

        Sort this list
    """
    from u_llm_sdk.types import Provider

    if provider == Provider.CLAUDE:
        # Claude uses native --system-prompt flag
        return prompt, {"system_prompt": system_prompt}
    else:
        # Codex/Gemini: prepend to prompt
        effective_prompt = f"{system_prompt}\n\n{prompt}"
        return effective_prompt, {}


__all__ = [
    "BaseSessionManager",
    "get_session_manager",
    "inject_system_prompt",
]
