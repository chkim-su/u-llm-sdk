"""Claude-specific session management.

Claude stores sessions in ~/.claude/projects/<path-key>/ as JSONL files.
The path key is derived from the project path using a specific encoding.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from u_llm_sdk.session.base import BaseSessionManager

if TYPE_CHECKING:
    from u_llm_sdk.session.message import SessionMessage


class ClaudeSessionManager(BaseSessionManager):
    """Session manager for Claude CLI.

    Session format: ~/.claude/projects/<path-key>/<uuid>.jsonl

    The path key is derived from the project path by:
    1. Replacing path separators with '-'
    2. Encoding special characters

    Example:
        >>> manager = ClaudeSessionManager("/home/user/myproject")
        >>> session_id = manager.create_from_system_prompt("You are a security expert.")
        >>> # Resume with: claude --resume <session_id>
    """

    def __init__(self, project_path: str):
        """Initialize Claude session manager.

        Args:
            project_path: Absolute path to the project directory
        """
        super().__init__(project_path)
        self._base_dir = Path.home() / ".claude" / "projects"
        self._project_dir = self._base_dir / self._path_to_key(project_path)

    def _path_to_key(self, path: str) -> str:
        """Convert a path to Claude's project key format.

        Claude uses a specific encoding for project paths in directory names.

        Args:
            path: Absolute file path

        Returns:
            Encoded path key for directory name
        """
        # Normalize path
        normalized = os.path.normpath(path)

        # Remove leading slash and replace separators
        if normalized.startswith("/"):
            normalized = normalized[1:]

        # Replace path separators with dash
        key = normalized.replace("/", "-").replace("\\", "-")

        # Encode special characters (simple version)
        key = re.sub(r"[^a-zA-Z0-9\-_]", "_", key)

        return key

    def create_from_system_prompt(
        self,
        system_prompt: str,
        assistant_acknowledgment: str = "Understood.",
    ) -> str:
        """Create a session with injected system context.

        Creates a JSONL file with a virtual conversation that
        seeds the context.

        Args:
            system_prompt: System context to inject
            assistant_acknowledgment: Assistant's acknowledgment

        Returns:
            Session ID (UUID)
        """
        from u_llm_sdk.session.message import SessionMessage

        messages = [
            SessionMessage(role="user", content=system_prompt),
            SessionMessage(role="assistant", content=assistant_acknowledgment),
        ]
        return self.create_session(messages)

    def create_session(self, messages: list["SessionMessage"]) -> str:
        """Create a session with message history.

        Args:
            messages: List of messages to pre-seed

        Returns:
            Session ID (UUID)
        """
        session_id = str(uuid.uuid4())

        # Ensure directory exists
        self._project_dir.mkdir(parents=True, exist_ok=True)

        session_path = self._project_dir / f"{session_id}.jsonl"

        # Write messages as JSONL
        with session_path.open("w", encoding="utf-8") as f:
            for msg in messages:
                entry = {
                    "type": msg.role,
                    "message": {
                        "role": msg.role,
                        "content": msg.resolve(),
                    },
                }
                f.write(json.dumps(entry) + "\n")

        return session_id

    def read_session(self, session_id: str) -> list[dict[str, Any]]:
        """Read messages from a session.

        Args:
            session_id: The session ID

        Returns:
            List of message dictionaries

        Raises:
            SessionNotFoundError: If session doesn't exist
        """
        session_path = self._project_dir / f"{session_id}.jsonl"

        if not session_path.exists():
            from u_llm_sdk.types import SessionNotFoundError

            raise SessionNotFoundError(f"Session not found: {session_id}")

        messages = []
        with session_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    if "message" in entry:
                        messages.append(entry["message"])

        return messages

    def list_sessions(self) -> list[str]:
        """List all available session IDs.

        Returns:
            List of session IDs
        """
        if not self._project_dir.exists():
            return []

        sessions = []
        for file in self._project_dir.glob("*.jsonl"):
            sessions.append(file.stem)

        return sorted(sessions)

    def get_session_path(self, session_id: str) -> str:
        """Get the file path for a session.

        Args:
            session_id: The session ID

        Returns:
            Absolute path to the session file
        """
        return str(self._project_dir / f"{session_id}.jsonl")

    def delete_session(self, session_id: str) -> bool:
        """Delete a session file.

        Args:
            session_id: The session ID

        Returns:
            True if deleted, False if not found
        """
        session_path = self._project_dir / f"{session_id}.jsonl"

        if session_path.exists():
            session_path.unlink()
            return True

        return False


__all__ = ["ClaudeSessionManager"]
