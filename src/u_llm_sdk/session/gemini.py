"""Gemini-specific session management.

Gemini stores sessions in ~/.gemini/tmp/<project-hash>/chats/ as JSON files
with a messages array format.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from u_llm_sdk.session.base import BaseSessionManager

if TYPE_CHECKING:
    from u_llm_sdk.session.message import SessionMessage


class GeminiSessionManager(BaseSessionManager):
    """Session manager for Gemini CLI.

    Session format: ~/.gemini/tmp/<project-hash>/chats/<uuid>.json

    Gemini uses a JSON format (not JSONL) with a messages array.
    The project hash is an MD5 hash of the project path.

    Example:
        >>> manager = GeminiSessionManager("/home/user/myproject")
        >>> session_id = manager.create_from_system_prompt("You are a Go expert.")
        >>> # Resume with: gemini --resume <session_id>
    """

    def __init__(self, project_path: str):
        """Initialize Gemini session manager.

        Args:
            project_path: Absolute path to the project directory
        """
        super().__init__(project_path)
        self._base_dir = Path.home() / ".gemini" / "tmp"
        self._project_dir = self._base_dir / self._path_to_hash(project_path) / "chats"

    def _path_to_hash(self, path: str) -> str:
        """Convert a path to Gemini's project hash format.

        Gemini uses MD5 hash of the project path for directory naming.

        Args:
            path: Absolute file path

        Returns:
            MD5 hash of the path
        """
        return hashlib.md5(path.encode("utf-8")).hexdigest()

    def create_from_system_prompt(
        self,
        system_prompt: str,
        assistant_acknowledgment: str = "Understood.",
    ) -> str:
        """Create a session with injected system context.

        For Gemini, system prompts are prepended to the first user message.

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

        session_path = self._project_dir / f"{session_id}.json"

        # Build session data in Gemini's JSON format
        session_data = {
            "id": session_id,
            "project_path": self.project_path,
            "created_at": datetime.now().isoformat(),
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.resolve(),
                }
                for msg in messages
            ],
        }

        # Write as JSON (not JSONL)
        with session_path.open("w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2)

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
        session_path = self._project_dir / f"{session_id}.json"

        if not session_path.exists():
            from u_llm_sdk.types import SessionNotFoundError

            raise SessionNotFoundError(f"Session not found: {session_id}")

        with session_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        return data.get("messages", [])

    def list_sessions(self) -> list[str]:
        """List all available session IDs.

        Returns:
            List of session IDs
        """
        if not self._project_dir.exists():
            return []

        sessions = []
        for file in self._project_dir.glob("*.json"):
            sessions.append(file.stem)

        return sorted(sessions)

    def get_session_path(self, session_id: str) -> str:
        """Get the file path for a session.

        Args:
            session_id: The session ID

        Returns:
            Absolute path to the session file
        """
        return str(self._project_dir / f"{session_id}.json")

    def delete_session(self, session_id: str) -> bool:
        """Delete a session file.

        Args:
            session_id: The session ID

        Returns:
            True if deleted, False if not found
        """
        session_path = self._project_dir / f"{session_id}.json"

        if session_path.exists():
            session_path.unlink()
            return True

        return False


__all__ = ["GeminiSessionManager"]
