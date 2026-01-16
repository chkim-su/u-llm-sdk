"""Codex-specific session management.

Codex stores sessions in ~/.codex/sessions/YYYY/MM/DD/ as JSONL files
with a date-based directory structure.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from u_llm_sdk.session.base import BaseSessionManager

if TYPE_CHECKING:
    from u_llm_sdk.session.message import SessionMessage


class CodexSessionManager(BaseSessionManager):
    """Session manager for Codex CLI.

    Session format: ~/.codex/sessions/YYYY/MM/DD/<uuid>.jsonl

    Codex uses a date-based directory structure for session organization.
    The session file includes metadata about instructions in the
    session_meta field.

    Example:
        >>> manager = CodexSessionManager("/home/user/myproject")
        >>> session_id = manager.create_from_system_prompt("You are a Rust expert.")
        >>> # Resume with: codex resume <session_id>
    """

    def __init__(self, project_path: str):
        """Initialize Codex session manager.

        Args:
            project_path: Absolute path to the project directory
        """
        super().__init__(project_path)
        self._base_dir = Path.home() / ".codex" / "sessions"

    def _get_date_dir(self) -> Path:
        """Get the directory for today's sessions.

        Returns:
            Path to today's session directory
        """
        now = datetime.now()
        return self._base_dir / str(now.year) / f"{now.month:02d}" / f"{now.day:02d}"

    def create_from_system_prompt(
        self,
        system_prompt: str,
        assistant_acknowledgment: str = "Understood.",
    ) -> str:
        """Create a session with injected system context.

        For Codex, system prompts are stored in session_meta.instructions
        and also prepended to the first user message.

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
        return self.create_session(messages, instructions=system_prompt)

    def create_session(
        self,
        messages: list["SessionMessage"],
        instructions: str | None = None,
    ) -> str:
        """Create a session with message history.

        Args:
            messages: List of messages to pre-seed
            instructions: Optional system instructions to store in metadata

        Returns:
            Session ID (UUID)
        """
        session_id = str(uuid.uuid4())

        # Get date-based directory
        session_dir = self._get_date_dir()
        session_dir.mkdir(parents=True, exist_ok=True)

        session_path = session_dir / f"{session_id}.jsonl"

        # Write session metadata first
        with session_path.open("w", encoding="utf-8") as f:
            # Session metadata entry
            meta = {
                "type": "session_meta",
                "session_id": session_id,
                "project_path": self.project_path,
                "created_at": datetime.now().isoformat(),
            }
            if instructions:
                meta["instructions"] = instructions
            f.write(json.dumps(meta) + "\n")

            # Write messages
            for msg in messages:
                entry = {
                    "type": msg.role,
                    "content": msg.resolve(),
                }
                f.write(json.dumps(entry) + "\n")

        return session_id

    def read_session(self, session_id: str) -> list[dict[str, Any]]:
        """Read messages from a session.

        Searches through date directories to find the session.

        Args:
            session_id: The session ID

        Returns:
            List of message dictionaries

        Raises:
            SessionNotFoundError: If session doesn't exist
        """
        session_path = self._find_session_path(session_id)

        if session_path is None:
            from u_llm_sdk.types import SessionNotFoundError

            raise SessionNotFoundError(f"Session not found: {session_id}")

        messages = []
        with session_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    # Skip metadata entries
                    if entry.get("type") in ("user", "assistant"):
                        messages.append({
                            "role": entry["type"],
                            "content": entry.get("content", ""),
                        })

        return messages

    def _find_session_path(self, session_id: str) -> Path | None:
        """Find a session file by searching date directories.

        Args:
            session_id: The session ID to find

        Returns:
            Path to session file or None if not found
        """
        if not self._base_dir.exists():
            return None

        # Search through year/month/day directories
        for year_dir in sorted(self._base_dir.iterdir(), reverse=True):
            if not year_dir.is_dir():
                continue
            for month_dir in sorted(year_dir.iterdir(), reverse=True):
                if not month_dir.is_dir():
                    continue
                for day_dir in sorted(month_dir.iterdir(), reverse=True):
                    if not day_dir.is_dir():
                        continue
                    session_path = day_dir / f"{session_id}.jsonl"
                    if session_path.exists():
                        return session_path

        return None

    def list_sessions(self) -> list[str]:
        """List all available session IDs.

        Returns:
            List of session IDs (most recent first)
        """
        if not self._base_dir.exists():
            return []

        sessions = []

        # Search through all date directories
        for year_dir in sorted(self._base_dir.iterdir(), reverse=True):
            if not year_dir.is_dir():
                continue
            for month_dir in sorted(year_dir.iterdir(), reverse=True):
                if not month_dir.is_dir():
                    continue
                for day_dir in sorted(month_dir.iterdir(), reverse=True):
                    if not day_dir.is_dir():
                        continue
                    for file in day_dir.glob("*.jsonl"):
                        sessions.append(file.stem)

        return sessions

    def get_session_path(self, session_id: str) -> str:
        """Get the file path for a session.

        Args:
            session_id: The session ID

        Returns:
            Absolute path to the session file

        Raises:
            SessionNotFoundError: If session doesn't exist
        """
        path = self._find_session_path(session_id)

        if path is None:
            from u_llm_sdk.types import SessionNotFoundError

            raise SessionNotFoundError(f"Session not found: {session_id}")

        return str(path)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session file.

        Args:
            session_id: The session ID

        Returns:
            True if deleted, False if not found
        """
        path = self._find_session_path(session_id)

        if path is not None:
            path.unlink()
            return True

        return False


__all__ = ["CodexSessionManager"]
