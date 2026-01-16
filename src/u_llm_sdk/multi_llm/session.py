"""Session Management Module.

This module provides session storage and management for multi-LLM orchestration.
It handles session lifecycle, state persistence, and orchestrator switching.

Key Components:
    - SessionStore: Abstract interface for session storage
    - InMemorySessionStore: In-memory session storage
    - FileSessionStore: File-based persistent storage
    - SessionManager: High-level session lifecycle management

Usage:
    >>> from u_llm_sdk.multi_llm import SessionManager, InMemorySessionStore
    >>> from u_llm_sdk.types import SessionConfig, Provider
    >>>
    >>> store = InMemorySessionStore()
    >>> manager = SessionManager(store)
    >>>
    >>> # Create a new session
    >>> session = await manager.create_session(SessionConfig(
    ...     session_id="session-001",
    ...     orchestrator_provider=Provider.GEMINI,
    ... ))
    >>>
    >>> # Switch orchestrator
    >>> await manager.switch_orchestrator("session-001", Provider.CLAUDE)
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional, Protocol, runtime_checkable

from u_llm_sdk.types import (
    Provider,
    SessionConfig,
    OrchestratorState,
    Task,
    ConsensusResult,
    EscalationRequest,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Session Store Protocol
# =============================================================================


@runtime_checkable
class SessionStore(Protocol):
    """Protocol for session storage backends.

    Implementations must provide CRUD operations for sessions
    and support concurrent access patterns.
    """

    async def create(self, config: SessionConfig) -> OrchestratorState:
        """Create a new session.

        Args:
            config: Session configuration

        Returns:
            Initial orchestrator state

        Raises:
            SessionExistsError: If session already exists
        """
        ...

    async def get(self, session_id: str) -> Optional[OrchestratorState]:
        """Get session state by ID.

        Args:
            session_id: Session identifier

        Returns:
            Session state or None if not found
        """
        ...

    async def update(self, state: OrchestratorState) -> None:
        """Update session state.

        Args:
            state: Updated orchestrator state

        Raises:
            SessionNotFoundError: If session doesn't exist
        """
        ...

    async def delete(self, session_id: str) -> bool:
        """Delete a session.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted, False if not found
        """
        ...

    async def list_sessions(self) -> list[str]:
        """List all session IDs.

        Returns:
            List of session identifiers
        """
        ...

    async def exists(self, session_id: str) -> bool:
        """Check if session exists.

        Args:
            session_id: Session identifier

        Returns:
            True if session exists
        """
        ...


# =============================================================================
# Exceptions
# =============================================================================


class SessionError(Exception):
    """Base exception for session errors."""

    pass


class SessionExistsError(SessionError):
    """Raised when trying to create a session that already exists."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        super().__init__(f"Session already exists: {session_id}")


class SessionNotFoundError(SessionError):
    """Raised when session is not found."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        super().__init__(f"Session not found: {session_id}")


# =============================================================================
# In-Memory Session Store
# =============================================================================


class InMemorySessionStore:
    """In-memory session store for development and testing.

    Thread-safe implementation using per-session asyncio locks.
    Data is lost when the process terminates.

    Attributes:
        sessions: Internal session storage
        configs: Session configurations
    """

    def __init__(self) -> None:
        """Initialize in-memory store."""
        self._sessions: dict[str, OrchestratorState] = {}
        self._configs: dict[str, SessionConfig] = {}
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()  # For structural changes only
        self._created_at: dict[str, datetime] = {}

    def _get_session_lock(self, session_id: str) -> asyncio.Lock:
        """Get or create a lock for a specific session."""
        if session_id not in self._session_locks:
            self._session_locks[session_id] = asyncio.Lock()
        return self._session_locks[session_id]

    async def create(self, config: SessionConfig) -> OrchestratorState:
        """Create a new session in memory."""
        async with self._global_lock:
            if config.session_id in self._sessions:
                raise SessionExistsError(config.session_id)

            # Create session lock before releasing global lock
            self._session_locks[config.session_id] = asyncio.Lock()

            state = OrchestratorState(
                session_id=config.session_id,
                current_provider=config.orchestrator_provider,
                session_context="",
                active_tasks=[],
                pending_escalations=[],
                consensus_history=[],
            )

            self._sessions[config.session_id] = state
            self._configs[config.session_id] = config
            self._created_at[config.session_id] = datetime.now()

            logger.info(f"Created session: {config.session_id}")
            return state

    async def get(self, session_id: str) -> Optional[OrchestratorState]:
        """Get session state from memory."""
        lock = self._get_session_lock(session_id)
        async with lock:
            return self._sessions.get(session_id)

    async def update(self, state: OrchestratorState) -> None:
        """Update session state in memory."""
        lock = self._get_session_lock(state.session_id)
        async with lock:
            if state.session_id not in self._sessions:
                raise SessionNotFoundError(state.session_id)

            self._sessions[state.session_id] = state
            logger.debug(f"Updated session: {state.session_id}")

    async def delete(self, session_id: str) -> bool:
        """Delete session from memory."""
        async with self._global_lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                self._configs.pop(session_id, None)
                self._created_at.pop(session_id, None)
                self._session_locks.pop(session_id, None)
                logger.info(f"Deleted session: {session_id}")
                return True
            return False

    async def list_sessions(self) -> list[str]:
        """List all session IDs."""
        async with self._global_lock:
            return list(self._sessions.keys())

    async def exists(self, session_id: str) -> bool:
        """Check if session exists."""
        # Read-only check, no lock needed for dict membership
        return session_id in self._sessions

    async def get_config(self, session_id: str) -> Optional[SessionConfig]:
        """Get session configuration.

        Args:
            session_id: Session identifier

        Returns:
            Session config or None
        """
        lock = self._get_session_lock(session_id)
        async with lock:
            return self._configs.get(session_id)

    async def get_session_age(self, session_id: str) -> Optional[timedelta]:
        """Get session age.

        Args:
            session_id: Session identifier

        Returns:
            Session age or None if not found
        """
        lock = self._get_session_lock(session_id)
        async with lock:
            created = self._created_at.get(session_id)
            if created:
                return datetime.now() - created
            return None

    async def cleanup_old_sessions(self, max_age: timedelta) -> int:
        """Remove sessions older than max_age.

        Args:
            max_age: Maximum session age

        Returns:
            Number of sessions removed
        """
        async with self._global_lock:
            now = datetime.now()
            to_remove = [
                sid
                for sid, created in self._created_at.items()
                if now - created > max_age
            ]

            for sid in to_remove:
                self._sessions.pop(sid, None)
                self._configs.pop(sid, None)
                self._created_at.pop(sid, None)
                self._session_locks.pop(sid, None)

            if to_remove:
                logger.info(f"Cleaned up {len(to_remove)} old sessions")

            return len(to_remove)


# =============================================================================
# File-Based Session Store
# =============================================================================


class FileSessionStore:
    """File-based persistent session store.

    Stores each session as a JSON file in the specified directory.
    Suitable for single-process deployments needing persistence.
    Uses per-session locks for better concurrency.

    Attributes:
        base_dir: Directory for session files
    """

    def __init__(self, base_dir: Path) -> None:
        """Initialize file-based store.

        Args:
            base_dir: Directory for session files (created if needed)
        """
        self.base_dir = Path(base_dir).expanduser()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()

    def _get_session_lock(self, session_id: str) -> asyncio.Lock:
        """Get or create a lock for a specific session."""
        if session_id not in self._session_locks:
            self._session_locks[session_id] = asyncio.Lock()
        return self._session_locks[session_id]

    def _session_path(self, session_id: str) -> Path:
        """Get file path for session."""
        return self.base_dir / f"{session_id}.json"

    def _config_path(self, session_id: str) -> Path:
        """Get file path for session config."""
        return self.base_dir / f"{session_id}.config.json"

    async def create(self, config: SessionConfig) -> OrchestratorState:
        """Create a new session file."""
        async with self._global_lock:
            session_path = self._session_path(config.session_id)
            if session_path.exists():
                raise SessionExistsError(config.session_id)

            # Create session lock before releasing global lock
            self._session_locks[config.session_id] = asyncio.Lock()

            state = OrchestratorState(
                session_id=config.session_id,
                current_provider=config.orchestrator_provider,
                session_context="",
                active_tasks=[],
                pending_escalations=[],
                consensus_history=[],
            )

            # Save state
            session_data = state.to_dict()
            session_data["_created_at"] = datetime.now().isoformat()
            session_path.write_text(json.dumps(session_data, indent=2))

            # Save config
            config_path = self._config_path(config.session_id)
            config_path.write_text(json.dumps(config.to_dict(), indent=2))

            logger.info(f"Created session file: {session_path}")
            return state

    async def get(self, session_id: str) -> Optional[OrchestratorState]:
        """Load session state from file."""
        lock = self._get_session_lock(session_id)
        async with lock:
            session_path = self._session_path(session_id)
            if not session_path.exists():
                return None

            try:
                data = json.loads(session_path.read_text())
                # Remove internal metadata
                data.pop("_created_at", None)
                return OrchestratorState.from_dict(data)
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Failed to load session {session_id}: {e}")
                return None

    async def update(self, state: OrchestratorState) -> None:
        """Update session file."""
        lock = self._get_session_lock(state.session_id)
        async with lock:
            session_path = self._session_path(state.session_id)
            if not session_path.exists():
                raise SessionNotFoundError(state.session_id)

            # Preserve created_at
            try:
                existing = json.loads(session_path.read_text())
                created_at = existing.get("_created_at")
            except (json.JSONDecodeError, FileNotFoundError):
                created_at = datetime.now().isoformat()

            session_data = state.to_dict()
            session_data["_created_at"] = created_at
            session_data["_updated_at"] = datetime.now().isoformat()

            session_path.write_text(json.dumps(session_data, indent=2))
            logger.debug(f"Updated session file: {session_path}")

    async def delete(self, session_id: str) -> bool:
        """Delete session file."""
        async with self._global_lock:
            session_path = self._session_path(session_id)
            config_path = self._config_path(session_id)

            deleted = False
            if session_path.exists():
                session_path.unlink()
                deleted = True

            if config_path.exists():
                config_path.unlink()

            self._session_locks.pop(session_id, None)

            if deleted:
                logger.info(f"Deleted session file: {session_path}")

            return deleted

    async def list_sessions(self) -> list[str]:
        """List all session IDs from files."""
        async with self._global_lock:
            sessions = []
            for path in self.base_dir.glob("*.json"):
                if not path.name.endswith(".config.json"):
                    sessions.append(path.stem)
            return sessions

    async def exists(self, session_id: str) -> bool:
        """Check if session file exists."""
        return self._session_path(session_id).exists()

    async def get_config(self, session_id: str) -> Optional[SessionConfig]:
        """Load session configuration from file."""
        lock = self._get_session_lock(session_id)
        async with lock:
            config_path = self._config_path(session_id)
            if not config_path.exists():
                return None

            try:
                data = json.loads(config_path.read_text())
                return SessionConfig.from_dict(data)
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Failed to load config {session_id}: {e}")
                return None

    async def cleanup_old_sessions(self, max_age: timedelta) -> int:
        """Remove session files older than max_age.

        Args:
            max_age: Maximum session age

        Returns:
            Number of sessions removed
        """
        async with self._global_lock:
            now = datetime.now()
            removed = 0

            for path in self.base_dir.glob("*.json"):
                if path.name.endswith(".config.json"):
                    continue

                try:
                    data = json.loads(path.read_text())
                    created_str = data.get("_created_at")
                    if created_str:
                        created = datetime.fromisoformat(created_str)
                        if now - created > max_age:
                            session_id = path.stem
                            path.unlink()
                            config_path = self._config_path(session_id)
                            if config_path.exists():
                                config_path.unlink()
                            self._session_locks.pop(session_id, None)
                            removed += 1
                except (json.JSONDecodeError, ValueError):
                    continue

            if removed:
                logger.info(f"Cleaned up {removed} old session files")

            return removed


# =============================================================================
# Session Manager
# =============================================================================


class SessionManager:
    """High-level session lifecycle manager.

    Handles session creation, orchestrator switching, task management,
    and session cleanup. Works with any SessionStore implementation.

    Attributes:
        store: Session storage backend
    """

    def __init__(self, store: SessionStore) -> None:
        """Initialize session manager.

        Args:
            store: Session storage backend
        """
        self.store = store
        self._event_handlers: dict[str, list[Callable]] = {
            "session_created": [],
            "session_deleted": [],
            "orchestrator_switched": [],
            "task_added": [],
            "task_completed": [],
        }

    # -------------------------------------------------------------------------
    # Session Lifecycle
    # -------------------------------------------------------------------------

    async def create_session(
        self,
        config: Optional[SessionConfig] = None,
        session_id: Optional[str] = None,
        orchestrator: Provider = Provider.GEMINI,
    ) -> OrchestratorState:
        """Create a new orchestration session.

        Args:
            config: Full session configuration (preferred)
            session_id: Session ID (auto-generated if not provided)
            orchestrator: Initial orchestrator provider

        Returns:
            Initial orchestrator state

        Raises:
            SessionExistsError: If session already exists
        """
        if config is None:
            config = SessionConfig(
                session_id=session_id or f"session-{uuid.uuid4().hex[:8]}",
                orchestrator_provider=orchestrator,
            )

        state = await self.store.create(config)
        await self._emit("session_created", state)

        logger.info(
            f"Session created: {config.session_id} "
            f"(orchestrator: {orchestrator.value})"
        )
        return state

    async def get_session(self, session_id: str) -> Optional[OrchestratorState]:
        """Get session state.

        Args:
            session_id: Session identifier

        Returns:
            Session state or None
        """
        return await self.store.get(session_id)

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted
        """
        state = await self.store.get(session_id)
        deleted = await self.store.delete(session_id)

        if deleted and state:
            await self._emit("session_deleted", state)
            logger.info(f"Session deleted: {session_id}")

        return deleted

    async def list_sessions(self) -> list[str]:
        """List all active sessions.

        Returns:
            List of session IDs
        """
        return await self.store.list_sessions()

    # -------------------------------------------------------------------------
    # Orchestrator Management
    # -------------------------------------------------------------------------

    async def switch_orchestrator(
        self,
        session_id: str,
        new_provider: Provider,
        preserve_context: bool = True,
    ) -> OrchestratorState:
        """Switch the orchestrator for a session.

        Transfers session state to a new orchestrator provider while
        preserving active tasks, escalations, and consensus history.

        Args:
            session_id: Session identifier
            new_provider: New orchestrator provider
            preserve_context: Whether to preserve session context

        Returns:
            Updated orchestrator state

        Raises:
            SessionNotFoundError: If session not found
        """
        state = await self.store.get(session_id)
        if state is None:
            raise SessionNotFoundError(session_id)

        old_provider = state.current_provider

        if old_provider == new_provider:
            logger.warning(f"Orchestrator unchanged: {new_provider.value}")
            return state

        # Create new state with transferred data
        new_state = OrchestratorState(
            session_id=session_id,
            current_provider=new_provider,
            session_context=state.session_context if preserve_context else "",
            active_tasks=state.active_tasks,
            pending_escalations=state.pending_escalations,
            consensus_history=state.consensus_history,
        )

        await self.store.update(new_state)
        await self._emit(
            "orchestrator_switched",
            {
                "session_id": session_id,
                "old_provider": old_provider,
                "new_provider": new_provider,
            },
        )

        logger.info(
            f"Orchestrator switched: {old_provider.value} -> {new_provider.value} "
            f"(session: {session_id})"
        )
        return new_state

    async def get_current_orchestrator(self, session_id: str) -> Optional[Provider]:
        """Get the current orchestrator provider.

        Args:
            session_id: Session identifier

        Returns:
            Current provider or None
        """
        state = await self.store.get(session_id)
        return state.current_provider if state else None

    # -------------------------------------------------------------------------
    # Task Management
    # -------------------------------------------------------------------------

    async def add_task(self, session_id: str, task: Task) -> OrchestratorState:
        """Add a task to the session.

        Args:
            session_id: Session identifier
            task: Task to add

        Returns:
            Updated state

        Raises:
            SessionNotFoundError: If session not found
        """
        state = await self.store.get(session_id)
        if state is None:
            raise SessionNotFoundError(session_id)

        # Check for duplicate
        existing_ids = {t.task_id for t in state.active_tasks}
        if task.task_id in existing_ids:
            logger.warning(f"Task already exists: {task.task_id}")
            return state

        state.active_tasks.append(task)
        await self.store.update(state)
        await self._emit("task_added", {"session_id": session_id, "task": task})

        logger.debug(f"Task added: {task.task_id} (session: {session_id})")
        return state

    async def complete_task(
        self, session_id: str, task_id: str
    ) -> Optional[OrchestratorState]:
        """Mark a task as completed and remove from active tasks.

        Args:
            session_id: Session identifier
            task_id: Task identifier

        Returns:
            Updated state or None if not found
        """
        state = await self.store.get(session_id)
        if state is None:
            raise SessionNotFoundError(session_id)

        # Find and remove task
        task = None
        for i, t in enumerate(state.active_tasks):
            if t.task_id == task_id:
                task = state.active_tasks.pop(i)
                break

        if task is None:
            logger.warning(f"Task not found: {task_id}")
            return state

        await self.store.update(state)
        await self._emit(
            "task_completed", {"session_id": session_id, "task_id": task_id}
        )

        logger.debug(f"Task completed: {task_id} (session: {session_id})")
        return state

    async def get_active_tasks(self, session_id: str) -> list[Task]:
        """Get all active tasks for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of active tasks
        """
        state = await self.store.get(session_id)
        return state.active_tasks if state else []

    # -------------------------------------------------------------------------
    # Escalation Management
    # -------------------------------------------------------------------------

    async def add_escalation(
        self, session_id: str, escalation: EscalationRequest
    ) -> OrchestratorState:
        """Add a pending escalation to the session.

        Args:
            session_id: Session identifier
            escalation: Escalation request

        Returns:
            Updated state
        """
        state = await self.store.get(session_id)
        if state is None:
            raise SessionNotFoundError(session_id)

        state.pending_escalations.append(escalation)
        await self.store.update(state)

        logger.debug(f"Escalation added (session: {session_id})")
        return state

    async def resolve_escalation(
        self, session_id: str, task_id: str
    ) -> Optional[EscalationRequest]:
        """Resolve and remove an escalation by task ID.

        Args:
            session_id: Session identifier
            task_id: Original task ID

        Returns:
            Resolved escalation or None
        """
        state = await self.store.get(session_id)
        if state is None:
            raise SessionNotFoundError(session_id)

        # Find and remove escalation
        escalation = None
        for i, e in enumerate(state.pending_escalations):
            if e.original_task.task_id == task_id:
                escalation = state.pending_escalations.pop(i)
                break

        if escalation:
            await self.store.update(state)
            logger.debug(f"Escalation resolved: {task_id} (session: {session_id})")

        return escalation

    # -------------------------------------------------------------------------
    # Consensus History
    # -------------------------------------------------------------------------

    async def add_consensus_result(
        self, session_id: str, result: ConsensusResult
    ) -> OrchestratorState:
        """Add a consensus result to session history.

        Args:
            session_id: Session identifier
            result: Consensus result

        Returns:
            Updated state
        """
        state = await self.store.get(session_id)
        if state is None:
            raise SessionNotFoundError(session_id)

        state.consensus_history.append(result)
        await self.store.update(state)

        logger.debug(
            f"Consensus added: success={result.success} (session: {session_id})"
        )
        return state

    async def get_consensus_history(self, session_id: str) -> list[ConsensusResult]:
        """Get consensus history for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of consensus results
        """
        state = await self.store.get(session_id)
        return state.consensus_history if state else []

    # -------------------------------------------------------------------------
    # Context Management
    # -------------------------------------------------------------------------

    async def update_context(
        self, session_id: str, context: str, append: bool = False
    ) -> OrchestratorState:
        """Update session context.

        Args:
            session_id: Session identifier
            context: New context or context to append
            append: If True, append to existing context

        Returns:
            Updated state
        """
        state = await self.store.get(session_id)
        if state is None:
            raise SessionNotFoundError(session_id)

        if append and state.session_context:
            state.session_context = f"{state.session_context}\n\n{context}"
        else:
            state.session_context = context

        await self.store.update(state)
        return state

    # -------------------------------------------------------------------------
    # Event Handling
    # -------------------------------------------------------------------------

    def on(self, event: str, handler: Callable) -> None:
        """Register an event handler.

        Args:
            event: Event name (session_created, session_deleted,
                   orchestrator_switched, task_added, task_completed)
            handler: Async handler function
        """
        if event in self._event_handlers:
            self._event_handlers[event].append(handler)

    async def _emit(self, event: str, data: any) -> None:
        """Emit an event to all handlers."""
        handlers = self._event_handlers.get(event, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                logger.error(f"Event handler error ({event}): {e}")

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    async def cleanup_old_sessions(
        self, max_age_hours: int = 24
    ) -> int:
        """Remove sessions older than specified age.

        Args:
            max_age_hours: Maximum session age in hours

        Returns:
            Number of sessions removed
        """
        if hasattr(self.store, "cleanup_old_sessions"):
            max_age = timedelta(hours=max_age_hours)
            return await self.store.cleanup_old_sessions(max_age)
        return 0

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    async def get_session_summary(self, session_id: str) -> Optional[dict]:
        """Get a summary of session state.

        Args:
            session_id: Session identifier

        Returns:
            Summary dict or None
        """
        state = await self.store.get(session_id)
        if state is None:
            return None

        return {
            "session_id": state.session_id,
            "current_orchestrator": state.current_provider.value,
            "active_tasks_count": len(state.active_tasks),
            "pending_escalations_count": len(state.pending_escalations),
            "consensus_count": len(state.consensus_history),
            "has_context": bool(state.session_context),
            "context_length": len(state.session_context),
        }
