"""ChronicleAdapter - InterventionHook implementation for Chronicle recording.

Records ExecutionRecord and FailureRecord to Chronicle storage based on
LLM provider execution events.

IMPORTANT: This module imports from mv_rag.chronicle, which means it
creates a dependency from U-llm-sdk → MV-rag. This is intentional for
the "No API / Local-First" architecture where all packages run in-process.

If you need network isolation, use the HTTP-based approach via RAGClient instead.
"""

from __future__ import annotations

import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Optional, Protocol

from u_llm_sdk.types import LLMResult, PreActionContext
from u_llm_sdk.types.chronicle import (
    ExecutionRecord,
    ExecutionOutcome,
    FailureRecord,
    ErrorFingerprint,
)

if TYPE_CHECKING:
    from mv_rag.chronicle import ChronicleStore


class InterventionHook(Protocol):
    """Protocol for intervention hooks (matches hooks.py)."""

    async def on_pre_action(
        self,
        prompt: str,
        provider: str,
        model: Optional[str] = None,
        session_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Optional[PreActionContext]: ...

    async def on_post_action(
        self,
        result: LLMResult,
        pre_action_context: Optional[PreActionContext],
        run_id: Optional[str] = None,
    ) -> None: ...


@dataclass
class PendingExecution:
    """Tracks an in-progress execution for timing and context."""

    run_id: str
    session_id: str
    provider: str
    model: Optional[str]
    prompt_hash: str
    start_time: float
    logical_step: int


@dataclass
class ChronicleConfig:
    """Configuration for ChronicleAdapter.

    Attributes:
        decision_id_prefix: Prefix for synthetic decision IDs
        record_prompts: Whether to store prompt hashes
        failure_threshold: Exit code above which to create FailureRecord
        generate_fingerprints: Whether to generate ErrorFingerprints
    """

    decision_id_prefix: str = "dec_synthetic_"
    record_prompts: bool = True
    failure_threshold: int = 0
    generate_fingerprints: bool = True

    def to_dict(self) -> dict:
        return {
            "decision_id_prefix": self.decision_id_prefix,
            "record_prompts": self.record_prompts,
            "failure_threshold": self.failure_threshold,
            "generate_fingerprints": self.generate_fingerprints,
        }


class ChronicleAdapter:
    """InterventionHook that records to Chronicle storage.

    This adapter bridges the InterventionHook protocol with Chronicle's
    ExecutionRecord and FailureRecord storage. It:

    1. Records ExecutionRecord on every LLM completion (on_post_action)
    2. Records FailureRecord when execution fails
    3. Generates ErrorFingerprint for failure similarity matching
    4. Optionally chains to another hook (e.g., RAGClient)

    ID Canonicalization (per CHRONICLE_ARCHITECTURE.md):
        - ExecutionRecord.record_id = f"exec_{run_id}"
        - This ensures TimeKeeper event IDs map 1:1 to Chronicle records

    Usage:
        >>> from mv_rag.chronicle import ChronicleStore
        >>> store = ChronicleStore("/path/to/chronicle.db")
        >>>
        >>> # Standalone
        >>> adapter = ChronicleAdapter(store)
        >>>
        >>> # Chained with RAGClient
        >>> adapter = ChronicleAdapter(store, inner_hook=rag_client)
        >>>
        >>> # Use with provider
        >>> provider = ClaudeProvider(config, intervention_hook=adapter)
    """

    def __init__(
        self,
        store: "ChronicleStore",
        config: Optional[ChronicleConfig] = None,
        inner_hook: Optional[InterventionHook] = None,
        project_root: Optional[str] = None,
    ):
        """Initialize the Chronicle adapter.

        Args:
            store: ChronicleStore instance for recording
            config: Configuration options
            inner_hook: Optional hook to chain (receives all calls first)
            project_root: Project root for context extraction
        """
        self._store = store
        self._config = config or ChronicleConfig()
        self._inner_hook = inner_hook
        self._project_root = project_root

        # Track pending executions by run_id
        self._pending: dict[str, PendingExecution] = {}

        # Session-based logical step counters
        self._logical_steps: dict[str, int] = {}

    async def on_pre_action(
        self,
        prompt: str,
        provider: str,
        model: Optional[str] = None,
        session_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Optional[PreActionContext]:
        """Called before LLM action. Starts execution timer.

        Args:
            prompt: The prompt being sent
            provider: Provider name
            model: Model name
            session_id: Session ID
            run_id: Run ID (used for record_id canonicalization)

        Returns:
            PreActionContext from inner hook, or None
        """
        # Generate run_id if not provided
        actual_run_id = run_id or str(uuid.uuid4())
        actual_session_id = session_id or "default"

        # Get and increment logical step for this session
        logical_step = self._logical_steps.get(actual_session_id, 0) + 1
        self._logical_steps[actual_session_id] = logical_step

        # Record pending execution
        self._pending[actual_run_id] = PendingExecution(
            run_id=actual_run_id,
            session_id=actual_session_id,
            provider=provider,
            model=model,
            prompt_hash=self._hash_prompt(prompt) if self._config.record_prompts else "",
            start_time=time.time(),
            logical_step=logical_step,
        )

        # Chain to inner hook if present
        if self._inner_hook:
            return await self._inner_hook.on_pre_action(
                prompt, provider, model, session_id, run_id
            )

        return None

    async def on_post_action(
        self,
        result: LLMResult,
        pre_action_context: Optional[PreActionContext],
        run_id: Optional[str] = None,
    ) -> None:
        """Called after LLM action. Records ExecutionRecord and optionally FailureRecord.

        Args:
            result: The LLMResult from execution
            pre_action_context: Context from pre_action (if any)
            run_id: Run ID for record lookup
        """
        # Chain to inner hook first (fire-and-forget pattern)
        if self._inner_hook:
            try:
                await self._inner_hook.on_post_action(result, pre_action_context, run_id)
            except Exception:
                pass  # Don't let inner hook failures affect Chronicle recording

        # Get pending execution
        actual_run_id = run_id or "unknown"
        pending = self._pending.pop(actual_run_id, None)

        if pending is None:
            # No matching pre_action - create minimal context
            pending = PendingExecution(
                run_id=actual_run_id,
                session_id="default",
                provider="unknown",
                model=None,
                prompt_hash="",
                start_time=time.time(),
                logical_step=1,
            )

        # Calculate duration
        duration_ms = int((time.time() - pending.start_time) * 1000)

        # Determine outcome
        outcome = self._determine_outcome(result)

        # Create synthetic decision_id (for now, until full Decision integration)
        decision_id = f"{self._config.decision_id_prefix}{pending.run_id}"

        # Create ExecutionRecord
        exec_record = ExecutionRecord.create(
            event_id=pending.run_id,
            session_id=pending.session_id,
            logical_step=pending.logical_step,
            decision_id=decision_id,
            tool_name=f"llm:{pending.provider}",
            input_args={"model": pending.model, "prompt_hash": pending.prompt_hash},
            exit_code=0 if result.success else 1,
            duration_ms=duration_ms,
            output_summary=self._summarize_output(result),
            outcome=outcome,
        )

        # Save ExecutionRecord
        try:
            self._store.save_execution(exec_record)
        except Exception as e:
            # Log but don't raise - Chronicle failures shouldn't block execution
            import logging
            logging.getLogger(__name__).warning(
                f"Failed to save ExecutionRecord: {e}"
            )

        # Create FailureRecord if execution failed
        if not result.success and self._config.generate_fingerprints:
            await self._record_failure(exec_record, result, pending)

    async def _record_failure(
        self,
        exec_record: ExecutionRecord,
        result: LLMResult,
        pending: PendingExecution,
    ) -> None:
        """Create and save FailureRecord for failed execution."""
        # Generate error fingerprint
        fingerprint = self._generate_fingerprint(result, pending)

        # Create FailureRecord
        failure = FailureRecord.create(
            session_id=pending.session_id,
            logical_step=pending.logical_step,
            execution_id=exec_record.record_id,
            symptom=self._extract_symptom(result),
            error_fingerprint=fingerprint,
            cause=result.error if hasattr(result, "error") else None,
        )

        try:
            self._store.save_failure(failure)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                f"Failed to save FailureRecord: {e}"
            )

    def _determine_outcome(self, result: LLMResult) -> ExecutionOutcome:
        """Determine ExecutionOutcome from LLMResult."""
        if result.success:
            return ExecutionOutcome.SUCCESS

        # Check for specific failure modes
        stop_reason = getattr(result, "stop_reason", None)

        if stop_reason == "timeout":
            return ExecutionOutcome.TIMEOUT
        elif stop_reason == "abort" or stop_reason == "cancelled":
            return ExecutionOutcome.ABORT
        elif stop_reason == "error":
            return ExecutionOutcome.ERROR

        # Default to FAILURE for non-success
        return ExecutionOutcome.FAILURE

    def _generate_fingerprint(
        self,
        result: LLMResult,
        pending: PendingExecution,
    ) -> ErrorFingerprint:
        """Generate ErrorFingerprint from failed result."""
        # Extract error type from result
        error_text = getattr(result, "error", "") or ""
        stop_reason = getattr(result, "stop_reason", "") or ""

        # Determine error type
        if "timeout" in error_text.lower() or stop_reason == "timeout":
            error_type = "TimeoutError"
        elif "rate" in error_text.lower() or "limit" in error_text.lower():
            error_type = "RateLimitError"
        elif "auth" in error_text.lower() or "key" in error_text.lower():
            error_type = "AuthenticationError"
        else:
            error_type = "LLMExecutionError"

        # Normalize message (remove dynamic parts)
        normalized = self._normalize_error_message(error_text)

        # Extract stack trace if available
        stack_top_3 = None
        if hasattr(result, "traceback") and result.traceback:
            lines = result.traceback.split("\n")
            stack_top_3 = tuple(lines[:3]) if len(lines) >= 3 else tuple(lines)

        return ErrorFingerprint(
            error_type=error_type,
            normalized_message=normalized,
            error_code=stop_reason if stop_reason else None,
            stack_top_3=stack_top_3,
            affected_file=None,  # LLM errors typically don't have a specific file
        )

    def _normalize_error_message(self, message: str) -> str:
        """Normalize error message for fingerprinting.

        Applies rules from CHRONICLE_ARCHITECTURE.md:
        1. Strip timestamps
        2. Replace UUIDs with <UUID>
        3. Replace paths with <PATH>
        4. Replace numbers with <NUM>
        5. Normalize whitespace
        """
        import re

        # Start with original
        normalized = message

        # Strip timestamps (ISO format)
        normalized = re.sub(
            r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?",
            "<TIMESTAMP>",
            normalized,
        )

        # Replace UUIDs
        normalized = re.sub(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            "<UUID>",
            normalized,
            flags=re.IGNORECASE,
        )

        # Replace file paths (Unix and Windows)
        normalized = re.sub(
            r"(?:/[^\s:]+)+|(?:[A-Z]:\\[^\s:]+)+",
            "<PATH>",
            normalized,
        )

        # Replace standalone numbers (but not in words)
        normalized = re.sub(r"\b\d+\b", "<NUM>", normalized)

        # Normalize whitespace
        normalized = " ".join(normalized.split())

        # Truncate to reasonable length
        if len(normalized) > 200:
            normalized = normalized[:200] + "..."

        return normalized

    def _extract_symptom(self, result: LLMResult) -> str:
        """Extract human-readable symptom from result."""
        if hasattr(result, "error") and result.error:
            return result.error[:200]

        if hasattr(result, "stop_reason") and result.stop_reason:
            return f"Stopped: {result.stop_reason}"

        return "LLM execution failed with unknown error"

    def _summarize_output(self, result: LLMResult) -> str:
        """Create summary of LLM output for ExecutionRecord."""
        if not result.success:
            return f"[FAILED] {self._extract_symptom(result)}"

        text = result.text if hasattr(result, "text") else str(result)
        if len(text) > 100:
            return text[:100] + "..."
        return text

    def _hash_prompt(self, prompt: str) -> str:
        """Create hash of prompt for reproducibility tracking."""
        import hashlib
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]

    def get_session_stats(self, session_id: str) -> dict:
        """Get statistics for a session.

        Args:
            session_id: Session to get stats for

        Returns:
            Dict with execution counts, success rate, etc.
        """
        return self._store.get_stats()

    def reset_session(self, session_id: str) -> None:
        """Reset logical step counter for a session.

        Args:
            session_id: Session to reset
        """
        self._logical_steps.pop(session_id, None)
