"""Multi-LLM RAG Integration Module.

This module extends the base InterventionHook to provide Multi-LLM orchestration
specific functionality, including:

1. Orchestration Decision Logging:
   - Task routing decisions (which provider for which task type)
   - Brainstorming outcomes and consensus results
   - Escalation patterns and resolutions

2. Pattern Search:
   - Find similar past requests and their routing decisions
   - Learn from historical orchestration outcomes

3. Feedback Collection:
   - Success/failure rates per provider-task combination
   - Consensus convergence patterns

Architecture:
    MultiLLMRAGHook wraps RAGClient and adds orchestration-specific endpoints:
    - POST /api/v1/orchestration/decision (save routing decision)
    - POST /api/v1/orchestration/brainstorm (save brainstorm result)
    - GET /api/v1/orchestration/similar (search similar patterns)

Usage:
    >>> from u_llm_sdk.multi_llm import MultiLLMRAGHook
    >>> from u_llm_sdk.rag_client import RAGClientConfig
    >>>
    >>> hook = MultiLLMRAGHook(RAGClientConfig(base_url="http://localhost:8000"))
    >>>
    >>> # Save routing decision
    >>> await hook.save_routing_decision(
    ...     request="Build auth system",
    ...     task_type=TaskType.CODE_IMPLEMENTATION,
    ...     assigned_provider=Provider.CLAUDE,
    ...     routing_reason="Code task -> Claude",
    ... )
    >>>
    >>> # Search similar patterns
    >>> patterns = await hook.search_similar_patterns("Build login system")
    >>> for p in patterns:
    ...     print(f"Similar: {p.request} -> {p.assigned_provider}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Optional

from u_llm_sdk.types import (
    ConsensusResult,
    EscalationRequest,
    LLMResult,
    PreActionContext,
    Provider,
)

if TYPE_CHECKING:
    from ..rag_client import RAGClient, RAGClientConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Task Type Enum (for routing decisions)
# =============================================================================


class TaskType(Enum):
    """Task type for routing decisions.

    Used to categorize tasks for provider assignment.
    """

    CODE_IMPLEMENTATION = "code_implementation"
    CODE_REVIEW = "code_review"
    DEEP_ANALYSIS = "deep_analysis"
    ARCHITECTURE_DESIGN = "architecture_design"
    DEBUGGING = "debugging"
    CLARIFICATION = "clarification"
    BRAINSTORM = "brainstorm"
    GENERAL = "general"


# =============================================================================
# Data Classes for Orchestration Feedback
# =============================================================================


@dataclass
class RoutingDecision:
    """A routing decision made by the orchestrator.

    Attributes:
        request: Original user request
        task_type: Categorized task type
        assigned_provider: Provider assigned to this task
        routing_reason: Why this provider was chosen
        confidence: Confidence in the routing (0.0-1.0)
        context_factors: Factors that influenced the decision
        timestamp: When the decision was made
    """

    request: str
    task_type: TaskType
    assigned_provider: Provider
    routing_reason: str
    confidence: float = 1.0
    context_factors: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "request": self.request,
            "task_type": self.task_type.value,
            "assigned_provider": self.assigned_provider.value,
            "routing_reason": self.routing_reason,
            "confidence": self.confidence,
            "context_factors": self.context_factors,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> RoutingDecision:
        """Create from dictionary."""
        timestamp_value = data.get("timestamp")
        timestamp = (
            datetime.fromisoformat(timestamp_value)
            if isinstance(timestamp_value, str)
            else timestamp_value or datetime.now()
        )

        return cls(
            request=data["request"],
            task_type=TaskType(data["task_type"]),
            assigned_provider=Provider(data["assigned_provider"]),
            routing_reason=data["routing_reason"],
            confidence=data.get("confidence", 1.0),
            context_factors=data.get("context_factors", []),
            timestamp=timestamp,
        )


@dataclass
class BrainstormOutcome:
    """Outcome of a brainstorming session.

    Attributes:
        topic: The brainstorm topic
        consensus_result: Final consensus result
        rounds_taken: Number of rounds to reach consensus
        escalated_to_user: Whether user intervention was needed
        success: Whether consensus was reached
        participating_providers: Providers that participated
        timestamp: When the brainstorm concluded
    """

    topic: str
    consensus_result: Optional[ConsensusResult]
    rounds_taken: int
    escalated_to_user: bool
    success: bool
    participating_providers: list[Provider] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "topic": self.topic,
            "consensus_result": (
                self.consensus_result.to_dict() if self.consensus_result else None
            ),
            "rounds_taken": self.rounds_taken,
            "escalated_to_user": self.escalated_to_user,
            "success": self.success,
            "participating_providers": [p.value for p in self.participating_providers],
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> BrainstormOutcome:
        """Create from dictionary."""
        timestamp_value = data.get("timestamp")
        timestamp = (
            datetime.fromisoformat(timestamp_value)
            if isinstance(timestamp_value, str)
            else timestamp_value or datetime.now()
        )

        consensus_data = data.get("consensus_result")
        consensus_result = (
            ConsensusResult.from_dict(consensus_data) if consensus_data else None
        )

        return cls(
            topic=data["topic"],
            consensus_result=consensus_result,
            rounds_taken=data["rounds_taken"],
            escalated_to_user=data.get("escalated_to_user", False),
            success=data.get("success", False),
            participating_providers=[
                Provider(p) for p in data.get("participating_providers", [])
            ],
            timestamp=timestamp,
        )


@dataclass
class EscalationOutcome:
    """Outcome of an escalation request.

    Attributes:
        escalation_request: The original escalation request
        resolved: Whether the escalation was resolved
        resolution_source: How it was resolved (orchestrator, user, timeout)
        resolution_time_ms: Time to resolution in milliseconds
        clarifications_provided: Number of clarifications provided
        timestamp: When the escalation was resolved
    """

    escalation_request: EscalationRequest
    resolved: bool
    resolution_source: str  # "orchestrator", "user", "timeout"
    resolution_time_ms: int = 0
    clarifications_provided: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "escalation_request": self.escalation_request.to_dict(),
            "resolved": self.resolved,
            "resolution_source": self.resolution_source,
            "resolution_time_ms": self.resolution_time_ms,
            "clarifications_provided": self.clarifications_provided,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> EscalationOutcome:
        """Create from dictionary."""
        timestamp_value = data.get("timestamp")
        timestamp = (
            datetime.fromisoformat(timestamp_value)
            if isinstance(timestamp_value, str)
            else timestamp_value or datetime.now()
        )

        return cls(
            escalation_request=EscalationRequest.from_dict(data["escalation_request"]),
            resolved=data.get("resolved", False),
            resolution_source=data.get("resolution_source", "unknown"),
            resolution_time_ms=data.get("resolution_time_ms", 0),
            clarifications_provided=data.get("clarifications_provided", 0),
            timestamp=timestamp,
        )


@dataclass
class PatternMatch:
    """A similar pattern found from historical data.

    Attributes:
        request: The historical request
        task_type: How the task was categorized
        assigned_provider: Which provider was assigned
        outcome_success: Whether the execution was successful
        similarity_score: How similar to the query (0.0-1.0)
        context_similarity: Similarity of context factors
    """

    request: str
    task_type: TaskType
    assigned_provider: Provider
    outcome_success: bool
    similarity_score: float
    context_similarity: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "request": self.request,
            "task_type": self.task_type.value,
            "assigned_provider": self.assigned_provider.value,
            "outcome_success": self.outcome_success,
            "similarity_score": self.similarity_score,
            "context_similarity": self.context_similarity,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PatternMatch:
        """Create from dictionary."""
        return cls(
            request=data["request"],
            task_type=TaskType(data["task_type"]),
            assigned_provider=Provider(data["assigned_provider"]),
            outcome_success=data.get("outcome_success", True),
            similarity_score=data.get("similarity_score", 0.0),
            context_similarity=data.get("context_similarity", 0.0),
        )


@dataclass
class OrchestrationHint:
    """Hints for orchestration based on historical patterns.

    Provided in pre-action context to guide routing decisions.

    Attributes:
        suggested_provider: Suggested provider based on patterns
        suggested_task_type: Suggested task categorization
        confidence: Confidence in suggestions (0.0-1.0)
        similar_patterns: Relevant similar patterns found
        warnings: Any warnings based on past failures
    """

    suggested_provider: Optional[Provider] = None
    suggested_task_type: Optional[TaskType] = None
    confidence: float = 0.0
    similar_patterns: list[PatternMatch] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "suggested_provider": (
                self.suggested_provider.value if self.suggested_provider else None
            ),
            "suggested_task_type": (
                self.suggested_task_type.value if self.suggested_task_type else None
            ),
            "confidence": self.confidence,
            "similar_patterns": [p.to_dict() for p in self.similar_patterns],
            "warnings": self.warnings,
        }

    @classmethod
    def from_dict(cls, data: dict) -> OrchestrationHint:
        """Create from dictionary."""
        suggested_provider_value = data.get("suggested_provider")
        suggested_provider = (
            Provider(suggested_provider_value) if suggested_provider_value else None
        )

        suggested_task_type_value = data.get("suggested_task_type")
        suggested_task_type = (
            TaskType(suggested_task_type_value) if suggested_task_type_value else None
        )

        return cls(
            suggested_provider=suggested_provider,
            suggested_task_type=suggested_task_type,
            confidence=data.get("confidence", 0.0),
            similar_patterns=[
                PatternMatch.from_dict(p) for p in data.get("similar_patterns", [])
            ],
            warnings=data.get("warnings", []),
        )


# =============================================================================
# Multi-LLM RAG Hook
# =============================================================================


class MultiLLMRAGHook:
    """RAG hook specialized for Multi-LLM orchestration.

    This hook extends RAGClient functionality with orchestration-specific
    features like routing decision logging and pattern search.

    It implements the InterventionHook protocol for use with providers,
    while adding orchestration-specific methods.

    Example:
        >>> hook = MultiLLMRAGHook(config)
        >>>
        >>> # Use with orchestrator
        >>> orchestrator = GeminiOrchestrator(providers, rag_hook=hook)
        >>>
        >>> # Save routing decision after task assignment
        >>> await hook.save_routing_decision(
        ...     request="Build authentication",
        ...     task_type=TaskType.CODE_IMPLEMENTATION,
        ...     assigned_provider=Provider.CLAUDE,
        ...     routing_reason="Code implementation -> Claude (specialized)",
        ... )
        >>>
        >>> # Search for similar patterns
        >>> hints = await hook.get_orchestration_hints("Build OAuth flow")
        >>> if hints.suggested_provider:
        ...     print(f"Suggested: {hints.suggested_provider}")
    """

    def __init__(
        self,
        config: "RAGClientConfig",
        *,
        pattern_search_limit: int = 5,
        min_similarity_threshold: float = 0.6,
    ):
        """Initialize MultiLLMRAGHook.

        Args:
            config: RAGClient configuration
            pattern_search_limit: Max patterns to return in search
            min_similarity_threshold: Minimum similarity for pattern matches
        """
        # Import here to avoid circular import
        from ..rag_client import RAGClient

        self._rag_client = RAGClient(config)
        self._pattern_search_limit = pattern_search_limit
        self._min_similarity_threshold = min_similarity_threshold
        self._config = config

    # -------------------------------------------------------------------------
    # InterventionHook Protocol Implementation
    # -------------------------------------------------------------------------

    async def on_pre_action(
        self,
        prompt: str,
        provider: str,
        model: Optional[str] = None,
        session_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Optional[PreActionContext]:
        """Called before LLM action - delegates to RAGClient.

        Args:
            prompt: The user's prompt
            provider: Provider name
            model: Model name (optional)
            session_id: Session ID (optional)
            run_id: Run ID (optional)

        Returns:
            PreActionContext if context should be injected, None otherwise
        """
        return await self._rag_client.on_pre_action(
            prompt=prompt,
            provider=provider,
            model=model,
            session_id=session_id,
            run_id=run_id,
        )

    async def on_post_action(
        self,
        result: LLMResult,
        pre_action_context: Optional[PreActionContext] = None,
        run_id: Optional[str] = None,
    ) -> None:
        """Called after LLM action - delegates to RAGClient.

        Args:
            result: The LLM execution result
            pre_action_context: Context that was injected (if any)
            run_id: Run ID (optional)
        """
        await self._rag_client.on_post_action(
            result=result,
            pre_action_context=pre_action_context,
            run_id=run_id,
        )

    # -------------------------------------------------------------------------
    # Orchestration-Specific Methods
    # -------------------------------------------------------------------------

    async def save_routing_decision(
        self,
        request: str,
        task_type: TaskType,
        assigned_provider: Provider,
        routing_reason: str,
        *,
        confidence: float = 1.0,
        context_factors: Optional[list[str]] = None,
        session_id: Optional[str] = None,
    ) -> bool:
        """Save a routing decision for pattern learning.

        Args:
            request: Original user request
            task_type: Categorized task type
            assigned_provider: Provider assigned
            routing_reason: Why this provider was chosen
            confidence: Confidence in routing (0.0-1.0)
            context_factors: Factors that influenced decision
            session_id: Session ID for correlation

        Returns:
            True if saved successfully, False on error (fail-open)
        """
        decision = RoutingDecision(
            request=request,
            task_type=task_type,
            assigned_provider=assigned_provider,
            routing_reason=routing_reason,
            confidence=confidence,
            context_factors=context_factors or [],
        )

        try:
            response = await self._rag_client._post_with_retry(
                "/api/v1/orchestration/decision",
                json={
                    "decision": decision.to_dict(),
                    "session_id": session_id,
                },
            )
            response.raise_for_status()
            logger.info(
                f"Saved routing decision: {task_type.value} -> "
                f"{assigned_provider.value}"
            )
            return True

        except Exception as e:
            logger.warning(f"Failed to save routing decision (ignored): {e}")
            return False

    async def save_brainstorm_outcome(
        self,
        topic: str,
        consensus_result: Optional[ConsensusResult],
        rounds_taken: int,
        escalated_to_user: bool,
        success: bool,
        *,
        participating_providers: Optional[list[Provider]] = None,
        session_id: Optional[str] = None,
    ) -> bool:
        """Save a brainstorm outcome for pattern learning.

        Args:
            topic: The brainstorm topic
            consensus_result: Final consensus result
            rounds_taken: Number of rounds taken
            escalated_to_user: Whether user intervention was needed
            success: Whether consensus was reached
            participating_providers: Providers that participated
            session_id: Session ID for correlation

        Returns:
            True if saved successfully, False on error (fail-open)
        """
        outcome = BrainstormOutcome(
            topic=topic,
            consensus_result=consensus_result,
            rounds_taken=rounds_taken,
            escalated_to_user=escalated_to_user,
            success=success,
            participating_providers=participating_providers
            or [Provider.GEMINI, Provider.CLAUDE, Provider.CODEX],
        )

        try:
            response = await self._rag_client._post_with_retry(
                "/api/v1/orchestration/brainstorm",
                json={
                    "outcome": outcome.to_dict(),
                    "session_id": session_id,
                },
            )
            response.raise_for_status()
            logger.info(
                f"Saved brainstorm outcome: topic='{topic[:50]}...', "
                f"success={success}, rounds={rounds_taken}"
            )
            return True

        except Exception as e:
            logger.warning(f"Failed to save brainstorm outcome (ignored): {e}")
            return False

    async def save_escalation_outcome(
        self,
        escalation_request: EscalationRequest,
        resolved: bool,
        resolution_source: str,
        *,
        resolution_time_ms: int = 0,
        clarifications_provided: int = 0,
        session_id: Optional[str] = None,
    ) -> bool:
        """Save an escalation outcome for pattern learning.

        Args:
            escalation_request: The original escalation request
            resolved: Whether it was resolved
            resolution_source: How it was resolved (orchestrator, user, timeout)
            resolution_time_ms: Time to resolution
            clarifications_provided: Number of clarifications
            session_id: Session ID for correlation

        Returns:
            True if saved successfully, False on error (fail-open)
        """
        outcome = EscalationOutcome(
            escalation_request=escalation_request,
            resolved=resolved,
            resolution_source=resolution_source,
            resolution_time_ms=resolution_time_ms,
            clarifications_provided=clarifications_provided,
        )

        try:
            response = await self._rag_client._post_with_retry(
                "/api/v1/orchestration/escalation",
                json={
                    "outcome": outcome.to_dict(),
                    "session_id": session_id,
                },
            )
            response.raise_for_status()
            logger.info(
                f"Saved escalation outcome: resolved={resolved}, "
                f"source={resolution_source}"
            )
            return True

        except Exception as e:
            logger.warning(f"Failed to save escalation outcome (ignored): {e}")
            return False

    async def search_similar_patterns(
        self,
        request: str,
        *,
        task_type: Optional[TaskType] = None,
        limit: Optional[int] = None,
    ) -> list[PatternMatch]:
        """Search for similar historical patterns.

        Args:
            request: Request to find similar patterns for
            task_type: Filter by task type (optional)
            limit: Max patterns to return (default: self._pattern_search_limit)

        Returns:
            List of similar patterns, sorted by similarity
        """
        limit = limit or self._pattern_search_limit

        try:
            params = {
                "q": request,
                "limit": limit,
                "min_similarity": self._min_similarity_threshold,
            }
            if task_type:
                params["task_type"] = task_type.value

            response = await self._rag_client._get_with_retry(
                "/api/v1/orchestration/similar",
                params=params,
            )
            response.raise_for_status()

            data = response.json()
            patterns = [PatternMatch.from_dict(p) for p in data.get("patterns", [])]

            logger.debug(f"Found {len(patterns)} similar patterns for: {request[:50]}...")
            return patterns

        except Exception as e:
            logger.warning(f"Pattern search failed (returning empty): {e}")
            return []

    async def get_orchestration_hints(
        self,
        request: str,
        *,
        task_type: Optional[TaskType] = None,
    ) -> OrchestrationHint:
        """Get orchestration hints based on historical patterns.

        This is the main method for pre-routing guidance. It searches for
        similar patterns and derives suggestions for the orchestrator.

        Args:
            request: The user request to get hints for
            task_type: Known task type (if already determined)

        Returns:
            OrchestrationHint with suggestions and warnings
        """
        patterns = await self.search_similar_patterns(
            request=request,
            task_type=task_type,
        )

        if not patterns:
            return OrchestrationHint()

        # Analyze patterns for suggestions
        provider_votes: dict[Provider, tuple[int, float]] = {}  # (count, total_score)
        task_type_votes: dict[TaskType, int] = {}
        warnings: list[str] = []

        for pattern in patterns:
            # Vote for provider
            p = pattern.assigned_provider
            count, score = provider_votes.get(p, (0, 0.0))
            provider_votes[p] = (count + 1, score + pattern.similarity_score)

            # Vote for task type
            t = pattern.task_type
            task_type_votes[t] = task_type_votes.get(t, 0) + 1

            # Check for failures
            if not pattern.outcome_success and pattern.similarity_score > 0.8:
                warnings.append(
                    f"Similar request '{pattern.request[:30]}...' failed with "
                    f"{pattern.assigned_provider.value}"
                )

        # Find best provider
        best_provider = None
        best_score = 0.0
        for provider, (count, total_score) in provider_votes.items():
            avg_score = total_score / count if count > 0 else 0
            if avg_score > best_score:
                best_score = avg_score
                best_provider = provider

        # Find best task type
        best_task_type = None
        if task_type_votes:
            best_task_type = max(task_type_votes.keys(), key=lambda k: task_type_votes[k])

        # Calculate confidence
        confidence = min(best_score, 1.0) if best_provider else 0.0

        return OrchestrationHint(
            suggested_provider=best_provider,
            suggested_task_type=best_task_type,
            confidence=confidence,
            similar_patterns=patterns,
            warnings=warnings,
        )

    async def record_task_outcome(
        self,
        request: str,
        task_type: TaskType,
        assigned_provider: Provider,
        success: bool,
        *,
        error_message: Optional[str] = None,
        execution_time_ms: int = 0,
        session_id: Optional[str] = None,
    ) -> bool:
        """Record the outcome of a task execution.

        This provides feedback to improve future routing decisions.

        Args:
            request: The original request
            task_type: How the task was categorized
            assigned_provider: Which provider executed it
            success: Whether execution succeeded
            error_message: Error message if failed
            execution_time_ms: Execution time in milliseconds
            session_id: Session ID for correlation

        Returns:
            True if recorded successfully, False on error (fail-open)
        """
        try:
            response = await self._rag_client._post_with_retry(
                "/api/v1/orchestration/outcome",
                json={
                    "request": request,
                    "task_type": task_type.value,
                    "assigned_provider": assigned_provider.value,
                    "success": success,
                    "error_message": error_message,
                    "execution_time_ms": execution_time_ms,
                    "session_id": session_id,
                },
            )
            response.raise_for_status()
            logger.debug(
                f"Recorded task outcome: {task_type.value}/{assigned_provider.value} "
                f"-> {'success' if success else 'failure'}"
            )
            return True

        except Exception as e:
            logger.warning(f"Failed to record task outcome (ignored): {e}")
            return False

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def close(self) -> None:
        """Close the underlying RAG client."""
        await self._rag_client.close()

    async def __aenter__(self):
        """Async context manager entry."""
        await self._rag_client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._rag_client.__aexit__(exc_type, exc_val, exc_tb)


# =============================================================================
# No-Op Implementation for Testing
# =============================================================================


class NoOpMultiLLMRAGHook:
    """No-operation hook for testing without MV-rag.

    All methods return sensible defaults without making any API calls.
    """

    async def on_pre_action(
        self,
        prompt: str,
        provider: str,
        model: Optional[str] = None,
        session_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Optional[PreActionContext]:
        """Always returns None (no injection)."""
        return None

    async def on_post_action(
        self,
        result: LLMResult,
        pre_action_context: Optional[PreActionContext] = None,
        run_id: Optional[str] = None,
    ) -> None:
        """Does nothing."""
        pass

    async def save_routing_decision(self, *args, **kwargs) -> bool:
        """Always returns True (pretend success)."""
        return True

    async def save_brainstorm_outcome(self, *args, **kwargs) -> bool:
        """Always returns True (pretend success)."""
        return True

    async def save_escalation_outcome(self, *args, **kwargs) -> bool:
        """Always returns True (pretend success)."""
        return True

    async def search_similar_patterns(self, *args, **kwargs) -> list[PatternMatch]:
        """Always returns empty list."""
        return []

    async def get_orchestration_hints(self, *args, **kwargs) -> OrchestrationHint:
        """Always returns empty hint."""
        return OrchestrationHint()

    async def record_task_outcome(self, *args, **kwargs) -> bool:
        """Always returns True (pretend success)."""
        return True

    async def close(self) -> None:
        """Does nothing."""
        pass

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        pass


__all__ = [
    # Enums
    "TaskType",
    # Data classes
    "RoutingDecision",
    "BrainstormOutcome",
    "EscalationOutcome",
    "PatternMatch",
    "OrchestrationHint",
    # Hooks
    "MultiLLMRAGHook",
    "NoOpMultiLLMRAGHook",
]
