"""Orchestration State Types and Execution Context.

Contains all state-related types for orchestration:
- ExecutionState: Pipeline state enum
- WorkOrderState: Individual WorkOrder state enum
- Event/EventType: State transition events
- ExecutionContext: Central state container
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from u_llm_sdk.types import LLMResult

    from .contracts import Evidence, ExecutionPlan, ReviewReport, WorkOrder
    from .delta_compliance import DeltaAnalyzer, RepoSnapshot
    from .file_scope import FileSetManager
    from .git_integration import GitManager
    from .lock_manager import LockManager


# =============================================================================
# Execution States
# =============================================================================


class ExecutionState(Enum):
    """Pipeline execution state."""

    INIT = "init"
    PLANNING = "planning"
    DISPATCHING = "dispatching"
    WAITING_PR = "waiting_pr"
    REVIEWING = "reviewing"
    MERGING = "merging"
    REJECTED = "rejected"
    REDISPATCHING = "redispatching"
    AGGREGATING = "aggregating"
    COMPLETED = "completed"
    FAILED = "failed"


class WorkOrderState(Enum):
    """Individual WorkOrder state."""

    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    PR_SUBMITTED = "pr_submitted"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    MERGED = "merged"
    SKIPPED = "skipped"


# =============================================================================
# Events
# =============================================================================


@dataclass
class Event:
    """State transition trigger event."""

    type: str
    work_order_id: Optional[str] = None
    payload: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class EventType:
    """Event type constants."""

    PLAN_COMPLETE = "plan_complete"
    DISPATCH_COMPLETE = "dispatch_complete"
    PR_OPENED = "pr_opened"
    PR_UPDATED = "pr_updated"
    REVIEW_PASS = "review_pass"
    REVIEW_FAIL = "review_fail"
    MERGE_COMPLETE = "merge_complete"
    LAYER_COMPLETE = "layer_complete"
    ALL_MERGED = "all_merged"
    AGGREGATE_COMPLETE = "aggregate_complete"
    ERROR = "error"


# =============================================================================
# Execution Context
# =============================================================================


@dataclass
class ExecutionContext:
    """Execution context (state holder).

    Contains all state for a single orchestration run.
    """

    objective: str
    cwd: str

    state: ExecutionState = ExecutionState.INIT

    # Plan
    plan: Optional["ExecutionPlan"] = None
    work_orders: dict[str, "WorkOrder"] = field(default_factory=dict)
    work_order_states: dict[str, WorkOrderState] = field(default_factory=dict)

    # Layers (parallel execution units)
    layers: list[list[str]] = field(default_factory=list)
    current_layer: int = 0

    # Results
    results: dict[str, "LLMResult"] = field(default_factory=dict)
    evidences: dict[str, "Evidence"] = field(default_factory=dict)
    reviews: dict[str, "ReviewReport"] = field(default_factory=dict)

    # Event queue
    event_queue: list[Event] = field(default_factory=list)
    event_history: list[Event] = field(default_factory=list)

    # Final output
    merged_branches: list[str] = field(default_factory=list)
    final_output: Optional[str] = None

    # Timing
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    # Resource managers (initialized later)
    file_set_manager: Optional["FileSetManager"] = None
    lock_manager: Optional["LockManager"] = None
    git_manager: Optional["GitManager"] = None
    delta_analyzer: Optional["DeltaAnalyzer"] = None

    # Lock tracking
    held_locks: dict[str, list[str]] = field(default_factory=dict)  # wo_id -> [locks]

    # Branch tracking
    wo_branches: dict[str, str] = field(default_factory=dict)  # wo_id -> branch_name

    # Worktree tracking (P0 safety: isolated execution directories)
    wo_worktrees: dict[str, str] = field(
        default_factory=dict
    )  # wo_id -> worktree_path
    wo_base_commits: dict[str, str] = field(
        default_factory=dict
    )  # wo_id -> base_commit

    # Delta compliance tracking
    before_snapshots: dict[str, "RepoSnapshot"] = field(
        default_factory=dict
    )  # wo_id -> snapshot

    # MED-07: Context build error tracking
    context_build_errors: dict[str, str] = field(
        default_factory=dict
    )  # wo_id -> error message

    def push_event(self, event: Event) -> None:
        """Add event to queue."""
        self.event_queue.append(event)

    def pop_event(self) -> Optional[Event]:
        """Get next event from queue."""
        if self.event_queue:
            event = self.event_queue.pop(0)
            self.event_history.append(event)
            return event
        return None

    def get_duration_ms(self) -> int:
        """Get execution duration in milliseconds."""
        end = self.end_time or time.time()
        return int((end - self.start_time) * 1000)

    def to_summary(self) -> dict:
        """Generate execution summary."""
        completed = sum(
            1
            for s in self.work_order_states.values()
            if s == WorkOrderState.MERGED
        )
        failed = sum(
            1
            for s in self.work_order_states.values()
            if s == WorkOrderState.REJECTED
        )
        skipped = sum(
            1
            for s in self.work_order_states.values()
            if s == WorkOrderState.SKIPPED
        )

        return {
            "state": self.state.value,
            "objective": self.objective,
            "work_orders_total": len(self.work_orders),
            "work_orders_completed": completed,
            "work_orders_failed": failed,
            "work_orders_skipped": skipped,
            "layers_total": len(self.layers),
            "current_layer": self.current_layer,
            "merged_branches": self.merged_branches,
            "duration_ms": self.get_duration_ms(),
        }


__all__ = [
    "ExecutionState",
    "WorkOrderState",
    "Event",
    "EventType",
    "ExecutionContext",
]
