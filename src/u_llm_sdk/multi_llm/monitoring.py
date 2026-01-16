"""Monitoring and Logging Module for Multi-LLM Orchestration.

This module provides comprehensive monitoring, logging, and debugging capabilities:
- Structured logging with context
- Debug mode for detailed tracing
- Metric exporters (Prometheus, JSON)
- Event emitters for external monitoring

Key Components:
    - OrchestrationLogger: Structured logging with correlation IDs
    - DebugMode: Detailed tracing for development
    - MetricExporter: Export metrics to various formats
    - EventEmitter: Emit events for external monitoring systems

Usage:
    >>> from u_llm_sdk.multi_llm.monitoring import (
    ...     OrchestrationLogger,
    ...     DebugMode,
    ...     MetricExporter,
    ... )
    >>>
    >>> # Structured logging
    >>> logger = OrchestrationLogger(session_id="abc123")
    >>> logger.info("Starting brainstorm", topic="Architecture")
    >>>
    >>> # Debug mode
    >>> with DebugMode():
    ...     # Detailed tracing enabled
    ...     result = await orchestrator.process()
    >>>
    >>> # Export metrics
    >>> exporter = MetricExporter(metrics)
    >>> prometheus_output = exporter.to_prometheus()
"""

import asyncio
import json
import logging
import sys
import time
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Protocol,
    Set,
    TypeVar,
    Union,
)

from u_llm_sdk.types import Provider

from .performance import PerformanceMetrics

# Configure module logger
module_logger = logging.getLogger(__name__)

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# Log Levels and Event Types
# =============================================================================


class LogLevel(Enum):
    """Log severity levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class EventType(Enum):
    """Types of orchestration events."""

    # Session events
    SESSION_START = "session_start"
    SESSION_END = "session_end"

    # Request events
    REQUEST_RECEIVED = "request_received"
    REQUEST_PROCESSED = "request_processed"

    # Routing events
    TASK_ROUTED = "task_routed"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"

    # Brainstorm events
    BRAINSTORM_START = "brainstorm_start"
    BRAINSTORM_ROUND = "brainstorm_round"
    BRAINSTORM_END = "brainstorm_end"

    # Consensus events
    CONSENSUS_VOTE = "consensus_vote"
    CONSENSUS_REACHED = "consensus_reached"
    CONSENSUS_FAILED = "consensus_failed"

    # Escalation events
    ESCALATION_STARTED = "escalation_started"
    ESCALATION_RESOLVED = "escalation_resolved"

    # Provider events
    PROVIDER_CALL = "provider_call"
    PROVIDER_RESPONSE = "provider_response"
    PROVIDER_ERROR = "provider_error"

    # System events
    METRIC_RECORDED = "metric_recorded"
    WARNING_TRIGGERED = "warning_triggered"
    ERROR_OCCURRED = "error_occurred"


# =============================================================================
# Log Entry Data Classes
# =============================================================================


@dataclass
class LogEntry:
    """Structured log entry."""

    timestamp: str
    level: str
    message: str
    event_type: Optional[str] = None
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    provider: Optional[str] = None
    operation: Optional[str] = None
    latency_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    stack_trace: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = {}
        for k, v in asdict(self).items():
            if v is not None and v != {}:
                result[k] = v
        return result

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


@dataclass
class EventRecord:
    """Record of an orchestration event."""

    event_type: EventType
    timestamp: float = field(default_factory=time.time)
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    provider: Optional[Provider] = None
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "correlation_id": self.correlation_id,
            "provider": self.provider.value if self.provider else None,
            "data": self.data,
        }


# =============================================================================
# Orchestration Logger
# =============================================================================


class OrchestrationLogger:
    """Structured logger for multi-LLM orchestration.

    Features:
        - Correlation IDs for request tracing
        - Session context propagation
        - Structured JSON output option
        - Provider-specific logging

    Usage:
        >>> logger = OrchestrationLogger(session_id="abc123")
        >>> logger.info("Processing request", request_id="req-1")
        >>>
        >>> with logger.operation("brainstorm"):
        ...     logger.debug("Starting round 1")
        ...     # ...
        ...     logger.info("Round 1 complete", votes={"a": 2, "b": 1})
    """

    def __init__(
        self,
        name: str = "orchestration",
        session_id: Optional[str] = None,
        json_output: bool = False,
        log_level: LogLevel = LogLevel.INFO,
    ):
        """Initialize orchestration logger.

        Args:
            name: Logger name
            session_id: Session identifier for correlation
            json_output: If True, output structured JSON logs
            log_level: Minimum log level
        """
        self._logger = logging.getLogger(name)
        self._session_id = session_id
        self._json_output = json_output
        self._log_level = log_level
        self._correlation_id: Optional[str] = None
        self._current_operation: Optional[str] = None
        self._current_provider: Optional[Provider] = None
        self._log_buffer: List[LogEntry] = []
        self._buffer_enabled = False

    @property
    def session_id(self) -> Optional[str]:
        """Get current session ID."""
        return self._session_id

    @session_id.setter
    def session_id(self, value: str) -> None:
        """Set session ID."""
        self._session_id = value

    def new_correlation_id(self) -> str:
        """Generate and set a new correlation ID."""
        self._correlation_id = str(uuid.uuid4())[:8]
        return self._correlation_id

    @contextmanager
    def operation(
        self, name: str, provider: Optional[Provider] = None
    ) -> Iterator[None]:
        """Context manager for logging an operation.

        Args:
            name: Operation name
            provider: Provider performing the operation
        """
        old_operation = self._current_operation
        old_provider = self._current_provider
        self._current_operation = name
        self._current_provider = provider

        start = time.time()
        self.debug(f"Starting {name}")

        try:
            yield
        except Exception as e:
            self.error(f"Failed {name}", error=str(e))
            raise
        finally:
            elapsed_ms = (time.time() - start) * 1000
            self.debug(f"Completed {name}", latency_ms=elapsed_ms)
            self._current_operation = old_operation
            self._current_provider = old_provider

    def _create_entry(
        self,
        level: LogLevel,
        message: str,
        event_type: Optional[EventType] = None,
        error: Optional[Exception] = None,
        **kwargs: Any,
    ) -> LogEntry:
        """Create a log entry."""
        entry = LogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            level=level.value,
            message=message,
            event_type=event_type.value if event_type else None,
            session_id=self._session_id,
            correlation_id=self._correlation_id,
            provider=self._current_provider.value if self._current_provider else None,
            operation=self._current_operation,
            metadata=kwargs,
        )

        if error:
            entry.error = str(error)
            entry.stack_trace = traceback.format_exc()

        return entry

    def _log(
        self,
        level: LogLevel,
        message: str,
        event_type: Optional[EventType] = None,
        error: Optional[Exception] = None,
        **kwargs: Any,
    ) -> None:
        """Internal log method."""
        entry = self._create_entry(level, message, event_type, error, **kwargs)

        if self._buffer_enabled:
            self._log_buffer.append(entry)

        # Get Python log level
        py_level = getattr(logging, level.value.upper())

        if self._json_output:
            self._logger.log(py_level, entry.to_json())
        else:
            # Format human-readable message
            parts = [message]
            if self._session_id:
                parts.insert(0, f"[{self._session_id[:8]}]")
            if self._correlation_id:
                parts.insert(1, f"({self._correlation_id})")
            if kwargs:
                parts.append(str(kwargs))
            self._logger.log(py_level, " ".join(parts))

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self._log(LogLevel.DEBUG, message, **kwargs)

    def info(
        self, message: str, event_type: Optional[EventType] = None, **kwargs: Any
    ) -> None:
        """Log info message."""
        self._log(LogLevel.INFO, message, event_type, **kwargs)

    def warning(
        self, message: str, event_type: Optional[EventType] = None, **kwargs: Any
    ) -> None:
        """Log warning message."""
        self._log(LogLevel.WARNING, message, event_type, **kwargs)

    def error(
        self,
        message: str,
        error: Optional[Exception] = None,
        event_type: Optional[EventType] = None,
        **kwargs: Any,
    ) -> None:
        """Log error message."""
        self._log(LogLevel.ERROR, message, event_type, error, **kwargs)

    def critical(
        self,
        message: str,
        error: Optional[Exception] = None,
        event_type: Optional[EventType] = None,
        **kwargs: Any,
    ) -> None:
        """Log critical message."""
        self._log(LogLevel.CRITICAL, message, event_type, error, **kwargs)

    # Convenience methods for common events

    def log_request(self, request: str, **kwargs: Any) -> None:
        """Log incoming request."""
        self.info(
            "Request received",
            event_type=EventType.REQUEST_RECEIVED,
            request=request[:100],
            **kwargs,
        )

    def log_routing(
        self, task: str, provider: Provider, reason: str, **kwargs: Any
    ) -> None:
        """Log task routing decision."""
        self.info(
            f"Routed to {provider.value}",
            event_type=EventType.TASK_ROUTED,
            task=task[:50],
            provider=provider.value,
            reason=reason,
            **kwargs,
        )

    def log_provider_call(
        self,
        provider: Provider,
        prompt_length: int,
        **kwargs: Any,
    ) -> None:
        """Log provider API call."""
        self.debug(
            f"Calling {provider.value}",
            event_type=EventType.PROVIDER_CALL,
            prompt_length=prompt_length,
            **kwargs,
        )

    def log_provider_response(
        self,
        provider: Provider,
        latency_ms: float,
        response_length: int,
        **kwargs: Any,
    ) -> None:
        """Log provider response."""
        self.debug(
            f"Response from {provider.value}",
            event_type=EventType.PROVIDER_RESPONSE,
            latency_ms=latency_ms,
            response_length=response_length,
            **kwargs,
        )

    def log_brainstorm_round(
        self,
        round_number: int,
        participants: List[Provider],
        **kwargs: Any,
    ) -> None:
        """Log brainstorm round."""
        self.info(
            f"Brainstorm round {round_number}",
            event_type=EventType.BRAINSTORM_ROUND,
            round_number=round_number,
            participants=[p.value for p in participants],
            **kwargs,
        )

    def log_consensus(
        self,
        success: bool,
        vote_breakdown: Dict[str, str],
        **kwargs: Any,
    ) -> None:
        """Log consensus result."""
        event = EventType.CONSENSUS_REACHED if success else EventType.CONSENSUS_FAILED
        self.info(
            f"Consensus {'reached' if success else 'not reached'}",
            event_type=event,
            vote_breakdown=vote_breakdown,
            **kwargs,
        )

    def enable_buffer(self) -> None:
        """Enable log buffering."""
        self._buffer_enabled = True
        self._log_buffer.clear()

    def disable_buffer(self) -> List[LogEntry]:
        """Disable log buffering and return buffered entries."""
        self._buffer_enabled = False
        entries = self._log_buffer.copy()
        self._log_buffer.clear()
        return entries


# =============================================================================
# Debug Mode
# =============================================================================


class DebugMode:
    """Context manager for enabling detailed debug tracing.

    Features:
        - Temporarily lower log level
        - Capture all log entries
        - Performance profiling
        - Call stack tracing

    Usage:
        >>> with DebugMode() as debug:
        ...     result = await orchestrator.process()
        >>> print(debug.get_logs())
        >>> print(debug.get_profile())
    """

    _active: bool = False
    _instance: Optional["DebugMode"] = None

    def __init__(
        self,
        capture_logs: bool = True,
        profile: bool = True,
        trace_calls: bool = False,
    ):
        """Initialize debug mode.

        Args:
            capture_logs: Capture all log entries
            profile: Enable performance profiling
            trace_calls: Enable call stack tracing (expensive)
        """
        self.capture_logs = capture_logs
        self.profile = profile
        self.trace_calls = trace_calls

        self._logs: List[LogEntry] = []
        self._profile_data: Dict[str, Dict[str, Any]] = {}
        self._call_traces: List[str] = []
        self._original_level: Optional[int] = None
        self._start_time: float = 0

    @classmethod
    def is_active(cls) -> bool:
        """Check if debug mode is active."""
        return cls._active

    def __enter__(self) -> "DebugMode":
        """Enter debug mode."""
        DebugMode._active = True
        DebugMode._instance = self
        self._start_time = time.time()

        # Lower log level
        root_logger = logging.getLogger()
        self._original_level = root_logger.level
        root_logger.setLevel(logging.DEBUG)

        module_logger.debug("Debug mode enabled")
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit debug mode."""
        DebugMode._active = False
        DebugMode._instance = None

        # Restore log level
        if self._original_level is not None:
            root_logger = logging.getLogger()
            root_logger.setLevel(self._original_level)

        elapsed = time.time() - self._start_time
        module_logger.debug(f"Debug mode disabled (elapsed: {elapsed:.2f}s)")

    def record_log(self, entry: LogEntry) -> None:
        """Record a log entry."""
        if self.capture_logs:
            self._logs.append(entry)

    def record_profile(self, name: str, latency_ms: float, **data: Any) -> None:
        """Record profiling data."""
        if self.profile:
            if name not in self._profile_data:
                self._profile_data[name] = {
                    "count": 0,
                    "total_ms": 0,
                    "min_ms": float("inf"),
                    "max_ms": 0,
                }
            self._profile_data[name]["count"] += 1
            self._profile_data[name]["total_ms"] += latency_ms
            self._profile_data[name]["min_ms"] = min(
                self._profile_data[name]["min_ms"], latency_ms
            )
            self._profile_data[name]["max_ms"] = max(
                self._profile_data[name]["max_ms"], latency_ms
            )
            self._profile_data[name].update(data)

    def record_call_trace(self) -> None:
        """Record current call stack."""
        if self.trace_calls:
            self._call_traces.append("".join(traceback.format_stack()))

    def get_logs(self) -> List[LogEntry]:
        """Get captured log entries."""
        return self._logs.copy()

    def get_profile(self) -> Dict[str, Dict[str, Any]]:
        """Get profiling data."""
        result = {}
        for name, data in self._profile_data.items():
            result[name] = {
                **data,
                "avg_ms": data["total_ms"] / data["count"] if data["count"] > 0 else 0,
            }
        return result

    def get_call_traces(self) -> List[str]:
        """Get recorded call traces."""
        return self._call_traces.copy()

    def summary(self) -> str:
        """Generate debug summary."""
        lines = ["Debug Mode Summary", "=" * 40]

        lines.append(f"\nLogs captured: {len(self._logs)}")

        if self._profile_data:
            lines.append("\nProfiling:")
            for name, data in self.get_profile().items():
                lines.append(
                    f"  {name}: {data['count']} calls, "
                    f"avg={data['avg_ms']:.1f}ms"
                )

        return "\n".join(lines)


def debug_trace(func: F) -> F:
    """Decorator to trace function calls in debug mode.

    Usage:
        >>> @debug_trace
        ... async def my_function():
        ...     pass
    """

    @wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        if DebugMode.is_active() and DebugMode._instance:
            DebugMode._instance.record_call_trace()
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                elapsed_ms = (time.time() - start) * 1000
                DebugMode._instance.record_profile(func.__name__, elapsed_ms)
                return result
            except Exception as e:
                elapsed_ms = (time.time() - start) * 1000
                DebugMode._instance.record_profile(
                    func.__name__, elapsed_ms, error=str(e)
                )
                raise
        return await func(*args, **kwargs)

    @wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        if DebugMode.is_active() and DebugMode._instance:
            DebugMode._instance.record_call_trace()
            start = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed_ms = (time.time() - start) * 1000
                DebugMode._instance.record_profile(func.__name__, elapsed_ms)
                return result
            except Exception as e:
                elapsed_ms = (time.time() - start) * 1000
                DebugMode._instance.record_profile(
                    func.__name__, elapsed_ms, error=str(e)
                )
                raise
        return func(*args, **kwargs)

    if asyncio.iscoroutinefunction(func):
        return async_wrapper  # type: ignore
    return sync_wrapper  # type: ignore


# =============================================================================
# Metric Exporter
# =============================================================================


class MetricExporter:
    """Export performance metrics to various formats.

    Supported formats:
        - JSON: Structured JSON output
        - Prometheus: Prometheus text format
        - InfluxDB: Line protocol format

    Usage:
        >>> exporter = MetricExporter(metrics, prefix="multivers")
        >>> print(exporter.to_prometheus())
        >>> print(exporter.to_json())
    """

    def __init__(
        self,
        metrics: PerformanceMetrics,
        prefix: str = "orchestration",
        labels: Optional[Dict[str, str]] = None,
    ):
        """Initialize metric exporter.

        Args:
            metrics: Performance metrics to export
            prefix: Metric name prefix
            labels: Common labels for all metrics
        """
        self._metrics = metrics
        self.prefix = prefix
        self.labels = labels or {}

    def _format_labels(self, extra_labels: Optional[Dict[str, str]] = None) -> str:
        """Format labels for Prometheus."""
        all_labels = {**self.labels, **(extra_labels or {})}
        if not all_labels:
            return ""
        label_str = ",".join(f'{k}="{v}"' for k, v in all_labels.items())
        return f"{{{label_str}}}"

    def to_json(self) -> str:
        """Export metrics as JSON.

        Returns:
            JSON string of all metrics
        """
        data = {
            "prefix": self.prefix,
            "labels": self.labels,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": self._metrics.to_dict(),
        }
        return json.dumps(data, indent=2)

    def to_prometheus(self) -> str:
        """Export metrics in Prometheus text format.

        Returns:
            Prometheus exposition format string
        """
        lines = []
        timestamp = int(time.time() * 1000)

        # Latency metrics by provider
        for provider, stats in self._metrics.latency_by_provider.items():
            labels = self._format_labels({"provider": provider.value})

            lines.append(
                f"# HELP {self.prefix}_latency_ms "
                f"Latency in milliseconds by provider"
            )
            lines.append(f"# TYPE {self.prefix}_latency_ms gauge")
            lines.append(
                f"{self.prefix}_latency_avg_ms{labels} {stats.avg_ms} {timestamp}"
            )
            lines.append(
                f"{self.prefix}_latency_count{labels} {stats.count} {timestamp}"
            )

        # Throughput metrics
        throughput = self._metrics.throughput
        labels = self._format_labels()

        lines.append(
            f"# HELP {self.prefix}_requests_total Total requests processed"
        )
        lines.append(f"# TYPE {self.prefix}_requests_total counter")
        lines.append(
            f"{self.prefix}_requests_total{labels} "
            f"{throughput.total_requests} {timestamp}"
        )

        lines.append(
            f"# HELP {self.prefix}_success_rate Success rate percentage"
        )
        lines.append(f"# TYPE {self.prefix}_success_rate gauge")
        lines.append(
            f"{self.prefix}_success_rate{labels} "
            f"{throughput.success_rate} {timestamp}"
        )

        # Memory metrics
        memory = self._metrics.memory
        lines.append(
            f"# HELP {self.prefix}_memory_bytes Current memory usage in bytes"
        )
        lines.append(f"# TYPE {self.prefix}_memory_bytes gauge")
        lines.append(
            f"{self.prefix}_memory_current_bytes{labels} "
            f"{memory.current_bytes} {timestamp}"
        )
        lines.append(
            f"{self.prefix}_memory_peak_bytes{labels} "
            f"{memory.peak_bytes} {timestamp}"
        )

        return "\n".join(lines)

    def to_influxdb(self, measurement: Optional[str] = None) -> str:
        """Export metrics in InfluxDB line protocol format.

        Args:
            measurement: Measurement name (defaults to prefix)

        Returns:
            InfluxDB line protocol string
        """
        measurement = measurement or self.prefix
        lines = []
        timestamp = int(time.time() * 1_000_000_000)  # nanoseconds

        # Format tags
        tags = ",".join(f"{k}={v}" for k, v in self.labels.items())
        if tags:
            tags = "," + tags

        # Latency fields
        for provider, stats in self._metrics.latency_by_provider.items():
            fields = (
                f"latency_avg_ms={stats.avg_ms},"
                f"latency_count={stats.count}i,"
                f"latency_min_ms={stats.min_ms if stats.min_ms != float('inf') else 0},"
                f"latency_max_ms={stats.max_ms}"
            )
            lines.append(
                f"{measurement},provider={provider.value}{tags} {fields} {timestamp}"
            )

        # Throughput fields
        throughput = self._metrics.throughput
        fields = (
            f"requests_total={throughput.total_requests}i,"
            f"requests_success={throughput.successful_requests}i,"
            f"requests_failed={throughput.failed_requests}i,"
            f"success_rate={throughput.success_rate}"
        )
        lines.append(f"{measurement}_throughput{tags} {fields} {timestamp}")

        # Memory fields
        memory = self._metrics.memory
        fields = (
            f"current_bytes={memory.current_bytes}i,"
            f"peak_bytes={memory.peak_bytes}i"
        )
        lines.append(f"{measurement}_memory{tags} {fields} {timestamp}")

        return "\n".join(lines)


# =============================================================================
# Event Emitter
# =============================================================================


class EventHandler(Protocol):
    """Protocol for event handlers."""

    async def handle(self, event: EventRecord) -> None:
        """Handle an event."""
        ...


class EventEmitter:
    """Emit orchestration events to handlers.

    Features:
        - Multiple handler support
        - Async event handling
        - Event filtering
        - Event history

    Usage:
        >>> emitter = EventEmitter()
        >>>
        >>> @emitter.on(EventType.TASK_COMPLETED)
        ... async def handle_completion(event):
        ...     print(f"Task completed: {event.data}")
        >>>
        >>> await emitter.emit(EventType.TASK_COMPLETED, data={"task": "auth"})
    """

    def __init__(
        self, max_history: int = 100, logger: Optional[OrchestrationLogger] = None
    ):
        """Initialize event emitter.

        Args:
            max_history: Maximum events to keep in history
            logger: Optional logger for event logging
        """
        self._handlers: Dict[EventType, List[Callable[[EventRecord], Any]]] = (
            defaultdict(list)
        )
        self._global_handlers: List[Callable[[EventRecord], Any]] = []
        self._history: List[EventRecord] = []
        self._max_history = max_history
        self._logger = logger

    def on(
        self, event_type: Optional[EventType] = None
    ) -> Callable[[Callable[[EventRecord], Any]], Callable[[EventRecord], Any]]:
        """Decorator to register an event handler.

        Args:
            event_type: Event type to handle (None for all events)

        Returns:
            Decorator function
        """

        def decorator(
            handler: Callable[[EventRecord], Any]
        ) -> Callable[[EventRecord], Any]:
            if event_type is None:
                self._global_handlers.append(handler)
            else:
                self._handlers[event_type].append(handler)
            return handler

        return decorator

    def add_handler(
        self,
        handler: Callable[[EventRecord], Any],
        event_type: Optional[EventType] = None,
    ) -> None:
        """Add an event handler.

        Args:
            handler: Handler function
            event_type: Event type to handle (None for all events)
        """
        if event_type is None:
            self._global_handlers.append(handler)
        else:
            self._handlers[event_type].append(handler)

    def remove_handler(
        self,
        handler: Callable[[EventRecord], Any],
        event_type: Optional[EventType] = None,
    ) -> bool:
        """Remove an event handler.

        Args:
            handler: Handler to remove
            event_type: Event type (None for global handlers)

        Returns:
            True if handler was removed
        """
        try:
            if event_type is None:
                self._global_handlers.remove(handler)
            else:
                self._handlers[event_type].remove(handler)
            return True
        except ValueError:
            return False

    async def emit(
        self,
        event_type: EventType,
        session_id: Optional[str] = None,
        provider: Optional[Provider] = None,
        **data: Any,
    ) -> None:
        """Emit an event.

        Args:
            event_type: Type of event
            session_id: Session identifier
            provider: Provider related to event
            **data: Event data
        """
        event = EventRecord(
            event_type=event_type,
            session_id=session_id,
            provider=provider,
            data=data,
        )

        # Add to history
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]

        # Log if logger available
        if self._logger:
            self._logger.debug(
                f"Event: {event_type.value}",
                event_type=event_type,
                **data,
            )

        # Call handlers
        handlers = self._handlers.get(event_type, []) + self._global_handlers

        for handler in handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                module_logger.error(f"Event handler error: {e}")

    def get_history(
        self,
        event_type: Optional[EventType] = None,
        limit: int = 50,
    ) -> List[EventRecord]:
        """Get event history.

        Args:
            event_type: Filter by event type
            limit: Maximum events to return

        Returns:
            List of event records
        """
        events = self._history
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return events[-limit:]

    def clear_history(self) -> None:
        """Clear event history."""
        self._history.clear()


# =============================================================================
# Monitoring Dashboard Data
# =============================================================================


@dataclass
class DashboardData:
    """Data structure for monitoring dashboard."""

    # Session info
    session_id: Optional[str] = None
    start_time: Optional[float] = None
    status: str = "idle"

    # Current activity
    current_operation: Optional[str] = None
    current_provider: Optional[Provider] = None

    # Metrics
    requests_processed: int = 0
    success_rate: float = 0.0
    avg_latency_ms: float = 0.0

    # Provider status
    provider_status: Dict[str, str] = field(default_factory=dict)

    # Recent events
    recent_events: List[Dict[str, Any]] = field(default_factory=list)

    # Alerts
    alerts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "status": self.status,
            "current_operation": self.current_operation,
            "current_provider": (
                self.current_provider.value if self.current_provider else None
            ),
            "requests_processed": self.requests_processed,
            "success_rate": self.success_rate,
            "avg_latency_ms": self.avg_latency_ms,
            "provider_status": self.provider_status,
            "recent_events": self.recent_events,
            "alerts": self.alerts,
        }


class MonitoringDashboard:
    """Aggregate monitoring data for dashboard display.

    Usage:
        >>> dashboard = MonitoringDashboard(metrics, emitter)
        >>> data = dashboard.get_data()
        >>> print(json.dumps(data.to_dict(), indent=2))
    """

    def __init__(
        self,
        metrics: PerformanceMetrics,
        emitter: Optional[EventEmitter] = None,
    ):
        """Initialize dashboard.

        Args:
            metrics: Performance metrics
            emitter: Event emitter for recent events
        """
        self._metrics = metrics
        self._emitter = emitter
        self._session_id: Optional[str] = None
        self._start_time: Optional[float] = None
        self._alerts: List[str] = []

    def set_session(self, session_id: str) -> None:
        """Set current session."""
        self._session_id = session_id
        self._start_time = time.time()

    def add_alert(self, message: str) -> None:
        """Add an alert."""
        self._alerts.append(message)
        if len(self._alerts) > 10:
            self._alerts = self._alerts[-10:]

    def clear_alerts(self) -> None:
        """Clear all alerts."""
        self._alerts.clear()

    def get_data(self) -> DashboardData:
        """Get current dashboard data."""
        # Calculate average latency across all providers
        total_latency = 0.0
        total_count = 0
        for stats in self._metrics.latency_by_provider.values():
            total_latency += stats.total_ms
            total_count += stats.count
        avg_latency = total_latency / total_count if total_count > 0 else 0

        # Get recent events
        recent = []
        if self._emitter:
            for event in self._emitter.get_history(limit=10):
                recent.append(event.to_dict())

        return DashboardData(
            session_id=self._session_id,
            start_time=self._start_time,
            status="active" if self._session_id else "idle",
            requests_processed=self._metrics.throughput.total_requests,
            success_rate=self._metrics.throughput.success_rate,
            avg_latency_ms=avg_latency,
            recent_events=recent,
            alerts=self._alerts.copy(),
        )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Log levels and events
    "LogLevel",
    "EventType",
    # Log entries
    "LogEntry",
    "EventRecord",
    # Logging
    "OrchestrationLogger",
    # Debug
    "DebugMode",
    "debug_trace",
    # Export
    "MetricExporter",
    # Events
    "EventHandler",
    "EventEmitter",
    # Dashboard
    "DashboardData",
    "MonitoringDashboard",
]
