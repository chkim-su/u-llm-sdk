"""Performance Optimization Module for Multi-LLM Orchestration.

This module provides utilities for optimizing multi-LLM orchestration performance:
- Parallel execution with concurrency control
- Latency tracking and optimization
- Memory management for large discussion logs
- Connection pooling and resource management

Key Components:
    - ParallelExecutor: Concurrent task execution with limits
    - LatencyTracker: Provider-specific latency monitoring
    - MemoryManager: Discussion log memory optimization
    - ResourcePool: Connection and resource pooling
    - PerformanceMetrics: Aggregated performance data

Usage:
    >>> from u_llm_sdk.multi_llm.performance import (
    ...     ParallelExecutor,
    ...     LatencyTracker,
    ...     PerformanceMetrics,
    ... )
    >>>
    >>> # Parallel execution with concurrency limit
    >>> executor = ParallelExecutor(max_concurrency=3)
    >>> results = await executor.execute([task1, task2, task3])
    >>>
    >>> # Track latency across providers
    >>> tracker = LatencyTracker()
    >>> async with tracker.measure(Provider.CLAUDE):
    ...     result = await claude.run(prompt)
    >>> print(tracker.get_stats(Provider.CLAUDE))
"""

import asyncio
import logging
import sys
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
)

from u_llm_sdk.types import Provider

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


# =============================================================================
# Performance Metrics Data Classes
# =============================================================================


class MetricType(Enum):
    """Types of performance metrics."""

    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    MEMORY_USAGE = "memory_usage"
    QUEUE_DEPTH = "queue_depth"


@dataclass
class LatencyStats:
    """Statistics for latency measurements."""

    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0
    last_ms: float = 0.0

    @property
    def avg_ms(self) -> float:
        """Average latency in milliseconds."""
        return self.total_ms / self.count if self.count > 0 else 0.0

    def record(self, latency_ms: float) -> None:
        """Record a latency measurement."""
        self.count += 1
        self.total_ms += latency_ms
        self.min_ms = min(self.min_ms, latency_ms)
        self.max_ms = max(self.max_ms, latency_ms)
        self.last_ms = latency_ms

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "count": self.count,
            "avg_ms": round(self.avg_ms, 2),
            "min_ms": round(self.min_ms, 2) if self.min_ms != float("inf") else None,
            "max_ms": round(self.max_ms, 2),
            "last_ms": round(self.last_ms, 2),
        }


@dataclass
class ThroughputStats:
    """Statistics for throughput measurements."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def success_rate(self) -> float:
        """Success rate as a percentage."""
        return (
            (self.successful_requests / self.total_requests * 100)
            if self.total_requests > 0
            else 0.0
        )

    @property
    def requests_per_second(self) -> float:
        """Average requests per second."""
        elapsed = time.time() - self.start_time
        return self.total_requests / elapsed if elapsed > 0 else 0.0

    def record_success(self) -> None:
        """Record a successful request."""
        self.total_requests += 1
        self.successful_requests += 1

    def record_failure(self) -> None:
        """Record a failed request."""
        self.total_requests += 1
        self.failed_requests += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": round(self.success_rate, 2),
            "requests_per_second": round(self.requests_per_second, 4),
        }


@dataclass
class MemoryStats:
    """Statistics for memory usage."""

    peak_bytes: int = 0
    current_bytes: int = 0
    allocation_count: int = 0

    def record_allocation(self, size_bytes: int) -> None:
        """Record memory allocation."""
        self.current_bytes += size_bytes
        self.peak_bytes = max(self.peak_bytes, self.current_bytes)
        self.allocation_count += 1

    def record_deallocation(self, size_bytes: int) -> None:
        """Record memory deallocation."""
        self.current_bytes = max(0, self.current_bytes - size_bytes)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "peak_mb": round(self.peak_bytes / (1024 * 1024), 2),
            "current_mb": round(self.current_bytes / (1024 * 1024), 2),
            "allocation_count": self.allocation_count,
        }


@dataclass
class PerformanceMetrics:
    """Aggregated performance metrics for multi-LLM orchestration."""

    latency_by_provider: Dict[Provider, LatencyStats] = field(default_factory=dict)
    latency_by_operation: Dict[str, LatencyStats] = field(default_factory=dict)
    throughput: ThroughputStats = field(default_factory=ThroughputStats)
    memory: MemoryStats = field(default_factory=MemoryStats)
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

    def get_provider_latency(self, provider: Provider) -> LatencyStats:
        """Get or create latency stats for a provider."""
        if provider not in self.latency_by_provider:
            self.latency_by_provider[provider] = LatencyStats()
        return self.latency_by_provider[provider]

    def get_operation_latency(self, operation: str) -> LatencyStats:
        """Get or create latency stats for an operation."""
        if operation not in self.latency_by_operation:
            self.latency_by_operation[operation] = LatencyStats()
        return self.latency_by_operation[operation]

    def to_dict(self) -> Dict[str, Any]:
        """Convert all metrics to dictionary."""
        return {
            "latency_by_provider": {
                p.value: s.to_dict() for p, s in self.latency_by_provider.items()
            },
            "latency_by_operation": {
                op: s.to_dict() for op, s in self.latency_by_operation.items()
            },
            "throughput": self.throughput.to_dict(),
            "memory": self.memory.to_dict(),
            "custom": self.custom_metrics,
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = ["Performance Metrics Summary", "=" * 40]

        if self.latency_by_provider:
            lines.append("\nLatency by Provider:")
            for provider, stats in self.latency_by_provider.items():
                lines.append(
                    f"  {provider.value}: avg={stats.avg_ms:.1f}ms, "
                    f"count={stats.count}"
                )

        lines.append(f"\nThroughput: {self.throughput.success_rate:.1f}% success")
        lines.append(
            f"Memory: {self.memory.current_bytes / (1024*1024):.2f}MB current"
        )

        return "\n".join(lines)


# =============================================================================
# Latency Tracker
# =============================================================================


class LatencyTracker:
    """Track latency for providers and operations.

    Usage:
        >>> tracker = LatencyTracker()
        >>> async with tracker.measure(Provider.CLAUDE):
        ...     result = await claude.run(prompt)
        >>> print(tracker.get_stats(Provider.CLAUDE).avg_ms)
    """

    def __init__(self, metrics: Optional[PerformanceMetrics] = None):
        """Initialize latency tracker.

        Args:
            metrics: Optional shared metrics instance
        """
        self._metrics = metrics or PerformanceMetrics()
        self._active_measurements: Dict[str, float] = {}

    @property
    def metrics(self) -> PerformanceMetrics:
        """Get underlying metrics."""
        return self._metrics

    @asynccontextmanager
    async def measure(
        self, provider: Optional[Provider] = None, operation: Optional[str] = None
    ) -> AsyncIterator[None]:
        """Context manager for measuring latency.

        Args:
            provider: Provider to track (optional)
            operation: Operation name to track (optional)

        Example:
            >>> async with tracker.measure(Provider.CLAUDE, "brainstorm"):
            ...     result = await run_brainstorm()
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000

            if provider:
                self._metrics.get_provider_latency(provider).record(elapsed_ms)

            if operation:
                self._metrics.get_operation_latency(operation).record(elapsed_ms)

            logger.debug(
                f"Latency: provider={provider}, operation={operation}, "
                f"elapsed={elapsed_ms:.1f}ms"
            )

    def get_stats(self, provider: Provider) -> LatencyStats:
        """Get latency stats for a provider."""
        return self._metrics.get_provider_latency(provider)

    def get_operation_stats(self, operation: str) -> LatencyStats:
        """Get latency stats for an operation."""
        return self._metrics.get_operation_latency(operation)

    def record(
        self,
        latency_ms: float,
        provider: Optional[Provider] = None,
        operation: Optional[str] = None,
    ) -> None:
        """Manually record a latency measurement.

        Args:
            latency_ms: Latency in milliseconds
            provider: Provider to track
            operation: Operation name to track
        """
        if provider:
            self._metrics.get_provider_latency(provider).record(latency_ms)
        if operation:
            self._metrics.get_operation_latency(operation).record(latency_ms)


# =============================================================================
# Parallel Executor
# =============================================================================


@dataclass
class TaskResult(Generic[T]):
    """Result of a parallel task execution."""

    index: int
    success: bool
    result: Optional[T] = None
    error: Optional[Exception] = None
    latency_ms: float = 0.0


class ParallelExecutor:
    """Execute multiple tasks in parallel with concurrency control.

    Features:
        - Configurable concurrency limit
        - Timeout per task
        - Error handling (fail-fast or continue)
        - Latency tracking per task

    Usage:
        >>> executor = ParallelExecutor(max_concurrency=3, timeout_seconds=30)
        >>>
        >>> async def process(x):
        ...     return x * 2
        >>>
        >>> tasks = [lambda: process(i) for i in range(10)]
        >>> results = await executor.execute(tasks)
    """

    def __init__(
        self,
        max_concurrency: int = 5,
        timeout_seconds: float = 60.0,
        fail_fast: bool = False,
        tracker: Optional[LatencyTracker] = None,
    ):
        """Initialize parallel executor.

        Args:
            max_concurrency: Maximum concurrent tasks
            timeout_seconds: Timeout per task in seconds
            fail_fast: If True, cancel remaining tasks on first failure
            tracker: Optional latency tracker
        """
        self.max_concurrency = max_concurrency
        self.timeout_seconds = timeout_seconds
        self.fail_fast = fail_fast
        self._tracker = tracker
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._active_tasks: Set[asyncio.Task] = set()

    async def execute(
        self,
        tasks: List[Callable[[], Awaitable[T]]],
        operation_name: Optional[str] = None,
    ) -> List[TaskResult[T]]:
        """Execute tasks in parallel with concurrency control.

        Args:
            tasks: List of async callables to execute
            operation_name: Optional operation name for tracking

        Returns:
            List of TaskResult objects in original order
        """
        if not tasks:
            return []

        results: List[TaskResult[T]] = [
            TaskResult(index=i, success=False) for i in range(len(tasks))
        ]

        async def run_task(index: int, task: Callable[[], Awaitable[T]]) -> None:
            start = time.perf_counter()
            async with self._semaphore:
                try:
                    result = await asyncio.wait_for(
                        task(), timeout=self.timeout_seconds
                    )
                    elapsed_ms = (time.perf_counter() - start) * 1000

                    results[index] = TaskResult(
                        index=index,
                        success=True,
                        result=result,
                        latency_ms=elapsed_ms,
                    )

                    if self._tracker and operation_name:
                        self._tracker.record(elapsed_ms, operation=operation_name)

                except asyncio.TimeoutError:
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    results[index] = TaskResult(
                        index=index,
                        success=False,
                        error=TimeoutError(
                            f"Task {index} timed out after {self.timeout_seconds}s"
                        ),
                        latency_ms=elapsed_ms,
                    )
                    logger.warning(f"Task {index} timed out after {self.timeout_seconds}s")

                except Exception as e:
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    results[index] = TaskResult(
                        index=index,
                        success=False,
                        error=e,
                        latency_ms=elapsed_ms,
                    )
                    logger.error(f"Task {index} failed: {e}")

        # Create all tasks
        async_tasks = [
            asyncio.create_task(run_task(i, task)) for i, task in enumerate(tasks)
        ]
        self._active_tasks.update(async_tasks)

        try:
            if self.fail_fast:
                # Cancel on first failure
                done, pending = await asyncio.wait(
                    async_tasks, return_when=asyncio.FIRST_EXCEPTION
                )
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            else:
                # Wait for all tasks
                await asyncio.gather(*async_tasks, return_exceptions=True)
        finally:
            self._active_tasks -= set(async_tasks)

        return results

    async def execute_with_providers(
        self,
        tasks: Dict[Provider, Callable[[], Awaitable[T]]],
    ) -> Dict[Provider, TaskResult[T]]:
        """Execute tasks mapped to providers.

        Args:
            tasks: Dictionary mapping providers to async callables

        Returns:
            Dictionary mapping providers to results
        """
        providers = list(tasks.keys())
        callables = [tasks[p] for p in providers]

        results = await self.execute(callables)

        return {provider: results[i] for i, provider in enumerate(providers)}

    async def cancel_active(self) -> int:
        """Cancel all active tasks.

        Returns:
            Number of tasks cancelled
        """
        cancelled = 0
        for task in list(self._active_tasks):
            if not task.done():
                task.cancel()
                cancelled += 1

        return cancelled


# =============================================================================
# Memory Manager
# =============================================================================


class MemoryManager:
    """Manage memory for large discussion logs and session data.

    Features:
        - Track memory allocation/deallocation
        - Implement discussion log compaction (without summarization)
        - Memory pressure warnings
        - Garbage collection hints

    Note: This manager tracks LOGICAL memory usage, not actual Python
    memory. Use for understanding data structure sizes.
    """

    def __init__(
        self,
        max_memory_mb: float = 512.0,
        warning_threshold: float = 0.8,
        metrics: Optional[PerformanceMetrics] = None,
    ):
        """Initialize memory manager.

        Args:
            max_memory_mb: Maximum memory budget in megabytes
            warning_threshold: Warn when usage exceeds this fraction
            metrics: Optional shared metrics instance
        """
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self.warning_threshold = warning_threshold
        self._metrics = metrics or PerformanceMetrics()
        self._allocations: Dict[str, int] = {}

    @property
    def current_usage_bytes(self) -> int:
        """Get current memory usage in bytes."""
        return self._metrics.memory.current_bytes

    @property
    def usage_ratio(self) -> float:
        """Get current usage as fraction of max."""
        return self.current_usage_bytes / self.max_memory_bytes

    def allocate(self, key: str, size_bytes: int) -> bool:
        """Allocate memory for a data structure.

        Args:
            key: Unique identifier for the allocation
            size_bytes: Size in bytes

        Returns:
            True if allocation succeeded, False if would exceed limit
        """
        if self.current_usage_bytes + size_bytes > self.max_memory_bytes:
            logger.warning(
                f"Memory allocation denied: {size_bytes} bytes would exceed "
                f"limit of {self.max_memory_bytes} bytes"
            )
            return False

        if key in self._allocations:
            # Update existing allocation
            old_size = self._allocations[key]
            self._metrics.memory.record_deallocation(old_size)

        self._allocations[key] = size_bytes
        self._metrics.memory.record_allocation(size_bytes)

        if self.usage_ratio > self.warning_threshold:
            logger.warning(
                f"Memory usage at {self.usage_ratio:.1%}, "
                f"threshold is {self.warning_threshold:.1%}"
            )

        return True

    def deallocate(self, key: str) -> int:
        """Deallocate memory for a data structure.

        Args:
            key: Unique identifier for the allocation

        Returns:
            Size deallocated in bytes, 0 if key not found
        """
        if key not in self._allocations:
            return 0

        size = self._allocations.pop(key)
        self._metrics.memory.record_deallocation(size)
        return size

    def estimate_discussion_size(
        self, entry_count: int, avg_message_length: int = 500
    ) -> int:
        """Estimate memory size for discussion entries.

        Args:
            entry_count: Number of discussion entries
            avg_message_length: Average message length in characters

        Returns:
            Estimated size in bytes
        """
        # Rough estimate: ~2 bytes per char + object overhead
        return entry_count * (avg_message_length * 2 + 200)

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            **self._metrics.memory.to_dict(),
            "max_mb": round(self.max_memory_bytes / (1024 * 1024), 2),
            "usage_ratio": round(self.usage_ratio, 4),
            "active_allocations": len(self._allocations),
        }


# =============================================================================
# Resource Pool
# =============================================================================


@dataclass
class PooledResource(Generic[T]):
    """A pooled resource with usage tracking."""

    resource: T
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    use_count: int = 0

    def mark_used(self) -> None:
        """Mark resource as used."""
        self.last_used = time.time()
        self.use_count += 1


class ResourcePool(Generic[T]):
    """Generic resource pool for connections and heavy objects.

    Features:
        - Configurable pool size
        - Resource expiry
        - Usage tracking

    Usage:
        >>> pool = ResourcePool(create_func=create_connection, max_size=5)
        >>> async with pool.acquire() as conn:
        ...     await conn.execute(query)
    """

    def __init__(
        self,
        create_func: Callable[[], Awaitable[T]],
        max_size: int = 10,
        max_idle_seconds: float = 300.0,
        destroy_func: Optional[Callable[[T], Awaitable[None]]] = None,
    ):
        """Initialize resource pool.

        Args:
            create_func: Async function to create a resource
            max_size: Maximum pool size
            max_idle_seconds: Expire resources idle longer than this
            destroy_func: Optional async function to destroy a resource
        """
        self._create_func = create_func
        self._destroy_func = destroy_func
        self.max_size = max_size
        self.max_idle_seconds = max_idle_seconds
        self._available: List[PooledResource[T]] = []
        self._in_use: Set[PooledResource[T]] = set()
        self._lock = asyncio.Lock()

    @property
    def size(self) -> int:
        """Current pool size."""
        return len(self._available) + len(self._in_use)

    @property
    def available_count(self) -> int:
        """Number of available resources."""
        return len(self._available)

    async def _create_resource(self) -> PooledResource[T]:
        """Create a new pooled resource."""
        resource = await self._create_func()
        return PooledResource(resource=resource)

    async def _destroy_resource(self, pooled: PooledResource[T]) -> None:
        """Destroy a pooled resource."""
        if self._destroy_func:
            try:
                await self._destroy_func(pooled.resource)
            except Exception as e:
                logger.warning(f"Error destroying resource: {e}")

    async def _cleanup_expired(self) -> int:
        """Remove expired resources from pool."""
        now = time.time()
        expired = []
        remaining = []

        for pooled in self._available:
            if now - pooled.last_used > self.max_idle_seconds:
                expired.append(pooled)
            else:
                remaining.append(pooled)

        self._available = remaining

        for pooled in expired:
            await self._destroy_resource(pooled)

        return len(expired)

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[T]:
        """Acquire a resource from the pool.

        Yields:
            The pooled resource
        """
        async with self._lock:
            # Cleanup expired resources
            await self._cleanup_expired()

            # Try to get an available resource
            if self._available:
                pooled = self._available.pop()
            elif self.size < self.max_size:
                # Create new resource
                pooled = await self._create_resource()
            else:
                # Pool exhausted - wait for a resource
                # In production, implement proper waiting queue
                raise RuntimeError("Resource pool exhausted")

            self._in_use.add(pooled)

        pooled.mark_used()

        try:
            yield pooled.resource
        finally:
            async with self._lock:
                self._in_use.discard(pooled)
                self._available.append(pooled)

    async def close(self) -> None:
        """Close the pool and destroy all resources."""
        async with self._lock:
            for pooled in self._available:
                await self._destroy_resource(pooled)
            self._available.clear()

            # Note: In-use resources will be cleaned up when released

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            "size": self.size,
            "available": self.available_count,
            "in_use": len(self._in_use),
            "max_size": self.max_size,
        }


# =============================================================================
# Performance Optimizer
# =============================================================================


class PerformanceOptimizer:
    """Coordinator for all performance optimization components.

    Provides a unified interface for:
        - Parallel execution
        - Latency tracking
        - Memory management
        - Performance reporting

    Usage:
        >>> optimizer = PerformanceOptimizer()
        >>>
        >>> # Track latency
        >>> async with optimizer.track_latency(Provider.CLAUDE):
        ...     result = await claude.run(prompt)
        >>>
        >>> # Execute in parallel
        >>> results = await optimizer.parallel_execute(tasks)
        >>>
        >>> # Get metrics
        >>> print(optimizer.get_metrics().summary())
    """

    def __init__(
        self,
        max_concurrency: int = 5,
        max_memory_mb: float = 512.0,
        timeout_seconds: float = 60.0,
    ):
        """Initialize performance optimizer.

        Args:
            max_concurrency: Maximum concurrent tasks
            max_memory_mb: Maximum memory budget in MB
            timeout_seconds: Default timeout for tasks
        """
        self._metrics = PerformanceMetrics()
        self._tracker = LatencyTracker(self._metrics)
        self._memory = MemoryManager(max_memory_mb, metrics=self._metrics)
        self._executor = ParallelExecutor(
            max_concurrency=max_concurrency,
            timeout_seconds=timeout_seconds,
            tracker=self._tracker,
        )

    @property
    def metrics(self) -> PerformanceMetrics:
        """Get performance metrics."""
        return self._metrics

    @property
    def tracker(self) -> LatencyTracker:
        """Get latency tracker."""
        return self._tracker

    @property
    def memory(self) -> MemoryManager:
        """Get memory manager."""
        return self._memory

    @property
    def executor(self) -> ParallelExecutor:
        """Get parallel executor."""
        return self._executor

    @asynccontextmanager
    async def track_latency(
        self, provider: Optional[Provider] = None, operation: Optional[str] = None
    ) -> AsyncIterator[None]:
        """Track latency for an operation.

        Args:
            provider: Provider to track
            operation: Operation name to track
        """
        async with self._tracker.measure(provider, operation):
            yield

    async def parallel_execute(
        self,
        tasks: List[Callable[[], Awaitable[T]]],
        operation_name: Optional[str] = None,
    ) -> List[TaskResult[T]]:
        """Execute tasks in parallel.

        Args:
            tasks: List of async callables
            operation_name: Optional operation name for tracking

        Returns:
            List of TaskResult objects
        """
        return await self._executor.execute(tasks, operation_name)

    def record_success(self) -> None:
        """Record a successful operation."""
        self._metrics.throughput.record_success()

    def record_failure(self) -> None:
        """Record a failed operation."""
        self._metrics.throughput.record_failure()

    def get_metrics(self) -> PerformanceMetrics:
        """Get all performance metrics."""
        return self._metrics

    def get_summary(self) -> str:
        """Get human-readable metrics summary."""
        return self._metrics.summary()

    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self._metrics = PerformanceMetrics()
        self._tracker = LatencyTracker(self._metrics)


# =============================================================================
# Batch Processor
# =============================================================================


class BatchProcessor(Generic[T, R]):
    """Process items in batches for improved throughput.

    Features:
        - Configurable batch size
        - Automatic batching with timeout
        - Parallel batch processing

    Usage:
        >>> processor = BatchProcessor(
        ...     process_batch=lambda items: [item * 2 for item in items],
        ...     batch_size=10,
        ... )
        >>> results = await processor.process([1, 2, 3, 4, 5])
    """

    def __init__(
        self,
        process_batch: Callable[[List[T]], Awaitable[List[R]]],
        batch_size: int = 10,
        max_wait_seconds: float = 1.0,
    ):
        """Initialize batch processor.

        Args:
            process_batch: Async function to process a batch
            batch_size: Maximum items per batch
            max_wait_seconds: Maximum time to wait for batch to fill
        """
        self._process_batch = process_batch
        self.batch_size = batch_size
        self.max_wait_seconds = max_wait_seconds
        self._pending: List[Tuple[T, asyncio.Future[R]]] = []
        self._lock = asyncio.Lock()
        self._batch_task: Optional[asyncio.Task] = None

    async def process(self, items: List[T]) -> List[R]:
        """Process items, automatically batching them.

        Args:
            items: Items to process

        Returns:
            List of results in same order as input
        """
        if not items:
            return []

        # Create futures for all items
        futures: List[asyncio.Future[R]] = []

        async with self._lock:
            for item in items:
                future: asyncio.Future[R] = asyncio.get_event_loop().create_future()
                futures.append(future)
                self._pending.append((item, future))

            # Check if we should process immediately
            if len(self._pending) >= self.batch_size:
                await self._process_pending_batch()
            elif self._batch_task is None:
                # Start timeout task
                self._batch_task = asyncio.create_task(self._timeout_batch())

        # Wait for all results
        return await asyncio.gather(*futures)

    async def _process_pending_batch(self) -> None:
        """Process pending items as a batch."""
        if not self._pending:
            return

        # Take up to batch_size items
        batch_items = self._pending[: self.batch_size]
        self._pending = self._pending[self.batch_size :]

        items = [item for item, _ in batch_items]
        futures = [future for _, future in batch_items]

        try:
            results = await self._process_batch(items)

            for future, result in zip(futures, results):
                if not future.done():
                    future.set_result(result)

        except Exception as e:
            for future in futures:
                if not future.done():
                    future.set_exception(e)

    async def _timeout_batch(self) -> None:
        """Process batch after timeout."""
        await asyncio.sleep(self.max_wait_seconds)

        async with self._lock:
            self._batch_task = None
            await self._process_pending_batch()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Metrics
    "MetricType",
    "LatencyStats",
    "ThroughputStats",
    "MemoryStats",
    "PerformanceMetrics",
    # Tracking
    "LatencyTracker",
    # Execution
    "TaskResult",
    "ParallelExecutor",
    # Memory
    "MemoryManager",
    # Resources
    "PooledResource",
    "ResourcePool",
    # Optimization
    "PerformanceOptimizer",
    # Batch processing
    "BatchProcessor",
]
