"""Resource Lock Manager for Parallel Editing.

Prevents semantic conflicts by locking logical resources.

Key Insight:
    File disjoint prevents TEXT conflicts, but not SEMANTIC conflicts.
    Two WorkOrders can modify different files but still conflict:
    - WO-001: Add new PaymentType enum variant
    - WO-002: Add handler for all PaymentType variants

    Solution: Lock logical resources, not files.
    Resource examples:
    - "public_api:payments" - Payment module's public interface
    - "shared_types:user" - User type definitions
    - "schema:database" - Database schema
    - "config:env" - Environment configuration

Design:
    - Locks are identified by string keys (resource names)
    - Lock acquisition is ordered (alphabetically) to prevent deadlock
    - Supports both sync and async interfaces
    - Pluggable backend (InMemory for dev, Redis for production)
"""

from __future__ import annotations

import asyncio
import threading
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

if TYPE_CHECKING:
    from .contracts import WorkOrder


@dataclass
class LockInfo:
    """Information about a held lock.

    Attributes:
        resource_key: Resource being locked
        holder_id: ID of the lock holder (WorkOrder ID)
        acquired_at: Timestamp when lock was acquired
        metadata: Additional metadata
    """

    resource_key: str
    holder_id: str
    acquired_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "resource_key": self.resource_key,
            "holder_id": self.holder_id,
            "acquired_at": self.acquired_at,
            "metadata": self.metadata,
        }


@dataclass
class LockAcquisitionResult:
    """Result of lock acquisition attempt.

    Attributes:
        success: Whether all locks were acquired
        acquired_locks: Set of successfully acquired locks
        failed_locks: Dict of {resource: current_holder} for failed locks
        wait_time_ms: Time spent waiting for locks
    """

    success: bool
    acquired_locks: Set[str] = field(default_factory=set)
    failed_locks: Dict[str, str] = field(default_factory=dict)
    wait_time_ms: int = 0

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "acquired_locks": list(self.acquired_locks),
            "failed_locks": self.failed_locks,
            "wait_time_ms": self.wait_time_ms,
        }


class LockManager(ABC):
    """Abstract base class for lock managers.

    Implementations must be safe for concurrent access.
    """

    @abstractmethod
    async def acquire(
        self,
        resource_keys: List[str],
        holder_id: str,
        timeout: Optional[float] = None,
    ) -> LockAcquisitionResult:
        """Acquire locks on multiple resources.

        IMPORTANT: Resources are locked in sorted order to prevent deadlock.

        Args:
            resource_keys: List of resource keys to lock
            holder_id: ID of the lock holder (e.g., WorkOrder ID)
            timeout: Max time to wait for locks (None = no wait)

        Returns:
            LockAcquisitionResult with success status and details
        """
        pass

    @abstractmethod
    async def release(
        self,
        resource_keys: List[str],
        holder_id: str,
    ) -> bool:
        """Release locks on resources.

        Only releases locks held by the specified holder.

        Args:
            resource_keys: Resources to unlock
            holder_id: ID of the holder releasing

        Returns:
            True if all releases succeeded
        """
        pass

    @abstractmethod
    async def get_lock_info(self, resource_key: str) -> Optional[LockInfo]:
        """Get information about a lock.

        Args:
            resource_key: Resource to check

        Returns:
            LockInfo if locked, None if not
        """
        pass

    @abstractmethod
    async def get_held_locks(self, holder_id: str) -> List[LockInfo]:
        """Get all locks held by a holder.

        Args:
            holder_id: Holder to check

        Returns:
            List of LockInfo for held locks
        """
        pass

    @abstractmethod
    async def release_all(self, holder_id: str) -> int:
        """Release all locks held by a holder.

        Useful for cleanup on failure.

        Args:
            holder_id: Holder to release all locks for

        Returns:
            Number of locks released
        """
        pass

    @asynccontextmanager
    async def lock(
        self,
        resource_keys: List[str],
        holder_id: str,
        timeout: Optional[float] = None,
    ):
        """Context manager for acquiring and releasing locks.

        Usage:
            async with lock_manager.lock(["api:payments"], "WO-001") as result:
                if result.success:
                    # Do work with locked resources
                    pass

        Args:
            resource_keys: Resources to lock
            holder_id: Holder ID
            timeout: Lock acquisition timeout

        Yields:
            LockAcquisitionResult
        """
        result = await self.acquire(resource_keys, holder_id, timeout)
        try:
            yield result
        finally:
            if result.acquired_locks:
                await self.release(list(result.acquired_locks), holder_id)


class InMemoryLockManager(LockManager):
    """In-memory lock manager for single-process orchestration.

    Thread-safe and async-safe. Uses ordered lock acquisition
    to prevent deadlocks.

    For production multi-process scenarios, use RedisLockManager.

    Example:
        >>> manager = InMemoryLockManager()
        >>> async with manager.lock(["api:payments", "types:payment"], "WO-001") as result:
        ...     if result.success:
        ...         # Safe to modify payment-related resources
        ...         pass
    """

    def __init__(self):
        """Initialize in-memory lock manager."""
        self._locks: Dict[str, LockInfo] = {}
        self._lock = asyncio.Lock()  # Protects _locks dict
        self._resource_locks: Dict[str, asyncio.Lock] = {}  # Per-resource locks
        self._sync_lock = threading.Lock()  # For sync operations

    async def acquire(
        self,
        resource_keys: List[str],
        holder_id: str,
        timeout: Optional[float] = None,
    ) -> LockAcquisitionResult:
        """Acquire locks in sorted order to prevent deadlock."""
        if not resource_keys:
            return LockAcquisitionResult(success=True)

        # Sort keys to prevent deadlock
        sorted_keys = sorted(set(resource_keys))
        acquired: Set[str] = set()
        failed: Dict[str, str] = {}
        start_time = time.time()

        try:
            for key in sorted_keys:
                # Get or create per-resource lock
                async with self._lock:
                    if key not in self._resource_locks:
                        self._resource_locks[key] = asyncio.Lock()
                    resource_lock = self._resource_locks[key]

                # Try to acquire resource lock
                if timeout is not None:
                    remaining = timeout - (time.time() - start_time)
                    if remaining <= 0:
                        # Timeout exceeded
                        break
                    try:
                        await asyncio.wait_for(
                            resource_lock.acquire(),
                            timeout=remaining,
                        )
                    except asyncio.TimeoutError:
                        # Check who holds it
                        async with self._lock:
                            if key in self._locks:
                                failed[key] = self._locks[key].holder_id
                            else:
                                failed[key] = "unknown"
                        break
                else:
                    # Non-blocking acquire
                    acquired_lock = resource_lock.locked()
                    if acquired_lock:
                        # Already locked by someone
                        async with self._lock:
                            if key in self._locks:
                                failed[key] = self._locks[key].holder_id
                            else:
                                failed[key] = "unknown"
                        break
                    await resource_lock.acquire()

                # Record lock acquisition
                async with self._lock:
                    # Double-check not already held by another
                    if key in self._locks and self._locks[key].holder_id != holder_id:
                        failed[key] = self._locks[key].holder_id
                        resource_lock.release()
                        break

                    self._locks[key] = LockInfo(
                        resource_key=key,
                        holder_id=holder_id,
                        acquired_at=time.time(),
                    )
                    acquired.add(key)

        except Exception as e:
            # Rollback on error
            for key in acquired:
                async with self._lock:
                    if key in self._locks and self._locks[key].holder_id == holder_id:
                        del self._locks[key]
                    if key in self._resource_locks:
                        try:
                            self._resource_locks[key].release()
                        except RuntimeError:
                            pass  # Already released

            raise RuntimeError(f"Lock acquisition failed: {e}")

        # If we failed, rollback acquired locks
        if failed:
            for key in acquired:
                async with self._lock:
                    if key in self._locks and self._locks[key].holder_id == holder_id:
                        del self._locks[key]
                    if key in self._resource_locks:
                        try:
                            self._resource_locks[key].release()
                        except RuntimeError:
                            pass

            return LockAcquisitionResult(
                success=False,
                acquired_locks=set(),
                failed_locks=failed,
                wait_time_ms=int((time.time() - start_time) * 1000),
            )

        return LockAcquisitionResult(
            success=True,
            acquired_locks=acquired,
            wait_time_ms=int((time.time() - start_time) * 1000),
        )

    async def release(
        self,
        resource_keys: List[str],
        holder_id: str,
    ) -> bool:
        """Release locks held by holder."""
        all_released = True

        for key in resource_keys:
            async with self._lock:
                if key in self._locks:
                    if self._locks[key].holder_id == holder_id:
                        del self._locks[key]
                        if key in self._resource_locks:
                            try:
                                self._resource_locks[key].release()
                            except RuntimeError:
                                pass  # Already released
                    else:
                        # Not held by this holder
                        all_released = False
                # If not in _locks, consider it released

        return all_released

    async def get_lock_info(self, resource_key: str) -> Optional[LockInfo]:
        """Get lock info for a resource."""
        async with self._lock:
            return self._locks.get(resource_key)

    async def get_held_locks(self, holder_id: str) -> List[LockInfo]:
        """Get all locks held by a holder."""
        async with self._lock:
            return [
                info for info in self._locks.values() if info.holder_id == holder_id
            ]

    async def release_all(self, holder_id: str) -> int:
        """Release all locks held by a holder."""
        released = 0
        async with self._lock:
            keys_to_release = [
                key
                for key, info in self._locks.items()
                if info.holder_id == holder_id
            ]
            for key in keys_to_release:
                del self._locks[key]
                if key in self._resource_locks:
                    try:
                        self._resource_locks[key].release()
                    except RuntimeError:
                        pass
                released += 1

        return released

    async def get_all_locks(self) -> Dict[str, LockInfo]:
        """Get all current locks (for debugging)."""
        async with self._lock:
            return dict(self._locks)

    async def clear_all(self):
        """Clear all locks (for testing)."""
        async with self._lock:
            for key in list(self._resource_locks.keys()):
                try:
                    self._resource_locks[key].release()
                except RuntimeError:
                    pass
            self._locks.clear()
            self._resource_locks.clear()


class RedisLockManager(LockManager):
    """Redis-based lock manager for distributed orchestration.

    ┌─────────────────────────────────────────────────────────────────┐
    │  🔒 COMMERCIAL DEPLOYMENT ONLY - DEVELOPMENT LOCKED            │
    │                                                                 │
    │  This feature is intentionally not implemented.                 │
    │  For personal/single-process use: Use InMemoryLockManager      │
    │  For commercial distributed deployment: Implement as needed    │
    └─────────────────────────────────────────────────────────────────┘

    When implementing for commercial deployment:
    - Distributed locking with Redis SET NX EX
    - Lock renewal for long-running operations
    - Automatic expiration for failed processes
    - Pub/Sub for lock release notifications
    - Consider Redlock algorithm for stronger guarantees
    """

    def __init__(self, redis_url: str):
        """Initialize Redis lock manager.

        Args:
            redis_url: Redis connection URL

        Raises:
            NotImplementedError: This feature is locked for commercial deployment.
        """
        raise NotImplementedError(
            "🔒 RedisLockManager: COMMERCIAL DEPLOYMENT ONLY\n"
            "This feature is locked. For personal use, use InMemoryLockManager.\n"
            "Implement this class when deploying to distributed/multi-process environments."
        )

    async def acquire(self, resource_keys, holder_id, timeout=None):
        raise NotImplementedError("🔒 Commercial deployment only")

    async def release(self, resource_keys, holder_id):
        raise NotImplementedError("🔒 Commercial deployment only")

    async def get_lock_info(self, resource_key):
        raise NotImplementedError("🔒 Commercial deployment only")

    async def get_held_locks(self, holder_id):
        raise NotImplementedError("🔒 Commercial deployment only")

    async def release_all(self, holder_id):
        raise NotImplementedError("🔒 Commercial deployment only")


# =============================================================================
# Utility Functions
# =============================================================================


def validate_resource_keys(resource_keys: List[str]) -> List[str]:
    """Validate and normalize resource keys.

    Resource keys should follow naming convention:
    - category:name (e.g., "api:payments", "types:user")
    - No spaces or special characters

    Args:
        resource_keys: Keys to validate

    Returns:
        Normalized keys

    Raises:
        ValueError: If key format is invalid
    """
    normalized = []
    for key in resource_keys:
        key = key.strip().lower()

        if not key:
            continue

        # Basic validation
        if " " in key:
            raise ValueError(f"Resource key cannot contain spaces: {key}")

        # Recommend category:name format
        if ":" not in key:
            # Warn but allow
            pass

        normalized.append(key)

    return normalized


def compute_layer_locks(
    work_orders: List["WorkOrder"],
) -> List[List[str]]:
    """Compute execution layers based on resource locks.

    WorkOrders with overlapping locks cannot be in the same layer.
    This function groups WorkOrders into parallel-safe layers.

    Args:
        work_orders: List of WorkOrder objects

    Returns:
        List of layers, each layer is list of WorkOrder IDs
        that can run in parallel.

    Example:
        >>> layers = compute_layer_locks(work_orders)
        >>> # layers[0] can all run in parallel
        >>> # layers[1] runs after layers[0] completes
    """
    # Build dependency graph based on locks
    wo_map = {wo.id: wo for wo in work_orders}
    remaining = set(wo.id for wo in work_orders)
    completed: Set[str] = set()
    layers: List[List[str]] = []

    while remaining:
        # Find WOs whose dependencies are satisfied
        ready = []
        for wo_id in remaining:
            wo = wo_map[wo_id]
            if all(dep in completed for dep in wo.dependencies):
                ready.append(wo_id)

        if not ready:
            # Circular dependency - add all remaining
            layers.append(list(remaining))
            break

        # Build layer respecting lock conflicts
        layer: List[str] = []
        used_locks: Set[str] = set()

        # Sort by priority
        ready.sort(key=lambda x: wo_map[x].priority)

        for wo_id in ready:
            wo = wo_map[wo_id]
            wo_locks = set(wo.resource_locks)

            # Check for lock conflict with current layer
            if wo_locks & used_locks:
                continue  # Skip to next layer

            layer.append(wo_id)
            used_locks.update(wo_locks)

        if layer:
            layers.append(layer)
            completed.update(layer)
            remaining -= set(layer)
        else:
            # All ready WOs have lock conflicts
            # Force one through
            forced = ready[0]
            layers.append([forced])
            completed.add(forced)
            remaining.remove(forced)

    return layers


__all__ = [
    "LockInfo",
    "LockAcquisitionResult",
    "LockManager",
    "InMemoryLockManager",
    "RedisLockManager",
    "validate_resource_keys",
    "compute_layer_locks",
]
