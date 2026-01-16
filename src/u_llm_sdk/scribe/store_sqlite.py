"""SQLite-based Scribe Store.

Two-table design for efficient queries:
1. scribe_items: Append-only event/history table (all versions)
2. scribe_heads: Index table pointing to current active head per section/key

This design ensures:
- No actual deletion (audit trail preserved)
- O(1) lookup of current active items
- Efficient phase-based queries via joins
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional

from .types import (
    Clearance,
    ChangeReason,
    PHASE_SECTION_MAP,
    PHASE_CLEARANCE_MAP,
    SECTIONS,
    ScribeItem,
    ScribeStatus,
    ScribeType,
)

logger = logging.getLogger(__name__)


# =============================================================================
# DDL (Schema Definition)
# =============================================================================

DDL_SCRIBE_ITEMS = """
CREATE TABLE IF NOT EXISTS scribe_items (
    -- Primary key: unique per version
    rowid INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Identity
    id TEXT NOT NULL,
    type TEXT NOT NULL,

    -- Lifecycle
    status TEXT NOT NULL DEFAULT 'active',
    version INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    updated_by TEXT NOT NULL DEFAULT '',

    -- Public Layer (always visible in DEFAULT)
    public_notice TEXT NOT NULL DEFAULT '',
    public_summary TEXT NOT NULL DEFAULT '',

    -- Sealed Layer (only visible in ESCALATED/AUDIT)
    sealed_payload_json TEXT NOT NULL DEFAULT '{}',

    -- Provenance (JSON array of source IDs)
    provenance_json TEXT NOT NULL DEFAULT '[]',

    -- Change Tracking
    supersedes TEXT,
    superseded_by TEXT,
    retract_reason TEXT
);

-- Index for fast lookup by id (latest version)
CREATE INDEX IF NOT EXISTS idx_scribe_items_id ON scribe_items(id);

-- Index for type-based queries
CREATE INDEX IF NOT EXISTS idx_scribe_items_type ON scribe_items(type);

-- Index for status filtering
CREATE INDEX IF NOT EXISTS idx_scribe_items_status ON scribe_items(status);
"""

DDL_SCRIBE_HEADS = """
CREATE TABLE IF NOT EXISTS scribe_heads (
    -- Composite key: type + id
    type TEXT NOT NULL,
    id TEXT NOT NULL,

    -- Points to the current active rowid in scribe_items
    head_rowid INTEGER NOT NULL,

    -- Last update timestamp for cache invalidation
    updated_at TEXT NOT NULL,

    PRIMARY KEY (type, id),
    FOREIGN KEY (head_rowid) REFERENCES scribe_items(rowid)
);

-- Index for type-based section queries
CREATE INDEX IF NOT EXISTS idx_scribe_heads_type ON scribe_heads(type);
"""


# =============================================================================
# Store Implementation
# =============================================================================


class ScribeStore:
    """SQLite-backed Scribe store with two-table design.

    Usage:
        >>> store = ScribeStore(Path(".chronicle/scribe.db"))
        >>> store.upsert_section(
        ...     ScribeType.CONVENTION,
        ...     "conv:naming",
        ...     public_summary="Use snake_case for functions.",
        ...     public_notice="Naming convention for Python code.",
        ...     sealed_payload={"examples": ["def my_func():", ...]},
        ...     provenance=["chronicle:decision:123"],
        ...     updated_by="phase:prepare",
        ... )
        >>> context = store.get_for_phase("plan", Clearance.DEFAULT)
    """

    def __init__(self, db_path: Path):
        """Initialize store.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize schema
        with self._connection() as conn:
            conn.executescript(DDL_SCRIBE_ITEMS)
            conn.executescript(DDL_SCRIBE_HEADS)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # =========================================================================
    # Core Operations (Only 3 mutations allowed)
    # =========================================================================

    def upsert_section(
        self,
        item_type: ScribeType,
        item_id: str,
        *,
        public_summary: str,
        public_notice: str,
        sealed_payload: Optional[dict] = None,
        provenance: Optional[list[str]] = None,
        updated_by: str = "",
    ) -> ScribeItem:
        """Insert or update a section item (most common operation).

        If item_id already exists:
        - Creates new version with version++
        - Updates scribe_heads to point to new version

        Args:
            item_type: ScribeType for this item
            item_id: Unique identifier within type
            public_summary: Brief description (always visible)
            public_notice: Change notice (always visible)
            sealed_payload: Detailed content (only ESCALATED+)
            provenance: Links to Chronicle/SCIP sources
            updated_by: Which phase/agent is making this change

        Returns:
            The created/updated ScribeItem
        """
        now = datetime.now()
        sealed = sealed_payload or {}
        prov = provenance or []

        with self._connection() as conn:
            # Check if exists
            existing = conn.execute(
                """
                SELECT si.* FROM scribe_items si
                JOIN scribe_heads sh ON si.rowid = sh.head_rowid
                WHERE sh.type = ? AND sh.id = ?
                """,
                (item_type.value, item_id),
            ).fetchone()

            if existing:
                new_version = existing["version"] + 1
                created_at = existing["created_at"]
            else:
                new_version = 1
                created_at = now.isoformat()

            # Insert new item
            cursor = conn.execute(
                """
                INSERT INTO scribe_items (
                    id, type, status, version,
                    created_at, updated_at, updated_by,
                    public_notice, public_summary,
                    sealed_payload_json, provenance_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item_id,
                    item_type.value,
                    ScribeStatus.ACTIVE.value,
                    new_version,
                    created_at,
                    now.isoformat(),
                    updated_by,
                    public_notice,
                    public_summary,
                    json.dumps(sealed),
                    json.dumps(prov),
                ),
            )
            new_rowid = cursor.lastrowid

            # Update head pointer
            conn.execute(
                """
                INSERT OR REPLACE INTO scribe_heads (type, id, head_rowid, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (item_type.value, item_id, new_rowid, now.isoformat()),
            )

            logger.debug(f"Upserted {item_type.value}:{item_id} v{new_version}")

        return ScribeItem(
            id=item_id,
            type=item_type,
            status=ScribeStatus.ACTIVE,
            version=new_version,
            created_at=datetime.fromisoformat(created_at),
            updated_at=now,
            updated_by=updated_by,
            public_notice=public_notice,
            public_summary=public_summary,
            sealed_payload=sealed,
            provenance=prov,
        )

    def supersede(
        self,
        old_id: str,
        new_item: ScribeItem,
        reason: str,
    ) -> ScribeItem:
        """Replace an item with a new version.

        Sets old item's status to SUPERSEDED and creates new item.
        Old item's public_notice is updated to indicate supersession.

        Args:
            old_id: ID of the item to supersede
            new_item: New item to replace with
            reason: Human-readable reason for change

        Returns:
            The newly created item

        Raises:
            ValueError: If old_id doesn't exist
        """
        now = datetime.now()

        with self._connection() as conn:
            # Get old item's head
            old_row = conn.execute(
                """
                SELECT si.*, sh.type as head_type FROM scribe_items si
                JOIN scribe_heads sh ON si.rowid = sh.head_rowid
                WHERE sh.id = ?
                """,
                (old_id,),
            ).fetchone()

            if not old_row:
                raise ValueError(f"Item not found: {old_id}")

            # Update old item's status
            conn.execute(
                """
                UPDATE scribe_items
                SET status = ?,
                    superseded_by = ?,
                    public_notice = ?,
                    updated_at = ?
                WHERE rowid = (
                    SELECT head_rowid FROM scribe_heads WHERE id = ?
                )
                """,
                (
                    ScribeStatus.SUPERSEDED.value,
                    new_item.id,
                    f"📝 Superseded: {reason}. New: {new_item.id}",
                    now.isoformat(),
                    old_id,
                ),
            )

            # Insert new item with supersedes link
            new_item.supersedes = old_id
            new_item.updated_at = now

            cursor = conn.execute(
                """
                INSERT INTO scribe_items (
                    id, type, status, version,
                    created_at, updated_at, updated_by,
                    public_notice, public_summary,
                    sealed_payload_json, provenance_json,
                    supersedes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    new_item.id,
                    new_item.type.value,
                    ScribeStatus.ACTIVE.value,
                    new_item.version,
                    now.isoformat(),
                    now.isoformat(),
                    new_item.updated_by,
                    new_item.public_notice,
                    new_item.public_summary,
                    json.dumps(new_item.sealed_payload),
                    json.dumps(new_item.provenance),
                    old_id,
                ),
            )
            new_rowid = cursor.lastrowid

            # Update head to point to new item
            conn.execute(
                """
                INSERT OR REPLACE INTO scribe_heads (type, id, head_rowid, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (new_item.type.value, new_item.id, new_rowid, now.isoformat()),
            )

            logger.info(f"Superseded {old_id} with {new_item.id}: {reason}")

        return new_item

    def retract(
        self,
        item_id: str,
        reason: ChangeReason,
        retracted_by: str,
        replacement_hint: Optional[str] = None,
    ) -> None:
        """Retract (invalidate) an item.

        Item's content is sealed, but public_notice is updated
        to indicate retraction and reason.

        Args:
            item_id: ID of the item to retract
            reason: Why this is being retracted
            retracted_by: Which phase/agent is retracting
            replacement_hint: Optional hint about what to use instead
        """
        now = datetime.now()

        notice = f"❌ Retracted: {reason.value}"
        if replacement_hint:
            notice += f". See: {replacement_hint}"

        with self._connection() as conn:
            result = conn.execute(
                """
                UPDATE scribe_items
                SET status = ?,
                    retract_reason = ?,
                    public_notice = ?,
                    updated_at = ?,
                    updated_by = ?
                WHERE rowid = (
                    SELECT head_rowid FROM scribe_heads WHERE id = ?
                )
                """,
                (
                    ScribeStatus.RETRACTED.value,
                    reason.value,
                    notice,
                    now.isoformat(),
                    retracted_by,
                    item_id,
                ),
            )

            if result.rowcount == 0:
                logger.warning(f"Retract failed - item not found: {item_id}")
            else:
                logger.info(f"Retracted {item_id}: {reason.value}")

    # =========================================================================
    # Query Operations
    # =========================================================================

    def get_item(self, item_id: str) -> Optional[ScribeItem]:
        """Get the current (head) version of an item."""
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT si.* FROM scribe_items si
                JOIN scribe_heads sh ON si.rowid = sh.head_rowid
                WHERE sh.id = ?
                """,
                (item_id,),
            ).fetchone()

            if row:
                return self._row_to_item(row)
            return None

    def get_items_by_type(
        self,
        item_type: ScribeType,
        *,
        include_retracted: bool = False,
    ) -> list[ScribeItem]:
        """Get all current items of a specific type."""
        with self._connection() as conn:
            if include_retracted:
                rows = conn.execute(
                    """
                    SELECT si.* FROM scribe_items si
                    JOIN scribe_heads sh ON si.rowid = sh.head_rowid
                    WHERE sh.type = ?
                    ORDER BY si.updated_at DESC
                    """,
                    (item_type.value,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT si.* FROM scribe_items si
                    JOIN scribe_heads sh ON si.rowid = sh.head_rowid
                    WHERE sh.type = ? AND si.status = 'active'
                    ORDER BY si.updated_at DESC
                    """,
                    (item_type.value,),
                ).fetchall()

            return [self._row_to_item(row) for row in rows]

    def get_active_sections(
        self,
        section_names: list[str],
        clearance: Clearance = Clearance.DEFAULT,
    ) -> dict[str, list[ScribeItem]]:
        """Get items for specified sections.

        Args:
            section_names: Section names from SECTIONS
            clearance: Visibility level

        Returns:
            Dict mapping section name → list of items
        """
        result: dict[str, list[ScribeItem]] = {}

        for section_name in section_names:
            section = SECTIONS.get(section_name)
            if not section:
                continue

            items = []
            for item_type in section.types:
                type_items = self.get_items_by_type(
                    item_type,
                    include_retracted=(clearance != Clearance.DEFAULT),
                )
                items.extend(type_items)

            result[section_name] = items

        return result

    def get_for_phase(
        self,
        phase_name: str,
        clearance: Optional[Clearance] = None,
    ) -> str:
        """Get formatted injection text for a phase.

        This is the main injection API.

        Args:
            phase_name: Phase name (prepare, design, plan, etc.)
            clearance: Override clearance level (uses default if None)

        Returns:
            Formatted markdown text ready for injection
        """
        section_names = PHASE_SECTION_MAP.get(phase_name, [])
        if not section_names:
            return ""

        if clearance is None:
            clearance = PHASE_CLEARANCE_MAP.get(phase_name, Clearance.DEFAULT)

        sections = self.get_active_sections(section_names, clearance)

        lines = []
        for section_name in section_names:
            section = SECTIONS.get(section_name)
            items = sections.get(section_name, [])

            if not items:
                continue

            lines.append(section.header)
            lines.append("")

            for item in items:
                if clearance == Clearance.DEFAULT:
                    lines.append(item.to_public_text())
                else:
                    lines.append(item.to_escalated_text())

            lines.append("")

        return "\n".join(lines)

    def get_digest(self, phase_name: str) -> str:
        """Get a digest of scribe state for cache invalidation.

        Returns a hash of all public_notice + public_summary
        for sections relevant to this phase.

        This is used as part of EvidenceGate cache key.
        """
        section_names = PHASE_SECTION_MAP.get(phase_name, [])
        sections = self.get_active_sections(section_names, Clearance.DEFAULT)

        content_parts = []
        for section_name, items in sorted(sections.items()):
            for item in sorted(items, key=lambda x: x.id):
                content_parts.append(f"{item.id}:{item.version}:{item.public_notice}")

        if not content_parts:
            return "empty"

        combined = "|".join(content_parts)
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    # =========================================================================
    # History Operations (for AUDIT clearance)
    # =========================================================================

    def get_history(self, item_id: str) -> list[ScribeItem]:
        """Get full version history of an item (for AUDIT)."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM scribe_items
                WHERE id = ?
                ORDER BY version DESC
                """,
                (item_id,),
            ).fetchall()

            return [self._row_to_item(row) for row in rows]

    def get_change_log(
        self,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[ScribeItem]:
        """Get recent changes across all items."""
        with self._connection() as conn:
            if since:
                rows = conn.execute(
                    """
                    SELECT * FROM scribe_items
                    WHERE updated_at > ?
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    (since.isoformat(), limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM scribe_items
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()

            return [self._row_to_item(row) for row in rows]

    # =========================================================================
    # Helpers
    # =========================================================================

    def _row_to_item(self, row: sqlite3.Row) -> ScribeItem:
        """Convert database row to ScribeItem."""
        return ScribeItem(
            id=row["id"],
            type=ScribeType(row["type"]),
            status=ScribeStatus(row["status"]),
            version=row["version"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            updated_by=row["updated_by"],
            public_notice=row["public_notice"],
            public_summary=row["public_summary"],
            sealed_payload=json.loads(row["sealed_payload_json"]),
            provenance=json.loads(row["provenance_json"]),
            supersedes=row["supersedes"],
            superseded_by=row["superseded_by"],
            retract_reason=ChangeReason(row["retract_reason"]) if row["retract_reason"] else None,
        )

    def clear(self) -> None:
        """Clear all data (for testing only)."""
        with self._connection() as conn:
            conn.execute("DELETE FROM scribe_heads")
            conn.execute("DELETE FROM scribe_items")


__all__ = ["ScribeStore"]
