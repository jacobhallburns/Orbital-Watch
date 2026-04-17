"""
SQLite-backed local TLE cache with configurable TTL.

Usage pattern — wrap any source call with the cache:

    cache = TLECache()
    record = cache.get(norad_id=25544)
    if record is None:
        record = source.fetch_by_norad_id(25544)
        cache.put(record)

The cache stores raw TLE line1/line2 strings and re-parses on retrieval
so the full TLERecord dataclass (including fetched_at) is always current.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from orbitalwatch.tle.model import TLERecord
from orbitalwatch.tle.parser import parse

logger = logging.getLogger(__name__)

_DEFAULT_TTL_SECONDS = 3600          # 1 hour
_DEFAULT_DB_PATH = Path.home() / ".orbitalwatch" / "tle_cache.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tle_cache (
    norad_id    INTEGER PRIMARY KEY,
    name        TEXT    NOT NULL,
    line1       TEXT    NOT NULL,
    line2       TEXT    NOT NULL,
    source      TEXT    NOT NULL DEFAULT 'unknown',
    fetched_at  TEXT    NOT NULL   -- ISO-8601 UTC
);
"""


class TLECache:
    """SQLite-backed cache for TLERecord objects.

    Args:
        db_path:     Path to the SQLite database file.
                     Defaults to ``~/.orbitalwatch/tle_cache.db``.
        ttl_seconds: How long a cached entry is considered fresh.
                     Defaults to 3600 (1 hour). Pass 0 to disable TTL
                     (always return cached values if present).
    """

    def __init__(
        self,
        db_path: Path | str | None = None,
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
    ) -> None:
        self.db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH
        self.ttl_seconds = ttl_seconds
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, norad_id: int) -> TLERecord | None:
        """Return a cached TLERecord for *norad_id*, or None if absent/stale.

        A record is considered stale if its ``fetched_at`` timestamp is older
        than ``self.ttl_seconds`` ago.  Pass ``ttl_seconds=0`` at construction
        to always return cached values regardless of age.
        """
        row = self._conn.execute(
            "SELECT name, line1, line2, source, fetched_at FROM tle_cache WHERE norad_id = ?",
            (norad_id,),
        ).fetchone()
        if row is None:
            return None

        name, line1, line2, source, fetched_at_iso = row
        fetched_at = datetime.fromisoformat(fetched_at_iso)

        if self.ttl_seconds > 0:
            age = datetime.now(tz=timezone.utc) - fetched_at
            if age > timedelta(seconds=self.ttl_seconds):
                logger.debug("Cache miss (stale) for NORAD %d (age %s)", norad_id, age)
                return None

        try:
            record = parse(name, line1, line2, source=source)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Corrupt cache entry for NORAD %d: %s", norad_id, exc)
            self.invalidate(norad_id)
            return None

        logger.debug("Cache hit for NORAD %d", norad_id)
        return record

    def put(self, record: TLERecord) -> None:
        """Insert or replace a TLERecord in the cache."""
        fetched_at_iso = record.fetched_at.astimezone(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT OR REPLACE INTO tle_cache (norad_id, name, line1, line2, source, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                record.norad_id,
                record.name,
                record.line1,
                record.line2,
                record.source,
                fetched_at_iso,
            ),
        )
        self._conn.commit()
        logger.debug("Cached NORAD %d (%s)", record.norad_id, record.name)

    def put_many(self, records: list[TLERecord]) -> None:
        """Bulk-insert a list of TLERecords into the cache."""
        now_iso = datetime.now(tz=timezone.utc).isoformat()
        rows = [
            (r.norad_id, r.name, r.line1, r.line2, r.source, now_iso)
            for r in records
        ]
        self._conn.executemany(
            """
            INSERT OR REPLACE INTO tle_cache (norad_id, name, line1, line2, source, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self._conn.commit()
        logger.debug("Bulk-cached %d records", len(rows))

    def invalidate(self, norad_id: int) -> None:
        """Remove a single entry from the cache."""
        self._conn.execute("DELETE FROM tle_cache WHERE norad_id = ?", (norad_id,))
        self._conn.commit()

    def clear(self) -> None:
        """Remove all entries from the cache."""
        self._conn.execute("DELETE FROM tle_cache")
        self._conn.commit()

    def count(self) -> int:
        """Return the total number of cached entries."""
        row = self._conn.execute("SELECT COUNT(*) FROM tle_cache").fetchone()
        return row[0] if row else 0

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    def __enter__(self) -> "TLECache":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
