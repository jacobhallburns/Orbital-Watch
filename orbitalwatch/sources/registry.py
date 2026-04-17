"""
Source registry with priority ordering, health tracking, and degraded-mode fallback.

When a source fails the registry:
1. Logs a warning with source name and reason.
2. Falls back to the SQLite cache automatically.
3. If the cache has no entry, tries the next registered source.
4. If all sources and the cache fail, raises ``AllSourcesFailedError`` with a
   full status report.

A source failure NEVER silently returns empty results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from orbitalwatch.sources.base import BaseSource
from orbitalwatch.sources.cache import TLECache
from orbitalwatch.tle.model import TLERecord

logger = logging.getLogger(__name__)


@dataclass
class SourceStatus:
    """Health snapshot for a single registered source."""

    name: str
    healthy: bool
    last_checked: datetime
    last_success: Optional[datetime]
    error_message: Optional[str]
    cache_age_seconds: Optional[float]


class AllSourcesFailedError(RuntimeError):
    """Raised when every registered source and the cache have all failed.

    Attributes:
        statuses: Per-source health snapshots at the time of failure.
    """

    def __init__(self, message: str, statuses: list[SourceStatus]) -> None:
        super().__init__(message)
        self.statuses = statuses


class SourceRegistry:
    """Manages multiple TLE sources with priority ordering and health tracking.

    Args:
        cache:  Shared ``TLECache`` instance.  A default in-memory-equivalent
                cache is created when not provided.
        strict: When ``True``, any source failure raises immediately instead of
                falling back.  Useful for tests that want deterministic failure.

    Example::

        registry = SourceRegistry()
        registry.register(CelesTrakSource(), priority=10)
        registry.register(MyBackupSource(), priority=20)

        record = registry.fetch_by_norad_id(25544)
        for status in registry.source_health():
            print(status.name, "healthy" if status.healthy else status.error_message)
    """

    def __init__(
        self,
        cache: TLECache | None = None,
        strict: bool = False,
    ) -> None:
        self._sources: list[tuple[int, BaseSource]] = []
        self._cache: TLECache = cache if cache is not None else TLECache()
        self._strict = strict
        self._statuses: dict[str, SourceStatus] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, source: BaseSource, priority: int = 100) -> None:
        """Register a source.

        Args:
            source:   Any ``BaseSource`` implementation.
            priority: Lower numbers are tried first.  Sources with equal
                      priority are tried in registration order.
        """
        self._sources.append((priority, source))
        self._sources.sort(key=lambda x: x[0])
        self._statuses[source.name] = SourceStatus(
            name=source.name,
            healthy=True,
            last_checked=datetime.now(tz=timezone.utc),
            last_success=None,
            error_message=None,
            cache_age_seconds=None,
        )

    # ------------------------------------------------------------------
    # Fetch methods
    # ------------------------------------------------------------------

    def fetch_by_norad_id(self, norad_id: int) -> TLERecord:
        """Fetch a single satellite with cache fallback.

        Tries each registered source in priority order.  On source failure:
        checks the cache, returns cached data if present, otherwise moves to
        the next source.  Raises ``AllSourcesFailedError`` only when every
        source and the cache have been exhausted.

        Args:
            norad_id: Integer NORAD catalog number.

        Returns:
            A ``TLERecord`` from the first source (or cache) that succeeds.

        Raises:
            AllSourcesFailedError: If all sources fail and the cache has no
                entry for this NORAD ID.
        """
        if not self._sources:
            cached = self._cache.get(norad_id)
            if cached is not None:
                return cached
            raise AllSourcesFailedError(
                f"No sources registered and no cache entry for NORAD {norad_id}.",
                [],
            )

        errors: list[tuple[str, str]] = []

        for _, source in self._sources:
            try:
                record = source.fetch_by_norad_id(norad_id)
                self._mark_success(source.name)
                self._cache.put(record)
                return record
            except Exception as exc:
                reason = str(exc)
                self._mark_failure(source.name, reason)
                logger.warning(
                    "Source %r failed for NORAD %d: %s", source.name, norad_id, reason
                )
                errors.append((source.name, reason))

                if self._strict:
                    raise AllSourcesFailedError(
                        f"Source {source.name!r} failed (strict=True): {reason}",
                        list(self._statuses.values()),
                    ) from exc

                cached = self._cache.get(norad_id)
                if cached is not None:
                    logger.warning(
                        "Falling back to cache for NORAD %d (age may be stale)", norad_id
                    )
                    return cached

        raise AllSourcesFailedError(
            f"All sources failed for NORAD {norad_id}. "
            + "; ".join(f"{n}: {e}" for n, e in errors),
            list(self._statuses.values()),
        )

    def fetch_category(self, category: str) -> list[TLERecord]:
        """Fetch a category with source fallback.

        Unlike ``fetch_by_norad_id``, category fetches cannot fall back to the
        cache (the cache is keyed by NORAD ID, not category name).  All sources
        are attempted in order; if all fail, ``AllSourcesFailedError`` is raised.

        Args:
            category: Category name recognised by at least one registered source.

        Returns:
            List of ``TLERecord`` objects from the first source that succeeds.

        Raises:
            AllSourcesFailedError: If every source fails.
        """
        if not self._sources:
            raise AllSourcesFailedError(
                f"No sources registered; cannot fetch category {category!r}.",
                [],
            )

        errors: list[tuple[str, str]] = []

        for _, source in self._sources:
            try:
                records = source.fetch_category(category)
                self._mark_success(source.name)
                self._cache.put_many(records)
                return records
            except Exception as exc:
                reason = str(exc)
                self._mark_failure(source.name, reason)
                logger.warning(
                    "Source %r failed for category %r: %s", source.name, category, reason
                )
                errors.append((source.name, reason))

                if self._strict:
                    raise AllSourcesFailedError(
                        f"Source {source.name!r} failed (strict=True): {reason}",
                        list(self._statuses.values()),
                    ) from exc

        raise AllSourcesFailedError(
            f"All sources failed for category {category!r}. "
            + "; ".join(f"{n}: {e}" for n, e in errors),
            list(self._statuses.values()),
        )

    def fetch_many(self, norad_ids: list[int]) -> list[TLERecord]:
        """Fetch multiple satellites, each with the same fallback logic as
        ``fetch_by_norad_id``.

        Silently skips individual IDs that cannot be found on any source or in
        the cache.  Raises ``AllSourcesFailedError`` only if every source fails
        for *every* requested ID (i.e., the registry itself is broken).

        Args:
            norad_ids: List of integer NORAD IDs.

        Returns:
            List of ``TLERecord`` objects for IDs that were successfully resolved.
        """
        results: list[TLERecord] = []
        for nid in norad_ids:
            try:
                results.append(self.fetch_by_norad_id(nid))
            except AllSourcesFailedError:
                logger.debug("All sources failed for NORAD %d; skipping", nid)
        return results

    # ------------------------------------------------------------------
    # Health reporting
    # ------------------------------------------------------------------

    def source_health(self) -> list[SourceStatus]:
        """Return current health status for every registered source.

        Returns:
            List of ``SourceStatus`` objects, one per registered source, in
            registration order.
        """
        return [status for _, src in self._sources for status in [self._statuses[src.name]]]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mark_success(self, name: str) -> None:
        now = datetime.now(tz=timezone.utc)
        prev = self._statuses.get(name)
        self._statuses[name] = SourceStatus(
            name=name,
            healthy=True,
            last_checked=now,
            last_success=now,
            error_message=None,
            cache_age_seconds=prev.cache_age_seconds if prev else None,
        )

    def _mark_failure(self, name: str, error: str) -> None:
        now = datetime.now(tz=timezone.utc)
        prev = self._statuses.get(name)
        self._statuses[name] = SourceStatus(
            name=name,
            healthy=False,
            last_checked=now,
            last_success=prev.last_success if prev else None,
            error_message=error,
            cache_age_seconds=prev.cache_age_seconds if prev else None,
        )
