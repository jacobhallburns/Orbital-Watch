"""
Abstract base class for all TLE data sources.

Every source — whether CelesTrak, Space-Track, a classified internal feed, or a
local file — must implement the three abstract methods below.  The rest of the
library uses only this interface, so swapping sources is transparent.

How to add a custom source
--------------------------
1.  Subclass ``BaseSource`` and implement ``name``, ``fetch_by_norad_id``,
    ``fetch_category``, and ``fetch_many``.
2.  Register the instance with ``SourceRegistry.register()``.
3.  The rest of the library works identically regardless of source.

Minimal example::

    from orbitalwatch.sources.base import BaseSource, SourceFetchError
    from orbitalwatch.tle.model import TLERecord
    from orbitalwatch.tle.parser import parse


    class MyCustomSource(BaseSource):
        \"\"\"Example showing how to add a proprietary or classified data source.

        Implement these three methods and register with
        SourceRegistry.register(). The rest of the library works
        identically regardless of source.

        All methods must raise SourceFetchError (or a subclass) on failure —
        never return empty results when the fetch itself failed.
        \"\"\"

        @property
        def name(self) -> str:
            return "my-classified-feed"

        def fetch_by_norad_id(self, norad_id: int) -> TLERecord:
            raw = my_internal_api.get_tle(norad_id)
            if raw is None:
                raise SourceFetchError(f"NORAD {norad_id} not found in {self.name}")
            return parse(raw.name, raw.line1, raw.line2, source=self.name)

        def fetch_category(self, category: str) -> list[TLERecord]:
            raw_list = my_internal_api.get_category(category)
            if raw_list is None:
                raise SourceFetchError(
                    f"Category {category!r} unavailable from {self.name}"
                )
            return [parse(r.name, r.line1, r.line2, source=self.name) for r in raw_list]

        def fetch_many(self, norad_ids: list[int]) -> list[TLERecord]:
            # Sources with a batch endpoint should override this for efficiency.
            results = []
            for nid in norad_ids:
                try:
                    results.append(self.fetch_by_norad_id(nid))
                except SourceFetchError:
                    pass  # silent skip; caller compares returned IDs if needed
            return results

Contract rules
--------------
- ``fetch_by_norad_id`` MUST raise ``SourceFetchError`` (or subclass) if the
  object cannot be found.  It must never return a wrong satellite.
- ``fetch_category`` MUST raise ``SourceFetchError`` if the category is
  unrecognised or the request fails.  It may return an empty list only if the
  category genuinely contains zero objects.
- ``fetch_many`` may silently skip individual IDs that are not found (return
  fewer results than requested), but MUST raise ``SourceFetchError`` if the
  batch request itself fails.
- All returned ``TLERecord`` instances must have ``source`` set to ``self.name``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from orbitalwatch.tle.model import TLERecord


class SourceFetchError(RuntimeError):
    """Raised when a source cannot fulfil a fetch request."""


class BaseSource(ABC):
    """Abstract contract that every TLE data source must satisfy.

    See module docstring for a complete implementation example.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable label for this source, e.g. ``'celestrak'``."""

    @abstractmethod
    def fetch_by_norad_id(self, norad_id: int) -> TLERecord:
        """Return the TLERecord for a single satellite by NORAD catalog number.

        Args:
            norad_id: Integer NORAD ID, e.g. ``25544`` for the ISS.

        Returns:
            A ``TLERecord`` with ``source`` set to ``self.name``.

        Raises:
            SourceFetchError: If the satellite cannot be found or the request
                fails.  Never return a wrong satellite.
        """

    @abstractmethod
    def fetch_category(self, category: str) -> list[TLERecord]:
        """Return all satellites in a named category / group.

        Args:
            category: Category name recognised by this source (e.g. ``'stations'``,
                ``'starlink'``, ``'active'``).  Valid names are source-specific.

        Returns:
            List of ``TLERecord`` objects — one per satellite.  May be empty only
            if the category genuinely has zero members.

        Raises:
            SourceFetchError: On network failure or unrecognised category.
        """

    @abstractmethod
    def fetch_many(self, norad_ids: list[int]) -> list[TLERecord]:
        """Return TLERecords for a batch of NORAD IDs.

        Sources with a native batch API should override this for efficiency.
        The default pattern — looping over ``fetch_by_norad_id`` — is
        acceptable but slow for large lists.

        Args:
            norad_ids: List of integer NORAD IDs to fetch.

        Returns:
            List of ``TLERecord`` objects.  Objects not found are silently
            omitted; compare returned NORAD IDs to the input list if strict
            presence checking is required.

        Raises:
            SourceFetchError: If the batch request itself fails (distinct from
                individual IDs simply not being present in the catalog).
        """
