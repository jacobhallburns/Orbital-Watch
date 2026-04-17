"""
CelesTrak data source adapter.

Fetches TLE catalogs and individual satellite records from CelesTrak's
free, no-auth API endpoint:

    https://celestrak.org/NORAD/elements/gp.php

Documented at https://celestrak.org/NORAD/documentation/gp-data-formats.php

Supported operations:
- fetch_category(category)    — all satellites in a named group
- fetch_active()              — full active-satellites catalog
- fetch_by_norad_id(norad_id) — single satellite lookup

All methods return TLERecord instances via the shared parser.
"""

from __future__ import annotations

import logging

import requests

from orbitalwatch.sources.base import BaseSource, SourceFetchError
from orbitalwatch.tle.model import TLERecord
from orbitalwatch.tle.parser import TLEParseError, parse_many

logger = logging.getLogger(__name__)

# Documented at https://celestrak.org/NORAD/documentation/gp-data-formats.php
_GP_URL = "https://celestrak.org/NORAD/elements/gp.php"

# Documented group names accepted by the GROUP query parameter.
KNOWN_CATEGORIES: tuple[str, ...] = (
    "stations",
    "visual",
    "active",
    "analyst",
    "last-30-days",
    "weather",
    "noaa",
    "goes",
    "resource",
    "sarsat",
    "dmc",
    "tdrss",
    "argos",
    "planet",
    "spire",
    "geo",
    "intelsat",
    "ses",
    "iridium",
    "iridium-NEXT",
    "starlink",
    "oneweb",
    "orbcomm",
    "globalstar",
    "amateur",
    "x-comm",
    "other-comm",
    "gps-ops",
    "glo-ops",
    "galileo",
    "beidou",
    "sbas",
    "nnss",
    "radar",
    "cubesats",
    "other",
)

_SOURCE_LABEL = "celestrak"


class CelesTrakError(SourceFetchError):
    """Raised when a CelesTrak request fails and no cached fallback is available."""


class CelesTrakSource(BaseSource):
    """Fetch satellite TLEs from CelesTrak's public API.

    Uses ``FORMAT=TLE`` (classic 3-line TLE text), which is the most
    universally compatible format and maps directly to the internal parser.

    Args:
        timeout:    HTTP request timeout in seconds (default 15).
        session:    Optional pre-configured requests.Session (useful for testing).
    """

    def __init__(self, timeout: int = 15, session: requests.Session | None = None) -> None:
        self.timeout = timeout
        self._session = session or requests.Session()
        self._session.headers.update({"User-Agent": "orbitalwatch/0.1 (python)"})

    @property
    def name(self) -> str:
        return _SOURCE_LABEL

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_category(self, category: str) -> list[TLERecord]:
        """Fetch all TLEs for a named CelesTrak group/category.

        Args:
            category: One of the group names listed in KNOWN_CATEGORIES,
                      e.g. ``'stations'``, ``'starlink'``, ``'active'``.

        Returns:
            List of TLERecord objects, one per satellite in the group.

        Raises:
            CelesTrakError: On HTTP failure or unparseable response.
        """
        params = {"GROUP": category, "FORMAT": "TLE"}
        text = self._get_text(_GP_URL, params, context=f"category={category!r}")
        return self._parse_tle_text(text)

    def fetch_active(self) -> list[TLERecord]:
        """Fetch the full active-satellites catalog (~8 000 objects).

        Equivalent to ``fetch_category('active')``.
        """
        return self.fetch_category("active")

    def fetch_by_norad_id(self, norad_id: int) -> TLERecord:
        """Fetch a single satellite by its NORAD catalog number.

        Args:
            norad_id: Integer NORAD ID, e.g. ``25544`` for the ISS.

        Returns:
            A TLERecord for the requested object.

        Raises:
            CelesTrakError: If the satellite is not found or the request fails.
        """
        params = {"CATNR": str(norad_id), "FORMAT": "TLE"}
        text = self._get_text(_GP_URL, params, context=f"NORAD={norad_id}")
        records = self._parse_tle_text(text)
        if not records:
            raise CelesTrakError(f"No data returned for NORAD ID {norad_id}")
        if len(records) > 1:
            logger.warning(
                "NORAD ID %d matched %d records; returning first", norad_id, len(records)
            )
        return records[0]

    def fetch_many(self, norad_ids: list[int]) -> list[TLERecord]:
        """Fetch multiple satellites by NORAD ID.

        Loops over ``fetch_by_norad_id``; individual misses are logged and
        skipped rather than raising.

        Args:
            norad_ids: List of integer NORAD IDs.

        Returns:
            List of TLERecord objects for IDs that were found.

        Raises:
            CelesTrakError: If a request itself fails (network error, HTTP error).
        """
        results: list[TLERecord] = []
        for nid in norad_ids:
            try:
                results.append(self.fetch_by_norad_id(nid))
            except CelesTrakError as exc:
                logger.warning("Skipping NORAD %d: %s", nid, exc)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_text(self, url: str, params: dict[str, str], context: str) -> str:
        try:
            resp = self._session.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise CelesTrakError(f"CelesTrak request failed ({context}): {exc}") from exc
        return resp.text

    def _parse_tle_text(self, text: str) -> list[TLERecord]:
        """Parse a 3-line TLE text block returned by CelesTrak into TLERecords.

        Malformed individual entries are logged and skipped; no exception is
        raised unless the entire text is unparseable.
        """
        # CelesTrak returns clean 3-line sets; parse_many handles blank lines.
        try:
            return parse_many(text, source=_SOURCE_LABEL)
        except TLEParseError as exc:
            raise CelesTrakError(f"Failed to parse CelesTrak TLE response: {exc}") from exc
