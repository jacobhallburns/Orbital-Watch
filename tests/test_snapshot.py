"""Tests for AOIQuery.snapshot() and AOIRequiredError."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from orbitalwatch.geo.coordinates import PointRadius
from orbitalwatch.geo.timewindow import TimeWindow
from orbitalwatch.query.aoi_query import AOIQuery, AOIRequiredError, SatellitePosition
from orbitalwatch.sources.base import BaseSource, SourceFetchError
from orbitalwatch.sources.cache import TLECache
from orbitalwatch.sources.registry import SourceRegistry
from orbitalwatch.tle.model import TLERecord
from orbitalwatch.tle.parser import parse


# ---------------------------------------------------------------------------
# Real TLE fixtures
# ---------------------------------------------------------------------------

# ISS — LEO, well-known valid TLE
_ISS = (
    "ISS (ZARYA)",
    "1 25544U 98067A   24006.87666667  .00024151  00000-0  42368-3 0  9993",
    "2 25544  51.6399 317.5029 0002437  62.8843 297.2499 15.50047572435874",
)

# Hubble — LEO
_HST = (
    "HST",
    "1 20580U 90037B   24006.16779861  .00002052  00000-0  10893-3 0  9994",
    "2 20580  28.4697 183.7621 0002549 178.9479 181.1538 15.09762456906542",
)


class _StaticSource(BaseSource):
    """Returns a fixed list of TLERecords regardless of category."""

    def __init__(self, records: list[TLERecord]) -> None:
        self._records = records

    @property
    def name(self) -> str:
        return "static"

    def fetch_by_norad_id(self, norad_id: int) -> TLERecord:
        for r in self._records:
            if r.norad_id == norad_id:
                return r
        raise SourceFetchError(f"Not found: {norad_id}")

    def fetch_category(self, category: str) -> list[TLERecord]:
        return list(self._records)

    def fetch_many(self, norad_ids: list[int]) -> list[TLERecord]:
        return [r for r in self._records if r.norad_id in norad_ids]


# Use a UTC time close to the TLE epochs to get valid propagated positions.
# ISS epoch: 2024-01-06 ~21:02 UTC.  Propagating 2+ years gives bad results.
_SNAP_UTC = datetime(2024, 1, 7, 6, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def registry(tmp_path: Path) -> SourceRegistry:
    iss = parse(*_ISS, source="test")
    hst = parse(*_HST, source="test")
    cache = TLECache(db_path=tmp_path / "snap_test.db")
    reg = SourceRegistry(cache=cache)
    reg.register(_StaticSource([iss, hst]))
    return reg


# ---------------------------------------------------------------------------
# SatellitePosition value tests
# ---------------------------------------------------------------------------

class TestSnapshotValues:
    def test_returns_satellite_position_objects(self, registry):
        positions = AOIQuery().snapshot(registry, category="stations", utc=_SNAP_UTC)
        assert len(positions) >= 1
        for p in positions:
            assert isinstance(p, SatellitePosition)

    def test_lat_in_range(self, registry):
        positions = AOIQuery().snapshot(registry, utc=_SNAP_UTC)
        for p in positions:
            assert -90.0 <= p.lat <= 90.0, f"lat={p.lat} out of range"

    def test_lon_in_range(self, registry):
        positions = AOIQuery().snapshot(registry, utc=_SNAP_UTC)
        for p in positions:
            assert -180.0 <= p.lon <= 180.0, f"lon={p.lon} out of range"

    def test_alt_km_in_leo_to_geo_range(self, registry):
        positions = AOIQuery().snapshot(registry, utc=_SNAP_UTC)
        for p in positions:
            assert 160.0 <= p.alt_km <= 42_000.0, f"alt_km={p.alt_km} out of range"

    def test_velocity_in_expected_range(self, registry):
        positions = AOIQuery().snapshot(registry, utc=_SNAP_UTC)
        for p in positions:
            assert 1.0 <= p.velocity_kms <= 8.5, f"velocity_kms={p.velocity_kms} out of range"

    def test_utc_is_timezone_aware(self, registry):
        positions = AOIQuery().snapshot(registry, utc=_SNAP_UTC)
        for p in positions:
            assert p.utc.tzinfo is not None
            assert p.utc.utcoffset().total_seconds() == 0.0

    def test_satellite_field_is_tle_record(self, registry):
        positions = AOIQuery().snapshot(registry, utc=_SNAP_UTC)
        for p in positions:
            assert isinstance(p.satellite, TLERecord)

    def test_explicit_utc_respected(self, registry):
        positions = AOIQuery().snapshot(registry, utc=_SNAP_UTC)
        for p in positions:
            assert p.utc == _SNAP_UTC

    def test_returns_correct_count(self, registry):
        positions = AOIQuery().snapshot(registry, category="stations", utc=_SNAP_UTC)
        assert len(positions) == 2  # ISS + HST from the fixture


# ---------------------------------------------------------------------------
# AOIRequiredError
# ---------------------------------------------------------------------------

class TestAOIRequiredError:
    def test_run_without_aoi_raises(self, registry):
        query = AOIQuery()
        window = TimeWindow.next_n_hours(6)
        with pytest.raises(AOIRequiredError):
            query.run(aoi=None, window=window, sources=registry)

    def test_error_message_mentions_snapshot(self, registry):
        query = AOIQuery()
        window = TimeWindow.next_n_hours(6)
        with pytest.raises(AOIRequiredError, match="snapshot"):
            query.run(aoi=None, window=window, sources=registry)
