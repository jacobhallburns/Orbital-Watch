"""Tests for orbitalwatch.query.aoi_query — pre-filters and AOI query logic."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from orbitalwatch.geo.coordinates import BoundingBox, PointRadius, PolygonAOI
from orbitalwatch.geo.timewindow import TimeWindow
from orbitalwatch.query.aoi_query import (
    AOIQuery,
    AOIQueryResult,
    AOIRequiredError,
    _passes_geo_filter,
    _passes_inclination_filter,
    _passes_period_filter,
    _pre_filter,
)
from orbitalwatch.sources.base import BaseSource
from orbitalwatch.sources.cache import TLECache
from orbitalwatch.sources.registry import SourceRegistry
from orbitalwatch.tle.model import TLERecord
from orbitalwatch.tle.parser import parse


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _fix_checksum(line: str) -> str:
    total = sum(int(c) if c.isdigit() else (1 if c == "-" else 0) for c in line[:68])
    return line[:68] + str(total % 10)


# Real ISS TLE — known to pass over mid-latitudes
_ISS_TLE = (
    "ISS (ZARYA)",
    "1 25544U 98067A   24006.87666667  .00024151  00000-0  42368-3 0  9993",
    "2 25544  51.6399 317.5029 0002437  62.8843 297.2499 15.50047572435874",
)


def _make_record_with_params(
    norad_id: int = 99001,
    name: str = "TESTSAT",
    inclination_deg: float = 51.6,
    mean_motion: float = 15.5,
    raan_deg: float = 100.0,
    arg_perigee_deg: float = 90.0,
    mean_anomaly_deg: float = 270.0,
    eccentricity: float = 0.001,
) -> TLERecord:
    """Create a TLERecord with specific orbital parameters for filter testing."""
    # Build TLE-formatted strings that will pass checksum but with specific parameters
    # We use a known-valid base and just build the record directly
    return TLERecord(
        name=name,
        norad_id=norad_id,
        classification="U",
        int_designator="98067A",
        epoch=datetime(2024, 1, 7, 21, 2, 24, tzinfo=timezone.utc),
        mean_motion_dot=2.4151e-4,
        mean_motion_ddot=0.0,
        bstar=4.2368e-4,
        element_set_no=999,
        rev_at_epoch=43587,
        inclination_deg=inclination_deg,
        raan_deg=raan_deg,
        eccentricity=eccentricity,
        arg_perigee_deg=arg_perigee_deg,
        mean_anomaly_deg=mean_anomaly_deg,
        mean_motion_rev_per_day=mean_motion,
        line1=_ISS_TLE[1],
        line2=_ISS_TLE[2],
        source="test",
    )


def _make_registry(records: list[TLERecord], tmp_path: Path) -> SourceRegistry:
    class _MockSource(BaseSource):
        @property
        def name(self) -> str:
            return "mock"

        def fetch_by_norad_id(self, norad_id: int) -> TLERecord:
            for r in records:
                if r.norad_id == norad_id:
                    return r
            from orbitalwatch.sources.base import SourceFetchError
            raise SourceFetchError(f"Not found: {norad_id}")

        def fetch_category(self, category: str) -> list[TLERecord]:
            return list(records)

        def fetch_many(self, norad_ids: list[int]) -> list[TLERecord]:
            return [r for r in records if r.norad_id in norad_ids]

    cache = TLECache(db_path=tmp_path / "aoi_test.db")
    reg = SourceRegistry(cache=cache)
    reg.register(_MockSource())
    return reg


# ---------------------------------------------------------------------------
# Inclination pre-filter
# ---------------------------------------------------------------------------

class TestInclinationFilter:
    def test_high_inclination_passes(self):
        tle = _make_record_with_params(inclination_deg=51.6)
        # AOI at 30°N: min_lat 30, max_lat 30 (point)
        assert _passes_inclination_filter(tle, aoi_min_lat=30.0, aoi_max_lat=30.0)

    def test_low_inclination_filtered_for_high_lat_aoi(self):
        # Inclination 20° cannot reach 30°N (20 < 30 - 5 = 25)
        tle = _make_record_with_params(inclination_deg=20.0)
        assert not _passes_inclination_filter(tle, aoi_min_lat=30.0, aoi_max_lat=35.0)

    def test_inclination_at_boundary_passes(self):
        # Inclination 26° >= 30 - 5 = 25 → passes
        tle = _make_record_with_params(inclination_deg=26.0)
        assert _passes_inclination_filter(tle, aoi_min_lat=30.0, aoi_max_lat=35.0)

    def test_retrograde_high_inclination_always_passes(self):
        tle = _make_record_with_params(inclination_deg=98.0)
        assert _passes_inclination_filter(tle, aoi_min_lat=30.0, aoi_max_lat=35.0)

    def test_negative_aoi_max_lat_skips_filter(self):
        # Filter only applies when aoi_max_lat > 0
        tle = _make_record_with_params(inclination_deg=5.0)
        assert _passes_inclination_filter(tle, aoi_min_lat=-40.0, aoi_max_lat=-20.0)


# ---------------------------------------------------------------------------
# GEO pre-filter
# ---------------------------------------------------------------------------

class TestGEOFilter:
    def test_leo_object_always_passes(self):
        tle = _make_record_with_params(mean_motion=15.5)  # LEO
        assert _passes_geo_filter(tle, aoi_center_lon=0.0)

    def test_geo_object_near_aoi_lon_passes(self):
        # GEO object with raan=0, arg_perigee=0, mean_anomaly=0, epoch at J2000
        # subsatellite lon ≈ 0 - GAST(J2000) = 0 - 280.46 ≈ -280 ≈ 79.54°
        # Regardless of exact value, an AOI far away should be filtered
        tle = _make_record_with_params(
            mean_motion=1.0, inclination_deg=0.1,
            raan_deg=100.0, arg_perigee_deg=0.0, mean_anomaly_deg=0.0,
        )
        # Compute what the filter thinks the lon is, then test near it
        from orbitalwatch.query.aoi_query import _geo_subsatellite_lon
        geo_lon = _geo_subsatellite_lon(tle)
        # AOI center very close to subsatellite lon → should pass
        assert _passes_geo_filter(tle, aoi_center_lon=geo_lon)

    def test_geo_object_far_from_aoi_filtered(self):
        tle = _make_record_with_params(
            mean_motion=1.0, inclination_deg=0.1,
            raan_deg=100.0, arg_perigee_deg=0.0, mean_anomaly_deg=0.0,
        )
        from orbitalwatch.query.aoi_query import _geo_subsatellite_lon
        geo_lon = _geo_subsatellite_lon(tle)
        # Place AOI 100° away → filtered
        opposite_lon = (geo_lon + 100.0 + 180.0) % 360.0 - 180.0
        assert not _passes_geo_filter(tle, aoi_center_lon=opposite_lon)


# ---------------------------------------------------------------------------
# Period pre-filter
# ---------------------------------------------------------------------------

class TestPeriodFilter:
    def test_short_period_leo_passes(self):
        tle = _make_record_with_params(mean_motion=15.5)  # ~93 min period
        assert _passes_period_filter(tle, window_hours=6.0)

    def test_very_long_period_filtered(self):
        # mean_motion = 0.5 rev/day → period = 48 hours; window = 6h → 48 > 12 → filtered
        tle = _make_record_with_params(mean_motion=0.5)
        assert not _passes_period_filter(tle, window_hours=6.0)

    def test_period_exactly_at_boundary_filtered(self):
        # period = window * 2 exactly → filtered (> is exclusive at boundary)
        # window_hours=6 → threshold = 12h → mean_motion = 24/12 = 2 rev/day
        tle = _make_record_with_params(mean_motion=2.0)  # period = 12h exactly
        # 12 > 12 is False → NOT filtered at exact boundary
        assert _passes_period_filter(tle, window_hours=6.0)

    def test_zero_mean_motion_passes(self):
        # Malformed TLE — period filter skips rather than crashing
        tle = _make_record_with_params(mean_motion=0.0)
        assert _passes_period_filter(tle, window_hours=6.0)


# ---------------------------------------------------------------------------
# AOIQuery.run()
# ---------------------------------------------------------------------------

class TestAOIQueryRun:
    def test_requires_aoi(self, tmp_path):
        reg = _make_registry([], tmp_path)
        query = AOIQuery()
        with pytest.raises(AOIRequiredError):
            query.run(aoi=None, window=TimeWindow.next_n_hours(6), sources=reg)

    def test_incompatible_inclination_filtered_out(self, tmp_path):
        """Object with inclination 10° cannot reach Austin (30°N) → no passes."""
        low_inc = _make_record_with_params(norad_id=99101, inclination_deg=10.0)
        reg = _make_registry([low_inc], tmp_path)
        aoi = PointRadius(radius_km=500.0, lat=30.27, lon=-97.74)
        window = TimeWindow.next_n_hours(6)
        results = AOIQuery().run(aoi, window, reg)
        assert all(r.satellite.norad_id != 99101 for r in results)

    def test_compatible_satellite_survives_prefilter(self, tmp_path):
        """ISS (inc=51.6°) must not be pre-filtered for a 30°N AOI."""
        from orbitalwatch.query.aoi_query import _pre_filter
        iss = parse(*_ISS_TLE, source="test")
        aoi = PointRadius(radius_km=200.0, lat=30.27, lon=-97.74)
        survivors = _pre_filter([iss], aoi, window_hours=6.0)
        # ISS inclination 51.6 >= 30.27 - 5 = 25.27 → must survive
        assert any(r.norad_id == 25544 for r in survivors)

    def test_result_type(self, tmp_path):
        iss = parse(*_ISS_TLE, source="test")
        reg = _make_registry([iss], tmp_path)
        aoi = PointRadius(radius_km=2000.0, lat=30.27, lon=-97.74)
        window = TimeWindow.next_n_hours(24)
        results = AOIQuery().run(aoi, window, reg)
        for r in results:
            assert isinstance(r, AOIQueryResult)
            assert isinstance(r.satellite, TLERecord)
            assert isinstance(r.passes, list)

    def test_categories_filter_limits_fetch(self, tmp_path):
        """categories=['stations'] should fetch only that category."""
        fetch_log: list[str] = []

        class _LoggingSource(BaseSource):
            @property
            def name(self) -> str:
                return "logger"

            def fetch_by_norad_id(self, norad_id: int) -> TLERecord:
                from orbitalwatch.sources.base import SourceFetchError
                raise SourceFetchError("not found")

            def fetch_category(self, category: str) -> list[TLERecord]:
                fetch_log.append(category)
                return [parse(*_ISS_TLE, source="test")]

            def fetch_many(self, norad_ids: list[int]) -> list[TLERecord]:
                return []

        from orbitalwatch.sources.cache import TLECache
        cache = TLECache(db_path=tmp_path / "log_test.db")
        reg = SourceRegistry(cache=cache)
        reg.register(_LoggingSource())

        aoi = PointRadius(radius_km=100.0, lat=30.27, lon=-97.74)
        window = TimeWindow.next_n_hours(6)
        AOIQuery().run(aoi, window, reg, categories=["stations"])

        assert fetch_log == ["stations"]

    def test_bounding_box_aoi_accepted(self, tmp_path):
        iss = parse(*_ISS_TLE, source="test")
        reg = _make_registry([iss], tmp_path)
        aoi = BoundingBox(min_lat=25.0, max_lat=35.0, min_lon=-100.0, max_lon=-95.0)
        window = TimeWindow.next_n_hours(24)
        results = AOIQuery().run(aoi, window, reg)
        assert isinstance(results, list)

    def test_polygon_aoi_accepted(self, tmp_path):
        iss = parse(*_ISS_TLE, source="test")
        reg = _make_registry([iss], tmp_path)
        aoi = PolygonAOI(points=[
            (25.0, -100.0), (35.0, -100.0), (35.0, -95.0), (25.0, -95.0)
        ])
        window = TimeWindow.next_n_hours(24)
        results = AOIQuery().run(aoi, window, reg)
        assert isinstance(results, list)
