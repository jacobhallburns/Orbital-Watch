"""
Live integration tests that hit real CelesTrak endpoints.

These are skipped by default.  Run them explicitly with:

    pytest -m live
    pytest tests/test_live.py

They require a working internet connection and are intentionally
excluded from CI to avoid rate-limiting CelesTrak.
"""

from __future__ import annotations

import pytest

from orbitalwatch.sources.celestrak import CelesTrakSource, KNOWN_CATEGORIES
from orbitalwatch.tle.model import TLERecord

pytestmark = pytest.mark.live  # applied to every test in this module


@pytest.fixture(scope="module")
def source() -> CelesTrakSource:
    return CelesTrakSource(timeout=20)


# ---------------------------------------------------------------------------
# fetch_by_norad_id
# ---------------------------------------------------------------------------

class TestLiveFetchByNoradId:
    def test_iss_returns_tle_record(self, source):
        rec = source.fetch_by_norad_id(25544)
        assert isinstance(rec, TLERecord)

    def test_iss_norad_id_correct(self, source):
        rec = source.fetch_by_norad_id(25544)
        assert rec.norad_id == 25544

    def test_iss_name_contains_iss(self, source):
        rec = source.fetch_by_norad_id(25544)
        assert "ISS" in rec.name.upper()

    def test_iss_inclination_near_51_deg(self, source):
        rec = source.fetch_by_norad_id(25544)
        # ISS inclination is ~51.6°; allow ±1° for orbital evolution
        assert 50.0 <= rec.inclination_deg <= 53.0

    def test_iss_mean_motion_leo_range(self, source):
        rec = source.fetch_by_norad_id(25544)
        assert 14.0 <= rec.mean_motion_rev_per_day <= 17.0

    def test_iss_eccentricity_near_zero(self, source):
        rec = source.fetch_by_norad_id(25544)
        assert rec.eccentricity < 0.01

    def test_iss_source_label(self, source):
        rec = source.fetch_by_norad_id(25544)
        assert rec.source == "celestrak"

    def test_iss_epoch_has_utc_timezone(self, source):
        from datetime import timezone
        rec = source.fetch_by_norad_id(25544)
        assert rec.epoch.tzinfo is not None
        assert rec.epoch.tzinfo == timezone.utc

    def test_iss_tle_lines_valid_format(self, source):
        rec = source.fetch_by_norad_id(25544)
        l1, l2 = rec.tle_lines()
        assert l1.startswith("1 ")
        assert l2.startswith("2 ")
        assert len(l1) == 69
        assert len(l2) == 69

    def test_hst_norad_id(self, source):
        rec = source.fetch_by_norad_id(20580)
        assert rec.norad_id == 20580

    def test_hst_inclination_near_28_deg(self, source):
        rec = source.fetch_by_norad_id(20580)
        assert 27.0 <= rec.inclination_deg <= 30.0


# ---------------------------------------------------------------------------
# fetch_category
# ---------------------------------------------------------------------------

class TestLiveFetchCategory:
    def test_stations_returns_list(self, source):
        records = source.fetch_category("stations")
        assert isinstance(records, list)

    def test_stations_nonempty(self, source):
        records = source.fetch_category("stations")
        assert len(records) > 0

    def test_stations_all_tle_records(self, source):
        records = source.fetch_category("stations")
        assert all(isinstance(r, TLERecord) for r in records)

    def test_stations_contains_iss(self, source):
        records = source.fetch_category("stations")
        ids = {r.norad_id for r in records}
        assert 25544 in ids

    def test_visual_category_nonempty(self, source):
        # 'visual' is a small, consistently-open group (~100 objects).
        # Avoid 'active'/'starlink' which 403 for anonymous programmatic access.
        records = source.fetch_category("visual")
        assert len(records) > 10

    def test_all_records_have_valid_inclination(self, source):
        records = source.fetch_category("stations")
        for r in records:
            assert 0.0 <= r.inclination_deg <= 180.0, (
                f"{r.name}: inclination {r.inclination_deg} out of range"
            )

    def test_all_records_have_valid_eccentricity(self, source):
        records = source.fetch_category("stations")
        for r in records:
            assert 0.0 <= r.eccentricity < 1.0, (
                f"{r.name}: eccentricity {r.eccentricity} out of range"
            )


# ---------------------------------------------------------------------------
# fetch_active
# ---------------------------------------------------------------------------

class TestLiveFetchActive:
    def test_returns_nonempty_catalog(self, source):
        # CelesTrak 403s the full 'active' catalog for anonymous programmatic
        # access from some IPs; fetch_active() still works when it succeeds.
        # This test is marked xfail on 403 so the suite stays green in CI.
        try:
            records = source.fetch_active()
        except Exception as exc:
            if "403" in str(exc):
                pytest.xfail("CelesTrak 403 on active catalog from this IP")
            raise
        assert len(records) > 0

    def test_all_are_tle_records(self, source):
        try:
            records = source.fetch_active()
        except Exception as exc:
            if "403" in str(exc):
                pytest.xfail("CelesTrak 403 on active catalog from this IP")
            raise
        assert all(isinstance(r, TLERecord) for r in records)

    def test_no_duplicate_norad_ids(self, source):
        try:
            records = source.fetch_active()
        except Exception as exc:
            if "403" in str(exc):
                pytest.xfail("CelesTrak 403 on active catalog from this IP")
            raise
        ids = [r.norad_id for r in records]
        assert len(ids) == len(set(ids)), "Duplicate NORAD IDs in active catalog"
