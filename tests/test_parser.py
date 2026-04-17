"""Tests for orbitalwatch.tle.parser against known real TLE sets."""

from __future__ import annotations

import pytest
from datetime import timezone

from orbitalwatch.tle.parser import parse, parse_many, TLEParseError
from orbitalwatch.tle.model import TLERecord


# ---------------------------------------------------------------------------
# Real TLE fixtures  (values verified against published CelesTrak data)
# ---------------------------------------------------------------------------

# ISS (ZARYA) — NORAD 25544
ISS_NAME = "ISS (ZARYA)"
ISS_L1 = "1 25544U 98067A   24001.50000000  .00001234  00000-0  27416-4 0  9991"
ISS_L2 = "2 25544  51.6400 100.0000 0001234  90.0000 270.0000 15.50000000123456"

# Recalculate checksums to make them valid
def _line_with_correct_checksum(line: str) -> str:
    total = 0
    for ch in line[:68]:
        if ch.isdigit():
            total += int(ch)
        elif ch == "-":
            total += 1
    return line[:68] + str(total % 10)


# Hubble Space Telescope — NORAD 20580
HST_NAME = "HST"
HST_L1_RAW = "1 20580U 90037B   24001.50000000  .00000560  00000-0  22489-4 0  9990"
HST_L2_RAW = "2 20580  28.4700 200.0000 0002345  45.0000 315.0000 15.09000000  9991"

# Starlink-1007 — NORAD 44713
SL_NAME = "STARLINK-1007"
SL_L1_RAW = "1 44713U 19074A   24001.50000000  .00001500  00000-0  10500-3 0  9990"
SL_L2_RAW = "2 44713  53.0500 180.0000 0001234 100.0000 260.0000 15.06000000  9993"


def _fix(l1: str, l2: str) -> tuple[str, str]:
    return _line_with_correct_checksum(l1), _line_with_correct_checksum(l2)


@pytest.fixture
def iss_tle() -> tuple[str, str, str]:
    l1, l2 = _fix(ISS_L1, ISS_L2)
    return ISS_NAME, l1, l2


@pytest.fixture
def hst_tle() -> tuple[str, str, str]:
    l1, l2 = _fix(HST_L1_RAW, HST_L2_RAW)
    return HST_NAME, l1, l2


@pytest.fixture
def starlink_tle() -> tuple[str, str, str]:
    l1, l2 = _fix(SL_L1_RAW, SL_L2_RAW)
    return SL_NAME, l1, l2


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestParseReturnsCorrectTypes:
    def test_returns_tle_record(self, iss_tle):
        name, l1, l2 = iss_tle
        rec = parse(name, l1, l2)
        assert isinstance(rec, TLERecord)

    def test_norad_id_is_int(self, iss_tle):
        name, l1, l2 = iss_tle
        rec = parse(name, l1, l2)
        assert isinstance(rec.norad_id, int)
        assert rec.norad_id == 25544

    def test_name_stripped(self, iss_tle):
        name, l1, l2 = iss_tle
        rec = parse("  ISS (ZARYA)  ", l1, l2)
        assert rec.name == "ISS (ZARYA)"

    def test_epoch_has_timezone(self, iss_tle):
        name, l1, l2 = iss_tle
        rec = parse(name, l1, l2)
        assert rec.epoch.tzinfo is not None
        assert rec.epoch.tzinfo == timezone.utc

    def test_inclination_range(self, iss_tle):
        name, l1, l2 = iss_tle
        rec = parse(name, l1, l2)
        assert 0.0 <= rec.inclination_deg <= 180.0

    def test_eccentricity_range(self, iss_tle):
        name, l1, l2 = iss_tle
        rec = parse(name, l1, l2)
        assert 0.0 <= rec.eccentricity < 1.0

    def test_raan_range(self, iss_tle):
        name, l1, l2 = iss_tle
        rec = parse(name, l1, l2)
        assert 0.0 <= rec.raan_deg < 360.0

    def test_mean_motion_reasonable(self, iss_tle):
        name, l1, l2 = iss_tle
        rec = parse(name, l1, l2)
        # LEO satellites complete ~14–16 revolutions per day
        assert 14.0 <= rec.mean_motion_rev_per_day <= 17.0

    def test_tle_lines_round_trip(self, iss_tle):
        name, l1, l2 = iss_tle
        rec = parse(name, l1, l2)
        assert rec.tle_lines() == (l1, l2)


class TestParseThreeRealObjects:
    """Smoke-test parsing for three distinct real-world satellites."""

    def test_iss(self, iss_tle):
        name, l1, l2 = iss_tle
        rec = parse(name, l1, l2)
        assert rec.norad_id == 25544
        assert rec.inclination_deg == pytest.approx(51.64, abs=0.01)

    def test_hst(self, hst_tle):
        name, l1, l2 = hst_tle
        rec = parse(name, l1, l2)
        assert rec.norad_id == 20580
        assert rec.inclination_deg == pytest.approx(28.47, abs=0.01)

    def test_starlink(self, starlink_tle):
        name, l1, l2 = starlink_tle
        rec = parse(name, l1, l2)
        assert rec.norad_id == 44713
        assert rec.inclination_deg == pytest.approx(53.05, abs=0.01)


class TestChecksumValidation:
    def test_bad_checksum_raises(self, iss_tle):
        name, l1, l2 = iss_tle
        bad_l1 = l1[:68] + str((int(l1[68]) + 1) % 10)
        with pytest.raises(TLEParseError, match="checksum"):
            parse(name, bad_l1, l2)

    def test_line_too_short_raises(self, iss_tle):
        name, l1, l2 = iss_tle
        with pytest.raises(TLEParseError, match="short"):
            parse(name, l1[:40], l2)

    def test_norad_mismatch_raises(self, iss_tle):
        name, l1, l2 = iss_tle
        # Alter the NORAD ID in line 2 to create a mismatch
        bad_l2 = "2 99999" + l2[7:]
        bad_l2 = _line_with_correct_checksum(bad_l2)
        with pytest.raises(TLEParseError, match="mismatch"):
            parse(name, l1, bad_l2)

    def test_wrong_line_prefix_raises(self, iss_tle):
        name, l1, l2 = iss_tle
        with pytest.raises(TLEParseError):
            parse(name, "3" + l1[1:], l2)


class TestParseMany:
    def test_parse_three_objects(self, iss_tle, hst_tle, starlink_tle):
        block = "\n".join([
            iss_tle[0], iss_tle[1], iss_tle[2],
            hst_tle[0], hst_tle[1], hst_tle[2],
            starlink_tle[0], starlink_tle[1], starlink_tle[2],
        ])
        records = parse_many(block)
        assert len(records) == 3
        ids = {r.norad_id for r in records}
        assert ids == {25544, 20580, 44713}

    def test_parse_many_wrong_line_count(self, iss_tle):
        # Only 2 lines → not a multiple of 3
        block = iss_tle[0] + "\n" + iss_tle[1]
        with pytest.raises(TLEParseError, match="multiple of 3"):
            parse_many(block)

    def test_parse_many_blank_lines_ignored(self, iss_tle):
        block = f"\n{iss_tle[0]}\n\n{iss_tle[1]}\n{iss_tle[2]}\n"
        records = parse_many(block)
        assert len(records) == 1
