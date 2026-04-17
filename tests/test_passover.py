"""
Tests for orbitalwatch.predict.passover.

Uses ISS over Austin TX across a 24-hour window to validate PassEvent
fields and that the predictor finds real passes.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from orbitalwatch.geo.location import GroundStation
from orbitalwatch.predict.passover import PassEvent, PassoverPredictor
from orbitalwatch.tle.model import TLERecord

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
_ISS_NAME = "ISS (ZARYA)"
_ISS_L1 = "1 25544U 98067A   23027.52753472  .00011243  00000-0  20659-3 0  9992"
_ISS_L2 = "2 25544  51.6421 163.2570 0003972  42.3938  91.3924 15.49937454379798"

AUSTIN = GroundStation(name="Austin TX", lat=30.2672, lon=-97.7431, elev_m=150.0)

_WINDOW_START = datetime(2023, 1, 27, 0, 0, 0, tzinfo=timezone.utc)
_WINDOW_END = _WINDOW_START + timedelta(hours=24)
_MIN_EL = 10.0


def _parse_iss() -> TLERecord:
    epoch_str = _ISS_L1[18:32].strip()
    yr2 = int(epoch_str[:2])
    year = 2000 + yr2 if yr2 < 57 else 1900 + yr2
    epoch = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=float(epoch_str[2:]) - 1)

    return TLERecord(
        name=_ISS_NAME,
        norad_id=int(_ISS_L1[2:7]),
        classification=_ISS_L1[7],
        int_designator=_ISS_L1[9:17].strip(),
        epoch=epoch,
        mean_motion_dot=float(_ISS_L1[33:43]),
        mean_motion_ddot=0.0,
        bstar=0.0,
        element_set_no=int(_ISS_L1[64:68]),
        rev_at_epoch=int(_ISS_L2[63:68]),
        inclination_deg=float(_ISS_L2[8:16]),
        raan_deg=float(_ISS_L2[17:25]),
        eccentricity=float("0." + _ISS_L2[26:33]),
        arg_perigee_deg=float(_ISS_L2[34:42]),
        mean_anomaly_deg=float(_ISS_L2[43:51]),
        mean_motion_rev_per_day=float(_ISS_L2[52:63]),
        line1=_ISS_L1,
        line2=_ISS_L2,
    )


@pytest.fixture(scope="module")
def iss_passes() -> list[PassEvent]:
    predictor = PassoverPredictor()
    return predictor.passes_in_window(
        _parse_iss(), AUSTIN, _WINDOW_START, _WINDOW_END, min_elevation_deg=_MIN_EL
    )


# ---------------------------------------------------------------------------
# passes_in_window tests
# ---------------------------------------------------------------------------

class TestPassesInWindow:
    def test_at_least_two_passes_in_24h(self, iss_passes: list[PassEvent]):
        assert len(iss_passes) >= 2, (
            f"Expected >= 2 ISS passes over Austin TX in 24 h, got {len(iss_passes)}"
        )

    def test_rise_before_set(self, iss_passes: list[PassEvent]):
        for p in iss_passes:
            assert p.rise_utc < p.set_utc, (
                f"rise_utc {p.rise_utc} is not before set_utc {p.set_utc}"
            )

    def test_duration_matches_rise_set_difference(self, iss_passes: list[PassEvent]):
        for p in iss_passes:
            expected = (p.set_utc - p.rise_utc).total_seconds()
            assert abs(p.duration_seconds - expected) < 1.0, (
                f"duration_seconds={p.duration_seconds:.1f} s but "
                f"set-rise={expected:.1f} s (diff > 1 s)"
            )

    def test_max_elevation_above_minimum(self, iss_passes: list[PassEvent]):
        for p in iss_passes:
            assert p.max_elevation_deg >= _MIN_EL, (
                f"max_elevation_deg={p.max_elevation_deg:.2f}° < min {_MIN_EL}°"
            )

    def test_azimuths_in_valid_range(self, iss_passes: list[PassEvent]):
        for p in iss_passes:
            for az, label in [
                (p.rise_az, "rise_az"),
                (p.max_az, "max_az"),
                (p.set_az, "set_az"),
            ]:
                assert 0.0 <= az < 360.0, f"{label}={az:.2f}° is outside [0, 360)"

    def test_passes_ordered_chronologically(self, iss_passes: list[PassEvent]):
        for a, b in zip(iss_passes, iss_passes[1:]):
            assert a.set_utc <= b.rise_utc, (
                f"Pass overlap: previous set {a.set_utc} > next rise {b.rise_utc}"
            )

    def test_passes_within_requested_window(self, iss_passes: list[PassEvent]):
        for p in iss_passes:
            assert p.rise_utc >= _WINDOW_START
            assert p.set_utc <= _WINDOW_END


# ---------------------------------------------------------------------------
# next_passes tests
# ---------------------------------------------------------------------------

class TestNextPasses:
    def test_returns_exact_count(self):
        predictor = PassoverPredictor()
        passes = predictor.next_passes(_parse_iss(), AUSTIN, count=3, start_utc=_WINDOW_START)
        assert len(passes) == 3

    def test_next_passes_are_non_overlapping(self):
        predictor = PassoverPredictor()
        passes = predictor.next_passes(_parse_iss(), AUSTIN, count=3, start_utc=_WINDOW_START)
        for a, b in zip(passes, passes[1:]):
            assert a.set_utc <= b.rise_utc, (
                f"Overlapping passes: set {a.set_utc} > rise {b.rise_utc}"
            )

    def test_next_passes_start_after_start_utc(self):
        predictor = PassoverPredictor()
        passes = predictor.next_passes(_parse_iss(), AUSTIN, count=2, start_utc=_WINDOW_START)
        for p in passes:
            assert p.rise_utc >= _WINDOW_START
