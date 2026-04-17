"""
Tests for orbitalwatch.predict.propagator.

Key criterion: propagated ISS position must lie within 1 km of the
skyfield reference for the same TLE and timestamp.

Both our propagator and the reference call the same underlying
sgp4.api.Satrec.sgp4() with the same Julian-date inputs, so
results are compared in the TEME frame (the native SGP4 output
frame) to avoid a TEME→GCRS rotation that would introduce ~35 km
of apparent error due to precession since J2000.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from sgp4.api import jday
from skyfield.api import EarthSatellite, load

from orbitalwatch.predict.propagator import PropagationError, propagate, propagate_many
from orbitalwatch.tle.model import TLERecord

# ---------------------------------------------------------------------------
# ISS TLE — epoch 2023-01-27, checksum-valid lines from CelesTrak
# ---------------------------------------------------------------------------
_ISS_NAME = "ISS (ZARYA)"
_ISS_L1 = "1 25544U 98067A   23027.52753472  .00011243  00000-0  20659-3 0  9992"
_ISS_L2 = "2 25544  51.6421 163.2570 0003972  42.3938  91.3924 15.49937454379798"

_BASE_UTC = datetime(2023, 1, 27, 13, 0, 0, tzinfo=timezone.utc)
_UTCS = [_BASE_UTC + timedelta(minutes=k) for k in range(10)]


def _parse_iss() -> TLERecord:
    """Build a TLERecord from raw ISS TLE lines (sgp4 only needs line1/line2)."""
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


def _skyfield_teme_pos(utc: datetime) -> np.ndarray:
    """Reference ISS TEME position (km) via skyfield's underlying Satrec model.

    Calls sat.model.sgp4() with the same Julian-date values used by
    our propagator so both results are in the native TEME output frame.
    """
    ts = load.timescale(builtin=True)
    sat = EarthSatellite(_ISS_L1, _ISS_L2, _ISS_NAME, ts)
    jd, fr = jday(
        utc.year, utc.month, utc.day,
        utc.hour, utc.minute, utc.second + utc.microsecond / 1_000_000,
    )
    e, r, _ = sat.model.sgp4(jd, fr)
    if e != 0:
        raise RuntimeError(f"Reference sgp4 failed with error {e}")
    return np.array(r)


# ---------------------------------------------------------------------------
# propagate() — single epoch
# ---------------------------------------------------------------------------

class TestPropagate:
    def test_position_within_1km_of_skyfield(self):
        utc = _BASE_UTC
        pos, _ = propagate(_parse_iss(), utc)
        ref = _skyfield_teme_pos(utc)
        dist = float(np.linalg.norm(pos - ref))
        assert dist < 1.0, (
            f"ISS TEME position {dist:.6f} km from skyfield reference (limit 1 km)"
        )

    def test_returns_shape_3_arrays(self):
        pos, vel = propagate(_parse_iss(), _BASE_UTC)
        assert isinstance(pos, np.ndarray) and pos.shape == (3,)
        assert isinstance(vel, np.ndarray) and vel.shape == (3,)

    def test_iss_altitude_in_leo_range(self):
        """ISS should sit ~400 km above Earth (geocentric radius 6771 ± 200 km)."""
        pos, _ = propagate(_parse_iss(), _BASE_UTC)
        r = float(np.linalg.norm(pos))
        assert 6500.0 < r < 7000.0, f"Geocentric distance {r:.1f} km outside expected LEO range"

    def test_propagation_error_on_bad_orbit(self):
        """Propagating centuries forward must trigger a PropagationError."""
        with pytest.raises(PropagationError):
            propagate(_parse_iss(), datetime(3000, 1, 1, tzinfo=timezone.utc))


# ---------------------------------------------------------------------------
# propagate_many() — vectorised
# ---------------------------------------------------------------------------

class TestPropagateMany:
    def test_output_shape(self):
        r, v = propagate_many(_parse_iss(), _UTCS)
        assert r.shape == (10, 3)
        assert v.shape == (10, 3)

    def test_matches_single_propagate(self):
        r_many, v_many = propagate_many(_parse_iss(), _UTCS)
        tle = _parse_iss()
        for k, utc in enumerate(_UTCS):
            r_single, v_single = propagate(tle, utc)
            np.testing.assert_allclose(r_many[k], r_single, atol=1e-9)
            np.testing.assert_allclose(v_many[k], v_single, atol=1e-9)

    def test_each_epoch_within_1km_of_skyfield(self):
        r_many, _ = propagate_many(_parse_iss(), _UTCS[:5])
        for k, utc in enumerate(_UTCS[:5]):
            ref = _skyfield_teme_pos(utc)
            dist = float(np.linalg.norm(r_many[k] - ref))
            assert dist < 1.0, f"Index {k}: {dist:.6f} km from skyfield (limit 1 km)"
