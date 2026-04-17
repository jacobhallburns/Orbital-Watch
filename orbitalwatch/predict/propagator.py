"""
SGP4/SDP4 propagation wrapper.

Wraps the sgp4 library with TLERecord as input, returning ECI position
and velocity as numpy arrays. Supports both single-epoch and vectorised
propagation over a time array.
"""

from __future__ import annotations

from datetime import datetime
from typing import Union

import numpy as np
from sgp4.api import Satrec, jday

from orbitalwatch.tle.model import TLERecord

_SGP4_ERRORS: dict[int, str] = {
    1: "mean eccentricity not in [0, 1) or semi-major axis < 0.95 er",
    2: "mean motion is less than 0.0",
    3: "perturbed eccentricity not in [0, 1)",
    4: "semi-latus rectum is less than 0.0",
    5: "epoch elements are sub-orbital",
    6: "satellite has decayed",
}


class PropagationError(Exception):
    """Raised when SGP4 propagation returns a non-zero error code."""


def _make_satrec(tle: TLERecord) -> Satrec:
    return Satrec.twoline2rv(tle.line1, tle.line2)


def _to_jd(utc: datetime) -> tuple[float, float]:
    return jday(
        utc.year, utc.month, utc.day,
        utc.hour, utc.minute, utc.second + utc.microsecond / 1_000_000,
    )


def propagate(tle: TLERecord, utc: datetime) -> tuple[np.ndarray, np.ndarray]:
    """Propagate satellite to a single UTC datetime.

    Returns (position_km, velocity_km_s), each a (3,) numpy array in the
    TEME frame (as produced by SGP4).

    Raises:
        PropagationError: if SGP4 returns a non-zero error code.
    """
    sat = _make_satrec(tle)
    jd, fr = _to_jd(utc)
    e, r, v = sat.sgp4(jd, fr)
    if e != 0:
        msg = _SGP4_ERRORS.get(e, f"SGP4 error code {e}")
        raise PropagationError(f"SGP4 propagation failed: {msg}")
    return np.array(r, dtype=float), np.array(v, dtype=float)


def propagate_many(
    tle: TLERecord,
    utcs: Union[list[datetime], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised propagation over an array of UTC datetimes.

    Args:
        tle:  The satellite TLE record.
        utcs: Sequence of timezone-aware UTC datetime objects.

    Returns:
        (positions_km, velocities_km_s) as (N, 3) numpy arrays in the
        TEME frame.

    Raises:
        PropagationError: if any propagation step returns a non-zero error.
    """
    sat = _make_satrec(tle)
    n = len(utcs)
    jds = np.empty(n, dtype=float)
    frs = np.empty(n, dtype=float)
    for i, utc in enumerate(utcs):
        jds[i], frs[i] = _to_jd(utc)

    e, r, v = sat.sgp4_array(jds, frs)

    bad = np.flatnonzero(e != 0)
    if bad.size > 0:
        code = int(e[bad[0]])
        msg = _SGP4_ERRORS.get(code, f"SGP4 error code {code}")
        raise PropagationError(
            f"SGP4 propagation failed at index {bad[0]}: {msg}"
        )

    return r, v
