"""
Ground-station pass prediction.

Given a TLERecord and a GroundStation, finds overhead passes using a
coarse 60-second scan to detect horizon crossings, bisection to refine
rise/set times, and a fine scan to locate max elevation.

Skyfield handles all ECI → topocentric az/el transforms.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Union

import numpy as np
from skyfield.api import EarthSatellite, load, wgs84

from orbitalwatch.geo.location import GroundStation
from orbitalwatch.tle.model import TLERecord


@dataclass
class PassEvent:
    rise_utc: datetime
    set_utc: datetime
    max_elevation_deg: float
    rise_az: float
    max_az: float
    set_az: float
    duration_seconds: float


class PassoverPredictor:
    """Predicts overhead satellite passes for a ground station."""

    def __init__(self, coarse_step_s: int = 60) -> None:
        self._step_s = coarse_step_s
        self._ts = load.timescale(builtin=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_sat(self, tle: TLERecord) -> EarthSatellite:
        return EarthSatellite(tle.line1, tle.line2, tle.name, self._ts)

    def _make_topos(self, station: GroundStation):
        return wgs84.latlon(station.lat, station.lon, elevation_m=station.elev_m)

    def _altaz_at(self, sat, topos, utc: datetime) -> tuple[float, float]:
        """Return (elevation_deg, azimuth_deg) for a single UTC datetime."""
        t = self._ts.from_datetime(utc)
        alt, az, _ = (sat - topos).at(t).altaz()
        return float(alt.degrees), float(az.degrees)

    def _elevation_at(self, sat, topos, utc: datetime) -> float:
        return self._altaz_at(sat, topos, utc)[0]

    def _altaz_array(
        self, sat, topos, times: list[datetime]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised az/el for a list of UTC datetimes."""
        t = self._ts.from_datetimes(times)
        alt, az, _ = (sat - topos).at(t).altaz()
        return alt.degrees, az.degrees

    def _bisect_crossing(
        self,
        sat,
        topos,
        t_lo: datetime,
        t_hi: datetime,
        rising: bool,
        tol_s: float = 0.5,
    ) -> datetime:
        """Narrow horizon crossing to within tol_s seconds.

        Convention: when rising=True, elevation goes from negative (t_lo)
        to non-negative (t_hi); when False, the reverse.
        """
        while (t_hi - t_lo).total_seconds() > tol_s:
            t_mid = t_lo + (t_hi - t_lo) / 2
            el = self._elevation_at(sat, topos, t_mid)
            # rising=True:  el>=0 → crossing is in [t_lo, t_mid] → shrink t_hi
            # rising=False: el>=0 → crossing is in [t_mid, t_hi] → shrink t_lo
            if (el >= 0) == rising:
                t_hi = t_mid
            else:
                t_lo = t_mid
        return t_lo + (t_hi - t_lo) / 2

    def _find_max_elevation(
        self,
        sat,
        topos,
        rise_time: datetime,
        set_time: datetime,
        fine_step_s: float = 10.0,
    ) -> tuple[float, float, datetime]:
        """Return (max_el_deg, az_at_max_deg, time_of_max) via fine scan."""
        duration_s = (set_time - rise_time).total_seconds()
        n = max(int(duration_s / fine_step_s), 10)
        times = [
            rise_time + timedelta(seconds=duration_s * k / n) for k in range(n + 1)
        ]
        elevs, azs = self._altaz_array(sat, topos, times)
        idx = int(np.argmax(elevs))
        return float(elevs[idx]), float(azs[idx]), times[idx]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def passes_in_window(
        self,
        tle: TLERecord,
        station: GroundStation,
        start_utc: datetime,
        end_utc: datetime,
        min_elevation_deg: float = 10.0,
    ) -> list[PassEvent]:
        """Return all passes with max elevation >= min_elevation_deg in [start, end]."""
        sat = self._make_sat(tle)
        topos = self._make_topos(station)

        step = timedelta(seconds=self._step_s)
        times: list[datetime] = []
        t = start_utc
        while t <= end_utc:
            times.append(t)
            t += step
        if len(times) < 2:
            return []

        elevs, _ = self._altaz_array(sat, topos, times)
        passes: list[PassEvent] = []
        i = 0

        # Handle satellite already above horizon at window start.
        if elevs[0] >= 0:
            rise_time = start_utc
            j = 0
            while j < len(times) - 1 and elevs[j] >= 0:
                j += 1
            set_time = (
                end_utc
                if elevs[j] >= 0
                else self._bisect_crossing(sat, topos, times[j - 1], times[j], rising=False)
            )
            max_el, max_az, _ = self._find_max_elevation(sat, topos, rise_time, set_time)
            if max_el >= min_elevation_deg:
                _, rise_az = self._altaz_at(sat, topos, rise_time)
                _, set_az = self._altaz_at(sat, topos, set_time)
                passes.append(PassEvent(
                    rise_utc=rise_time,
                    set_utc=set_time,
                    max_elevation_deg=max_el,
                    rise_az=rise_az,
                    max_az=max_az,
                    set_az=set_az,
                    duration_seconds=(set_time - rise_time).total_seconds(),
                ))
            i = j

        while i < len(times) - 1:
            if elevs[i] < 0 and elevs[i + 1] >= 0:
                rise_time = self._bisect_crossing(
                    sat, topos, times[i], times[i + 1], rising=True
                )
                j = i + 1
                while j < len(times) - 1 and elevs[j] >= 0:
                    j += 1
                set_time = (
                    end_utc
                    if elevs[j] >= 0
                    else self._bisect_crossing(
                        sat, topos, times[j - 1], times[j], rising=False
                    )
                )
                max_el, max_az, _ = self._find_max_elevation(sat, topos, rise_time, set_time)
                if max_el >= min_elevation_deg:
                    _, rise_az = self._altaz_at(sat, topos, rise_time)
                    _, set_az = self._altaz_at(sat, topos, set_time)
                    passes.append(PassEvent(
                        rise_utc=rise_time,
                        set_utc=set_time,
                        max_elevation_deg=max_el,
                        rise_az=rise_az,
                        max_az=max_az,
                        set_az=set_az,
                        duration_seconds=(set_time - rise_time).total_seconds(),
                    ))
                i = j
            else:
                i += 1

        return passes

    def next_passes(
        self,
        tle: TLERecord,
        station: GroundStation,
        count: int,
        start_utc: datetime,
        min_elevation_deg: float = 10.0,
    ) -> list[PassEvent]:
        """Return the next `count` passes after start_utc."""
        results: list[PassEvent] = []
        window_start = start_utc
        for _ in range(30):
            if len(results) >= count:
                break
            window_end = window_start + timedelta(days=1)
            results.extend(
                self.passes_in_window(tle, station, window_start, window_end, min_elevation_deg)
            )
            window_start = window_end
        return results[:count]
