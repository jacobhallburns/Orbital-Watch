"""
Ground station / observer location.

Planned functionality:
- GroundStation dataclass: name, lat (°N), lon (°E), elev_m.
- Helper: from_city(name) — look up common observing sites by name.
- Coordinate conversion helpers: geodetic ↔ ECEF ↔ ECI (via skyfield).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GroundStation:
    """An observer location on Earth's surface.

    Attributes:
        name:   Human-readable label.
        lat:    Geodetic latitude in decimal degrees (north positive).
        lon:    Geodetic longitude in decimal degrees (east positive).
        elev_m: Elevation above WGS-84 ellipsoid in metres.
    """

    name: str
    lat: float
    lon: float
    elev_m: float = 0.0
