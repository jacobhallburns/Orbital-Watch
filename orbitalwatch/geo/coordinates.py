"""
Area-of-Interest (AOI) coordinate types.

All three types convert internally to a Shapely geometry and expose a
consistent interface:

    .to_shapely() -> shapely.geometry.BaseGeometry
    .contains(lat, lon) -> bool
    .to_geojson() -> dict  (GeoJSON geometry object)

MGRS support requires the ``mgrs`` package (``pip install mgrs>=1.4``).
Shapely 2.0+ is required.

Coordinate conventions
----------------------
- ``lat`` / ``lon`` are always decimal degrees (WGS-84 geodetic).
- ``lon`` range: -180 to +180 (east positive).
- Shapely geometries use ``(x, y) = (lon, lat)`` internally.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from shapely.geometry import Point, Polygon, box, mapping
from shapely.geometry.base import BaseGeometry


def _get_mgrs():
    """Return an mgrs.MGRS() instance; raise ImportError with a helpful message if
    the package is missing."""
    try:
        import mgrs as _mgrs_lib
    except ImportError as exc:
        raise ImportError(
            "The 'mgrs' package is required for MGRS coordinate support. "
            "Install it with: pip install mgrs>=1.4"
        ) from exc
    return _mgrs_lib.MGRS()


def _mgrs_to_latlon(mgrs_str: str) -> tuple[float, float]:
    """Convert an MGRS string to (lat_deg, lon_deg)."""
    m = _get_mgrs()
    lat, lon = m.toLatLon(mgrs_str)
    return float(lat), float(lon)


@dataclass
class PointRadius:
    """Circular AOI defined by a centre point and radius.

    Provide either ``(lat, lon)`` or ``mgrs``; if ``mgrs`` is supplied the
    ``lat`` / ``lon`` fields are populated automatically in ``__post_init__``.

    Args:
        radius_km: Radius of the circle in kilometres.
        lat:       Geodetic latitude in decimal degrees (WGS-84, north +).
        lon:       Geodetic longitude in decimal degrees (east +).
        mgrs:      Optional MGRS grid reference string.  When provided,
                   overrides ``lat`` and ``lon``.

    Example::

        aoi = PointRadius(radius_km=50.0, lat=30.27, lon=-97.74)
        # or
        aoi = PointRadius(radius_km=50.0, mgrs="14RPU2119849375")
    """

    radius_km: float
    lat: float = 0.0
    lon: float = 0.0
    mgrs: Optional[str] = None

    def __post_init__(self) -> None:
        if self.mgrs is not None:
            self.lat, self.lon = _mgrs_to_latlon(self.mgrs)

    def to_shapely(self) -> BaseGeometry:
        """Return a Shapely polygon approximating the circle.

        The circle is approximated in geographic degrees.  One degree of
        latitude ≈ 111.32 km; the radius is converted to degrees using this
        constant (a conservative approximation valid for moderate latitudes).
        """
        radius_deg = self.radius_km / 111.32
        return Point(self.lon, self.lat).buffer(radius_deg, quad_segs=64)

    def contains(self, lat: float, lon: float) -> bool:
        """Return True if (lat, lon) falls within this AOI."""
        return bool(self.to_shapely().contains(Point(lon, lat)))

    def to_geojson(self) -> dict:
        """Return a GeoJSON geometry dict (type ``Polygon``)."""
        return dict(mapping(self.to_shapely()))

    def bounds(self) -> tuple[float, float, float, float]:
        """Return (min_lat, max_lat, min_lon, max_lon) bounding box."""
        deg = self.radius_km / 111.32
        return self.lat - deg, self.lat + deg, self.lon - deg, self.lon + deg


@dataclass
class BoundingBox:
    """Rectangular AOI aligned to lat/lon axes.

    Args:
        min_lat: Southern boundary (decimal degrees).
        max_lat: Northern boundary (decimal degrees).
        min_lon: Western boundary (decimal degrees, east +).
        max_lon: Eastern boundary (decimal degrees, east +).

    Example::

        aoi = BoundingBox(min_lat=29.0, max_lat=31.0, min_lon=-99.0, max_lon=-96.0)
    """

    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float

    def to_shapely(self) -> BaseGeometry:
        """Return a Shapely box geometry."""
        return box(self.min_lon, self.min_lat, self.max_lon, self.max_lat)

    def contains(self, lat: float, lon: float) -> bool:
        """Return True if (lat, lon) falls within this bounding box."""
        return (
            self.min_lat <= lat <= self.max_lat
            and self.min_lon <= lon <= self.max_lon
        )

    def to_geojson(self) -> dict:
        """Return a GeoJSON geometry dict (type ``Polygon``)."""
        return dict(mapping(self.to_shapely()))

    def bounds(self) -> tuple[float, float, float, float]:
        """Return (min_lat, max_lat, min_lon, max_lon)."""
        return self.min_lat, self.max_lat, self.min_lon, self.max_lon


@dataclass
class PolygonAOI:
    """Arbitrary polygon AOI.

    Provide either ``points`` (list of (lat, lon) tuples) or ``mgrs_points``
    (list of MGRS strings); the two formats are mutually exclusive.

    Args:
        points:      List of (lat, lon) tuples defining the polygon vertices.
        mgrs_points: List of MGRS grid reference strings defining the vertices.

    Example::

        aoi = PolygonAOI(points=[(30.0, -98.0), (31.0, -98.0), (31.0, -97.0), (30.0, -97.0)])
        # or
        aoi = PolygonAOI(mgrs_points=["14RPT0000000000", "14RPT5000000000", ...])
    """

    points: Optional[list[tuple[float, float]]] = None
    mgrs_points: Optional[list[str]] = None

    def __post_init__(self) -> None:
        if self.points is None and self.mgrs_points is None:
            raise ValueError("PolygonAOI requires either 'points' or 'mgrs_points'.")
        if self.points is not None and self.mgrs_points is not None:
            raise ValueError("Provide either 'points' or 'mgrs_points', not both.")

    def _latlon_pairs(self) -> list[tuple[float, float]]:
        """Return (lat, lon) pairs regardless of which input was used."""
        if self.mgrs_points is not None:
            return [_mgrs_to_latlon(p) for p in self.mgrs_points]
        assert self.points is not None
        return self.points

    def to_shapely(self) -> BaseGeometry:
        """Return a Shapely Polygon.

        Note: Shapely uses (x, y) = (lon, lat) ordering internally.
        """
        latlons = self._latlon_pairs()
        coords = [(lon, lat) for lat, lon in latlons]
        return Polygon(coords)

    def contains(self, lat: float, lon: float) -> bool:
        """Return True if (lat, lon) falls within this polygon."""
        return bool(self.to_shapely().contains(Point(lon, lat)))

    def to_geojson(self) -> dict:
        """Return a GeoJSON geometry dict (type ``Polygon``)."""
        return dict(mapping(self.to_shapely()))

    def bounds(self) -> tuple[float, float, float, float]:
        """Return (min_lat, max_lat, min_lon, max_lon) of the bounding box."""
        geom = self.to_shapely()
        minx, miny, maxx, maxy = geom.bounds
        return miny, maxy, minx, maxx
