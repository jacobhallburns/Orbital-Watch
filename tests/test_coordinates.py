"""Tests for orbitalwatch.geo.coordinates — AOI types."""

from __future__ import annotations

import pytest
from shapely.geometry import Point, Polygon, box

from orbitalwatch.geo.coordinates import BoundingBox, PointRadius, PolygonAOI


# ---------------------------------------------------------------------------
# PointRadius tests
# ---------------------------------------------------------------------------

class TestPointRadius:
    def test_contains_center(self):
        aoi = PointRadius(radius_km=100.0, lat=30.0, lon=-97.0)
        assert aoi.contains(30.0, -97.0)

    def test_contains_nearby_point(self):
        aoi = PointRadius(radius_km=100.0, lat=30.0, lon=-97.0)
        # ~50 km north (≈0.45°) — should be inside a 100 km circle
        assert aoi.contains(30.45, -97.0)

    def test_excludes_distant_point(self):
        aoi = PointRadius(radius_km=50.0, lat=30.0, lon=-97.0)
        # ~200 km away — should be outside a 50 km radius
        assert not aoi.contains(32.0, -97.0)

    def test_from_mgrs_sets_lat_lon(self):
        import mgrs as mgrs_lib
        m = mgrs_lib.MGRS()
        mgrs_str = m.toMGRS(30.27, -97.74)
        aoi = PointRadius(radius_km=50.0, mgrs=mgrs_str)
        assert abs(aoi.lat - 30.27) < 0.01
        assert abs(aoi.lon - (-97.74)) < 0.01

    def test_from_mgrs_contains_expected_point(self):
        import mgrs as mgrs_lib
        m = mgrs_lib.MGRS()
        mgrs_str = m.toMGRS(30.27, -97.74)
        aoi = PointRadius(radius_km=50.0, mgrs=mgrs_str)
        assert aoi.contains(30.27, -97.74)

    def test_to_shapely_returns_polygon(self):
        from shapely.geometry.base import BaseGeometry
        aoi = PointRadius(radius_km=100.0, lat=30.0, lon=-97.0)
        geom = aoi.to_shapely()
        assert isinstance(geom, BaseGeometry)
        assert geom.geom_type == "Polygon"

    def test_to_geojson_is_dict_with_type(self):
        aoi = PointRadius(radius_km=100.0, lat=30.0, lon=-97.0)
        gj = aoi.to_geojson()
        assert isinstance(gj, dict)
        assert gj.get("type") == "Polygon"
        assert "coordinates" in gj

    def test_bounds_returns_four_floats(self):
        aoi = PointRadius(radius_km=111.32, lat=0.0, lon=0.0)
        min_lat, max_lat, min_lon, max_lon = aoi.bounds()
        assert min_lat < 0 < max_lat
        assert min_lon < 0 < max_lon


# ---------------------------------------------------------------------------
# BoundingBox tests
# ---------------------------------------------------------------------------

class TestBoundingBox:
    def test_contains_interior_point(self):
        aoi = BoundingBox(min_lat=29.0, max_lat=31.0, min_lon=-99.0, max_lon=-96.0)
        assert aoi.contains(30.0, -97.5)

    def test_excludes_point_outside_lat(self):
        aoi = BoundingBox(min_lat=29.0, max_lat=31.0, min_lon=-99.0, max_lon=-96.0)
        assert not aoi.contains(32.0, -97.5)

    def test_excludes_point_outside_lon(self):
        aoi = BoundingBox(min_lat=29.0, max_lat=31.0, min_lon=-99.0, max_lon=-96.0)
        assert not aoi.contains(30.0, -100.0)

    def test_to_shapely_returns_polygon(self):
        aoi = BoundingBox(min_lat=29.0, max_lat=31.0, min_lon=-99.0, max_lon=-96.0)
        geom = aoi.to_shapely()
        assert geom.geom_type == "Polygon"

    def test_to_geojson_valid(self):
        aoi = BoundingBox(min_lat=29.0, max_lat=31.0, min_lon=-99.0, max_lon=-96.0)
        gj = aoi.to_geojson()
        assert gj["type"] == "Polygon"
        assert len(gj["coordinates"][0]) >= 4

    def test_bounds_returns_correct_values(self):
        aoi = BoundingBox(min_lat=29.0, max_lat=31.0, min_lon=-99.0, max_lon=-96.0)
        assert aoi.bounds() == (29.0, 31.0, -99.0, -96.0)


# ---------------------------------------------------------------------------
# PolygonAOI tests
# ---------------------------------------------------------------------------

class TestPolygonAOI:
    _SQUARE = [
        (29.0, -99.0),
        (31.0, -99.0),
        (31.0, -96.0),
        (29.0, -96.0),
    ]

    def test_contains_interior_point(self):
        aoi = PolygonAOI(points=self._SQUARE)
        assert aoi.contains(30.0, -97.5)

    def test_excludes_exterior_point(self):
        aoi = PolygonAOI(points=self._SQUARE)
        assert not aoi.contains(32.0, -97.5)

    def test_to_shapely_returns_polygon(self):
        aoi = PolygonAOI(points=self._SQUARE)
        geom = aoi.to_shapely()
        assert geom.geom_type == "Polygon"

    def test_to_geojson_valid(self):
        aoi = PolygonAOI(points=self._SQUARE)
        gj = aoi.to_geojson()
        assert gj["type"] == "Polygon"

    def test_from_mgrs_points_converts_correctly(self):
        import mgrs as mgrs_lib
        m = mgrs_lib.MGRS()
        corners = [(29.0, -99.0), (31.0, -99.0), (31.0, -96.0), (29.0, -96.0)]
        mgrs_pts = [m.toMGRS(lat, lon) for lat, lon in corners]
        aoi = PolygonAOI(mgrs_points=mgrs_pts)
        assert aoi.contains(30.0, -97.5)
        assert not aoi.contains(32.0, -97.5)

    def test_raises_without_points(self):
        with pytest.raises(ValueError):
            PolygonAOI()

    def test_raises_with_both_points_and_mgrs(self):
        import mgrs as mgrs_lib
        m = mgrs_lib.MGRS()
        pts = [(29.0, -99.0), (31.0, -99.0)]
        mgrs_pts = [m.toMGRS(lat, lon) for lat, lon in pts]
        with pytest.raises(ValueError):
            PolygonAOI(points=pts, mgrs_points=mgrs_pts)

    def test_bounds_reasonable(self):
        aoi = PolygonAOI(points=self._SQUARE)
        min_lat, max_lat, min_lon, max_lon = aoi.bounds()
        assert min_lat < max_lat
        assert min_lon < max_lon
