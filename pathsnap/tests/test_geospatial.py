"""
Tests for pathsnap.geospatial module.
"""

import numpy as np
import pytest
from pathsnap.geospatial.projection import (
    LocalMetricProjector,
    haversine_distance,
    compute_bearing,
    angular_difference,
    heading_cosine,
)
from pathsnap.geospatial.geometry import (
    point_to_segment_projection,
    segment_length,
)


def test_metric_projector_roundtrip():
    lon0, lat0 = 8.59, 49.84
    projector = LocalMetricProjector(lon0=lon0, lat0=lat0)

    # Origin should be (0, 0)
    origin_m = projector.project(np.array([lon0]), np.array([lat0]))
    assert np.allclose(origin_m, [[0.0, 0.0]], atol=1e-5)

    # Roundtrip test
    test_lons = np.array([8.59, 8.60, 8.58])
    test_lats = np.array([49.84, 49.85, 49.83])
    metric_coords = projector.project(test_lons, test_lats)
    unprojected = projector.unproject(metric_coords[:, 0], metric_coords[:, 1])

    assert np.allclose(unprojected[:, 0], test_lons, atol=1e-7)
    assert np.allclose(unprojected[:, 1], test_lats, atol=1e-7)


def test_bearing_and_angular_difference():
    # Due North: dx = 0, dy = 10 -> bearing = 0 rad
    b_north = compute_bearing(0.0, 10.0)
    assert np.isclose(b_north, 0.0)

    # Due East: dx = 10, dy = 0 -> bearing = pi/2 rad
    b_east = compute_bearing(10.0, 0.0)
    assert np.isclose(b_east, np.pi / 2.0)

    # Angular difference between North and East is pi/2
    diff = angular_difference(b_north, b_east)
    assert np.isclose(diff, np.pi / 2.0)
    assert np.isclose(heading_cosine(b_north, b_east), 0.0, atol=1e-7)

    # Opposite directions: difference is pi, cosine is -1
    b_south = compute_bearing(0.0, -10.0)
    assert np.isclose(angular_difference(b_north, b_south), np.pi)
    assert np.isclose(heading_cosine(b_north, b_south), -1.0)


def test_point_to_segment_projection():
    # Segment along x-axis from (0, 0) to (10, 0)
    a = np.array([0.0, 0.0])
    b = np.array([10.0, 0.0])

    # Point directly above midpoint at (5, 3)
    p = np.array([5.0, 3.0])
    dist, tau, proj_pt = point_to_segment_projection(p, a, b)

    assert np.isclose(dist, 3.0)
    assert np.isclose(tau, 0.5)
    assert np.allclose(proj_pt, [5.0, 0.0])

    # Point past end of segment at (15, 0)
    p_past = np.array([15.0, 0.0])
    dist_past, tau_past, proj_past = point_to_segment_projection(p_past, a, b)
    assert np.isclose(dist_past, 5.0)
    assert np.isclose(tau_past, 1.0)
    assert np.allclose(proj_past, [10.0, 0.0])
