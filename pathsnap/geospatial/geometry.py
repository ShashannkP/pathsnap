"""
pathsnap.geospatial.geometry
============================
Geometric calculation routines for points, vectors, and directed line segments in 2D metric space.
"""

import numpy as np
from typing import Tuple


def point_to_segment_projection(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """
    Computes orthogonal projection of point p onto line segment ab in 2D metric space.

    Parameters
    ----------
    p : array-like of shape (2,)
        Query point [x, y].
    a : array-like of shape (2,)
        Start node of road segment [x, y].
    b : array-like of shape (2,)
        End node of road segment [x, y].

    Returns
    -------
    dist : float
        Perpendicular distance from p to the nearest point on segment ab in meters.
    tau : float
        Scalar projection factor along ab:
        tau = 0.0 at point a, tau = 1.0 at point b. Clamped to [0.0, 1.0].
    proj_point : ndarray of shape (2,)
        Projected point coordinates on segment ab.
    """
    ab = b - a
    ab_len_sq = float(np.dot(ab, ab))

    if ab_len_sq < 1e-12:
        dist = float(np.linalg.norm(p - a))
        return dist, 0.0, a.copy()

    t_unclamped = float(np.dot(p - a, ab) / ab_len_sq)
    tau = max(0.0, min(1.0, t_unclamped))
    proj_point = a + tau * ab
    dist = float(np.linalg.norm(p - proj_point))

    return dist, tau, proj_point


def segment_length(a: np.ndarray, b: np.ndarray) -> float:
    """
    Computes Euclidean length of segment ab in meters.
    """
    return float(np.linalg.norm(b - a))
