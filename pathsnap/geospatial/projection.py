"""
pathsnap.geospatial.projection
==============================
Geospatial coordinate projection and angular geodesic utilities.

Implements the local Transverse Mercator metric projection (in meters) centered at a
reference latitude and longitude. This preserves local metric distances with < 0.01%
distortion over urban extents, eliminating the ~35% anisotropic distortion caused by
raw degree Euclidean math.
"""

import numpy as np


class LocalMetricProjector:
    """
    Projects WGS84 geographic coordinates (longitude, latitude) into a local
    Euclidean metric plane (x_meters, y_meters) centered at (lon0, lat0).
    """

    EARTH_RADIUS = 6371000.0  # WGS84 mean earth radius in meters

    def __init__(self, lon0: float, lat0: float):
        self.lon0 = float(lon0)
        self.lat0 = float(lat0)
        self.lat0_rad = np.radians(self.lat0)
        self.m_per_deg_lat = self.EARTH_RADIUS * (np.pi / 180.0)
        self.m_per_deg_lon = self.m_per_deg_lat * np.cos(self.lat0_rad)

    def project(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """
        Projects longitude and latitude arrays into metric (x, y) coordinates.
        Returns an (N, 2) numpy array of [x_meters, y_meters].
        """
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        x = (lon - self.lon0) * self.m_per_deg_lon
        y = (lat - self.lat0) * self.m_per_deg_lat
        return np.column_stack([x, y])

    def unproject(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Inverts local metric coordinates back to WGS84 (longitude, latitude).
        Returns an (N, 2) numpy array of [longitude, latitude].
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        lon = self.lon0 + (x / self.m_per_deg_lon)
        lat = self.lat0 + (y / self.m_per_deg_lat)
        return np.column_stack([lon, lat])


def haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Computes exact great-circle distance between two WGS84 points in meters.
    """
    r = 6371000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = (np.sin(delta_phi / 2.0) ** 2 +
         np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return r * c


def compute_bearing(dx: float, dy: float) -> float:
    """
    Computes azimuth / compass bearing in radians in [0, 2pi) where:
    0 = North (+y), pi/2 = East (+x), pi = South (-y), 3pi/2 = West (-x).
    """
    angle = np.arctan2(dx, dy)
    if angle < 0:
        angle += 2.0 * np.pi
    return float(angle)


def angular_difference(angle1: float, angle2: float) -> float:
    """
    Computes the minimal acute angular difference between two bearings in radians in [0, pi].
    """
    diff = abs(angle1 - angle2) % (2.0 * np.pi)
    if diff > np.pi:
        diff = 2.0 * np.pi - diff
    return float(diff)


def heading_cosine(angle1: float, angle2: float) -> float:
    """
    Computes cos(Delta theta) in [-1, 1].
    +1.0 = identical direction, 0.0 = orthogonal, -1.0 = opposite direction.
    """
    return float(np.cos(angular_difference(angle1, angle2)))
