"""
pathsnap.geospatial
===================
Geospatial projection and 2D vector geometry tools.
"""

from .projection import (
    LocalMetricProjector,
    haversine_distance,
    compute_bearing,
    angular_difference,
    heading_cosine,
)
from .geometry import (
    point_to_segment_projection,
    segment_length,
)

__all__ = [
    "LocalMetricProjector",
    "haversine_distance",
    "compute_bearing",
    "angular_difference",
    "heading_cosine",
    "point_to_segment_projection",
    "segment_length",
]
