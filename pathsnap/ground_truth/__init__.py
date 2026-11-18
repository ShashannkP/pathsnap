"""
pathsnap.ground_truth
=====================
Ground-truth route alignment and candidate supervisory label assignment.
"""

from .route_aligner import RouteAligner, ObservationAlignment

__all__ = ["RouteAligner", "ObservationAlignment"]
