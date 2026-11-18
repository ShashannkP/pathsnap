"""
pathsnap.evaluation
===================
Evaluation metrics and validation routines.
"""

from .metrics import (
    compute_candidate_classification_metrics,
    compute_observation_top1_accuracy,
    compute_route_continuity,
)

__all__ = [
    "compute_candidate_classification_metrics",
    "compute_observation_top1_accuracy",
    "compute_route_continuity",
]
