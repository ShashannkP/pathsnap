"""
pathsnap.candidate_generation
=============================
Spatial indexing and candidate road segment retrieval.
"""

from .spatial_index import SampledArcKDTree, CandidateMatch

__all__ = ["SampledArcKDTree", "CandidateMatch"]
