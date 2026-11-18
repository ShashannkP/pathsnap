"""
pathsnap.candidate_generation.spatial_index
===========================================
High-performance Sampled Arc KD-Tree for spatial candidate road-segment retrieval.

Interpolates spatial sample points along directed road segments at uniform 25m intervals,
eliminating the midpoint miss rate on long arterial roads and highway spans.
Provides an adaptive fallback to K-nearest points during extreme GPS multipath spikes.
"""

import numpy as np
from scipy.spatial import cKDTree
from typing import List, Tuple, Dict, NamedTuple
from ..preprocessing.network_preprocessor import RoadNetwork
from ..geospatial.geometry import point_to_segment_projection


class CandidateMatch(NamedTuple):
    """
    Holds spatial candidate attributes for a single candidate road segment.
    """
    arc_id: int
    dist_m: float          # Perpendicular distance in meters
    tau: float             # Projection factor [0.0, 1.0] along segment
    proj_point: np.ndarray # Projected [x, y] coordinates on segment


class SampledArcKDTree:
    """
    Discretizes road network segments into uniform metric sample points and
    indexes them in a scipy cKDTree for sub-millisecond candidate retrieval.
    """

    def __init__(
        self,
        sampling_interval_m: float = 25.0,
        search_radius_m: float = 35.0,
        fallback_k: int = 15,
    ):
        self.sampling_interval_m = sampling_interval_m
        self.search_radius_m = search_radius_m
        self.fallback_k = fallback_k

        self.tree: cKDTree = None
        self.sample_points: np.ndarray = None
        self.sample_arc_ids: np.ndarray = None
        self.network: RoadNetwork = None
        self.u_coords: np.ndarray = None
        self.v_coords: np.ndarray = None

    def build(self, network: RoadNetwork) -> "SampledArcKDTree":
        """
        Builds the sampled spatial KD-Tree for the given road network.
        """
        self.network = network
        arcs = network.arcs
        nodes_metric = network.nodes_metric
        arc_lengths = network.arc_lengths
        num_arcs = network.num_arcs

        self.u_coords = nodes_metric[arcs[:, 0]]
        self.v_coords = nodes_metric[arcs[:, 1]]

        pts_list: List[np.ndarray] = []
        arc_ids_list: List[int] = []

        interval = float(self.sampling_interval_m)

        for arc_id in range(num_arcs):
            a = self.u_coords[arc_id]
            b = self.v_coords[arc_id]
            length = arc_lengths[arc_id]

            # Always add start and end nodes
            pts_list.append(a)
            arc_ids_list.append(arc_id)
            pts_list.append(b)
            arc_ids_list.append(arc_id)

            # If segment is longer than sampling interval, interpolate intermediate points
            if length > interval:
                n_samples = int(length // interval)
                ab = b - a
                step = 1.0 / (n_samples + 1)
                for s in range(1, n_samples + 1):
                    pts_list.append(a + (s * step) * ab)
                    arc_ids_list.append(arc_id)

        self.sample_points = np.array(pts_list, dtype=float)
        self.sample_arc_ids = np.array(arc_ids_list, dtype=int)
        self.tree = cKDTree(self.sample_points)

        return self

    def query_point(
        self,
        p_metric: np.ndarray,
        radius_m: float = None,
        fallback_k: int = None,
    ) -> List[CandidateMatch]:
        """
        Retrieves, deduplicates, and scores candidates for a single metric point p.
        Returns candidates sorted in ascending order of exact perpendicular distance d_perp.
        """
        if radius_m is None:
            radius_m = self.search_radius_m
        if fallback_k is None:
            fallback_k = self.fallback_k

        # Step 1: Ball query
        sample_indices = self.tree.query_ball_point(p_metric, r=radius_m)

        # Step 2: Adaptive fallback if ball query is empty (severe GPS outlier / gap)
        if len(sample_indices) == 0:
            k = min(fallback_k, len(self.sample_points))
            _, sample_indices = self.tree.query(p_metric, k=k)
            if np.isscalar(sample_indices):
                sample_indices = [int(sample_indices)]
            else:
                sample_indices = list(sample_indices)

        # Step 3: Deduplicate candidate arc IDs
        candidate_arc_ids = set(self.sample_arc_ids[sample_indices])

        # Step 4: Compute exact perpendicular distance and projection factor for each candidate
        candidates: List[CandidateMatch] = []
        for arc_id in candidate_arc_ids:
            a = self.u_coords[arc_id]
            b = self.v_coords[arc_id]
            dist_m, tau, proj_pt = point_to_segment_projection(p_metric, a, b)
            candidates.append(CandidateMatch(
                arc_id=int(arc_id),
                dist_m=float(dist_m),
                tau=float(tau),
                proj_point=proj_pt,
            ))

        # Step 5: Sort candidates by exact perpendicular distance
        candidates.sort(key=lambda c: c.dist_m)
        return candidates

    def query_trajectory(
        self,
        coords_metric: np.ndarray,
        radius_m: float = None,
        fallback_k: int = None,
    ) -> List[List[CandidateMatch]]:
        """
        Retrieves candidates for all observations in a trajectory.
        """
        results: List[List[CandidateMatch]] = []
        for i in range(len(coords_metric)):
            cands = self.query_point(coords_metric[i], radius_m=radius_m, fallback_k=fallback_k)
            results.append(cands)
        return results
