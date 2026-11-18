"""
pathsnap.ground_truth.route_aligner
===================================
Robust ground-truth route alignment and candidate label assignment.

Aligns raw GPS observations with chronological .route arc sequences using a monotonic
forward-sliding window in projected metric space, assigning binary supervisory labels
(y = 1 for the true traversed segment, y = 0 for false alternatives) while explicitly
flagging ambiguous or unalignable observations to prevent noisy label contamination.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from ..preprocessing.gps_preprocessor import GPSTrajectory
from ..preprocessing.network_preprocessor import RoadNetwork
from ..candidate_generation.spatial_index import CandidateMatch
from ..geospatial.geometry import point_to_segment_projection


class ObservationAlignment:
    """
    Holds ground-truth alignment results for a single GPS observation.
    """
    def __init__(
        self,
        obs_idx: int,
        true_arc_id: Optional[int],
        dist_to_true_arc_m: float,
        route_sequence_idx: int,
        is_reliable: bool,
        is_candidate_miss: bool,
    ):
        self.obs_idx = obs_idx
        self.true_arc_id = true_arc_id
        self.dist_to_true_arc_m = dist_to_true_arc_m
        self.route_sequence_idx = route_sequence_idx
        self.is_reliable = is_reliable
        self.is_candidate_miss = is_candidate_miss


class RouteAligner:
    """
    Aligns GPS trajectory points to the reference .route arc sequence.
    """

    def __init__(
        self,
        max_valid_distance_m: float = 50.0,
        forward_window_size: int = 15,
        backward_window_size: int = 2,
    ):
        self.max_valid_distance_m = max_valid_distance_m
        self.forward_window_size = forward_window_size
        self.backward_window_size = backward_window_size

    def align_trajectory_to_route(
        self,
        trajectory: GPSTrajectory,
        network: RoadNetwork,
        route_path: str,
        candidates_per_obs: List[List[CandidateMatch]] = None,
    ) -> List[ObservationAlignment]:
        """
        Aligns every GPS point in trajectory to the ground-truth route arcs.

        Parameters
        ----------
        trajectory : GPSTrajectory
            Processed GPS trajectory.
        network : RoadNetwork
            Processed road network.
        route_path : str
            Path to .route file.
        candidates_per_obs : list of list of CandidateMatch, optional
            Pre-generated candidates to check candidate miss rate.

        Returns
        -------
        list of ObservationAlignment
        """
        # Load route arc IDs
        raw_route = np.loadtxt(route_path, dtype=int)
        if raw_route.ndim == 0:
            raw_route = np.array([int(raw_route)])
        route_arcs = list(raw_route)
        num_route_arcs = len(route_arcs)

        coords_metric = trajectory.coords_metric
        num_obs = len(coords_metric)

        # Pre-extract metric endpoints of all route arcs
        u_coords = network.u_coords
        v_coords = network.v_coords

        route_arc_coords = [
            (u_coords[arc_id], v_coords[arc_id]) for arc_id in route_arcs
        ]

        alignments: List[ObservationAlignment] = []
        curr_route_idx = 0

        for t in range(num_obs):
            p = coords_metric[t]

            # Search within sliding window around current route index
            w_start = max(0, curr_route_idx - self.backward_window_size)
            w_end = min(num_route_arcs, curr_route_idx + self.forward_window_size + 1)

            best_d = 1e9
            best_route_idx = curr_route_idx

            for r_idx in range(w_start, w_end):
                a, b = route_arc_coords[r_idx]
                d, _, _ = point_to_segment_projection(p, a, b)
                if d < best_d:
                    best_d = d
                    best_route_idx = r_idx

            is_reliable = best_d <= self.max_valid_distance_m
            true_arc_id = route_arcs[best_route_idx] if is_reliable else None

            # Advance current route index monotonically
            if is_reliable:
                curr_route_idx = best_route_idx

            # Check candidate miss
            is_miss = False
            if candidates_per_obs is not None and is_reliable:
                cands = candidates_per_obs[t]
                cand_arc_ids = {c.arc_id for c in cands}
                is_miss = (true_arc_id not in cand_arc_ids)

            alignments.append(ObservationAlignment(
                obs_idx=t,
                true_arc_id=true_arc_id,
                dist_to_true_arc_m=float(best_d),
                route_sequence_idx=best_route_idx,
                is_reliable=is_reliable,
                is_candidate_miss=is_miss,
            ))

        return alignments
