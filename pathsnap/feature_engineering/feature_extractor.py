"""
pathsnap.feature_engineering.feature_extractor
==============================================
Extracts the 18-dimensional feature representation for candidate road segments.

Features are organized into five operational categories:
1. Spatial Geometry (5 features)
2. Directional Alignment (4 features)
3. Temporal & Kinematics (3 features)
4. Graph Topology & Connectivity (3 features)
5. Candidate Ranking & Likelihoods (3 features)

Strictly free from target leakage; uses only permitted causal GPS observations and
road-network graph geometry.
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Any, Tuple
from ..preprocessing.gps_preprocessor import GPSTrajectory
from ..preprocessing.network_preprocessor import RoadNetwork
from ..candidate_generation.spatial_index import CandidateMatch
from ..geospatial.projection import angular_difference, heading_cosine


FEATURE_NAMES = [
    # Category A: Spatial Geometry (5)
    "dist_perp",           # Perpendicular distance in meters
    "arc_length",          # Candidate arc length in meters
    "proj_ratio",          # Scalar projection factor tau in [0, 1]
    "gps_displacement",    # Distance from previous GPS point in meters
    "dist_margin",         # Difference from closest candidate distance (meters)
    # Category B: Directional Alignment (4)
    "road_bearing",        # Candidate segment bearing [0, 2pi)
    "gps_bearing",         # Vehicle trajectory heading [0, 2pi)
    "angular_diff",        # Acute angle between road and GPS bearing [0, pi]
    "heading_cos",         # Cosine of angular difference [-1, 1]
    # Category C: Temporal & Kinematics (3)
    "dt_sec",              # Time delta from previous observation in seconds
    "speed_mps",           # Instantaneous speed in meters/second
    "accel_mps2",          # Estimated acceleration in meters/second^2
    # Category D: Graph Topology & Connectivity (3)
    "is_connected_prev",   # 1.0 if successor or same arc as previous best, else 0.0
    "net_to_euc_ratio",    # Network distance / Euclidean displacement ratio
    "target_out_degree",   # Out-degree of candidate target node v
    # Category E: Candidate Ranking & Likelihoods (3)
    "spatial_likelihood",  # Gaussian likelihood exp(-d^2 / 2*sigma^2)
    "cand_rank",           # Ordinal proximity rank (1 = closest)
    "cand_rel_dist",       # Relative distance proportion among observation's candidates
]


class FeatureExtractor:
    """
    Extracts 18-dimensional feature vectors for each candidate of every GPS observation.
    """

    def __init__(self, gps_sigma_m: float = 5.0, max_network_search_hops: int = 3):
        self.gps_sigma_m = gps_sigma_m
        self.max_network_search_hops = max_network_search_hops
        self.feature_names = FEATURE_NAMES

    def extract_features_for_trajectory(
        self,
        trajectory: GPSTrajectory,
        network: RoadNetwork,
        candidates_per_obs: List[List[CandidateMatch]],
    ) -> np.ndarray:
        """
        Extracts feature matrix for all candidates across a trajectory.

        Parameters
        ----------
        trajectory : GPSTrajectory
            Processed trajectory.
        network : RoadNetwork
            Processed road network.
        candidates_per_obs : list of list of CandidateMatch
            Generated candidates for each GPS observation.

        Returns
        -------
        ndarray of shape (Total_Candidates, 18)
        """
        df = trajectory.df
        num_obs = len(df)
        total_cands = sum(len(c) for c in candidates_per_obs)

        feature_matrix = np.zeros((total_cands, len(self.feature_names)), dtype=float)

        arcs = network.arcs
        arc_lengths = network.arc_lengths
        arc_bearings = network.arc_bearings
        graph = network.graph
        out_degrees = dict(graph.out_degree())

        # Pre-extract trajectory columns
        disp_arr = df["displacement_m"].values
        dt_arr = df["dt_sec"].values
        speed_arr = df["speed_mps"].values
        accel_arr = df["accel_mps2"].values
        bearing_arr = df["bearing_rad"].values

        row_ptr = 0
        prev_best_arc_id = -1
        prev_best_tau = 0.5

        for t in range(num_obs):
            cands = candidates_per_obs[t]
            num_cands = len(cands)
            if num_cands == 0:
                continue

            gps_disp = disp_arr[t]
            dt = dt_arr[t]
            speed = speed_arr[t]
            accel = accel_arr[t]
            gps_bearing = bearing_arr[t]

            # Candidate-level distance statistics for this observation
            min_dist = cands[0].dist_m  # Already sorted by distance
            sum_dist = sum(c.dist_m for c in cands) + 1e-4

            for rank_idx, cand in enumerate(cands):
                arc_id = cand.arc_id
                d_perp = cand.dist_m
                tau = cand.tau
                L_arc = float(arc_lengths[arc_id])
                theta_road = float(arc_bearings[arc_id])

                # Spatial geometry (5)
                f0_dist_perp = d_perp
                f1_arc_length = L_arc
                f2_proj_ratio = tau
                f3_gps_disp = gps_disp
                f4_dist_margin = d_perp - min_dist

                # Directional alignment (4)
                ang_diff = angular_difference(gps_bearing, theta_road)
                f5_road_bearing = theta_road
                f6_gps_bearing = gps_bearing
                f7_ang_diff = ang_diff
                f8_heading_cos = float(np.cos(ang_diff))

                # Temporal & Kinematics (3)
                f9_dt = dt
                f10_speed = speed
                f11_accel = accel

                # Graph topology & Connectivity (3)
                u_curr = int(arcs[arc_id, 0])
                v_curr = int(arcs[arc_id, 1])
                target_out_deg = float(out_degrees.get(v_curr, 0))

                if t == 0 or prev_best_arc_id == -1:
                    is_conn = 1.0
                    net_ratio = 1.0
                else:
                    u_prev = int(arcs[prev_best_arc_id, 0])
                    v_prev = int(arcs[prev_best_arc_id, 1])

                    if arc_id == prev_best_arc_id:
                        is_conn = 1.0
                        net_disp = abs(tau - prev_best_tau) * L_arc
                        net_ratio = net_disp / (gps_disp + 0.1)
                    elif v_prev == u_curr:
                        # Direct legal successor (1-hop)
                        is_conn = 1.0
                        L_prev = float(arc_lengths[prev_best_arc_id])
                        net_disp = (1.0 - prev_best_tau) * L_prev + tau * L_arc
                        net_ratio = net_disp / (gps_disp + 0.1)
                    elif v_prev in graph and any(u_curr in graph[nbr] for nbr in graph[v_prev]):
                        # 2-hop legal successor through an intersection
                        is_conn = 0.5
                        L_prev = float(arc_lengths[prev_best_arc_id])
                        net_disp = (1.0 - prev_best_tau) * L_prev + L_arc + 20.0
                        net_ratio = net_disp / (gps_disp + 0.1)
                    else:
                        is_conn = 0.0
                        net_ratio = 5.0  # Disconnected / multi-hop penalty cap

                f12_is_conn = is_conn
                f13_net_ratio = min(net_ratio, 10.0)  # Bound ratio to avoid extreme outliers
                f14_out_deg = target_out_deg

                # Candidate Ranking & Likelihoods (3)
                f15_spatial_lik = float(np.exp(-(d_perp ** 2) / (2.0 * (self.gps_sigma_m ** 2))))
                f16_rank = float(rank_idx + 1)
                f17_rel_dist = float(d_perp / sum_dist)

                feature_matrix[row_ptr] = [
                    f0_dist_perp,
                    f1_arc_length,
                    f2_proj_ratio,
                    f3_gps_disp,
                    f4_dist_margin,
                    f5_road_bearing,
                    f6_gps_bearing,
                    f7_ang_diff,
                    f8_heading_cos,
                    f9_dt,
                    f10_speed,
                    f11_accel,
                    f12_is_conn,
                    f13_net_ratio,
                    f14_out_deg,
                    f15_spatial_lik,
                    f16_rank,
                    f17_rel_dist,
                ]

                row_ptr += 1

            # Update previous best candidate using closest candidate (rank 0)
            prev_best_arc_id = cands[0].arc_id
            prev_best_tau = cands[0].tau

        return feature_matrix
