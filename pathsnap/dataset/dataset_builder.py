"""
pathsnap.dataset.dataset_builder
================================
Builds, validates, and partitions model-ready candidate-level datasets.

Constructs tabular datasets where each row corresponds to a single candidate road segment
for a given GPS observation, preserving metadata (trip_id, obs_idx, timestamp, cand_arc_id),
18 high-signal feature columns, and binary ground-truth labels (target: 1 = true segment, 0 = alternative).
Enforces strict trip-level separation for GroupKFold cross-validation.
"""

import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Any
from ..preprocessing.gps_preprocessor import GPSPreprocessor, GPSTrajectory
from ..preprocessing.network_preprocessor import NetworkPreprocessor, RoadNetwork
from ..candidate_generation.spatial_index import SampledArcKDTree, CandidateMatch
from ..ground_truth.route_aligner import RouteAligner, ObservationAlignment
from ..feature_engineering.feature_extractor import FeatureExtractor, FEATURE_NAMES


METADATA_COLUMNS = ["trip_id", "obs_idx", "timestamp", "cand_arc_id"]
TARGET_COLUMN = "target"


class DatasetBuilder:
    """
    Constructs candidate-level model-ready datasets from trip folders.
    """

    def __init__(
        self,
        sampling_interval_m: float = 25.0,
        search_radius_m: float = 35.0,
        fallback_k: int = 15,
        max_align_dist_m: float = 50.0,
    ):
        self.gps_preprocessor = GPSPreprocessor()
        self.network_preprocessor = NetworkPreprocessor()
        self.spatial_index = SampledArcKDTree(
            sampling_interval_m=sampling_interval_m,
            search_radius_m=search_radius_m,
            fallback_k=fallback_k,
        )
        self.route_aligner = RouteAligner(max_valid_distance_m=max_align_dist_m)
        self.feature_extractor = FeatureExtractor()

    def process_single_trip(
        self,
        trip_dir: str,
        trip_id: Optional[str] = None,
        has_ground_truth: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Processes a single trip directory and produces candidate-level dataset.

        Parameters
        ----------
        trip_dir : str
            Path to folder containing .track, .nodes, .arcs, and optional .route.
        trip_id : str, optional
            Trip identifier. Inferred from directory name if None.
        has_ground_truth : bool
            Whether to attempt label alignment using .route.

        Returns
        -------
        df : pd.DataFrame
            DataFrame with metadata, 18 features, and target (if has_ground_truth).
        stats : dict
            Diagnostic metrics for this trip.
        """
        if trip_id is None:
            trip_id = os.path.basename(os.path.normpath(trip_dir))

        track_path = os.path.join(trip_dir, f"{trip_id}.track")
        nodes_path = os.path.join(trip_dir, f"{trip_id}.nodes")
        arcs_path = os.path.join(trip_dir, f"{trip_id}.arcs")
        route_path = os.path.join(trip_dir, f"{trip_id}.route")

        # 1. Preprocess GPS Trajectory
        trajectory = self.gps_preprocessor.load_and_process(track_path, trip_id=trip_id)

        # 2. Preprocess Road Network using the trajectory's metric projector
        network = self.network_preprocessor.load_and_process(
            nodes_path=nodes_path,
            arcs_path=arcs_path,
            projector=trajectory.projector,
            trip_id=trip_id,
        )

        # 3. Build Spatial Index & Query Candidates
        index = SampledArcKDTree(
            sampling_interval_m=self.spatial_index.sampling_interval_m,
            search_radius_m=self.spatial_index.search_radius_m,
            fallback_k=self.spatial_index.fallback_k,
        ).build(network)

        candidates_per_obs = index.query_trajectory(trajectory.coords_metric)

        # 4. Align Ground-Truth Route if available
        alignments: Optional[List[ObservationAlignment]] = None
        has_route_file = os.path.exists(route_path) and has_ground_truth
        if has_route_file:
            alignments = self.route_aligner.align_trajectory_to_route(
                trajectory=trajectory,
                network=network,
                route_path=route_path,
                candidates_per_obs=candidates_per_obs,
            )

        # 5. Extract Features
        X_features = self.feature_extractor.extract_features_for_trajectory(
            trajectory=trajectory,
            network=network,
            candidates_per_obs=candidates_per_obs,
        )

        # 6. Assemble DataFrame rows
        rows = []
        row_idx = 0
        total_misses = 0
        total_reliable_obs = 0

        df_track = trajectory.df
        num_obs = len(df_track)

        for t in range(num_obs):
            cands = candidates_per_obs[t]
            ts = df_track.at[t, "timestamp"]

            true_arc_id = None
            is_reliable = False
            if alignments is not None:
                true_arc_id = alignments[t].true_arc_id
                is_reliable = alignments[t].is_reliable
                if is_reliable:
                    total_reliable_obs += 1
                    if alignments[t].is_candidate_miss:
                        total_misses += 1

            for cand in cands:
                cand_arc_id = cand.arc_id

                # Target assignment:
                # 1 if candidate is true road segment; 0 otherwise.
                # If ground truth was unalignable, label is set to -1 (excluded).
                if has_route_file:
                    if is_reliable and true_arc_id is not None:
                        target = 1 if cand_arc_id == true_arc_id else 0
                    else:
                        target = -1  # Ambiguous / unaligned
                else:
                    target = -1

                meta = {
                    "trip_id": trip_id,
                    "obs_idx": t,
                    "timestamp": ts,
                    "cand_arc_id": cand_arc_id,
                    TARGET_COLUMN: target,
                }
                rows.append(meta)
                row_idx += 1

        df_meta = pd.DataFrame(rows)
        df_feats = pd.DataFrame(X_features, columns=FEATURE_NAMES)
        df_result = pd.concat([df_meta, df_feats], axis=1)

        miss_rate = (total_misses / total_reliable_obs) if total_reliable_obs > 0 else 0.0

        stats = {
            "trip_id": trip_id,
            "num_observations": num_obs,
            "total_candidates": len(df_result),
            "avg_candidates_per_obs": len(df_result) / num_obs if num_obs > 0 else 0.0,
            "reliable_observations": total_reliable_obs,
            "candidate_misses": total_misses,
            "candidate_miss_rate": miss_rate,
            "positive_samples": int((df_result[TARGET_COLUMN] == 1).sum()),
            "negative_samples": int((df_result[TARGET_COLUMN] == 0).sum()),
            "unaligned_samples": int((df_result[TARGET_COLUMN] == -1).sum()),
        }

        return df_result, stats

    def build_dataset_from_directory(
        self,
        base_dir: str,
        trip_ids: Optional[List[str]] = None,
        max_trips: Optional[int] = None,
        filter_unaligned: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Builds a combined dataset across multiple trip directories.

        Parameters
        ----------
        base_dir : str
            Parent directory containing trip folders (e.g., 'Student folder/train').
        trip_ids : list of str, optional
            Specific trip IDs to load.
        max_trips : int, optional
            Limit to first N trips for rapid benchmarking.
        filter_unaligned : bool
            Whether to drop rows with target == -1 (ambiguous alignment).

        Returns
        -------
        df_combined : pd.DataFrame
            Combined candidate-level dataset.
        df_stats : pd.DataFrame
            Per-trip summary statistics.
        """
        if trip_ids is None:
            trip_ids = sorted([
                d for d in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, d))
            ])

        if max_trips is not None:
            trip_ids = trip_ids[:max_trips]

        trip_dfs = []
        stats_list = []

        for tid in trip_ids:
            tdir = os.path.join(base_dir, tid)
            df_trip, trip_stats = self.process_single_trip(tdir, trip_id=tid)
            trip_dfs.append(df_trip)
            stats_list.append(trip_stats)

        df_combined = pd.concat(trip_dfs, axis=0, ignore_index=True)
        df_stats = pd.DataFrame(stats_list)

        if filter_unaligned:
            df_combined = df_combined[df_combined[TARGET_COLUMN] >= 0].reset_index(drop=True)

        return df_combined, df_stats


def get_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Extracts (N, 18) float feature matrix."""
    return df[FEATURE_NAMES].values.astype(float)


def get_target_vector(df: pd.DataFrame) -> np.ndarray:
    """Extracts (N,) integer target vector (0 or 1)."""
    return df[TARGET_COLUMN].values.astype(int)


def get_groups(df: pd.DataFrame) -> np.ndarray:
    """Extracts (N,) array of trip_id values for GroupKFold."""
    return df["trip_id"].values
