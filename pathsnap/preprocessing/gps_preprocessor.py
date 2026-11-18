"""
pathsnap.preprocessing.gps_preprocessor
=======================================
Robust GPS trajectory loader, validator, and kinematic feature generator.

Strictly preserves all observations and chronological ordering while computing
kinematic diagnostics (time deltas, speeds, accelerations, headings, gap flags).
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from ..geospatial.projection import LocalMetricProjector, compute_bearing


class GPSTrajectory:
    """
    Holds processed GPS trajectory data for a single trip.
    """

    def __init__(
        self,
        trip_id: str,
        df: pd.DataFrame,
        coords_metric: np.ndarray,
        projector: LocalMetricProjector,
        metadata: dict,
    ):
        self.trip_id = trip_id
        self.df = df
        self.coords_metric = coords_metric  # (T, 2) array of [x_m, y_m]
        self.projector = projector
        self.metadata = metadata

    def __len__(self) -> int:
        return len(self.df)


class GPSPreprocessor:
    """
    Loads, cleans, validates, and computes kinematics for raw GPS .track files.
    """

    def __init__(
        self,
        gap_threshold_sec: float = 5.0,
        large_gap_threshold_sec: float = 60.0,
        speed_jump_threshold_mps: float = 40.0,  # 144 km/h
        stationary_threshold_m: float = 0.5,
    ):
        self.gap_threshold_sec = gap_threshold_sec
        self.large_gap_threshold_sec = large_gap_threshold_sec
        self.speed_jump_threshold_mps = speed_jump_threshold_mps
        self.stationary_threshold_m = stationary_threshold_m

    def load_and_process(self, track_path: str, trip_id: Optional[str] = None) -> GPSTrajectory:
        """
        Loads and validates a raw .track file.

        Parameters
        ----------
        track_path : str
            Path to .track file.
        trip_id : str, optional
            Trip identifier. Inferred from filename if not provided.

        Returns
        -------
        GPSTrajectory
        """
        if trip_id is None:
            trip_id = os.path.splitext(os.path.basename(track_path))[0]

        # Load raw data (whitespace-separated: longitude, latitude, timestamp)
        raw_data = np.loadtxt(track_path, dtype=float)
        if raw_data.ndim == 1 and len(raw_data) == 3:
            raw_data = raw_data.reshape(1, 3)

        num_obs = len(raw_data)
        if num_obs == 0:
            raise ValueError(f"Track file {track_path} contains zero records.")

        lon = raw_data[:, 0]
        lat = raw_data[:, 1]
        timestamps = raw_data[:, 2]

        # Data integrity checks
        nan_count = int(np.isnan(raw_data).sum())
        if nan_count > 0:
            raise ValueError(f"Trip {trip_id} track has {nan_count} NaN values.")

        # Coordinate boundary validation
        invalid_coords = ((lon < -180.0) | (lon > 180.0) | (lat < -90.0) | (lat > 90.0)).sum()
        if invalid_coords > 0:
            raise ValueError(f"Trip {trip_id} has {invalid_coords} out-of-bounds coordinates.")

        # Timestamp monotonicity validation
        dt_raw = np.diff(timestamps)
        is_strictly_monotonic = bool(np.all(dt_raw > 0))
        duplicate_ts_count = int((dt_raw == 0).sum())

        # Establish local metric projection centered at trajectory median
        lon0 = float(np.median(lon))
        lat0 = float(np.median(lat))
        projector = LocalMetricProjector(lon0=lon0, lat0=lat0)
        coords_metric = projector.project(lon, lat)

        # Compute kinematic vectors
        # Displacements
        dx = np.zeros(num_obs, dtype=float)
        dy = np.zeros(num_obs, dtype=float)
        disp = np.zeros(num_obs, dtype=float)
        dt = np.ones(num_obs, dtype=float)  # First point default dt=1.0

        dx[1:] = coords_metric[1:, 0] - coords_metric[:-1, 0]
        dy[1:] = coords_metric[1:, 1] - coords_metric[:-1, 1]
        disp[1:] = np.sqrt(dx[1:] ** 2 + dy[1:] ** 2)
        dt[1:] = np.maximum(dt_raw, 1e-4)

        # Speeds and accelerations
        speed = disp / dt
        accel = np.zeros(num_obs, dtype=float)
        accel[1:] = (speed[1:] - speed[:-1]) / dt[1:]

        # Heading calculation
        bearing = np.zeros(num_obs, dtype=float)
        # For moving points, calculate atan2(dx, dy)
        moving_mask = disp >= self.stationary_threshold_m
        for i in range(1, num_obs):
            if moving_mask[i]:
                bearing[i] = compute_bearing(dx[i], dy[i])
            else:
                bearing[i] = bearing[i - 1]  # Carry forward last known moving heading
        bearing[0] = bearing[1] if num_obs > 1 else 0.0

        # Anomaly and operational flags
        is_gap = dt > self.gap_threshold_sec
        is_large_gap = dt > self.large_gap_threshold_sec
        is_jump = speed > self.speed_jump_threshold_mps
        is_stationary = disp < self.stationary_threshold_m

        df = pd.DataFrame({
            "obs_idx": np.arange(num_obs, dtype=int),
            "lon": lon,
            "lat": lat,
            "timestamp": timestamps,
            "x_m": coords_metric[:, 0],
            "y_m": coords_metric[:, 1],
            "dx_m": dx,
            "dy_m": dy,
            "displacement_m": disp,
            "dt_sec": dt,
            "speed_mps": speed,
            "accel_mps2": accel,
            "bearing_rad": bearing,
            "is_gap": is_gap,
            "is_large_gap": is_large_gap,
            "is_jump": is_jump,
            "is_stationary": is_stationary,
        })

        metadata = {
            "trip_id": trip_id,
            "num_observations": num_obs,
            "lon0": lon0,
            "lat0": lat0,
            "duration_sec": float(timestamps[-1] - timestamps[0]) if num_obs > 1 else 0.0,
            "total_distance_m": float(disp.sum()),
            "mean_speed_mps": float(speed.mean()),
            "max_speed_mps": float(speed.max()),
            "gap_count": int(is_gap.sum()),
            "large_gap_count": int(is_large_gap.sum()),
            "jump_count": int(is_jump.sum()),
            "is_strictly_monotonic": is_strictly_monotonic,
            "duplicate_timestamps": duplicate_ts_count,
        }

        return GPSTrajectory(
            trip_id=trip_id,
            df=df,
            coords_metric=coords_metric,
            projector=projector,
            metadata=metadata,
        )
