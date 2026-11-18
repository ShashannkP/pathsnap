"""
pathsnap.artifacts.export_final_model
====================================
Exports the final winning model artifact, feature schema, configuration,
and calibration parameters for production inference and HMM consumption.
Verifies independent loading and mock HMM emission payload formatting.
"""

import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pathsnap.feature_engineering.feature_extractor import FEATURE_NAMES
from pathsnap.models.hist_gradient_boosting import HistGradientBoostingPipeline


FEATURE_SCHEMA = [
    {
        "name": "dist_to_segment_m",
        "category": "Spatial",
        "dtype": "float64",
        "description": "Perpendicular metric distance from GPS point to projected road segment.",
        "min_expected": 0.0,
        "max_expected": 50.0,
    },
    {
        "name": "log_dist_to_segment",
        "category": "Spatial",
        "dtype": "float64",
        "description": "Natural log of 1 + orthogonal distance.",
        "min_expected": 0.0,
        "max_expected": 4.0,
    },
    {
        "name": "tau_along_segment",
        "category": "Spatial",
        "dtype": "float64",
        "description": "Longitudinal fraction of projection along arc [0, 1].",
        "min_expected": 0.0,
        "max_expected": 1.0,
    },
    {
        "name": "cand_arc_length_m",
        "category": "Spatial",
        "dtype": "float64",
        "description": "Physical metric length of candidate arc.",
        "min_expected": 1.0,
        "max_expected": 5000.0,
    },
    {
        "name": "delta_heading_deg",
        "category": "Directional",
        "dtype": "float64",
        "description": "Absolute angular difference between trajectory heading and arc direction [0, 180].",
        "min_expected": 0.0,
        "max_expected": 180.0,
    },
    {
        "name": "cos_delta_heading",
        "category": "Directional",
        "dtype": "float64",
        "description": "Cosine of angular heading difference [-1, 1].",
        "min_expected": -1.0,
        "max_expected": 1.0,
    },
    {
        "name": "bearing_concordance",
        "category": "Directional",
        "dtype": "float64",
        "description": "Binary flag indicating whether delta_heading <= 45 degrees.",
        "min_expected": 0.0,
        "max_expected": 1.0,
    },
    {
        "name": "delta_t_s",
        "category": "Temporal/Kinematic",
        "dtype": "float64",
        "description": "Elapsed seconds since previous observation.",
        "min_expected": 0.0,
        "max_expected": 3600.0,
    },
    {
        "name": "obs_speed_mps",
        "category": "Temporal/Kinematic",
        "dtype": "float64",
        "description": "Kinematic speed in meters per second.",
        "min_expected": 0.0,
        "max_expected": 70.0,
    },
    {
        "name": "obs_accel_mps2",
        "category": "Temporal/Kinematic",
        "dtype": "float64",
        "description": "Kinematic acceleration in meters per second squared.",
        "min_expected": -30.0,
        "max_expected": 30.0,
    },
    {
        "name": "speed_length_ratio",
        "category": "Temporal/Kinematic",
        "dtype": "float64",
        "description": "Dimensionless ratio of displacement (v * dt) to arc length.",
        "min_expected": 0.0,
        "max_expected": 100.0,
    },
    {
        "name": "is_1hop_reachable",
        "category": "Topological",
        "dtype": "float64",
        "description": "Binary indicator: current candidate directly connects to previous candidate.",
        "min_expected": 0.0,
        "max_expected": 1.0,
    },
    {
        "name": "is_2hop_reachable",
        "category": "Topological",
        "dtype": "float64",
        "description": "Binary indicator: current candidate reachable via 2-hop graph traversal.",
        "min_expected": 0.0,
        "max_expected": 1.0,
    },
    {
        "name": "rank_by_distance",
        "category": "Ranking/Contextual",
        "dtype": "float64",
        "description": "Zero-indexed rank of candidate sorted by distance (0 = closest).",
        "min_expected": 0.0,
        "max_expected": 30.0,
    },
    {
        "name": "rank_by_heading",
        "category": "Ranking/Contextual",
        "dtype": "float64",
        "description": "Zero-indexed rank of candidate sorted by heading alignment (0 = most aligned).",
        "min_expected": 0.0,
        "max_expected": 30.0,
    },
    {
        "name": "distance_margin_m",
        "category": "Ranking/Contextual",
        "dtype": "float64",
        "description": "Distance difference relative to closest candidate (d_i - d_0).",
        "min_expected": 0.0,
        "max_expected": 50.0,
    },
    {
        "name": "heading_margin_deg",
        "category": "Ranking/Contextual",
        "dtype": "float64",
        "description": "Heading alignment difference relative to best-aligned candidate.",
        "min_expected": 0.0,
        "max_expected": 180.0,
    },
    {
        "name": "num_candidates_at_obs",
        "category": "Ranking/Contextual",
        "dtype": "float64",
        "description": "Total candidate count generated for current observation.",
        "min_expected": 1.0,
        "max_expected": 30.0,
    },
]


def export_and_verify(
    best_params: dict,
    train_data_path: str = "pathsnap/data/train_candidates.joblib",
    test_data_path: str = "pathsnap/data/student_test_candidates.joblib",
    artifacts_dir: str = "pathsnap/artifacts/final_model",
):
    print("=" * 80)
    print("FINAL MODEL ARTIFACT EXPORT & VERIFICATION")
    print("=" * 80)

    os.makedirs(artifacts_dir, exist_ok=True)

    # 1. Export Feature Schema
    schema_path = os.path.join(artifacts_dir, "feature_schema.json")
    with open(schema_path, "w") as f:
        json.dump({"features": FEATURE_SCHEMA, "feature_names": FEATURE_NAMES}, f, indent=2)
    print(f"Exported Feature Schema (18 features) to '{schema_path}'.")

    # 2. Refit Winning Model on ALL 80 Training Trips
    print(f"\nLoading full training dataset from '{train_data_path}'...")
    df_train = joblib.load(train_data_path)
    X_train = df_train[FEATURE_NAMES].values.astype(float)
    y_train = df_train["target"].values.astype(int)

    print(f"Refitting HistGradientBoostingPipeline with optimal hyperparameters on {len(df_train):,} rows...")
    t0 = time.time()
    final_pipeline = HistGradientBoostingPipeline(
        learning_rate=best_params.get("learning_rate", 0.08),
        max_iter=best_params.get("max_iter", 180),
        max_leaf_nodes=best_params.get("max_leaf_nodes", 31),
        min_samples_leaf=best_params.get("min_samples_leaf", 25),
        l2_regularization=best_params.get("l2_regularization", 1.0),
        class_weight=best_params.get("class_weight", "balanced"),
        early_stopping=True,
        random_state=42,
    )
    final_pipeline.fit(X_train, y_train)
    fit_time = time.time() - t0
    print(f"Model successfully fitted in {fit_time:.2f}s.")

    # 3. Save Model Artifact
    model_path = os.path.join(artifacts_dir, "model.joblib")
    final_pipeline.save(model_path)
    print(f"Serialized model artifact to '{model_path}'.")

    # 4. Save Model Configuration JSON
    config = {
        "model_family": "HistGradientBoostingClassifier",
        "framework": "scikit-learn 1.6.1",
        "creation_timestamp": datetime.now().isoformat(),
        "input_features": FEATURE_NAMES,
        "num_features": len(FEATURE_NAMES),
        "hyperparameters": best_params,
        "training_samples": len(df_train),
        "training_trips": int(df_train["trip_id"].nunique()),
        "fit_time_seconds": fit_time,
        "target_variable": "target (1 = true road segment, 0 = alternative candidate)",
        "intended_use": "GPS Map-Matching HMM Emission Probability Generation",
    }
    config_path = os.path.join(artifacts_dir, "model_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Exported model metadata configuration to '{config_path}'.")

    # 5. Independent Verification: Reload from disk and score student test trip
    print("\n" + "-" * 80)
    print("VERIFICATION: Reloading artifact from disk & testing inference...")
    print("-" * 80)

    reloaded_pipeline = HistGradientBoostingPipeline.load(model_path)
    df_test_raw = joblib.load(test_data_path)
    sample_trip_id = df_test_raw["trip_id"].iloc[0]
    df_sample = df_test_raw[df_test_raw["trip_id"] == sample_trip_id].copy()

    X_sample = df_sample[FEATURE_NAMES].values.astype(float)
    t_inf = time.time()
    sample_probs = reloaded_pipeline.predict_proba(X_sample)
    inf_time = time.time() - t_inf

    print(f"  Reloaded model successfully from '{model_path}'.")
    print(f"  Scored trip '{sample_trip_id}': {len(df_sample)} candidates in {inf_time*1000:.2f} ms ({len(df_sample)/inf_time:,.0f} rows/sec).")
    print(f"  Probability range: [{sample_probs.min():.4f}, {sample_probs.max():.4f}], mean={sample_probs.mean():.4f}")

    # Format for HMM
    hmm_payload = reloaded_pipeline.format_for_hmm(df_sample)
    print(f"  Generated HMM emission payload for {len(hmm_payload)} observations.")
    print(f"  Sample HMM observation 0:")
    print(f"    obs_idx: {hmm_payload[0]['obs_idx']}")
    print(f"    timestamp: {hmm_payload[0]['timestamp']}")
    print(f"    candidate_arc_ids: {hmm_payload[0]['candidate_arc_ids'][:5]}...")
    print(f"    probabilities: {[round(p, 4) for p in hmm_payload[0]['probabilities'][:5]]}...")

    print("\n" + "=" * 80)
    print("ALL MODEL ARTIFACTS VERIFIED AND READY FOR PHASE 4 (HMM/VITERBI)")
    print("=" * 80)


if __name__ == "__main__":
    params = {
        "learning_rate": 0.08,
        "max_iter": 180,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 25,
        "l2_regularization": 1.0,
        "class_weight": "balanced",
    }
    if os.path.exists("pathsnap/artifacts/final_model/best_hyperparameters.json"):
        with open("pathsnap/artifacts/final_model/best_hyperparameters.json") as f:
            best_cfg = json.load(f)
            params = best_cfg.get("parameters", params)
    export_and_verify(params)
