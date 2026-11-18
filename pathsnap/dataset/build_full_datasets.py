"""
pathsnap.dataset.build_full_datasets
====================================
Builds and persists candidate-level datasets for all 80 training trips and 10 student-testing trips.
Logs diagnostic candidate generation and route alignment statistics.
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pathsnap.dataset.dataset_builder import DatasetBuilder


def main():
    print("=" * 70)
    print("PATHSNAP FULL DATASET CONSTRUCTION PIPELINE")
    print("=" * 70)

    train_base_dir = "Student folder/train"
    test_base_dir = "Student folder/student_test"
    output_dir = "pathsnap/data"
    os.makedirs(output_dir, exist_ok=True)
    reports_dir = "pathsnap/reports"
    os.makedirs(reports_dir, exist_ok=True)

    builder = DatasetBuilder(
        sampling_interval_m=25.0,
        search_radius_m=35.0,
        fallback_k=15,
        max_align_dist_m=50.0,
    )

    # 1. Build Training Dataset (80 Trips)
    train_trips = sorted([
        d for d in os.listdir(train_base_dir)
        if os.path.isdir(os.path.join(train_base_dir, d))
    ])
    print(f"\n[1/2] Processing {len(train_trips)} Training Trips from '{train_base_dir}'...")

    t0 = time.time()
    train_dfs = []
    train_stats_list = []

    for i, tid in enumerate(train_trips, 1):
        tdir = os.path.join(train_base_dir, tid)
        step_t0 = time.time()
        df_trip, trip_stats = builder.process_single_trip(tdir, trip_id=tid, has_ground_truth=True)
        train_dfs.append(df_trip)
        train_stats_list.append(trip_stats)
        elapsed_step = time.time() - step_t0

        if i % 10 == 0 or i == len(train_trips):
            print(f"  Processed {i:2d}/{len(train_trips)} trips ({tid}) - {len(df_trip)} candidates ({elapsed_step:.2f}s)")

    df_train_raw = pd.concat(train_dfs, axis=0, ignore_index=True)
    df_train_stats = pd.DataFrame(train_stats_list)

    total_train_obs = df_train_stats["num_observations"].sum()
    total_train_cands = len(df_train_raw)
    total_train_reliable = df_train_stats["reliable_observations"].sum()
    total_train_misses = df_train_stats["candidate_misses"].sum()
    train_miss_rate = total_train_misses / total_train_reliable if total_train_reliable > 0 else 0.0

    print(f"\nTraining Set Summary:")
    print(f"  Total GPS Observations:     {total_train_obs:,}")
    print(f"  Total Candidates Generated: {total_train_cands:,}")
    print(f"  Avg Candidates per Obs:     {total_train_cands / total_train_obs:.2f}")
    print(f"  Reliably Aligned Obs:       {total_train_reliable:,} ({total_train_reliable/total_train_obs*100:.2f}%)")
    print(f"  Candidate Misses:           {total_train_misses:,} ({train_miss_rate*100:.3f}% miss rate)")
    print(f"  Theoretical Accuracy Cap:   {(1.0 - train_miss_rate)*100:.3f}%")

    # Filter unaligned rows (target >= 0) for training
    df_train_clean = df_train_raw[df_train_raw["target"] >= 0].reset_index(drop=True)
    train_pos = (df_train_clean["target"] == 1).sum()
    train_neg = (df_train_clean["target"] == 0).sum()
    print(f"  Clean Training Rows:        {len(df_train_clean):,}")
    print(f"  Class Ratio (Neg : Pos):    {train_neg / train_pos:.2f} : 1 ({train_pos:,} pos, {train_neg:,} neg)")

    train_out_path = os.path.join(output_dir, "train_candidates.joblib")
    print(f"  Saving to '{train_out_path}'...")
    joblib.dump(df_train_clean, train_out_path, compress=3)

    # 2. Build Student Testing Dataset (10 Trips)
    test_trips = sorted([
        d for d in os.listdir(test_base_dir)
        if os.path.isdir(os.path.join(test_base_dir, d))
    ])
    print(f"\n[2/2] Processing {len(test_trips)} Student Testing Trips from '{test_base_dir}'...")

    test_dfs = []
    test_stats_list = []

    for i, tid in enumerate(test_trips, 1):
        tdir = os.path.join(test_base_dir, tid)
        step_t0 = time.time()
        df_trip, trip_stats = builder.process_single_trip(tdir, trip_id=tid, has_ground_truth=True)
        test_dfs.append(df_trip)
        test_stats_list.append(trip_stats)
        elapsed_step = time.time() - step_t0
        print(f"  Processed {i:2d}/{len(test_trips)} trips ({tid}) - {len(df_trip)} candidates ({elapsed_step:.2f}s)")

    df_test_raw = pd.concat(test_dfs, axis=0, ignore_index=True)
    df_test_stats = pd.DataFrame(test_stats_list)

    total_test_obs = df_test_stats["num_observations"].sum()
    total_test_cands = len(df_test_raw)
    total_test_reliable = df_test_stats["reliable_observations"].sum()
    total_test_misses = df_test_stats["candidate_misses"].sum()
    test_miss_rate = total_test_misses / total_test_reliable if total_test_reliable > 0 else 0.0

    print(f"\nStudent Testing Set Summary:")
    print(f"  Total GPS Observations:     {total_test_obs:,}")
    print(f"  Total Candidates Generated: {total_test_cands:,}")
    print(f"  Avg Candidates per Obs:     {total_test_cands / total_test_obs:.2f}")
    print(f"  Reliably Aligned Obs:       {total_test_reliable:,} ({total_test_reliable/total_test_obs*100:.2f}%)")
    print(f"  Candidate Misses:           {total_test_misses:,} ({test_miss_rate*100:.3f}% miss rate)")
    print(f"  Theoretical Accuracy Cap:   {(1.0 - test_miss_rate)*100:.3f}%")

    test_out_path = os.path.join(output_dir, "student_test_candidates.joblib")
    print(f"  Saving to '{test_out_path}'...")
    joblib.dump(df_test_raw, test_out_path, compress=3)

    # Save stats CSV
    df_all_stats = pd.concat([
        df_train_stats.assign(partition="train"),
        df_test_stats.assign(partition="student_test")
    ], ignore_index=True)
    stats_csv_path = os.path.join(reports_dir, "dataset_construction_stats.csv")
    df_all_stats.to_csv(stats_csv_path, index=False)
    print(f"\nSaved per-trip diagnostic statistics to '{stats_csv_path}'.")

    total_time = time.time() - t0
    print(f"\nAll datasets constructed and verified in {total_time:.2f}s ({total_time/60:.2f} minutes).")


if __name__ == "__main__":
    main()
