"""
Multi-trip empirical benchmark script.
Evaluates candidate generation, miss rates, class imbalance, and timing across 10 training trips.
"""

import os
import time
import pandas as pd
import numpy as np
from pathsnap.dataset.dataset_builder import DatasetBuilder


def main():
    builder = DatasetBuilder(
        sampling_interval_m=25.0,
        search_radius_m=35.0,
        fallback_k=15,
        max_align_dist_m=50.0,
    )

    train_dir = "Student folder/train"
    trips_to_eval = sorted([
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    ])[:10]

    results = []

    for tid in trips_to_eval:
        tdir = os.path.join(train_dir, tid)
        t0 = time.perf_counter()
        df_cands, stats = builder.process_single_trip(tdir, trip_id=tid)
        dur = time.perf_counter() - t0
        stats["elapsed_sec"] = dur

        cands_per_obs = df_cands.groupby("obs_idx").size()
        stats["max_candidates"] = int(cands_per_obs.max())
        stats["min_candidates"] = int(cands_per_obs.min())
        results.append(stats)

        print(
            f"Trip {tid}: {stats['num_observations']:5d} GPS | "
            f"{stats['total_candidates']:6d} cands | "
            f"Miss Rate: {stats['candidate_miss_rate']*100:5.2f}% | "
            f"Avg Cands: {stats['avg_candidates_per_obs']:4.1f} | "
            f"Max Cands: {stats['max_candidates']:2d} | "
            f"Pos: {stats['positive_samples']:5d} | "
            f"Neg: {stats['negative_samples']:5d} | "
            f"Time: {dur:5.2f}s"
        )

    df_summary = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("MULTI-TRIP CANDIDATE GENERATION BENCHMARK (10 TRIPS)")
    print("=" * 80)
    tot_obs = df_summary["num_observations"].sum()
    tot_cands = df_summary["total_candidates"].sum()
    tot_rel = df_summary["reliable_observations"].sum()
    tot_miss = df_summary["candidate_misses"].sum()
    tot_pos = df_summary["positive_samples"].sum()
    tot_neg = df_summary["negative_samples"].sum()

    print(f"Total GPS Observations:           {tot_obs:,}")
    print(f"Total Candidate Rows Generated:    {tot_cands:,}")
    print(f"Reliable Aligned Observations:     {tot_rel:,} ({tot_rel/tot_obs*100:.2f}%)")
    print(f"Total Candidate Misses:            {tot_miss}")
    print(f"Dataset-Level Candidate Miss Rate: {tot_miss / tot_rel * 100:.3f}%")
    print(f"Dataset Theoretical Accuracy Cap:  {100.0 - (tot_miss / tot_rel * 100):.3f}%")
    print(f"Mean Candidates per Observation:   {df_summary['avg_candidates_per_obs'].mean():.2f}")
    print(f"Max Candidates (Single Obs):       {df_summary['max_candidates'].max()}")
    print(f"Min Candidates (Single Obs):       {df_summary['min_candidates'].min()}")
    print(f"Total Positive Candidates (y=1):   {tot_pos:,}")
    print(f"Total Negative Candidates (y=0):   {tot_neg:,}")
    print(f"Class Imbalance Ratio (Neg:Pos):   {tot_neg / tot_pos:.2f} : 1")
    print(f"Mean Processing Time per Trip:     {df_summary['elapsed_sec'].mean():.2f}s")
    print("=" * 80)

    # Save summary table
    df_summary.to_csv("pathsnap/reports/candidate_benchmark_10trips.csv", index=False)
    print("Saved summary table to pathsnap/reports/candidate_benchmark_10trips.csv")


if __name__ == "__main__":
    main()
