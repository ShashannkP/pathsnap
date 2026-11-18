"""
Verification of the 5-Model Benchmarking Framework.
Runs a trip-aware GroupKFold evaluation across:
1. Logistic Regression
2. Random Forest
3. Extra Trees
4. HistGradientBoosting
5. Multi-Layer Perceptron (MLP)
"""

import os
import time
import pandas as pd
import numpy as np
from pathsnap.dataset.dataset_builder import DatasetBuilder
from pathsnap.models.benchmark import get_benchmark_models, evaluate_model_cv


def main():
    builder = DatasetBuilder()
    train_dir = "Student folder/train"
    # Select 4 representative trips for quick verification
    test_trips = ["00000001", "00000003", "00000004", "00000005"]

    print(f"Building candidate dataset across {len(test_trips)} trips: {test_trips}...")
    df_data, df_stats = builder.build_dataset_from_directory(
        base_dir=train_dir,
        trip_ids=test_trips,
        filter_unaligned=True,
    )
    print(f"Dataset shape: {df_data.shape} ({len(df_data):,} candidate rows)")
    print(f"Trips in dataset: {df_data['trip_id'].nunique()}, Observations: {df_data.groupby(['trip_id', 'obs_idx']).ngroups}")
    print(f"Class distribution: Positive = {(df_data['target'] == 1).sum()}, Negative = {(df_data['target'] == 0).sum()}")

    models = get_benchmark_models(random_state=42)
    benchmark_results = []

    print("\n" + "=" * 90)
    print("RUNNING 5-MODEL TRIP-AWARE GROUPKFOLD BENCHMARK VERIFICATION")
    print("=" * 90)

    for name, model in models.items():
        print(f"Evaluating {name} with 3-fold GroupKFold...")
        res = evaluate_model_cv(name, model, df_data, n_splits=3)
        benchmark_results.append(res)
        print(
            f"  -> {name:20s} | "
            f"Macro-F1: {res['mean_macro_f1']:.4f} (+/- {res['std_macro_f1']:.4f}) | "
            f"Top-1 Acc: {res['mean_top1_acc']*100:.2f}% | "
            f"Precision: {res['mean_precision']:.4f} | "
            f"Recall: {res['mean_recall']:.4f} | "
            f"Train Time: {res['total_train_time_sec']:.2f}s | "
            f"Infer Time: {res['total_infer_time_sec']:.3f}s"
        )

    df_bench = pd.DataFrame(benchmark_results)
    df_bench.to_csv("pathsnap/reports/5_model_benchmark_verification.csv", index=False)
    print("=" * 90)
    print("Saved benchmark verification results to pathsnap/reports/5_model_benchmark_verification.csv")


if __name__ == "__main__":
    main()
