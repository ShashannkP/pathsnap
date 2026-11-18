"""
pathsnap.experiments.cross_validation
=====================================
Rigorous 5-fold GroupKFold cross-validation and Student-Test evaluation
across the FIVE distinct model families:
1. Logistic Regression (Linear / Probabilistic)
2. Random Forest (Bagging Ensemble)
3. Extra Trees (Extremely Randomized Trees)
4. HistGradientBoosting (Histogram Gradient Boosting - Leading Hypothesis)
5. Multi-Layer Perceptron (Connectionist / Deep Learning)

All folds are strictly grouped by trip_id (64 train / 16 val trips per fold).
Outputs:
- pathsnap/reports/cross_validation_folds.csv (fold-by-fold metrics)
- pathsnap/reports/cross_validation_results.csv (mean +- std per model)
- pathsnap/reports/student_testing_results.csv (evaluation on 10 student test trips)
- pathsnap/reports/full_model_comparison.csv (comprehensive benchmark comparison)
"""

import os
import sys
import time
import psutil
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    log_loss,
    brier_score_loss,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pathsnap.feature_engineering.feature_extractor import FEATURE_NAMES
from pathsnap.models.logistic_regression import LogisticRegressionPipeline
from pathsnap.models.random_forest import RandomForestPipeline
from pathsnap.models.extra_trees import ExtraTreesPipeline
from pathsnap.models.hist_gradient_boosting import HistGradientBoostingPipeline
from pathsnap.models.mlp import MLPPipeline
from pathsnap.evaluation.calibration import compute_calibration_metrics


def print_flush(*args, **kwargs):
    kwargs["flush"] = True
    print(*args, **kwargs)


def evaluate_predictions(
    df_eval: pd.DataFrame,
    y_prob: np.ndarray,
    model_name: str,
) -> Dict[str, float]:
    """
    Computes candidate-level and observation-level Top-1 metrics.
    """
    y_true = df_eval["target"].values.astype(int)
    y_pred = (y_prob >= 0.5).astype(int)

    # 1. Candidate-level classification metrics
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    pos_f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    precision = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    recall = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))

    clipped_prob = np.clip(y_prob, 1e-15, 1.0 - 1e-15)
    loss = float(log_loss(y_true, clipped_prob))
    brier = float(brier_score_loss(y_true, clipped_prob))

    # Calibration ECE
    calib = compute_calibration_metrics(y_true, clipped_prob, n_bins=10)
    ece = calib["ece"]

    # 2. Observation-level Top-1 Accuracy:
    # For each observation, does the candidate with highest predicted probability have target == 1?
    df_temp = df_eval[["trip_id", "obs_idx", "target"]].copy()
    df_temp["score"] = y_prob

    # Top-1 candidate per (trip_id, obs_idx)
    idx_top1 = df_temp.groupby(["trip_id", "obs_idx"])["score"].idxmax()
    top1_targets = df_temp.loc[idx_top1, "target"].values
    top1_accuracy = float(np.mean(top1_targets == 1))

    return {
        "macro_f1": macro_f1,
        "pos_f1": pos_f1,
        "precision": precision,
        "recall": recall,
        "accuracy": acc,
        "top1_accuracy": top1_accuracy,
        "log_loss": loss,
        "brier_score": brier,
        "ece": ece,
    }


def get_model_instances() -> Dict[str, Any]:
    """Instantiates fresh pipelines for the 5 model families."""
    return {
        "LogisticRegression": LogisticRegressionPipeline(C=1.0, max_iter=1000, class_weight="balanced"),
        "RandomForest": RandomForestPipeline(n_estimators=100, max_depth=16, class_weight="balanced_subsample", n_jobs=-1, random_state=42),
        "ExtraTrees": ExtraTreesPipeline(n_estimators=100, max_depth=16, class_weight="balanced_subsample", n_jobs=-1, random_state=42),
        "HistGradientBoosting": HistGradientBoostingPipeline(learning_rate=0.08, max_iter=150, max_leaf_nodes=31, class_weight="balanced", random_state=42),
        "MultiLayerPerceptron": MLPPipeline(hidden_layer_sizes=(64, 32), alpha=0.001, max_iter=150, random_state=42),
    }


def run_cross_validation():
    print_flush("=" * 80)
    print_flush("PATHSNAP 5-MODEL BENCHMARK: 5-FOLD GROUPKFOLD & STUDENT-TEST EVALUATION")
    print_flush("=" * 80)

    train_data_path = "pathsnap/data/train_candidates.joblib"
    test_data_path = "pathsnap/data/student_test_candidates.joblib"
    reports_dir = "pathsnap/reports"
    os.makedirs(reports_dir, exist_ok=True)

    if not os.path.exists(train_data_path) or not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Datasets not found in pathsnap/data. Run build_full_datasets.py first.")

    print_flush(f"\nLoading training dataset from '{train_data_path}'...")
    df_train = joblib.load(train_data_path)
    print_flush(f"Loaded {len(df_train):,} training candidate rows across {df_train['trip_id'].nunique()} unique trips.")

    print_flush(f"\nLoading student test dataset from '{test_data_path}'...")
    df_test_raw = joblib.load(test_data_path)
    df_test = df_test_raw[df_test_raw["target"] >= 0].reset_index(drop=True)
    print_flush(f"Loaded {len(df_test):,} clean student test candidate rows across {df_test['trip_id'].nunique()} unique trips.")

    X_train_full = df_train[FEATURE_NAMES].values.astype(float)
    y_train_full = df_train["target"].values.astype(int)
    groups = df_train["trip_id"].values

    X_test = df_test[FEATURE_NAMES].values.astype(float)
    y_test = df_test["target"].values.astype(int)

    # Setup 5-Fold GroupKFold
    n_splits = 5
    gkf = GroupKFold(n_splits=n_splits)

    fold_records: List[Dict[str, Any]] = []

    model_names = [
        "LogisticRegression",
        "RandomForest",
        "ExtraTrees",
        "HistGradientBoosting",
        "MultiLayerPerceptron",
    ]

    print_flush("\n" + "-" * 80)
    print_flush(f"PHASE A: 5-FOLD GROUPKFOLD CROSS-VALIDATION (Strictly Grouped by trip_id)")
    print_flush("-" * 80)

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train_full, y_train_full, groups=groups), 1):
        n_train_trips = len(np.unique(groups[train_idx]))
        n_val_trips = len(np.unique(groups[val_idx]))
        print_flush(f"\n>>> FOLD {fold_idx}/{n_splits} | Train: {len(train_idx):,} rows ({n_train_trips} trips) | Val: {len(val_idx):,} rows ({n_val_trips} trips)")

        X_tr, y_tr = X_train_full[train_idx], y_train_full[train_idx]
        X_val, y_val = X_train_full[val_idx], y_train_full[val_idx]
        df_val = df_train.iloc[val_idx].reset_index(drop=True)

        for m_name in model_names:
            models_dict = get_model_instances()
            model = models_dict[m_name]

            # Measure fit time and memory
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / (1024 * 1024)

            t0 = time.time()
            model.fit(X_tr, y_tr)
            fit_time = time.time() - t0

            # Measure inference time
            t1 = time.time()
            y_val_prob = model.predict_proba(X_val)
            infer_time = time.time() - t1
            mem_after = process.memory_info().rss / (1024 * 1024)
            peak_mem_mb = max(mem_after - mem_before, 0.0)

            metrics = evaluate_predictions(df_val, y_val_prob, m_name)
            metrics.update({
                "model": m_name,
                "fold": fold_idx,
                "train_trips": n_train_trips,
                "val_trips": n_val_trips,
                "train_rows": len(train_idx),
                "val_rows": len(val_idx),
                "fit_time_s": fit_time,
                "infer_time_s": infer_time,
                "throughput_rows_per_s": len(val_idx) / infer_time if infer_time > 0 else 0.0,
                "peak_memory_mb": peak_mem_mb,
            })
            fold_records.append(metrics)

            print_flush(f"  [{m_name:<20}] Top-1 Acc: {metrics['top1_accuracy']*100:.2f}% | Macro-F1: {metrics['macro_f1']:.4f} | Recall: {metrics['recall']:.4f} | LogLoss: {metrics['log_loss']:.4f} | Fit: {fit_time:.2f}s")

    # Aggregate Cross-Validation Results
    df_folds = pd.DataFrame(fold_records)
    folds_csv_path = os.path.join(reports_dir, "cross_validation_folds.csv")
    df_folds.to_csv(folds_csv_path, index=False)
    print_flush(f"\nSaved fold-by-fold records to '{folds_csv_path}'.")

    summary_metrics = ["top1_accuracy", "macro_f1", "precision", "recall", "accuracy", "log_loss", "brier_score", "ece", "fit_time_s", "infer_time_s"]
    cv_summary = []

    for m_name in model_names:
        sub = df_folds[df_folds["model"] == m_name]
        row = {"model": m_name}
        for metric in summary_metrics:
            row[f"{metric}_mean"] = sub[metric].mean()
            row[f"{metric}_std"] = sub[metric].std()
        cv_summary.append(row)

    df_cv_summary = pd.DataFrame(cv_summary)
    cv_summary_path = os.path.join(reports_dir, "cross_validation_results.csv")
    df_cv_summary.to_csv(cv_summary_path, index=False)
    print_flush(f"Saved GroupKFold summary to '{cv_summary_path}'.")

    # PHASE B: Fit on FULL 80 training trips & Evaluate on 10 Student Test Trips
    print_flush("\n" + "-" * 80)
    print_flush(f"PHASE B: FULL DATASET TRAINING (80 TRIPS) & STUDENT TESTING (10 TRIPS)")
    print_flush("-" * 80)

    test_records = []
    full_fitted_models = {}

    for m_name in model_names:
        print_flush(f"\n>>> Refitting {m_name} on ALL {len(df_train):,} training candidate rows...")
        models_dict = get_model_instances()
        model = models_dict[m_name]

        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)

        t0 = time.time()
        model.fit(X_train_full, y_train_full)
        full_fit_time = time.time() - t0

        t1 = time.time()
        y_test_prob = model.predict_proba(X_test)
        test_infer_time = time.time() - t1
        mem_after = process.memory_info().rss / (1024 * 1024)

        test_metrics = evaluate_predictions(df_test, y_test_prob, m_name)
        test_metrics.update({
            "model": m_name,
            "train_rows": len(df_train),
            "test_rows": len(df_test),
            "fit_time_s": full_fit_time,
            "infer_time_s": test_infer_time,
            "throughput_rows_per_s": len(df_test) / test_infer_time if test_infer_time > 0 else 0.0,
            "peak_memory_mb": max(mem_after - mem_before, 0.0),
        })
        test_records.append(test_metrics)
        full_fitted_models[m_name] = model

        print_flush(f"  [{m_name:<20}] Test Top-1 Acc: {test_metrics['top1_accuracy']*100:.2f}% | Test Macro-F1: {test_metrics['macro_f1']:.4f} | Precision: {test_metrics['precision']:.4f} | Recall: {test_metrics['recall']:.4f} | LogLoss: {test_metrics['log_loss']:.4f} | Fit: {full_fit_time:.2f}s")

    df_test_results = pd.DataFrame(test_records)
    test_results_path = os.path.join(reports_dir, "student_testing_results.csv")
    df_test_results.to_csv(test_results_path, index=False)
    print_flush(f"\nSaved Student Testing results to '{test_results_path}'.")

    # PHASE C: Construct Master Comparison Table
    print_flush("\n" + "=" * 80)
    print_flush("PHASE C: COMPREHENSIVE 5-MODEL BENCHMARK COMPARISON TABLE")
    print_flush("=" * 80)

    comparison_rows = []
    for m_name in model_names:
        cv_row = df_cv_summary[df_cv_summary["model"] == m_name].iloc[0]
        test_row = df_test_results[df_test_results["model"] == m_name].iloc[0]

        comparison_rows.append({
            "Model": m_name,
            "CV Top-1 Acc (%)": f"{cv_row['top1_accuracy_mean']*100:.2f} ± {cv_row['top1_accuracy_std']*100:.2f}",
            "CV Macro-F1": f"{cv_row['macro_f1_mean']:.4f} ± {cv_row['macro_f1_std']:.4f}",
            "CV Recall": f"{cv_row['recall_mean']:.4f} ± {cv_row['recall_std']:.4f}",
            "CV Log Loss": f"{cv_row['log_loss_mean']:.4f} ± {cv_row['log_loss_std']:.4f}",
            "CV Brier Score": f"{cv_row['brier_score_mean']:.4f} ± {cv_row['brier_score_std']:.4f}",
            "CV Fit Time (s)": f"{cv_row['fit_time_s_mean']:.2f} ± {cv_row['fit_time_s_std']:.2f}",
            "Test Top-1 Acc (%)": f"{test_row['top1_accuracy']*100:.2f}",
            "Test Macro-F1": f"{test_row['macro_f1']:.4f}",
            "Test Recall": f"{test_row['recall']:.4f}",
            "Test Log Loss": f"{test_row['log_loss']:.4f}",
            "Test Brier Score": f"{test_row['brier_score']:.4f}",
            "Full Fit Time (s)": f"{test_row['fit_time_s']:.2f}",
            "Test Throughput (rows/s)": f"{test_row['throughput_rows_per_s']:,.0f}",
        })

    df_master = pd.DataFrame(comparison_rows)
    master_csv_path = os.path.join(reports_dir, "full_model_comparison.csv")
    df_master.to_csv(master_csv_path, index=False)
    print_flush(df_master.to_string(index=False))
    print_flush(f"\nSaved master comparison table to '{master_csv_path}'.")

    return df_cv_summary, df_test_results, df_master

    return df_cv_summary, df_test_results, df_master


if __name__ == "__main__":
    run_cross_validation()
