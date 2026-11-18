"""
pathsnap.experiments.optuna_optimization
========================================
Rigorous & High-Efficiency Bayesian Hyperparameter Optimisation using Optuna (TPE).
1. Executes 30 Optuna trials using 3-fold GroupKFold on a geographically representative
   subset of 20 training trips (~300,000 candidate rows) for sub-3-second trial latency.
2. Validates the winning hyperparameter configuration on the FULL 80 training trips (1.2M rows)
   using 5-Fold GroupKFold cross-validation.
3. Refits on the entire 80-trip training partition and exports final model artifacts to
   pathsnap/artifacts/final_model/ with complete feature schema and metadata.
"""

import os
import sys
import time
import json
import joblib
import optuna
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss

optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pathsnap.feature_engineering.feature_extractor import FEATURE_NAMES
from pathsnap.models.hist_gradient_boosting import HistGradientBoostingPipeline
from pathsnap.artifacts.export_final_model import FEATURE_SCHEMA, export_and_verify


def print_flush(*args, **kwargs):
    kwargs["flush"] = True
    print(*args, **kwargs)


def run_optuna_optimization(
    n_trials: int = 30,
    data_path: str = "pathsnap/data/train_candidates.joblib",
    test_data_path: str = "pathsnap/data/student_test_candidates.joblib",
    reports_dir: str = "pathsnap/reports",
    artifacts_dir: str = "pathsnap/artifacts/final_model",
):
    print_flush("=" * 80)
    print_flush("PATHSNAP BAYESIAN HYPERPARAMETER OPTIMISATION (OPTUNA TPE)")
    print_flush("=" * 80)

    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    print_flush(f"\n[1/4] Loading full training dataset from '{data_path}'...")
    df_train = joblib.load(data_path)
    X_full = df_train[FEATURE_NAMES].values.astype(float)
    y_full = df_train["target"].values.astype(int)
    groups_full = df_train["trip_id"].values
    all_trips = sorted(df_train["trip_id"].unique())
    print_flush(f"Loaded {len(df_train):,} candidate rows across {len(all_trips)} trips.")

    # 1. Stratified Geographic Subset for Fast Tuning (Every 4th trip -> 20 representative trips)
    search_trips = set(all_trips[::4])  # Exactly 20 trips across diverse network domains
    mask_search = df_train["trip_id"].isin(search_trips).values
    df_search = df_train[mask_search].reset_index(drop=True)
    X_search = df_search[FEATURE_NAMES].values.astype(float)
    y_search = df_search["target"].values.astype(int)
    groups_search = df_search["trip_id"].values

    print_flush(f"\n[2/4] Constructed Representative Tuning Partition: {len(df_search):,} rows across {len(search_trips)} trips.")

    # Pre-slice search folds
    gkf_search = GroupKFold(n_splits=3)
    search_folds = []
    for tr_idx, val_idx in gkf_search.split(X_search, y_search, groups=groups_search):
        df_val_sub = df_search.iloc[val_idx][["trip_id", "obs_idx", "target"]].copy()
        search_folds.append({
            "X_tr": X_search[tr_idx],
            "y_tr": y_search[tr_idx],
            "X_val": X_search[val_idx],
            "y_val": y_search[val_idx],
            "df_val": df_val_sub,
        })

    trials_records: List[Dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        t0 = time.time()

        learning_rate = trial.suggest_float("learning_rate", 0.04, 0.16, log=True)
        max_iter = trial.suggest_int("max_iter", 80, 160, step=20)
        max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 24, 45)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 15, 40, step=5)
        l2_regularization = trial.suggest_float("l2_regularization", 0.1, 5.0, log=True)
        class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])

        fold_f1_scores = []
        fold_top1_accs = []
        fold_log_losses = []

        for fold_idx, fold in enumerate(search_folds, 1):
            pipeline = HistGradientBoostingPipeline(
                learning_rate=learning_rate,
                max_iter=max_iter,
                max_leaf_nodes=max_leaf_nodes,
                min_samples_leaf=min_samples_leaf,
                l2_regularization=l2_regularization,
                class_weight=class_weight,
                early_stopping=True,
                random_state=42 + fold_idx,
            )
            pipeline.fit(fold["X_tr"], fold["y_tr"])
            y_val_prob = pipeline.predict_proba(fold["X_val"])
            y_val_pred = (y_val_prob >= 0.5).astype(int)

            macro_f1 = float(f1_score(fold["y_val"], y_val_pred, average="macro", zero_division=0))
            loss = float(log_loss(fold["y_val"], np.clip(y_val_prob, 1e-15, 1.0 - 1e-15)))

            df_val_temp = fold["df_val"].copy()
            df_val_temp["score"] = y_val_prob
            idx_top1 = df_val_temp.groupby(["trip_id", "obs_idx"])["score"].idxmax()
            top1_acc = float(np.mean(df_val_temp.loc[idx_top1, "target"] == 1))

            fold_f1_scores.append(macro_f1)
            fold_top1_accs.append(top1_acc)
            fold_log_losses.append(loss)

        mean_macro_f1 = float(np.mean(fold_f1_scores))
        std_macro_f1 = float(np.std(fold_f1_scores))
        mean_top1_acc = float(np.mean(fold_top1_accs))
        mean_log_loss = float(np.mean(fold_log_losses))
        trial_time = time.time() - t0

        record = {
            "trial_number": trial.number,
            "learning_rate": learning_rate,
            "max_iter": max_iter,
            "max_leaf_nodes": max_leaf_nodes,
            "min_samples_leaf": min_samples_leaf,
            "l2_regularization": l2_regularization,
            "class_weight": str(class_weight),
            "search_macro_f1": mean_macro_f1,
            "search_macro_f1_std": std_macro_f1,
            "search_top1_acc": mean_top1_acc,
            "search_log_loss": mean_log_loss,
            "trial_time_s": trial_time,
        }
        trials_records.append(record)

        print_flush(
            f"Trial {trial.number:2d}/{n_trials} | "
            f"F1: {mean_macro_f1:.4f} ± {std_macro_f1:.4f} | "
            f"Top-1: {mean_top1_acc*100:.2f}% | "
            f"LogLoss: {mean_log_loss:.4f} | "
            f"LR: {learning_rate:.4f}, Leaves: {max_leaf_nodes}, L2: {l2_regularization:.2f} "
            f"({trial_time:.2f}s)"
        )

        return mean_macro_f1

    print_flush(f"\nLaunching Optuna Study ({n_trials} Trials, TPESampler, seed=42)...")
    t_opt_start = time.time()
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    opt_time = time.time() - t_opt_start
    print_flush(f"\nSuccessfully completed {n_trials} Bayesian optimization trials in {opt_time:.2f}s ({opt_time/60:.2f} minutes).")

    df_trials = pd.DataFrame(trials_records)
    trials_csv_path = os.path.join(reports_dir, "optuna_trials.csv")
    df_trials.to_csv(trials_csv_path, index=False)
    print_flush(f"Saved all {len(df_trials)} trial records to '{trials_csv_path}'.")

    best_trial = study.best_trial
    print_flush("\n" + "=" * 80)
    print_flush(f"OPTIMAL HYPERPARAMETERS FOUND (Trial {best_trial.number}):")
    print_flush("=" * 80)
    print_flush(f"  Search Tuning Macro-F1: {best_trial.value:.4f}")
    for k, v in best_trial.params.items():
        print_flush(f"  {k}: {v}")

    # 3. Final Verification: Run FULL 5-Fold GroupKFold across all 80 Training Trips
    print_flush("\n" + "=" * 80)
    print_flush("[3/4] FINAL VALIDATION: 5-Fold GroupKFold on FULL 80 Training Trips (1,208,139 rows)")
    print_flush("=" * 80)

    gkf_5 = GroupKFold(n_splits=5)
    f5_scores = []
    f5_top1s = []
    f5_recalls = []
    f5_precisions = []
    f5_losses = []

    best_p = best_trial.params

    for f_idx, (tr_idx, val_idx) in enumerate(gkf_5.split(X_full, y_full, groups=groups_full), 1):
        df_val_sub = df_train.iloc[val_idx][["trip_id", "obs_idx", "target"]].copy()
        final_pipe = HistGradientBoostingPipeline(
            learning_rate=best_p["learning_rate"],
            max_iter=best_p["max_iter"],
            max_leaf_nodes=best_p["max_leaf_nodes"],
            min_samples_leaf=best_p["min_samples_leaf"],
            l2_regularization=best_p["l2_regularization"],
            class_weight=best_p["class_weight"],
            early_stopping=True,
            random_state=42 + f_idx,
        )
        t_f = time.time()
        final_pipe.fit(X_full[tr_idx], y_full[tr_idx])
        fit_t = time.time() - t_f

        y_prob = final_pipe.predict_proba(X_full[val_idx])
        y_pred = (y_prob >= 0.5).astype(int)

        f1 = float(f1_score(y_full[val_idx], y_pred, average="macro", zero_division=0))
        rec = float(np.sum((y_full[val_idx] == 1) & (y_pred == 1)) / np.sum(y_full[val_idx] == 1))
        prec = float(np.sum((y_full[val_idx] == 1) & (y_pred == 1)) / max(np.sum(y_pred == 1), 1))
        loss = float(log_loss(y_full[val_idx], np.clip(y_prob, 1e-15, 1.0 - 1e-15)))

        df_val_sub["score"] = y_prob
        idx_top1 = df_val_sub.groupby(["trip_id", "obs_idx"])["score"].idxmax()
        top1 = float(np.mean(df_val_sub.loc[idx_top1, "target"] == 1))

        f5_scores.append(f1)
        f5_top1s.append(top1)
        f5_recalls.append(rec)
        f5_precisions.append(prec)
        f5_losses.append(loss)
        print_flush(f"  Fold {f_idx}/5: Macro-F1 = {f1:.4f} | Top-1 = {top1*100:.2f}% | Recall = {rec*100:.2f}% | LogLoss = {loss:.4f} ({fit_t:.2f}s)")

    final_cv_macro_f1 = float(np.mean(f5_scores))
    final_cv_top1_acc = float(np.mean(f5_top1s))
    final_cv_recall = float(np.mean(f5_recalls))
    final_cv_precision = float(np.mean(f5_precisions))
    final_cv_log_loss = float(np.mean(f5_losses))

    print_flush(f"\nFinal Optimized 5-Fold GroupKFold Cross-Validation Metrics:")
    print_flush(f"  Macro-F1:    {final_cv_macro_f1:.4f} ± {np.std(f5_scores):.4f}")
    print_flush(f"  Top-1 Acc:   {final_cv_top1_acc*100:.2f}% ± {np.std(f5_top1s)*100:.2f}%")
    print_flush(f"  Recall:      {final_cv_recall*100:.2f}% ± {np.std(f5_recalls)*100:.2f}%")
    print_flush(f"  Precision:   {final_cv_precision*100:.2f}% ± {np.std(f5_precisions)*100:.2f}%")
    print_flush(f"  Log Loss:    {final_cv_log_loss:.4f} ± {np.std(f5_losses):.4f}")

    best_config = {
        "model_family": "HistGradientBoostingClassifier",
        "best_trial_number": best_trial.number,
        "search_macro_f1": float(best_trial.value),
        "final_5fold_cv_macro_f1": final_cv_macro_f1,
        "final_5fold_cv_macro_f1_std": float(np.std(f5_scores)),
        "final_5fold_cv_top1_acc": final_cv_top1_acc,
        "final_5fold_cv_top1_acc_std": float(np.std(f5_top1s)),
        "final_5fold_cv_recall": final_cv_recall,
        "final_5fold_cv_precision": final_cv_precision,
        "final_5fold_cv_log_loss": final_cv_log_loss,
        "parameters": best_trial.params,
        "optimization_sampler": "TPESampler",
        "n_trials": n_trials,
        "feature_names": FEATURE_NAMES,
        "num_features": len(FEATURE_NAMES),
    }
    config_path = os.path.join(artifacts_dir, "best_hyperparameters.json")
    with open(config_path, "w") as f:
        json.dump(best_config, f, indent=2)
    print_flush(f"\nSaved optimal configuration to '{config_path}'.")

    # 4. Final Model Artifact Export & Independent Verification
    print_flush("\n" + "=" * 80)
    print_flush("[4/4] EXPORTING FINAL PRODUCTION ARTIFACT & VERIFYING HMM COMPATIBILITY")
    print_flush("=" * 80)

    export_and_verify(
        best_params=best_p,
        train_data_path=data_path,
        test_data_path=test_data_path,
        artifacts_dir=artifacts_dir,
    )

    return best_config


if __name__ == "__main__":
    run_optuna_optimization(n_trials=30)
