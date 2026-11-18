"""
pathsnap.models.benchmark
=========================
Experimentation and benchmarking framework for the five required model families.

Implements unified trip-aware GroupKFold evaluation across:
1. Logistic Regression (Linear baseline)
2. Random Forest (Bagging ensemble)
3. Extra Trees (Extremely randomized trees)
4. HistGradientBoosting (Gradient boosting leading candidate)
5. Multi-Layer Perceptron (Connectionist / Deep learning)
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from ..feature_engineering.feature_extractor import FEATURE_NAMES
from ..evaluation.metrics import (
    compute_candidate_classification_metrics,
    compute_observation_top1_accuracy,
)
from ..dataset.dataset_builder import TARGET_COLUMN


def get_benchmark_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Returns the five configured models from distinct algorithm families.
    """
    return {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=500, random_state=random_state)),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=16,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=random_state,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=100,
            max_depth=16,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=random_state,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=150,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=random_state,
        ),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                alpha=0.001,
                max_iter=50,
                random_state=random_state,
            )),
        ]),
    }


def evaluate_model_cv(
    model_name: str,
    model: Any,
    df_data: pd.DataFrame,
    n_splits: int = 5,
) -> Dict[str, Any]:
    """
    Evaluates a model using leak-free trip-level GroupKFold cross-validation.

    Parameters
    ----------
    model_name : str
        Name of algorithm.
    model : Scikit-learn estimator
        Estimator instance.
    df_data : pd.DataFrame
        Candidate-level dataset containing metadata, 18 features, and target.
    n_splits : int
        Number of trip-level folds.

    Returns
    -------
    dict with aggregated metrics.
    """
    X = df_data[FEATURE_NAMES].values.astype(float)
    y = df_data[TARGET_COLUMN].values.astype(int)
    groups = df_data["trip_id"].values

    gkf = GroupKFold(n_splits=n_splits)

    fold_metrics = []
    total_train_time = 0.0
    total_infer_time = 0.0

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        df_val = df_data.iloc[val_idx].copy()

        # Training
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0
        total_train_time += train_time

        # Inference
        t1 = time.perf_counter()
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_val)
            y_prob = probs[:, 1] if probs.shape[1] == 2 else probs[:, 0]
        else:
            y_prob = model.decision_function(X_val)
        y_pred = model.predict(X_val)
        infer_time = time.perf_counter() - t1
        total_infer_time += infer_time

        # Compute Metrics
        clf_metrics = compute_candidate_classification_metrics(y_val, y_pred, y_prob=y_prob)
        top1_metrics = compute_observation_top1_accuracy(df_val, y_prob)

        combined = {**clf_metrics, **top1_metrics}
        fold_metrics.append(combined)

    df_fold = pd.DataFrame(fold_metrics)
    summary = {
        "model": model_name,
        "cv_folds": n_splits,
        "mean_macro_f1": float(df_fold["macro_f1"].mean()),
        "std_macro_f1": float(df_fold["macro_f1"].std()),
        "mean_top1_acc": float(df_fold["top1_accuracy"].mean()),
        "std_top1_acc": float(df_fold["top1_accuracy"].std()),
        "mean_precision": float(df_fold["precision"].mean()),
        "mean_recall": float(df_fold["recall"].mean()),
        "mean_log_loss": float(df_fold["log_loss"].mean()) if "log_loss" in df_fold else float("nan"),
        "total_train_time_sec": float(total_train_time),
        "total_infer_time_sec": float(total_infer_time),
    }

    return summary
