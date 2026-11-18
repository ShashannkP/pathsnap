"""
pathsnap.evaluation.metrics
===========================
Comprehensive evaluation metrics for candidate classification, top-1 accuracy, and route continuity.

Calculates:
- Observation-level Top-1 Accuracy
- Candidate Miss Rate
- Route Continuity (%)
- Precision, Recall, Macro-F1, Accuracy
- Log Loss (Binary Cross-Entropy)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    confusion_matrix,
)
from typing import Dict, Any, List, Optional
from ..preprocessing.network_preprocessor import RoadNetwork


def compute_candidate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Computes standard classification metrics on candidate rows.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "binary_f1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
    }

    if y_prob is not None:
        y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-7, 1.0 - 1e-7)
        try:
            metrics["log_loss"] = float(log_loss(y_true, y_prob))
        except Exception:
            metrics["log_loss"] = float("nan")

    return metrics


def compute_observation_top1_accuracy(
    df_cands: pd.DataFrame,
    y_scores: np.ndarray,
) -> Dict[str, float]:
    """
    Computes Observation-level Top-1 Accuracy.
    For each GPS observation (grouped by trip_id, obs_idx), picks the candidate with
    highest predicted score and checks if target == 1.

    Parameters
    ----------
    df_cands : pd.DataFrame
        Must contain 'trip_id', 'obs_idx', and 'target'.
    y_scores : np.ndarray
        Model predicted probability or decision score for each candidate row.

    Returns
    -------
    dict with 'top1_accuracy', 'total_observations', 'correct_observations'
    """
    df_eval = df_cands[["trip_id", "obs_idx", "target"]].copy()
    df_eval["score"] = y_scores

    # Find candidate with max score per observation
    idx_top1 = df_eval.groupby(["trip_id", "obs_idx"])["score"].idxmax()
    top1_df = df_eval.loc[idx_top1]

    # Target == 1 indicates the top candidate was indeed the ground-truth segment
    correct = int((top1_df["target"] == 1).sum())
    total = len(top1_df)
    acc = float(correct / total) if total > 0 else 0.0

    return {
        "top1_accuracy": acc,
        "total_observations": total,
        "correct_observations": correct,
    }


def compute_route_continuity(
    predicted_arc_ids: List[int],
    network: RoadNetwork,
) -> Dict[str, Any]:
    """
    Computes Route Continuity: percentage of consecutive predicted road segments
    that form a legal, direct connection in the road network graph (v_t == u_{t+1} or same arc).
    """
    if len(predicted_arc_ids) <= 1:
        return {"continuity_pct": 100.0, "connected_transitions": 0, "total_transitions": 0}

    arcs = network.arcs
    num_transitions = len(predicted_arc_ids) - 1
    connected_count = 0

    for t in range(num_transitions):
        arc1 = predicted_arc_ids[t]
        arc2 = predicted_arc_ids[t + 1]

        if arc1 == arc2:
            # Remained on the same road segment
            connected_count += 1
        else:
            v1 = arcs[arc1, 1]
            u2 = arcs[arc2, 0]
            if v1 == u2:
                # Direct legal successor
                connected_count += 1

    continuity_pct = float(connected_count / num_transitions * 100.0) if num_transitions > 0 else 100.0

    return {
        "continuity_pct": continuity_pct,
        "connected_transitions": connected_count,
        "total_transitions": num_transitions,
    }
