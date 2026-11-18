"""
pathsnap.evaluation.calibration
===============================
Probability calibration analysis and evaluation.

Computes:
- Brier Score (mean squared error of predicted probabilities)
- Calibration curves (fraction of positives vs. mean predicted probability)
- Expected Calibration Error (ECE)
- Observation-level softmax normalisation:
  P_norm(c_i) = exp(s_i / T) / sum_j exp(s_j / T) or p_i / sum_j p_j
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Computes Brier score, ECE, and calibration curve coordinates.

    Parameters
    ----------
    y_true : np.ndarray
        Binary true labels (0 or 1).
    y_prob : np.ndarray
        Predicted probabilities in [0.0, 1.0].
    n_bins : int
        Number of equal-width bins for calibration curve.

    Returns
    -------
    dict with:
        brier_score : float
        ece : float (Expected Calibration Error)
        prob_true : list of float
        prob_pred : list of float
        bin_counts : list of int
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)

    # 1. Brier Score
    brier = float(brier_score_loss(y_true, y_prob))

    # 2. Calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")

    # 3. Expected Calibration Error (ECE)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_samples = len(y_prob)
    bin_counts = []

    for i in range(n_bins):
        bin_mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1]) if i < n_bins - 1 else (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        n_in_bin = np.sum(bin_mask)
        bin_counts.append(int(n_in_bin))
        if n_in_bin > 0:
            bin_acc = np.mean(y_true[bin_mask])
            bin_conf = np.mean(y_prob[bin_mask])
            ece += (n_in_bin / total_samples) * abs(bin_acc - bin_conf)

    return {
        "brier_score": brier,
        "ece": float(ece),
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist(),
        "bin_counts": bin_counts,
    }


def normalize_probabilities_by_observation(
    df_candidates: pd.DataFrame,
    prob_col: str = "prob",
    temperature: float = 1.0,
    method: str = "sum",
) -> np.ndarray:
    """
    Normalizes candidate probabilities within each GPS observation so that
    sum_{c in Candidates(t)} P(c) == 1.0.

    Parameters
    ----------
    df_candidates : pd.DataFrame
        DataFrame with 'obs_idx' and probability column.
    prob_col : str
        Name of probability column.
    temperature : float
        Temperature parameter for softmax (if method='softmax').
    method : str
        'sum' for direct ratio normalisation, 'softmax' for logit exponentiation.

    Returns
    -------
    normalized_probs : np.ndarray
        Array of normalized probabilities matching df_candidates length.
    """
    norm_probs = np.zeros(len(df_candidates), dtype=float)

    for _, group_idx in df_candidates.groupby("obs_idx").groups.items():
        sub_probs = df_candidates.loc[group_idx, prob_col].values
        if method == "softmax":
            logits = np.log(np.clip(sub_probs, 1e-12, 1.0 - 1e-12)) / temperature
            exp_logits = np.exp(logits - np.max(logits))
            norm = exp_logits / np.sum(exp_logits)
        else:
            s = np.sum(sub_probs)
            norm = (sub_probs / s) if s > 0 else np.ones_like(sub_probs) / len(sub_probs)

        norm_probs[group_idx] = norm

    return norm_probs
