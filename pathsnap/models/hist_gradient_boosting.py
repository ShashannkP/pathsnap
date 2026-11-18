"""
pathsnap.models.hist_gradient_boosting
======================================
Reusable HistGradientBoostingClassifier pipeline with HMM-compatible inference interfaces.

Provides:
- Robust training and validation with early stopping.
- Candidate probability estimation for HMM emission likelihoods.
- HMM emission payload formatting grouped by observation index.
- Model persistence via joblib.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from ..feature_engineering.feature_extractor import FEATURE_NAMES


class HistGradientBoostingPipeline:
    """
    Scikit-learn pipeline wrapper for HistGradientBoosting candidate classification.
    """

    def __init__(
        self,
        learning_rate: float = 0.05,
        max_iter: int = 200,
        max_leaf_nodes: int = 31,
        min_samples_leaf: int = 20,
        l2_regularization: float = 1.0,
        class_weight: Optional[str] = None,
        early_stopping: bool = True,
        random_state: int = 42,
    ):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.class_weight = class_weight
        self.early_stopping = early_stopping
        self.random_state = random_state

        self.model = HistGradientBoostingClassifier(
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            class_weight=self.class_weight,
            early_stopping=self.early_stopping,
            random_state=self.random_state,
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HistGradientBoostingPipeline":
        """
        Fits the HistGradientBoostingClassifier on candidate feature matrix X and binary target y.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts binary candidate labels (0 or 1)."""
        X = np.asarray(X, dtype=float)
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts candidate correctness probability P(y=1 | x).
        Returns 1D array of probabilities in [0.0, 1.0].
        """
        X = np.asarray(X, dtype=float)
        probs = self.model.predict_proba(X)
        # Return probability of positive class (col 1)
        if probs.shape[1] == 2:
            return probs[:, 1]
        elif probs.shape[1] == 1:
            # Single class edge case
            return probs[:, 0]
        return probs[:, -1]

    def format_for_hmm(
        self,
        df_candidates: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        """
        Formats candidate predictions into observation-grouped HMM emission payloads.

        Parameters
        ----------
        df_candidates : pd.DataFrame
            Must contain 'obs_idx', 'timestamp', 'cand_arc_id', and all 18 feature columns.

        Returns
        -------
        list of dict, one per observation:
            {
                'obs_idx': int,
                'timestamp': float,
                'candidate_arc_ids': list of int,
                'probabilities': list of float,
            }
        """
        X = df_candidates[FEATURE_NAMES].values.astype(float)
        probs = self.predict_proba(X)

        df_scored = df_candidates[["obs_idx", "timestamp", "cand_arc_id"]].copy()
        df_scored["prob"] = probs

        hmm_payload: List[Dict[str, Any]] = []
        for obs_idx, group in df_scored.groupby("obs_idx", sort=True):
            ts = float(group["timestamp"].iloc[0])
            arc_ids = group["cand_arc_id"].values.astype(int).tolist()
            cand_probs = group["prob"].values.astype(float).tolist()

            hmm_payload.append({
                "obs_idx": int(obs_idx),
                "timestamp": ts,
                "candidate_arc_ids": arc_ids,
                "probabilities": cand_probs,
            })

        return hmm_payload

    def save(self, filepath: str) -> None:
        """Serializes pipeline to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "HistGradientBoostingPipeline":
        """Loads pipeline from disk."""
        return joblib.load(filepath)
