"""
pathsnap.models.mlp
===================
Multi-Layer Perceptron (Neural Network) candidate classifier.
Uses standard feature scaling, Adam optimizer with early stopping and L2 penalty.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from ..feature_engineering.feature_extractor import FEATURE_NAMES


class MLPPipeline:
    """
    Standardized Multi-Layer Perceptron candidate classifier.
    """

    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (64, 32),
        activation: str = "relu",
        alpha: float = 0.001,
        learning_rate_init: float = 0.005,
        batch_size: int = 2048,
        max_iter: int = 150,
        early_stopping: bool = True,
        n_iter_no_change: int = 10,
        random_state: int = 42,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.random_state = random_state

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation=self.activation,
                alpha=self.alpha,
                learning_rate_init=self.learning_rate_init,
                batch_size=self.batch_size,
                max_iter=self.max_iter,
                early_stopping=self.early_stopping,
                n_iter_no_change=self.n_iter_no_change,
                random_state=self.random_state,
            ))
        ])
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPPipeline":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.pipeline.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return self.pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        probs = self.pipeline.predict_proba(X)
        if probs.shape[1] == 2:
            return probs[:, 1]
        elif probs.shape[1] == 1:
            return probs[:, 0]
        return probs[:, -1]

    def format_for_hmm(self, df_candidates: pd.DataFrame) -> List[Dict[str, Any]]:
        X = df_candidates[FEATURE_NAMES].values.astype(float)
        probs = self.predict_proba(X)
        df_scored = df_candidates[["obs_idx", "timestamp", "cand_arc_id"]].copy()
        df_scored["prob"] = probs

        hmm_payload = []
        for obs_idx, group in df_scored.groupby("obs_idx", sort=True):
            hmm_payload.append({
                "obs_idx": int(obs_idx),
                "timestamp": float(group["timestamp"].iloc[0]),
                "candidate_arc_ids": group["cand_arc_id"].values.astype(int).tolist(),
                "probabilities": group["prob"].values.astype(float).tolist(),
            })
        return hmm_payload

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "MLPPipeline":
        return joblib.load(filepath)
