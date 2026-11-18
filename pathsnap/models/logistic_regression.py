"""
pathsnap.models.logistic_regression
===================================
Logistic Regression baseline classifier for candidate map matching.
Features standard scaling and L2 regularization with balanced class weights.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from ..feature_engineering.feature_extractor import FEATURE_NAMES


class LogisticRegressionPipeline:
    """
    Standardized Logistic Regression candidate classifier.
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: Optional[str] = "balanced",
        solver: str = "lbfgs",
        random_state: int = 42,
    ):
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.solver = solver
        self.random_state = random_state

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=self.C,
                max_iter=self.max_iter,
                class_weight=self.class_weight,
                solver=self.solver,
                random_state=self.random_state,
                n_jobs=-1,
            ))
        ])
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionPipeline":
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
    def load(cls, filepath: str) -> "LogisticRegressionPipeline":
        return joblib.load(filepath)
