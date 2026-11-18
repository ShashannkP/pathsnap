"""
pathsnap.models.extra_trees
===========================
Extra Trees (Extremely Randomized Trees) candidate classifier.
Ensemble method combining extreme randomization with variance reduction.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.ensemble import ExtraTreesClassifier
from ..feature_engineering.feature_extractor import FEATURE_NAMES


class ExtraTreesPipeline:
    """
    Extra Trees candidate classifier.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 16,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        class_weight: Optional[str] = "balanced_subsample",
        max_features: str = "sqrt",
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.model = ExtraTreesClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=self.class_weight,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ExtraTreesPipeline":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        probs = self.model.predict_proba(X)
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
    def load(cls, filepath: str) -> "ExtraTreesPipeline":
        return joblib.load(filepath)
