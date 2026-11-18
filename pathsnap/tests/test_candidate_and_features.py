"""
Tests for Candidate Generation, Ground-Truth Alignment, Feature Extraction, and Model Pipeline.
"""

import os
import pytest
import numpy as np
import pandas as pd
from pathsnap.preprocessing.gps_preprocessor import GPSPreprocessor
from pathsnap.preprocessing.network_preprocessor import NetworkPreprocessor
from pathsnap.candidate_generation.spatial_index import SampledArcKDTree
from pathsnap.ground_truth.route_aligner import RouteAligner
from pathsnap.feature_engineering.feature_extractor import FeatureExtractor, FEATURE_NAMES
from pathsnap.dataset.dataset_builder import DatasetBuilder, get_feature_matrix, get_target_vector
from pathsnap.models.hist_gradient_boosting import HistGradientBoostingPipeline


@pytest.fixture
def sample_trip_dir():
    return "Student folder/train/00000001"


def test_candidate_generation_and_features(sample_trip_dir):
    builder = DatasetBuilder(
        sampling_interval_m=25.0,
        search_radius_m=35.0,
        fallback_k=15,
        max_align_dist_m=50.0,
    )

    df_cands, stats = builder.process_single_trip(sample_trip_dir, trip_id="00000001")

    # Verify dataset dimensions and structure
    assert len(df_cands) > 1000
    assert stats["num_observations"] == 1315
    assert stats["candidate_miss_rate"] < 0.005, f"Miss rate {stats['candidate_miss_rate']} exceeds threshold"

    # Verify all 18 features exist and contain zero NaNs
    for feat_name in FEATURE_NAMES:
        assert feat_name in df_cands.columns, f"Missing feature {feat_name}"
        assert not df_cands[feat_name].isna().any(), f"Feature {feat_name} has NaNs"

    # Verify target distribution
    pos_count = (df_cands["target"] == 1).sum()
    neg_count = (df_cands["target"] == 0).sum()
    assert pos_count > 1000, f"Expected > 1000 positive samples, got {pos_count}"
    assert neg_count > pos_count, "Expected negative candidates to outnumber positive candidates"

    # Test HistGradientBoosting pipeline on this single trip subset
    X = get_feature_matrix(df_cands)
    y = get_target_vector(df_cands)

    pipeline = HistGradientBoostingPipeline(max_iter=30, random_state=42)
    pipeline.fit(X, y)

    preds = pipeline.predict(X)
    probs = pipeline.predict_proba(X)

    assert len(preds) == len(X)
    assert len(probs) == len(X)
    assert probs.min() >= 0.0 and probs.max() <= 1.0

    # Test HMM formatting
    hmm_payload = pipeline.format_for_hmm(df_cands)
    assert len(hmm_payload) == 1315
    assert "candidate_arc_ids" in hmm_payload[0]
    assert "probabilities" in hmm_payload[0]
    assert len(hmm_payload[0]["candidate_arc_ids"]) == len(hmm_payload[0]["probabilities"])
