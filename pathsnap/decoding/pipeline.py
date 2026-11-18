"""End-to-end GPS map matching without reference route inputs."""

from pathlib import Path


from pathsnap.candidate_generation.spatial_index import SampledArcKDTree
from pathsnap.feature_engineering.feature_extractor import FeatureExtractor
from pathsnap.preprocessing.gps_preprocessor import GPSPreprocessor
from pathsnap.preprocessing.network_preprocessor import NetworkPreprocessor

from .viterbi import DecoderConfig, ViterbiDecoder


def prepare_trip(trip_dir, model):
    """Load GPS and roads, then generate and score candidates in observation order."""
    trip_dir = Path(trip_dir)
    trip_id = trip_dir.name
    trajectory = GPSPreprocessor().load_and_process(str(trip_dir / f"{trip_id}.track"), trip_id)
    network = NetworkPreprocessor().load_and_process(
        str(trip_dir / f"{trip_id}.nodes"), str(trip_dir / f"{trip_id}.arcs"),
        trajectory.projector, trip_id,
    )
    candidates = SampledArcKDTree(
        sampling_interval_m=25.0, search_radius_m=35.0, fallback_k=15,
    ).build(network).query_trajectory(trajectory.coords_metric)
    features = FeatureExtractor().extract_features_for_trajectory(trajectory, network, candidates)
    flat_scores = model.predict_proba(features) if len(features) else []
    probabilities = []
    start = 0
    for states in candidates:
        end = start + len(states)
        probabilities.append(flat_scores[start:end])
        start = end
    if start != len(flat_scores):
        raise ValueError("Features and candidates disagree on the number of rows.")
    return trajectory, network, candidates, probabilities


def decode_prepared(prepared, config=DecoderConfig(), sequential=True):
    trajectory, network, candidates, probabilities = prepared
    return ViterbiDecoder(network, config).decode(
        trajectory.df.timestamp.to_numpy(), trajectory.coords_metric,
        candidates, probabilities, sequential=sequential,
    )


def predict_trip(trip_dir, model, config=DecoderConfig()):
    """Predict a route using GPS and roads on disk and a loaded candidate model."""
    return decode_prepared(prepare_trip(trip_dir, model), config)

