"""
Tests for pathsnap.preprocessing module on actual dataset trip 00000001.
"""

import os
import pytest
from pathsnap.preprocessing.gps_preprocessor import GPSPreprocessor
from pathsnap.preprocessing.network_preprocessor import NetworkPreprocessor


@pytest.fixture
def sample_trip_dir():
    return "Student folder/train/00000001"


def test_gps_preprocessing(sample_trip_dir):
    track_path = os.path.join(sample_trip_dir, "00000001.track")
    assert os.path.exists(track_path), "Sample track file not found"

    preprocessor = GPSPreprocessor()
    traj = preprocessor.load_and_process(track_path, trip_id="00000001")

    assert traj.trip_id == "00000001"
    assert len(traj) == 1315
    assert traj.coords_metric.shape == (1315, 2)
    assert not traj.df.isna().any().any(), "Found unexpected NaNs in processed GPS dataframe"

    # Check that timestamps are strictly monotonic
    assert traj.metadata["is_strictly_monotonic"] is True
    assert traj.metadata["duplicate_timestamps"] == 0

    # Check kinematic columns
    assert "speed_mps" in traj.df.columns
    assert "bearing_rad" in traj.df.columns
    assert "dt_sec" in traj.df.columns


def test_network_preprocessing(sample_trip_dir):
    nodes_path = os.path.join(sample_trip_dir, "00000001.nodes")
    arcs_path = os.path.join(sample_trip_dir, "00000001.arcs")
    track_path = os.path.join(sample_trip_dir, "00000001.track")

    gps_prep = GPSPreprocessor()
    traj = gps_prep.load_and_process(track_path, trip_id="00000001")

    net_prep = NetworkPreprocessor()
    network = net_prep.load_and_process(nodes_path, arcs_path, traj.projector, trip_id="00000001")

    assert network.num_nodes == 88809
    assert network.num_arcs == 165790
    assert len(network.arc_lengths) == 165790
    assert len(network.arc_bearings) == 165790
    assert network.nodes_metric.shape == (88809, 2)

    # Check directed graph validity
    assert network.graph.number_of_nodes() == 88809
    assert network.num_arcs == 165790
    assert network.graph.number_of_edges() == 165498
    assert network.metadata["duplicate_arcs_count"] == 292
