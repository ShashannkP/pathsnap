from types import SimpleNamespace

import numpy as np

from pathsnap.decoding.viterbi import ViterbiDecoder


def state(arc, position=0.5):
    return SimpleNamespace(arc_id=arc, tau=position)


def decoder():
    return ViterbiDecoder(SimpleNamespace(
        arcs=np.array([[0, 1], [1, 2], [2, 3], [4, 5]]),
        arc_lengths=np.array([10.0, 10.0, 10.0, 10.0]),
        node_outgoing_arcs={0: [0], 1: [1], 2: [2], 3: [], 4: [3], 5: []},
    ))


def test_viterbi_keeps_connected_alternative_and_inserts_connector():
    candidates = [[state(0)], [state(3), state(2)]]
    scores = [[0.9], [0.95, 0.6]]
    coords = [[0, 0], [20, 0]]
    decoded = decoder().decode([0, 1], np.array(coords), candidates, scores)
    baseline = decoder().decode([0, 1], np.array(coords), candidates, scores, sequential=False)
    assert decoded.observation_arc_ids == [0, 2]
    assert decoded.route_segments == [[0, 1, 2]]
    assert decoded.connected_transitions == 1
    assert baseline.observation_arc_ids == [0, 3]
    assert baseline.breaks == {1: "no_plausible_transition"}


def test_missing_and_gap_restart_the_sequence():
    candidates = [[state(0)], [], [state(2)], [state(0)]]
    result = decoder().decode(
        [0, 1, 2, 99], np.zeros((4, 2)), candidates, [[0.5], [], [0.5], [0.5]],
    )
    assert result.observation_arc_ids == [0, None, 2, 0]
    assert result.route_segments == [[0], [2], [0]]
    assert result.breaks == {1: "missing_candidates", 3: "timestamp_gap"}


def test_global_backtracking_can_change_the_first_observation():
    candidates = [[state(3), state(0)], [state(2)]]
    result = decoder().decode(
        [0, 1], np.array([[0, 0], [20, 0]]), candidates, [[0.99, 0.2], [0.8]],
    )
    assert result.observation_arc_ids == [0, 2]


def test_directed_roads_do_not_allow_backward_travel_between_arcs():
    result = decoder().decode(
        [0, 1], np.zeros((2, 2)), [[state(2)], [state(0)]], [[0.5], [0.5]],
    )
    assert result.route_segments == [[2], [0]]
    assert result.breaks == {1: "no_plausible_transition"}
