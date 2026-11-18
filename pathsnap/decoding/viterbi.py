"""Directed map-matching transitions and log-space Viterbi decoding."""

from dataclasses import dataclass
from functools import lru_cache
from heapq import heappop, heappush
from math import inf

import numpy as np


@dataclass(frozen=True)
class DecoderConfig:
    beta_m: float = 25.0
    max_gap_s: float = 10.0
    max_speed_mps: float = 55.0
    distance_slack_m: float = 35.0
    reverse_jitter_m: float = 10.0

    def __post_init__(self):
        for value in (self.beta_m, self.max_gap_s, self.max_speed_mps):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("Beta, maximum gap, and maximum speed must be finite and positive.")
        for value in (self.distance_slack_m, self.reverse_jitter_m):
            if not np.isfinite(value) or value < 0:
                raise ValueError("Distance slack and reverse jitter must be finite and nonnegative.")


@dataclass(frozen=True)
class DecodedRoute:
    observation_arc_ids: list
    route_segments: list
    breaks: dict
    eligible_transitions: int
    connected_transitions: int


class ViterbiDecoder:
    """Score states by candidate probability and transitions by network discrepancy."""

    def __init__(self, network, config=DecoderConfig()):
        self.network = network
        self.config = config
        self.paths_from = lru_cache(maxsize=1024)(self._paths_from)

    def _paths_from(self, source):
        limit = self.config.max_speed_mps * self.config.max_gap_s + self.config.distance_slack_m
        distance = {source: 0.0}
        previous_arc = {}
        queue = [(0.0, source)]
        while queue:
            cost, node = heappop(queue)
            if cost != distance[node]:
                continue
            for arc in self.network.node_outgoing_arcs[node]:
                neighbor = int(self.network.arcs[arc, 1])
                next_cost = cost + max(float(self.network.arc_lengths[arc]), 0.001)
                if next_cost <= limit and next_cost < distance.get(neighbor, inf):
                    distance[neighbor] = next_cost
                    previous_arc[neighbor] = arc
                    heappush(queue, (next_cost, neighbor))
        return distance, previous_arc

    def _distance(self, before, after):
        arc_a, arc_b = before.arc_id, after.arc_id
        length_a = self.network.arc_lengths[arc_a]
        if arc_a == arc_b:
            progress = (after.tau - before.tau) * length_a
            if progress >= -self.config.reverse_jitter_m:
                return max(0.0, progress), False
        source = int(self.network.arcs[arc_a, 1])
        destination = int(self.network.arcs[arc_b, 0])
        distances, _ = self.paths_from(source)
        length_b = self.network.arc_lengths[arc_b]
        return (1 - before.tau) * length_a + distances.get(destination, inf) + after.tau * length_b, True

    def _connectors(self, before, after):
        source = int(self.network.arcs[before.arc_id, 1])
        node = int(self.network.arcs[after.arc_id, 0])
        _, previous = self.paths_from(source)
        result = []
        while node != source:
            arc = previous[node]
            result.append(int(arc))
            node = int(self.network.arcs[arc, 0])
        return result[::-1]

    def decode(self, timestamps, coords, candidates, probabilities, sequential=True):
        """Return one arc (or None) per observation and connected route segments."""
        if not len(timestamps) == len(coords) == len(candidates) == len(probabilities):
            raise ValueError("Inputs must contain the same number of observations.")
        chosen = [None] * len(candidates)
        breaks = {}
        ends = []
        back = [None] * len(candidates)
        scores = None
        for t, (states, values) in enumerate(zip(candidates, probabilities)):
            values = np.asarray(values, dtype=float)
            if values.shape != (len(states),) or not np.isfinite(values).all():
                raise ValueError(f"Invalid candidate probabilities at observation {t}.")
            if not states:
                breaks[t] = "missing_candidates"
                if scores is not None:
                    ends.append((t - 1, int(np.argmax(scores))))
                scores = None
                continue
            if not sequential:
                chosen[t] = int(np.argmax(values))
                continue
            emissions = np.log(np.clip(values, 1e-12, 1.0))
            if scores is None:
                scores = emissions
                continue
            dt = timestamps[t] - timestamps[t - 1]
            if not 0 < dt <= self.config.max_gap_s:
                ends.append((t - 1, int(np.argmax(scores))))
                breaks[t] = "timestamp_gap"
                scores = emissions
                continue
            observed_distance = np.linalg.norm(coords[t] - coords[t - 1])
            limit = self.config.max_speed_mps * dt + self.config.distance_slack_m
            best = np.full(len(states), -np.inf)
            parents = np.full(len(states), -1, dtype=int)
            for i, before in enumerate(candidates[t - 1]):
                if not np.isfinite(scores[i]):
                    continue
                for j, after in enumerate(states):
                    distance, _ = self._distance(before, after)
                    if distance <= limit:
                        value = scores[i] - abs(distance - observed_distance) / self.config.beta_m
                        if value > best[j]:
                            best[j] = value
                            parents[j] = i
            if np.isfinite(best).any():
                scores = best + emissions
                back[t] = parents
            else:
                ends.append((t - 1, int(np.argmax(scores))))
                breaks[t] = "no_plausible_transition"
                scores = emissions

        if scores is not None:
            ends.append((len(candidates) - 1, int(np.argmax(scores))))
        for t, state in ends:
            chosen[t] = state
            while back[t] is not None:
                state = int(back[t][state])
                t -= 1
                chosen[t] = state
        return self._assemble(timestamps, candidates, chosen, breaks)

    def _assemble(self, timestamps, candidates, chosen, breaks):
        arcs = [int(states[index].arc_id) if index is not None else None
                for states, index in zip(candidates, chosen)]
        segments = []
        connected = eligible = 0
        for t, arc in enumerate(arcs):
            if arc is None:
                continue
            if t == 0 or arcs[t - 1] is None:
                segments.append([arc])
                continue
            dt = timestamps[t] - timestamps[t - 1]
            if not 0 < dt <= self.config.max_gap_s:
                breaks[t] = "timestamp_gap"
                segments.append([arc])
                continue
            eligible += 1
            before = candidates[t - 1][chosen[t - 1]]
            after = candidates[t][chosen[t]]
            distance, via_nodes = self._distance(before, after)
            limit = self.config.max_speed_mps * dt + self.config.distance_slack_m
            if t in breaks or distance > limit:
                breaks[t] = "no_plausible_transition"
                segments.append([arc])
                continue
            connected += 1
            if via_nodes:
                segments[-1].extend(self._connectors(before, after))
                segments[-1].append(arc)
        return DecodedRoute(arcs, segments, breaks, eligible, connected)
