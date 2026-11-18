"""Tune the decoder on training trips and compare against independent classifier argmax."""

import argparse
from dataclasses import asdict
from itertools import groupby
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from pathsnap.decoding.pipeline import decode_prepared, prepare_trip
from pathsnap.decoding.viterbi import DecoderConfig
from pathsnap.ground_truth.route_aligner import RouteAligner
from pathsnap.models.hist_gradient_boosting import HistGradientBoostingPipeline


def edit_distance(left, right):
    """Unit-cost Levenshtein distance on ordered arc IDs."""
    if len(right) > len(left):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row, source in enumerate(left, 1):
        current = [row]
        for col, target in enumerate(right, 1):
            current.append(min(previous[col] + 1, current[-1] + 1,
                               previous[col - 1] + (source != target)))
        previous = current
    return previous[-1]


def deduplicate(arcs):
    return [arc for arc, _ in groupby(arcs)]


def load_reference(trip_dir, prepared):
    trajectory, network, candidates, _ = prepared
    route_file = trip_dir / f"{trip_dir.name}.route"
    alignments = RouteAligner().align_trajectory_to_route(
        trajectory, network, str(route_file), candidates,
    )
    route = np.loadtxt(route_file, dtype=int, ndmin=1).tolist()
    return alignments, deduplicate(route)


def evaluate(result, reference, measure_sequence=True):
    alignments, route = reference
    reliable = [alignment for alignment in alignments if alignment.is_reliable]
    correct = sum(result.observation_arc_ids[a.obs_idx] == a.true_arc_id for a in reliable)
    continuity = (
        result.connected_transitions / result.eligible_transitions
        if result.eligible_transitions else None
    )
    metrics = dict(
        observations=len(alignments), aligned_observations=len(reliable),
        assigned_observations=sum(arc is not None for arc in result.observation_arc_ids),
        pointwise_accuracy=correct / len(reliable) if reliable else None,
        eligible_transitions=result.eligible_transitions,
        connected_transitions=result.connected_transitions,
        continuity=continuity, breaks=len(result.breaks),
    )
    if measure_sequence:
        predicted = [arc for segment in result.route_segments for arc in deduplicate(segment)]
        metrics["normalized_route_edit_distance"] = edit_distance(predicted, route) / max(
            len(predicted), len(route), 1,
        )
    return metrics


def run(raw_root, model_path, output):
    splits = {name: sorted(path for path in (raw_root / name).iterdir() if path.is_dir())
              for name in ("train", "student_test")}
    if len(splits["train"]) != 80 or len(splits["student_test"]) != 10:
        raise ValueError("Expected 80 training trips and 10 student-test trips.")
    if {p.name for p in splits["train"]} & {p.name for p in splits["student_test"]}:
        raise ValueError("Training and student-test trips overlap.")
    output.mkdir(parents=True, exist_ok=False)
    model = HistGradientBoostingPipeline.load(str(model_path))
    configurations = {beta: DecoderConfig(beta_m=beta) for beta in (10.0, 25.0, 50.0)}
    calibration = []
    for trip in splits["train"][::20]:
        prepared = prepare_trip(trip, model)
        predictions = {beta: decode_prepared(prepared, config)
                       for beta, config in configurations.items()}
        reference = load_reference(trip, prepared)
        for beta, predicted in predictions.items():
            calibration.append(dict(trip_id=trip.name, beta_m=beta,
                                    **evaluate(predicted, reference, measure_sequence=False)))
        print(f"Decoder development trip {trip.name}", flush=True)
    development = pd.DataFrame(calibration)
    development.to_csv(output / "decoder_development.csv", index=False)
    mean_accuracy = development.groupby("beta_m").pointwise_accuracy.mean()
    config = configurations[float(mean_accuracy.idxmax())]
    (output / "decoder_config.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8",
    )

    results = []
    for trip in splits["student_test"]:
        start = perf_counter()
        prepared = prepare_trip(trip, model)
        prepare_seconds = perf_counter() - start
        predictions = {}
        for name, sequential in (("classifier_only", False), ("classifier_plus_hmm", True)):
            start = perf_counter()
            predicted = decode_prepared(prepared, config, sequential)
            predictions[name] = (predicted, perf_counter() - start)
        reference = load_reference(trip, prepared)
        for name, (predicted, elapsed) in predictions.items():
            results.append(dict(trip_id=trip.name, method=name, prepare_seconds=prepare_seconds,
                                decode_seconds=elapsed, **evaluate(predicted, reference)))
        (output / f"{trip.name}_predictions.json").write_text(
            json.dumps({name: asdict(route) for name, (route, _) in predictions.items()}),
            encoding="utf-8",
        )
        pd.DataFrame(results).to_csv(output / "per_trip.csv", index=False)
        print(f"Decoder student-test trip {trip.name}", flush=True)
    frame = pd.DataFrame(results)
    frame.groupby("method").mean(numeric_only=True).to_csv(output / "comparison.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("Student folder"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.raw_root, args.model, args.out)
