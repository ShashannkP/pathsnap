"""Tune the team's selected HistGradientBoosting model on all 80 training trips."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import optuna
import pandas as pd
import sklearn
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from pathsnap.experiments.cross_validation import evaluate_predictions
from pathsnap.feature_engineering.feature_extractor import FEATURE_NAMES
from pathsnap.models.hist_gradient_boosting import HistGradientBoostingPipeline


SEED = 42
BASELINE = dict(
    learning_rate=0.08, max_iter=150, max_leaf_nodes=31,
    min_samples_leaf=20, l2_regularization=1.0, class_weight="balanced",
)
DATA = Path("pathsnap/data")
PREVIOUS = Path("pathsnap/artifacts/final_model/best_hyperparameters.json")


def load_candidates(partition, expected_trips):
    path = DATA / f"{partition}_candidates.joblib"
    with path.open("rb") as handle:
        if handle.read(11) == b"version htt":
            raise ValueError(f"{path} is a Git LFS pointer. Download its LFS object first.")
    frame = joblib.load(path).reset_index(drop=True)
    if partition == "student_test":
        frame = frame.loc[frame.target != -1].reset_index(drop=True)
    if frame.trip_id.nunique() != expected_trips:
        raise ValueError(f"{path} must contain {expected_trips} distinct trips.")
    if set(frame.target.unique()) != {0, 1}:
        raise ValueError(f"{path} must contain binary, aligned candidate labels.")
    return frame, frame[FEATURE_NAMES].to_numpy(dtype=float), frame.target.to_numpy(dtype=int)


def make_model(params):
    # A fixed iteration count keeps internal random row holdouts out of the protocol.
    return HistGradientBoostingPipeline(**params, early_stopping=False, random_state=SEED)


def evaluate(frame, scores, name):
    metrics = evaluate_predictions(frame, scores, name)
    keys = ["trip_id", "obs_idx"]
    scored = frame[keys + ["cand_arc_id", "target"]].assign(score=scores)
    top = scored.loc[scored.groupby(keys).score.idxmax(), keys + ["cand_arc_id"]]
    reference = scored.loc[scored.target == 1, keys + ["cand_arc_id"]]
    # The cached data stores true arc IDs only when they were retrieved as candidates.
    covered = reference.merge(top, on=keys, suffixes=("_true", "_pred"), validate="one_to_one")
    by_trip = [
        f1_score(trip.cand_arc_id_true, trip.cand_arc_id_pred, average="macro", zero_division=0)
        for _, trip in covered.groupby("trip_id")
    ]
    metrics["covered_observation_macro_f1"] = float(np.mean(by_trip))
    metrics["candidate_miss_rate_with_rows"] = 1 - len(covered) / len(top)
    return metrics


def run(output, n_trials):
    output.mkdir(parents=True, exist_ok=False)
    train, features, target = load_candidates("train", 80)
    folds = list(GroupKFold(n_splits=5).split(features, target, train.trip_id))
    fold_manifest = [
        {"trip_id": trip, "fold": number}
        for number, (_, valid) in enumerate(folds, 1)
        for trip in sorted(train.iloc[valid].trip_id.unique())
    ]
    pd.DataFrame(fold_manifest).to_csv(output / "fold_manifest.csv", index=False)
    records = []

    def cross_validate(params, trial_number):
        rows = []
        for number, (fit, valid) in enumerate(folds, 1):
            model = make_model(params)
            start = perf_counter()
            model.fit(features[fit], target[fit])
            fit_time = perf_counter() - start
            start = perf_counter()
            scores = model.predict_proba(features[valid])
            score_time = perf_counter() - start
            metrics = evaluate(train.iloc[valid], scores, str(trial_number))
            rows.append(dict(trial=trial_number, fold=number, fit_seconds=fit_time,
                             inference_seconds=score_time, **metrics))
            print(f"Trial {trial_number} fold {number}/5: Macro F1 {metrics['macro_f1']:.4f}", flush=True)
            records.append(rows[-1])
            pd.DataFrame(records).to_csv(output / "trial_folds.csv", index=False)
        return float(np.mean([row["macro_f1"] for row in rows]))

    sampler = optuna.samplers.TPESampler(seed=SEED, n_startup_trials=4)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.enqueue_trial(BASELINE)
    study.enqueue_trial(json.loads(PREVIOUS.read_text(encoding="utf-8"))["parameters"])

    def objective(trial):
        params = dict(
            learning_rate=trial.suggest_float("learning_rate", 0.04, 0.16, log=True),
            max_iter=trial.suggest_int("max_iter", 80, 160, step=10),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 24, 45),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 15, 40, step=5),
            l2_regularization=trial.suggest_float("l2_regularization", 0.1, 5.0, log=True),
            class_weight=trial.suggest_categorical("class_weight", [None, "balanced"]),
        )
        return cross_validate(params, trial.number)

    start = perf_counter()
    study.optimize(
        objective, n_trials=n_trials,
        callbacks=[lambda s, _: s.trials_dataframe().to_csv(output / "optuna_trials.csv", index=False)],
    )
    search_time = perf_counter() - start
    export_comparison(output, train, features, target, study.trials[0].params,
                      study.best_trial.params, study.best_trial.number, n_trials, search_time)


def export_comparison(output, train, features, target, baseline, tuned, best_trial,
                      n_trials, search_time):
    test, test_features, _ = load_candidates("student_test", 10)
    if set(train.trip_id) & set(test.trip_id):
        raise ValueError("Training and student-test trip IDs overlap.")

    comparison = []
    fold_results = pd.read_csv(output / "trial_folds.csv")
    for name, trial, params in (("baseline", 0, baseline), ("tuned", best_trial, tuned)):
        model = make_model(params)
        start = perf_counter()
        model.fit(features, target)
        fit_time = perf_counter() - start
        start = perf_counter()
        scores = model.predict_proba(test_features)
        score_time = perf_counter() - start
        path = output / f"{name}_model.joblib"
        model.save(str(path))
        restored = HistGradientBoostingPipeline.load(str(path))
        np.testing.assert_allclose(restored.predict_proba(test_features[:512]), scores[:512])
        trip = test.loc[test.trip_id == test.trip_id.iloc[0]]
        assert len(restored.format_for_hmm(trip)) == trip.obs_idx.nunique()
        cv = fold_results[fold_results.trial == trial].drop(columns=["trial", "fold"])
        comparison.append(dict(
            model=name, trial=trial, full_fit_seconds=fit_time,
            test_inference_seconds=score_time, model_bytes=path.stat().st_size,
            **{f"cv_{key}": value for key, value in cv.mean().items()},
            **{f"student_{key}": value for key, value in evaluate(test, scores, name).items()},
        ))
        print(f"{name}: refitted and verified on student-test data", flush=True)
    pd.DataFrame(comparison).to_csv(output / "comparison.csv", index=False)
    metadata = dict(
        model_family="HistGradientBoostingClassifier", feature_names=FEATURE_NAMES,
        training_trips=80, training_rows=len(train), student_test_trips=10,
        student_test_rows=len(test), folds=5, seed=SEED, early_stopping=False,
        objective="Mean validation candidate-level macro F1", sampler="Optuna TPE",
        trials=n_trials, search_seconds=search_time, best_trial=best_trial,
        baseline_parameters=baseline, parameters=tuned,
        sklearn_version=sklearn.__version__, optuna_version=optuna.__version__,
    )
    (output / "model_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def finish_existing(output, n_trials):
    """Refit from complete search logs if evaluation or model export was interrupted."""
    trials = pd.read_csv(output / "optuna_trials.csv")
    folds = pd.read_csv(output / "trial_folds.csv")
    expected_folds = {(trial, fold) for trial in range(n_trials) for fold in range(1, 6)}
    if (
        trials.number.tolist() != list(range(n_trials))
        or not trials.state.eq("COMPLETE").all()
        or len(folds) != len(expected_folds)
        or set(zip(folds.trial, folds.fold)) != expected_folds
    ):
        raise ValueError("The saved study does not have all requested trials and folds.")

    def params_for(row):
        params = {key: row[f"params_{key}"] for key in BASELINE}
        for key in ("max_iter", "max_leaf_nodes", "min_samples_leaf"):
            params[key] = int(params[key])
        if pd.isna(params["class_weight"]):
            params["class_weight"] = None
        return params

    best = trials.loc[trials.value.idxmax()]
    duration = (
        pd.to_datetime(trials.datetime_complete).max()
        - pd.to_datetime(trials.datetime_start).min()
    ).total_seconds()
    train, features, target = load_candidates("train", 80)
    export_comparison(output, train, features, target, params_for(trials.iloc[0]),
                      params_for(best), int(best.number), n_trials, duration)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New results directory")
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--finish-existing", action="store_true",
                        help="Refit and evaluate from a completed search in --out")
    args = parser.parse_args()
    if args.trials < 5:
        parser.error("At least five trials are needed to include a TPE-guided trial.")
    if args.finish_existing:
        finish_existing(args.out, args.trials)
    else:
        run(args.out, args.trials)
