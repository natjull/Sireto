#!/usr/bin/env python3
"""Run the sealed V4.8 acceptor development experiment without random data."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402


SCHEMA_VERSION = "sireto-v4.8-acceptor-development-1"
EXPECTED_HASHES = {
    "contract": "cbeeb2394dad43e1c85a66841cd3497471ed943f44c5b99d8c4cc1733913b717",
    "partition_manifest": "f0e255b891dfb6b24d57f3b7423dd64a227908dbf68559b2da4572ea37791d33",
    "partition_assignments": "f828249172c36ce33a3279d294dfc5030e6d8eeb58baee9cf9e08130f13593b9",
    "historical_scenes": "8f3bc4633ada9eb6347e47a1029f0e69fa8946b1c3c1df38c72232f572088dc9",
    "current_labels": "e5e592d4dcd540273378dada7128f957b1d335df63fbc88f4c1377c0f9337bd2",
    "frozen_model": "16283b8aba5ed135846a74e9040c79e9f863f7e2bd658ca642ad444174b9a3fa",
    "frozen_metadata": "73199451b2de6ae383c9c0c58b10ab9c7393994a4efdec45f9c8e1e9f150691c",
}
VARIANT_WEIGHTS = {
    "BASE_REFIT": 0,
    "HARD_W1": 1,
    "HARD_W2": 2,
    "HARD_W4": 4,
}
FROZEN_VARIANT = "BASE_FROZEN"
FROZEN_THRESHOLD = 0.46313316267954524
MIN_AUTO_COUNT = 100


def _json_dump(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _assert_columns(frame: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [0.0, 1.0]
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = proportion + z * z / (2.0 * total)
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    )
    return [(centre - margin) / denominator, (centre + margin) / denominator]


def decision_metrics(
    scores: np.ndarray,
    targets: np.ndarray,
    threshold: float,
    *,
    ambiguous: np.ndarray | None = None,
) -> dict[str, Any]:
    scores = np.asarray(scores, dtype=float)
    targets = np.asarray(targets, dtype=int)
    auto = scores >= float(threshold)
    auto_count = int(auto.sum())
    correct_auto = int((auto & targets.astype(bool)).sum())
    error_auto = auto_count - correct_auto
    ambiguous_auto = (
        int((auto & np.asarray(ambiguous, dtype=bool)).sum())
        if ambiguous is not None
        else 0
    )
    return {
        "threshold": float(threshold),
        "row_count": int(len(scores)),
        "auto_count": auto_count,
        "correct_auto": correct_auto,
        "error_auto": error_auto,
        "review_count": int(len(scores) - auto_count),
        "precision": correct_auto / auto_count if auto_count else None,
        "precision_wilson_95": wilson_interval(correct_auto, auto_count),
        "coverage": auto_count / len(scores) if len(scores) else 0.0,
        "coverage_wilson_95": wilson_interval(auto_count, len(scores)),
        "ambiguous_auto": ambiguous_auto,
    }


def threshold_curve(
    scores: np.ndarray,
    targets: np.ndarray,
    *,
    ambiguous: np.ndarray,
) -> pd.DataFrame:
    scores = np.asarray(scores, dtype=float)
    candidates = np.unique(
        np.concatenate(
            [
                scores,
                [
                    np.nextafter(scores.max(), np.inf),
                    np.nextafter(scores.min(), -np.inf),
                ],
            ]
        )
    )
    rows = [
        decision_metrics(scores, targets, threshold, ambiguous=ambiguous)
        for threshold in candidates
    ]
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def select_threshold(
    scores: np.ndarray,
    targets: np.ndarray,
    *,
    ambiguous: np.ndarray,
    max_ambiguous_auto: int,
    min_auto_count: int = MIN_AUTO_COUNT,
) -> tuple[float, dict[str, Any]] | None:
    curve = threshold_curve(scores, targets, ambiguous=ambiguous)
    eligible = curve[
        curve["auto_count"].ge(min_auto_count)
        & (
            1000 * curve["correct_auto"].astype(int)
            >= 998 * curve["auto_count"].astype(int)
        )
        & curve["ambiguous_auto"].le(int(max_ambiguous_auto))
    ].copy()
    if eligible.empty:
        return None
    eligible["_precision_sort"] = (
        eligible["correct_auto"] / eligible["auto_count"]
    )
    winner = eligible.sort_values(
        ["auto_count", "_precision_sort", "threshold"],
        ascending=[False, False, False],
    ).iloc[0]
    threshold = float(winner["threshold"])
    metrics = decision_metrics(
        scores,
        targets,
        threshold,
        ambiguous=ambiguous,
    )
    return threshold, metrics


def make_model() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=42,
                ),
            ),
        ]
    )


def fit_model(
    frame: pd.DataFrame,
    *,
    feature_order: list[str],
    sample_weights: np.ndarray | None = None,
) -> Pipeline:
    if frame.empty:
        raise ValueError("Cannot fit V4.8 on an empty frame")
    targets = frame["acceptor_target"].astype(int).to_numpy()
    if np.unique(targets).size != 2:
        raise ValueError("V4.8 fit requires both target classes")
    weights = (
        np.ones(len(frame), dtype=float)
        if sample_weights is None
        else np.asarray(sample_weights, dtype=float)
    )
    if len(weights) != len(frame) or (weights <= 0).any():
        raise ValueError("Invalid V4.8 sample weights")
    model = make_model()
    model.fit(
        frame[feature_order].astype(float).to_numpy(),
        targets,
        model__sample_weight=weights,
    )
    return model


def model_scores(
    model: Any,
    frame: pd.DataFrame,
    *,
    feature_order: list[str],
) -> np.ndarray:
    if frame.empty:
        return np.array([], dtype=float)
    matrix = frame[feature_order].astype(float).to_numpy()
    return np.asarray(model.predict_proba(matrix)[:, 1], dtype=float)


def model_parameters(model: Pipeline) -> dict[str, Any]:
    logistic = model.named_steps["model"]
    return {
        "coefficients": logistic.coef_.astype(float).tolist(),
        "intercept": logistic.intercept_.astype(float).tolist(),
        "n_iter": logistic.n_iter_.astype(int).tolist(),
    }


def _load_development_frames(
    *,
    historical_scenes_path: Path,
    current_labels_path: Path,
    partition_assignments_path: Path,
    feature_order: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scenes = pd.read_parquet(historical_scenes_path)
    assignments = pd.read_parquet(partition_assignments_path)
    for frame in (scenes, assignments):
        frame["query_id"] = frame["query_id"].astype(str)
    _assert_columns(
        assignments,
        [
            "population",
            "query_id",
            "role",
            "hard_fold",
            "acceptor_target",
            "adjudication_label",
            "evidence_validated",
            "component_id",
        ],
        "partition assignments",
    )
    _assert_columns(
        scenes,
        [
            "query_id",
            "split",
            "label_kind",
            "is_exact_siret_correct",
            "acceptor_eligible",
            *feature_order,
        ],
        "historical scenes",
    )
    historical_assignments = assignments[assignments["population"].eq("historical")]
    current_assignments = assignments[assignments["population"].eq("current")]
    historical = scenes.merge(
        historical_assignments[
            [
                "query_id",
                "role",
                "hard_fold",
                "component_id",
            ]
        ],
        on="query_id",
        validate="one_to_one",
    )
    historical["acceptor_target"] = historical["is_exact_siret_correct"].astype(int)
    historical_fit = historical[
        historical["split"].eq("fit")
        & historical["acceptor_eligible"].astype(bool)
        & historical["role"].isin({"historical_fit", "historical_hard_support"})
    ].copy()
    historical_dev = historical[
        historical["split"].eq("dev")
        & historical["acceptor_eligible"].astype(bool)
        & historical["role"].eq("historical_dev")
    ].copy()

    random_ids = set(
        current_assignments.loc[
            current_assignments["role"].eq("random_sealed"), "query_id"
        ].astype(str)
    )
    development_assignments = current_assignments[
        current_assignments["role"].isin({"hard_oof", "hard_dev_locked"})
        & current_assignments["evidence_validated"].astype(bool)
        & current_assignments["acceptor_target"].notna()
    ].copy()
    if set(development_assignments["query_id"]) & random_ids:
        raise ValueError("Random V4.8 query entered development")
    development_ids = sorted(development_assignments["query_id"].astype(str))
    current_columns = [
        "query_id",
        "current_label_origin",
        *feature_order,
    ]
    current = pd.read_parquet(
        current_labels_path,
        columns=current_columns,
        filters=[("query_id", "in", development_ids)],
    )
    current["query_id"] = current["query_id"].astype(str)
    _assert_columns(current, current_columns, "current development scenes")
    if set(current["query_id"]) != set(development_ids):
        raise ValueError("Filtered V4.8 current-scene read is incomplete")
    if set(current["query_id"]) & random_ids:
        raise ValueError("Filtered V4.8 current-scene read opened a random row")
    development = development_assignments.merge(
        current[current_columns],
        on="query_id",
        validate="one_to_one",
    )
    hard_oof = development[development["role"].eq("hard_oof")].copy()
    hard_locked = development[development["role"].eq("hard_dev_locked")].copy()
    if (
        len(historical_fit) != 5501
        or len(historical_dev) != 1452
        or len(hard_oof) != 94
        or len(hard_locked) != 4
    ):
        raise ValueError(
            "V4.8 development population mismatch: "
            f"fit={len(historical_fit)}, dev={len(historical_dev)}, "
            f"hard_oof={len(hard_oof)}, hard_locked={len(hard_locked)}"
        )
    if hard_oof["hard_fold"].isna().any():
        raise ValueError("V4.8 hard OOF fold is missing")
    if set(hard_oof["hard_fold"].astype(int)) != set(range(5)):
        raise ValueError("V4.8 requires five non-empty hard folds")
    all_original_dev = scenes[
        scenes["split"].eq("dev") & scenes["acceptor_eligible"].astype(bool)
    ].copy()
    return historical_fit, historical_dev, hard_oof, hard_locked, all_original_dev


def _training_frame(
    historical_fit: pd.DataFrame,
    hard: pd.DataFrame,
    *,
    hard_weight: int,
    held_out_fold: int | None,
) -> tuple[pd.DataFrame, np.ndarray]:
    historical = historical_fit
    if held_out_fold is not None:
        historical = historical[
            historical["hard_fold"].isna()
            | historical["hard_fold"].astype("Int64").ne(held_out_fold)
        ]
    historical = historical.copy()
    historical["_source"] = "historical"
    if hard_weight == 0:
        combined = historical
        weights = np.ones(len(combined), dtype=float)
        return combined, weights
    hard_fit = hard
    if held_out_fold is not None:
        hard_fit = hard[hard["hard_fold"].astype(int).ne(held_out_fold)]
    hard_fit = hard_fit.copy()
    hard_fit["_source"] = "hard"
    combined = pd.concat([historical, hard_fit], ignore_index=True, sort=False)
    weights = np.where(combined["_source"].eq("hard"), float(hard_weight), 1.0)
    return combined, weights


def _paired_hard_metrics(predictions: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for variant, frame in predictions.groupby("variant", sort=False):
        wrong = frame["adjudication_label"].eq("TOP1_WRONG")
        correct = frame["adjudication_label"].eq("TOP1_CORRECT")
        ambiguous = frame["adjudication_label"].eq("AMBIGUOUS")
        output[str(variant)] = {
            "row_count": int(len(frame)),
            "wrong_count": int(wrong.sum()),
            "wrong_rejected": int((wrong & ~frame["auto"]).sum()),
            "wrong_auto": int((wrong & frame["auto"]).sum()),
            "wrong_rejection_wilson_95": wilson_interval(
                int((wrong & ~frame["auto"]).sum()), int(wrong.sum())
            ),
            "correct_count": int(correct.sum()),
            "correct_auto": int((correct & frame["auto"]).sum()),
            "correct_acceptance_rate": (
                float(frame.loc[correct, "auto"].mean()) if correct.any() else 0.0
            ),
            "correct_acceptance_wilson_95": wilson_interval(
                int((correct & frame["auto"]).sum()), int(correct.sum())
            ),
            "ambiguous_count": int(ambiguous.sum()),
            "ambiguous_auto": int((ambiguous & frame["auto"]).sum()),
        }
    return output


def _hard_segment_metrics(predictions: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant, variant_frame in predictions.groupby("variant", sort=False):
        for field in ("sampling_stratum", "current_label_origin"):
            for value, frame in variant_frame.groupby(field, dropna=False, sort=True):
                rows.append(
                    {
                        "variant": str(variant),
                        "segment_field": field,
                        "segment_value": _text(value),
                        "row_count": int(len(frame)),
                        "auto_count": int(frame["auto"].sum()),
                        "correct_count": int(
                            frame["adjudication_label"].eq("TOP1_CORRECT").sum()
                        ),
                        "correct_auto": int(
                            (
                                frame["adjudication_label"].eq("TOP1_CORRECT")
                                & frame["auto"]
                            ).sum()
                        ),
                        "wrong_count": int(
                            frame["adjudication_label"].eq("TOP1_WRONG").sum()
                        ),
                        "wrong_rejected": int(
                            (
                                frame["adjudication_label"].eq("TOP1_WRONG")
                                & ~frame["auto"]
                            ).sum()
                        ),
                        "ambiguous_auto": int(
                            (
                                frame["adjudication_label"].eq("AMBIGUOUS")
                                & frame["auto"]
                            ).sum()
                        ),
                    }
                )
    return rows


def _paired_transitions(predictions: pd.DataFrame) -> dict[str, Any]:
    base = predictions[predictions["variant"].eq("BASE_REFIT")][
        ["query_id", "adjudication_label", "auto"]
    ].rename(columns={"auto": "base_auto"})
    output: dict[str, Any] = {}
    for variant in ("HARD_W1", "HARD_W2", "HARD_W4"):
        candidate = predictions[predictions["variant"].eq(variant)][
            ["query_id", "auto"]
        ].rename(columns={"auto": "candidate_auto"})
        if candidate.empty:
            output[variant] = {"status": "UNAVAILABLE_NO_SAFE_OOF_THRESHOLD"}
            continue
        paired = base.merge(candidate, on="query_id", validate="one_to_one")
        if len(paired) != len(base):
            raise ValueError(f"Incomplete paired hard transitions for {variant}")
        transitions: dict[str, int] = {}
        for base_auto, candidate_auto, label in zip(
            paired["base_auto"],
            paired["candidate_auto"],
            paired["adjudication_label"],
            strict=True,
        ):
            key = (
                f"{'AUTO' if bool(base_auto) else 'REVIEW'}"
                f"→{'AUTO' if bool(candidate_auto) else 'REVIEW'}"
                f":{label}"
            )
            transitions[key] = transitions.get(key, 0) + 1
        output[variant] = transitions
    return output


def _variant_gate(
    *,
    frozen: dict[str, Any],
    variant: dict[str, Any],
    hard_metrics: dict[str, Any],
    base_hard_metrics: dict[str, Any],
) -> dict[str, Any]:
    precision_target = (
        1000 * int(variant["correct_auto"]) >= 998 * int(variant["auto_count"])
    )
    not_below_frozen = (
        int(variant["correct_auto"]) * int(frozen["auto_count"])
        >= int(frozen["correct_auto"]) * int(variant["auto_count"])
    )
    coverage_ok = float(variant["coverage"]) + 1e-15 >= float(frozen["coverage"]) - 0.02
    wrong_gain = int(hard_metrics["wrong_rejected"]) - int(
        base_hard_metrics["wrong_rejected"]
    )
    correct_loss = float(base_hard_metrics["correct_acceptance_rate"]) - float(
        hard_metrics["correct_acceptance_rate"]
    )
    ambiguous_ok = int(hard_metrics["ambiguous_auto"]) <= int(
        base_hard_metrics["ambiguous_auto"]
    )
    checks = {
        "historical_precision_at_least_0_998": precision_target,
        "historical_precision_not_below_frozen": not_below_frozen,
        "historical_coverage_within_2pp": coverage_ok,
        "wrong_rejected_gain_at_least_4": wrong_gain >= 4,
        "correct_acceptance_loss_at_most_5pp": correct_loss <= 0.05 + 1e-15,
        "no_extra_ambiguous_auto": ambiguous_ok,
    }
    return {
        "admissible": all(checks.values()),
        "checks": checks,
        "wrong_rejected_gain": wrong_gain,
        "correct_acceptance_loss": correct_loss,
    }


def run_development(
    *,
    contract_path: Path,
    partition_manifest_path: Path,
    partition_assignments_path: Path,
    historical_scenes_path: Path,
    current_labels_path: Path,
    frozen_model_path: Path,
    frozen_metadata_path: Path,
    output_root: Path,
    enforce_canonical: bool = True,
) -> Path:
    paths = {
        "contract": Path(contract_path).resolve(),
        "partition_manifest": Path(partition_manifest_path).resolve(),
        "partition_assignments": Path(partition_assignments_path).resolve(),
        "historical_scenes": Path(historical_scenes_path).resolve(),
        "current_labels": Path(current_labels_path).resolve(),
        "frozen_model": Path(frozen_model_path).resolve(),
        "frozen_metadata": Path(frozen_metadata_path).resolve(),
    }
    hashes = {name: file_sha256(path) for name, path in paths.items()}
    if enforce_canonical:
        mismatches = {
            name: (EXPECTED_HASHES[name], actual)
            for name, actual in hashes.items()
            if actual != EXPECTED_HASHES[name]
        }
        if mismatches:
            raise ValueError(f"V4.8 development input mismatch: {mismatches}")
    partition_manifest = json.loads(
        paths["partition_manifest"].read_text(encoding="utf-8")
    )
    if partition_manifest.get("invariants", {}).get("random_targets_exposed") is not False:
        raise ValueError("V4.8 partition manifest does not seal random targets")
    metadata = json.loads(paths["frozen_metadata"].read_text(encoding="utf-8"))
    feature_order = [str(value) for value in metadata["feature_order"]]
    if len(feature_order) != 80:
        raise ValueError("V4.8 requires exactly 80 features")

    historical_fit, historical_dev, hard_oof, hard_locked, original_dev = (
        _load_development_frames(
            historical_scenes_path=paths["historical_scenes"],
            current_labels_path=paths["current_labels"],
            partition_assignments_path=paths["partition_assignments"],
            feature_order=feature_order,
        )
    )
    frozen_model = joblib.load(paths["frozen_model"])
    original_scores = model_scores(frozen_model, original_dev, feature_order=feature_order)
    original_metrics = decision_metrics(
        original_scores,
        original_dev["is_exact_siret_correct"].astype(int).to_numpy(),
        FROZEN_THRESHOLD,
        ambiguous=original_dev["label_kind"].eq("AMBIGUOUS").to_numpy(),
    )
    if (
        original_metrics["row_count"] != 1456
        or original_metrics["auto_count"] != 1188
        or original_metrics["correct_auto"] != 1186
        or original_metrics["error_auto"] != 2
    ):
        raise ValueError(f"STOP_REPRODUCTION: frozen baseline={original_metrics}")

    dev_targets = historical_dev["acceptor_target"].astype(int).to_numpy()
    dev_ambiguous = historical_dev["label_kind"].eq("AMBIGUOUS").to_numpy()
    frozen_dev_scores = model_scores(
        frozen_model, historical_dev, feature_order=feature_order
    )
    frozen_dev_metrics = decision_metrics(
        frozen_dev_scores,
        dev_targets,
        FROZEN_THRESHOLD,
        ambiguous=dev_ambiguous,
    )
    max_ambiguous_auto = int(frozen_dev_metrics["ambiguous_auto"])

    complete_models: dict[str, Pipeline] = {}
    complete_thresholds: dict[str, float] = {}
    dev_predictions: list[pd.DataFrame] = []
    curves: list[pd.DataFrame] = []
    coefficients: dict[str, Any] = {
        FROZEN_VARIANT: {
            "model_sha256": hashes["frozen_model"],
            "parameters": model_parameters(frozen_model),
        }
    }
    frozen_dev_output = historical_dev[
        ["query_id", "label_kind", "acceptor_target"]
    ].copy()
    frozen_dev_output["variant"] = FROZEN_VARIANT
    frozen_dev_output["score"] = frozen_dev_scores
    frozen_dev_output["threshold"] = FROZEN_THRESHOLD
    frozen_dev_output["auto"] = frozen_dev_scores >= FROZEN_THRESHOLD
    dev_predictions.append(frozen_dev_output)
    frozen_curve = threshold_curve(
        frozen_dev_scores, dev_targets, ambiguous=dev_ambiguous
    )
    frozen_curve["variant"] = FROZEN_VARIANT
    curves.append(frozen_curve)

    reproducibility_scores: list[np.ndarray] = []
    for repetition in range(2):
        train, weights = _training_frame(
            historical_fit,
            hard_oof,
            hard_weight=0,
            held_out_fold=None,
        )
        repeated_model = fit_model(
            train, feature_order=feature_order, sample_weights=weights
        )
        reproducibility_scores.append(
            model_scores(repeated_model, historical_dev, feature_order=feature_order)
        )
        if repetition == 0:
            reproducibility_model = repeated_model
    if not np.allclose(
        reproducibility_scores[0],
        reproducibility_scores[1],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("STOP_REPRODUCTION: BASE_REFIT scores are not deterministic")
    first_parameters = model_parameters(reproducibility_model)
    second_parameters = model_parameters(repeated_model)
    if not np.allclose(
        np.asarray(first_parameters["coefficients"]),
        np.asarray(second_parameters["coefficients"]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("STOP_REPRODUCTION: BASE_REFIT coefficients differ")

    complete_dev_metrics: dict[str, dict[str, Any]] = {
        FROZEN_VARIANT: frozen_dev_metrics
    }
    for variant, hard_weight in VARIANT_WEIGHTS.items():
        if variant == "BASE_REFIT":
            model = reproducibility_model
        else:
            train, weights = _training_frame(
                historical_fit,
                hard_oof,
                hard_weight=hard_weight,
                held_out_fold=None,
            )
            model = fit_model(
                train, feature_order=feature_order, sample_weights=weights
            )
        scores = model_scores(model, historical_dev, feature_order=feature_order)
        selected = select_threshold(
            scores,
            dev_targets,
            ambiguous=dev_ambiguous,
            max_ambiguous_auto=max_ambiguous_auto,
        )
        if selected is None:
            complete_thresholds[variant] = float("nan")
            complete_dev_metrics[variant] = {
                "threshold": None,
                "row_count": len(historical_dev),
                "auto_count": 0,
                "correct_auto": 0,
                "error_auto": 0,
                "review_count": len(historical_dev),
                "precision": None,
                "precision_wilson_95": [0.0, 1.0],
                "coverage": 0.0,
                "ambiguous_auto": 0,
                "threshold_found": False,
            }
        else:
            threshold, metrics = selected
            complete_thresholds[variant] = threshold
            metrics["threshold_found"] = True
            complete_dev_metrics[variant] = metrics
        complete_models[variant] = model
        coefficients[variant] = model_parameters(model)
        output = historical_dev[["query_id", "label_kind", "acceptor_target"]].copy()
        output["variant"] = variant
        output["score"] = scores
        output["threshold"] = complete_thresholds[variant]
        output["auto"] = (
            scores >= complete_thresholds[variant]
            if np.isfinite(complete_thresholds[variant])
            else False
        )
        dev_predictions.append(output)
        curve = threshold_curve(scores, dev_targets, ambiguous=dev_ambiguous)
        curve["variant"] = variant
        curves.append(curve)

    hard_predictions: list[pd.DataFrame] = []
    fold_thresholds: dict[str, dict[str, float]] = {
        variant: {} for variant in VARIANT_WEIGHTS
    }
    variant_failures: dict[str, list[str]] = {}
    for variant, hard_weight in VARIANT_WEIGHTS.items():
        if not np.isfinite(complete_thresholds[variant]):
            if variant == "BASE_REFIT":
                raise ValueError("STOP_RETRAIN: BASE_REFIT has no safe complete threshold")
            variant_failures[variant] = ["complete_model_no_safe_threshold"]
            continue
        variant_predictions: list[pd.DataFrame] = []
        for fold in range(5):
            train, weights = _training_frame(
                historical_fit,
                hard_oof,
                hard_weight=hard_weight,
                held_out_fold=fold,
            )
            model = fit_model(
                train, feature_order=feature_order, sample_weights=weights
            )
            fold_dev_scores = model_scores(
                model, historical_dev, feature_order=feature_order
            )
            selected = select_threshold(
                fold_dev_scores,
                dev_targets,
                ambiguous=dev_ambiguous,
                max_ambiguous_auto=max_ambiguous_auto,
            )
            if selected is None:
                if variant == "BASE_REFIT":
                    raise ValueError(
                        f"STOP_RETRAIN: no safe dev threshold for {variant} fold {fold}"
                    )
                variant_failures[variant] = [f"fold_{fold}_no_safe_threshold"]
                fold_thresholds[variant] = {}
                variant_predictions = []
                break
            threshold, _ = selected
            fold_thresholds[variant][str(fold)] = threshold
            held_out = hard_oof[hard_oof["hard_fold"].astype(int).eq(fold)].copy()
            scores = model_scores(model, held_out, feature_order=feature_order)
            output = held_out[
                [
                    "query_id",
                    "sampling_stratum",
                    "current_label_origin",
                    "adjudication_label",
                    "acceptor_target",
                    "hard_fold",
                    "component_id",
                ]
            ].copy()
            output["variant"] = variant
            output["score"] = scores
            output["threshold"] = threshold
            output["auto"] = scores >= threshold
            output["prediction_is_group_oof"] = True
            variant_predictions.append(output)
        hard_predictions.extend(variant_predictions)
    hard_prediction_frame = pd.concat(hard_predictions, ignore_index=True)
    observed_counts = hard_prediction_frame.groupby("variant").size().to_dict()
    expected_variants = [
        variant for variant in VARIANT_WEIGHTS if variant not in variant_failures
    ]
    if hard_prediction_frame.groupby(["variant", "query_id"]).size().max() != 1 or (
        observed_counts != {variant: 94 for variant in expected_variants}
    ):
        raise ValueError("V4.8 hard OOF predictions are incomplete or duplicated")

    hard_metrics = _paired_hard_metrics(hard_prediction_frame)
    hard_segment_metrics = _hard_segment_metrics(hard_prediction_frame)
    paired_transitions = _paired_transitions(hard_prediction_frame)
    base_hard = hard_metrics["BASE_REFIT"]
    gates: dict[str, Any] = {}
    for variant in ("HARD_W1", "HARD_W2", "HARD_W4"):
        if variant not in variant_failures:
            gates[variant] = _variant_gate(
                frozen=frozen_dev_metrics,
                variant=complete_dev_metrics[variant],
                hard_metrics=hard_metrics[variant],
                base_hard_metrics=base_hard,
            )
            continue
        metrics = complete_dev_metrics[variant]
        threshold_found = bool(metrics.get("threshold_found"))
        precision_target = threshold_found and (
            1000 * int(metrics["correct_auto"])
            >= 998 * int(metrics["auto_count"])
        )
        not_below_frozen = threshold_found and (
            int(metrics["correct_auto"]) * int(frozen_dev_metrics["auto_count"])
            >= int(frozen_dev_metrics["correct_auto"]) * int(metrics["auto_count"])
        )
        coverage_ok = threshold_found and (
            float(metrics["coverage"]) + 1e-15
            >= float(frozen_dev_metrics["coverage"]) - 0.02
        )
        gates[variant] = {
            "admissible": False,
            "reason": variant_failures[variant],
            "checks": {
                "historical_precision_at_least_0_998": precision_target,
                "historical_precision_not_below_frozen": not_below_frozen,
                "historical_coverage_within_2pp": coverage_ok,
                "wrong_rejected_gain_at_least_4": False,
                "correct_acceptance_loss_at_most_5pp": False,
                "no_extra_ambiguous_auto": False,
            },
            "wrong_rejected_gain": None,
            "correct_acceptance_loss": None,
        }
    admissible = [variant for variant, gate in gates.items() if gate["admissible"]]
    winner: str | None = None
    if admissible:
        winner = sorted(
            admissible,
            key=lambda variant: (
                -hard_metrics[variant]["wrong_rejected"],
                -hard_metrics[variant]["correct_auto"],
                -complete_dev_metrics[variant]["coverage"],
                VARIANT_WEIGHTS[variant],
            ),
        )[0]
        development_verdict = "GO_RANDOM_OPEN_V48"
    else:
        historically_safe = any(
            gate["checks"]["historical_precision_at_least_0_998"]
            and gate["checks"]["historical_precision_not_below_frozen"]
            and gate["checks"]["historical_coverage_within_2pp"]
            for gate in gates.values()
        )
        development_verdict = (
            "PIVOT_FEATURES" if historically_safe else "STOP_RETRAIN"
        )

    locked_predictions: list[pd.DataFrame] = []
    for variant, model in [
        (FROZEN_VARIANT, frozen_model),
        *complete_models.items(),
    ]:
        threshold = (
            FROZEN_THRESHOLD
            if variant == FROZEN_VARIANT
            else complete_thresholds[variant]
        )
        scores = model_scores(model, hard_locked, feature_order=feature_order)
        output = hard_locked[
            [
                "query_id",
                "sampling_stratum",
                "current_label_origin",
                "adjudication_label",
                "acceptor_target",
                "component_id",
            ]
        ].copy()
        output["variant"] = variant
        output["score"] = scores
        output["threshold"] = threshold
        output["auto"] = scores >= threshold if np.isfinite(threshold) else False
        output["descriptive_only"] = True
        locked_predictions.append(output)
    locked_frame = pd.concat(locked_predictions, ignore_index=True)

    dev_frame = pd.concat(dev_predictions, ignore_index=True)
    curve_frame = pd.concat(curves, ignore_index=True)
    score_hashes = {
        "dev": hashlib.sha256(
            dev_frame.sort_values(["variant", "query_id"]).to_csv(index=False).encode()
        ).hexdigest(),
        "hard_oof": hashlib.sha256(
            hard_prediction_frame.sort_values(["variant", "query_id"])
            .to_csv(index=False)
            .encode()
        ).hexdigest(),
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "development_verdict": development_verdict,
        "winner": winner,
        "original_frozen_reproduction": original_metrics,
        "effective_dev_metrics": complete_dev_metrics,
        "hard_oof_metrics": hard_metrics,
        "hard_oof_segment_metrics": hard_segment_metrics,
        "hard_oof_paired_transitions": paired_transitions,
        "gates": gates,
        "fold_thresholds": fold_thresholds,
        "variant_failures": variant_failures,
        "score_hashes": score_hashes,
        "counts": {
            "historical_fit": len(historical_fit),
            "historical_dev": len(historical_dev),
            "hard_oof": len(hard_oof),
            "hard_dev_locked": len(hard_locked),
            "random_scored": 0,
        },
        "limitations": [
            "Threshold and historical metrics use the same development set.",
            "Random class counts were known in aggregate, but no random score or decision was read.",
            "This is an internal feasibility gate, not a 99.8% certification.",
        ],
    }
    identity = {
        "schema_version": SCHEMA_VERSION,
        "input_hashes": hashes,
        "runner_sha256": file_sha256(Path(__file__).resolve()),
        "partition_build_id": partition_manifest.get("build_id"),
        "variant_weights": VARIANT_WEIGHTS,
        "score_hashes": score_hashes,
        "development_verdict": development_verdict,
        "winner": winner,
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    target = Path(output_root).resolve() / build_id
    if target.exists():
        raise FileExistsError(f"Immutable V4.8 development artifact exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{build_id}.tmp-", dir=target.parent))
    try:
        output_frames = {
            "dev_predictions.parquet": dev_frame,
            "hard_oof_predictions.parquet": hard_prediction_frame,
            "hard_dev_locked_descriptive.parquet": locked_frame,
            "risk_coverage_curves.parquet": curve_frame,
        }
        for name, frame in output_frames.items():
            frame.to_parquet(staging / name, index=False)
        _json_dump(staging / "development_report.json", report)
        _json_dump(staging / "variant_coefficients.json", coefficients)
        if winner is not None:
            model_dir = staging / "winner"
            model_dir.mkdir()
            model_path = model_dir / "acceptor_model.joblib"
            joblib.dump(complete_models[winner], model_path)
            winner_metadata = {
                "schema_version": "sireto-v4.8-acceptor-winner-1",
                "variant": winner,
                "hard_weight": VARIANT_WEIGHTS[winner],
                "threshold": complete_thresholds[winner],
                "feature_order": feature_order,
                "partition_assignments_sha256": hashes["partition_assignments"],
                "random_scored": False,
                "test_opened": False,
            }
            _json_dump(model_dir / "metadata.json", winner_metadata)
            freeze = {
                "frozen_at": datetime.now(timezone.utc).isoformat(),
                "variant": winner,
                "threshold": complete_thresholds[winner],
                "model_sha256": file_sha256(model_path),
                "metadata_sha256": file_sha256(model_dir / "metadata.json"),
                "partition_assignments_sha256": hashes["partition_assignments"],
                "random_opened": False,
                "test_opened": False,
            }
            _json_dump(staging / "winner_freeze.json", freeze)
        output_hashes = {
            str(path.relative_to(staging)): file_sha256(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        manifest = {
            **identity,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "inputs": {name: str(path) for name, path in paths.items()},
            "outputs": output_hashes,
            "dependencies": {
                "python": sys.version,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "invariants": {
                "retrieval_trained": False,
                "ranker_trained": False,
                "features_added": False,
                "random_rows_scored": 0,
                "random_targets_read_from_partition": False,
                "test_opened": False,
            },
        }
        _json_dump(staging / "manifest.json", manifest)
        staging.rename(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def parse_args() -> argparse.Namespace:
    root = Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100")
    partition = root / "datasets/v4_8_acceptor_partitions/1c78764d5263afca"
    historical = root / "models/v4_1/f938abf6b8a87155"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("docs/v4_8_current_acceptor_feasibility_contract.md"),
    )
    parser.add_argument(
        "--partition-manifest", type=Path, default=partition / "manifest.json"
    )
    parser.add_argument(
        "--partition-assignments",
        type=Path,
        default=partition / "partition_assignments.parquet",
    )
    parser.add_argument(
        "--historical-scenes",
        type=Path,
        default=historical / "acceptor_scenes.parquet",
    )
    parser.add_argument(
        "--current-labels",
        type=Path,
        default=root
        / "audits/v4_7_current_adjudications/4cc5420fb5da0683/current_labels.parquet",
    )
    parser.add_argument(
        "--frozen-model",
        type=Path,
        default=historical / "acceptor/acceptor_model.joblib",
    )
    parser.add_argument(
        "--frozen-metadata",
        type=Path,
        default=historical / "acceptor/metadata.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "experiments/v4_8_acceptor_development",
    )
    parser.add_argument("--no-canonical-checks", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = run_development(
        contract_path=args.contract,
        partition_manifest_path=args.partition_manifest,
        partition_assignments_path=args.partition_assignments,
        historical_scenes_path=args.historical_scenes,
        current_labels_path=args.current_labels,
        frozen_model_path=args.frozen_model,
        frozen_metadata_path=args.frozen_metadata,
        output_root=args.output_root,
        enforce_canonical=not args.no_canonical_checks,
    )
    print(target)


if __name__ == "__main__":
    main()
