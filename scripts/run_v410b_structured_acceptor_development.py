#!/usr/bin/env python3
"""Run the preregistered V4.10b acceptor development experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import importlib.metadata
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402
from src.xgb_matcher.v410b_acceptor import (  # noqa: E402
    OrderPreservingSelectiveStandardScaler,
)


SCHEMA_VERSION = "sireto-v4.10b-structured-acceptor-development-1"
EXPECTED_PLAN_SHA256 = (
    "ab197ba0017b847d8f7d4c00721abaed5dd04fbcfb269ff9fadfb0d0afffd145"
)
DEFAULT_PLAN = Path("config/v4_10b_training_plan.json")
DEFAULT_EXECUTION_LOCK = Path("config/v4_10b_execution_lock.json")
SCALER_SOURCE = Path("src/xgb_matcher/v410b_acceptor.py")
ALLOWED_DATASET_FILES = (
    "historical_scenes.parquet",
    "consumed_hard_scenes.parquet",
    "descriptive_locked_scenes.parquet",
    "feature_catalog.json",
)
FROZEN_VARIANT = "BASE_FROZEN"
FOLDS = tuple(range(5))


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_bytes(_json_bytes(payload))


def _sha_order(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _assert_exact_columns(frame: pd.DataFrame, required: Sequence[str], name: str) -> None:
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"STOP_INPUT_INTEGRITY: {name} missing {sorted(missing)}")


def _assert_matrix(frame: pd.DataFrame, order: Sequence[str], name: str) -> None:
    _assert_exact_columns(frame, order, name)
    try:
        matrix = frame[list(order)].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"STOP_INPUT_INTEGRITY: {name} is non-numeric") from error
    if not np.isfinite(matrix).all():
        raise ValueError(f"STOP_INPUT_INTEGRITY: {name} is non-finite")


def wilson_interval(successes: int, total: int) -> list[float]:
    if total <= 0:
        return [0.0, 1.0]
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = proportion + z * z / (2.0 * total)
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / total
        + z * z / (4.0 * total * total)
    )
    return [(centre - margin) / denominator, (centre + margin) / denominator]


def decision_metrics(
    scores: np.ndarray,
    targets: np.ndarray,
    threshold: float,
    *,
    labels: np.ndarray | None = None,
) -> dict[str, Any]:
    score_values = np.asarray(scores, dtype=float)
    target_values = np.asarray(targets, dtype=int)
    if len(score_values) != len(target_values) or not np.isfinite(score_values).all():
        raise ValueError("STOP_INPUT_INTEGRITY: invalid scores or targets")
    auto = score_values >= float(threshold)
    auto_count = int(auto.sum())
    correct = int((auto & target_values.astype(bool)).sum())
    output = {
        "threshold": float(threshold),
        "row_count": int(len(score_values)),
        "auto_count": auto_count,
        "correct_auto": correct,
        "error_auto": auto_count - correct,
        "review_count": int(len(score_values) - auto_count),
        "precision": correct / auto_count if auto_count else None,
        "precision_wilson_95": wilson_interval(correct, auto_count),
        "coverage": auto_count / len(score_values) if len(score_values) else 0.0,
    }
    if labels is not None:
        label_values = np.asarray(labels).astype(str)
        output.update(
            {
                "correct_hard_auto": int(
                    (auto & (label_values == "TOP1_CORRECT")).sum()
                ),
                "wrong_hard_rejected": int(
                    ((~auto) & (label_values == "TOP1_WRONG")).sum()
                ),
                "ambiguous_hard_auto": int(
                    (auto & (label_values == "AMBIGUOUS")).sum()
                ),
            }
        )
    return output


def threshold_curve(scores: np.ndarray, targets: np.ndarray) -> pd.DataFrame:
    values = np.asarray(scores, dtype=float)
    candidates = np.unique(
        np.concatenate(
            [
                values,
                [
                    np.nextafter(values.max(), np.inf),
                    np.nextafter(values.min(), -np.inf),
                ],
            ]
        )
    )
    return pd.DataFrame(
        [decision_metrics(values, targets, value) for value in candidates]
    ).sort_values("threshold").reset_index(drop=True)


def select_threshold(
    scores: np.ndarray,
    targets: np.ndarray,
    *,
    minimum_auto_count: int = 100,
) -> tuple[float, dict[str, Any], pd.DataFrame] | None:
    curve = threshold_curve(scores, targets)
    eligible = curve[
        curve["auto_count"].ge(int(minimum_auto_count))
        & (
            1000 * curve["correct_auto"].astype(int)
            >= 998 * curve["auto_count"].astype(int)
        )
    ].copy()
    if eligible.empty:
        return None
    eligible["_precision"] = eligible["correct_auto"] / eligible["auto_count"]
    winner = eligible.sort_values(
        ["auto_count", "_precision", "threshold"],
        ascending=[False, False, False],
        kind="mergesort",
    ).iloc[0]
    threshold = float(winner["threshold"])
    return (
        threshold,
        decision_metrics(scores, targets, threshold),
        curve,
    )


def _model_scores(model: Any, frame: pd.DataFrame, order: Sequence[str]) -> np.ndarray:
    if frame.empty:
        return np.array([], dtype=float)
    return np.asarray(
        model.predict_proba(frame[list(order)].to_numpy(dtype=np.float64))[:, 1],
        dtype=float,
    )


def _logit_model(
    plan: Mapping[str, Any],
    *,
    feature_order: Sequence[str],
    continuous_features: Sequence[str],
    scale_all: bool,
) -> Pipeline:
    config = plan["models"][
        "current80_logit" if scale_all else "structured_logit"
    ]
    if scale_all:
        preprocessing: Any = StandardScaler()
    else:
        continuous_indices = [
            index
            for index, name in enumerate(feature_order)
            if name in set(continuous_features)
        ]
        preprocessing = OrderPreservingSelectiveStandardScaler(
            continuous_indices
        )
    return Pipeline(
        [
            ("preprocessing", preprocessing),
            (
                "model",
                LogisticRegression(
                    C=float(config["C"]),
                    class_weight=str(config["class_weight"]),
                    max_iter=int(config["max_iter"]),
                    solver=str(config["solver"]),
                    random_state=int(config["random_state"]),
                ),
            ),
        ]
    )


def _xgb_model(plan: Mapping[str, Any]) -> xgb.XGBClassifier:
    config = dict(plan["models"]["structured_xgb"])
    for key in ("early_stopping", "class_balance"):
        config.pop(key, None)
    return xgb.XGBClassifier(**config)


def _fit_model(
    family: str,
    train: pd.DataFrame,
    *,
    feature_order: Sequence[str],
    continuous_features: Sequence[str],
    plan: Mapping[str, Any],
    base_weights: np.ndarray,
) -> Any:
    targets = train["acceptor_target"].astype(int).to_numpy()
    if train.empty or set(np.unique(targets)) != {0, 1}:
        raise ValueError("STOP_INPUT_INTEGRITY: fit requires both target classes")
    if family == "current80_logit":
        model = _logit_model(
            plan,
            feature_order=feature_order,
            continuous_features=feature_order,
            scale_all=True,
        )
        model.fit(
            train[list(feature_order)].to_numpy(dtype=float),
            targets,
            model__sample_weight=base_weights,
        )
        return model
    if family == "structured_logit":
        model = _logit_model(
            plan,
            feature_order=feature_order,
            continuous_features=continuous_features,
            scale_all=False,
        )
        model.fit(
            train[list(feature_order)].to_numpy(dtype=float),
            targets,
            model__sample_weight=base_weights,
        )
        return model
    if family != "structured_xgb":
        raise ValueError(f"Unsupported V4.10 family: {family}")
    counts = np.bincount(targets, minlength=2).astype(float)
    if (counts == 0).any():
        raise ValueError("STOP_INPUT_INTEGRITY: XGB class balance is undefined")
    class_factors = len(targets) / (2.0 * counts)
    weights = base_weights * class_factors[targets]
    model = _xgb_model(plan)
    model.fit(train[list(feature_order)].to_numpy(dtype=float), targets, sample_weight=weights)
    return model


def _training_frame(
    historical_fit: pd.DataFrame,
    hard: pd.DataFrame,
    *,
    hard_weight: float,
    held_out_fold: int | None,
) -> tuple[pd.DataFrame, np.ndarray]:
    historical = historical_fit
    current = hard
    if held_out_fold is not None:
        historical = historical[
            ~(
                historical["role"].eq("historical_hard_support")
                & historical["hard_fold"].astype("Int64").eq(held_out_fold)
            )
        ]
        current = current[
            current["hard_fold"].astype(int).ne(held_out_fold)
        ]
    historical = historical.copy()
    current = current.copy()
    historical["_sample_weight"] = 1.0
    current["_sample_weight"] = float(hard_weight)
    combined = (
        pd.concat([historical, current], ignore_index=True)
        .sort_values("query_id", kind="mergesort")
        .reset_index(drop=True)
    )
    weights = combined.pop("_sample_weight").to_numpy(dtype=float)
    return combined, weights


def _load_plan(
    path: Path,
    *,
    enforce_canonical: bool,
) -> tuple[dict[str, Any], dict[str, Path], dict[str, str]]:
    plan_path = Path(path).resolve()
    observed_plan_hash = file_sha256(plan_path)
    if enforce_canonical and observed_plan_hash != EXPECTED_PLAN_SHA256:
        raise ValueError("STOP_INPUT_INTEGRITY: V4.10 training plan hash mismatch")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != "sireto-v4.10b-structured-acceptor-training-plan-1":
        raise ValueError("STOP_INPUT_INTEGRITY: unsupported V4.10 plan")
    dataset = plan["dataset"]
    root = Path(dataset["root"]).resolve()
    paths = {
        "plan": plan_path,
        "dataset_manifest": root / "manifest.json",
        **{name: root / name for name in ALLOWED_DATASET_FILES},
        "frozen_model": Path(plan["frozen_baseline"]["model_path"]).resolve(),
        "frozen_metadata": Path(plan["frozen_baseline"]["metadata_path"]).resolve(),
    }
    hashes = {name: file_sha256(value) for name, value in paths.items()}
    expected = {
        "plan": observed_plan_hash,
        "dataset_manifest": dataset["manifest"]["sha256"],
        **dataset["allowed_inputs"],
        "frozen_model": plan["frozen_baseline"]["model_sha256"],
        "frozen_metadata": plan["frozen_baseline"]["metadata_sha256"],
    }
    mismatches = {
        name: {"expected": expected[name], "observed": hashes[name]}
        for name in expected
        if hashes[name] != expected[name]
    }
    if mismatches:
        raise ValueError(f"STOP_INPUT_INTEGRITY: input hash mismatch {mismatches}")
    return plan, paths, hashes


def _runtime_versions() -> dict[str, str]:
    return {
        package: importlib.metadata.version(package)
        for package in (
            "numpy",
            "pandas",
            "pyarrow",
            "scikit-learn",
            "xgboost",
            "joblib",
        )
    }


def _validate_runtime(plan: Mapping[str, Any]) -> dict[str, str]:
    observed = _runtime_versions()
    expected = {
        str(key): str(value)
        for key, value in plan["runtime"]["required_versions"].items()
    }
    if observed != expected:
        raise ValueError(
            f"STOP_RUNTIME_INTEGRITY: versions {observed} != {expected}"
        )
    return observed


def _git_blob_sha256(commit: str, relative_path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=Path(__file__).resolve().parent.parent,
        check=True,
        capture_output=True,
    )
    return hashlib.sha256(result.stdout).hexdigest()


def validate_execution_lock(
    plan: Mapping[str, Any],
    paths: Mapping[str, Path],
    hashes: Mapping[str, str],
    execution_lock_path: Path,
    *,
    verify_git_commit: bool = True,
) -> tuple[dict[str, Any], str]:
    """Validate an externally created immutable lock before semantic reads."""

    lock_path = Path(execution_lock_path).resolve()
    raw = lock_path.read_bytes()
    lock_hash = hashlib.sha256(raw).hexdigest()
    lock = json.loads(raw)
    policy = plan["runtime"]["execution_lock"]
    exact_fields = [str(value) for value in policy["exact_fields"]]
    if set(lock) != set(exact_fields):
        raise ValueError("STOP_EXECUTION_LOCK: fields differ from preregistration")
    expected = {
        "schema_version": policy["schema_version"],
        "training_plan_sha256": hashes["plan"],
        "runner_sha256": file_sha256(Path(__file__).resolve()),
        "dataset_manifest_sha256": hashes["dataset_manifest"],
        "feature_catalog_sha256": hashes["feature_catalog.json"],
    }
    for key, value in expected.items():
        if str(lock.get(key)) != str(value):
            raise ValueError(f"STOP_EXECUTION_LOCK: {key} mismatch")
    commit = str(lock.get("runner_commit") or "")
    if not commit:
        raise ValueError("STOP_EXECUTION_LOCK: runner_commit missing")
    if verify_git_commit:
        try:
            committed_hash = _git_blob_sha256(
                commit,
                "scripts/run_v410b_structured_acceptor_development.py",
            )
        except subprocess.CalledProcessError as error:
            raise ValueError("STOP_EXECUTION_LOCK: runner commit unavailable") from error
        if committed_hash != expected["runner_sha256"]:
            raise ValueError("STOP_EXECUTION_LOCK: commit does not pin runner bytes")
        scaler_path = Path(__file__).resolve().parent.parent / SCALER_SOURCE
        try:
            committed_scaler_hash = _git_blob_sha256(commit, str(SCALER_SOURCE))
        except subprocess.CalledProcessError as error:
            raise ValueError("STOP_EXECUTION_LOCK: scaler source commit unavailable") from error
        if committed_scaler_hash != file_sha256(scaler_path):
            raise ValueError("STOP_EXECUTION_LOCK: commit does not pin scaler source bytes")
    return lock, lock_hash


def _load_frames(
    plan: Mapping[str, Any],
    paths: Mapping[str, Path],
    *,
    enforce_canonical: bool = True,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    list[str],
    list[str],
    list[str],
    list[str],
]:
    manifest = json.loads(paths["dataset_manifest"].read_text(encoding="utf-8"))
    catalog = json.loads(paths["feature_catalog.json"].read_text(encoding="utf-8"))
    current80 = [str(value) for value in catalog["current80_feature_order"]]
    structured = [str(value) for value in catalog["structured_feature_order"]]
    if (
        len(current80) != int(plan["dataset"]["feature_orders"]["current80_count"])
        or _sha_order(current80)
        != plan["dataset"]["feature_orders"]["current80_sha256"]
        or len(structured)
        != int(plan["dataset"]["feature_orders"]["structured_count"])
        or _sha_order(structured)
        != plan["dataset"]["feature_orders"]["structured_sha256"]
    ):
        raise ValueError("STOP_INPUT_INTEGRITY: V4.10 feature orders changed")
    specifications = {str(item["name"]): item for item in catalog["features"]}
    if set(structured) - set(specifications):
        raise ValueError("STOP_INPUT_INTEGRITY: structured feature spec missing")
    if any(
        not specifications[name].get("model_allowed")
        or not specifications[name].get("structured_allowed")
        or specifications[name].get("alias_of") is not None
        for name in structured
    ):
        raise ValueError("STOP_INPUT_INTEGRITY: forbidden feature entered structured order")
    scaled = [str(value) for value in catalog["structured_scaled_feature_order"]]
    unscaled = [
        str(value) for value in catalog["structured_unscaled_feature_order"]
    ]
    feature_plan = plan["dataset"]["feature_orders"]
    if (
        len(scaled) != int(feature_plan["structured_scaled_count"])
        or _sha_order(scaled) != feature_plan["structured_scaled_sha256"]
        or len(unscaled) != int(feature_plan["structured_unscaled_count"])
        or _sha_order(unscaled) != feature_plan["structured_unscaled_sha256"]
        or set(scaled) & set(unscaled)
        or [name for name in structured if name in set(scaled)] != scaled
        or [name for name in structured if name in set(unscaled)] != unscaled
        or set(structured) != set(scaled) | set(unscaled)
    ):
        raise ValueError("STOP_INPUT_INTEGRITY: scaled/unscaled orders changed")
    if any(
        specifications[name]["kind"] not in {"continuous", "count"}
        for name in scaled
    ) or any(specifications[name]["kind"] != "binary" for name in unscaled):
        raise ValueError("STOP_INPUT_INTEGRITY: feature kind/scaler policy changed")
    metadata_columns = [
        "query_id",
        "split",
        "training_eligible",
        "role",
        "hard_fold",
        "acceptor_target",
        "adjudication_label",
        "hard_component_id",
        "ranker_prediction_is_out_of_sample",
        "prediction_origin",
    ]
    train_columns = list(dict.fromkeys([*metadata_columns, *current80, *structured]))
    dev_columns = train_columns
    baseline_only_columns = list(dict.fromkeys([*metadata_columns, *current80]))
    historical_path = paths["historical_scenes.parquet"]
    historical_fit = pd.read_parquet(
        historical_path,
        columns=train_columns,
        filters=[
            ("split", "=", "fit"),
            ("training_eligible", "=", True),
            ("role", "in", ["historical_fit", "historical_hard_support"]),
        ],
    )
    effective_dev = pd.read_parquet(
        historical_path,
        columns=dev_columns,
        filters=[
            ("split", "=", "dev"),
            ("training_eligible", "=", True),
            ("role", "=", "historical_dev"),
        ],
    )
    excluded_dev = pd.read_parquet(
        historical_path,
        columns=baseline_only_columns,
        filters=[
            ("split", "=", "dev"),
            ("training_eligible", "=", True),
            ("role", "=", "historical_random_excluded"),
        ],
    )
    hard = pd.read_parquet(paths["consumed_hard_scenes.parquet"])
    # The locked parquet is deliberately never opened after its byte hash.
    locked_rows = int(
        manifest["volumes"]["descriptive_locked"]
    )
    expected_rows = plan["dataset"]["expected_rows"]
    if (
        int(manifest["volumes"]["historical_scenes"])
        != int(expected_rows["historical_scenes"])
        or len(hard) != int(expected_rows["development_consumed"])
        or locked_rows != int(expected_rows["descriptive_locked"])
    ):
        raise ValueError("STOP_INPUT_INTEGRITY: V4.10 row counts changed")
    if enforce_canonical and (
        len(historical_fit) != 5_501
        or len(effective_dev) != 1_452
        or len(excluded_dev) != 4
    ):
        raise ValueError("STOP_INPUT_INTEGRITY: predicate-read populations changed")
    expected_roles = {
        str(key): int(value)
        for key, value in plan["dataset"]["expected_historical_roles"].items()
    }
    if manifest["population_audit"]["historical"]["role_counts"] != expected_roles:
        raise ValueError("STOP_INPUT_INTEGRITY: historical role counts changed")
    for name, frame in (
        ("historical_fit", historical_fit),
        ("effective_dev", effective_dev),
        ("hard", hard),
    ):
        _assert_matrix(frame, current80, f"{name}.current80")
        _assert_matrix(frame, structured, f"{name}.structured")
        if not frame["ranker_prediction_is_out_of_sample"].astype(bool).all():
            raise ValueError(f"STOP_INPUT_INTEGRITY: {name} lacks ranker OOS proof")
        if frame["query_id"].astype(str).duplicated().any():
            raise ValueError(f"STOP_INPUT_INTEGRITY: {name} query_id duplicated")
    hard_counts = hard["hard_fold"].astype(int).value_counts().sort_index().to_dict()
    expected_folds = {
        int(key): int(value)
        for key, value in plan["dataset"]["expected_hard_folds"].items()
    }
    if hard_counts != expected_folds:
        raise ValueError("STOP_INPUT_INTEGRITY: hard folds changed")
    if hard["adjudication_label"].value_counts().to_dict() != plan["dataset"][
        "expected_hard_labels"
    ]:
        raise ValueError("STOP_INPUT_INTEGRITY: hard labels changed")
    _assert_matrix(excluded_dev, current80, "excluded_dev.current80")
    if (
        excluded_dev["query_id"].astype(str).duplicated().any()
        or not excluded_dev["ranker_prediction_is_out_of_sample"].astype(bool).all()
    ):
        raise ValueError("STOP_INPUT_INTEGRITY: excluded baseline dev invalid")
    populations = {
        "historical_fit": set(historical_fit["query_id"].astype(str)),
        "effective_dev": set(effective_dev["query_id"].astype(str)),
        "excluded_dev": set(excluded_dev["query_id"].astype(str)),
        "hard": set(hard["query_id"].astype(str)),
    }
    population_names = list(populations)
    for left_index, left_name in enumerate(population_names):
        for right_name in population_names[left_index + 1 :]:
            if populations[left_name] & populations[right_name]:
                raise ValueError(
                    f"STOP_LEAKAGE: query overlap {left_name}/{right_name}"
                )
    original_dev = pd.concat(
        [
            effective_dev[baseline_only_columns],
            excluded_dev[baseline_only_columns],
        ],
        ignore_index=True,
    )
    if set(hard["role"].astype(str)) != {"hard_oof"}:
        raise ValueError("STOP_INPUT_INTEGRITY: consumed hard role changed")
    if [int(value) for value in plan["cross_validation"]["folds"]] != list(FOLDS):
        raise ValueError("STOP_INPUT_INTEGRITY: cross-validation folds changed")
    return (
        historical_fit.sort_values("query_id"),
        original_dev.sort_values("query_id"),
        effective_dev.sort_values("query_id"),
        hard.sort_values("query_id"),
        manifest,
        current80,
        structured,
        scaled,
        unscaled,
    )


def _variant_gate(
    historical: Mapping[str, Any],
    baseline: Mapping[str, Any],
    hard: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    gate = plan["development_gate"]
    correct_auto = int(historical["correct_auto"])
    auto_count = int(historical["auto_count"])
    checks = {
        "wrong_hard_rejected": int(hard["wrong_hard_rejected"])
        >= int(gate["minimum_wrong_hard_rejected"]),
        "ambiguous_hard_auto": int(hard["ambiguous_hard_auto"])
        <= int(gate["maximum_ambiguous_hard_auto"]),
        "correct_hard_auto": int(hard["correct_hard_auto"])
        >= int(gate["minimum_correct_hard_auto"]),
        "historical_precision": 1000 * correct_auto >= 998 * auto_count,
        "historical_precision_noninferior": (
            correct_auto * int(baseline["auto_count"])
            >= auto_count * int(baseline["correct_auto"])
        ),
        "historical_coverage": (
            int(baseline["auto_count"]) - auto_count <= 29
        ),
    }
    return {"admissible": all(checks.values()), "checks": checks}


def _assert_variants(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    variants = [dict(item) for item in plan["variants"]]
    identifiers = [str(item["id"]) for item in variants]
    if (
        len(variants) != 10
        or len(set(identifiers)) != 10
        or identifiers[0] != FROZEN_VARIANT
        or variants[0]["role"] != "comparator"
        or sum(item["role"] == "control" for item in variants) != 3
        or sum(item["role"] == "promotion_candidate" for item in variants) != 6
    ):
        raise ValueError("STOP_INPUT_INTEGRITY: variant registry changed")
    for item in variants[1:]:
        if item["role"] == "control":
            valid = (
                item["family"] == "current80_logit"
                and item["feature_order"] == "current80"
            )
        else:
            valid = (
                item["family"] in {"structured_logit", "structured_xgb"}
                and item["feature_order"] == "structured"
            )
        if not valid or float(item["hard_weight"]) not in {1.0, 2.0, 4.0}:
            raise ValueError("STOP_INPUT_INTEGRITY: variant contract changed")
    return variants


def _hard_decision_metrics(predictions: pd.DataFrame) -> dict[str, Any]:
    auto = predictions["auto"].astype(bool).to_numpy()
    labels = predictions["adjudication_label"].astype(str).to_numpy()
    targets = predictions["acceptor_target"].astype(int).to_numpy()
    correct = int((auto & targets.astype(bool)).sum())
    auto_count = int(auto.sum())
    return {
        "row_count": int(len(predictions)),
        "auto_count": auto_count,
        "correct_auto": correct,
        "error_auto": auto_count - correct,
        "review_count": int(len(predictions) - auto_count),
        "precision": correct / auto_count if auto_count else None,
        "coverage": auto_count / len(predictions) if len(predictions) else 0.0,
        "correct_hard_auto": int((auto & (labels == "TOP1_CORRECT")).sum()),
        "wrong_hard_rejected": int(((~auto) & (labels == "TOP1_WRONG")).sum()),
        "ambiguous_hard_auto": int((auto & (labels == "AMBIGUOUS")).sum()),
    }


def _model_diagnostic(
    model: Any,
    family: str,
    feature_order: Sequence[str],
) -> dict[str, Any]:
    if family in {"current80_logit", "structured_logit"}:
        estimator = model.named_steps["model"]
        coefficients = np.asarray(estimator.coef_[0], dtype=float)
        if len(coefficients) != len(feature_order):
            raise ValueError("STOP_REPRODUCTION: coefficient width changed")
        return {
            "kind": "logistic_coefficients",
            "feature_count": len(feature_order),
            "feature_order_sha256": _sha_order(feature_order),
            "intercept": float(np.asarray(estimator.intercept_)[0]),
            "by_feature": {
                name: float(value)
                for name, value in zip(feature_order, coefficients, strict=True)
            },
        }
    importances = np.asarray(model.feature_importances_, dtype=float)
    if len(importances) != len(feature_order):
        raise ValueError("STOP_REPRODUCTION: importance width changed")
    return {
        "kind": "xgb_feature_importances",
        "feature_count": len(feature_order),
        "feature_order_sha256": _sha_order(feature_order),
        "importance_type": str(model.importance_type),
        "by_feature": {
            name: float(value)
            for name, value in zip(feature_order, importances, strict=True)
        },
    }


def _development_verdict(
    complete_variants: Sequence[str],
    eligible_variants: Sequence[str],
    plan: Mapping[str, Any],
) -> str:
    if not complete_variants:
        return "STOP_STRUCTURED_ACCEPTOR"
    if eligible_variants:
        return str(plan["selection"]["success_verdict"])
    return "PIVOT_STRUCTURED_FEATURES"


def _baseline_variant_result(
    original_metrics: Mapping[str, Any],
    effective_metrics: Mapping[str, Any],
    hard_diagnostic: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "family": "frozen",
        "role": "comparator",
        "feature_order": "current80",
        "hard_weight": 0,
        "complete_safe_thresholds": False,
        "passes_development_gate": False,
        "promotion_eligible": False,
        "reason": "COMPARATOR_ONLY",
        "original_dev_metrics": dict(original_metrics),
        "historical_metrics": dict(effective_metrics),
        "hard_diagnostic": {
            "prediction_is_group_oof": False,
            "used_for_gate_or_selection": False,
            "metrics": dict(hard_diagnostic),
        },
    }


def _assert_variant_results_complete(
    variant_results: Mapping[str, Any],
    variants: Sequence[Mapping[str, Any]],
) -> None:
    expected = {str(item["id"]) for item in variants}
    if len(variant_results) != len(variants) or set(variant_results) != expected:
        raise ValueError("STOP_OUTPUT_INTEGRITY: variant results are incomplete")


def _population_usage(
    plan: Mapping[str, Any],
    *,
    historical_fit_rows: int,
    effective_dev_rows: int,
    hard_rows: int,
    excluded_dev_rows: int,
) -> dict[str, Any]:
    return {
        "historical_fit": int(historical_fit_rows),
        "historical_dev_effective": int(effective_dev_rows),
        "hard_consumed": int(hard_rows),
        "historical_excluded_dev": {
            "read": int(excluded_dev_rows),
            "baseline_scored": int(excluded_dev_rows),
            "trained_variant_scored": 0,
            "threshold_used": 0,
            "gate_used": 0,
        },
        "historical_random_excluded_other_44": {
            "materialized": 0,
            "read": 0,
            "scored": 0,
        },
        "random_v48": {"read": 0, "scored": 0},
        "descriptive_locked": {
            "byte_hash_checked": int(
                plan["dataset"]["expected_rows"]["descriptive_locked"]
            ),
            "read": 0,
            "scored": 0,
        },
        "fresh_dev": {"read": 0, "scored": 0},
        "final_test": {"read": 0, "scored": 0},
    }


def _fit_twice(
    family: str,
    train: pd.DataFrame,
    *,
    feature_order: Sequence[str],
    continuous_features: Sequence[str],
    plan: Mapping[str, Any],
    base_weights: np.ndarray,
    dev: pd.DataFrame,
    held_hard: pd.DataFrame | None,
) -> tuple[
    Any,
    np.ndarray,
    np.ndarray | None,
    float | None,
    dict[str, Any] | None,
    pd.DataFrame,
    dict[str, Any],
]:
    """Fit twice and prove score, threshold, and decision reproducibility."""

    tolerance = float(plan["determinism"]["repeat_fit_score_absolute_tolerance"])
    outcomes: list[
        tuple[
            Any,
            np.ndarray,
            np.ndarray | None,
            float | None,
            dict[str, Any] | None,
            pd.DataFrame,
        ]
    ] = []
    for _ in range(2):
        model = _fit_model(
            family,
            train,
            feature_order=feature_order,
            continuous_features=continuous_features,
            plan=plan,
            base_weights=base_weights,
        )
        dev_scores = _model_scores(model, dev, feature_order)
        selected = select_threshold(
            dev_scores,
            dev["acceptor_target"].astype(int).to_numpy(),
            minimum_auto_count=int(plan["threshold_selection"]["minimum_auto_count"]),
        )
        if selected is None:
            threshold = None
            metrics = None
            curve = threshold_curve(
                dev_scores,
                dev["acceptor_target"].astype(int).to_numpy(),
            )
        else:
            threshold, metrics, curve = selected
        hard_scores = (
            _model_scores(model, held_hard, feature_order)
            if held_hard is not None
            else None
        )
        outcomes.append((model, dev_scores, hard_scores, threshold, metrics, curve))
    first, second = outcomes
    dev_difference = float(np.max(np.abs(first[1] - second[1])))
    hard_difference = (
        float(np.max(np.abs(first[2] - second[2])))
        if first[2] is not None and len(first[2])
        else 0.0
    )
    same_threshold = first[3] == second[3]
    same_dev_decisions = (
        True
        if first[3] is None
        else np.array_equal(first[1] >= first[3], second[1] >= second[3])
    )
    same_hard_decisions = (
        True
        if first[2] is None or first[3] is None
        else np.array_equal(first[2] >= first[3], second[2] >= second[3])
    )
    diagnostic = {
        "dev_max_score_absolute_difference": dev_difference,
        "held_hard_max_score_absolute_difference": hard_difference,
        "threshold_identical": same_threshold,
        "dev_auto_decisions_identical": same_dev_decisions,
        "held_hard_auto_decisions_identical": same_hard_decisions,
        "tolerance": tolerance,
        "passed": (
            dev_difference <= tolerance
            and hard_difference <= tolerance
            and same_threshold
            and same_dev_decisions
            and same_hard_decisions
        ),
    }
    if not diagnostic["passed"]:
        raise ValueError("STOP_REPRODUCTION: repeated fit differs")
    return (*first, diagnostic)


def run_development(
    training_plan_path: Path,
    *,
    execution_lock_path: Path = DEFAULT_EXECUTION_LOCK,
    output_root: Path | None = None,
    enforce_canonical: bool = True,
) -> Path:
    plan, paths, hashes = _load_plan(
        training_plan_path, enforce_canonical=enforce_canonical
    )
    runtime_versions = _validate_runtime(plan)
    execution_lock, execution_lock_hash = validate_execution_lock(
        plan,
        paths,
        hashes,
        execution_lock_path,
        verify_git_commit=enforce_canonical,
    )
    (
        historical_fit,
        original_dev,
        effective_dev,
        hard,
        dataset_manifest,
        current80,
        structured,
        scaled,
        unscaled,
    ) = _load_frames(plan, paths, enforce_canonical=enforce_canonical)
    variants_all = _assert_variants(plan)
    frozen_metadata = json.loads(paths["frozen_metadata"].read_text(encoding="utf-8"))
    if [str(value) for value in frozen_metadata["feature_order"]] != current80:
        raise ValueError("STOP_REPRODUCTION: frozen feature order changed")
    frozen = joblib.load(paths["frozen_model"])
    frozen_threshold = float(plan["frozen_baseline"]["threshold"])
    original_scores = _model_scores(frozen, original_dev, current80)
    effective_scores = _model_scores(frozen, effective_dev, current80)
    original_metrics = decision_metrics(
        original_scores,
        original_dev["acceptor_target"].astype(int).to_numpy(),
        frozen_threshold,
    )
    effective_baseline = decision_metrics(
        effective_scores,
        effective_dev["acceptor_target"].astype(int).to_numpy(),
        frozen_threshold,
    )
    if enforce_canonical:
        for key, expected in plan["frozen_baseline"]["original_dev_expected"].items():
            if original_metrics[key] != expected:
                raise ValueError(f"STOP_REPRODUCTION: original BASE_FROZEN {key}")
        for key, expected in plan["frozen_baseline"]["effective_dev_expected"].items():
            if effective_baseline[key] != expected:
                raise ValueError(f"STOP_REPRODUCTION: effective BASE_FROZEN {key}")

    variants = [item for item in variants_all if item["id"] != FROZEN_VARIANT]
    variant_results: dict[str, Any] = {}
    fitted_model_diagnostics: dict[str, Any] = {}
    final_models: dict[str, Any] = {}
    final_thresholds: dict[str, float] = {}
    dev_outputs: list[pd.DataFrame] = []
    hard_outputs: list[pd.DataFrame] = []
    curves: list[pd.DataFrame] = []
    determinism: dict[str, Any] = {"logical_fit_count": 0, "physical_fit_count": 0, "variants": {}}
    base_dev = effective_dev[["query_id", "acceptor_target", "adjudication_label"]].copy()
    base_dev["variant"] = FROZEN_VARIANT
    base_dev["score"] = effective_scores
    base_dev["threshold"] = frozen_threshold
    base_dev["auto"] = effective_scores >= frozen_threshold
    dev_outputs.append(base_dev)
    base_hard_scores = _model_scores(frozen, hard, current80)
    base_hard = hard[
        ["query_id", "acceptor_target", "adjudication_label", "hard_fold", "hard_component_id"]
    ].copy()
    base_hard["variant"] = FROZEN_VARIANT
    base_hard["score"] = base_hard_scores
    base_hard["threshold"] = frozen_threshold
    base_hard["auto"] = base_hard_scores >= frozen_threshold
    base_hard["prediction_is_group_oof"] = False
    base_hard["prediction_origin"] = plan["baseline_hard_diagnostic"]["prediction_origin"]
    hard_outputs.append(base_hard)
    baseline_hard_diagnostic = _hard_decision_metrics(base_hard)
    variant_results[FROZEN_VARIANT] = _baseline_variant_result(
        original_metrics,
        effective_baseline,
        baseline_hard_diagnostic,
    )

    for variant in variants:
        variant_id = str(variant["id"])
        family = str(variant["family"])
        hard_weight = float(variant["hard_weight"])
        order = current80 if variant["feature_order"] == "current80" else structured
        continuous_order = order if family == "current80_logit" else scaled
        fold_predictions: list[pd.DataFrame] = []
        fold_thresholds: dict[str, float] = {}
        variant_determinism: dict[str, Any] = {"folds": {}}
        failed_reason: str | None = None
        for fold in FOLDS:
            train, weights = _training_frame(
                historical_fit,
                hard,
                hard_weight=hard_weight,
                held_out_fold=fold,
            )
            expected_rows = int(plan["cross_validation"]["expected_fold_train_rows"][str(fold)])
            if enforce_canonical and len(train) != expected_rows:
                raise ValueError(f"STOP_LEAKAGE: fold {fold} train rows changed")
            held_out = hard[hard["hard_fold"].astype(int).eq(fold)].copy()
            (
                model,
                dev_scores,
                scores,
                threshold,
                _,
                curve,
                repeat_diagnostic,
            ) = _fit_twice(
                family,
                train,
                feature_order=order,
                continuous_features=continuous_order,
                plan=plan,
                base_weights=weights,
                dev=effective_dev,
                held_hard=held_out,
            )
            determinism["logical_fit_count"] += 1
            determinism["physical_fit_count"] += 2
            variant_determinism["folds"][str(fold)] = repeat_diagnostic
            curve["variant"] = variant_id
            curve["model_scope"] = "fold"
            curve["fold"] = fold
            curves.append(curve)
            output = held_out[
                [
                    "query_id",
                    "acceptor_target",
                    "adjudication_label",
                    "hard_fold",
                    "hard_component_id",
                ]
            ].copy()
            output["variant"] = variant_id
            output["score"] = scores
            output["threshold"] = np.nan if threshold is None else threshold
            output["auto"] = (
                pd.array([pd.NA] * len(output), dtype="boolean")
                if threshold is None
                else pd.array(scores >= threshold, dtype="boolean")
            )
            output["prediction_is_group_oof"] = True
            output["prediction_origin"] = "v410b_group_oof"
            fold_predictions.append(output)
            if threshold is None:
                failed_reason = failed_reason or (
                    f"DISQUALIFIED_NO_SAFE_THRESHOLD_FOLD_{fold}"
                )
            else:
                fold_thresholds[str(fold)] = threshold
        hard_prediction = (
            pd.concat(fold_predictions, ignore_index=True)
            if fold_predictions
            else pd.DataFrame()
        )
        if (
            len(hard_prediction) != len(hard)
            or hard_prediction["query_id"].duplicated().any()
        ):
            raise ValueError("STOP_LEAKAGE: incomplete hard group-OOF prediction")
        hard_outputs.append(hard_prediction)
        train, weights = _training_frame(
            historical_fit,
            hard,
            hard_weight=hard_weight,
            held_out_fold=None,
        )
        if enforce_canonical and len(train) != 5_595:
            raise ValueError("STOP_LEAKAGE: full-model train rows changed")
        (
            model,
            scores,
            _,
            threshold,
            historical_metrics,
            curve,
            repeat_diagnostic,
        ) = _fit_twice(
            family,
            train,
            feature_order=order,
            continuous_features=continuous_order,
            plan=plan,
            base_weights=weights,
            dev=effective_dev,
            held_hard=None,
        )
        fitted_model_diagnostics[variant_id] = _model_diagnostic(
            model, family, order
        )
        determinism["logical_fit_count"] += 1
        determinism["physical_fit_count"] += 2
        variant_determinism["full"] = repeat_diagnostic
        determinism["variants"][variant_id] = variant_determinism
        curve["variant"] = variant_id
        curve["model_scope"] = "full"
        curve["fold"] = pd.NA
        curves.append(curve)
        if threshold is None:
            failed_reason = failed_reason or "DISQUALIFIED_NO_SAFE_THRESHOLD_FINAL"
        output = effective_dev[
            ["query_id", "acceptor_target", "adjudication_label"]
        ].copy()
        output["variant"] = variant_id
        output["score"] = scores
        output["threshold"] = np.nan if threshold is None else threshold
        output["auto"] = (
            pd.array([pd.NA] * len(output), dtype="boolean")
            if threshold is None
            else pd.array(scores >= threshold, dtype="boolean")
        )
        dev_outputs.append(output)
        if failed_reason is not None:
            variant_results[variant_id] = {
                "complete_safe_thresholds": False,
                "passes_development_gate": False,
                "promotion_eligible": False,
                "reason": failed_reason,
                "role": str(variant["role"]),
                "fold_thresholds": fold_thresholds,
            }
            continue
        assert historical_metrics is not None
        hard_metrics = _hard_decision_metrics(hard_prediction)
        gate = _variant_gate(
            historical_metrics, effective_baseline, hard_metrics, plan
        )
        variant_results[variant_id] = {
            "complete_safe_thresholds": True,
            "passes_development_gate": bool(gate["admissible"]),
            "promotion_eligible": bool(
                gate["admissible"] and variant["role"] == "promotion_candidate"
            ),
            "family": family,
            "hard_weight": hard_weight,
            "feature_order": str(variant["feature_order"]),
            "role": str(variant["role"]),
            "historical_metrics": historical_metrics,
            "hard_oof_metrics": hard_metrics,
            "fold_thresholds": fold_thresholds,
            "full_threshold": threshold,
            "gate": gate,
        }
        final_models[variant_id] = model
        final_thresholds[variant_id] = threshold
    _assert_variant_results_complete(variant_results, variants_all)
    complete_variants = [
        variant_id
        for variant_id, result in variant_results.items()
        if result.get("complete_safe_thresholds")
    ]
    if enforce_canonical and (
        determinism["logical_fit_count"] != 54
        or determinism["physical_fit_count"] != 108
    ):
        raise ValueError("STOP_REPRODUCTION: expected 54 logical fits fitted twice")
    eligible = [
        variant_id
        for variant_id, result in variant_results.items()
        if result.get("passes_development_gate")
        and result.get("role") == "promotion_candidate"
    ]
    simplicity = plan["selection"]["family_simplicity"]
    weight_by_id = {str(item["id"]): float(item["hard_weight"]) for item in variants}
    provisional_order = sorted(
        eligible,
        key=lambda variant_id: (
            -variant_results[variant_id]["hard_oof_metrics"]["wrong_hard_rejected"],
            -variant_results[variant_id]["hard_oof_metrics"]["correct_hard_auto"],
            -variant_results[variant_id]["historical_metrics"]["auto_count"],
            int(simplicity[variant_results[variant_id]["family"]]),
            weight_by_id[variant_id],
        ),
    )
    verdict = _development_verdict(complete_variants, eligible, plan)
    dev_frame = pd.concat(dev_outputs, ignore_index=True)
    hard_frame = pd.concat(hard_outputs, ignore_index=True)
    curve_frame = (
        pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    )
    score_hashes = {
        "dev": hashlib.sha256(
            dev_frame.sort_values(["variant", "query_id"]).to_csv(index=False).encode()
        ).hexdigest(),
        "hard_oof": hashlib.sha256(
            hard_frame.sort_values(["variant", "query_id"]).to_csv(index=False).encode()
        ).hexdigest(),
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "verdict": verdict,
        "fresh_dev_eligible_variants": eligible,
        "provisional_order": provisional_order,
        "baseline_reproduction": {
            "original_dev": original_metrics,
            "effective_dev": effective_baseline,
        },
        "baseline_hard_diagnostic": {
            "used_for_gate_or_selection": False,
            "metrics": baseline_hard_diagnostic,
        },
        "variants": variant_results,
        "determinism": determinism,
        "score_hashes": score_hashes,
        "population_usage": _population_usage(
            plan,
            historical_fit_rows=len(historical_fit),
            effective_dev_rows=len(effective_dev),
            hard_rows=len(hard),
            excluded_dev_rows=len(original_dev) - len(effective_dev),
        ),
    }
    identity = {
        "schema_version": SCHEMA_VERSION,
        "input_hashes": hashes,
        "execution_lock_sha256": execution_lock_hash,
        "runner_sha256": file_sha256(Path(__file__).resolve()),
        "scaler_source_sha256": file_sha256(
            Path(__file__).resolve().parent.parent / SCALER_SOURCE
        ),
        "dataset_build_id": dataset_manifest.get("build_id"),
        "score_hashes": score_hashes,
        "fresh_dev_eligible_variants": eligible,
        "provisional_order": provisional_order,
        "verdict": verdict,
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    root = Path(output_root or plan["output_root"]).resolve()
    target = root / build_id
    if target.exists():
        raise FileExistsError(f"Immutable V4.10 artifact exists: {target}")
    root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{build_id}-", dir=root))
    try:
        dev_frame.to_parquet(stage / "dev_predictions.parquet", index=False)
        hard_frame.to_parquet(stage / "hard_oof_predictions.parquet", index=False)
        curve_frame.to_parquet(stage / "threshold_curves.parquet", index=False)
        _json_dump(stage / "variant_results.json", {"variants": variant_results})
        _json_dump(
            stage / "model_diagnostics.json",
            {
                "determinism": determinism,
                "baseline_hard_diagnostic": baseline_hard_diagnostic,
                "models": fitted_model_diagnostics,
                "scaled_feature_count": len(scaled),
                "unscaled_feature_count": len(unscaled),
            },
        )
        _json_dump(stage / "development_report.json", report)
        for variant_id in eligible:
            bundle_dir = stage / "bundles" / variant_id
            bundle_dir.mkdir(parents=True)
            model_path = bundle_dir / "acceptor_model.joblib"
            joblib.dump(final_models[variant_id], model_path)
            bundle_variant = next(item for item in variants if item["id"] == variant_id)
            bundle_metadata = {
                "schema_version": "sireto-v4.10b-acceptor-bundle-1",
                "variant": variant_id,
                "family": bundle_variant["family"],
                "threshold": final_thresholds[variant_id],
                "feature_order_name": bundle_variant["feature_order"],
                "feature_order": (
                    current80
                    if bundle_variant["feature_order"] == "current80"
                    else structured
                ),
                "training_plan_sha256": hashes["plan"],
                "dataset_manifest_sha256": hashes["dataset_manifest"],
                "execution_lock_sha256": execution_lock_hash,
                "scaler_source_sha256": identity["scaler_source_sha256"],
                "random_v48_scored": False,
                "historical_excluded_dev_used_for_training_threshold_or_gate": False,
                "fresh_scored": False,
                "test_scored": False,
            }
            _json_dump(bundle_dir / "metadata.json", bundle_metadata)
        required_before_manifest = set(plan["outputs"]["required_files"]) - {
            "manifest.json"
        }
        observed_before_manifest = {
            path.name
            for path in stage.iterdir()
            if path.is_file()
        }
        if not required_before_manifest.issubset(observed_before_manifest):
            raise ValueError("STOP_OUTPUT_INTEGRITY: required output missing")
        outputs = {
            str(path.relative_to(stage)): file_sha256(path)
            for path in sorted(stage.rglob("*"))
            if path.is_file()
        }
        manifest = {
            **identity,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "plan_sha256": hashes["plan"],
            "execution_lock": execution_lock,
            "runtime_versions": runtime_versions,
            "variant_results": variant_results,
            "output_hashes": outputs,
            "population_usage": report["population_usage"],
            "determinism": determinism,
            "invariants": {
                "variant_count": 10,
                "trained_variant_count": 9,
                "hard_group_oof_folds": 5,
                "locked_rows_read": 0,
                "locked_rows_scored": 0,
                "random_v48_rows_read": 0,
                "random_v48_rows_scored": 0,
                "historical_excluded_dev_rows_read": int(
                    len(original_dev) - len(effective_dev)
                ),
                "historical_excluded_dev_rows_baseline_scored": int(
                    len(original_dev) - len(effective_dev)
                ),
                "historical_excluded_dev_rows_trained_variant_scored": 0,
                "historical_excluded_dev_rows_threshold_used": 0,
                "historical_excluded_dev_rows_gate_used": 0,
                "historical_random_excluded_other_44_rows_materialized": 0,
                "fresh_rows_read_or_scored": 0,
                "test_rows_read_or_scored": 0,
                "threshold_selected_on_hard": False,
            },
        }
        _json_dump(stage / "manifest.json", manifest)
        stage.rename(target)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return target


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--execution-lock", type=Path, default=DEFAULT_EXECUTION_LOCK)
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print(
        run_development(
            args.training_plan,
            execution_lock_path=args.execution_lock,
            output_root=args.output_root,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
