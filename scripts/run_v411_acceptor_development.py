#!/usr/bin/env python3
"""Run the frozen V4.11 compact acceptor development comparison."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xgb_matcher.v411_acceptor import (  # noqa: E402
    COMPACT_LOGIT,
    MONOTONIC_XGB,
    V411_ACCEPTOR_FAMILIES,
    build_v411_acceptor,
)
from src.xgb_matcher.v411_scene import (  # noqa: E402
    V411_ACCEPTOR_FEATURE_NAMES,
    V411_BINARY_FEATURE_NAMES,
    V411_MONOTONIC_CONSTRAINTS,
    V411_SCALED_FEATURE_NAMES,
    validate_v411_scene_frame,
)
from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402


SCHEMA_VERSION = "sireto-v4.11-compact-acceptor-development-1"
PLAN_SCHEMA_VERSION = "sireto-v4.11-compact-acceptor-training-plan-1"
LOCK_SCHEMA_VERSION = "sireto-v4.11-compact-acceptor-execution-lock-1"
EXPERIMENT_ID = "V411_COMPACT_ACCEPTOR_DEVELOPMENT"
ACCEPTOR_SOURCE = Path("src/xgb_matcher/v411_acceptor.py")
SCENE_SOURCE = Path("src/xgb_matcher/v411_scene.py")
SCENES_FILENAME = "acceptor_scenes.parquet"
BASELINE_THRESHOLD = 0.46313316267954524

EXPECTED_MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    COMPACT_LOGIT: {
        "C": 0.1,
        "solver": "lbfgs",
        "tol": 0.0001,
        "class_weight": None,
        "max_iter": 5000,
        "random_state": 42,
    },
    MONOTONIC_XGB: {
        "n_estimators": 400,
        "learning_rate": 0.03,
        "max_depth": 2,
        "min_child_weight": 20,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 10,
        "reg_alpha": 1,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "n_jobs": 8,
        "random_state": 42,
    },
}


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_bytes(_json_bytes(payload))


def _order_sha256(values: Sequence[Any]) -> str:
    return hashlib.sha256(
        "\n".join(str(value) for value in values).encode("utf-8")
    ).hexdigest()


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
    labels: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64)
    target_values = np.asarray(targets, dtype=int)
    label_values = np.asarray(labels).astype(str)
    if (
        len(values) != len(target_values)
        or len(values) != len(label_values)
        or not np.isfinite(values).all()
    ):
        raise ValueError("STOP_INPUT_INTEGRITY: invalid decisions")
    auto = values >= float(threshold)
    auto_count = int(auto.sum())
    correct = int((auto & target_values.astype(bool)).sum())
    ambiguous_auto = int((auto & (label_values == "AMBIGUOUS")).sum())
    return {
        "threshold": float(threshold),
        "row_count": int(len(values)),
        "auto_count": auto_count,
        "correct_auto": correct,
        "error_auto": auto_count - correct,
        "review_count": int(len(values) - auto_count),
        "precision": correct / auto_count if auto_count else None,
        "precision_wilson_95": wilson_interval(correct, auto_count),
        "coverage": auto_count / len(values) if len(values) else 0.0,
        "ambiguous_auto": ambiguous_auto,
    }


def threshold_curve(
    scores: np.ndarray,
    targets: np.ndarray,
    labels: np.ndarray,
) -> pd.DataFrame:
    values = np.asarray(scores, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("STOP_INPUT_INTEGRITY: threshold scores invalid")
    candidates = np.unique(
        np.concatenate([values, [np.nextafter(values.max(), np.inf)]])
    )
    return pd.DataFrame(
        [
            decision_metrics(values, targets, labels, threshold)
            for threshold in candidates
        ]
    ).sort_values("threshold", kind="mergesort")


def select_threshold(
    scores: np.ndarray,
    targets: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, dict[str, Any], pd.DataFrame] | None:
    """Use the exact integer 99.8% rule and frozen tie-break."""

    curve = threshold_curve(scores, targets, labels)
    eligible = curve[
        curve["auto_count"].gt(0)
        & (
            1000 * curve["correct_auto"].astype(int)
            >= 998 * curve["auto_count"].astype(int)
        )
    ]
    if eligible.empty:
        return None
    winner = eligible.sort_values(
        ["auto_count", "threshold"],
        ascending=[False, False],
        kind="mergesort",
    ).iloc[0]
    threshold = float(winner["threshold"])
    return (
        threshold,
        decision_metrics(scores, targets, labels, threshold),
        curve.reset_index(drop=True),
    )


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


def _load_plan(
    plan_path: Path,
) -> tuple[dict[str, Any], dict[str, Path], dict[str, str]]:
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (
        plan.get("schema_version") != PLAN_SCHEMA_VERSION
        or plan.get("experiment_id") != EXPERIMENT_ID
    ):
        raise ValueError("STOP_INPUT_INTEGRITY: unsupported V4.11 plan")
    if list(plan.get("families") or []) != list(V411_ACCEPTOR_FAMILIES):
        raise ValueError("STOP_INPUT_INTEGRITY: acceptor family registry changed")
    if plan.get("models") != EXPECTED_MODEL_CONFIGS:
        raise ValueError("STOP_INPUT_INTEGRITY: V4.11 model config changed")
    feature_contract = plan.get("feature_contract") or {}
    expected_feature = {
        "order": V411_ACCEPTOR_FEATURE_NAMES,
        "order_sha256": _order_sha256(V411_ACCEPTOR_FEATURE_NAMES),
        "scaled": V411_SCALED_FEATURE_NAMES,
        "binary": V411_BINARY_FEATURE_NAMES,
        "monotonic_constraints": V411_MONOTONIC_CONSTRAINTS,
        "monotonic_constraints_sha256": _order_sha256(
            V411_MONOTONIC_CONSTRAINTS
        ),
    }
    if feature_contract != expected_feature:
        raise ValueError("STOP_INPUT_INTEGRITY: V4.11 feature contract changed")
    selection = plan.get("selection") or {}
    if selection != {
        "precision_milli": 998,
        "coverage_milli": 800,
        "maximum_ambiguous_auto": 0,
        "critical_family_minimum_rows": 100,
        "critical_family_maximum_coverage_loss_basis_points": 200,
        "tie_break": [
            "coverage_desc",
            "errors_asc",
            "COMPACT_LOGIT_first",
        ],
        "repetitions": 2,
    }:
        raise ValueError("STOP_INPUT_INTEGRITY: V4.11 selection policy changed")
    scene_root = Path(plan["scene_dataset"]["root"]).resolve()
    paths = {
        "plan": plan_path,
        "scene_manifest": scene_root / "manifest.json",
        "scenes": scene_root / SCENES_FILENAME,
        "baseline_model": Path(plan["baseline"]["model_path"]).resolve(),
        "baseline_metadata": Path(plan["baseline"]["metadata_path"]).resolve(),
        "baseline_scenes": Path(plan["baseline"]["scenes_path"]).resolve(),
    }
    hashes = {name: file_sha256(path) for name, path in paths.items()}
    expected_hashes = {
        "scene_manifest": plan["scene_dataset"]["manifest_sha256"],
        "scenes": plan["scene_dataset"]["scenes_sha256"],
        "baseline_model": plan["baseline"]["model_sha256"],
        "baseline_metadata": plan["baseline"]["metadata_sha256"],
        "baseline_scenes": plan["baseline"]["scenes_sha256"],
    }
    mismatches = {
        name: {"expected": expected, "observed": hashes[name]}
        for name, expected in expected_hashes.items()
        if hashes[name] != expected
    }
    if mismatches:
        raise ValueError(f"STOP_INPUT_INTEGRITY: hash mismatch {mismatches}")
    if float(plan["baseline"]["threshold"]) != BASELINE_THRESHOLD:
        raise ValueError("STOP_INPUT_INTEGRITY: baseline threshold changed")
    observed_versions = _runtime_versions()
    expected_versions = {
        str(key): str(value)
        for key, value in plan["runtime"]["required_versions"].items()
    }
    if observed_versions != expected_versions:
        raise ValueError(
            f"STOP_RUNTIME_INTEGRITY: {observed_versions} != {expected_versions}"
        )
    return plan, paths, hashes


def _git_blob_sha256(commit: str, relative_path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True,
        check=True,
    )
    return hashlib.sha256(result.stdout).hexdigest()


def validate_execution_lock(
    execution_lock_path: Path,
    paths: Mapping[str, Path],
    hashes: Mapping[str, str],
    *,
    verify_git_commit: bool = True,
) -> tuple[dict[str, Any], str]:
    lock_path = Path(execution_lock_path).resolve()
    raw = lock_path.read_bytes()
    lock = json.loads(raw)
    expected_fields = {
        "schema_version",
        "training_plan_sha256",
        "runner_sha256",
        "acceptor_source_sha256",
        "scene_source_sha256",
        "scene_manifest_sha256",
        "runner_commit",
    }
    if set(lock) != expected_fields:
        raise ValueError("STOP_EXECUTION_LOCK: fields changed")
    expected = {
        "schema_version": LOCK_SCHEMA_VERSION,
        "training_plan_sha256": hashes["plan"],
        "runner_sha256": file_sha256(Path(__file__).resolve()),
        "acceptor_source_sha256": file_sha256(
            Path(__file__).resolve().parent.parent / ACCEPTOR_SOURCE
        ),
        "scene_source_sha256": file_sha256(
            Path(__file__).resolve().parent.parent / SCENE_SOURCE
        ),
        "scene_manifest_sha256": hashes["scene_manifest"],
    }
    for key, value in expected.items():
        if str(lock.get(key)) != str(value):
            raise ValueError(f"STOP_EXECUTION_LOCK: {key} mismatch")
    commit = str(lock["runner_commit"])
    if not commit:
        raise ValueError("STOP_EXECUTION_LOCK: runner commit missing")
    if verify_git_commit:
        for relative, observed in (
            ("scripts/run_v411_acceptor_development.py", expected["runner_sha256"]),
            (str(ACCEPTOR_SOURCE), expected["acceptor_source_sha256"]),
            (str(SCENE_SOURCE), expected["scene_source_sha256"]),
        ):
            try:
                committed = _git_blob_sha256(commit, relative)
            except subprocess.CalledProcessError as error:
                raise ValueError(
                    "STOP_EXECUTION_LOCK: runner commit unavailable"
                ) from error
            if committed != observed:
                raise ValueError(
                    f"STOP_EXECUTION_LOCK: commit does not pin {relative}"
                )
    return lock, hashlib.sha256(raw).hexdigest()


def _load_scenes(
    plan: Mapping[str, Any],
    paths: Mapping[str, Path],
    *,
    enforce_canonical: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest = json.loads(paths["scene_manifest"].read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "sireto-v4.11-compact-acceptor-dataset-1":
        raise ValueError("STOP_INPUT_INTEGRITY: unsupported scene manifest")
    scenes = pd.read_parquet(paths["scenes"])
    validate_v411_scene_frame(scenes)
    required = {
        "query_id",
        "split",
        "dev_partition",
        "label_kind",
        "acceptor_target",
        "predicted_siret",
        "ground_truth_siret",
        "ranker_prediction_is_out_of_sample",
        "input_siret_state",
        "source_segment",
    }
    if not required.issubset(scenes.columns):
        raise ValueError("STOP_INPUT_INTEGRITY: scene metadata missing")
    if scenes["query_id"].astype(str).duplicated().any():
        raise ValueError("STOP_INPUT_INTEGRITY: duplicate scene")
    if not scenes["ranker_prediction_is_out_of_sample"].astype(bool).all():
        raise ValueError("STOP_LEAKAGE: ranker predictions are not OOS")
    eligible = scenes["label_kind"].isin(["MATCH_EXACT", "AMBIGUOUS"])
    if scenes.loc[eligible, "acceptor_target"].isna().any():
        raise ValueError("STOP_INPUT_INTEGRITY: eligible target missing")
    if scenes.loc[~eligible, "acceptor_target"].notna().any():
        raise ValueError("STOP_INPUT_INTEGRITY: unresolved used as negative")
    fit = scenes[scenes["split"].eq("fit") & eligible].copy()
    threshold = scenes[
        scenes["dev_partition"].eq("threshold_dev") & eligible
    ].copy()
    comparison = scenes[
        scenes["dev_partition"].eq("comparison_dev") & eligible
    ].copy()
    unresolved = scenes[~eligible].copy()
    populations = [set(frame["query_id"].astype(str)) for frame in (fit, threshold, comparison)]
    if any(
        populations[left] & populations[right]
        for left in range(3)
        for right in range(left + 1, 3)
    ):
        raise ValueError("STOP_LEAKAGE: acceptor populations overlap")
    if set(fit["acceptor_target"].astype(int)) != {0, 1}:
        raise ValueError("STOP_INPUT_INTEGRITY: fit requires both classes")
    if enforce_canonical and (
        len(scenes) != 7003
        or len(fit) != 5547
        or len(threshold) != 710
        or len(comparison) != 746
        or len(unresolved) != 0
        or fit["label_kind"].value_counts().to_dict()
        != {"MATCH_EXACT": 4666, "AMBIGUOUS": 881}
        or threshold["label_kind"].value_counts().to_dict()
        != {"MATCH_EXACT": 583, "AMBIGUOUS": 127}
        or comparison["label_kind"].value_counts().to_dict()
        != {"MATCH_EXACT": 634, "AMBIGUOUS": 112}
    ):
        raise ValueError("STOP_INPUT_INTEGRITY: canonical populations changed")
    return (
        fit.sort_values("query_id", kind="mergesort"),
        threshold.sort_values("query_id", kind="mergesort"),
        comparison.sort_values("query_id", kind="mergesort"),
        unresolved.sort_values("query_id", kind="mergesort"),
    )


def _stack_dependencies(paths: Mapping[str, Path]) -> dict[str, Any]:
    """Resolve and verify every non-acceptor dependency of the frozen stack."""

    scene_manifest = json.loads(
        paths["scene_manifest"].read_text(encoding="utf-8")
    )
    inputs = scene_manifest.get("inputs") or {}

    def pinned_input(name: str) -> tuple[Path, str]:
        record = inputs.get(name)
        if not isinstance(record, Mapping):
            raise ValueError(f"STOP_INPUT_INTEGRITY: scene input missing: {name}")
        path = Path(str(record.get("path"))).resolve()
        expected = str(record.get("sha256"))
        if file_sha256(path) != expected:
            raise ValueError(f"STOP_INPUT_INTEGRITY: scene input drift: {name}")
        return path, expected

    retrieval_manifest_path, retrieval_manifest_sha = pinned_input(
        "retrieval_dataset_manifest"
    )
    ranker_manifest_path, ranker_manifest_sha = pinned_input(
        "ranker_artifact_manifest"
    )
    taxonomy_path, taxonomy_sha = pinned_input("taxonomy")
    contract_path, contract_sha = pinned_input("contract")
    scene_source_path, scene_source_sha = pinned_input("scene_source")
    site_function_source_path, site_function_source_sha = pinned_input(
        "site_function_source"
    )
    ranker_manifest = json.loads(
        ranker_manifest_path.read_text(encoding="utf-8")
    )
    if (
        ranker_manifest.get("schema_version")
        != "sireto-v4.11-input-blind-ranker-c-development-1"
        or ranker_manifest.get("verdict") != "GO_RANKER_C"
    ):
        raise ValueError("STOP_INPUT_INTEGRITY: bundle ranker-C is not promotable")
    full_fit_record = (ranker_manifest.get("outputs") or {}).get(
        "ranker_c/full_fit.json"
    )
    if not isinstance(full_fit_record, Mapping):
        raise ValueError("STOP_INPUT_INTEGRITY: bundle ranker-C model missing")
    ranker_model_path = ranker_manifest_path.parent / "ranker_c/full_fit.json"
    ranker_model_sha = str(full_fit_record.get("sha256"))
    if file_sha256(ranker_model_path) != ranker_model_sha:
        raise ValueError("STOP_INPUT_INTEGRITY: bundle ranker-C model drift")
    retrieval_manifest = json.loads(
        retrieval_manifest_path.read_text(encoding="utf-8")
    )
    return {
        "retrieval": {
            "manifest_path": str(retrieval_manifest_path),
            "manifest_sha256": retrieval_manifest_sha,
            "build_id": retrieval_manifest.get("build_id"),
        },
        "ranker_c": {
            "manifest_path": str(ranker_manifest_path),
            "manifest_sha256": ranker_manifest_sha,
            "build_id": ranker_manifest.get("build_id"),
            "model_path": str(ranker_model_path),
            "model_sha256": ranker_model_sha,
        },
        "scene": {
            "manifest_path": str(paths["scene_manifest"]),
            "manifest_sha256": file_sha256(paths["scene_manifest"]),
            "source_path": str(scene_source_path),
            "source_sha256": scene_source_sha,
            "site_function_source_path": str(site_function_source_path),
            "site_function_source_sha256": site_function_source_sha,
            "taxonomy_path": str(taxonomy_path),
            "taxonomy_sha256": taxonomy_sha,
        },
        "contract": {
            "path": str(contract_path),
            "sha256": contract_sha,
        },
    }


def _scores(model: Any, frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(
        model.predict_proba(
            frame[V411_ACCEPTOR_FEATURE_NAMES].to_numpy(dtype=np.float64)
        )[:, 1],
        dtype=np.float64,
    )


def _baseline_decisions(
    comparison: pd.DataFrame,
    plan: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> pd.DataFrame:
    metadata = json.loads(paths["baseline_metadata"].read_text(encoding="utf-8"))
    if (
        metadata.get("schema_version") != "v4.1-acceptor-1"
        or float(metadata.get("threshold")) != BASELINE_THRESHOLD
    ):
        raise ValueError("STOP_INPUT_INTEGRITY: baseline metadata changed")
    feature_order = [str(value) for value in metadata["feature_order"]]
    baseline = pd.read_parquet(paths["baseline_scenes"])
    if baseline["query_id"].astype(str).duplicated().any():
        raise ValueError("STOP_INPUT_INTEGRITY: duplicate baseline scene")
    ids = set(comparison["query_id"].astype(str))
    baseline = baseline[baseline["query_id"].astype(str).isin(ids)].copy()
    required_columns = {*feature_order, "acceptor_eligible", "predicted_siret"}
    if len(baseline) != len(comparison) or not required_columns.issubset(
        baseline.columns
    ):
        raise ValueError("STOP_INPUT_INTEGRITY: baseline comparison incomplete")
    model = joblib.load(paths["baseline_model"])
    baseline["baseline_score"] = np.asarray(
        model.predict_proba(
            baseline[feature_order].to_numpy(dtype=np.float64)
        )[:, 1],
        dtype=np.float64,
    )
    # Reproduce the end-to-end V4.1 stack: a failed deterministic precheck
    # forces REVIEW before the frozen acceptor threshold is applied.
    baseline["baseline_auto"] = (
        baseline["acceptor_eligible"].astype(bool)
        & baseline["baseline_score"].ge(BASELINE_THRESHOLD)
    )
    current = comparison[
        ["query_id", "label_kind", "ground_truth_siret"]
    ].copy()
    output = current.merge(
        baseline[["query_id", "predicted_siret", "baseline_score", "baseline_auto"]],
        on="query_id",
        validate="one_to_one",
    )
    output["baseline_correct"] = (
        output["baseline_auto"]
        & output["label_kind"].eq("MATCH_EXACT")
        & output["predicted_siret"].astype(str).eq(
            output["ground_truth_siret"].astype(str)
        )
    )
    return output


def _binary_metrics(auto: np.ndarray, correct: np.ndarray) -> dict[str, Any]:
    auto_values = np.asarray(auto, dtype=bool)
    correct_values = np.asarray(correct, dtype=bool)
    count = len(auto_values)
    auto_count = int(auto_values.sum())
    correct_auto = int((auto_values & correct_values).sum())
    return {
        "row_count": count,
        "auto_count": auto_count,
        "correct_auto": correct_auto,
        "error_auto": auto_count - correct_auto,
        "precision": correct_auto / auto_count if auto_count else None,
        "coverage": auto_count / count if count else 0.0,
    }


def _critical_family_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    masks: dict[str, pd.Series] = {}
    for column in ("input_siret_state", "source_segment"):
        for value in sorted(frame[column].astype(str).unique()):
            masks[f"{column}={value}"] = frame[column].astype(str).eq(value)
    siren_count = frame["top1_siren_candidate_count"].astype(float)
    masks["top1_siren_candidate_count>1"] = siren_count.gt(1)
    masks["top1_siren_candidate_count=1"] = siren_count.eq(1)
    masks["top1_siren_candidate_count=0"] = siren_count.eq(0)
    role_count = frame["role_crm_count"].astype(float)
    masks["role_crm_count>0"] = role_count.gt(0)
    masks["role_crm_count=0"] = role_count.eq(0)
    return masks


def family_comparison(
    comparison: pd.DataFrame,
    candidate_auto: np.ndarray,
    baseline: pd.DataFrame,
    *,
    minimum_rows: int,
) -> tuple[list[dict[str, Any]], bool]:
    candidate_auto_values = np.asarray(candidate_auto, dtype=bool)
    candidate_correct = (
        comparison["acceptor_target"].astype(int).to_numpy().astype(bool)
    )
    baseline = baseline.copy()
    baseline["query_id"] = baseline["query_id"].astype(str)
    baseline_indexed = baseline.set_index("query_id").loc[
        comparison["query_id"].astype(str)
    ]
    baseline_auto = baseline_indexed["baseline_auto"].astype(bool).to_numpy()
    baseline_correct = baseline_indexed["baseline_correct"].astype(bool).to_numpy()
    rows: list[dict[str, Any]] = []
    all_gate = True
    for family, mask_series in _critical_family_masks(comparison).items():
        mask = mask_series.to_numpy(dtype=bool)
        candidate = _binary_metrics(
            candidate_auto_values[mask], candidate_correct[mask]
        )
        reference = _binary_metrics(baseline_auto[mask], baseline_correct[mask])
        gated = int(mask.sum()) >= int(minimum_rows)
        if family == "top1_siren_candidate_count=0":
            # The frozen gate names only >1 and =1.  Empty pools remain
            # published as an integrity diagnostic, never added post hoc.
            gated = False
        precision_noninferior = (
            True
            if reference["auto_count"] == 0
            else candidate["auto_count"] > 0
            and candidate["correct_auto"] * reference["auto_count"]
            >= reference["correct_auto"] * candidate["auto_count"]
        )
        coverage_noninferior = (
            100 * candidate["auto_count"]
            >= 100 * reference["auto_count"] - 2 * candidate["row_count"]
        )
        passed = (not gated) or (precision_noninferior and coverage_noninferior)
        all_gate = all_gate and passed
        rows.append(
            {
                "family": family,
                "gated": gated,
                "precision_noninferior": precision_noninferior,
                "coverage_loss_at_most_2_points": coverage_noninferior,
                "passed": passed,
                "candidate": candidate,
                "baseline": reference,
            }
        )
    return rows, all_gate


def _fit_repeated(
    family: str,
    config: dict[str, Any],
    fit: pd.DataFrame,
    scored_frames: Sequence[pd.DataFrame],
    repetitions: int,
) -> tuple[Any, list[np.ndarray]]:
    matrix = fit[V411_ACCEPTOR_FEATURE_NAMES].to_numpy(dtype=np.float64)
    targets = fit["acceptor_target"].astype(int).to_numpy()
    reference_scores: list[np.ndarray] | None = None
    first_model: Any = None
    for repetition in range(repetitions):
        model = build_v411_acceptor(family, config)
        model.fit(matrix, targets)
        current = [_scores(model, frame) for frame in scored_frames]
        if repetition == 0:
            first_model = model
            reference_scores = current
        elif reference_scores is None or any(
            not np.array_equal(left, right)
            for left, right in zip(reference_scores, current, strict=True)
        ):
            raise ValueError(
                f"STOP_REPRODUCTION: {family} scores differ across repetitions"
            )
    if reference_scores is None:
        raise AssertionError("V4.11 repetitions did not execute")
    return first_model, reference_scores


def _variant_gate(
    metrics: Mapping[str, Any],
    family_gate: bool,
) -> dict[str, Any]:
    auto = int(metrics["auto_count"])
    correct = int(metrics["correct_auto"])
    rows = int(metrics["row_count"])
    checks = {
        "precision_at_least_99_8": auto > 0 and 1000 * correct >= 998 * auto,
        "coverage_at_least_80": 1000 * auto >= 800 * rows,
        "ambiguous_auto_zero": int(metrics["ambiguous_auto"]) == 0,
        "critical_families": bool(family_gate),
    }
    return {"eligible": all(checks.values()), "checks": checks}


def _winner(variants: Sequence[Mapping[str, Any]]) -> str | None:
    eligible = [variant for variant in variants if variant["gate"]["eligible"]]
    if not eligible:
        return None
    ordered = sorted(
        eligible,
        key=lambda item: (
            -int(item["comparison_metrics"]["auto_count"]),
            int(item["comparison_metrics"]["error_auto"]),
            0 if item["family"] == COMPACT_LOGIT else 1,
        ),
    )
    return str(ordered[0]["family"])


def run_development(
    *,
    plan_path: Path,
    execution_lock_path: Path,
    output_root: Path,
    enforce_canonical: bool = True,
    verify_git_commit: bool = True,
) -> Path:
    plan, paths, hashes = _load_plan(plan_path)
    lock, lock_sha256 = validate_execution_lock(
        execution_lock_path,
        paths,
        hashes,
        verify_git_commit=verify_git_commit,
    )
    fit, threshold_dev, comparison_dev, unresolved = _load_scenes(
        plan, paths, enforce_canonical=enforce_canonical
    )
    baseline = _baseline_decisions(comparison_dev, plan, paths)
    stack_dependencies = _stack_dependencies(paths)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "training_plan_sha256": hashes["plan"],
        "execution_lock_sha256": lock_sha256,
        "scene_manifest_sha256": hashes["scene_manifest"],
        "scenes_sha256": hashes["scenes"],
        "runner_sha256": file_sha256(Path(__file__).resolve()),
        "acceptor_source_sha256": file_sha256(
            Path(__file__).resolve().parent.parent / ACCEPTOR_SOURCE
        ),
        "feature_order_sha256": _order_sha256(V411_ACCEPTOR_FEATURE_NAMES),
        "monotonic_constraints_sha256": _order_sha256(
            V411_MONOTONIC_CONSTRAINTS
        ),
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / build_id
    if target.exists():
        raise FileExistsError(f"Immutable V4.11 experiment exists: {target}")
    staging = Path(tempfile.mkdtemp(prefix=f".{build_id}.tmp-", dir=output_root))
    variants: list[dict[str, Any]] = []
    models: dict[str, Any] = {}
    prediction_frames: list[pd.DataFrame] = []
    curve_frames: list[pd.DataFrame] = []
    family_rows: list[dict[str, Any]] = []
    try:
        for family in V411_ACCEPTOR_FAMILIES:
            model, score_sets = _fit_repeated(
                family,
                plan["models"][family],
                fit,
                [threshold_dev, comparison_dev],
                int(plan["selection"]["repetitions"]),
            )
            threshold_scores, comparison_scores = score_sets
            selected = select_threshold(
                threshold_scores,
                threshold_dev["acceptor_target"].astype(int).to_numpy(),
                threshold_dev["label_kind"].astype(str).to_numpy(),
            )
            if selected is None:
                curve = threshold_curve(
                    threshold_scores,
                    threshold_dev["acceptor_target"].astype(int).to_numpy(),
                    threshold_dev["label_kind"].astype(str).to_numpy(),
                )
                curve.insert(0, "family", family)
                curve_frames.append(curve)
                for partition, frame, scores in (
                    ("threshold_dev", threshold_dev, threshold_scores),
                    ("comparison_dev", comparison_dev, comparison_scores),
                ):
                    predictions = frame[
                        [
                            "query_id",
                            "label_kind",
                            "ground_truth_siret",
                            "predicted_siret",
                            "acceptor_target",
                        ]
                    ].copy()
                    predictions.insert(0, "model_family", family)
                    predictions.insert(1, "evaluation_partition", partition)
                    predictions["score"] = scores
                    predictions["threshold"] = np.nan
                    predictions["decision"] = "REVIEW"
                    prediction_frames.append(predictions)
                variants.append(
                    {
                        "family": family,
                        "threshold": None,
                        "threshold_metrics": None,
                        "comparison_metrics": None,
                        "gate": {
                            "eligible": False,
                            "checks": {"safe_threshold_exists": False},
                        },
                    }
                )
                models[family] = model
                continue
            threshold, threshold_metrics, curve = selected
            comparison_metrics = decision_metrics(
                comparison_scores,
                comparison_dev["acceptor_target"].astype(int).to_numpy(),
                comparison_dev["label_kind"].astype(str).to_numpy(),
                threshold,
            )
            candidate_auto = comparison_scores >= threshold
            families, families_passed = family_comparison(
                comparison_dev,
                candidate_auto,
                baseline,
                minimum_rows=int(
                    plan["selection"]["critical_family_minimum_rows"]
                ),
            )
            gate = _variant_gate(comparison_metrics, families_passed)
            variants.append(
                {
                    "family": family,
                    "threshold": threshold,
                    "threshold_metrics": threshold_metrics,
                    "comparison_metrics": comparison_metrics,
                    "gate": gate,
                }
            )
            models[family] = model
            curve.insert(0, "family", family)
            curve_frames.append(curve)
            for row in families:
                family_rows.append({"model_family": family, **row})
            for partition, frame, scores in (
                ("threshold_dev", threshold_dev, threshold_scores),
                ("comparison_dev", comparison_dev, comparison_scores),
            ):
                predictions = frame[
                    [
                        "query_id",
                        "label_kind",
                        "ground_truth_siret",
                        "predicted_siret",
                        "acceptor_target",
                    ]
                ].copy()
                predictions.insert(0, "model_family", family)
                predictions.insert(1, "evaluation_partition", partition)
                predictions["score"] = scores
                predictions["threshold"] = threshold
                predictions["decision"] = np.where(
                    (
                        predictions["label_kind"].ne("UNRESOLVED")
                        & predictions["score"].ge(threshold)
                    ),
                    "AUTO_MATCH",
                    "REVIEW",
                )
                prediction_frames.append(predictions)

        winner = _winner(variants)
        verdict = (
            "GO_FREEZE_V411_CANDIDATE"
            if winner is not None
            else "PIVOT_COMPACT_ACCEPTOR"
        )
        pd.concat(prediction_frames, ignore_index=True).to_parquet(
            staging / "predictions.parquet", index=False
        )
        pd.concat(curve_frames, ignore_index=True).to_parquet(
            staging / "threshold_curves.parquet", index=False
        )
        unresolved_predictions = unresolved[
            [
                "query_id",
                "label_kind",
                "ground_truth_siret",
                "predicted_siret",
                "acceptor_target",
            ]
        ].copy()
        unresolved_predictions["score"] = np.nan
        unresolved_predictions["threshold"] = np.nan
        unresolved_predictions["decision"] = "REVIEW"
        unresolved_predictions["review_reason"] = "UNRESOLVED"
        unresolved_predictions.to_parquet(
            staging / "unresolved_review.parquet", index=False
        )
        _json_dump(staging / "critical_family_metrics.json", {"rows": family_rows})
        baseline_metrics = _binary_metrics(
            baseline["baseline_auto"].astype(bool).to_numpy(),
            baseline["baseline_correct"].astype(bool).to_numpy(),
        )
        result = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "build_id": build_id,
            "verdict": verdict,
            "winner": winner,
            "variants": variants,
            "baseline_comparison_metrics": baseline_metrics,
            "populations": {
                "fit": len(fit),
                "threshold_dev": len(threshold_dev),
                "comparison_dev": len(comparison_dev),
                "unresolved_forced_review": len(unresolved),
            },
            "reproduction": {
                "repetitions": int(plan["selection"]["repetitions"]),
                "scores_bit_exact": True,
            },
        }
        _json_dump(staging / "results.json", result)
        bundle_outputs: dict[str, Any] = {}
        if winner is not None:
            bundle = staging / "bundle"
            bundle.mkdir()
            model_path = bundle / "acceptor_model.joblib"
            joblib.dump(models[winner], model_path)
            threshold = next(
                float(item["threshold"])
                for item in variants
                if item["family"] == winner
            )
            bundle_metadata = {
                "schema_version": "sireto-v4.11-acceptor-bundle-1",
                "model_bundle_id": build_id,
                "model_family": winner,
                "threshold": threshold,
                "decision_rule": "AUTO_MATCH if score >= threshold else REVIEW",
                "feature_order": V411_ACCEPTOR_FEATURE_NAMES,
                "scaled_features": V411_SCALED_FEATURE_NAMES,
                "binary_features": V411_BINARY_FEATURE_NAMES,
                "monotonic_constraints": V411_MONOTONIC_CONSTRAINTS,
                "scene_dataset_manifest_sha256": hashes["scene_manifest"],
                "training_plan_sha256": hashes["plan"],
                "unresolved_policy": "FORCE_REVIEW",
            }
            _json_dump(bundle / "metadata.json", bundle_metadata)
            stack_manifest = {
                "schema_version": "sireto-v4.11-end-to-end-bundle-1",
                "model_bundle_id": build_id,
                "decision_rule": bundle_metadata["decision_rule"],
                "unresolved_policy": "FORCE_REVIEW",
                "components": {
                    **stack_dependencies,
                    "acceptor": {
                        "model_path": "acceptor_model.joblib",
                        "model_sha256": file_sha256(model_path),
                        "metadata_path": "metadata.json",
                        "metadata_sha256": file_sha256(bundle / "metadata.json"),
                        "source_path": str(ACCEPTOR_SOURCE),
                        "source_sha256": file_sha256(
                            Path(__file__).resolve().parent.parent
                            / ACCEPTOR_SOURCE
                        ),
                    },
                },
            }
            _json_dump(bundle / "stack_manifest.json", stack_manifest)
            bundle_outputs = {
                "bundle/acceptor_model.joblib": {
                    "sha256": file_sha256(model_path),
                    "size_bytes": int(model_path.stat().st_size),
                },
                "bundle/metadata.json": {
                    "sha256": file_sha256(bundle / "metadata.json"),
                    "size_bytes": int((bundle / "metadata.json").stat().st_size),
                },
                "bundle/stack_manifest.json": {
                    "sha256": file_sha256(bundle / "stack_manifest.json"),
                    "size_bytes": int(
                        (bundle / "stack_manifest.json").stat().st_size
                    ),
                },
            }
        output_paths = [
            staging / "predictions.parquet",
            staging / "threshold_curves.parquet",
            staging / "unresolved_review.parquet",
            staging / "critical_family_metrics.json",
            staging / "results.json",
        ]
        outputs = {
            path.name: {
                "sha256": file_sha256(path),
                "size_bytes": int(path.stat().st_size),
                "row_count": (
                    int(len(pd.read_parquet(path)))
                    if path.suffix == ".parquet"
                    else None
                ),
            }
            for path in output_paths
        }
        outputs.update(bundle_outputs)
        manifest = {
            **identity,
            "build_identity": identity,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "runner_commit": lock["runner_commit"],
            "inputs": {
                name: {
                    "path": str(paths[name]),
                    "sha256": hashes[name],
                }
                for name in paths
            },
            "outputs": outputs,
            "verdict": verdict,
            "winner": winner,
            "invariants": {
                "fit_ranker_predictions_out_of_fold": True,
                "threshold_dev_only_for_threshold": True,
                "comparison_dev_only_for_selection": True,
                "unresolved_excluded_and_forced_review": True,
                "consumed_hard_opened": False,
                "random_or_locked_opened": False,
                "unseen_225_opened": False,
                "final_test_opened": False,
                "repetitions": 2,
                "scores_bit_exact": True,
            },
        }
        _json_dump(staging / "manifest.json", manifest)
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--execution-lock", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = run_development(
        plan_path=args.plan,
        execution_lock_path=args.execution_lock,
        output_root=args.output_root,
    )
    print(target)


if __name__ == "__main__":
    main()
