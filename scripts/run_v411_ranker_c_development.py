#!/usr/bin/env python3
"""Train and gate the V4.11 input-blind ranker C.

This runner consumes only a validated V4.11 dataset passed explicitly on the
command line.  It creates five out-of-fold models for fit scenes and one model
trained on all eligible fit queries for dev scoring.  The complete procedure
is repeated and required to be bit-exact before an immutable artifact is
published.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_v411_input_blind_dataset import (  # noqa: E402
    CANDIDATE_CEILING,
    DEFAULT_CONTRACT,
    EXPERIMENT_ID,
    RANKER_C_FEATURE_ORDER,
    feature_order_sha256,
    validate_artifact as validate_dataset_artifact,
)
from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402


SCHEMA_VERSION = "sireto-v4.11-input-blind-ranker-c-development-1"
RUN_ID = "V411_INPUT_BLIND_RANKER_C"
SEED = 42
EXPECTED_QUERY_COUNT = 7_003
EXPECTED_FIT_COUNT = 5_547
EXPECTED_DEV_COUNT = 1_456
TARGET_DEV_HIT_AT_1 = 0.998
MAX_LARGE_SEGMENT_REGRESSION = 0.02
LARGE_SEGMENT_MINIMUM = 100
BASELINE_MODEL_PATH = Path(
    "/Volumes/CATNAT_DATA/SIRETO_RECALL100/models/"
    "v4_6_aligned_ranker/421f2cd0cc436af7/ranker_b/ranker.json"
)
BASELINE_METADATA_PATH = BASELINE_MODEL_PATH.with_name("metadata.json")
EXPECTED_BASELINE_MODEL_SHA256 = (
    "ffa0014e1650f679651da91b4b52ef53636eb4fee804666afb8f7756a90c50d7"
)
EXPECTED_BASELINE_METADATA_SHA256 = (
    "39eb014b8c833c79cd50027db110b63144ad482e7466c7f97f8fcdd98b519f11"
)
RANKER_PARAMS: dict[str, Any] = {
    "objective": "rank:pairwise",
    "eval_metric": "ndcg@1",
    "n_estimators": 800,
    "learning_rate": 0.035,
    "max_depth": 6,
    "min_child_weight": 3,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 5.0,
    "random_state": SEED,
    "n_jobs": -1,
    "tree_method": "hist",
}
PREDICTION_COLUMNS = [
    "query_id",
    "candidate_siret",
    "candidate_siren",
    "retrieval_rank",
    "is_ground_truth",
    "ranker_score",
    "prediction_origin",
    "oof_fold",
    "ranker_rank",
]


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _external_path(path: Path, *, name: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(Path("/Volumes/CATNAT_DATA")):
        raise ValueError(f"{name} must be located under /Volumes/CATNAT_DATA")
    return resolved


def _dependency_versions() -> dict[str, str]:
    output: dict[str, str] = {}
    for package in ("numpy", "pandas", "pyarrow", "scikit-learn", "xgboost"):
        try:
            output[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            output[package] = "missing"
    return output


def load_dataset(
    dataset_dir: Path,
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Validate and load the single explicit V4.11 input artifact."""

    dataset_dir = Path(dataset_dir).resolve()
    validate_dataset_artifact(dataset_dir)
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("Unexpected V4.11 dataset experiment")
    if manifest.get("contract_sha256") != file_sha256(DEFAULT_CONTRACT):
        raise ValueError("V4.11 dataset was built against another contract")
    ranker_spec = manifest.get("ranker_c") or {}
    if list(ranker_spec.get("feature_order") or []) != RANKER_C_FEATURE_ORDER:
        raise ValueError("V4.11 manifest ranker feature order changed")
    if (
        ranker_spec.get("feature_order_sha256")
        != feature_order_sha256(RANKER_C_FEATURE_ORDER)
    ):
        raise ValueError("V4.11 manifest ranker feature-order hash changed")
    integrity = manifest.get("integrity") or {}
    if integrity.get("retrieval_gate_passed") is not True:
        raise ValueError("PIVOT_INPUT_BLIND_RETRIEVAL: dataset gate did not pass")
    if integrity.get("positive_injection") is not False:
        raise ValueError("V4.11 dataset suggests positive injection")
    if int((integrity.get("pool_size") or {}).get("max", -1)) > CANDIDATE_CEILING:
        raise ValueError("V4.11 dataset candidate ceiling changed")

    queries = pd.read_parquet(dataset_dir / "queries.parquet")
    query_audit = pd.read_parquet(dataset_dir / "query_audit.parquet")
    labels = pd.read_parquet(dataset_dir / "labels.parquet")
    assignments = pd.read_parquet(dataset_dir / "split_assignments.parquet")
    candidates = pd.read_parquet(dataset_dir / "candidates_sparse_top100.parquet")
    for frame in (queries, query_audit, labels, assignments, candidates):
        frame["query_id"] = frame["query_id"].astype(str)
    if len(queries) != EXPECTED_QUERY_COUNT:
        raise ValueError("V4.11 ranker requires exactly 7,003 queries")
    split_counts = assignments["split"].value_counts().to_dict()
    if split_counts != {"fit": EXPECTED_FIT_COUNT, "dev": EXPECTED_DEV_COUNT}:
        raise ValueError(f"Unexpected V4.11 split counts: {split_counts}")
    if assignments["query_id"].duplicated().any():
        raise ValueError("V4.11 assignments are not unique")
    if set(assignments["query_id"]) != set(queries["query_id"]):
        raise ValueError("V4.11 assignments do not cover the query population")
    if set(labels["query_id"]) != set(queries["query_id"]):
        raise ValueError("V4.11 labels do not cover the query population")
    if set(query_audit["query_id"]) != set(queries["query_id"]):
        raise ValueError("V4.11 query audit does not cover the query population")
    if list(candidates.columns[-45:]) != RANKER_C_FEATURE_ORDER:
        raise ValueError("V4.11 candidate feature columns are not in frozen order")
    matrix = candidates[RANKER_C_FEATURE_ORDER].to_numpy(dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("V4.11 ranker feature matrix is not finite")
    if candidates.duplicated(["query_id", "candidate_siret"]).any():
        raise ValueError("V4.11 candidate pools contain duplicate SIRETs")
    if (
        candidates.groupby("query_id", sort=False).size().max()
        > CANDIDATE_CEILING
    ):
        raise ValueError("V4.11 candidate ceiling exceeded")
    assignments = assignments.copy()
    assignments["oof_fold"] = assignments["oof_fold"].astype(int)
    if set(assignments["oof_fold"]) != set(range(5)):
        raise ValueError("V4.11 requires the five frozen OOF folds")
    component_fold_count = assignments.groupby("siren_component_id")[
        "oof_fold"
    ].nunique()
    if int(component_fold_count.max()) != 1:
        raise ValueError("A SIREN component spans multiple OOF folds")
    candidates = candidates.merge(
        assignments[["query_id", "split", "oof_fold"]],
        on="query_id",
        how="left",
        validate="many_to_one",
    )
    labels = labels.merge(
        assignments[["query_id", "split", "oof_fold"]],
        on="query_id",
        how="left",
        validate="one_to_one",
    )
    return manifest, queries, query_audit, labels, assignments, candidates


def eligible_ranker_rows(
    candidates: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    """Keep exact queries whose true SIRET is physically present in the pool."""

    exact = labels[labels["label_kind"].astype(str).eq("MATCH_EXACT")][
        ["query_id", "ground_truth_siret"]
    ].copy()
    exact["ground_truth_siret"] = exact["ground_truth_siret"].fillna("").astype(str)
    scoped = candidates.merge(exact, on="query_id", how="inner", validate="many_to_one")
    scoped["is_ground_truth"] = (
        scoped["candidate_siret"].fillna("").astype(str)
        == scoped["ground_truth_siret"]
    ).astype(np.int8)
    positive_counts = scoped.groupby("query_id")["is_ground_truth"].sum()
    eligible_ids = set(positive_counts[positive_counts.eq(1)].index.astype(str))
    output = scoped[scoped["query_id"].isin(eligible_ids)].drop(
        columns=["ground_truth_siret"]
    )
    return output


def fit_ranker(rows: pd.DataFrame) -> xgb.XGBRanker:
    ordered = rows.sort_values(
        ["query_id", "candidate_siret"], kind="mergesort"
    ).copy()
    if ordered.empty:
        raise ValueError("Cannot train ranker C without eligible candidate rows")
    positives = ordered.groupby("query_id")["is_ground_truth"].sum()
    if not positives.eq(1).all():
        raise ValueError("Every ranker C training group must have one positive")
    groups = ordered.groupby("query_id", sort=False).size().to_numpy()
    model = xgb.XGBRanker(**RANKER_PARAMS)
    model.fit(
        ordered[RANKER_C_FEATURE_ORDER].to_numpy(dtype=np.float32),
        ordered["is_ground_truth"].to_numpy(dtype=np.int8),
        group=groups,
        verbose=False,
    )
    return model


def score_rows(
    model: xgb.XGBRanker,
    rows: pd.DataFrame,
    *,
    origin: str,
    fold: int | None,
) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=PREDICTION_COLUMNS)
    output = rows[
        [
            "query_id",
            "candidate_siret",
            "candidate_siren",
            "retrieval_rank",
            "is_ground_truth",
        ]
    ].copy()
    output["ranker_score"] = model.predict(
        rows[RANKER_C_FEATURE_ORDER].to_numpy(dtype=np.float32)
    ).astype(np.float32)
    if not np.isfinite(output["ranker_score"].to_numpy(dtype=np.float64)).all():
        raise ValueError("Ranker C produced a non-finite score")
    output["prediction_origin"] = origin
    output["oof_fold"] = pd.Series(
        [fold] * len(output), index=output.index, dtype="Int8"
    )
    output = output.sort_values(
        ["query_id", "ranker_score", "retrieval_rank", "candidate_siret"],
        ascending=[True, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    output["ranker_rank"] = (
        output.groupby("query_id", sort=False).cumcount() + 1
    ).astype(np.int16)
    return output[PREDICTION_COLUMNS]


def train_once(
    candidates: pd.DataFrame,
    labels: pd.DataFrame,
) -> tuple[dict[str, xgb.XGBRanker], pd.DataFrame, dict[str, Any]]:
    fit_labels = labels[labels["split"].eq("fit")].copy()
    fit_candidates = candidates[candidates["split"].eq("fit")].copy()
    models: dict[str, xgb.XGBRanker] = {}
    prediction_parts: list[pd.DataFrame] = []
    training: dict[str, Any] = {"folds": {}}
    for fold in range(5):
        validation_ids = set(
            fit_labels.loc[fit_labels["oof_fold"].eq(fold), "query_id"]
        )
        train_labels = fit_labels[fit_labels["oof_fold"].ne(fold)]
        train_rows = eligible_ranker_rows(fit_candidates, train_labels)
        exact_training_count = int(
            train_labels["label_kind"].astype(str).eq("MATCH_EXACT").sum()
        )
        model = fit_ranker(train_rows)
        models[f"oof_fold_{fold}"] = model
        validation_rows = fit_candidates[
            fit_candidates["query_id"].isin(validation_ids)
        ]
        prediction_parts.append(
            score_rows(
                model,
                validation_rows,
                origin="ranker_c_oof",
                fold=fold,
            )
        )
        training["folds"][str(fold)] = {
            "training_query_count": int(train_rows["query_id"].nunique()),
            "training_candidate_count": int(len(train_rows)),
            "exact_query_count": exact_training_count,
            "exact_missing_positive_excluded": (
                exact_training_count - int(train_rows["query_id"].nunique())
            ),
            "scored_query_count": int(len(validation_ids)),
            "scored_candidate_count": int(len(validation_rows)),
        }
    final_rows = eligible_ranker_rows(fit_candidates, fit_labels)
    final_model = fit_ranker(final_rows)
    models["full_fit"] = final_model
    dev_rows = candidates[candidates["split"].eq("dev")]
    prediction_parts.append(
        score_rows(
            final_model,
            dev_rows,
            origin="ranker_c_dev",
            fold=None,
        )
    )
    predictions = pd.concat(prediction_parts, ignore_index=True)
    predictions = predictions.sort_values(
        ["query_id", "ranker_rank", "candidate_siret"], kind="mergesort"
    ).reset_index(drop=True)
    expected_candidate_count = int(len(candidates))
    if len(predictions) != expected_candidate_count:
        raise ValueError("Ranker C predictions do not cover every candidate")
    if predictions.duplicated(["query_id", "candidate_siret"]).any():
        raise ValueError("Ranker C predictions contain duplicate candidates")
    full_fit_exact_count = int(
        fit_labels["label_kind"].astype(str).eq("MATCH_EXACT").sum()
    )
    training["full_fit"] = {
        "training_query_count": int(final_rows["query_id"].nunique()),
        "training_candidate_count": int(len(final_rows)),
        "exact_query_count": full_fit_exact_count,
        "exact_missing_positive_excluded": (
            full_fit_exact_count - int(final_rows["query_id"].nunique())
        ),
        "dev_scored_query_count": int(dev_rows["query_id"].nunique()),
        "dev_scored_candidate_count": int(len(dev_rows)),
    }
    return models, predictions, training


def _save_models(models: Mapping[str, xgb.XGBRanker], directory: Path) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=False)
    hashes: dict[str, str] = {}
    for name in sorted(models):
        path = directory / f"{name}.json"
        models[name].save_model(path)
        hashes[name] = file_sha256(path)
    return hashes


def assert_bit_exact_repetitions(
    first_predictions: pd.DataFrame,
    second_predictions: pd.DataFrame,
    first_model_hashes: Mapping[str, str],
    second_model_hashes: Mapping[str, str],
) -> dict[str, Any]:
    identity_columns = [name for name in PREDICTION_COLUMNS if name != "ranker_score"]
    left = first_predictions.sort_values(
        ["query_id", "candidate_siret"], kind="mergesort"
    ).reset_index(drop=True)
    right = second_predictions.sort_values(
        ["query_id", "candidate_siret"], kind="mergesort"
    ).reset_index(drop=True)
    if not left[identity_columns].equals(right[identity_columns]):
        raise ValueError("Repeated ranker C prediction identities/ranks differ")
    score_left = left["ranker_score"].to_numpy(dtype=np.float32)
    score_right = right["ranker_score"].to_numpy(dtype=np.float32)
    if not np.array_equal(score_left, score_right):
        raise ValueError("Repeated ranker C scores are not bit-exact")
    if dict(first_model_hashes) != dict(second_model_hashes):
        raise ValueError("Repeated ranker C model files are not bit-exact")
    return {
        "model_files_bit_exact": True,
        "scores_bit_exact": True,
        "ranks_bit_exact": True,
        "max_score_absolute_delta": 0.0,
        "model_hashes": dict(first_model_hashes),
    }


def build_top1(
    predictions: pd.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    """Produce one explicit top-1 row per query, including retrieval misses."""

    pool_state = (
        predictions.groupby("query_id", sort=False)
        .agg(
            pool_candidate_count=("candidate_siret", "size"),
            truth_present_in_pool=("is_ground_truth", "max"),
        )
        .reset_index()
    )
    top1 = predictions[predictions["ranker_rank"].eq(1)].copy()
    top1 = assignments[
        ["query_id", "split", "oof_fold"]
    ].merge(
        top1,
        on="query_id",
        how="left",
        validate="one_to_one",
        suffixes=("_assignment", ""),
    )
    top1 = top1.merge(
        pool_state,
        on="query_id",
        how="left",
        validate="one_to_one",
    )
    top1["pool_candidate_count"] = (
        top1["pool_candidate_count"].fillna(0).astype(int)
    )
    top1["empty_pool"] = top1["pool_candidate_count"].eq(0)
    top1["truth_present_in_pool"] = (
        top1["truth_present_in_pool"].fillna(0).astype(bool)
    )
    top1["truth_absent_from_pool"] = ~top1["truth_present_in_pool"]
    if len(top1) != len(assignments) or top1["query_id"].duplicated().any():
        raise ValueError("Ranker C top-1 does not contain one row per query")
    return top1


def exact_metrics(
    top1: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    split: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    exact = labels[
        labels["split"].eq(split)
        & labels["label_kind"].astype(str).eq("MATCH_EXACT")
    ][["query_id", "ground_truth_siret", "ground_truth_siren"]].copy()
    evaluated = exact.merge(
        top1[
            [
                "query_id",
                "candidate_siret",
                "candidate_siren",
                "empty_pool",
                "truth_absent_from_pool",
            ]
        ],
        on="query_id",
        how="left",
        validate="one_to_one",
    )
    expected_siret = evaluated["ground_truth_siret"].fillna("").astype(str)
    predicted_siret = evaluated["candidate_siret"].fillna("").astype(str)
    evaluated["siret_hit"] = predicted_siret.eq(expected_siret)
    expected_siren = evaluated["ground_truth_siren"].fillna("").astype(str)
    expected_siren = expected_siren.where(
        expected_siren.str.len().eq(9), expected_siret.str[:9]
    )
    evaluated["siren_hit"] = (
        evaluated["candidate_siren"].fillna("").astype(str).eq(expected_siren)
    )
    count = len(evaluated)
    return evaluated, {
        "exact_count": count,
        "siret_successes": int(evaluated["siret_hit"].sum()),
        "siret_hit_at_1": (
            float(evaluated["siret_hit"].mean()) if count else 0.0
        ),
        "siren_successes": int(evaluated["siren_hit"].sum()),
        "siren_hit_at_1": (
            float(evaluated["siren_hit"].mean()) if count else 0.0
        ),
        "empty_pool_count": int(evaluated["empty_pool"].fillna(True).sum()),
        "truth_absent_from_pool_count": int(
            evaluated["truth_absent_from_pool"].fillna(True).sum()
        ),
        "retrieval_miss_count": int(
            evaluated["truth_absent_from_pool"].fillna(True).sum()
        ),
    }


def load_masked_ranker_b(
    model_path: Path = BASELINE_MODEL_PATH,
    metadata_path: Path = BASELINE_METADATA_PATH,
) -> tuple[xgb.XGBRanker, list[str], dict[str, Any]]:
    if file_sha256(model_path) != EXPECTED_BASELINE_MODEL_SHA256:
        raise ValueError("Masked ranker B model hash mismatch")
    if file_sha256(metadata_path) != EXPECTED_BASELINE_METADATA_SHA256:
        raise ValueError("Masked ranker B metadata hash mismatch")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_order = list(metadata.get("feature_order") or [])
    if len(feature_order) != 64:
        raise ValueError("Masked ranker B must have exactly 64 features")
    model = xgb.XGBRanker()
    model.load_model(model_path)
    return model, feature_order, {
        "model_path": str(model_path),
        "model_sha256": EXPECTED_BASELINE_MODEL_SHA256,
        "metadata_path": str(metadata_path),
        "metadata_sha256": EXPECTED_BASELINE_METADATA_SHA256,
    }


def masked_ranker_b_matrix(
    candidates: pd.DataFrame,
    feature_order: Sequence[str],
) -> np.ndarray:
    """Project V4.11 pools into ranker B with identifier signals masked."""

    projected = pd.DataFrame(
        0.0, index=candidates.index, columns=list(feature_order), dtype=np.float32
    )
    shared = [name for name in RANKER_C_FEATURE_ORDER[:-1] if name in projected]
    projected.loc[:, shared] = candidates[shared].to_numpy(dtype=np.float32)
    recip = candidates["retrieval_rank_recip"].to_numpy(dtype=np.float32)
    projected["admission_rank_recip"] = recip
    projected["admission_current_sparse_rank_recip"] = recip
    projected["admission_fusion_score"] = (
        1.0
        / (
            60.0
            + candidates["retrieval_rank"].to_numpy(dtype=np.float32)
        )
    )
    projected["admission_channel_count"] = 1.0
    projected["candidate_is_active"] = 1.0
    projected["candidate_from_sparse"] = 1.0
    # Other per-channel ranks not materialized by V4.11 stay zero.
    # All five identifier-dependent features required by the contract stay zero.
    forbidden = {
        "input_siret_exact_match",
        "input_siren_exact_match",
        "candidate_from_input_siret",
        "candidate_from_input_siren",
        "candidate_from_closed_alias",
    }
    if projected[list(forbidden)].to_numpy().any():
        raise AssertionError("Masked ranker B identifier features are non-zero")
    return projected[list(feature_order)].to_numpy(dtype=np.float32)


def masked_ranker_b_projection_metadata() -> dict[str, Any]:
    """Describe the exact diagnostic projection applied to ranker B."""

    return {
        "shared_candidate_features": RANKER_C_FEATURE_ORDER[:-1],
        "admission_rank_recip": "1/retrieval_rank",
        "admission_current_sparse_rank_recip": "1/retrieval_rank",
        "admission_fusion_score": "1/(60+retrieval_rank)",
        "admission_channel_count": 1.0,
        "candidate_is_active": 1.0,
        "candidate_from_sparse": 1.0,
        "all_other_ranker_b_only_features": 0.0,
    }


def score_masked_ranker_b(
    candidates: pd.DataFrame,
    *,
    model_path: Path = BASELINE_MODEL_PATH,
    metadata_path: Path = BASELINE_METADATA_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    model, feature_order, identity = load_masked_ranker_b(
        model_path, metadata_path
    )
    dev = candidates[candidates["split"].eq("dev")].copy()
    output = dev[
        [
            "query_id",
            "candidate_siret",
            "candidate_siren",
            "retrieval_rank",
            "is_ground_truth",
        ]
    ].copy()
    output["ranker_score"] = model.predict(
        masked_ranker_b_matrix(dev, feature_order)
    ).astype(np.float32)
    output["prediction_origin"] = "ranker_b_masked_dev"
    output["oof_fold"] = pd.Series(
        [None] * len(output), index=output.index, dtype="Int8"
    )
    output = output.sort_values(
        ["query_id", "ranker_score", "retrieval_rank", "candidate_siret"],
        ascending=[True, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    output["ranker_rank"] = (
        output.groupby("query_id", sort=False).cumcount() + 1
    ).astype(np.int16)
    return output[PREDICTION_COLUMNS], {
        **identity,
        "projection": masked_ranker_b_projection_metadata(),
    }


def segment_comparison(
    c_hits: pd.DataFrame,
    b_hits: pd.DataFrame,
    query_audit: pd.DataFrame,
) -> tuple[dict[str, Any], bool]:
    paired = c_hits[["query_id", "siret_hit"]].rename(
        columns={"siret_hit": "c_hit"}
    ).merge(
        b_hits[["query_id", "siret_hit"]].rename(columns={"siret_hit": "b_hit"}),
        on="query_id",
        validate="one_to_one",
    ).merge(query_audit, on="query_id", how="left", validate="one_to_one")
    output: dict[str, Any] = {}
    passed = True
    for column in ("input_siret_state", "source_segment"):
        if column not in paired:
            raise ValueError(f"V4.11 query audit is missing {column}")
        output[column] = {}
        for value, group in paired.groupby(column, dropna=False):
            count = len(group)
            b_rate = float(group["b_hit"].mean())
            c_rate = float(group["c_hit"].mean())
            regression = b_rate - c_rate
            gated = count >= LARGE_SEGMENT_MINIMUM
            if gated and regression > MAX_LARGE_SEGMENT_REGRESSION + 1e-15:
                passed = False
            output[column][str(value)] = {
                "count": int(count),
                "ranker_b_masked_hit_at_1": b_rate,
                "ranker_c_hit_at_1": c_rate,
                "ranker_c_delta": c_rate - b_rate,
                "gated": gated,
                "passes": (not gated) or regression <= MAX_LARGE_SEGMENT_REGRESSION,
            }
    return output, passed


def build_artifact(
    *,
    dataset_dir: Path,
    output_root: Path,
    baseline_model_path: Path = BASELINE_MODEL_PATH,
    baseline_metadata_path: Path = BASELINE_METADATA_PATH,
) -> Path:
    output_root = _external_path(output_root, name="output_root")
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    (
        dataset_manifest,
        _queries,
        query_audit,
        labels,
        assignments,
        candidates,
    ) = load_dataset(dataset_dir)

    source_path = Path(__file__).resolve()
    identity = {
        "schema_version": SCHEMA_VERSION,
        "run_id": RUN_ID,
        "contract_sha256": file_sha256(DEFAULT_CONTRACT),
        "runner_source_sha256": file_sha256(source_path),
        "dataset_manifest_sha256": file_sha256(Path(dataset_dir) / "manifest.json"),
        "dataset_build_id": dataset_manifest["build_id"],
        "feature_order": RANKER_C_FEATURE_ORDER,
        "feature_order_sha256": feature_order_sha256(RANKER_C_FEATURE_ORDER),
        "ranker_params": RANKER_PARAMS,
        "folds": [0, 1, 2, 3, 4],
        "repetitions": 2,
        "baseline_model_sha256": EXPECTED_BASELINE_MODEL_SHA256,
        "baseline_metadata_sha256": EXPECTED_BASELINE_METADATA_SHA256,
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    target = output_root / build_id
    if target.exists():
        raise FileExistsError(f"Immutable V4.11 ranker artifact exists: {target}")
    staging = Path(tempfile.mkdtemp(prefix=f".{build_id}.tmp-", dir=output_root))
    repeat_staging = Path(tempfile.mkdtemp(prefix=".v411-ranker-repeat-"))
    try:
        first_models, first_predictions, first_training = train_once(
            candidates, labels
        )
        first_hashes = _save_models(first_models, staging / "ranker_c")
        second_models, second_predictions, second_training = train_once(
            candidates, labels
        )
        second_hashes = _save_models(second_models, repeat_staging / "ranker_c")
        if first_training != second_training:
            raise ValueError("Repeated ranker C training populations differ")
        determinism = assert_bit_exact_repetitions(
            first_predictions,
            second_predictions,
            first_hashes,
            second_hashes,
        )

        top1 = build_top1(first_predictions, assignments)
        fit_hits, fit_metrics = exact_metrics(top1, labels, split="fit")
        dev_hits, dev_metrics = exact_metrics(top1, labels, split="dev")
        masked_predictions, baseline_identity = score_masked_ranker_b(
            candidates,
            model_path=baseline_model_path,
            metadata_path=baseline_metadata_path,
        )
        masked_top1 = build_top1(masked_predictions, assignments[
            assignments["split"].eq("dev")
        ])
        b_hits, b_metrics = exact_metrics(masked_top1, labels, split="dev")
        segments, segments_pass = segment_comparison(
            dev_hits, b_hits, query_audit
        )
        checks = {
            "dataset_retrieval_gate_passed": True,
            "dev_siret_hit_at_1_gte_0_998": (
                dev_metrics["siret_hit_at_1"] >= TARGET_DEV_HIT_AT_1
            ),
            "no_large_segment_regression_gt_2pp_vs_masked_ranker_b": segments_pass,
            "model_files_bit_exact": determinism["model_files_bit_exact"],
            "prediction_scores_bit_exact": determinism["scores_bit_exact"],
            "prediction_ranks_bit_exact": determinism["ranks_bit_exact"],
            "candidate_ceiling_lte_100": (
                int((dataset_manifest["integrity"]["pool_size"])["max"])
                <= CANDIDATE_CEILING
            ),
            "positive_injection_false": (
                dataset_manifest["integrity"]["positive_injection"] is False
            ),
        }
        verdict = (
            "GO_RANKER_C"
            if all(bool(value) for value in checks.values())
            else "PIVOT_INPUT_BLIND_RANKER"
        )

        predictions_path = staging / "predictions_ranker_c_oof_dev.parquet"
        top1_path = staging / "top1_ranker_c_oof_dev.parquet"
        masked_path = staging / "predictions_ranker_b_masked_dev.parquet"
        evaluated_path = staging / "evaluated_exact_dev.parquet"
        first_predictions.to_parquet(predictions_path, index=False)
        top1.to_parquet(top1_path, index=False)
        masked_predictions.to_parquet(masked_path, index=False)
        dev_hits.to_parquet(evaluated_path, index=False)
        report = {
            "schema_version": SCHEMA_VERSION,
            "build_id": build_id,
            "verdict": verdict,
            "metrics": {
                "fit_oof": fit_metrics,
                "dev": dev_metrics,
                "ranker_b_masked_dev_diagnostic": b_metrics,
                "segments_vs_ranker_b_masked": segments,
            },
            "checks": checks,
            "determinism": determinism,
            "training": first_training,
            "baseline_diagnostic": baseline_identity,
            "timing": {"total_seconds": time.perf_counter() - started},
            "limitations": [
                "Historical dev has already been consumed and is not final proof.",
                "Ranker B masked is diagnostic and cannot be promoted.",
                "No acceptor, threshold, fresh challenge or final test was opened.",
            ],
        }
        report_path = staging / "ranker_c_report.json"
        _json_dump(report_path, report)
        metadata = {
            **identity,
            "build_id": build_id,
            "verdict": verdict,
            "model_hashes": first_hashes,
            "training": first_training,
            "dependencies": _dependency_versions(),
        }
        metadata_path = staging / "ranker_c" / "metadata.json"
        _json_dump(metadata_path, metadata)
        output_paths = sorted(
            path
            for path in staging.rglob("*")
            if path.is_file() and path.name != "manifest.json"
        )
        outputs = {
            str(path.relative_to(staging)): {
                "sha256": file_sha256(path),
                "size_bytes": int(path.stat().st_size),
            }
            for path in output_paths
        }
        manifest = {
            **identity,
            "build_identity": identity,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "inputs": {
                "dataset": str(Path(dataset_dir).resolve()),
                "dataset_manifest_sha256": identity["dataset_manifest_sha256"],
                "baseline_diagnostic": baseline_identity,
                "runner_source": str(source_path),
            },
            "outputs": outputs,
            "verdict": verdict,
            "checks": checks,
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "logical_cpu_count": os.cpu_count(),
                "dependencies": _dependency_versions(),
            },
            "invariants": {
                "input_siret_or_siren_used_as_feature": False,
                "positive_injection": False,
                "acceptor_loaded_or_trained": False,
                "threshold_selected_or_applied": False,
                "fresh_challenge_opened": False,
                "final_test_opened": False,
                "five_oof_models": True,
                "one_full_fit_model": True,
                "two_repetitions_bit_exact": bool(
                    determinism["model_files_bit_exact"]
                    and determinism["scores_bit_exact"]
                    and determinism["ranks_bit_exact"]
                ),
            },
        }
        _json_dump(staging / "manifest.json", manifest)
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(repeat_staging, ignore_errors=True)
    return target


def validate_artifact(artifact_dir: Path) -> None:
    artifact_dir = Path(artifact_dir).resolve()
    manifest = json.loads(
        (artifact_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported V4.11 ranker artifact schema")
    identity = manifest.get("build_identity") or {}
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    if manifest.get("build_id") != build_id or artifact_dir.name != build_id:
        raise ValueError("V4.11 ranker artifact build identity mismatch")
    if list(manifest.get("feature_order") or []) != RANKER_C_FEATURE_ORDER:
        raise ValueError("V4.11 ranker artifact feature order changed")
    for relative, record in (manifest.get("outputs") or {}).items():
        path = artifact_dir / relative
        if file_sha256(path) != record.get("sha256"):
            raise ValueError(f"V4.11 ranker output hash mismatch: {relative}")
    report = json.loads(
        (artifact_dir / "ranker_c_report.json").read_text(encoding="utf-8")
    )
    if report.get("checks") != manifest.get("checks"):
        raise ValueError("V4.11 ranker report checks differ from manifest")
    expected = (
        "GO_RANKER_C"
        if all(bool(value) for value in report["checks"].values())
        else "PIVOT_INPUT_BLIND_RANKER"
    )
    if report.get("verdict") != expected or manifest.get("verdict") != expected:
        raise ValueError("V4.11 ranker verdict does not follow frozen gates")
    predictions = pd.read_parquet(
        artifact_dir / "predictions_ranker_c_oof_dev.parquet"
    )
    if list(predictions.columns) != PREDICTION_COLUMNS:
        raise ValueError("V4.11 ranker prediction schema changed")
    if not np.isfinite(
        predictions["ranker_score"].to_numpy(dtype=np.float64)
    ).all():
        raise ValueError("V4.11 ranker artifact has non-finite scores")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--baseline-model", type=Path, default=BASELINE_MODEL_PATH)
    parser.add_argument(
        "--baseline-metadata", type=Path, default=BASELINE_METADATA_PATH
    )
    parser.add_argument("--validate-artifact", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate_artifact is not None:
        validate_artifact(args.validate_artifact)
        print(args.validate_artifact)
        return
    missing = [
        name for name in ("dataset", "output_root") if getattr(args, name) is None
    ]
    if missing:
        raise SystemExit(f"Missing required arguments: {', '.join(missing)}")
    output = build_artifact(
        dataset_dir=args.dataset,
        output_root=args.output_root,
        baseline_model_path=args.baseline_model,
        baseline_metadata_path=args.baseline_metadata,
    )
    print(output)


if __name__ == "__main__":
    main()
