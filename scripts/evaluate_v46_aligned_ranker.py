#!/usr/bin/env python3
"""Train and gate the V4.6 ranker aligned with frozen V4.2-B pools."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import binomtest
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_v41_training_dataset import FEATURE_ORDER  # noqa: E402
from scripts.build_v46_aligned_dataset import (  # noqa: E402
    EXPECTED_CONTRACT_SHA256,
    EXPECTED_V42_CONFIG_SIGNATURE,
    candidate_content_sha256,
    validate_artifact as validate_dataset_artifact,
)
from scripts.train_v41_models import DEFAULT_RANKER_PARAMS  # noqa: E402
from scripts.train_v9_ranker import eligible_ranker_rows  # noqa: E402
from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402


SCHEMA_VERSION = "sireto-v4.6-aligned-ranker-evaluation-1"
EXPERIMENT_ID = "V46_ALIGNED_RANKER_V42B"
SEED = 42
EXPECTED_QUERY_COUNT = 7_003
EXPECTED_FIT_COUNT = 5_547
EXPECTED_DEV_COUNT = 1_456
EXPECTED_DEV_EXACT_COUNT = 1_217
EXPECTED_BASELINE_MODEL_SHA256 = (
    "720b0d2d44971477198112f03606eb303bc2f61c06bfdaf48b576b6df4551080"
)
EXPECTED_BASELINE_METADATA_SHA256 = (
    "5f5edd2a342fd4e8e2e3754bc3bca0f24b8dd93aec7f899f9d727cb54195757b"
)
EXPECTED_BASELINE_TRAIN_RETRIEVAL_SIGNATURE = (
    "189aeae6efead3595a586413871fbc388fde900d3b243d338b70d6a9de5a9db3"
)
SEGMENT_COLUMNS = ("input_siret_state", "source_segment")
BOOTSTRAP_REPLICATIONS = 10_000
TARGET_HIT_AT_1 = 0.99
MIN_NET_CORRECTIONS = 4
MAX_LARGE_SEGMENT_REGRESSION = 0.01
MAX_SEGMENT_NET_LOSSES = 2
MAX_LATENCY_RATIO = 1.25
MAX_TOTAL_SECONDS = 8 * 60 * 60


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


def _output_hash(manifest: Mapping[str, Any], filename: str) -> str:
    record = (manifest.get("outputs") or {}).get(filename)
    if isinstance(record, Mapping):
        return str(record.get("sha256") or "")
    return str(record or "")


def _dependency_versions() -> dict[str, str]:
    output: dict[str, str] = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "xgboost"):
        try:
            output[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            output[name] = "missing"
    return output


def _mac_hardware() -> dict[str, Any]:
    """Return stable Mac hardware facts without making them a model feature."""

    output: dict[str, Any] = {
        "platform_machine": platform.machine(),
        "processor": platform.processor(),
    }
    try:
        completed = subprocess.run(
            ["system_profiler", "-json", "SPHardwareDataType"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        payload = json.loads(completed.stdout)
        records = payload.get("SPHardwareDataType") or []
        if records:
            record = records[0]
            output.update(
                {
                    "model_name": record.get("machine_name"),
                    "model_identifier": record.get("machine_model"),
                    "chip": record.get("chip_type"),
                    "memory": record.get("physical_memory"),
                }
            )
    except (FileNotFoundError, subprocess.SubprocessError, json.JSONDecodeError):
        output["system_profiler_status"] = "UNAVAILABLE"
    return output


def load_aligned_dataset(
    dataset_dir: Path,
    replica_dir: Path,
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:
    """Load two independently built datasets and fail closed on any drift."""

    dataset_dir = Path(dataset_dir).resolve()
    replica_dir = Path(replica_dir).resolve()
    if dataset_dir == replica_dir:
        raise ValueError("V4.6 primary and replica dataset paths must be distinct")
    validate_dataset_artifact(dataset_dir)
    validate_dataset_artifact(replica_dir)
    primary_manifest_path = dataset_dir / "manifest.json"
    replica_manifest_path = replica_dir / "manifest.json"
    primary_manifest_sha256 = file_sha256(primary_manifest_path)
    replica_manifest_sha256 = file_sha256(replica_manifest_path)
    if primary_manifest_sha256 == replica_manifest_sha256:
        raise ValueError("V4.6 independent manifests must be distinct")
    manifest = json.loads(primary_manifest_path.read_text("utf-8"))
    replica = json.loads(replica_manifest_path.read_text("utf-8"))
    for item in (manifest, replica):
        if item.get("experiment_id") != EXPERIMENT_ID:
            raise ValueError("Unexpected V4.6 dataset experiment")
        if item.get("contract_sha256") != EXPECTED_CONTRACT_SHA256:
            raise ValueError("V4.6 dataset contract hash mismatch")
        signature = item.get("retrieval_config_signature") or (
            item.get("retrieval") or {}
        ).get("config_signature")
        if signature != EXPECTED_V42_CONFIG_SIGNATURE:
            raise ValueError("V4.6 dataset is not retrieval V4.2-B")
        if item.get("positive_injection") is not False:
            raise ValueError("V4.6 dataset suggests positive injection")
    primary_hash = str(
        (manifest.get("integrity") or {}).get("candidate_content_sha256") or ""
    )
    replica_hash = str(
        (replica.get("integrity") or {}).get("candidate_content_sha256") or ""
    )
    if not primary_hash or primary_hash != replica_hash:
        raise ValueError("Independent V4.6 candidate builds differ")
    for filename in ("queries.parquet", "labels.parquet", "split_assignments.parquet"):
        if _output_hash(manifest, filename) != _output_hash(replica, filename):
            raise ValueError(f"Independent V4.6 inputs differ: {filename}")

    queries = pd.read_parquet(dataset_dir / "queries.parquet")
    labels = pd.read_parquet(dataset_dir / "labels.parquet")
    candidates = pd.read_parquet(dataset_dir / "candidates_v42b.parquet")
    assignments = pd.read_parquet(dataset_dir / "split_assignments.parquet")
    if len(queries) != EXPECTED_QUERY_COUNT or len(labels) != EXPECTED_QUERY_COUNT:
        raise ValueError("V4.6 population must contain exactly 7,003 queries")
    if len(assignments) != EXPECTED_QUERY_COUNT:
        raise ValueError("V4.6 split assignments are incomplete")
    if list((manifest.get("feature_order") or [])) != list(FEATURE_ORDER):
        raise ValueError("V4.6 feature order differs from the frozen 64 features")
    missing = set(FEATURE_ORDER) - set(candidates)
    if missing:
        raise ValueError(f"V4.6 candidate features missing: {sorted(missing)}")
    observed_content = candidate_content_sha256(candidates)
    if observed_content != primary_hash:
        raise ValueError("V4.6 candidate content hash does not reproduce")

    query_ids = queries["query_id"].astype(str)
    if query_ids.duplicated().any():
        raise ValueError("V4.6 query IDs are not unique")
    assignments = assignments.copy()
    assignments["query_id"] = assignments["query_id"].astype(str)
    queries = queries.copy()
    labels = labels.copy()
    candidates = candidates.copy()
    for frame in (queries, labels, candidates):
        frame["query_id"] = frame["query_id"].astype(str)
    if assignments["query_id"].duplicated().any():
        raise ValueError("V4.6 assignments are not unique")
    if set(query_ids) != set(assignments["query_id"]):
        raise ValueError("V4.6 assignments do not match queries")
    counts = assignments["split"].value_counts().to_dict()
    if counts != {"fit": EXPECTED_FIT_COUNT, "dev": EXPECTED_DEV_COUNT}:
        raise ValueError(f"Unexpected V4.6 split counts: {counts}")
    dev_ids = set(assignments.loc[assignments["split"].eq("dev"), "query_id"])
    dev_exact = labels[
        labels["query_id"].isin(dev_ids)
        & labels["label_kind"].astype(str).eq("MATCH_EXACT")
    ]
    if len(dev_exact) != EXPECTED_DEV_EXACT_COUNT:
        raise ValueError("V4.6 dev must contain exactly 1,217 exact labels")

    labels = labels.merge(
        assignments[["query_id", "siren_component_id", "split", "oof_fold"]],
        on="query_id",
        how="left",
        validate="one_to_one",
    )
    candidates = candidates.merge(
        assignments[["query_id", "siren_component_id", "split", "oof_fold"]],
        on="query_id",
        how="left",
        validate="many_to_one",
    )
    replica_check = {
        "primary_manifest_sha256": primary_manifest_sha256,
        "replica_manifest_sha256": replica_manifest_sha256,
        "primary_path": str(dataset_dir),
        "replica_path": str(replica_dir),
        "candidate_content_sha256": primary_hash,
        "primary_build_seconds": float(
            (manifest.get("timing") or {}).get("total_seconds") or math.inf
        ),
        "replica_build_seconds": float(
            (replica.get("timing") or {}).get("total_seconds") or math.inf
        ),
        "independent_builds_match": True,
    }
    return manifest, queries, labels, candidates, assignments, replica_check


def load_frozen_ranker(model_dir: Path) -> tuple[xgb.XGBRanker, dict[str, Any]]:
    model_dir = Path(model_dir).resolve()
    model_path = model_dir / "ranker" / "ranker.json"
    metadata_path = model_dir / "ranker" / "metadata.json"
    if file_sha256(model_path) != EXPECTED_BASELINE_MODEL_SHA256:
        raise ValueError("Frozen ranker A model hash mismatch")
    if file_sha256(metadata_path) != EXPECTED_BASELINE_METADATA_SHA256:
        raise ValueError("Frozen ranker A metadata hash mismatch")
    metadata = json.loads(metadata_path.read_text("utf-8"))
    if list(metadata.get("feature_order") or []) != list(FEATURE_ORDER):
        raise ValueError("Frozen ranker A feature order mismatch")
    if (
        metadata.get("retrieval_signature")
        != EXPECTED_BASELINE_TRAIN_RETRIEVAL_SIGNATURE
    ):
        raise ValueError("Frozen ranker A retrieval signature mismatch")
    model = xgb.XGBRanker()
    model.load_model(model_path)
    return model, {
        "path": str(model_dir),
        "model_sha256": EXPECTED_BASELINE_MODEL_SHA256,
        "metadata_sha256": EXPECTED_BASELINE_METADATA_SHA256,
        "trained_retrieval_signature": EXPECTED_BASELINE_TRAIN_RETRIEVAL_SIGNATURE,
    }


def _new_ranker(seed: int) -> xgb.XGBRanker:
    return xgb.XGBRanker(
        **DEFAULT_RANKER_PARAMS,
        random_state=seed,
    )


def fit_ranker(
    rows: pd.DataFrame,
    *,
    seed: int,
    feature_order: Sequence[str] = FEATURE_ORDER,
) -> xgb.XGBRanker:
    ordered = rows.sort_values(
        ["query_id", "candidate_siret"], kind="stable"
    ).copy()
    if ordered.empty:
        raise ValueError("Cannot fit a ranker without candidate rows")
    groups = ordered.groupby("query_id", sort=False).size().to_numpy()
    model = _new_ranker(seed)
    model.fit(
        ordered[list(feature_order)].astype(float).to_numpy(),
        ordered["is_ground_truth"].astype(int).to_numpy(),
        group=groups,
        verbose=False,
    )
    return model


def rank_scored_rows(
    rows: pd.DataFrame,
    scores: Sequence[float],
    *,
    origin: str,
    fold: int | None,
) -> pd.DataFrame:
    """Apply the sole V4.6 tie-break: score, retrieval rank, then SIRET."""

    output = rows[
        [
            "query_id",
            "candidate_siret",
            "candidate_siren",
            "retrieval_rank",
            "is_ground_truth",
        ]
    ].copy()
    output["score"] = np.asarray(scores, dtype=np.float64)
    output["prediction_origin"] = origin
    output["fold"] = fold
    output = output.sort_values(
        ["query_id", "score", "retrieval_rank", "candidate_siret"],
        ascending=[True, False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    output["rank"] = output.groupby("query_id", sort=False).cumcount() + 1
    return output


def score_rows(
    model: xgb.XGBRanker,
    rows: pd.DataFrame,
    *,
    origin: str,
    fold: int | None,
) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(
            columns=[
                "query_id",
                "candidate_siret",
                "candidate_siren",
                "retrieval_rank",
                "is_ground_truth",
                "score",
                "prediction_origin",
                "fold",
                "rank",
            ]
        )
    scores = model.predict(rows[list(FEATURE_ORDER)].astype(float).to_numpy())
    return rank_scored_rows(rows, scores, origin=origin, fold=fold)


def append_missing_prediction_sentinels(
    predictions: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    include_splits: set[str],
    origin_by_split: Mapping[str, str],
) -> pd.DataFrame:
    """Represent zero-candidate queries without inventing a SIRET."""

    expected = assignments[assignments["split"].isin(include_splits)].copy()
    expected["query_id"] = expected["query_id"].astype(str)
    observed = set(predictions["query_id"].astype(str))
    missing = expected[~expected["query_id"].isin(observed)]
    sentinel_rows = [
        {
            "query_id": str(row.query_id),
            "candidate_siret": None,
            "candidate_siren": None,
            "retrieval_rank": 0,
            "is_ground_truth": 0,
            "score": -math.inf,
            "prediction_origin": origin_by_split[str(row.split)],
            "fold": int(row.oof_fold) if str(row.split) == "fit" else None,
            "rank": 1,
        }
        for row in missing.itertuples(index=False)
    ]
    output = (
        pd.concat([predictions, pd.DataFrame(sentinel_rows)], ignore_index=True)
        if sentinel_rows
        else predictions.copy()
    )
    top1 = output[output["rank"].eq(1)]
    top1_counts = top1.groupby("query_id").size()
    if len(top1_counts) != len(expected) or int(top1_counts.max()) != 1:
        raise ValueError("V4.6 predictions do not contain exactly one top-1 per query")
    if set(top1_counts.index.astype(str)) != set(expected["query_id"]):
        raise ValueError("V4.6 top-1 prediction coverage differs from assignments")
    return output


def train_aligned_ranker(
    candidates: pd.DataFrame,
    labels: pd.DataFrame,
) -> tuple[xgb.XGBRanker, pd.DataFrame, dict[str, Any]]:
    fit_labels = labels[labels["split"].eq("fit")].copy()
    fit_candidates = candidates[candidates["split"].eq("fit")].copy()
    prediction_parts: list[pd.DataFrame] = []
    exclusion_counts: dict[str, int] = {}
    for fold in range(5):
        validation_ids = set(
            fit_labels.loc[fit_labels["oof_fold"].eq(fold), "query_id"]
        )
        train_labels = fit_labels[fit_labels["oof_fold"].ne(fold)]
        train_ids = set(train_labels["query_id"])
        scoped = fit_candidates[fit_candidates["query_id"].isin(train_ids)]
        eligible = eligible_ranker_rows(scoped, train_labels)
        exact_count = int(train_labels["label_kind"].eq("MATCH_EXACT").sum())
        eligible_query_count = int(eligible["query_id"].nunique())
        exclusion_counts[f"fold_{fold}"] = exact_count - eligible_query_count
        model = fit_ranker(eligible, seed=SEED + fold)
        prediction_parts.append(
            score_rows(
                model,
                fit_candidates[fit_candidates["query_id"].isin(validation_ids)],
                origin="v46_b_oof",
                fold=fold,
            )
        )
    final_rows = eligible_ranker_rows(fit_candidates, fit_labels)
    exact_fit_count = int(fit_labels["label_kind"].eq("MATCH_EXACT").sum())
    exclusion_counts["final_fit"] = (
        exact_fit_count - int(final_rows["query_id"].nunique())
    )
    final_model = fit_ranker(final_rows, seed=SEED)
    prediction_parts.append(
        score_rows(
            final_model,
            candidates[candidates["split"].eq("dev")],
            origin="v46_b_dev",
            fold=None,
        )
    )
    predictions = pd.concat(prediction_parts, ignore_index=True)
    predictions = append_missing_prediction_sentinels(
        predictions,
        labels[["query_id", "split", "oof_fold"]].drop_duplicates("query_id"),
        include_splits={"fit", "dev"},
        origin_by_split={"fit": "v46_b_oof", "dev": "v46_b_dev"},
    )
    return final_model, predictions, {
        "exact_queries_excluded_missing_positive": exclusion_counts,
        "fit_candidate_rows": int(len(final_rows)),
        "fit_query_count": int(final_rows["query_id"].nunique()),
    }


def exact_hits(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    exact = labels[
        labels["split"].eq("dev")
        & labels["label_kind"].astype(str).eq("MATCH_EXACT")
    ][["query_id", "ground_truth_siret", "ground_truth_siren"]].copy()
    exact["query_id"] = exact["query_id"].astype(str)
    top1 = predictions[predictions["rank"].eq(1)][
        ["query_id", "candidate_siret", "candidate_siren", "score"]
    ].copy()
    evaluated = exact.merge(top1, on="query_id", how="left", validate="one_to_one")
    evaluated["siret_hit"] = evaluated["candidate_siret"].fillna("").astype(str).eq(
        evaluated["ground_truth_siret"].fillna("").astype(str)
    )
    expected_siren = evaluated["ground_truth_siren"].fillna("").astype(str)
    expected_siren = expected_siren.where(
        expected_siren.str.len().eq(9),
        evaluated["ground_truth_siret"].fillna("").astype(str).str[:9],
    )
    evaluated["siren_hit"] = evaluated["candidate_siren"].fillna("").astype(str).eq(
        expected_siren
    )
    return evaluated


def paired_statistics(
    a_hits: Sequence[bool],
    b_hits: Sequence[bool],
) -> dict[str, Any]:
    a = np.asarray(a_hits, dtype=bool)
    b = np.asarray(b_hits, dtype=bool)
    if len(a) != EXPECTED_DEV_EXACT_COUNT or len(b) != len(a):
        raise ValueError("Paired V4.6 statistics require exactly 1,217 cases")
    a_only = int(np.sum(a & ~b))
    b_only = int(np.sum(~a & b))
    both = int(np.sum(a & b))
    neither = int(np.sum(~a & ~b))
    delta = b.astype(np.int8) - a.astype(np.int8)
    rng = np.random.default_rng(SEED)
    bootstrap = np.empty(BOOTSTRAP_REPLICATIONS, dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATIONS):
        bootstrap[index] = float(
            delta[rng.integers(0, len(delta), size=len(delta))].mean()
        )
    discordant = a_only + b_only
    p_value = (
        1.0
        if discordant == 0
        else float(
            binomtest(
                min(a_only, b_only),
                n=discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )
    )
    return {
        "both_correct": both,
        "a_correct_b_wrong": a_only,
        "a_wrong_b_correct": b_only,
        "both_wrong": neither,
        "net_corrections": b_only - a_only,
        "delta_absolute": float(delta.mean()),
        "bootstrap_95_low": float(np.percentile(bootstrap, 2.5)),
        "bootstrap_95_high": float(np.percentile(bootstrap, 97.5)),
        "mcnemar_exact_two_sided_p": p_value,
    }


def segment_metrics(
    queries: pd.DataFrame,
    a: pd.DataFrame,
    b: pd.DataFrame,
) -> tuple[dict[str, Any], bool, bool]:
    rows = a[["query_id", "siret_hit"]].rename(
        columns={"siret_hit": "a_hit"}
    ).merge(
        b[["query_id", "siret_hit"]].rename(columns={"siret_hit": "b_hit"}),
        on="query_id",
        validate="one_to_one",
    )
    rows = rows.merge(
        queries[["query_id", *SEGMENT_COLUMNS]],
        on="query_id",
        how="left",
        validate="one_to_one",
    )
    result: dict[str, Any] = {}
    large_ok = True
    family_ok = True
    global_count = int(len(rows))
    global_a = int(rows["a_hit"].sum())
    global_b = int(rows["b_hit"].sum())
    result["GLOBAL"] = {
        "count": global_count,
        "a_successes": global_a,
        "b_successes": global_b,
        "a_hit_at_1": global_a / global_count,
        "b_hit_at_1": global_b / global_count,
        "delta": (global_b - global_a) / global_count,
        "net_loss": global_a - global_b,
    }
    for column in SEGMENT_COLUMNS:
        for value, group in rows.groupby(column, dropna=False):
            count = int(len(group))
            a_success = int(group["a_hit"].sum())
            b_success = int(group["b_hit"].sum())
            delta = (b_success - a_success) / count
            net_loss = a_success - b_success
            if count >= 100 and delta < -MAX_LARGE_SEGMENT_REGRESSION:
                large_ok = False
            if net_loss > MAX_SEGMENT_NET_LOSSES:
                family_ok = False
            result[f"{column}={value}"] = {
                "count": count,
                "a_successes": a_success,
                "b_successes": b_success,
                "a_hit_at_1": a_success / count,
                "b_hit_at_1": b_success / count,
                "delta": delta,
                "net_loss": net_loss,
            }
    return result, large_ok, family_ok


def validate_repeat_predictions(
    first: pd.DataFrame,
    second: pd.DataFrame,
) -> dict[str, Any]:
    keys = ["query_id", "candidate_siret", "prediction_origin", "fold"]
    left = first.sort_values(keys, kind="stable").reset_index(drop=True)
    right = second.sort_values(keys, kind="stable").reset_index(drop=True)
    if not left[keys].fillna("").equals(right[keys].fillna("")):
        raise ValueError("Repeated V4.6 prediction identities differ")
    max_score_delta = float(
        np.max(np.abs(left["score"].to_numpy() - right["score"].to_numpy()))
    )
    top1_left = set(
        map(tuple, left.loc[left["rank"].eq(1), ["query_id", "candidate_siret"]].values)
    )
    top1_right = set(
        map(tuple, right.loc[right["rank"].eq(1), ["query_id", "candidate_siret"]].values)
    )
    return {
        "top1_identical": top1_left == top1_right,
        "max_score_absolute_delta": max_score_delta,
        "scores_within_1e_12": max_score_delta <= 1e-12,
    }


def query_latency(
    model: xgb.XGBRanker,
    dev_candidates: pd.DataFrame,
    dev_query_ids: Sequence[str],
) -> dict[str, Any]:
    groups = {
        str(query_id): group
        for query_id, group in dev_candidates.sort_values(
            ["query_id", "candidate_siret"], kind="stable"
        ).groupby("query_id", sort=False)
    }
    ordered_ids = [str(value) for value in dev_query_ids]
    if len(ordered_ids) != EXPECTED_DEV_COUNT or len(set(ordered_ids)) != len(
        ordered_ids
    ):
        raise ValueError("Latency requires exactly 1,456 unique dev queries")
    unknown = set(groups) - set(ordered_ids)
    if unknown:
        raise ValueError("Latency candidates contain an unknown dev query")
    warmup = next((group for group in groups.values() if not group.empty), None)
    if warmup is None:
        raise ValueError("Latency cannot be measured without any candidate")
    model.predict(warmup[list(FEATURE_ORDER)].astype(float).to_numpy())
    durations: list[float] = []
    for _ in range(3):
        for query_id in ordered_ids:
            group = groups.get(query_id)
            started = time.perf_counter_ns()
            if group is not None:
                matrix = group[list(FEATURE_ORDER)].astype(float).to_numpy()
                model.predict(matrix)
            durations.append((time.perf_counter_ns() - started) / 1_000_000.0)
    return {
        "unit": "milliseconds_per_query",
        "warmup_query_count": 1,
        "repetitions": 3,
        "measurement_count": len(durations),
        "zero_candidate_query_count": len(ordered_ids) - len(groups),
        "p50": float(np.percentile(durations, 50)),
        "p95": float(np.percentile(durations, 95)),
    }


def evaluate_gates(
    *,
    a: pd.DataFrame,
    b: pd.DataFrame,
    paired: Mapping[str, Any],
    segment_large_ok: bool,
    segment_family_ok: bool,
    deterministic: Mapping[str, Any],
    latency_a: Mapping[str, Any],
    latency_b: Mapping[str, Any],
    total_seconds: float,
    integrity_ok: bool,
) -> tuple[dict[str, bool], str]:
    a_siret = float(a["siret_hit"].mean())
    b_siret = float(b["siret_hit"].mean())
    a_siren = float(a["siren_hit"].mean())
    b_siren = float(b["siren_hit"].mean())
    checks = {
        "b_siret_hit_at_1_gte_0_99": b_siret >= TARGET_HIT_AT_1,
        "net_corrections_gte_4": int(paired["net_corrections"])
        >= MIN_NET_CORRECTIONS,
        "bootstrap_low_strictly_positive": float(
            paired["bootstrap_95_low"]
        )
        > 0.0,
        "mcnemar_p_lt_0_05": float(
            paired["mcnemar_exact_two_sided_p"]
        )
        < 0.05,
        "no_global_siren_regression": b_siren >= a_siren,
        "large_segments_within_1pp": bool(segment_large_ok),
        "no_segment_loses_more_than_two_net": bool(segment_family_ok),
        "repeated_top1_identical": bool(deterministic["top1_identical"]),
        "repeated_scores_within_1e_12": bool(
            deterministic["scores_within_1e_12"]
        ),
        "b_latency_p95_lte_1_25x_a": float(latency_b["p95"])
        <= MAX_LATENCY_RATIO * float(latency_a["p95"]),
        "local_total_under_eight_hours": total_seconds < MAX_TOTAL_SECONDS,
        "input_integrity": bool(integrity_ok),
    }
    return checks, "GO_ALIGN_RANKER" if all(checks.values()) else "KEEP_RANKER_A"


def build_artifact(
    *,
    dataset_dir: Path,
    replica_dataset_dir: Path,
    baseline_model_dir: Path,
    output_root: Path,
) -> Path:
    output_root = _external_path(output_root, name="output_root")
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    (
        dataset_manifest,
        queries,
        labels,
        candidates,
        assignments,
        replica_check,
    ) = load_aligned_dataset(dataset_dir, replica_dataset_dir)
    baseline, baseline_identity = load_frozen_ranker(baseline_model_dir)
    dev_candidates = candidates[candidates["split"].eq("dev")].copy()
    baseline_scoring_started = time.perf_counter()
    baseline_predictions = score_rows(
        baseline, dev_candidates, origin="v46_a_dev", fold=None
    )
    baseline_predictions = append_missing_prediction_sentinels(
        baseline_predictions,
        assignments,
        include_splits={"dev"},
        origin_by_split={"dev": "v46_a_dev"},
    )
    baseline_scoring_seconds = time.perf_counter() - baseline_scoring_started

    aligned_training_started = time.perf_counter()
    aligned, aligned_predictions, training = train_aligned_ranker(candidates, labels)
    aligned_training_seconds = time.perf_counter() - aligned_training_started
    repeated_training_started = time.perf_counter()
    repeated, repeated_predictions, repeated_training = train_aligned_ranker(
        candidates, labels
    )
    repeated_training_seconds = time.perf_counter() - repeated_training_started
    deterministic = validate_repeat_predictions(
        aligned_predictions, repeated_predictions
    )
    if training != repeated_training:
        raise ValueError("Repeated V4.6 training diagnostics differ")
    a_hits = exact_hits(baseline_predictions, labels).sort_values("query_id")
    b_hits = exact_hits(
        aligned_predictions[aligned_predictions["prediction_origin"].eq("v46_b_dev")],
        labels,
    ).sort_values("query_id")
    if not a_hits["query_id"].reset_index(drop=True).equals(
        b_hits["query_id"].reset_index(drop=True)
    ):
        raise ValueError("A/B exact dev populations differ")
    paired = paired_statistics(a_hits["siret_hit"], b_hits["siret_hit"])
    segments, segment_large_ok, segment_family_ok = segment_metrics(
        queries, a_hits, b_hits
    )
    dev_query_ids = assignments.loc[
        assignments["split"].eq("dev"), "query_id"
    ].sort_values(kind="stable")
    latency_a_started = time.perf_counter()
    latency_a = query_latency(baseline, dev_candidates, dev_query_ids)
    latency_a_wall_seconds = time.perf_counter() - latency_a_started
    latency_b_started = time.perf_counter()
    latency_b = query_latency(aligned, dev_candidates, dev_query_ids)
    latency_b_wall_seconds = time.perf_counter() - latency_b_started
    elapsed_before_gate = time.perf_counter() - started
    dataset_seconds = float(replica_check["primary_build_seconds"])
    replica_dataset_seconds = float(replica_check["replica_build_seconds"])
    total_seconds = dataset_seconds + replica_dataset_seconds + elapsed_before_gate
    checks, verdict = evaluate_gates(
        a=a_hits,
        b=b_hits,
        paired=paired,
        segment_large_ok=segment_large_ok,
        segment_family_ok=segment_family_ok,
        deterministic=deterministic,
        latency_a=latency_a,
        latency_b=latency_b,
        total_seconds=total_seconds,
        integrity_ok=bool(replica_check["independent_builds_match"]),
    )

    source_path = Path(__file__).resolve()
    identity = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "contract_sha256": EXPECTED_CONTRACT_SHA256,
        "dataset_manifest_sha256": file_sha256(Path(dataset_dir) / "manifest.json"),
        "replica_manifest_sha256": file_sha256(
            Path(replica_dataset_dir) / "manifest.json"
        ),
        "candidate_content_sha256": replica_check["candidate_content_sha256"],
        "baseline_model_sha256": EXPECTED_BASELINE_MODEL_SHA256,
        "evaluator_source_sha256": file_sha256(source_path),
        "feature_order": list(FEATURE_ORDER),
        "seed": SEED,
        "ranker_params": {**DEFAULT_RANKER_PARAMS, "random_state": SEED},
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    target = output_root / build_id
    if target.exists():
        raise FileExistsError(f"Immutable V4.6 evaluation exists: {target}")
    staging = Path(tempfile.mkdtemp(prefix=f".{build_id}.tmp-", dir=output_root))
    try:
        model_dir = staging / "ranker_b"
        model_dir.mkdir()
        model_path = model_dir / "ranker.json"
        aligned.save_model(model_path)
        baseline_path = staging / "predictions_a_dev.parquet"
        aligned_path = staging / "predictions_b_oof_dev.parquet"
        paired_path = staging / "paired_dev.parquet"
        corrections_path = staging / "corrections_a_to_b.parquet"
        regressions_path = staging / "regressions_a_to_b.parquet"
        baseline_predictions.to_parquet(baseline_path, index=False)
        aligned_predictions.to_parquet(aligned_path, index=False)
        paired_rows = a_hits[
            ["query_id", "ground_truth_siret", "candidate_siret", "siret_hit", "siren_hit"]
        ].rename(
            columns={
                "candidate_siret": "a_candidate_siret",
                "siret_hit": "a_siret_hit",
                "siren_hit": "a_siren_hit",
            }
        ).merge(
            b_hits[
                ["query_id", "candidate_siret", "siret_hit", "siren_hit"]
            ].rename(
                columns={
                    "candidate_siret": "b_candidate_siret",
                    "siret_hit": "b_siret_hit",
                    "siren_hit": "b_siren_hit",
                }
            ),
            on="query_id",
            validate="one_to_one",
        )
        paired_rows.to_parquet(paired_path, index=False)
        paired_rows[
            ~paired_rows["a_siret_hit"].astype(bool)
            & paired_rows["b_siret_hit"].astype(bool)
        ].to_parquet(corrections_path, index=False)
        paired_rows[
            paired_rows["a_siret_hit"].astype(bool)
            & ~paired_rows["b_siret_hit"].astype(bool)
        ].to_parquet(regressions_path, index=False)
        report = {
            "schema_version": SCHEMA_VERSION,
            "build_id": build_id,
            "verdict": verdict,
            "training_authorized": True,
            "promotion_authorized": verdict == "GO_ALIGN_RANKER",
            "metrics": {
                "dev_exact_count": EXPECTED_DEV_EXACT_COUNT,
                "a": {
                    "siret_successes": int(a_hits["siret_hit"].sum()),
                    "siret_hit_at_1": float(a_hits["siret_hit"].mean()),
                    "siren_successes": int(a_hits["siren_hit"].sum()),
                    "siren_hit_at_1": float(a_hits["siren_hit"].mean()),
                },
                "b": {
                    "siret_successes": int(b_hits["siret_hit"].sum()),
                    "siret_hit_at_1": float(b_hits["siret_hit"].mean()),
                    "siren_successes": int(b_hits["siren_hit"].sum()),
                    "siren_hit_at_1": float(b_hits["siren_hit"].mean()),
                },
                "paired": paired,
                "segments": segments,
                "latency_a": latency_a,
                "latency_b": latency_b,
                "timing": {
                    "dataset_primary_seconds": dataset_seconds,
                    "dataset_replica_seconds": replica_dataset_seconds,
                    "baseline_batch_scoring_seconds": baseline_scoring_seconds,
                    "aligned_training_oof_and_final_seconds": (
                        aligned_training_seconds
                    ),
                    "determinism_repeat_training_seconds": (
                        repeated_training_seconds
                    ),
                    "latency_a_wall_seconds": latency_a_wall_seconds,
                    "latency_b_wall_seconds": latency_b_wall_seconds,
                    "evaluation_seconds": elapsed_before_gate,
                    "gate_total_seconds": total_seconds,
                },
            },
            "checks": checks,
            "determinism": deterministic,
            "training": training,
            "replica_check": replica_check,
            "paired_case_lists": {
                "corrections_filename": corrections_path.name,
                "correction_count": int(
                    (
                        ~paired_rows["a_siret_hit"].astype(bool)
                        & paired_rows["b_siret_hit"].astype(bool)
                    ).sum()
                ),
                "regressions_filename": regressions_path.name,
                "regression_count": int(
                    (
                        paired_rows["a_siret_hit"].astype(bool)
                        & ~paired_rows["b_siret_hit"].astype(bool)
                    ).sum()
                ),
            },
            "limitations": [
                "The historical dev was previously used to select ranker A.",
                "This paired gate is not an independent production validation.",
                "No acceptor, threshold, V4.4 label, random label or final test was used.",
            ],
        }
        report_path = staging / "evaluation_report.json"
        _json_dump(report_path, report)
        metadata = {
            **identity,
            "build_id": build_id,
            "model_sha256": file_sha256(model_path),
            "verdict": verdict,
            "promoted": verdict == "GO_ALIGN_RANKER",
            "retrieval_signature": EXPECTED_V42_CONFIG_SIGNATURE,
            "feature_order": list(FEATURE_ORDER),
            "dependencies": _dependency_versions(),
            "hardware": _mac_hardware(),
            "training": training,
        }
        metadata_path = model_dir / "metadata.json"
        _json_dump(metadata_path, metadata)
        outputs = {
            str(path.relative_to(staging)): {
                "sha256": file_sha256(path),
                "size_bytes": int(path.stat().st_size),
            }
            for path in (
                model_path,
                metadata_path,
                baseline_path,
                aligned_path,
                paired_path,
                corrections_path,
                regressions_path,
                report_path,
            )
        }
        manifest = {
            **identity,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "build_identity": identity,
            "inputs": {
                "dataset": str(Path(dataset_dir).resolve()),
                "replica_dataset": str(Path(replica_dataset_dir).resolve()),
                "baseline": baseline_identity,
                "evaluator_source": str(source_path),
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
                "hardware": _mac_hardware(),
            },
            "invariants": {
                "acceptor_loaded_or_trained": False,
                "threshold_selected_or_applied": False,
                "v44_or_random_labels_read": False,
                "final_test_read": False,
                "positive_injection": False,
                "candidate_ceiling": 100,
                "paired_dev_only_selection": True,
            },
        }
        _json_dump(staging / "manifest.json", manifest)
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def validate_artifact(artifact_dir: Path) -> None:
    artifact_dir = Path(artifact_dir)
    manifest = json.loads((artifact_dir / "manifest.json").read_text("utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported V4.6 evaluation schema")
    identity = manifest.get("build_identity") or {}
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    if build_id != manifest.get("build_id") or artifact_dir.name != build_id:
        raise ValueError("V4.6 evaluation build identity mismatch")
    for relative, record in (manifest.get("outputs") or {}).items():
        path = artifact_dir / relative
        if file_sha256(path) != record.get("sha256"):
            raise ValueError(f"V4.6 evaluation output hash mismatch: {relative}")
    report = json.loads((artifact_dir / "evaluation_report.json").read_text("utf-8"))
    if report.get("verdict") != manifest.get("verdict"):
        raise ValueError("V4.6 evaluation verdict mismatch")
    if report.get("checks") != manifest.get("checks"):
        raise ValueError("V4.6 evaluation checks mismatch")
    expected = (
        "GO_ALIGN_RANKER"
        if all(bool(value) for value in report["checks"].values())
        else "KEEP_RANKER_A"
    )
    if report.get("verdict") != expected:
        raise ValueError("V4.6 evaluation verdict does not follow its gates")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--replica-dataset", type=Path)
    parser.add_argument("--baseline-model-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--validate-artifact", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate_artifact:
        validate_artifact(args.validate_artifact)
        print(args.validate_artifact)
        return
    required = ("dataset", "replica_dataset", "baseline_model_dir", "output_root")
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise SystemExit(f"Missing required arguments: {', '.join(missing)}")
    output = build_artifact(
        dataset_dir=args.dataset,
        replica_dataset_dir=args.replica_dataset,
        baseline_model_dir=args.baseline_model_dir,
        output_root=args.output_root,
    )
    print(output)


if __name__ == "__main__":
    main()
