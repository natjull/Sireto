"""Parent-side parity and latency checks for persistent V4.12 workers."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .v412_service_bundle import (
    EXPECTED_FILES,
    _capture_exact,
    _identity,
    _path_chain,
)
from .v412_service_run import TIMING_COLUMNS


REFERENCE_ROOT = Path(
    "/Volumes/CATNAT_DATA/SIRETO_RECALL100/references/"
    "v4_12_service_parity/b4b7fef24c5e7036"
)
REFERENCE_MANIFEST_SHA256 = (
    "cbcb3303107cd00f895561b49b8ad3a26e5c8e3df8a07777817e7a6ed97f2340"
)
MAX_RSS_BYTES = 8 * 1024**3
BASE_FILES = frozenset(
    {
        "candidates_features.parquet",
        "ranker.parquet",
        "scenes.parquet",
        "acceptor.parquet",
        "timings.parquet",
    }
)
V412G_FILES = BASE_FILES | {
    "query_evidence.parquet",
    "candidate_evidence.parquet",
    "guard.parquet",
}


def nearest_rank(values: list[int], percentile: float) -> int:
    if not values or not 0.0 < percentile <= 1.0:
        raise ValueError(
            "STOP_V412_SERVICE_INTEGRITY: invalid percentile input"
        )
    ordered = sorted(int(value) for value in values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def _json(payload: bytes, label: str) -> dict[str, Any]:
    def object_pairs(pairs):
        parsed_object = {}
        for key, value in pairs:
            if key in parsed_object:
                raise ValueError(
                    "STOP_V412_SERVICE_INTEGRITY: "
                    f"duplicate key in {label}"
                )
            parsed_object[key] = value
        return parsed_object

    try:
        parsed = json.loads(payload, object_pairs_hook=object_pairs)
    except Exception as exc:
        raise ValueError(
            f"STOP_V412_SERVICE_INTEGRITY: invalid {label}"
        ) from exc
    if type(parsed) is not dict:
        raise ValueError(
            f"STOP_V412_SERVICE_INTEGRITY: invalid {label}"
        )
    return parsed


def _parquet(
    path: Path,
    sha256: str,
    *,
    expected_size: int | None = None,
) -> pd.DataFrame:
    payload = _capture_exact(path, sha256)
    if expected_size is not None and len(payload) != expected_size:
        raise ValueError(
            "STOP_V412_SERVICE_INTEGRITY: worker file size changed"
        )
    return pq.read_table(pa.BufferReader(payload)).to_pandas()


def _capture_untrusted_regular(path: Path) -> bytes:
    """Capture a not-yet-hashed manifest through one stable regular-file FD."""
    chain_before = _path_chain(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(
                "STOP_V412_SERVICE_INTEGRITY: manifest is not regular"
            )
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        payload = b"".join(chunks)
        current = os.stat(path, follow_symlinks=False)
        if (
            _identity(before) != _identity(after)
            or _identity(before) != _identity(current)
            or len(payload) != before.st_size
            or _path_chain(path) != chain_before
        ):
            raise ValueError(
                "STOP_V412_SERVICE_INTEGRITY: manifest changed while read"
            )
        return payload
    finally:
        os.close(descriptor)


def _load_reference() -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    manifest = _json(
        _capture_exact(
            REFERENCE_ROOT / "manifest.json",
            REFERENCE_MANIFEST_SHA256,
        ),
        "reference manifest",
    )
    if (
        manifest.get("schema_version")
        != "sireto-v4.12-service-parity-reference-2"
        or manifest.get("build_id") != "b4b7fef24c5e7036"
        or manifest.get("expected_query_count") != 1456
        or manifest.get("expected_candidate_count") != 145236
        or manifest.get("max_candidates") != 100
    ):
        raise ValueError(
            "STOP_V412_SERVICE_INTEGRITY: reference manifest changed"
        )
    outputs = manifest.get("outputs")
    if type(outputs) is not dict:
        raise ValueError(
            "STOP_V412_SERVICE_INTEGRITY: reference outputs missing"
        )
    frames = {
        name: _parquet(
            REFERENCE_ROOT / name,
            str(record["sha256"]),
        )
        for name, record in outputs.items()
        if name.endswith(".parquet")
    }
    return manifest, frames


def _load_worker(
    output: Path,
    *,
    expected_mode: str,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    manifest_path = output / "manifest.json"
    payload = _capture_untrusted_regular(manifest_path)
    manifest = _json(payload, "worker manifest")
    files = manifest.get("files")
    if type(files) is not dict:
        raise ValueError(
            "STOP_V412_SERVICE_INTEGRITY: worker files missing"
        )
    expected_files = BASE_FILES if expected_mode == "v411" else V412G_FILES
    if set(files) != expected_files:
        raise ValueError(
            "STOP_V412_SERVICE_INTEGRITY: worker file set changed"
        )
    frames: dict[str, pd.DataFrame] = {}
    for name, record in files.items():
        if (
            type(record) is not dict
            or set(record)
            != {"sha256", "size_bytes", "row_count", "columns"}
            or type(record.get("sha256")) is not str
            or len(record["sha256"]) != 64
            or type(record.get("size_bytes")) is not int
            or record["size_bytes"] < 0
            or type(record.get("row_count")) is not int
            or record["row_count"] < 0
            or type(record.get("columns")) is not list
        ):
            raise ValueError(
                "STOP_V412_SERVICE_INTEGRITY: invalid worker file record"
            )
        frame = _parquet(
            output / name,
            record["sha256"],
            expected_size=record["size_bytes"],
        )
        if (
            len(frame) != record["row_count"]
            or list(frame.columns) != record["columns"]
        ):
            raise ValueError(
                "STOP_V412_SERVICE_INTEGRITY: worker file metadata changed"
            )
        frames[name] = frame
    if set(output.iterdir()) != {
        manifest_path,
        *(output / name for name in files),
    }:
        raise ValueError(
            "STOP_V412_SERVICE_INTEGRITY: worker output tree changed"
        )
    return manifest, frames


def _assert_exact(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    label: str,
) -> None:
    try:
        pd.testing.assert_frame_equal(
            observed.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=True,
            check_exact=True,
        )
    except AssertionError as exc:
        raise ValueError(
            f"STOP_V412_SERVICE_INTEGRITY: {label} parity changed: {exc}"
        ) from exc


def _latency_summary(timings: pd.DataFrame) -> dict[str, int]:
    values = timings["total_wall_ns"].astype(int).tolist()
    evidence = timings["evidence_guard_ns"].astype(int).tolist()
    summary = {
        "count": len(values),
        "p50_ns": nearest_rank(values, 0.50),
        "p95_ns": nearest_rank(values, 0.95),
        "p99_ns": nearest_rank(values, 0.99),
        "maximum_ns": max(values),
        "evidence_guard_p50_ns": nearest_rank(evidence, 0.50),
        "evidence_guard_p95_ns": nearest_rank(evidence, 0.95),
        "evidence_guard_p99_ns": nearest_rank(evidence, 0.99),
        "evidence_guard_maximum_ns": max(evidence),
    }
    for column in TIMING_COLUMNS[1:-2]:
        column_values = timings[column].astype(int).tolist()
        summary[f"{column}_p50_ns"] = nearest_rank(column_values, 0.50)
        summary[f"{column}_p95_ns"] = nearest_rank(column_values, 0.95)
        summary[f"{column}_p99_ns"] = nearest_rank(column_values, 0.99)
        summary[f"{column}_maximum_ns"] = max(column_values)
    return summary


def evaluate_paired_gate(
    v411: Mapping[str, Any],
    v412g: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify already integrity/parity-validated gate worker summaries."""
    if (
        v411.get("mode") != "v411"
        or v412g.get("mode") != "v412g"
        or v411.get("query_count") != v412g.get("query_count")
        or v411.get("query_count") != 1456
        or v411.get("pid") == v412g.get("pid")
    ):
        raise ValueError(
            "STOP_V412_SERVICE_INTEGRITY: paired worker identity changed"
        )
    persistence_reasons: list[str] = []
    for label, summary in (("v411", v411), ("v412g", v412g)):
        counters = summary["counters"]
        for name in (
            "lookup_missing_count",
            "sealed_key_miss_count",
            "cache_rebuild_count",
            "cache_write_count",
        ):
            if counters[name] != 0:
                persistence_reasons.append(f"{label}:{name}")
        for name in ("cache_rebuild_api_absent", "cache_write_api_absent"):
            if counters.get(name) is not True:
                persistence_reasons.append(f"{label}:{name}")
        if summary["model_load_count"] != 1:
            persistence_reasons.append(f"{label}:model_load_count")
        if summary["store_load_count"] != 1:
            persistence_reasons.append(f"{label}:store_load_count")

    latency_v411 = v411["latency"]
    latency_v412g = v412g["latency"]
    cost_reasons: list[str] = []
    if v411["peak_rss_bytes"] >= MAX_RSS_BYTES:
        cost_reasons.append("v411:peak_rss")
    if v412g["peak_rss_bytes"] >= MAX_RSS_BYTES:
        cost_reasons.append("v412g:peak_rss")
    if latency_v412g["p95_ns"] >= 2 * latency_v411["p95_ns"]:
        cost_reasons.append("full_latency_ratio")
    v411_totals = v411["_total_wall_ns"]
    v412g_totals = v412g["_total_wall_ns"]
    if len(v411_totals) != 1456 or len(v412g_totals) != 1456:
        raise ValueError(
            "STOP_V412_SERVICE_INTEGRITY: paired timing population changed"
        )
    paired_delta = [
        int(after) - int(before)
        for before, after in zip(v411_totals, v412g_totals, strict=True)
    ]
    delta_summary = {
        "p50_ns": nearest_rank(paired_delta, 0.50),
        "p95_ns": nearest_rank(paired_delta, 0.95),
        "p99_ns": nearest_rank(paired_delta, 0.99),
        "maximum_ns": max(paired_delta),
        "minimum_ns": min(paired_delta),
    }
    verdict = (
        "GO_V412_SERVICE_FREEZE"
        if not persistence_reasons and not cost_reasons
        else "PIVOT_V412_SERVICE_IMPLEMENTATION"
    )
    return {
        "verdict": verdict,
        "persistence_reasons": persistence_reasons,
        "cost_reasons": cost_reasons,
        "p95_ratio": (
            latency_v412g["p95_ns"] / latency_v411["p95_ns"]
        ),
        "paired_total_delta": delta_summary,
    }


def validate_worker_output(
    output: Path,
    *,
    expected_mode: str,
    expected_phase: str,
    expected_nonce: str,
    expected_pid: int,
    expected_parent_pid: int,
    expected_execution_lock_sha256: str,
) -> dict[str, Any]:
    if expected_mode not in {"v411", "v412g"}:
        raise ValueError(
            "STOP_V412_SERVICE_INTEGRITY: invalid expected worker mode"
        )
    if (
        expected_phase not in {"diagnostic", "gate"}
        or type(expected_pid) is not int
        or expected_pid <= 0
        or type(expected_parent_pid) is not int
        or expected_parent_pid <= 0
        or len(expected_nonce) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected_nonce
        )
        or len(expected_execution_lock_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected_execution_lock_sha256
        )
    ):
        raise ValueError(
            "STOP_V412_SERVICE_INTEGRITY: invalid parent attestation"
        )
    reference_manifest, reference = _load_reference()
    manifest, observed = _load_worker(
        output,
        expected_mode=expected_mode,
    )
    if (
        manifest.get("schema_version")
        != "sireto-v4.12-persistent-service-run-1"
        or manifest.get("mode") != expected_mode
        or manifest.get("phase") != expected_phase
        or manifest.get("run_nonce") != expected_nonce
        or manifest.get("execution_lock_sha256")
        != expected_execution_lock_sha256
        or manifest.get("query_count") != 1456
        or manifest.get("candidate_count") != 145236
        or manifest.get("warmup_excluded") is not True
        or manifest.get("pid") != expected_pid
        or manifest.get("parent_pid") != expected_parent_pid
        or type(manifest.get("peak_rss_bytes")) is not int
        or manifest["peak_rss_bytes"] <= 0
        or manifest.get("network_denied") is not True
    ):
        raise ValueError(
            "STOP_V412_SERVICE_INTEGRITY: worker manifest gate failed"
        )
    counters = manifest.get("counters")
    if (
        type(counters) is not dict
        or counters.get("query_count") != 1456
        or counters.get("maximum_candidate_count", 101) > 100
        or any(
            type(counters.get(name)) is not int
            or counters[name] < 0
            for name in (
                "lookup_missing_count",
                "sealed_key_miss_count",
                "cache_rebuild_count",
                "cache_write_count",
                "evidence_cache_hit_count",
                "evidence_cache_miss_count",
                "evidence_cache_eviction_count",
            )
        )
        or type(manifest.get("model_load_count")) is not int
        or manifest["model_load_count"] < 0
        or type(manifest.get("store_load_count")) is not int
        or manifest["store_load_count"] < 0
    ):
        raise ValueError(
            "STOP_V412_SERVICE_INTEGRITY: worker counter gate failed"
        )
    expected_hashes = {
        role: digest for role, (_path, digest) in EXPECTED_FILES.items()
    }
    if manifest.get("asset_hashes") != expected_hashes:
        raise ValueError(
            "STOP_V412_SERVICE_INTEGRITY: worker asset identity changed"
        )

    file_mapping = {
        "candidates_features.parquet": "candidates_features.parquet",
        "ranker.parquet": "ranker_reference.parquet",
        "scenes.parquet": "scenes_reference.parquet",
    }
    for observed_name, expected_name in file_mapping.items():
        _assert_exact(
            observed[observed_name],
            reference[expected_name],
            label=observed_name,
        )
    acceptor = observed["acceptor.parquet"].reset_index(drop=True)
    expected_acceptor = reference["acceptor_reference.parquet"].reset_index(
        drop=True
    )
    _assert_exact(
        acceptor.drop(columns=["score"]),
        expected_acceptor.drop(columns=["score"]),
        label="acceptor non-score",
    )
    differences = np.abs(
        acceptor["score"].to_numpy(dtype=np.float64)
        - expected_acceptor["score"].to_numpy(dtype=np.float64)
    )
    if not np.isfinite(differences).all() or (differences > 1e-15).any():
        raise ValueError(
            "STOP_V412_SERVICE_INTEGRITY: acceptor score parity changed"
        )

    if expected_mode == "v412g":
        for observed_name, expected_name in (
            ("query_evidence.parquet", "query_evidence.parquet"),
            ("candidate_evidence.parquet", "candidate_evidence.parquet"),
            ("guard.parquet", "guard_reference.parquet"),
        ):
            _assert_exact(
                observed[observed_name],
                reference[expected_name],
                label=observed_name,
            )
    timings = observed["timings.parquet"]
    queries = reference["queries.parquet"]
    if (
        tuple(timings.columns) != TIMING_COLUMNS
        or len(timings) != 1456
        or timings["query_id"].astype(str).tolist()
        != queries["query_id"].astype(str).tolist()
        or (
            timings[list(TIMING_COLUMNS[1:])].to_numpy(dtype=np.int64)
            < 0
        ).any()
    ):
        raise ValueError(
            "STOP_V412_SERVICE_INTEGRITY: worker timing table changed"
        )
    if expected_mode == "v411" and (
        timings[
            [
                "evidence_route_load_index_ns",
                "evidence_search_aggregate_ns",
                "guard_ns",
                "evidence_guard_ns",
            ]
        ]
        .to_numpy(dtype=np.int64)
        .any()
    ):
        raise ValueError(
            "STOP_V412_SERVICE_INTEGRITY: V4.11 measured evidence"
        )
    component_values = timings[
        [
            "retrieval_lookup_ns",
            "hydrate_feature_ns",
            "ranker_ns",
            "scene_acceptor_ns",
        ]
    ].to_numpy(dtype=np.int64)
    if (component_values <= 0).any():
        raise ValueError(
            "STOP_V412_SERVICE_INTEGRITY: inactive core timing"
        )
    evidence_sum = (
        timings["evidence_route_load_index_ns"].astype(np.int64)
        + timings["evidence_search_aggregate_ns"].astype(np.int64)
        + timings["guard_ns"].astype(np.int64)
    )
    if not np.array_equal(
        evidence_sum.to_numpy(dtype=np.int64),
        timings["evidence_guard_ns"].to_numpy(dtype=np.int64),
    ):
        raise ValueError(
            "STOP_V412_SERVICE_INTEGRITY: evidence timing sum changed"
        )
    if expected_mode == "v412g" and (
        timings[
            [
                "evidence_route_load_index_ns",
                "evidence_search_aggregate_ns",
                "guard_ns",
            ]
        ]
        .to_numpy(dtype=np.int64)
        <= 0
    ).any():
        raise ValueError(
            "STOP_V412_SERVICE_INTEGRITY: inactive evidence timing"
        )
    accounted = (
        timings[
            [
                "retrieval_lookup_ns",
                "hydrate_feature_ns",
                "ranker_ns",
                "scene_acceptor_ns",
            ]
        ]
        .sum(axis=1)
        .astype(np.int64)
        + evidence_sum
    )
    if (
        timings["total_wall_ns"].to_numpy(dtype=np.int64)
        < accounted.to_numpy(dtype=np.int64)
    ).any():
        raise ValueError(
            "STOP_V412_SERVICE_INTEGRITY: total timing undercounts stages"
        )
    return {
        "mode": expected_mode,
        "pid": manifest["pid"],
        "peak_rss_bytes": manifest["peak_rss_bytes"],
        "query_count": reference_manifest["expected_query_count"],
        "candidate_count": reference_manifest["expected_candidate_count"],
        "latency": _latency_summary(timings),
        "counters": counters,
        "evidence_cache": manifest["evidence_cache"],
        "model_load_count": manifest["model_load_count"],
        "store_load_count": manifest["store_load_count"],
        "parity": "EXACT_7_STAGES" if expected_mode == "v412g" else "EXACT_5_STAGES",
        "_total_wall_ns": timings["total_wall_ns"].astype(int).tolist(),
    }


__all__ = [
    "MAX_RSS_BYTES",
    "REFERENCE_MANIFEST_SHA256",
    "REFERENCE_ROOT",
    "nearest_rank",
    "evaluate_paired_gate",
    "validate_worker_output",
]
