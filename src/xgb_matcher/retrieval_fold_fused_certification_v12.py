"""Post-fit certification for one immutable fold-fused v12 OOF run.

The OOF executor is intentionally unable to consume POSTFIT parity: those
models do not exist until the run completes.  This joiner verifies the exact
OOF manifest and its three model bytes against the later parity replay, then
publishes a no-refit certificate.  Runtime remains a subsequent, separately
sealed gate.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import pyarrow.parquet as pq

from .retrieval_fold_fused_oof_v12 import (
    FoldFusedInvariantError,
    GateThresholds,
    MANIFEST_SCHEMA_VERSION,
    TRAIN_FOLDS,
    sha256_file,
)
from .retrieval_fold_fused_runtime_v12 import load_parity_evidence_v12
from .retrieval_fold_fused_sharded_oof_v12 import (
    ENGINE_ID as SHARDED_ENGINE_ID,
    SHARDED_RUN_SCHEMA_VERSION,
    ShardSpecV12,
    _feature_names_sha256_v12,
    _outcome_from_row_v12,
    aggregate_sharded_gates_v12,
    load_fitted_site_model_records_v12,
    load_heldout_master_record_v12,
    load_shard_build_manifest_v12,
    load_shard_replay_manifest_v12,
)


CERTIFICATION_SCHEMA_VERSION = (
    "sireto-retrieval-v12-fold-fused-postfit-certificate-1"
)
_NATIONAL_QUERY_COUNT = 23_571
_NATIONAL_SHARDS = (
    {
        "shard_id": "fold2",
        "fold": 2,
        "query_start": 0,
        "query_stop": 7_767,
        "candidate_start": 0,
        "candidate_stop": 13_750_106,
    },
    {
        "shard_id": "fold3",
        "fold": 3,
        "query_start": 7_767,
        "query_stop": 15_734,
        "candidate_start": 13_750_106,
        "candidate_stop": 27_705_638,
    },
    {
        "shard_id": "fold4",
        "fold": 4,
        "query_start": 15_734,
        "query_stop": 23_571,
        "candidate_start": 27_705_638,
        "candidate_stop": 41_590_480,
    },
)
_SCIENTIFIC_LINEAGE_KEYS = (
    "scene_ir_sha256",
    "scene_ir_manifest_sha256",
    "source_candidate_sha256",
    "source_manifest_sha256",
    "label_ledger_sha256",
    "label_ledger_manifest_sha256",
    "scientific_config_sha256",
    "executor_config_sha256",
    "official_sidecar_manifest_sha256",
    "mmap_sidecar_manifest_sha256",
    "reader_parity_evidence_sha256",
    "parent_model_bundle_sha256",
    "query_sample_sha256",
)
_SHARDED_EXECUTION_LINEAGE_KEYS = (
    "sharded_implementation_sha256",
    "runtime_environment_sha256",
)
_NATIONAL_EXECUTION_LINEAGE_KEYS = (
    "physical_benchmark_manifest_sha256",
    "execution_authorization_sha256",
    "runtime_resource_guard_sha256",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _resolve_hashed_artifact(
    manifest_path: Path,
    record: Mapping[str, Any],
    *,
    label: str,
) -> Path:
    path = Path(str(record.get("path") or ""))
    if not path.is_absolute():
        path = manifest_path.parent / path
    path = path.resolve()
    if not path.is_file() or sha256_file(path) != record.get("sha256"):
        raise FoldFusedInvariantError(f"OOF artifact hash mismatch: {label}")
    return path


def _validate_hashed_sections(
    path: Path,
    raw: Mapping[str, Any],
    *,
    exact_outputs: set[str] | None = None,
) -> tuple[Mapping[str, Any], Mapping[str, Path]]:
    models = raw.get("site_models")
    if not isinstance(models, Mapping) or set(models) != {
        str(value) for value in TRAIN_FOLDS
    }:
        raise FoldFusedInvariantError("OOF manifest does not seal three site models")
    for outer in TRAIN_FOLDS:
        record = models[str(outer)]
        if not isinstance(record, Mapping):
            raise FoldFusedInvariantError("invalid OOF site model record")
        _resolve_hashed_artifact(path, record, label=f"site model {outer}")

    resolved_outputs: dict[str, Path] = {}
    for section in ("train_matrices", "outputs"):
        records = raw.get(section)
        if not isinstance(records, Mapping) or not records:
            raise FoldFusedInvariantError(f"OOF manifest has no {section}")
        if section == "train_matrices" and set(records) != {
            str(value) for value in TRAIN_FOLDS
        }:
            raise FoldFusedInvariantError("OOF train matrices must cover outers 2/3/4")
        if section == "outputs" and exact_outputs is not None and set(records) != exact_outputs:
            raise FoldFusedInvariantError("sharded OOF outputs are incomplete or unexpected")
        for key, record in records.items():
            if not isinstance(record, Mapping):
                raise FoldFusedInvariantError(f"invalid OOF {section} record")
            resolved = _resolve_hashed_artifact(
                path, record, label=f"{section}/{key}"
            )
            if section == "outputs":
                resolved_outputs[str(key)] = resolved
    return models, resolved_outputs


def _validate_legacy_oof_manifest(
    path: Path, raw: Mapping[str, Any]
) -> Mapping[str, Any]:
    required = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_kind": "NESTED_OUTER_OOF_FOLD_FUSED",
        "status": "EXECUTED_NO_READY_CLAIM",
        "opened_folds": list(TRAIN_FOLDS),
        "fold0_opened": False,
        "fold1_opened": False,
        "positive_injection": False,
        "logical_recipe_changes": False,
    }
    if any(raw.get(key) != value for key, value in required.items()):
        raise FoldFusedInvariantError("OOF manifest is not certifiable")
    parity = raw.get("parity") or {}
    gates = raw.get("gates") or {}
    lineage = raw.get("lineage") or {}
    if (
        parity.get("status") != "NOT_RUN"
        or gates.get("parity_gate") != "NOT_RUN"
        or gates.get("decision") != "NO_READY_DECISION"
        or gates.get("gate_eligible") is not False
        or lineage.get("parity_evidence_sha256") is not None
    ):
        raise FoldFusedInvariantError(
            "OOF must remain PREFIT-only/NO_READY before postfit certification"
        )
    _validate_hashed_sections(path, raw)
    return raw


def _validate_national_specs_v12(
    raw: Mapping[str, Any],
) -> tuple[tuple[ShardSpecV12, ...], tuple[tuple[str, int], ...], str]:
    specs_raw = raw.get("specs")
    if not isinstance(specs_raw, list) or len(specs_raw) != 3:
        raise FoldFusedInvariantError(
            "sharded OOF must seal exactly three national fold shards"
        )
    specs = tuple(ShardSpecV12.from_manifest(value) for value in specs_raw)
    for ordinal, (spec, expected) in enumerate(zip(specs, _NATIONAL_SHARDS, strict=True)):
        expected_query_count = expected["query_stop"] - expected["query_start"]
        expected_candidate_count = (
            expected["candidate_stop"] - expected["candidate_start"]
        )
        if (
            spec.shard_id != expected["shard_id"]
            or spec.shard_ordinal != ordinal
            or spec.sequence_start != expected["query_start"]
            or spec.sequence_stop != expected["query_stop"]
            or spec.query_count != expected_query_count
            or spec.folds != (expected["fold"],)
            or spec.query_folds != (expected["fold"],) * expected_query_count
            or spec.source_ordinals
            != tuple(range(expected["query_start"], expected["query_stop"]))
            or spec.source_ranges
            != ((expected["candidate_start"], expected["candidate_stop"]),)
            or spec.candidate_row_count != expected_candidate_count
        ):
            raise FoldFusedInvariantError(
                f"sharded OOF national range drifted: {expected['shard_id']}"
            )
    sequence = tuple(
        (query_id, fold)
        for spec in specs
        for query_id, fold in zip(spec.query_ids, spec.query_folds, strict=True)
    )
    if (
        len(sequence) != _NATIONAL_QUERY_COUNT
        or len({query_id for query_id, _fold in sequence}) != _NATIONAL_QUERY_COUNT
    ):
        raise FoldFusedInvariantError(
            "sharded OOF national query population overlaps or is incomplete"
        )
    sequence_sha = _sha256_json(
        [[query_id, fold] for query_id, fold in sequence]
    )
    return specs, sequence, sequence_sha


def _validate_national_execution_authority_v12(
    raw: Mapping[str, Any],
    *,
    concurrency: int,
    specs: tuple[ShardSpecV12, ...],
    query_sequence_sha256: str,
) -> Mapping[str, str]:
    lineage_raw = raw.get("lineage")
    required_lineage_keys = {
        *_SCIENTIFIC_LINEAGE_KEYS,
        *_SHARDED_EXECUTION_LINEAGE_KEYS,
        *_NATIONAL_EXECUTION_LINEAGE_KEYS,
    }
    if (
        not isinstance(lineage_raw, Mapping)
        or set(lineage_raw) != required_lineage_keys
        or any(not _is_sha256(value) for value in lineage_raw.values())
    ):
        raise FoldFusedInvariantError(
            "sharded OOF national execution lineage is incomplete"
        )
    lineage = {str(key): str(value) for key, value in lineage_raw.items()}

    evidence = raw.get("execution_evidence")
    expected_evidence_keys = {
        "phase",
        "physical_benchmark_manifest_sha256",
        "execution_authorization_sha256",
        "runtime_resource_guard",
        "fixed_shard_count",
        "maximum_concurrent_workers",
    }
    if not isinstance(evidence, Mapping) or set(evidence) != expected_evidence_keys:
        raise FoldFusedInvariantError(
            "sharded OOF national execution evidence is incomplete"
        )
    guard_raw = evidence.get("runtime_resource_guard")
    if not isinstance(guard_raw, Mapping):
        raise FoldFusedInvariantError("sharded OOF runtime guard is missing")
    expected_guard_keys = {
        "schema_version",
        "status",
        "selection_reason",
        "physical_authorized_maximum_concurrent_workers",
        "selected_maximum_concurrent_workers",
        "fixed_shard_count",
        "memory_metric",
        "memory_used_before_bytes",
        "memory_used_after_bytes",
        "memory_used_peak_bytes",
        "memory_threshold_bytes",
        "swapout_metric",
        "swapout_source",
        "swapout_page_size_bytes",
        "swapout_before_pages",
        "swapout_after_pages",
        "swapout_delta_pages",
        "swapout_delta_bytes",
        "observation_seconds",
        "observation_started_at_unix_seconds",
        "observation_finished_at_unix_seconds",
        "measurement_valid",
        "measured_before_first_shard_open",
        "worker_count_change_after_measurement",
        "automatic_retry_allowed",
        "record_sha256",
    }
    if set(guard_raw) != expected_guard_keys:
        raise FoldFusedInvariantError(
            "sharded OOF runtime guard fields are incomplete or unexpected"
        )
    guard = dict(guard_raw)
    guard_sha = str(guard.pop("record_sha256", ""))
    if (
        not _is_sha256(guard_sha)
        or _sha256_json(guard) != guard_sha
        or guard.get("schema_version")
        != "sireto-retrieval-v12-national-worker-guard-1"
        or guard.get("selected_maximum_concurrent_workers") != concurrency
        or guard.get("fixed_shard_count") != 3
        or guard.get("memory_metric") != "psutil.virtual_memory().used"
        or guard.get("memory_threshold_bytes") != 14 * 1024**3
        or guard.get("swapout_metric") != "vm_stat.Swapouts"
        or guard.get("observation_seconds") != 5.0
        or guard.get("measured_before_first_shard_open") is not True
        or guard.get("worker_count_change_after_measurement") is not False
        or guard.get("automatic_retry_allowed") is not False
    ):
        raise FoldFusedInvariantError("sharded OOF runtime guard drifted")

    authorized = guard.get("physical_authorized_maximum_concurrent_workers")
    measurement_valid = guard.get("measurement_valid")
    memory_before = guard.get("memory_used_before_bytes")
    memory_after = guard.get("memory_used_after_bytes")
    memory_peak = guard.get("memory_used_peak_bytes")
    swap_before = guard.get("swapout_before_pages")
    swap_after = guard.get("swapout_after_pages")
    swap_delta = guard.get("swapout_delta_pages")
    swap_page_size = guard.get("swapout_page_size_bytes")
    swap_delta_bytes = guard.get("swapout_delta_bytes")
    swap_source = guard.get("swapout_source")
    timestamps = (
        guard.get("observation_started_at_unix_seconds"),
        guard.get("observation_finished_at_unix_seconds"),
    )

    def optional_nonnegative_integer(value: Any) -> bool:
        return value is None or (
            isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0
        )

    if (
        authorized not in (2, 3)
        or not isinstance(measurement_valid, bool)
        or any(
            not optional_nonnegative_integer(value)
            for value in (
                memory_before,
                memory_after,
                memory_peak,
                swap_before,
                swap_after,
                swap_delta,
                swap_page_size,
                swap_delta_bytes,
            )
        )
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in timestamps
        )
        or (
            swap_source is not None
            and (not isinstance(swap_source, str) or not swap_source)
        )
        or (swap_source is None) != (swap_page_size is None)
        or (swap_page_size is not None and swap_page_size <= 0)
    ):
        raise FoldFusedInvariantError(
            "sharded OOF runtime guard measurements are invalid"
        )
    expected_memory_peak = (
        max(memory_before, memory_after)
        if memory_before is not None and memory_after is not None
        else None
    )
    if swap_before is not None and swap_after is not None:
        if swap_after < swap_before or swap_page_size is None:
            raise FoldFusedInvariantError(
                "sharded OOF runtime guard swap counters are invalid"
            )
        expected_swap_delta = swap_after - swap_before
        expected_swap_delta_bytes = expected_swap_delta * swap_page_size
    else:
        expected_swap_delta = None
        expected_swap_delta_bytes = None
    if (
        memory_peak != expected_memory_peak
        or swap_delta != expected_swap_delta
        or swap_delta_bytes != expected_swap_delta_bytes
        or (
            measurement_valid is True
            and (
                memory_before is None
                or memory_after is None
                or memory_peak is None
                or swap_before is None
                or swap_after is None
                or swap_delta is None
                or swap_page_size is None
                or swap_page_size <= 0
                or swap_source is None
            )
        )
    ):
        raise FoldFusedInvariantError(
            "sharded OOF runtime guard measurements are inconsistent"
        )
    w3_resources_pass = bool(
        measurement_valid
        and memory_peak is not None
        and memory_peak <= 14 * 1024**3
        and swap_delta == 0
    )
    expected_selected = 3 if authorized == 3 and w3_resources_pass else 2
    if authorized == 2:
        expected_reason = "PHYSICAL_BENCHMARK_AUTHORIZED_W2_ONLY"
    elif not measurement_valid:
        expected_reason = "W3_MEASUREMENT_UNAVAILABLE_FALLBACK_W2"
    elif memory_peak is not None and memory_peak > 14 * 1024**3:
        expected_reason = "W3_MEMORY_LIMIT_EXCEEDED_FALLBACK_W2"
    elif swap_delta != 0:
        expected_reason = "W3_SWAPOUT_ACTIVITY_FALLBACK_W2"
    else:
        expected_reason = "W3_CURRENT_STATE_PASS"
    expected_status = "PASS_W3" if expected_selected == 3 else "PASS_W2_FALLBACK"
    if (
        concurrency != expected_selected
        or guard.get("status") != expected_status
        or guard.get("selection_reason") != expected_reason
    ):
        raise FoldFusedInvariantError(
            "sharded OOF runtime guard decision/concurrency drifted"
        )
    if (
        evidence.get("phase") != "NATIONAL_OOF_FOLDS_2_3_4"
        or evidence.get("fixed_shard_count") != 3
        or evidence.get("maximum_concurrent_workers") != concurrency
        or evidence.get("physical_benchmark_manifest_sha256")
        != lineage["physical_benchmark_manifest_sha256"]
        or evidence.get("execution_authorization_sha256")
        != lineage["execution_authorization_sha256"]
        or guard_sha != lineage["runtime_resource_guard_sha256"]
    ):
        raise FoldFusedInvariantError(
            "sharded OOF authorization/guard lineage drifted"
        )

    digests = raw.get("digests")
    run_identity = raw.get("run_identity")
    expected_identity = {
        "engine_id": SHARDED_ENGINE_ID,
        "run_kind": "NESTED_OUTER_OOF_FOLD_FUSED_SHARDED",
        "query_count": _NATIONAL_QUERY_COUNT,
        "query_sequence_sha256": query_sequence_sha256,
        "lineage": lineage,
        "spec_sha256": [spec.digest for spec in specs],
        "shard_count": 3,
        "maximum_concurrent_workers": concurrency,
        "execution_evidence": dict(evidence),
        "digests": digests,
        "miniature_fit": False,
    }
    if (
        not isinstance(digests, Mapping)
        or run_identity != expected_identity
        or raw.get("run_id") != _sha256_json(expected_identity)
    ):
        raise FoldFusedInvariantError("sharded OOF national run identity drifted")
    return lineage


def _validate_national_worker_manifests_v12(
    path: Path,
    raw: Mapping[str, Any],
    *,
    specs: tuple[ShardSpecV12, ...],
    lineage: Mapping[str, str],
) -> None:
    build_hashes = raw.get("worker_build_manifest_sha256")
    replay_hashes = raw.get("worker_replay_manifest_sha256")
    if (
        not isinstance(build_hashes, list)
        or not isinstance(replay_hashes, list)
        or len(build_hashes) != 3
        or len(replay_hashes) != 3
        or any(not _is_sha256(value) for value in (*build_hashes, *replay_hashes))
    ):
        raise FoldFusedInvariantError(
            "sharded OOF worker manifest lineage is incomplete"
        )
    root = path.parent.resolve()
    for ordinal, spec in enumerate(specs):
        build_path = root / "worker_build" / spec.shard_id / "manifest.json"
        replay_path = root / "worker_replay" / spec.shard_id / "manifest.json"
        try:
            build = load_shard_build_manifest_v12(build_path, verify_hashes=True)
            replay = load_shard_replay_manifest_v12(replay_path, verify_hashes=True)
        except (FileNotFoundError, OSError, json.JSONDecodeError) as error:
            raise FoldFusedInvariantError(
                f"sharded OOF worker manifest is unreadable: {spec.shard_id}"
            ) from error
        if (
            build.manifest_sha256 != build_hashes[ordinal]
            or replay.manifest_sha256 != replay_hashes[ordinal]
            or build.spec.as_manifest() != spec.as_manifest()
            or replay.spec.as_manifest() != spec.as_manifest()
            or dict(build.lineage) != dict(lineage)
            or dict(replay.lineage) != dict(lineage)
            or replay.build_manifest_sha256 != build.manifest_sha256
        ):
            raise FoldFusedInvariantError(
                f"sharded OOF worker lineage drifted: {spec.shard_id}"
            )


def _validate_sharded_oof_manifest(
    path: Path, raw: Mapping[str, Any]
) -> Mapping[str, Any]:
    required = {
        "schema_version": SHARDED_RUN_SCHEMA_VERSION,
        "engine_id": SHARDED_ENGINE_ID,
        "run_kind": "NESTED_OUTER_OOF_FOLD_FUSED_SHARDED",
        "status": "EXECUTED_NO_READY_CLAIM",
        "folds": list(TRAIN_FOLDS),
        "fold0_opened": False,
        "fold1_opened": False,
        "positive_injection": False,
        "logical_recipe_changes": False,
        "miniature_fit": False,
        "shard_count": 3,
        "query_count": _NATIONAL_QUERY_COUNT,
    }
    if any(raw.get(key) != value for key, value in required.items()):
        raise FoldFusedInvariantError("sharded OOF manifest is not certifiable")
    query_count = int(raw.get("query_count", 0))
    concurrency = int(raw.get("maximum_concurrent_workers", 0))
    if (
        query_count != _NATIONAL_QUERY_COUNT
        or concurrency not in (2, 3)
        or raw.get("worker_count") != concurrency
    ):
        raise FoldFusedInvariantError(
            "sharded OOF national population/concurrency drifted"
        )

    specs, expected_sequence, query_sequence_sha = _validate_national_specs_v12(raw)
    lineage = _validate_national_execution_authority_v12(
        raw,
        concurrency=concurrency,
        specs=specs,
        query_sequence_sha256=query_sequence_sha,
    )
    _validate_national_worker_manifests_v12(
        path, raw, specs=specs, lineage=lineage
    )

    models, outputs = _validate_hashed_sections(
        path,
        raw,
        exact_outputs={"top500", "top100", "outcomes"},
    )
    loaded_models = load_fitted_site_model_records_v12(
        models,
        handle_loader=lambda _model_path, _outer: None,
        root=path.parent,
        verify_hashes=True,
    )
    if set(loaded_models) != set(TRAIN_FOLDS):
        raise FoldFusedInvariantError("sharded OOF site model catalog drifted")

    heldout_record = raw.get("heldout_master")
    if not isinstance(heldout_record, Mapping):
        raise FoldFusedInvariantError("sharded OOF heldout master is missing")
    heldout = load_heldout_master_record_v12(
        heldout_record, root=path.parent, verify_hashes=True
    )
    if (
        heldout.query_count != query_count
        or heldout.query_sequence_sha256 != query_sequence_sha
        or path.parent.resolve() not in heldout.path.resolve().parents
        or path.parent.resolve() not in heldout.manifest_path.resolve().parents
    ):
        raise FoldFusedInvariantError("sharded OOF heldout population drifted")

    train = raw["train_matrices"]
    for outer in TRAIN_FOLDS:
        record = train[str(outer)]
        expected_fit_folds = [value for value in TRAIN_FOLDS if value != outer]
        feature_names = tuple(str(value) for value in record.get("feature_names") or ())
        feature_schema = str(record.get("feature_schema_sha256") or "")
        if (
            record.get("fit_folds") != expected_fit_folds
            or int(record.get("rows", 0)) <= 0
            or int(record.get("queries", 0)) <= 0
            or len(feature_names) != 319
            or _feature_names_sha256_v12(feature_names) != feature_schema
            or loaded_models[outer].feature_names != feature_names
            or not _is_sha256(record.get("group_sequence_sha256"))
        ):
            raise FoldFusedInvariantError(
                f"sharded OOF train matrix contract drifted: outer {outer}"
            )

    output_records = raw["outputs"]
    if (
        int(output_records["outcomes"].get("rows", -1)) != query_count
        or not query_count
        <= int(output_records["top500"].get("rows", -1))
        <= 500 * query_count
        or not query_count
        <= int(output_records["top100"].get("rows", -1))
        <= 100 * query_count
    ):
        raise FoldFusedInvariantError("sharded OOF output cardinality/caps drifted")
    outcomes_table = pq.read_table(outputs["outcomes"])
    if outcomes_table.num_rows != query_count:
        raise FoldFusedInvariantError("sharded OOF outcome Parquet cardinality drifted")
    outcomes = tuple(
        _outcome_from_row_v12(row) for row in outcomes_table.to_pylist()
    )
    if (
        tuple((outcome.query_id, outcome.fold) for outcome in outcomes)
        != expected_sequence
    ):
        raise FoldFusedInvariantError("sharded OOF outcome population/order drifted")
    expected_gates = aggregate_sharded_gates_v12(outcomes, GateThresholds())
    if raw.get("gates") != expected_gates:
        raise FoldFusedInvariantError(
            "sharded OOF gates were not recomputed from the sealed outcomes"
        )
    if lineage.get("parity_evidence_sha256") is not None:
        raise FoldFusedInvariantError(
            "sharded OOF must remain PREFIT-only before postfit certification"
        )

    expected_digests = {
        "train_matrix": {
            str(outer): str(train[str(outer)].get("sha256") or "")
            for outer in TRAIN_FOLDS
        },
        "group_sequence": {
            str(outer): str(
                train[str(outer)].get("group_sequence_sha256") or ""
            )
            for outer in TRAIN_FOLDS
        },
        "heldout_spool": heldout.sha256,
        "model_by_outer": {
            str(outer): loaded_models[outer].sha256 for outer in TRAIN_FOLDS
        },
        "prerank500": str(output_records["top500"].get("sha256") or ""),
        "top100": str(output_records["top100"].get("sha256") or ""),
        "outcomes": str(output_records["outcomes"].get("sha256") or ""),
    }
    if raw.get("digests") != expected_digests:
        raise FoldFusedInvariantError("sharded OOF national digest surface drifted")
    return raw


def _validate_oof_manifest(path: Path) -> Mapping[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    schema = raw.get("schema_version")
    if schema == MANIFEST_SCHEMA_VERSION:
        return _validate_legacy_oof_manifest(path, raw)
    if schema == SHARDED_RUN_SCHEMA_VERSION:
        return _validate_sharded_oof_manifest(path, raw)
    raise FoldFusedInvariantError("OOF manifest schema is not certifiable")


def certify_fold_fused_oof_postfit_v12(
    *,
    oof_manifest_path: Path,
    postfit_parity_manifest_path: Path,
    output_path: Path,
) -> Path:
    """Bind exact POSTFIT parity to one OOF run, without fitting anything."""

    oof_manifest_path = Path(oof_manifest_path).resolve()
    postfit_parity_manifest_path = Path(postfit_parity_manifest_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    oof = _validate_oof_manifest(oof_manifest_path)
    oof_sha = sha256_file(oof_manifest_path)
    lineage = oof["lineage"]
    parity_lineage_keys = (
        "scene_ir_sha256",
        "scene_ir_manifest_sha256",
        "source_candidate_sha256",
        "source_manifest_sha256",
        "label_ledger_sha256",
        "label_ledger_manifest_sha256",
        "scientific_config_sha256",
        "executor_config_sha256",
        "official_sidecar_manifest_sha256",
        "mmap_sidecar_manifest_sha256",
        "reader_parity_evidence_sha256",
        "parent_model_bundle_sha256",
        "query_sample_sha256",
    )
    expected_lineage = {
        key: str(lineage.get(key) or "") for key in parity_lineage_keys
    }
    if any(len(value) != 64 for value in expected_lineage.values()):
        raise FoldFusedInvariantError("OOF lineage required by parity is incomplete")
    _evidence, parity_sha = load_parity_evidence_v12(
        postfit_parity_manifest_path,
        expected_lineage=expected_lineage,
    )
    parity = json.loads(postfit_parity_manifest_path.read_text(encoding="utf-8"))
    if parity.get("site_model_manifest_sha256") != oof_sha:
        raise FoldFusedInvariantError(
            "POSTFIT parity was not run against this exact OOF manifest"
        )
    oof_models = oof["site_models"]
    parity_models = parity.get("site_model_sha256") or {}
    parity_recipes = parity.get("site_model_recipe_sha256") or {}
    for outer in TRAIN_FOLDS:
        record = oof_models[str(outer)]
        if (
            parity_models.get(str(outer)) != record.get("sha256")
            or parity_recipes.get(str(outer)) != record.get("recipe_sha256")
        ):
            raise FoldFusedInvariantError(
                f"POSTFIT parity/site model binding mismatch: outer {outer}"
            )

    gates = oof["gates"]
    if gates.get("prerank_recall_at_500", {}).get("rate", 0.0) < gates.get(
        "thresholds", {}
    ).get("prerank_recall_at_500", float("inf")):
        decision = "PIVOT_GENERATION"
    elif gates.get("recall_gates_pass") is not True:
        decision = "PIVOT_ADMISSION"
    else:
        decision = "PENDING_RUNTIME_PROFILE"
    certificate = {
        "schema_version": CERTIFICATION_SCHEMA_VERSION,
        "status": "POSTFIT_PARITY_CERTIFIED_RUNTIME_PENDING",
        "decision": decision,
        "gate_eligible": False,
        "refit_performed": False,
        "fold0_opened": False,
        "fold1_opened": False,
        "runtime_gate": "NOT_EVALUATED",
        "master_decision_blocking": False,
        "lineage": {
            "oof_manifest_sha256": oof_sha,
            "postfit_parity_manifest_sha256": parity_sha,
            "prefit_parity_evidence_sha256": lineage.get(
                "prefit_parity_evidence_sha256"
            ),
            "reader_parity_evidence_sha256": lineage.get(
                "reader_parity_evidence_sha256"
            ),
            "site_model_sha256": {
                str(outer): oof_models[str(outer)]["sha256"]
                for outer in TRAIN_FOLDS
            },
            "site_model_recipe_sha256": {
                str(outer): oof_models[str(outer)]["recipe_sha256"]
                for outer in TRAIN_FOLDS
            },
        },
        "gates": {
            "master_recall_at_2000": gates.get("master_recall_at_2000"),
            "master_headroom_target_met": gates.get("master_headroom_target_met"),
            "prerank_recall_at_500": gates.get("prerank_recall_at_500"),
            "exact_recall_at_100": gates.get("exact_recall_at_100"),
            "exact_recall_at_100_by_outer_fold": gates.get(
                "exact_recall_at_100_by_outer_fold"
            ),
            "recall_gates_pass": gates.get("recall_gates_pass"),
            "postfit_parity_gate": "PASS",
            "latency_gate": "NOT_EVALUATED",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output_path.name}.", dir=output_path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        temporary_path.write_text(_canonical_json(certificate) + "\n", encoding="utf-8")
        os.rename(temporary_path, output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return output_path


__all__ = [
    "CERTIFICATION_SCHEMA_VERSION",
    "certify_fold_fused_oof_postfit_v12",
]
