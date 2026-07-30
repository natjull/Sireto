from __future__ import annotations

import copy
import contextlib
import hashlib
import importlib.util
import json
import multiprocessing
import os
import platform
import shutil
import stat
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_RSS_LIMIT = 8 * 1024 * 1024 * 1024
SPEC = importlib.util.spec_from_file_location(
    "evaluate_v412_unit_retrieval",
    ROOT / "scripts/evaluate_v412_unit_retrieval.py",
)
assert SPEC and SPEC.loader
evaluator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluator)


def _load_evaluator_for_child(root: str):
    spec = importlib.util.spec_from_file_location(
        f"evaluator_child_{os.getpid()}",
        Path(root) / "scripts/evaluate_v412_unit_retrieval.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _slot_lock_child(root: str, attempt_root: str, slot: str, queue) -> None:
    module = _load_evaluator_for_child(root)
    try:
        with module._exclusive_slot_lock(Path(attempt_root), slot):
            queue.put("ACQUIRED")
    except module.EvaluationStopped:
        queue.put("BLOCKED")


def _promotion_child(
    root: str, source: str, destination: str, start, queue
) -> None:
    module = _load_evaluator_for_child(root)
    start.wait()
    try:
        module._promote(Path(source), Path(destination))
        queue.put(("PROMOTED", Path(source).name))
    except module.EvaluationStopped:
        queue.put(("BLOCKED", Path(source).name))


def _base_plan(tmp_path: Path) -> dict:
    plan = json.loads(
        (ROOT / "config/v4_12_unit_retrieval_evaluator_plan.json").read_text()
    )
    plan["outputs"] = {
        "evaluation_root": str(tmp_path / "evaluation"),
        "audit_root": str(tmp_path / "audit"),
        "attempt_root": str(tmp_path / "attempt"),
        "temp_root": str(tmp_path / "temp"),
        "runtime_files": [
            "query_outcomes.parquet",
            "metrics.json",
            "integrity.json",
            "manifest.json",
        ],
        "audit_files": [
            "open_ledger.parquet",
            "provenance.json",
            "manifest.json",
        ],
    }
    counts = {
        "global": {"total": 4, "MATCH_EXACT": 3, "AMBIGUOUS": 1},
        "threshold_dev": {"total": 2, "MATCH_EXACT": 1, "AMBIGUOUS": 1},
        "comparison_dev": {"total": 2, "MATCH_EXACT": 2, "AMBIGUOUS": 0},
    }
    plan["oracle"]["population"] = counts
    plan["evaluation_spec"]["population_counts"] = counts
    plan["evaluation_spec"]["join"].update(
        {
            "expected_query_count": 4,
            "expected_candidate_count": 6,
            "expected_minimum_pool_size": 1,
            "expected_maximum_pool_size": 2,
            "expected_under_ceiling_query_count": 4,
            "expected_empty_query_count": 0,
        }
    )
    plan["evaluation_spec"]["gate"].update(
        {"coverage_minimum": 0.7, "recall_at_100_minimum": 0.6}
    )
    plan["evaluation_spec"]["latency"]["query_count"] = 4
    # A full scientific-Python process can reserve several GiB of virtual
    # address space before this fixture starts. Use the same deterministic
    # kernel ceiling as the immutable production plan.
    plan["max_rss_bytes"] = SYNTHETIC_RSS_LIMIT
    return plan


def _schema(fields: list[tuple[str, pa.DataType, bool]]) -> pa.Schema:
    return pa.schema(
        [pa.field(name, dtype, nullable=nullable) for name, dtype, nullable in fields],
        metadata=None,
    )


def _tables() -> tuple[pa.Table, pa.Table, pa.Table]:
    candidates = pa.Table.from_pylist(
        [
            {"query_id": "q1", "candidate_rank": 1, "candidate_siret": "11111111100001"},
            {"query_id": "q1", "candidate_rank": 2, "candidate_siret": "99999999900009"},
            {"query_id": "q2", "candidate_rank": 1, "candidate_siret": "22222222200002"},
            {"query_id": "q3", "candidate_rank": 1, "candidate_siret": "88888888800008"},
            {"query_id": "q3", "candidate_rank": 2, "candidate_siret": "33333333300003"},
            {"query_id": "q4", "candidate_rank": 1, "candidate_siret": "77777777700007"},
        ],
        schema=_schema(
            [
                ("query_id", pa.string(), False),
                ("candidate_rank", pa.uint8(), False),
                ("candidate_siret", pa.string(), False),
            ]
        ),
    )
    statuses = pa.Table.from_pylist(
        [
            {"query_id": "q1", "candidate_count": 2},
            {"query_id": "q2", "candidate_count": 1},
            {"query_id": "q3", "candidate_count": 2},
            {"query_id": "q4", "candidate_count": 1},
        ],
        schema=_schema(
            [
                ("query_id", pa.string(), False),
                ("candidate_count", pa.uint8(), False),
            ]
        ),
    )
    oracle = pa.Table.from_pylist(
        [
            {
                "query_id": "q1",
                "dev_partition": "threshold_dev",
                "label_kind": "MATCH_EXACT",
                "ground_truth_siret": "11111111100001",
                "ground_truth_siren": "111111111",
            },
            {
                "query_id": "q2",
                "dev_partition": "threshold_dev",
                "label_kind": "AMBIGUOUS",
                "ground_truth_siret": None,
                "ground_truth_siren": None,
            },
            {
                "query_id": "q3",
                "dev_partition": "comparison_dev",
                "label_kind": "MATCH_EXACT",
                "ground_truth_siret": "33333333300003",
                "ground_truth_siren": "333333333",
            },
            {
                "query_id": "q4",
                "dev_partition": "comparison_dev",
                "label_kind": "MATCH_EXACT",
                "ground_truth_siret": "44444444400004",
                "ground_truth_siren": "444444444",
            },
        ],
        schema=_schema(
            [
                ("query_id", pa.string(), False),
                ("dev_partition", pa.string(), False),
                ("label_kind", pa.string(), False),
                ("ground_truth_siret", pa.string(), True),
                ("ground_truth_siren", pa.string(), True),
            ]
        ),
    )
    return candidates, statuses, oracle


def _write_json(path: Path, value: dict) -> None:
    path.write_bytes(evaluator.canonical_json(value))


def _reseal_self_consistent_hashes(evaluation: Path, audit: Path) -> None:
    integrity_path = evaluation / "integrity.json"
    integrity = json.loads(integrity_path.read_text())
    integrity["metrics_sha256"] = hashlib.sha256(
        (evaluation / "metrics.json").read_bytes()
    ).hexdigest()
    integrity_path.write_bytes(evaluator.canonical_json(integrity))
    manifest_path = evaluation / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["metrics.json"] = evaluator._json_record(
        evaluation / "metrics.json"
    )
    manifest["files"]["integrity.json"] = evaluator._json_record(integrity_path)
    manifest_path.write_bytes(evaluator.canonical_json(manifest))
    provenance_path = audit / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["evaluation_manifest_sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    provenance_path.write_bytes(evaluator.canonical_json(provenance))
    audit_manifest_path = audit / "manifest.json"
    audit_manifest = json.loads(audit_manifest_path.read_text())
    audit_manifest["files"]["provenance.json"] = evaluator._json_record(
        provenance_path
    )
    audit_manifest_path.write_bytes(evaluator.canonical_json(audit_manifest))


def _fixture_inputs(tmp_path: Path, plan: dict) -> tuple[dict[str, Path], dict[str, str]]:
    root = tmp_path / "inputs"
    root.mkdir()
    candidates, statuses, oracle = _tables()
    paths = {role: root / f"{role}.json" for role in plan["artifact_contract"]["ledger_role_order"]}
    paths["worker_candidates_top100"] = root / "candidates.parquet"
    paths["worker_query_status"] = root / "statuses.parquet"
    paths["oracle_dev"] = root / "oracle.parquet"
    pq.write_table(candidates, paths["worker_candidates_top100"])
    pq.write_table(statuses, paths["worker_query_status"])
    pq.write_table(oracle, paths["oracle_dev"])
    worker = plan["prerequisite"]["worker_build_id"]
    parity = plan["prerequisite"]["parity_build_id"]
    oracle_id = plan["oracle"]["build_id"]
    payloads = {
        "worker_manifest": {
            "worker_build_id": worker,
            "verdict": "SEALED_V412_UNIT_RETRIEVAL",
        },
        "worker_integrity": {
            "worker_build_id": worker,
            "durations_ns": plan["evaluation_spec"]["latency"]["durations_ns"],
        },
        "worker_audit_manifest": {"worker_build_id": worker},
        "parity_manifest": {"parity_build_id": parity},
        "parity_provenance": {"parity_build_id": parity},
        "parity_result": {
            "worker_build_id": worker,
            "parity_build_id": parity,
            "verdict": "GO_V412_UNIT_RETRIEVAL_PARITY",
        },
        "oracle_manifest": {"build_id": oracle_id},
        "oracle_integrity": {"build_id": oracle_id},
        "oracle_audit_manifest": {"build_id": oracle_id},
    }
    for role, payload in payloads.items():
        _write_json(paths[role], payload)
    hashes = {
        role: hashlib.sha256(path.read_bytes()).hexdigest()
        for role, path in paths.items()
    }
    return paths, hashes


def _worker_fixture(tmp_path: Path) -> tuple[dict, dict, dict[int, int]]:
    plan = _base_plan(tmp_path)
    paths, hashes = _fixture_inputs(tmp_path, plan)
    plan["input_paths"].update(
        {role: str(path) for role, path in paths.items()}
    )
    plan["input_hashes"].update(hashes)
    fds = {role: os.open(path, os.O_RDONLY) for role, path in paths.items()}
    spec = {
        "schema_version": evaluator.WORKER_SPEC_SCHEMA,
        "evaluator_build_id": "e" * 64,
        "attempt_id": "a" * 64,
        "worker_build_id": plan["prerequisite"]["worker_build_id"],
        "oracle_build_id": plan["oracle"]["build_id"],
        "parity_build_id": plan["prerequisite"]["parity_build_id"],
        "input_fds": fds,
        "input_paths": {role: str(path) for role, path in paths.items()},
        "input_hashes": plan["input_hashes"],
        "git_commit": "b" * 40,
        "source_hashes": {"synthetic.py": "c" * 64},
        "plan_sha256": "d" * 64,
        "lock_sha256": "f" * 64,
        "evaluation_spec": plan["evaluation_spec"],
        "artifact_contract": plan["artifact_contract"],
        "runtime": plan["runtime"],
        "evaluation_stage": str(tmp_path / "stage" / "evaluation.stage"),
        "audit_stage": str(tmp_path / "stage" / "audit.stage"),
        "max_rss_bytes": plan["max_rss_bytes"],
    }
    (tmp_path / "stage").mkdir()
    return plan, spec, fds


def _package_validation(spec: dict, fds: dict[str, int]) -> dict:
    return {
        **spec,
        "input_snapshots": {
            role: {
                "absolute_path": spec["input_paths"][role],
                "projection": evaluator.DATA_PROJECTIONS.get(
                    role, "FULL_JSON_EXACT_KEYSET"
                ),
                "size_bytes_before": snapshot["size_bytes"],
                "sha256_before": snapshot["sha256"],
                "size_bytes_after": snapshot["size_bytes"],
                "sha256_after": snapshot["sha256"],
            }
            for role, fd in fds.items()
            for snapshot in [
                evaluator._snapshot_fd(fd, spec["max_rss_bytes"])
            ]
        },
    }


def _full_package_validation(
    plan: dict, spec: dict, fds: dict[str, int]
) -> dict:
    validation = _package_validation(spec, fds)
    validation["input_paths"] = plan["input_paths"]
    for role in plan["attempt_protocol"]["computed_runtime_role_order"]:
        path = Path(plan["input_paths"][role])
        validation["input_snapshots"][role] = {
            "absolute_path": str(path),
            "projection": plan["attempt_protocol"][
                "computed_input_projections"
            ][role],
            "size_bytes_before": path.stat().st_size,
            "sha256_before": plan["input_hashes"][role],
            "size_bytes_after": path.stat().st_size,
            "sha256_after": plan["input_hashes"][role],
        }
    return validation


def _write_computed_attestation(
    plan: dict,
    spec: dict,
    fds: dict[str, int],
    attempt_dir: Path,
    evaluation: Path,
    audit: Path,
) -> tuple[str, dict]:
    validation = _full_package_validation(plan, spec, fds)
    digest = evaluator._create_computed_attestation(
        plan,
        attempt_dir,
        spec["evaluator_build_id"],
        spec["attempt_id"],
        evaluation,
        audit,
        validation,
    )
    return digest, validation


@pytest.fixture(autouse=True)
def _reset_artifact_contract() -> None:
    evaluator._ACTIVE_ARTIFACT_CONTRACT = None


def test_plan_projection_and_gate_are_strict(tmp_path: Path) -> None:
    production_plan = json.loads(
        (ROOT / "config/v4_12_unit_retrieval_evaluator_plan.json").read_text()
    )
    assert production_plan["max_rss_bytes"] == 8 * 1024 * 1024 * 1024
    assert evaluator.ADMIN_RSS_LIMIT == production_plan["max_rss_bytes"]
    plan = _base_plan(tmp_path)
    assert plan["max_rss_bytes"] == SYNTHETIC_RSS_LIMIT
    evaluator.validate_plan(plan)
    assert list(plan["evaluation_spec"]) == plan["identity_projections"]["evaluation_spec_keys"]
    assert plan["evaluation_spec"]["gate"]["gate_statistic"] == "OBSERVED_RATE_FROM_RAW_COUNTS"
    assert "input_snapshots.json" not in (
        ROOT / "scripts/evaluate_v412_unit_retrieval.py"
    ).read_text()
    broken = copy.deepcopy(plan)
    broken["evaluation_spec"]["extra"] = True
    with pytest.raises(evaluator.EvaluationStopped):
        evaluator.validate_plan(broken)
    broken = copy.deepcopy(plan)
    broken["attempt_protocol"][
        "oracle_roles_opened_only_after_commit"
    ] = list(reversed(evaluator.ORACLE_ROLE_ORDER))
    with pytest.raises(evaluator.EvaluationStopped, match="oracle boundary"):
        evaluator.validate_plan(broken)


def test_json_rejects_duplicates_and_nonfinite() -> None:
    with pytest.raises(evaluator.EvaluationStopped):
        evaluator.parse_json(b'{"a":1,"a":2}\\n', "duplicate")
    with pytest.raises(evaluator.EvaluationStopped):
        evaluator.parse_json(b'{"a":NaN}\\n', "nan")


def test_openat_rejects_symlink_and_fd_survives_path_swap(tmp_path: Path) -> None:
    original = tmp_path / "original"
    original.write_bytes(b"sealed")
    link = tmp_path / "link"
    link.symlink_to(original)
    with pytest.raises((evaluator.EvaluationStopped, OSError)):
        evaluator._openat_anchored(link)
    digest = hashlib.sha256(b"sealed").hexdigest()
    limit = 512 * 1024 * 1024
    fd, snapshot = evaluator._open_locked(original, digest, limit)
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"attacker")
    os.replace(replacement, original)
    try:
        assert evaluator._read_fd(fd, limit) == b"sealed"
        after = evaluator._snapshot_fd(fd, limit)
        assert after["sha256"] == snapshot["sha256"]
        assert after["inode"] == snapshot["inode"]
        assert after != snapshot
    finally:
        os.close(fd)


def test_wilson_and_raw_gate_statistic() -> None:
    low, high = evaluator.wilson(0, 10, 1.959963984540054)
    assert low == 0.0
    assert 0.0 < high < 1.0
    low, high = evaluator.wilson(10, 10, 2.5758293035489004)
    assert 0.0 < low < 1.0
    assert high == 1.0


def test_outcome_payload_is_byte_exact_and_missing_formula(tmp_path: Path) -> None:
    plan = _base_plan(tmp_path)
    evaluator._ACTIVE_ARTIFACT_CONTRACT = plan["artifact_contract"]
    table, detail, payload, missing = evaluator.evaluate_tables(
        *_tables(), plan["evaluation_spec"]
    )
    assert table.column("query_id").to_pylist() == ["q1", "q2", "q3", "q4"]
    lines = payload.splitlines()
    assert lines[0] == (
        b"q1\x00threshold_dev\x00MATCH_EXACT\x002\x001\x001\x001\x001\x001"
    )
    assert lines[1].endswith(b"\x001\x00\\N\x00\\N\x00\\N\x00\\N\x00\\N")
    assert lines[3].endswith(b"\x001\x00\\N\x000\x000\x000\x000")
    assert missing == 1
    metrics, metric_missing = evaluator.build_metrics(
        detail["rows"],
        plan["evaluation_spec"],
        {
            "evaluator_build_id": "e" * 64,
            "attempt_id": "a" * 64,
            "worker_build_id": "w" * 64,
            "oracle_build_id": "o" * 64,
            "parity_build_id": "p" * 64,
        },
    )
    assert metric_missing == 1
    assert metrics["gates"]["gate_statistic"] == "OBSERVED_RATE_FROM_RAW_COUNTS"
    assert metrics["v412_measurements"][0]["recall_at"]["100"]["success_count"] == 2
    assert metrics["verdict"] == "GO_V412_UNIT_RETRIEVAL_EVALUATION"


def test_truth_absent_is_miss_and_ambiguous_is_outside_recall(tmp_path: Path) -> None:
    plan = _base_plan(tmp_path)
    evaluator._ACTIVE_ARTIFACT_CONTRACT = plan["artifact_contract"]
    _table, detail, _payload, missing = evaluator.evaluate_tables(
        *_tables(), plan["evaluation_spec"]
    )
    metrics, _ = evaluator.build_metrics(
        detail["rows"],
        plan["evaluation_spec"],
        {
            "evaluator_build_id": "e" * 64,
            "attempt_id": "a" * 64,
            "worker_build_id": "w" * 64,
            "oracle_build_id": "o" * 64,
            "parity_build_id": "p" * 64,
        },
    )
    assert missing == 1
    assert metrics["v412_measurements"][0]["coverage"]["denominator_count"] == 4
    assert metrics["v412_measurements"][0]["recall_at"]["100"]["denominator_count"] == 3


@pytest.mark.parametrize("side", ["oracle", "status"])
@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate"])
def test_oracle_status_ids_must_be_exact(
    tmp_path: Path, side: str, mutation: str
) -> None:
    plan = _base_plan(tmp_path)
    evaluator._ACTIVE_ARTIFACT_CONTRACT = plan["artifact_contract"]
    candidates, statuses, oracle = _tables()
    selected = oracle if side == "oracle" else statuses
    rows = selected.to_pylist()
    if mutation == "missing":
        rows.pop()
    elif mutation == "extra":
        rows.append({**rows[-1], "query_id": "q-extra"})
    else:
        rows[-1] = {**rows[-1], "query_id": rows[0]["query_id"]}
    changed = pa.Table.from_pylist(rows, schema=selected.schema)
    with pytest.raises(
        evaluator.EvaluationStopped, match="oracle/status population"
    ):
        evaluator.evaluate_tables(
            candidates,
            changed if side == "status" else statuses,
            changed if side == "oracle" else oracle,
            plan["evaluation_spec"],
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("candidate_rank", 3, "ranks are not contiguous"),
        ("candidate_siret", "11111111100001", "duplicate candidate SIRET"),
        ("candidate_siret", "NOT_A_SIRET", "invalid or duplicate candidate"),
    ],
)
def test_candidate_rows_reject_gaps_duplicates_and_invalid_siret(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    plan = _base_plan(tmp_path)
    evaluator._ACTIVE_ARTIFACT_CONTRACT = plan["artifact_contract"]
    candidates, statuses, oracle = _tables()
    rows = candidates.to_pylist()
    rows[1][field] = value
    changed = pa.Table.from_pylist(rows, schema=candidates.schema)
    with pytest.raises(evaluator.EvaluationStopped, match=message):
        evaluator.evaluate_tables(
            changed, statuses, oracle, plan["evaluation_spec"]
        )


def test_rejects_101_candidates(tmp_path: Path) -> None:
    plan = _base_plan(tmp_path)
    evaluator._ACTIVE_ARTIFACT_CONTRACT = plan["artifact_contract"]
    candidates, statuses, oracle = _tables()
    rows = [
        {"query_id": "q1", "candidate_rank": i, "candidate_siret": f"111111111{i:05d}"}
        for i in range(1, 102)
    ]
    candidates = pa.Table.from_pylist(rows, schema=candidates.schema)
    status_rows = statuses.to_pylist()
    status_rows[0]["candidate_count"] = 101
    statuses = pa.Table.from_pylist(status_rows, schema=statuses.schema)
    plan["evaluation_spec"]["join"]["expected_candidate_count"] = 101
    with pytest.raises(evaluator.EvaluationStopped):
        evaluator.evaluate_tables(candidates, statuses, oracle, plan["evaluation_spec"])


def test_worker_builds_and_validates_synthetic_packages(tmp_path: Path) -> None:
    plan, spec, fds = _worker_fixture(tmp_path)
    try:
        evaluation, audit = evaluator.worker_execute(spec)
        evaluator.validate_packages(
            evaluation,
            audit,
            evaluator_build_id=spec["evaluator_build_id"],
            attempt_id=spec["attempt_id"],
            artifact=plan["artifact_contract"],
            validation=_package_validation(spec, fds),
        )
        metrics = json.loads((evaluation / "metrics.json").read_text())
        assert metrics["population_order"] == ["global", "threshold_dev", "comparison_dev"]
        assert [row["name"] for row in metrics["frozen_references"]] == [
            "historical_all",
            "v2_exact",
            "v3_exact_identifiable",
        ]
        integrity = json.loads((evaluation / "integrity.json").read_text())
        assert integrity["missing_truth_count"] == 1
        ledger = pq.read_table(audit / "open_ledger.parquet")
        assert ledger.num_rows == 12
    finally:
        for fd in fds.values():
            os.close(fd)


def test_output_keyset_tamper_is_rejected(tmp_path: Path) -> None:
    plan, spec, fds = _worker_fixture(tmp_path)
    try:
        evaluation, audit = evaluator.worker_execute(spec)
        metrics_path = evaluation / "metrics.json"
        metrics = json.loads(metrics_path.read_text())
        metrics["unexpected"] = True
        metrics_path.chmod(0o600)
        metrics_path.write_bytes(evaluator.canonical_json(metrics))
        with pytest.raises(evaluator.EvaluationStopped):
            evaluator.validate_packages(
                evaluation,
                audit,
                evaluator_build_id=spec["evaluator_build_id"],
                attempt_id=spec["attempt_id"],
                artifact=plan["artifact_contract"],
                validation=_package_validation(spec, fds),
            )
    finally:
        for fd in fds.values():
            os.close(fd)


def test_manifest_files_and_provenance_are_exact(tmp_path: Path) -> None:
    plan, spec, fds = _worker_fixture(tmp_path)
    try:
        evaluation, audit = evaluator.worker_execute(spec)
        manifest_path = evaluation / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"]["extra.json"] = {
            "path": "extra.json",
            "size_bytes": 0,
            "sha256": "0" * 64,
        }
        manifest_path.write_bytes(evaluator.canonical_json(manifest))
        with pytest.raises(
            evaluator.EvaluationStopped, match="manifest files keyset"
        ):
            evaluator.validate_packages(
                evaluation,
                audit,
                evaluator_build_id=spec["evaluator_build_id"],
                attempt_id=spec["attempt_id"],
                artifact=plan["artifact_contract"],
                validation=_package_validation(spec, fds),
            )
        (tmp_path / "second").mkdir()
        evaluation, audit = evaluator.worker_execute(
            {
                **spec,
                "evaluation_stage": str(tmp_path / "second" / "evaluation.stage"),
                "audit_stage": str(tmp_path / "second" / "audit.stage"),
            }
        )
        provenance_path = audit / "provenance.json"
        provenance = json.loads(provenance_path.read_text())
        provenance["source_hashes"] = {"forged.py": "f" * 64}
        provenance_path.write_bytes(evaluator.canonical_json(provenance))
        audit_manifest_path = audit / "manifest.json"
        audit_manifest = json.loads(audit_manifest_path.read_text())
        audit_manifest["files"]["provenance.json"] = evaluator._json_record(
            provenance_path
        )
        audit_manifest_path.write_bytes(
            evaluator.canonical_json(audit_manifest)
        )
        with pytest.raises(
            evaluator.EvaluationStopped, match="provenance binding"
        ):
            evaluator.validate_packages(
                evaluation,
                audit,
                evaluator_build_id=spec["evaluator_build_id"],
                attempt_id=spec["attempt_id"],
                artifact=plan["artifact_contract"],
                validation=_package_validation(spec, fds),
            )
    finally:
        for fd in fds.values():
            os.close(fd)


def test_self_consistent_hash_tamper_cannot_forge_metrics(tmp_path: Path) -> None:
    plan, spec, fds = _worker_fixture(tmp_path)
    try:
        evaluation, audit = evaluator.worker_execute(spec)
        metrics_path = evaluation / "metrics.json"
        metrics = json.loads(metrics_path.read_text())
        metrics["v412_measurements"][0]["coverage"] = evaluator._proportion(
            2,
            4,
            plan["evaluation_spec"]["confidence_interval"],
        )
        metrics_path.write_bytes(evaluator.canonical_json(metrics))
        _reseal_self_consistent_hashes(evaluation, audit)
        with pytest.raises(evaluator.EvaluationStopped, match="mismatch|recomputation"):
            evaluator.validate_packages(
                evaluation,
                audit,
                evaluator_build_id=spec["evaluator_build_id"],
                attempt_id=spec["attempt_id"],
                artifact=plan["artifact_contract"],
                validation=_package_validation(spec, fds),
            )
    finally:
        for fd in fds.values():
            os.close(fd)


def test_ledger_fake_equal_sizes_are_rejected_against_parent_snapshots(
    tmp_path: Path,
) -> None:
    plan, spec, fds = _worker_fixture(tmp_path)
    try:
        evaluation, audit = evaluator.worker_execute(spec)
        ledger_path = audit / "open_ledger.parquet"
        ledger = pq.read_table(ledger_path).to_pylist()
        ledger[0]["size_bytes_before"] += 1
        ledger[0]["size_bytes_after"] += 1
        pq.write_table(
            pa.Table.from_pylist(
                ledger,
                schema=pq.read_schema(ledger_path),
            ),
            ledger_path,
            compression="zstd",
        )
        audit_manifest_path = audit / "manifest.json"
        audit_manifest = json.loads(audit_manifest_path.read_text())
        audit_manifest["files"]["open_ledger.parquet"] = evaluator._json_record(
            ledger_path
        )
        audit_manifest_path.write_bytes(
            evaluator.canonical_json(audit_manifest)
        )
        with pytest.raises(evaluator.EvaluationStopped, match="ledger binding"):
            evaluator.validate_packages(
                evaluation,
                audit,
                evaluator_build_id=spec["evaluator_build_id"],
                attempt_id=spec["attempt_id"],
                artifact=plan["artifact_contract"],
                validation=_package_validation(spec, fds),
            )
    finally:
        for fd in fds.values():
            os.close(fd)


def _attempt_plan_lock(tmp_path: Path) -> tuple[dict, dict]:
    plan = _base_plan(tmp_path)
    plan["input_hashes"] = {role: "1" * 64 for role in plan["input_paths"]}
    lock = {"input_hashes": plan["input_hashes"]}
    return plan, lock


def test_receipt_rerun_policy_and_oracle_guard(tmp_path: Path) -> None:
    plan, lock = _attempt_plan_lock(tmp_path)
    attempt_dir, _slot, _attempt = evaluator.ensure_receipt(
        plan, lock, "2" * 64, "3" * 64, now=lambda: "2026-01-01T00:00:00Z"
    )
    with pytest.raises(evaluator.EvaluationStopped, match="before durable commit"):
        evaluator._open_oracle_roles_after_commit(plan, lock, attempt_dir)
    repeated_dir, repeated_slot, repeated_attempt = evaluator.ensure_receipt(
        plan, lock, "2" * 64, "3" * 64, now=lambda: "NEVER_PERSISTED"
    )
    assert (repeated_dir, repeated_slot, repeated_attempt) == (
        attempt_dir,
        _slot,
        _attempt,
    )
    receipt = json.loads((attempt_dir / "receipt.json").read_text())
    assert receipt["schema_version"] == plan["attempt_protocol"][
        "schema_versions"
    ]["receipt"]
    assert receipt["created_at_utc"] == "2026-01-01T00:00:00Z"
    changed = copy.deepcopy(lock)
    changed["input_hashes"] = dict(lock["input_hashes"])
    changed["input_hashes"]["oracle_dev"] = "4" * 64
    with pytest.raises(evaluator.EvaluationStopped, match="another policy"):
        evaluator.ensure_receipt(
            plan, changed, "2" * 64, "5" * 64, now=lambda: "2026-01-01T00:00:01Z"
        )


def test_run_evaluation_uses_contractual_oracle_handshake_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _base_plan(tmp_path)
    plan_path = tmp_path / "plan.json"
    lock_path = tmp_path / "lock.json"
    plan_path.write_bytes(evaluator.canonical_json(plan))
    lock_path.write_bytes(evaluator.canonical_json({"synthetic": True}))
    fake_source = tmp_path / "scripts" / "evaluate.py"
    fake_source.parent.mkdir()
    fake_source.write_text("# synthetic")
    monkeypatch.setattr(evaluator, "__file__", str(fake_source))
    monkeypatch.setattr(evaluator, "PLAN_PATH", Path("plan.json"))
    monkeypatch.setattr(evaluator, "LOCK_PATH", Path("lock.json"))
    monkeypatch.setattr(evaluator, "validate_plan", lambda value: None)
    monkeypatch.setattr(evaluator, "validate_lock", lambda *args: None)
    monkeypatch.setattr(evaluator, "_runtime", lambda: plan["runtime"])
    monkeypatch.setattr(
        evaluator, "_identity", lambda *args: ("e" * 64, {})
    )
    monkeypatch.setattr(
        evaluator,
        "_attempt_identities",
        lambda *args: ("s" * 64, "a" * 64, {}),
    )

    @contextlib.contextmanager
    def fake_slot(*_args):
        yield {"synthetic": True}

    monkeypatch.setattr(evaluator, "_exclusive_slot_lock", fake_slot)
    monkeypatch.setattr(evaluator, "_verify_slot_lock", lambda *_args: None)
    attempt_dir = tmp_path / "attempt"
    attempt_dir.mkdir()
    monkeypatch.setattr(
        evaluator,
        "ensure_receipt",
        lambda *args: (attempt_dir, "s" * 64, "a" * 64),
    )
    monkeypatch.setattr(evaluator, "_package_validation", lambda *args: {})
    monkeypatch.setattr(evaluator, "recover_publication", lambda *args: None)
    nonoracle_roles = tuple(
        plan["attempt_protocol"]["pre_oracle_revalidation_roles"]
    )

    def descriptors(roles):
        return {role: os.open("/dev/null", os.O_RDONLY) for role in roles}

    nonoracle = descriptors(nonoracle_roles)
    oracle = descriptors(evaluator.ORACLE_ROLE_ORDER)
    monkeypatch.setattr(
        evaluator,
        "_open_roles",
        lambda _plan, _lock, roles: (
            {role: nonoracle[role] for role in roles},
            {role: {} for role in roles},
        ),
    )
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        evaluator, "_prepare_worker_execution", lambda *args: {"fds": []}
    )
    monkeypatch.setattr(
        evaluator,
        "_worker_spec",
        lambda *args: {
            "evaluator_build_id": "e" * 64,
            "attempt_id": "a" * 64,
        },
    )
    monkeypatch.setattr(
        evaluator,
        "_start_worker_pre_oracle",
        lambda *args: {"synthetic": True},
    )
    monkeypatch.setattr(evaluator, "_resnapshot", lambda *args: None)
    monkeypatch.setattr(evaluator, "append_event", lambda *args, **kwargs: {})

    def open_oracle(_plan, _lock, _attempt):
        observed["opened"] = tuple(oracle)
        return oracle, {role: {} for role in oracle}

    monkeypatch.setattr(
        evaluator, "_open_oracle_roles_after_commit", open_oracle
    )

    def send_oracle(_handle, roles, received):
        observed["sent_roles"] = tuple(roles)
        observed["sent_fds"] = tuple(received)

    monkeypatch.setattr(evaluator, "_send_oracle_fds", send_oracle)
    monkeypatch.setattr(evaluator, "_finish_worker", lambda *args: None)
    monkeypatch.setattr(
        evaluator, "_computed_input_snapshots", lambda *args: {}
    )
    expected_result = (tmp_path / "final-eval", tmp_path / "final-audit")
    monkeypatch.setattr(
        evaluator, "publish_packages", lambda *args: expected_result
    )
    result = evaluator.run_evaluation(plan_path, lock_path)
    assert result == expected_result
    assert observed == {
        "opened": evaluator.ORACLE_ROLE_ORDER,
        "sent_roles": evaluator.ORACLE_ROLE_ORDER,
        "sent_fds": evaluator.ORACLE_ROLE_ORDER,
    }
    assert tuple(
        plan["attempt_protocol"]["oracle_roles_opened_only_after_commit"]
    ) == evaluator.ORACLE_ROLE_ORDER


def test_state_machine_rejects_post_oracle_jump_without_commit(
    tmp_path: Path,
) -> None:
    plan, lock = _attempt_plan_lock(tmp_path)
    attempt_dir, _slot, _attempt = evaluator.ensure_receipt(
        plan, lock, "2" * 64, "3" * 64, now=lambda: "t0"
    )
    with pytest.raises(evaluator.EvaluationStopped, match="invalid attempt transition"):
        evaluator.append_event(
            attempt_dir,
            plan["attempt_protocol"],
            state="RECOVERABLE",
            phase="COMPUTED_STAGING_VALID",
            oracle_open_committed=True,
            evaluator_build_id="e" * 64,
            reason_code=None,
            now=lambda: "t1",
        )


def test_state_cache_missing_and_stale_reconstructs_from_journal(tmp_path: Path) -> None:
    plan, lock = _attempt_plan_lock(tmp_path)
    attempt_dir, _slot, _attempt = evaluator.ensure_receipt(
        plan, lock, "2" * 64, "3" * 64, now=lambda: "t0"
    )
    initial = json.loads((attempt_dir / "state.json").read_text())
    evaluator.append_event(
        attempt_dir,
        plan["attempt_protocol"],
        state="STARTED",
        phase="ORACLE_OPEN_COMMITTED",
        oracle_open_committed=True,
        evaluator_build_id="e" * 64,
        reason_code=None,
        now=lambda: "t1",
    )
    (attempt_dir / "state.json").unlink()
    rebuilt = evaluator.recover_state_cache(attempt_dir, plan["attempt_protocol"])
    assert rebuilt["oracle_open_committed"] is True
    (attempt_dir / "state.json").write_bytes(evaluator.canonical_json(initial))
    rebuilt = evaluator.recover_state_cache(attempt_dir, plan["attempt_protocol"])
    assert rebuilt["sequence"] == 1


def test_state_cache_temp_crash_stays_outside_slot_and_is_never_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, lock = _attempt_plan_lock(tmp_path)
    attempt_dir, _slot, _attempt = evaluator.ensure_receipt(
        plan, lock, "2" * 64, "3" * 64, now=lambda: "t0"
    )
    state = json.loads((attempt_dir / "state.json").read_text())
    original_replace = evaluator.os.replace
    with monkeypatch.context() as patch:
        patch.setattr(
            evaluator.os,
            "replace",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError("synthetic pre-rename crash")
            ),
        )
        with pytest.raises(OSError, match="synthetic"):
            evaluator._atomic_json_cache(
                attempt_dir / "state.json",
                state,
                plan["attempt_protocol"],
            )
    attempt_root = attempt_dir.parent
    orphans = sorted(attempt_root.glob(".state-cache-*.tmp"))
    assert len(orphans) == 1
    assert {path.name for path in attempt_dir.iterdir()} == set(
        plan["attempt_protocol"]["attempt_tree_before_computed_attestation"]
    )
    monkeypatch.setattr(evaluator.os, "replace", original_replace)
    evaluator._atomic_json_cache(
        attempt_dir / "state.json", state, plan["attempt_protocol"]
    )
    assert orphans[0].exists()
    assert sorted(attempt_root.glob(".state-cache-*.tmp")) == orphans


def test_state_cache_conflict_and_partial_journal_stop(tmp_path: Path) -> None:
    plan, lock = _attempt_plan_lock(tmp_path)
    attempt_dir, _slot, _attempt = evaluator.ensure_receipt(
        plan, lock, "2" * 64, "3" * 64, now=lambda: "t0"
    )
    state = json.loads((attempt_dir / "state.json").read_text())
    state["oracle_open_committed"] = True
    (attempt_dir / "state.json").write_bytes(evaluator.canonical_json(state))
    with pytest.raises(evaluator.EvaluationStopped, match="conflicts"):
        evaluator.recover_state_cache(attempt_dir, plan["attempt_protocol"])
    with (attempt_dir / "events.jsonl").open("ab") as stream:
        stream.write(b'{"partial":')
    with pytest.raises(evaluator.EvaluationStopped, match="partial"):
        evaluator.load_event_chain(attempt_dir, plan["attempt_protocol"])


def test_profile_is_closed_and_exact(tmp_path: Path) -> None:
    template = (ROOT / "config/v4_12_unit_retrieval_evaluator.sb").read_text()
    allowed = tmp_path / "allowed"
    allowed.write_text("x")
    rendered = evaluator.render_profile(
        template,
        allowed_files=[allowed],
        staging_root=tmp_path / "stage",
        forbidden_roots=[tmp_path / "forbidden"],
    )
    assert "(deny default)" in rendered
    assert "(deny network*)" in rendered
    assert "(deny process-fork)" in rendered
    assert str(allowed) in rendered
    assert "@@" not in rendered


def test_lock_binds_every_runtime_role_path_and_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _base_plan(tmp_path)
    payload = b"sealed-byte-identity"
    digest = hashlib.sha256(payload).hexdigest()
    plan["input_paths"] = {
        role: str(tmp_path / "sealed" / role) for role in plan["input_paths"]
    }
    plan["input_hashes"] = {role: digest for role in plan["input_paths"]}
    source_hashes = {relative: digest for relative in plan["future_sources"]}
    plan_sha = digest
    lock = {
        "schema_version": evaluator.LOCK_SCHEMA,
        "purpose": evaluator.LOCK_PURPOSE,
        "audit_verdict": evaluator.LOCK_VERDICT,
        "git_commit": "a" * 40,
        "source_hashes": source_hashes,
        "input_paths": plan["input_paths"],
        "input_hashes": plan["input_hashes"],
        "evaluation_spec_sha256": evaluator._evaluation_spec_sha(plan),
        "runtime": plan["runtime"],
        "outputs": plan["outputs"],
        "sandbox": {
            "executable": plan["input_paths"]["sandbox_executable"],
            "executable_sha256": digest,
            "python_framework_app": plan["input_paths"]["python_framework_app"],
            "python_framework_app_sha256": digest,
            "python_framework_library": plan["input_paths"][
                "python_framework_library"
            ],
            "python_framework_library_sha256": digest,
            "git_executable": plan["input_paths"]["git_executable"],
            "git_executable_sha256": digest,
            "system_read_roots": ["/System", "/usr", "/opt/homebrew"],
            "device_read_paths": ["/dev/null", "/dev/urandom", "/dev/fd"],
            "network_allowed": False,
            "fork_allowed": False,
            "write_scope": "PRIVATE_EVALUATOR_STAGING_ONLY",
        },
        "max_rss_bytes": plan["max_rss_bytes"],
    }
    monkeypatch.setattr(evaluator, "_read_path", lambda *_args, **_kwargs: payload)

    def fake_git(_repo, *args, binary=False):
        if args[:2] == ("cat-file", "-t"):
            return "commit\n"
        return payload if binary else payload.decode()

    monkeypatch.setattr(evaluator, "_git", fake_git)
    evaluator.validate_lock(plan, lock, tmp_path, plan_sha)
    substituted = copy.deepcopy(lock)
    substituted["sandbox"]["python_framework_app"] = str(
        tmp_path / "attacker-python"
    )
    with pytest.raises(evaluator.EvaluationStopped, match="role binding"):
        evaluator.validate_lock(plan, substituted, tmp_path, plan_sha)


def test_parent_rss_monitor_kills_process_above_limit() -> None:
    with pytest.raises(evaluator.EvaluationStopped, match="RSS limit exceeded"):
        evaluator._run_with_rss_ceiling(
            [
                sys.executable,
                "-c",
                "x=bytearray(128*1024*1024)",
            ],
            pass_fds=(),
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            limit=64 * 1024 * 1024,
        )


@pytest.mark.skipif(
    platform.system() != "Darwin" or not Path("/usr/bin/sandbox-exec").is_file(),
    reason="native Seatbelt test requires macOS",
)
def test_native_sandbox_worker_uses_only_synthetic_fds(tmp_path: Path) -> None:
    plan, spec, fds = _worker_fixture(tmp_path)
    python_app = Path(plan["input_paths"]["python_framework_app"])
    if not python_app.is_file():
        pytest.skip("locked Python app is unavailable")
    lock = {
        "sandbox": {
            "executable": "/usr/bin/sandbox-exec",
            "python_framework_app": str(python_app),
            "system_read_roots": ["/System", "/usr", "/opt/homebrew"],
            "device_read_paths": ["/dev/null", "/dev/urandom", "/dev/fd"],
        },
        "input_paths": {
            **{role: path for role, path in spec["input_paths"].items()},
        },
        "source_hashes": {
            "scripts/evaluate_v412_unit_retrieval.py": hashlib.sha256(
                (ROOT / "scripts/evaluate_v412_unit_retrieval.py").read_bytes()
            ).hexdigest(),
            "config/v4_12_unit_retrieval_evaluator.sb": hashlib.sha256(
                (ROOT / "config/v4_12_unit_retrieval_evaluator.sb").read_bytes()
            ).hexdigest(),
        },
    }
    runtime_fds = {
        role: os.open(Path(plan["input_paths"][role]), os.O_RDONLY)
        for role in (
            "sandbox_executable",
            "python_framework_app",
            "python_framework_library",
        )
    }
    execution = evaluator._prepare_worker_execution(
        plan,
        lock,
        runtime_fds,
        Path(spec["evaluation_stage"]).parent,
    )
    try:
        evaluator._invoke_worker(
            plan,
            lock,
            spec,
            fds,
            Path(spec["evaluation_stage"]).parent,
            execution,
        )
        evaluator.validate_packages(
            Path(spec["evaluation_stage"]),
            Path(spec["audit_stage"]),
            evaluator_build_id=spec["evaluator_build_id"],
            attempt_id=spec["attempt_id"],
            artifact=plan["artifact_contract"],
            validation=_package_validation(spec, fds),
        )
    finally:
        for fd in [
            *fds.values(),
            *runtime_fds.values(),
            *execution["fds"],
        ]:
            os.close(fd)


@pytest.mark.skipif(
    platform.system() != "Darwin" or not Path("/usr/bin/sandbox-exec").is_file(),
    reason="native Seatbelt test requires macOS",
)
def test_worker_is_already_loaded_before_post_verify_substitution(
    tmp_path: Path,
) -> None:
    plan, spec, fds = _worker_fixture(tmp_path)
    lock = {
        "sandbox": {
            "executable": "/usr/bin/sandbox-exec",
            "system_read_roots": ["/System", "/usr", "/opt/homebrew"],
            "device_read_paths": ["/dev/null", "/dev/urandom", "/dev/fd"],
        },
        "source_hashes": {
            "scripts/evaluate_v412_unit_retrieval.py": hashlib.sha256(
                (ROOT / "scripts/evaluate_v412_unit_retrieval.py").read_bytes()
            ).hexdigest(),
            "config/v4_12_unit_retrieval_evaluator.sb": hashlib.sha256(
                (ROOT / "config/v4_12_unit_retrieval_evaluator.sb").read_bytes()
            ).hexdigest(),
        },
    }
    runtime_fds = {
        role: os.open(Path(plan["input_paths"][role]), os.O_RDONLY)
        for role in (
            "sandbox_executable",
            "python_framework_app",
            "python_framework_library",
        )
    }
    execution = evaluator._prepare_worker_execution(
        plan,
        lock,
        runtime_fds,
        Path(spec["evaluation_stage"]).parent,
    )
    oracle_roles = evaluator.ORACLE_ROLE_ORDER
    pre_fds = {role: fd for role, fd in fds.items() if role not in oracle_roles}
    pre_spec = dict(spec)
    pre_spec["input_fds"] = pre_fds
    handle = None
    try:
        handle = evaluator._start_worker_pre_oracle(
            plan,
            lock,
            pre_spec,
            pre_fds,
            Path(spec["evaluation_stage"]).parent,
            execution,
        )
        assert handle["process"].returncode is None
        source = Path(execution["source"])
        os.chmod(execution["root"], 0o700)
        replacement = source.with_name("post-verify-attacker.py")
        replacement.write_text("raise SystemExit('attacker ran')\n")
        os.replace(replacement, source)
        evaluator._send_oracle_fds(handle, oracle_roles, fds)
        evaluator._finish_worker(handle)
        handle = None
        evaluator.validate_packages(
            Path(spec["evaluation_stage"]),
            Path(spec["audit_stage"]),
            evaluator_build_id=spec["evaluator_build_id"],
            attempt_id=spec["attempt_id"],
            artifact=plan["artifact_contract"],
            validation=_package_validation(spec, fds),
        )
    finally:
        if handle is not None:
            evaluator._abort_worker(handle)
        for fd in [
            *fds.values(),
            *runtime_fds.values(),
            *execution["fds"],
        ]:
            try:
                os.close(fd)
            except OSError:
                pass


@pytest.mark.skipif(
    platform.system() != "Darwin" or not Path("/usr/bin/sandbox-exec").is_file(),
    reason="native Seatbelt test requires macOS",
)
def test_ready_pre_oracle_crash_restarts_same_attempt_with_fresh_launch_root(
    tmp_path: Path,
) -> None:
    plan, spec, fds = _worker_fixture(tmp_path)
    lock = {
        "sandbox": {
            "executable": "/usr/bin/sandbox-exec",
            "system_read_roots": ["/System", "/usr", "/opt/homebrew"],
            "device_read_paths": ["/dev/null", "/dev/urandom", "/dev/fd"],
        },
        "source_hashes": {
            "scripts/evaluate_v412_unit_retrieval.py": hashlib.sha256(
                (ROOT / "scripts/evaluate_v412_unit_retrieval.py").read_bytes()
            ).hexdigest(),
            "config/v4_12_unit_retrieval_evaluator.sb": hashlib.sha256(
                (ROOT / "config/v4_12_unit_retrieval_evaluator.sb").read_bytes()
            ).hexdigest(),
        },
    }
    runtime_fds = {
        role: os.open(Path(plan["input_paths"][role]), os.O_RDONLY)
        for role in (
            "sandbox_executable",
            "python_framework_app",
            "python_framework_library",
        )
    }
    stage_base = Path(spec["evaluation_stage"]).parent
    oracle_roles = evaluator.ORACLE_ROLE_ORDER
    pre_fds = {role: fd for role, fd in fds.items() if role not in oracle_roles}
    pre_spec = dict(spec)
    pre_spec["input_fds"] = pre_fds
    first_execution = evaluator._prepare_worker_execution(
        plan, lock, runtime_fds, stage_base
    )
    second_execution = None
    first_handle = None
    second_handle = None
    try:
        first_handle = evaluator._start_worker_pre_oracle(
            plan, lock, pre_spec, pre_fds, stage_base, first_execution
        )
        assert first_handle["process"].returncode is None
        evaluator._abort_worker(first_handle)
        first_handle = None
        # The first launch remains untouched. The identical attempt/policy
        # receives a new sealed launch root and a new exclusive spec path.
        second_execution = evaluator._prepare_worker_execution(
            plan, lock, runtime_fds, stage_base
        )
        assert second_execution["root"] != first_execution["root"]
        second_handle = evaluator._start_worker_pre_oracle(
            plan, lock, pre_spec, pre_fds, stage_base, second_execution
        )
        evaluator._send_oracle_fds(second_handle, oracle_roles, fds)
        evaluator._finish_worker(second_handle)
        second_handle = None
        evaluator.validate_packages(
            Path(spec["evaluation_stage"]),
            Path(spec["audit_stage"]),
            evaluator_build_id=spec["evaluator_build_id"],
            attempt_id=spec["attempt_id"],
            artifact=plan["artifact_contract"],
            validation=_package_validation(spec, fds),
        )
    finally:
        if first_handle is not None:
            evaluator._abort_worker(first_handle)
        if second_handle is not None:
            evaluator._abort_worker(second_handle)
        for fd in [
            *fds.values(),
            *runtime_fds.values(),
            *first_execution["fds"],
            *(second_execution["fds"] if second_execution else []),
        ]:
            try:
                os.close(fd)
            except OSError:
                pass


def test_private_sealed_worker_source_detects_substitution(tmp_path: Path) -> None:
    plan, _spec, data_fds = _worker_fixture(tmp_path)
    lock = {
        "sandbox": {
            "executable": "/usr/bin/sandbox-exec",
            "system_read_roots": ["/System", "/usr", "/opt/homebrew"],
            "device_read_paths": ["/dev/null", "/dev/urandom", "/dev/fd"],
        },
        "source_hashes": {
            "scripts/evaluate_v412_unit_retrieval.py": hashlib.sha256(
                (ROOT / "scripts/evaluate_v412_unit_retrieval.py").read_bytes()
            ).hexdigest(),
            "config/v4_12_unit_retrieval_evaluator.sb": hashlib.sha256(
                (ROOT / "config/v4_12_unit_retrieval_evaluator.sb").read_bytes()
            ).hexdigest(),
        },
    }
    runtime_fds = {
        role: os.open(Path(plan["input_paths"][role]), os.O_RDONLY)
        for role in (
            "sandbox_executable",
            "python_framework_app",
            "python_framework_library",
        )
    }
    execution = evaluator._prepare_worker_execution(
        plan, lock, runtime_fds, tmp_path / "sealed-stage"
    )
    try:
        source = Path(execution["source"])
        os.chmod(execution["root"], 0o700)
        os.chmod(source, 0o600)
        replacement = source.with_name("attacker")
        replacement.write_bytes(b"print('substituted')\n")
        os.replace(replacement, source)
        with pytest.raises(evaluator.EvaluationStopped, match="substitution"):
            evaluator._verify_sealed_execution(
                execution, plan["max_rss_bytes"]
            )
    finally:
        for fd in [
            *data_fds.values(),
            *runtime_fds.values(),
            *execution["fds"],
        ]:
            os.close(fd)


def test_final_evaluation_without_audit_is_stop(tmp_path: Path) -> None:
    plan, lock = _attempt_plan_lock(tmp_path)
    attempt_dir, _slot, attempt_id = evaluator.ensure_receipt(
        plan, lock, "2" * 64, "3" * 64, now=lambda: "t0"
    )
    evaluator_id = "e" * 64
    final_eval = Path(plan["outputs"]["evaluation_root"]) / evaluator_id
    final_eval.mkdir(parents=True)
    with pytest.raises(evaluator.EvaluationStopped, match="publication artifact"):
        evaluator.recover_publication(
            plan, attempt_dir, evaluator_id, attempt_id, {}
        )


def test_post_oracle_recovery_before_computed_event_is_stopped(
    tmp_path: Path,
) -> None:
    plan, spec, fds = _worker_fixture(tmp_path)
    lock = {"input_hashes": spec["input_hashes"]}
    attempt_dir, _slot, attempt_id = evaluator.ensure_receipt(
        plan, lock, "2" * 64, "3" * 64, now=lambda: "t0"
    )
    spec.update(
        {
            "attempt_id": attempt_id,
            "plan_sha256": "2" * 64,
            "lock_sha256": "3" * 64,
        }
    )
    evaluator.append_event(
        attempt_dir,
        plan["attempt_protocol"],
        state="STARTED",
        phase="ORACLE_OPEN_COMMITTED",
        oracle_open_committed=True,
        evaluator_build_id=spec["evaluator_build_id"],
        now=lambda: "t1",
    )
    try:
        stage_eval, stage_audit = evaluator.worker_execute(spec)
        _attestation_sha, validation = _write_computed_attestation(
            plan, spec, fds, attempt_dir, stage_eval, stage_audit
        )
        validation.pop("input_snapshots")
        with pytest.raises(
            evaluator.EvaluationStopped,
            match="lacks committed computed attestation",
        ):
            evaluator.recover_publication(
                plan,
                attempt_dir,
                spec["evaluator_build_id"],
                attempt_id,
                validation,
            )
        evaluator.mark_attempt_stopped(
            attempt_dir,
            plan["attempt_protocol"],
            "CRASH_BEFORE_COMPUTED_EVENT",
        )
        state = evaluator.recover_state_cache(
            attempt_dir, plan["attempt_protocol"]
        )
        assert state["state"] == "STOPPED"
        assert state["computed_attestation_sha256"] is None
        assert {path.name for path in attempt_dir.iterdir()} == set(
            plan["attempt_protocol"]["attempt_tree_with_computed_attestation"]
        )
    finally:
        for fd in fds.values():
            os.close(fd)


def test_pending_crash_recovers_by_promotion_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, spec, fds = _worker_fixture(tmp_path)
    lock = {"input_hashes": spec["input_hashes"]}
    attempt_dir, _slot, attempt_id = evaluator.ensure_receipt(
        plan, lock, "2" * 64, "3" * 64, now=lambda: "t0"
    )
    spec["attempt_id"] = attempt_id
    spec["plan_sha256"] = "2" * 64
    spec["lock_sha256"] = "3" * 64
    evaluator.append_event(
        attempt_dir,
        plan["attempt_protocol"],
        state="STARTED",
        phase="ORACLE_OPEN_COMMITTED",
        oracle_open_committed=True,
        evaluator_build_id=spec["evaluator_build_id"],
        reason_code=None,
        now=lambda: "t1",
    )
    try:
        evaluation_stage, audit_stage = evaluator.worker_execute(spec)
        attestation_sha, validation = _write_computed_attestation(
            plan,
            spec,
            fds,
            attempt_dir,
            evaluation_stage,
            audit_stage,
        )
        evaluator.append_event(
            attempt_dir,
            plan["attempt_protocol"],
            state="RECOVERABLE",
            phase="COMPUTED_STAGING_VALID",
            oracle_open_committed=True,
            evaluator_build_id=spec["evaluator_build_id"],
            computed_attestation_sha256=attestation_sha,
            reason_code=None,
            now=lambda: "t2",
        )
        pending_eval = Path(plan["outputs"]["evaluation_root"]) / (
            f".pending-{spec['evaluator_build_id']}-{attempt_id}"
        )
        pending_audit = Path(plan["outputs"]["audit_root"]) / (
            f".pending-{spec['evaluator_build_id']}-{attempt_id}"
        )
        evaluator._promote(audit_stage, pending_audit)
        evaluator._promote(evaluation_stage, pending_eval)
        evaluator.append_event(
            attempt_dir,
            plan["attempt_protocol"],
            state="RECOVERABLE",
            phase="PENDING_BOTH_VALID",
            oracle_open_committed=True,
            evaluator_build_id=spec["evaluator_build_id"],
            reason_code="SYNTHETIC_CRASH_POINT",
            now=lambda: "t3",
        )
        recovery_validation = dict(validation)
        recovery_validation.pop("input_snapshots")
        monkeypatch.setattr(
            evaluator,
            "_open_roles",
            lambda *_args, **_kwargs: pytest.fail(
                "post-oracle recovery reopened inputs"
            ),
        )
        final_eval, final_audit = evaluator.recover_publication(
            plan,
            attempt_dir,
            spec["evaluator_build_id"],
            attempt_id,
            recovery_validation,
        )
        assert final_eval.is_dir() and final_audit.is_dir()
        state = evaluator.recover_state_cache(
            attempt_dir, plan["attempt_protocol"]
        )
        assert state["state"] == "FINAL"
        assert not pending_eval.exists() and not pending_audit.exists()
    finally:
        for fd in fds.values():
            os.close(fd)


@pytest.mark.parametrize(
    "crash_window",
    (
        "AUDIT_PENDING_ONLY",
        "BOTH_PENDING_BEFORE_EVENT",
        "FINAL_AUDIT_BEFORE_EVENT",
        "FINAL_EVALUATION_BEFORE_EVENT",
    ),
)
def test_every_publication_crash_window_is_promotion_only_and_idempotent(
    tmp_path: Path, crash_window: str
) -> None:
    plan, spec, fds = _worker_fixture(tmp_path)
    lock = {"input_hashes": spec["input_hashes"]}
    attempt_dir, _slot, attempt_id = evaluator.ensure_receipt(
        plan, lock, "2" * 64, "3" * 64, now=lambda: "t0"
    )
    spec["attempt_id"] = attempt_id
    spec["plan_sha256"] = "2" * 64
    spec["lock_sha256"] = "3" * 64
    stage_base = Path(plan["outputs"]["temp_root"]) / attempt_id
    stage_base.mkdir(parents=True)
    spec["evaluation_stage"] = str(stage_base / "evaluation.stage")
    spec["audit_stage"] = str(stage_base / "audit.stage")
    evaluator.append_event(
        attempt_dir,
        plan["attempt_protocol"],
        state="STARTED",
        phase="ORACLE_OPEN_COMMITTED",
        oracle_open_committed=True,
        evaluator_build_id=spec["evaluator_build_id"],
        reason_code=None,
        now=lambda: "t1",
    )
    try:
        stage_eval, stage_audit = evaluator.worker_execute(spec)
        pending_eval = Path(plan["outputs"]["evaluation_root"]) / (
            f".pending-{spec['evaluator_build_id']}-{attempt_id}"
        )
        pending_audit = Path(plan["outputs"]["audit_root"]) / (
            f".pending-{spec['evaluator_build_id']}-{attempt_id}"
        )
        final_eval = Path(plan["outputs"]["evaluation_root"]) / spec[
            "evaluator_build_id"
        ]
        final_audit = Path(plan["outputs"]["audit_root"]) / spec[
            "evaluator_build_id"
        ]
        attestation_sha, validation = _write_computed_attestation(
            plan, spec, fds, attempt_dir, stage_eval, stage_audit
        )
        evaluator.append_event(
            attempt_dir,
            plan["attempt_protocol"],
            state="RECOVERABLE",
            phase="COMPUTED_STAGING_VALID",
            oracle_open_committed=True,
            evaluator_build_id=spec["evaluator_build_id"],
            computed_attestation_sha256=attestation_sha,
            reason_code=None,
            now=lambda: "t2",
        )
        evaluator._promote(stage_audit, pending_audit)
        if crash_window in {
            "BOTH_PENDING_BEFORE_EVENT",
            "FINAL_AUDIT_BEFORE_EVENT",
            "FINAL_EVALUATION_BEFORE_EVENT",
        }:
            evaluator._promote(stage_eval, pending_eval)
        if crash_window in {
            "FINAL_AUDIT_BEFORE_EVENT",
            "FINAL_EVALUATION_BEFORE_EVENT",
        }:
            evaluator.append_event(
                attempt_dir,
                plan["attempt_protocol"],
                state="RECOVERABLE",
                phase="PENDING_BOTH_VALID",
                oracle_open_committed=True,
                evaluator_build_id=spec["evaluator_build_id"],
                reason_code=None,
                now=lambda: "t3",
            )
            evaluator._promote(pending_audit, final_audit)
            evaluator._freeze(final_audit)
        if crash_window == "FINAL_EVALUATION_BEFORE_EVENT":
            evaluator.append_event(
                attempt_dir,
                plan["attempt_protocol"],
                state="RECOVERABLE",
                phase="AUDIT_FINAL",
                oracle_open_committed=True,
                evaluator_build_id=spec["evaluator_build_id"],
                reason_code=None,
                now=lambda: "t4",
            )
            evaluator._promote(pending_eval, final_eval)
            evaluator._freeze(final_eval)
        recovery_validation = dict(validation)
        recovery_validation.pop("input_snapshots")
        recovered = evaluator.recover_publication(
            plan,
            attempt_dir,
            spec["evaluator_build_id"],
            attempt_id,
            recovery_validation,
        )
        assert recovered == (final_eval, final_audit)
        assert evaluator.recover_publication(
            plan,
            attempt_dir,
            spec["evaluator_build_id"],
            attempt_id,
            recovery_validation,
        ) == recovered
        assert evaluator.recover_state_cache(
            attempt_dir, plan["attempt_protocol"]
        )["state"] == "FINAL"
    finally:
        for fd in fds.values():
            os.close(fd)


def test_publication_is_non_clobber(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "value").write_text("new")
    (destination / "value").write_text("sealed")
    with pytest.raises(evaluator.EvaluationStopped, match="already exists"):
        evaluator._promote(source, destination)
    assert (destination / "value").read_text() == "sealed"
    assert (source / "value").read_text() == "new"


def test_exclusive_promotion_has_no_exists_rename_race(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    queue = context.Queue()
    destination = tmp_path / "published"
    sources = [tmp_path / "source-a", tmp_path / "source-b"]
    for index, source in enumerate(sources):
        source.mkdir()
        (source / "winner").write_text(str(index))
    processes = [
        context.Process(
            target=_promotion_child,
            args=(str(ROOT), str(source), str(destination), start, queue),
        )
        for source in sources
    ]
    for process in processes:
        process.start()
    start.set()
    results = [queue.get(timeout=10) for _ in processes]
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0
    assert sorted(result[0] for result in results) == ["BLOCKED", "PROMOTED"]
    assert (destination / "winner").read_text() in {"0", "1"}
    assert sum(source.exists() for source in sources) == 1


def test_measurement_slot_lock_is_interprocess_exclusive(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    attempt_root = tmp_path / "attempts"
    with evaluator._exclusive_slot_lock(attempt_root, "s" * 64) as record:
        child = context.Process(
            target=_slot_lock_child,
            args=(str(ROOT), str(attempt_root), "s" * 64, queue),
        )
        child.start()
        assert queue.get(timeout=10) == "BLOCKED"
        child.join(timeout=10)
        assert child.exitcode == 0
    lock_path = attempt_root / record["lock_name"]
    lock_stat = lock_path.stat()
    assert stat.S_ISREG(lock_stat.st_mode)
    assert lock_stat.st_uid == os.geteuid()
    assert stat.S_IMODE(lock_stat.st_mode) == 0o600
    with evaluator._exclusive_slot_lock(attempt_root, "s" * 64):
        pass
    assert lock_path.exists()


def test_slot_path_swap_cannot_create_a_second_lock(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    attempt_root = tmp_path / "attempts"
    slot = "f" * 64
    with evaluator._exclusive_slot_lock(attempt_root, slot) as record:
        lock_path = attempt_root / record["lock_name"]
        displaced = attempt_root / "displaced-lock"
        os.replace(lock_path, displaced)
        lock_path.write_bytes(b"attacker replacement")
        with pytest.raises(evaluator.EvaluationStopped, match="substitution"):
            evaluator._verify_slot_lock(record)
        child = context.Process(
            target=_slot_lock_child,
            args=(str(ROOT), str(attempt_root), slot, queue),
        )
        child.start()
        assert queue.get(timeout=10) == "BLOCKED"
        child.join(timeout=10)
        assert child.exitcode == 0


def test_pre_oracle_resnapshot_detects_substitution(tmp_path: Path) -> None:
    path = tmp_path / "input"
    path.write_bytes(b"sealed")
    limit = 512 * 1024 * 1024
    fd, before = evaluator._open_locked(
        path, hashlib.sha256(b"sealed").hexdigest(), limit
    )
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"changed")
    os.replace(replacement, path)
    try:
        with pytest.raises(evaluator.EvaluationStopped, match="changed before oracle"):
            evaluator._resnapshot({"role": fd}, {"role": before}, limit)
    finally:
        os.close(fd)


def test_cli_smoke_never_needs_plan_or_lock(capsys: pytest.CaptureFixture[str]) -> None:
    assert evaluator.main(["--smoke"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "SMOKE_OK"
