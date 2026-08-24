from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import pyarrow as pa
import pyarrow.parquet as pq

import xgb_matcher.retrieval_fold_fused_logical_io_v13 as logical_io
import xgb_matcher.retrieval_fold_fused_physical_benchmark_v14 as physical
from xgb_matcher.retrieval_fold_fused_oof_v12 import sha256_file
from xgb_matcher.retrieval_fold_fused_physical_benchmark_v12 import (
    PhysicalResourceMetricsV12,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/retrieval_all_human_v14_physical_execution.json"
DOC = ROOT / "docs/retrieval_all_human_v14_physical_execution_contract.md"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return sha256_file(path)


def _sha_map(prefix: str) -> dict[str, str]:
    return {
        name: _digest(f"{prefix}-{name}")
        for name in physical._REQUIRED_LOGICAL_DIGESTS
    }


def _anchor(tmp_path: Path) -> physical.ImmutableW1AnchorV13:
    benchmark = tmp_path / "w1" / "manifest.json"
    trial = tmp_path / "w1" / "trial_w1" / "manifest.json"
    benchmark_sha = _write_json(benchmark, {"kind": "immutable-v12-w1"})
    trial_sha = _write_json(trial, {"kind": "immutable-v12-trial"})
    return physical.ImmutableW1AnchorV13(
        benchmark_manifest_path=benchmark,
        benchmark_manifest_sha256=benchmark_sha,
        trial_manifest_path=trial,
        trial_manifest_sha256=trial_sha,
        wall_seconds=1918.9878759580315,
        materialized_bytes=464378945,
        query_count=512,
        folds=(2, 3, 4),
    )


def _product(
    path: Path,
    *,
    implementation_sha: str,
    plan_sha: str,
    prefit_sha: str,
    rehydration: bool | None,
) -> physical.FragmentW3TrialProductV13:
    manifest_sha = _write_json(path, {"kind": "fresh-v14-w3"})
    roots = _sha_map("exact")
    return physical.FragmentW3TrialProductV13(
        manifest_path=path,
        manifest_sha256=manifest_sha,
        query_count=512,
        folds=(2, 3, 4),
        v13_implementation_sha256=implementation_sha,
        sharded_execution_plan_sha256=plan_sha,
        prefit_parity_manifest_sha256=prefit_sha,
        lineage={
            "scene": _digest("scene"),
            "sharded_implementation_sha256": _digest("sharded"),
            "runtime_environment_sha256": _digest("runtime"),
            "v13_implementation_sha256": implementation_sha,
        },
        logical_digests=roots,
        immutable_w1_logical_digests=roots,
        materialized_bytes=300_000_000,
        projected_national_storage_bytes=10 * 1024**3,
        artifact_inventory_sha256=_digest("inventory"),
        fragment_dataset_manifest_sha256=_digest("fragments"),
        replay_dataset_manifest_sha256=_digest("replay"),
        physical_merge_files_present=False,
        full_payload_outside_top100=False,
        compact_rehydration_exact=rehydration,
    )


def _resources() -> PhysicalResourceMetricsV12:
    return PhysicalResourceMetricsV12(
        wall_seconds=700.0,
        coordinator_cpu_seconds=1.0,
        children_cpu_seconds=2.0,
        machine_busy_cpu_seconds=3.0,
        peak_process_tree_rss_bytes=8 * 1024**3,
        peak_machine_used_memory_bytes=12 * 1024**3,
        minimum_machine_available_memory_bytes=4 * 1024**3,
        swap_out_delta_bytes=0,
        process_read_bytes=1,
        process_write_bytes=2,
        machine_read_bytes=3,
        machine_write_bytes=4,
        sample_interval_seconds=0.05,
    )


def _logical(tmp_path: Path) -> physical.LogicalIOEvidenceV13:
    manifest = tmp_path / "logical" / "manifest.json"
    digest = _write_json(manifest, {"kind": "logical-io-v14"})
    return physical.LogicalIOEvidenceV13(
        manifest_path=manifest,
        manifest_sha256=digest,
        status="PASS_V14_IO",
        gate_pass=True,
        projection_sha256=(
            "c47aa0d64d96ad756df1a0b333328df69c13071d55861f85ba5b808e6813e9e8"
        ),
        unique_row_groups=368,
        logical_compressed_bytes=2696076099,
        cross_actor_overlap_count=0,
        w3_union_equals_w1=True,
        logical_amplification_ratio=1.0,
        materialized_ratio=0.75,
    )


def test_v14_preregistration_is_distinct_and_terminal_v13_is_pinned() -> None:
    config = physical.load_physical_config_v14(CONFIG)
    assert sha256_file(CONFIG) == physical.CONFIG_SHA256
    assert sha256_file(DOC) == "1c0aeaa1e147ba8d725e4a034a21dbb288405281b815f3e0a0a399591cc72427"
    assert config["revision_scope"] == "PHYSICAL_HASH_PROTOCOL_ONLY"
    assert config["scientific_contract_changed"] is False
    assert config["terminal_v13_tombstone"] == {
        "status": "TERMINAL_STOP_V13_NO_RETRY",
        "commit": "914c475",
        "receipt_id": "bdcba90d29bd7da0e84fb805936fe0f99f8518ce9661a30751412c005f6d6799",
        "v13_salvage_allowed": False,
        "v13_retry_allowed": False,
        "prior_draft_receipt_authority": False,
    }
    assert config["authorities"]["v13_terminal_report"]["sha256"] == (
        "bb26ca8c8876ba0f2ba415d03e2511f24e79810b257ebaa3448b0c2b64c94bf4"
    )
    assert config["immutable_v12_stop"]["w1_trial"]["manifest_sha256"] == (
        "67d37417237d0e93b933c95759699e871d5b3a825cb4a03deeed470391a36a74"
    )
    assert "v14" in config["benchmark_512"]["canonical_output_dir"]
    assert "v14" in config["national_execution_protocol"]["canonical_output_dir"]
    assert config["benchmark_512"]["fresh_trial_count"] == 1
    assert config["benchmark_512"]["v13_artifact_salvage_allowed"] is False


def test_v14_marker_is_canonical_irreversible_and_strict(tmp_path: Path) -> None:
    output = tmp_path / "benchmark-v14"
    marker = tmp_path / ".v14-attempt.json"
    config = {
        "benchmark_512": {
            "canonical_output_dir": str(output),
            "attempt_consumed_marker_path": str(marker),
        }
    }
    observed, digest = physical._consume_attempt(config, output)
    assert observed == marker.resolve()
    physical._validate_attempt(config, observed, digest)
    with pytest.raises(physical.PhysicalBenchmarkInvariantErrorV14):
        physical._consume_attempt(config, output)
    with pytest.raises(physical.PhysicalBenchmarkInvariantErrorV14):
        physical._consume_attempt(config, tmp_path / "alternate")


def test_v14_post_o_excl_fault_can_publish_durable_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "benchmark-v14"
    marker = tmp_path / ".v14-attempt.json"
    config = {
        "benchmark_512": {
            "canonical_output_dir": str(output),
            "attempt_consumed_marker_path": str(marker),
        }
    }
    real_fsync = physical._fsync_directory
    monkeypatch.setattr(
        physical,
        "_fsync_directory",
        lambda _path: (_ for _ in ()).throw(OSError("fsync fault")),
    )
    with pytest.raises(physical._PostOExclMarkerErrorV14) as failure:
        physical._consume_attempt(config, output)
    assert marker.exists()
    monkeypatch.setattr(physical, "_fsync_directory", real_fsync)
    manifest = physical._publish_exception_stop(
        output=output,
        config_path=CONFIG,
        marker=marker,
        frozen_sha256=_digest("v14-implementation"),
        frozen_files=(),
        phase="ATOMIC_ATTEMPT_MARKER_CREATION",
        error=failure.value,
    )
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    assert raw["status"] == "STOP_V14_EXECUTION_EXCEPTION"
    assert raw["execution_authorization_published"] is False
    assert raw["automatic_retry_allowed"] is False
    assert marker.exists()


def test_v14_functional_projection_recomputes_and_rejects_forged_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    anchor = _anchor(tmp_path)
    prefit = tmp_path / "prefit.json"
    prefit_sha = _write_json(prefit, {"kind": "prefit"})
    input_path = tmp_path / "w1-input.parquet"
    input_path.write_bytes(b"sealed-input")
    inputs = [
        {
            "role": "w1",
            "path": str(input_path.resolve()),
            "sha256": sha256_file(input_path),
            "size_bytes": input_path.stat().st_size,
        }
    ]
    roots = _sha_map("v14")
    monkeypatch.setattr(
        physical,
        "_compute_w1_functional_projection",
        lambda *args, **kwargs: (dict(roots), list(inputs)),
    )
    monkeypatch.setattr(
        physical,
        "v13_implementation_lineage",
        lambda _root: (_digest("v13-engine"), ()),
    )
    monkeypatch.setattr(
        physical,
        "v14_implementation_lineage",
        lambda _root: (_digest("v14-outer"), ()),
    )
    kwargs = {
        "anchor": anchor,
        "workspace_root": tmp_path,
        "output_root": tmp_path / "projection",
        "prefit_parity_manifest_path": prefit,
        "expected_prefit_parity_manifest_sha256": prefit_sha,
        "expected_prefit_lineage": {"scene": _digest("scene")},
        "sample_query_ids": tuple(f"q{i:03d}" for i in range(512)),
    }
    projection = physical.build_immutable_w1_functional_projection_v14(**kwargs)
    assert len(projection.manifest_path.parent.name) == 64
    assert projection.manifest_path.parent.name == projection.projection_id
    raw = json.loads(projection.manifest_path.read_text(encoding="utf-8"))
    raw["logical_digests"]["outcomes"] = _digest("forged")
    raw["identity"]["logical_digests"]["outcomes"] = _digest("forged")
    _write_json(projection.manifest_path, raw)
    with pytest.raises(physical.PhysicalBenchmarkInvariantErrorV14):
        physical.build_immutable_w1_functional_projection_v14(**kwargs)


def test_v14_prepared_authorities_reject_alternate_plan_before_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = {
        "scene": tmp_path / "scene",
        "labels": tmp_path / "labels.parquet",
        "sidecars": tmp_path / "sidecars",
        "mmap": tmp_path / "mmap",
        "reader": tmp_path / "reader.json",
        "parent": tmp_path / "parent.json",
        "scientific": tmp_path / "scientific.json",
        "executor": tmp_path / "executor.json",
        "source": tmp_path / "source.parquet",
        "prefit": tmp_path / "prefit.json",
    }
    for name in ("scene", "sidecars", "mmap"):
        paths[name].mkdir()
    for name in ("labels", "reader", "parent", "scientific", "executor", "source", "prefit"):
        paths[name].write_text(name + "\n", encoding="utf-8")
    plan = tmp_path / "plan.json"
    plan_raw = {
        "input_lineage": {
            "scene_ir": {"directory": str(paths["scene"])},
            "labels": {"ledger_path": str(paths["labels"])},
            "official_sidecars": {
                "sidecar_manifest_path": str(paths["sidecars"] / "manifest.json"),
                "mmap_manifest_path": str(paths["mmap"] / "manifest.json"),
                "reader_parity_manifest_path": str(paths["reader"]),
            },
            "parent_model_bundle": {"manifest_path": str(paths["parent"])},
            "prefit_parity": {
                "manifest_path": str(paths["prefit"]),
                "manifest_sha256": sha256_file(paths["prefit"]),
            },
        }
    }
    plan_sha = _write_json(plan, plan_raw)
    names = {
        "scene_ir_build_id": "scene_ir_sha256",
        "scene_ir_manifest_sha256": "scene_ir_manifest_sha256",
        "source_candidate_sha256": "source_candidate_sha256",
        "source_manifest_sha256": "source_manifest_sha256",
        "label_ledger_sha256": "label_ledger_sha256",
        "label_ledger_manifest_sha256": "label_ledger_manifest_sha256",
        "official_sidecar_manifest_sha256": "official_sidecar_manifest_sha256",
        "mmap_sidecar_manifest_sha256": "mmap_sidecar_manifest_sha256",
        "reader_parity_manifest_sha256": "reader_parity_evidence_sha256",
        "parent_model_bundle_sha256": "parent_model_bundle_sha256",
        "runtime_query_sample_sha256": "query_sample_sha256",
    }
    input_lineage = {key: _digest(key) for key in names}
    input_lineage["source_candidate_path"] = str(paths["source"])
    sample_path = tmp_path / "runtime-sample.parquet"
    sample_ids = tuple(f"sample-{index}" for index in range(512))
    pq.write_table(
        pa.table({"query_id": sample_ids, "fold": [2] * 512}), sample_path
    )
    input_lineage["runtime_query_sample_path"] = str(sample_path)
    input_lineage["runtime_query_sample_sha256"] = sha256_file(sample_path)
    input_lineage["runtime_query_sample_rows"] = 512
    config = {
        "input_lineage": input_lineage,
        "authorities": {
            "v12_sharded_plan": {"path": str(plan), "sha256": plan_sha},
            "scientific_config": {
                "path": str(paths["scientific"]),
                "sha256": _digest("scientific"),
            },
            "fold_fused_config": {
                "path": str(paths["executor"]),
                "sha256": _digest("executor"),
            },
        },
        "scientific_invariants": {
            "master_recall_at_2000": 0.995,
            "prerank_recall_at_500": 0.994,
            "exact_recall_at_100": 0.992,
            "exact_recall_at_100_wilson95_lower": 0.99,
            "minimum_each_outer_fold_exact_recall_at_100": 0.988,
        },
    }
    lineage = {lineage_name: input_lineage[config_name] for config_name, lineage_name in names.items()}
    lineage["scientific_config_sha256"] = _digest("scientific")
    lineage["executor_config_sha256"] = _digest("executor")
    expected_paths = {
        "workspace_root": str(tmp_path.resolve()),
        "scene_ir": str(paths["scene"].resolve()),
        "labels": str(paths["labels"].resolve()),
        "sidecars": str(paths["sidecars"].resolve()),
        "mmap_sidecar": str(paths["mmap"].resolve()),
        "reader_parity_manifest": str(paths["reader"].resolve()),
        "parent_model_bundle": str(paths["parent"].resolve()),
        "scientific_config": str(paths["scientific"].resolve()),
        "executor_config": str(paths["executor"].resolve()),
        "sharded_execution_plan": str(plan.resolve()),
        "prefit_parity_manifest": str(paths["prefit"].resolve()),
    }
    runtime = {
        "schema_version": physical.RUNTIME_PAYLOAD_SCHEMA_VERSION,
        "paths": expected_paths,
        "lineage": dict(lineage),
        "sharded_execution_plan_sha256": plan_sha,
        "prefit_parity_manifest_sha256": sha256_file(paths["prefit"]),
        "replay_generation_inputs_opened": False,
        "fold0_opened": False,
        "fold1_opened": False,
    }
    national_ids = tuple(f"national-{index}" for index in range(23571))
    admission = SimpleNamespace(
        fit_candidate_sampling={"maximum_candidates_per_query": 128}
    )
    cascade = object()
    executor = object()
    monkeypatch.setattr(
        physical.AdmissionConfigV12, "load", lambda _path: admission
    )
    monkeypatch.setattr(
        physical.RetrievalTop100CascadeV12Config, "load", lambda _path: cascade
    )
    monkeypatch.setattr(
        physical, "load_executor_config_v12", lambda *a, **k: executor
    )
    index = pa.table(
        {
            "query_id": national_ids,
            "fold": [2] * len(national_ids),
            "row_start": list(range(len(national_ids))),
            "row_count": [1] * len(national_ids),
        }
    )
    scene_cache = SimpleNamespace(
            source=paths["source"],
            cache_dir=paths["scene"].resolve(),
            manifest={"build_id": _digest("scene-build")},
            index=index,
    )
    monkeypatch.setattr(
        physical,
        "SceneIRCache",
        lambda *a, **k: SimpleNamespace(
            source=paths["source"].resolve(),
            cache_dir=paths["scene"].resolve(),
            manifest={"build_id": _digest("scene-build")},
            index=index,
        ),
    )
    prepared = SimpleNamespace(
        scene_cache=scene_cache,
        sharded_execution_plan_path=plan,
        sharded_execution_plan_sha256=plan_sha,
        prefit_parity_manifest_path=paths["prefit"],
        prefit_parity_manifest_sha256=sha256_file(paths["prefit"]),
        lineage=lineage,
        gates=physical._frozen_gates_v14(config),
        admission=admission,
        cascade=cascade,
        executor=executor,
        caps=physical.ExecutionCaps(fit_sample=128),
        sample_query_ids=sample_ids,
        national_query_ids=national_ids,
        runtime_payload=runtime,
    )
    physical._validate_prepared_authorities_v14(
        prepared, config=config, workspace_root=tmp_path
    )
    prepared.sharded_execution_plan_path = tmp_path / "alternate-plan.json"
    with pytest.raises(physical.PhysicalBenchmarkInvariantErrorV14):
        physical._validate_prepared_authorities_v14(
            prepared, config=config, workspace_root=tmp_path
        )
    prepared.sharded_execution_plan_path = plan
    prepared.runtime_payload = {**runtime, "unregistered": True}
    with pytest.raises(physical.PhysicalBenchmarkInvariantErrorV14):
        physical._validate_prepared_authorities_v14(
            prepared, config=config, workspace_root=tmp_path
        )
    prepared.runtime_payload = runtime
    prepared.caps = physical.ExecutionCaps(fit_sample=1)
    with pytest.raises(physical.PhysicalBenchmarkInvariantErrorV14):
        physical._validate_prepared_authorities_v14(
            prepared, config=config, workspace_root=tmp_path
        )
    prepared.caps = physical.ExecutionCaps(fit_sample=128)
    prepared.runtime_payload = {**runtime, "schema_version": "forged"}
    with pytest.raises(physical.PhysicalBenchmarkInvariantErrorV14):
        physical._validate_prepared_authorities_v14(
            prepared, config=config, workspace_root=tmp_path
        )
    prepared.runtime_payload = runtime
    prepared.sample_query_ids = tuple(reversed(sample_ids))
    with pytest.raises(physical.PhysicalBenchmarkInvariantErrorV14):
        physical._validate_prepared_authorities_v14(
            prepared, config=config, workspace_root=tmp_path
        )
    prepared.sample_query_ids = sample_ids
    prepared.scene_cache.index = index.set_column(
        1, "fold", pa.array([3] + [2] * (len(national_ids) - 1))
    )
    with pytest.raises(physical.PhysicalBenchmarkInvariantErrorV14):
        physical._validate_prepared_authorities_v14(
            prepared, config=config, workspace_root=tmp_path
        )


def test_v14_assessment_runs_one_rehydration_only_after_timer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "benchmark"
    marker = tmp_path / ".marker"
    marker.write_text("marker\n", encoding="utf-8")
    plan = tmp_path / "plan.json"
    prefit_sha = _digest("prefit")
    plan_sha = _write_json(
        plan,
        {"input_lineage": {"prefit_parity": {"manifest_sha256": prefit_sha}}},
    )
    config = {"authorities": {"v12_sharded_plan": {"path": str(plan), "sha256": plan_sha}}}
    anchor = _anchor(tmp_path)
    implementation = _digest("v13-engine")
    resources = _resources()
    logical = _logical(tmp_path)
    claimed: list[physical.FragmentW3TrialProductV13] = []
    verifies: list[bool] = []

    def execute(**kwargs: Any) -> physical.FragmentW3TrialProductV13:
        value = _product(
            Path(kwargs["output_dir"]) / "manifest.json",
            implementation_sha=implementation,
            plan_sha=plan_sha,
            prefit_sha=prefit_sha,
            rehydration=None,
        )
        claimed.append(value)
        return value

    def reload(*args: Any, **kwargs: Any) -> physical.FragmentW3TrialProductV13:
        verifies.append(bool(kwargs["verify_rehydration"]))
        return replace(claimed[0], compact_rehydration_exact=True)

    monkeypatch.setattr(physical, "load_physical_config_v14", lambda _p: config)
    monkeypatch.setattr(physical, "_validate_prepared_authorities_v14", lambda *a, **k: None)
    monkeypatch.setattr(physical, "_validate_attempt", lambda *a, **k: None)
    monkeypatch.setattr(physical, "load_immutable_w1_anchor_v13", lambda _c: anchor)
    monkeypatch.setattr(physical, "execute_fragment_w3_trial_v14", execute)
    monkeypatch.setattr(physical, "measure_physical_trial_v12", lambda callback: (callback(), resources))
    monkeypatch.setattr(physical, "load_fragment_w3_trial_product_v13", reload)
    monkeypatch.setattr(
        physical,
        "v14_implementation_lineage",
        lambda _root: (_digest("v14"), ()),
    )
    monkeypatch.setattr(physical, "compare_w3_to_immutable_w1_io_v13", lambda **k: object())
    monkeypatch.setattr(physical, "logical_io_evidence_from_gate_v13", lambda _a: logical)
    monkeypatch.setattr(
        physical,
        "_derive_physical_checks_v13",
        lambda **kwargs: ({key: True for key in (
            "logical_bridge_exact", "no_physical_merge_files",
            "top500_compact_top100_full", "compact_rehydration_exact",
            "logical_io_pass", "ram_pass", "swap_pass", "speed_pass",
            "storage_pass")}, 2.5),
    )
    dag = physical._OfficialDAGV14(
        prepared=SimpleNamespace(),
        workspace_root=tmp_path,
        w1_functional_projection=object(),
        w1_io_projection=SimpleNamespace(
            manifest_path=tmp_path / "io.json", manifest_sha256=_digest("io")
        ),
        capability=object(),
    )
    assessment = physical._assess_official(
        config_path=tmp_path / "config.json",
        output=output,
        marker=marker,
        marker_sha256=_digest("marker"),
        dag=dag,
        frozen_sha256=_digest("v14"),
        frozen_files=(),
    )
    assert verifies == [True]
    assert assessment.product.compact_rehydration_exact is True


def test_v14_authorization_reopens_without_second_rehydration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "canonical-benchmark"
    output.mkdir()
    marker = tmp_path / ".marker"
    marker.write_text("marker\n", encoding="utf-8")
    plan = tmp_path / "plan.json"
    prefit_sha = _digest("prefit")
    plan_sha = _write_json(
        plan,
        {"input_lineage": {"prefit_parity": {"manifest_sha256": prefit_sha}}},
    )
    config = {
        "benchmark_512": {
            "canonical_output_dir": str(output),
            "attempt_consumed_marker_path": str(marker),
        },
        "authorities": {
            "v12_sharded_plan": {"path": str(plan), "sha256": plan_sha},
            "v13_stop_receipt": {"sha256": _digest("receipt")},
        },
    }
    anchor = _anchor(tmp_path)
    implementation = _digest("v13-engine")
    product = _product(
        output / "trial_w3" / "manifest.json",
        implementation_sha=implementation,
        plan_sha=plan_sha,
        prefit_sha=prefit_sha,
        rehydration=True,
    )
    logical = _logical(tmp_path)
    resources = _resources()
    checks = {key: True for key in (
        "logical_bridge_exact", "no_physical_merge_files",
        "top500_compact_top100_full", "compact_rehydration_exact",
        "logical_io_pass", "ram_pass", "swap_pass", "speed_pass", "storage_pass")}
    assessment = physical.PhysicalBenchmarkAssessmentV14(
        config_path=tmp_path / "config.json",
        output=output,
        marker=marker.resolve(),
        marker_sha256=_digest("marker"),
        anchor=anchor,
        product=product,
        resources=resources,
        logical=logical,
        checks=checks,
        speedup=1918.9878759580315 / 700.0,
        v14_implementation_sha256=_digest("v14"),
        v14_implementation_files=(),
        official_capability=physical._CAPABILITY_TOKEN_V14,
    )
    monkeypatch.setattr(physical, "load_physical_config_v14", lambda _p: config)
    benchmark_path = output / "manifest.json"
    _write_json(
        benchmark_path,
        physical._benchmark_manifest(
            assessment, status="PASS_V14_PHYSICAL_BENCHMARK"
        ),
    )
    benchmark_sha = sha256_file(benchmark_path)
    auth_path = output / "execution_authorization_v14.json"
    _write_json(
        auth_path,
        physical._authorization_document(assessment, benchmark_sha),
    )
    verifies: list[bool] = []
    monkeypatch.setattr(physical, "load_immutable_w1_anchor_v13", lambda _c: anchor)
    monkeypatch.setattr(physical, "_validate_attempt", lambda *a, **k: None)
    monkeypatch.setattr(
        physical,
        "v14_implementation_lineage",
        lambda _root: (_digest("v14"), ()),
    )
    monkeypatch.setattr(
        physical,
        "load_fragment_w3_trial_product_v13",
        lambda *a, **k: verifies.append(bool(k["verify_rehydration"]))
        or replace(product, compact_rehydration_exact=None),
    )
    monkeypatch.setattr(logical_io, "load_logical_io_gate_v13", lambda *a, **k: object())
    monkeypatch.setattr(physical, "logical_io_evidence_from_gate_v13", lambda _a: logical)
    monkeypatch.setattr(
        physical,
        "_derive_physical_checks_v13",
        lambda **kwargs: (dict(checks), assessment.speedup),
    )
    observed = physical.load_execution_authorization_v14(
        auth_path,
        expected_authorization_sha256=sha256_file(auth_path),
        benchmark_manifest_path=benchmark_path,
        expected_benchmark_manifest_sha256=benchmark_sha,
        workspace_root=tmp_path,
    )
    assert observed["status"] == "AUTHORIZED_V14_W3_FOR_NATIONAL_OOF"
    assert verifies == [False]
