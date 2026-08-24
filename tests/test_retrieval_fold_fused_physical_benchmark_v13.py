from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pytest
import pyarrow as pa
import pyarrow.parquet as pq

import xgb_matcher.retrieval_fold_fused_physical_benchmark_v13 as physical
from xgb_matcher.retrieval_fold_fused_oof_v12 import (
    ExecutionCaps,
    FeatureBatch,
    FullMasterStatistics,
    MasterContext,
    sha256_file,
)
from xgb_matcher.retrieval_fold_fused_physical_benchmark_v12 import (
    PhysicalResourceMetricsV12,
)
from xgb_matcher.retrieval_fold_fused_fragment_dataset_v13 import merkle_root_v13
from xgb_matcher.retrieval_fold_fused_fragment_runtime_v13 import (
    _canonical_query_sequence_from_output_v13,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/retrieval_all_human_v13_physical_execution.json"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return sha256_file(path)


def _sha_map(prefix: str) -> dict[str, str]:
    return {
        name: _digest(f"{prefix}-{name}")
        for name in physical.REQUIRED_LOGICAL_BRIDGE_DIGESTS
    }


def _anchor(tmp_path: Path) -> physical.ImmutableW1AnchorV13:
    benchmark = tmp_path / "immutable_w1" / "manifest.json"
    trial = benchmark.parent / "trial_w1" / "manifest.json"
    trial_sha = _write_json(trial, {"kind": "w1"})
    benchmark_sha = _write_json(benchmark, {"kind": "v12-stop"})
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
    root: Path,
    *,
    implementation_sha256: str,
    sharded_execution_plan_sha256: str,
    prefit_parity_manifest_sha256: str,
    exact: bool = True,
    materialized: int = 300_000_000,
    projected_storage: int = 10 * 1024**3,
    merge: bool = False,
    payload_leak: bool = False,
    rehydration: bool = True,
) -> physical.FragmentW3TrialProductV13:
    manifest = root / "manifest.json"
    manifest_sha = _write_json(manifest, {"kind": "w3"})
    reference = _sha_map("same")
    observed = dict(reference)
    if not exact:
        observed["top100_full"] = _digest("drift")
    return physical.FragmentW3TrialProductV13(
        manifest_path=manifest,
        manifest_sha256=manifest_sha,
        query_count=512,
        folds=(2, 3, 4),
        v13_implementation_sha256=implementation_sha256,
        sharded_execution_plan_sha256=sharded_execution_plan_sha256,
        prefit_parity_manifest_sha256=prefit_parity_manifest_sha256,
        lineage={
            "scene": _digest("scene"),
            "sharded_implementation_sha256": _digest("sharded"),
            "runtime_environment_sha256": _digest("runtime"),
            "v13_implementation_sha256": implementation_sha256,
        },
        logical_digests=observed,
        immutable_w1_logical_digests=reference,
        materialized_bytes=materialized,
        projected_national_storage_bytes=projected_storage,
        artifact_inventory_sha256=_digest("inventory"),
        fragment_dataset_manifest_sha256=_digest("fragments"),
        replay_dataset_manifest_sha256=_digest("replay"),
        physical_merge_files_present=merge,
        full_payload_outside_top100=payload_leak,
        compact_rehydration_exact=rehydration,
    )


def _logical(
    root: Path,
    *,
    gate_pass: bool = True,
    overlap: int = 0,
    union_equal: bool = True,
    logical_ratio: float = 1.0,
    materialized_ratio: float = 0.75,
) -> physical.LogicalIOEvidenceV13:
    manifest = root / "manifest.json"
    manifest_sha = _write_json(manifest, {"kind": "io"})
    return physical.LogicalIOEvidenceV13(
        manifest_path=manifest,
        manifest_sha256=manifest_sha,
        status="PASS_V13_IO" if gate_pass else "STOP_V13_IO",
        gate_pass=gate_pass,
        projection_sha256=(
            "c47aa0d64d96ad756df1a0b333328df69c13071d55861f85ba5b808e6813e9e8"
        ),
        unique_row_groups=368,
        logical_compressed_bytes=2696076099,
        cross_actor_overlap_count=overlap,
        w3_union_equals_w1=union_equal,
        logical_amplification_ratio=logical_ratio,
        materialized_ratio=materialized_ratio,
    )


def _resources(
    *,
    wall: float = 700.0,
    ram: int = 12 * 1024**3,
    swap: int = 0,
    process_io: int = 999_999_999_999,
    machine_io: int = 888_888_888_888,
) -> PhysicalResourceMetricsV12:
    return PhysicalResourceMetricsV12(
        wall_seconds=wall,
        coordinator_cpu_seconds=1.0,
        children_cpu_seconds=2.0,
        machine_busy_cpu_seconds=3.0,
        peak_process_tree_rss_bytes=8 * 1024**3,
        peak_machine_used_memory_bytes=ram,
        minimum_machine_available_memory_bytes=4 * 1024**3,
        swap_out_delta_bytes=swap,
        process_read_bytes=process_io,
        process_write_bytes=0,
        machine_read_bytes=machine_io,
        machine_write_bytes=0,
        sample_interval_seconds=0.05,
    )


def _run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    product_options: dict[str, Any] | None = None,
    logical_options: dict[str, Any] | None = None,
    resources: PhysicalResourceMetricsV12 | None = None,
) -> tuple[Path, list[str]]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    output = tmp_path / "benchmark"
    marker = tmp_path / ".retrieval_v13_w3_attempt_consumed.json"
    config["benchmark_512"]["canonical_output_dir"] = str(output)
    config["benchmark_512"]["attempt_consumed_marker_path"] = str(marker)
    anchor = _anchor(tmp_path)
    calls: list[str] = []
    implementation_sha = _digest("implementation")
    plan_sha = str(config["authorities"]["v12_sharded_plan"]["sha256"])
    plan_path = ROOT / str(config["authorities"]["v12_sharded_plan"]["path"])
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    prefit_sha = str(plan["input_lineage"]["prefit_parity"]["manifest_sha256"])
    monkeypatch.setattr(physical, "load_physical_config_v13", lambda _path: config)
    monkeypatch.setattr(
        physical, "load_immutable_w1_anchor_v13", lambda _config: anchor
    )

    def runner(output: Path, observed_anchor: physical.ImmutableW1AnchorV13):
        calls.append("runner")
        assert observed_anchor == anchor
        return _product(
            output,
            implementation_sha256=implementation_sha,
            sharded_execution_plan_sha256=plan_sha,
            prefit_parity_manifest_sha256=prefit_sha,
            **(product_options or {}),
        )

    def strict_reload(_manifest_path: Path, **kwargs):
        calls.append("rehydrate_post_timer")
        assert kwargs["verify_rehydration"] is True
        return runner_product["value"]

    monkeypatch.setattr(
        physical, "load_fragment_w3_trial_product_v13", strict_reload
    )
    monkeypatch.setattr(
        physical,
        "v13_implementation_lineage",
        lambda _root: (implementation_sha, ()),
    )
    runner_product: dict[str, physical.FragmentW3TrialProductV13] = {}

    def io_builder(product, observed_anchor, output):
        calls.append("logical_io")
        assert observed_anchor == anchor
        assert isinstance(product, physical.FragmentW3TrialProductV13)
        return _logical(output, **(logical_options or {}))

    def measurer(callback):
        calls.append("measure_start")
        value = callback()
        runner_product["value"] = value
        calls.append("measure_stop")
        return value, resources or _resources()

    manifest = physical.run_physical_benchmark_test_harness_v13(
        config_path=CONFIG_PATH,
        output_dir=output,
        runner=runner,
        logical_io_builder=io_builder,
        measurer=measurer,
    )
    return manifest, calls


def _seal_synthetic_official_authorization_for_loader_test(
    manifest_path: Path,
) -> Path:
    """Turn valid harness evidence into a loader-only official fixture.

    The real harness is intentionally unable to do this.  This helper writes
    the exact already-sealed PASS/auth schemas only so the strict reopening
    path can be tested without exposing an injectable production publisher.
    """

    benchmark = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert benchmark["status"] == "STOP_V13_UNTRUSTED_EXECUTION_HOOKS"
    assert all(benchmark["checks"].values())
    benchmark["status"] = "PASS_V13_PHYSICAL_BENCHMARK"
    benchmark["gate_pass"] = True
    benchmark["execution_authorization_published"] = True
    benchmark_sha = _write_json(manifest_path, benchmark)
    identity = benchmark["benchmark_identity"]
    authorization_identity = {
        "engine_id": physical.ENGINE_ID,
        "benchmark_manifest_sha256": benchmark_sha,
        "config_sha256": physical.CONFIG_SHA256,
        "v13_implementation_sha256": identity["v13_implementation_sha256"],
        "immutable_w1_benchmark_sha256": identity[
            "immutable_w1_benchmark_sha256"
        ],
        "w3_trial_manifest_sha256": identity["w3_trial_manifest_sha256"],
        "logical_io_manifest_sha256": identity["logical_io_manifest_sha256"],
        "authorized_worker_count": 3,
        "authorized_scope": "NATIONAL_OOF_FOLDS_2_3_4_PHYSICAL_EXECUTION",
        "sharded_implementation_sha256": identity[
            "sharded_implementation_sha256"
        ],
        "runtime_environment_sha256": identity["runtime_environment_sha256"],
        "attempt_marker_path": identity["attempt_marker_path"],
        "attempt_marker_sha256": identity["attempt_marker_sha256"],
        "sharded_execution_plan_sha256": identity[
            "sharded_execution_plan_sha256"
        ],
        "prefit_parity_manifest_sha256": identity[
            "prefit_parity_manifest_sha256"
        ],
    }
    authorization = {
        "schema_version": physical.AUTHORIZATION_SCHEMA_VERSION,
        "status": "AUTHORIZED_V13_W3_FOR_NATIONAL_OOF",
        "authorization_id": physical._sha256_json(authorization_identity),
        "authorization_identity": authorization_identity,
        "folds": [2, 3, 4],
        "fold0_opened": False,
        "fold1_opened": False,
        "positive_injection": False,
        "automatic_national_launch": False,
    }
    authorization_path = manifest_path.parent / "execution_authorization.json"
    _write_json(authorization_path, authorization)
    return authorization_path


def _rebind_authorization_to_rehashed_benchmark(
    manifest_path: Path, authorization_path: Path
) -> tuple[str, str]:
    benchmark_sha = sha256_file(manifest_path)
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    authorization["authorization_identity"][
        "benchmark_manifest_sha256"
    ] = benchmark_sha
    authorization["authorization_id"] = physical._sha256_json(
        authorization["authorization_identity"]
    )
    authorization_sha = _write_json(authorization_path, authorization)
    return benchmark_sha, authorization_sha


def _patch_authorization_artifact_reopen_for_test(
    monkeypatch: pytest.MonkeyPatch,
    benchmark: Mapping[str, Any],
) -> list[bool]:
    identity = benchmark["benchmark_identity"]
    w3 = benchmark["w3"]
    observed_verify: list[bool] = []

    def load_product(_path: Path, **kwargs):
        observed_verify.append(bool(kwargs["verify_rehydration"]))
        return SimpleNamespace(
            lineage=w3["lineage"],
            bridge_exact=True,
            physical_merge_files_present=False,
            full_payload_outside_top100=False,
            materialized_bytes=w3["materialized_bytes"],
            projected_national_storage_bytes=w3[
                "projected_national_storage_bytes"
            ],
            artifact_inventory_sha256=identity["artifact_inventory_sha256"],
            fragment_dataset_manifest_sha256=identity[
                "fragment_dataset_manifest_sha256"
            ],
            replay_dataset_manifest_sha256=identity[
                "replay_dataset_manifest_sha256"
            ],
            sharded_execution_plan_sha256=identity[
                "sharded_execution_plan_sha256"
            ],
            prefit_parity_manifest_sha256=identity[
                "prefit_parity_manifest_sha256"
            ],
            logical_digests=identity["logical_digests"],
        )

    monkeypatch.setattr(
        physical, "load_fragment_w3_trial_product_v13", load_product
    )
    import xgb_matcher.retrieval_fold_fused_logical_io_v13 as logical_io

    monkeypatch.setattr(
        logical_io,
        "load_logical_io_gate_v13",
        lambda *_args, **_kwargs: SimpleNamespace(
            gate_pass=True, status="PASS_V13_IO"
        ),
    )
    logical_record = benchmark["logical_io"]
    logical_evidence = physical.LogicalIOEvidenceV13(
        manifest_path=Path(logical_record["manifest_path"]),
        manifest_sha256=logical_record["manifest_sha256"],
        status=logical_record["status"],
        gate_pass=logical_record["gate_pass"],
        projection_sha256=logical_record["projection_sha256"],
        unique_row_groups=logical_record["unique_row_groups"],
        logical_compressed_bytes=logical_record["logical_compressed_bytes"],
        cross_actor_overlap_count=logical_record["cross_actor_overlap_count"],
        w3_union_equals_w1=logical_record["w3_union_equals_w1"],
        logical_amplification_ratio=logical_record[
            "logical_amplification_ratio"
        ],
        materialized_ratio=logical_record["materialized_ratio"],
    )
    monkeypatch.setattr(
        physical,
        "logical_io_evidence_from_gate_v13",
        lambda _gate: logical_evidence,
    )
    return observed_verify


def test_amended_config_is_exactly_content_pinned() -> None:
    config = physical.load_physical_config_v13(CONFIG_PATH)
    assert sha256_file(CONFIG_PATH) == physical.CONFIG_SHA256
    assert config["payload_policy"]["prerank500"]["required_compact_fields"] == [
        "query_id",
        "fold",
        "candidate_siret",
        "candidate_parent",
        "rank",
        "site_score",
        "selection_reason",
    ]
    assert config["payload_policy"]["prerank500"]["audit_join_key"] == [
        "query_id",
        "candidate_siret",
    ]
    assert config["payload_policy"]["prerank500"][
        "row_level_hash_repetition_forbidden"
    ] is True
    assert len(config["benchmark_512"]["required_negative_tests"]) == 37
    assert config["benchmark_512"][
        "authorization_capable_entrypoint_owns_preflight_and_w1_projections"
    ] is True
    assert config["benchmark_512"][
        "generic_prepared_or_projection_injection_allowed"
    ] is False
    assert config["cache_neutral_io_gate"]["pre_w3_source_validation"][
        "verify_source_hash_argument"
    ] is False


def test_config_byte_drift_is_rejected(tmp_path: Path) -> None:
    drift = tmp_path / "config.json"
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["benchmark_512"]["minimum_speedup"] = 2.19
    _write_json(drift, payload)
    with pytest.raises(
        physical.PhysicalBenchmarkInvariantErrorV13,
        match="preregistered amendments",
    ):
        physical.load_physical_config_v13(drift)


def test_canonical_feature_order_comes_from_w1_scene_output_not_sample_order(
    tmp_path: Path,
) -> None:
    query_ids = tuple(f"q-{index:04d}" for index in range(512))
    scene_order = query_ids[100:170] + query_ids[:100] + query_ids[170:]
    folds = (2,) * 170 + (3,) * 171 + (4,) * 171
    path = tmp_path / "w1_outcomes.parquet"
    pq.write_table(
        pa.table(
            {
                "query_id": scene_order,
                "fold": folds,
            }
        ),
        path,
        row_group_size=37,
    )
    observed = _canonical_query_sequence_from_output_v13(path)
    assert observed == scene_order
    assert observed != query_ids
    leaves = {
        query_id: {
            "kind": "site_feature_values",
            "query_id": query_id,
            "fold": fold,
            "fused_features_sha256": _digest(query_id),
        }
        for query_id, fold in zip(scene_order, folds, strict=True)
    }
    assert merkle_root_v13([leaves[value] for value in observed]) != merkle_root_v13(
        [leaves[value] for value in query_ids]
    )


def test_trial_internal_evidence_must_be_relative_and_inventory_bound(
    tmp_path: Path,
) -> None:
    trial_root = tmp_path / "trial"
    internal = trial_root / "build" / "manifest.json"
    digest = _write_json(internal, {"kind": "build"})
    inventory = {
        "build/manifest.json": {
            "role": "support",
            "family": "support",
            "path": "build/manifest.json",
            "sha256": digest,
            "size_bytes": internal.stat().st_size,
        }
    }
    evidence = {
        "build_fragment_dataset": {
            "path": str(internal.resolve()),
            "sha256": digest,
        }
    }
    with pytest.raises(
        physical.PhysicalBenchmarkInvariantErrorV13,
        match="safe trial-relative path",
    ):
        physical._resolve_trial_evidence_record_v13(
            trial_root=trial_root,
            evidence=evidence,
            inventory=inventory,
            name="build_fragment_dataset",
        )

    evidence["build_fragment_dataset"]["path"] = "build/manifest.json"
    resolved, observed = physical._resolve_trial_evidence_record_v13(
        trial_root=trial_root,
        evidence=evidence,
        inventory=inventory,
        name="build_fragment_dataset",
    )
    assert resolved == internal.resolve()
    assert observed == digest

    del inventory["build/manifest.json"]
    with pytest.raises(
        physical.PhysicalBenchmarkInvariantErrorV13,
        match="inventory-bound",
    ):
        physical._resolve_trial_evidence_record_v13(
            trial_root=trial_root,
            evidence=evidence,
            inventory=inventory,
            name="build_fragment_dataset",
        )


def test_top500_full_payload_root_is_differentially_rehydrated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import xgb_matcher.retrieval_fold_fused_fragment_dataset_v13 as fragments
    import xgb_matcher.retrieval_fold_fused_fragment_outputs_v13 as outputs

    context = MasterContext(
        query_id="q-payload",
        scene_fold=2,
        outer_fold=2,
        candidate_sirets=("12345678900011",),
        candidate_parents=("123456789",),
    )
    payload = {"official_evidence_sources": ["SIRENE"], "name": "ACME"}
    names = tuple(f"feature_{index:03d}" for index in range(319))
    batch = FeatureBatch(
        query_id=context.query_id,
        scene_fold=context.scene_fold,
        outer_fold=context.outer_fold,
        candidate_indices=(0,),
        candidate_sirets=context.candidate_sirets,
        candidate_parents=context.candidate_parents,
        feature_names=names,
        values=np.arange(319, dtype=np.float32).reshape(1, 319),
        statistics_sha256=_digest("statistics"),
        row_payloads=(payload,),
    )

    class FeatureContext:
        def compute_full_master_statistics(self, observed: MasterContext):
            return FullMasterStatistics(
                query_id=observed.query_id,
                outer_fold=observed.outer_fold,
                master_count=1,
                candidate_order_sha256=observed.candidate_order_sha256,
                statistics_sha256=batch.statistics_sha256,
                complete=True,
            )

        def materialize(self, *_args, **_kwargs):
            return batch

    compact = pa.Table.from_pylist(
        [
            {
                "query_id": context.query_id,
                "fold": 2,
                "candidate_siret": context.candidate_sirets[0],
                "candidate_parent": context.candidate_parents[0],
                "rank": 1,
                "site_score": 0.75,
                "selection_reason": "ranker",
            }
        ],
        schema=outputs.TOP500_SCHEMA,
    )
    full = pa.Table.from_pylist(
        [
            {
                **compact.to_pylist()[0],
                "payload_json": outputs._canonical_json(payload),
            }
        ],
        schema=outputs.TOP100_SCHEMA,
    )
    feature_root = merkle_root_v13(
        [outputs.feature_batch_logical_leaf_v13(context, batch)]
    )
    payload_root = merkle_root_v13(
        [outputs.prerank500_full_payload_leaf_v13(full)]
    )
    monkeypatch.setattr(
        fragments,
        "iter_heldout_contexts_from_fragments_v13",
        lambda _build: iter((context,)),
    )
    monkeypatch.setattr(
        outputs,
        "iter_compact_output_query_tables_v13",
        lambda _replay, family: iter((compact if family == "top500" else full,)),
    )
    replay = SimpleNamespace(
        feature_values_logical_merkle_root=feature_root,
        prerank500_full_payload_logical_merkle_root=payload_root,
    )
    rows, _digest_value = physical._validate_compact_rehydration_v13(
        object(), replay, feature_context=FeatureContext()
    )
    assert rows == 1

    # Even a self-rehashed compact manifest/root cannot substitute another
    # pre-compaction payload: the independent heldout reconstruction differs.
    tampered_replay = SimpleNamespace(
        feature_values_logical_merkle_root=feature_root,
        prerank500_full_payload_logical_merkle_root=_digest("self-rehashed-tamper"),
    )
    with pytest.raises(
        physical.PhysicalBenchmarkInvariantErrorV13,
        match="payload/proofs differ",
    ):
        physical._validate_compact_rehydration_v13(
            object(), tampered_replay, feature_context=FeatureContext()
        )

def test_injected_test_harness_cannot_publish_pass_or_authorization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest_path, calls = _run(monkeypatch, tmp_path)
    assert calls == [
        "measure_start",
        "runner",
        "measure_stop",
        "rehydrate_post_timer",
        "logical_io",
    ]
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert raw["status"] == "STOP_V13_UNTRUSTED_EXECUTION_HOOKS"
    assert raw["gate_pass"] is False
    assert raw["immutable_v12_stop_preserved"] is True
    assert raw["w3"]["worker_count"] == 3
    assert raw["national_oof_launched"] is False
    assert raw["process_and_machine_io_decision_blocking"] is False
    assert all(raw["checks"].values())

    authorization_path = manifest_path.parent / "execution_authorization.json"
    assert not authorization_path.exists()


def test_injectable_assessment_cannot_reach_official_publisher(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}
    real_publisher = physical._publish_untrusted_assessment_v13

    def capture(assessment):
        captured["assessment"] = assessment
        return real_publisher(assessment)

    monkeypatch.setattr(physical, "_publish_untrusted_assessment_v13", capture)
    manifest, _calls = _run(monkeypatch, tmp_path)
    assessment = captured["assessment"]
    assert assessment.authorization_capability is None
    assert not (manifest.parent / "execution_authorization.json").exists()

    with pytest.raises(
        physical.PhysicalBenchmarkInvariantErrorV13,
        match="refuses an injectable assessment",
    ):
        physical._publish_official_assessment_v13(assessment)
    with pytest.raises(
        physical.PhysicalBenchmarkInvariantErrorV13,
        match="lacks official callback-free DAG provenance",
    ):
        physical._authorization_document_v13(
            assessment,
            benchmark_manifest_sha256=sha256_file(manifest),
        )


def test_process_and_machine_io_counters_are_diagnostic_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest, _ = _run(
        monkeypatch,
        tmp_path,
        resources=_resources(
            process_io=10**15,
            machine_io=10**15,
        ),
    )
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    assert raw["status"] == "STOP_V13_UNTRUSTED_EXECUTION_HOOKS"
    assert raw["process_and_machine_io_decision_blocking"] is False


def test_authorization_reopens_exact_pass_without_second_rehydration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest, _calls = _run(monkeypatch, tmp_path)
    authorization_path = _seal_synthetic_official_authorization_for_loader_test(
        manifest
    )
    benchmark = json.loads(manifest.read_text(encoding="utf-8"))
    identity = benchmark["benchmark_identity"]
    w3 = benchmark["w3"]
    observed_verify: list[bool] = []

    def load_product(_path: Path, **kwargs):
        observed_verify.append(bool(kwargs["verify_rehydration"]))
        return SimpleNamespace(
            lineage=w3["lineage"],
            bridge_exact=True,
            physical_merge_files_present=False,
            full_payload_outside_top100=False,
            materialized_bytes=w3["materialized_bytes"],
            projected_national_storage_bytes=w3[
                "projected_national_storage_bytes"
            ],
            artifact_inventory_sha256=identity["artifact_inventory_sha256"],
            fragment_dataset_manifest_sha256=identity[
                "fragment_dataset_manifest_sha256"
            ],
            replay_dataset_manifest_sha256=identity[
                "replay_dataset_manifest_sha256"
            ],
            sharded_execution_plan_sha256=identity[
                "sharded_execution_plan_sha256"
            ],
            prefit_parity_manifest_sha256=identity[
                "prefit_parity_manifest_sha256"
            ],
            logical_digests=identity["logical_digests"],
        )

    monkeypatch.setattr(
        physical, "load_fragment_w3_trial_product_v13", load_product
    )
    import xgb_matcher.retrieval_fold_fused_logical_io_v13 as logical_io

    monkeypatch.setattr(
        logical_io,
        "load_logical_io_gate_v13",
        lambda *_args, **_kwargs: SimpleNamespace(
            gate_pass=True, status="PASS_V13_IO"
        ),
    )
    logical_record = benchmark["logical_io"]
    logical_evidence = physical.LogicalIOEvidenceV13(
        manifest_path=Path(logical_record["manifest_path"]),
        manifest_sha256=logical_record["manifest_sha256"],
        status=logical_record["status"],
        gate_pass=logical_record["gate_pass"],
        projection_sha256=logical_record["projection_sha256"],
        unique_row_groups=logical_record["unique_row_groups"],
        logical_compressed_bytes=logical_record["logical_compressed_bytes"],
        cross_actor_overlap_count=logical_record["cross_actor_overlap_count"],
        w3_union_equals_w1=logical_record["w3_union_equals_w1"],
        logical_amplification_ratio=logical_record[
            "logical_amplification_ratio"
        ],
        materialized_ratio=logical_record["materialized_ratio"],
    )
    monkeypatch.setattr(
        physical,
        "logical_io_evidence_from_gate_v13",
        lambda _gate: logical_evidence,
    )
    authorization = physical.load_execution_authorization_v13(
        authorization_path,
        expected_authorization_sha256=sha256_file(authorization_path),
        benchmark_manifest_path=manifest,
        expected_benchmark_manifest_sha256=sha256_file(manifest),
        workspace_root=ROOT,
    )
    assert authorization["status"] == "AUTHORIZED_V13_W3_FOR_NATIONAL_OOF"
    assert observed_verify == [False]


@pytest.mark.parametrize(
    "tamper",
    ["wall", "ram", "swap", "storage", "logical_ratio", "w1_wall"],
)
def test_authorization_recomputes_rehashed_physical_facts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tamper: str,
) -> None:
    manifest, _calls = _run(monkeypatch, tmp_path)
    authorization_path = _seal_synthetic_official_authorization_for_loader_test(
        manifest
    )
    benchmark = json.loads(manifest.read_text(encoding="utf-8"))
    if tamper == "wall":
        benchmark["w3"]["resources"]["wall_seconds"] = 900.0
        benchmark["w3"]["speedup_over_immutable_w1"] = (
            benchmark["immutable_w1"]["wall_seconds"] / 900.0
        )
    elif tamper == "ram":
        benchmark["w3"]["resources"][
            "peak_machine_used_memory_bytes"
        ] = 15 * 1024**3
    elif tamper == "swap":
        benchmark["w3"]["resources"]["swap_out_delta_bytes"] = 1
    elif tamper == "storage":
        benchmark["w3"]["projected_national_storage_bytes"] = 21 * 1024**3
    elif tamper == "logical_ratio":
        benchmark["logical_io"]["logical_amplification_ratio"] = 1.01
    elif tamper == "w1_wall":
        benchmark["immutable_w1"]["wall_seconds"] += 1.0
    else:  # pragma: no cover - parametrization is frozen above
        raise AssertionError(tamper)
    _write_json(manifest, benchmark)
    benchmark_sha, authorization_sha = _rebind_authorization_to_rehashed_benchmark(
        manifest, authorization_path
    )
    observed_verify = _patch_authorization_artifact_reopen_for_test(
        monkeypatch, benchmark
    )
    with pytest.raises(
        physical.PhysicalBenchmarkInvariantErrorV13,
        match="artifact-derived|gates were not reproduced|immutable W1 anchor",
    ):
        physical.load_execution_authorization_v13(
            authorization_path,
            expected_authorization_sha256=authorization_sha,
            benchmark_manifest_path=manifest,
            expected_benchmark_manifest_sha256=benchmark_sha,
            workspace_root=ROOT,
        )
    # W1 report substitution fails before W3 reopen; all other substitutions
    # reopen once without repeating the expensive independent rehydration.
    assert observed_verify in ([], [False])


@pytest.mark.parametrize(
    ("product_options", "logical_options", "resources", "expected"),
    [
        ({"exact": False}, {}, None, "STOP_V13_PARITY"),
        ({"merge": True}, {}, None, "STOP_V13_ARTIFACT_PROTOCOL"),
        ({"payload_leak": True}, {}, None, "STOP_V13_PAYLOAD_PROTOCOL"),
        ({"rehydration": False}, {}, None, "STOP_V13_PARITY"),
        ({}, {"overlap": 1}, None, "STOP_V13_IO"),
        ({}, {}, _resources(ram=14 * 1024**3 + 1), "STOP_V13_RAM"),
        ({}, {}, _resources(swap=1), "STOP_V13_SWAP"),
        ({}, {}, _resources(wall=872.2672163445598 + 0.01), "STOP_V13_SPEED"),
        (
            {"projected_storage": 20 * 1024**3 + 1},
            {},
            None,
            "STOP_V13_STORAGE",
        ),
    ],
)
def test_each_failed_gate_publishes_stop_without_authorization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    product_options: dict[str, Any],
    logical_options: dict[str, Any],
    resources: PhysicalResourceMetricsV12 | None,
    expected: str,
) -> None:
    manifest, _ = _run(
        monkeypatch,
        tmp_path,
        product_options=product_options,
        logical_options=logical_options,
        resources=resources,
    )
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    assert raw["status"] == expected
    assert raw["gate_pass"] is False
    assert raw["execution_authorization_published"] is False
    assert not (manifest.parent / "execution_authorization.json").exists()


def test_materialized_bytes_are_cross_checked_outside_logical_io_builder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest, _ = _run(
        monkeypatch,
        tmp_path,
        product_options={"materialized": 534035787},
        logical_options={"materialized_ratio": 1.0},
    )
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    assert raw["status"] == "STOP_V13_IO"
    assert raw["checks"]["logical_io_pass"] is False


def test_authorization_tampering_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest, _ = _run(monkeypatch, tmp_path)
    path = _seal_synthetic_official_authorization_for_loader_test(manifest)
    original_sha = sha256_file(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["authorization_identity"]["authorized_worker_count"] = 2
    _write_json(path, raw)
    with pytest.raises(
        physical.PhysicalBenchmarkInvariantErrorV13,
        match="authorization or benchmark bytes drifted",
    ):
        physical.load_execution_authorization_v13(
            path,
            expected_authorization_sha256=original_sha,
            benchmark_manifest_path=manifest,
            expected_benchmark_manifest_sha256=sha256_file(manifest),
            workspace_root=ROOT,
        )


def test_existing_output_refuses_second_trial_before_runner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    monkeypatch.setattr(physical, "load_physical_config_v13", lambda _path: config)
    monkeypatch.setattr(
        physical,
        "load_immutable_w1_anchor_v13",
        lambda _config: _anchor(tmp_path),
    )
    output = tmp_path / "exists"
    output.mkdir()
    called = False

    def runner(_output, _anchor):
        nonlocal called
        called = True
        raise AssertionError("must not run")

    with pytest.raises(
        physical.PhysicalBenchmarkInvariantErrorV13,
        match="noncanonical or already exists",
    ):
        physical.run_physical_benchmark_test_harness_v13(
            config_path=CONFIG_PATH,
            output_dir=output,
            runner=runner,
            logical_io_builder=lambda *_args: None,
            measurer=lambda callback: (callback(), _resources()),
        )
    assert called is False


def test_callback_exception_consumes_attempt_and_publishes_durable_stop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    output = tmp_path / "benchmark"
    marker = tmp_path / ".retrieval_v13_w3_attempt_consumed.json"
    config["benchmark_512"]["canonical_output_dir"] = str(output)
    config["benchmark_512"]["attempt_consumed_marker_path"] = str(marker)
    monkeypatch.setattr(physical, "load_physical_config_v13", lambda _path: config)
    monkeypatch.setattr(
        physical, "load_immutable_w1_anchor_v13", lambda _config: _anchor(tmp_path)
    )
    monkeypatch.setattr(
        physical,
        "v13_implementation_lineage",
        lambda _root: (_digest("implementation"), ("module.py",)),
    )
    calls = 0

    def runner(_output: Path, _anchor: physical.ImmutableW1AnchorV13):
        nonlocal calls
        calls += 1
        raise RuntimeError("deliberate callback failure")

    def measurer(callback):
        return callback(), _resources()

    with pytest.raises(RuntimeError, match="deliberate callback failure"):
        physical.run_physical_benchmark_test_harness_v13(
            config_path=CONFIG_PATH,
            output_dir=output,
            runner=runner,
            logical_io_builder=lambda *_args: None,
            measurer=measurer,
        )
    assert calls == 1
    assert marker.is_file()
    stop = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert stop["status"] == "STOP_V13_EXECUTION_EXCEPTION"
    assert stop["attempt_marker"]["sha256"] == sha256_file(marker)
    assert stop["execution_authorization_published"] is False
    assert not (output / "execution_authorization.json").exists()

    with pytest.raises(
        physical.PhysicalBenchmarkInvariantErrorV13,
        match="noncanonical or already exists",
    ):
        physical.run_physical_benchmark_test_harness_v13(
            config_path=CONFIG_PATH,
            output_dir=output,
            runner=runner,
            logical_io_builder=lambda *_args: None,
            measurer=measurer,
        )
    assert calls == 1


def test_harness_post_o_excl_marker_fault_publishes_durable_stop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    output = tmp_path / "benchmark"
    marker = tmp_path / ".retrieval_v13_w3_attempt_consumed.json"
    config["benchmark_512"]["canonical_output_dir"] = str(output)
    config["benchmark_512"]["attempt_consumed_marker_path"] = str(marker)
    monkeypatch.setattr(physical, "load_physical_config_v13", lambda _path: config)
    monkeypatch.setattr(
        physical,
        "v13_implementation_lineage",
        lambda _root: (_digest("implementation"), ("module.py",)),
    )
    real_write = physical.os.write
    write_calls = 0

    def partial_marker_write(descriptor: int, payload: bytes) -> int:
        nonlocal write_calls
        write_calls += 1
        real_write(descriptor, payload[:17])
        raise OSError("partial marker write failed")

    monkeypatch.setattr(physical.os, "write", partial_marker_write)
    runner_called = False

    def runner(*_args):
        nonlocal runner_called
        runner_called = True
        raise AssertionError("runner must not start after marker failure")

    call = dict(
        config_path=CONFIG_PATH,
        output_dir=output,
        runner=runner,
        logical_io_builder=lambda *_args: None,
        measurer=lambda callback: (callback(), _resources()),
    )
    with pytest.raises(
        physical._OneShotAttemptConsumptionErrorV13,
        match="after O_EXCL",
    ):
        physical.run_physical_benchmark_test_harness_v13(**call)
    assert marker.is_file()
    assert write_calls == 1
    assert marker.stat().st_size == 17
    assert runner_called is False
    stop = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert stop["status"] == "STOP_V13_EXECUTION_EXCEPTION"
    assert stop["phase"] == "ATOMIC_ATTEMPT_MARKER_CREATION"
    assert stop["attempt_marker"]["sha256"] == sha256_file(marker)
    assert stop["execution_authorization_published"] is False
    with pytest.raises(
        physical.PhysicalBenchmarkInvariantErrorV13,
        match="noncanonical or already exists",
    ):
        physical.run_physical_benchmark_test_harness_v13(**call)


def test_build_orchestrator_uses_three_fold_owned_workers_and_no_merge(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    lineage = {"scene": _digest("scene")}

    class Spec:
        def __init__(self, fold: int, ordinal: int, count: int) -> None:
            self.folds = (fold,)
            self.shard_id = f"fold{fold}"
            self.shard_ordinal = ordinal
            self.query_count = count
            self.digest = _digest(f"spec-{fold}")

        def as_manifest(self):
            return {"fold": self.folds[0], "ordinal": self.shard_ordinal}

    specs = (Spec(2, 0, 170), Spec(3, 1, 171), Spec(4, 2, 171))
    builds = tuple(SimpleNamespace(spec=spec) for spec in specs)
    prepared = SimpleNamespace(
        lineage=lineage,
        scene_cache=SimpleNamespace(index=object()),
        sample_query_ids=tuple(str(index) for index in range(512)),
        runtime_payload={},
        caps=ExecutionCaps(),
        executor=SimpleNamespace(maximum_temporary_gib=20),
        sharded_execution_plan_sha256=_digest("plan"),
    )
    observed: dict[str, Any] = {}
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    canonical = tmp_path / "benchmark"
    marker = tmp_path / ".retrieval_v13_w3_attempt_consumed.json"
    config["benchmark_512"]["canonical_output_dir"] = str(canonical)
    config["benchmark_512"]["attempt_consumed_marker_path"] = str(marker)
    _write_json(
        marker,
        {
            "schema_version": "sireto-retrieval-v13-one-shot-attempt-1",
            "status": "ATTEMPT_IRREVERSIBLY_CONSUMED_BEFORE_CALLBACK",
            "config_sha256": physical.CONFIG_SHA256,
            "canonical_output_dir": str(canonical),
            "marker_path": str(marker),
            "retry_allowed": False,
        },
    )
    monkeypatch.setattr(physical, "load_physical_config_v13", lambda _path: config)
    monkeypatch.setattr(
        physical,
        "sharded_execution_lineage_v12",
        lambda *_args, **_kwargs: lineage,
    )
    monkeypatch.setattr(
        physical,
        "plan_query_shards",
        lambda *_args, **_kwargs: specs,
    )

    def spawn(entrypoint, jobs, *, max_workers):
        observed["entrypoint"] = entrypoint
        observed["jobs"] = jobs
        observed["max_workers"] = max_workers
        return tuple(tmp_path / f"build-{index}.json" for index in range(3))

    monkeypatch.setattr(physical, "spawn_manifest_workers_v12", spawn)
    monkeypatch.setattr(
        physical,
        "load_shard_build_manifest_v12",
        lambda path: builds[int(path.stem.rsplit("-", 1)[1])],
    )
    sentinel = object()

    def finish(**kwargs):
        observed["finish"] = kwargs
        return sentinel

    result = physical.execute_fragment_w3_trial_v13(
        prepared=prepared,
        workspace_root=ROOT,
        output_dir=canonical / "trial_w3",
        anchor=_anchor(tmp_path),
        trial_finisher=finish,
        attempt_marker_path=marker,
        attempt_marker_sha256=sha256_file(marker),
    )
    assert result is sentinel
    assert observed["max_workers"] == 3
    assert len(observed["jobs"]) == 3
    assert [job.payload["phase"] for job in observed["jobs"]] == [
        "BUILD",
        "BUILD",
        "BUILD",
    ]
    assert observed["finish"]["build_artifacts"] == builds
    assert not (canonical / "trial_w3" / "train_matrices").exists()
    assert not (canonical / "trial_w3" / "heldout_master_spool").exists()


def test_direct_w3_execution_without_one_shot_capability_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    canonical = tmp_path / "benchmark"
    config["benchmark_512"]["canonical_output_dir"] = str(canonical)
    config["benchmark_512"]["attempt_consumed_marker_path"] = str(
        tmp_path / ".retrieval_v13_w3_attempt_consumed.json"
    )
    monkeypatch.setattr(physical, "load_physical_config_v13", lambda _path: config)
    with pytest.raises(
        physical.PhysicalBenchmarkInvariantErrorV13,
        match="sealed one-shot capability",
    ):
        physical.execute_fragment_w3_trial_v13(
            prepared=SimpleNamespace(),
            workspace_root=ROOT,
            output_dir=canonical / "trial_w3",
            anchor=_anchor(tmp_path),
            trial_finisher=lambda **_kwargs: None,
        )
    assert not canonical.exists()


def test_official_runner_owns_cache_neutral_preflight_before_marker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    parameters = inspect.signature(physical.run_physical_benchmark_v13).parameters
    assert not {
        "prepared",
        "w1_functional_projection",
        "w1_io_projection",
        "runner",
        "logical_io_builder",
        "measurer",
    }.intersection(parameters)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    output = tmp_path / "benchmark"
    marker = tmp_path / ".retrieval_v13_w3_attempt_consumed.json"
    config["benchmark_512"]["canonical_output_dir"] = str(output)
    config["benchmark_512"]["attempt_consumed_marker_path"] = str(marker)
    monkeypatch.setattr(physical, "load_physical_config_v13", lambda _path: config)
    observed: dict[str, Any] = {}

    def prepare(**kwargs):
        observed.update(kwargs)
        raise RuntimeError("stop after cache-neutral preflight policy capture")

    monkeypatch.setattr(physical, "prepare_sharded_execution_v12", prepare)
    with pytest.raises(RuntimeError, match="policy capture"):
        physical.run_physical_benchmark_v13(
            config_path=CONFIG_PATH,
            output_dir=output,
            workspace_root=ROOT,
            scene_ir=tmp_path / "scene",
            query_sample=tmp_path / "sample",
            labels=tmp_path / "labels",
            sidecars=tmp_path / "sidecars",
            mmap_sidecar=tmp_path / "mmap",
            reader_parity_manifest=tmp_path / "reader",
            parent_model_bundle=tmp_path / "parent",
            scientific_config=tmp_path / "science.json",
            executor_config=tmp_path / "executor.json",
            prefit_parity_manifest=tmp_path / "prefit.json",
            sharded_execution_plan=tmp_path / "plan.json",
            immutable_projection_root=tmp_path / "projections",
        )
    assert observed["verify_source_hash"] is False
    assert "prepared" not in observed
    assert not marker.exists()
    assert not output.exists()


def test_official_w1_projection_strict_reload_fails_before_marker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import xgb_matcher.retrieval_fold_fused_fragment_runtime_v13 as fragment_runtime
    import xgb_matcher.retrieval_fold_fused_logical_io_v13 as logical_io

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    output = tmp_path / "benchmark"
    marker = tmp_path / ".retrieval_v13_w3_attempt_consumed.json"
    config["benchmark_512"]["canonical_output_dir"] = str(output)
    config["benchmark_512"]["attempt_consumed_marker_path"] = str(marker)
    monkeypatch.setattr(physical, "load_physical_config_v13", lambda _path: config)
    monkeypatch.setattr(
        physical,
        "prepare_sharded_execution_v12",
        lambda **_kwargs: SimpleNamespace(
            executor=object(),
            prefit_parity_manifest_path=tmp_path / "prefit.json",
            prefit_parity_manifest_sha256=_digest("prefit"),
            lineage={"scene": _digest("scene")},
            sample_query_ids=tuple(str(index) for index in range(512)),
        ),
    )
    monkeypatch.setattr(
        physical,
        "require_external_sharded_output_v12",
        lambda _path, **_kwargs: output,
    )
    anchor = _anchor(tmp_path)
    monkeypatch.setattr(physical, "load_immutable_w1_anchor_v13", lambda _config: anchor)
    functional = SimpleNamespace(
        manifest_path=tmp_path / "functional" / "manifest.json",
        manifest_sha256=_digest("functional"),
    )
    monkeypatch.setattr(
        fragment_runtime,
        "build_immutable_w1_functional_projection_v13",
        lambda **_kwargs: functional,
    )
    monkeypatch.setattr(
        fragment_runtime,
        "load_immutable_w1_functional_projection_v13",
        lambda *_args, **_kwargs: functional,
    )
    io_projection = SimpleNamespace(
        manifest_path=tmp_path / "logical" / "manifest.json",
        manifest_sha256=_digest("logical"),
    )
    monkeypatch.setattr(
        logical_io,
        "build_immutable_w1_io_projection_v13",
        lambda **_kwargs: io_projection,
    )

    def reject_reload(*_args, **_kwargs):
        raise RuntimeError("strict W1 I/O projection reload failed")

    monkeypatch.setattr(
        logical_io, "load_immutable_w1_io_projection_v13", reject_reload
    )
    with pytest.raises(RuntimeError, match="strict W1 I/O"):
        physical.run_physical_benchmark_v13(
            config_path=CONFIG_PATH,
            output_dir=output,
            workspace_root=ROOT,
            scene_ir=tmp_path / "scene",
            query_sample=tmp_path / "sample",
            labels=tmp_path / "labels",
            sidecars=tmp_path / "sidecars",
            mmap_sidecar=tmp_path / "mmap",
            reader_parity_manifest=tmp_path / "reader",
            parent_model_bundle=tmp_path / "parent",
            scientific_config=tmp_path / "science.json",
            executor_config=tmp_path / "executor.json",
            prefit_parity_manifest=tmp_path / "prefit.json",
            sharded_execution_plan=tmp_path / "plan.json",
            immutable_projection_root=tmp_path / "projections",
        )
    assert not marker.exists()
    assert not output.exists()


def test_official_post_o_excl_marker_fault_publishes_durable_stop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import xgb_matcher.retrieval_fold_fused_fragment_runtime_v13 as fragment_runtime
    import xgb_matcher.retrieval_fold_fused_logical_io_v13 as logical_io

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    output = tmp_path / "benchmark"
    marker = tmp_path / ".retrieval_v13_w3_attempt_consumed.json"
    config["benchmark_512"]["canonical_output_dir"] = str(output)
    config["benchmark_512"]["attempt_consumed_marker_path"] = str(marker)
    monkeypatch.setattr(physical, "load_physical_config_v13", lambda _path: config)
    prepared = SimpleNamespace(
        executor=object(),
        prefit_parity_manifest_path=tmp_path / "prefit.json",
        prefit_parity_manifest_sha256=_digest("prefit"),
        lineage={"scene": _digest("scene")},
        sample_query_ids=tuple(str(index) for index in range(512)),
    )
    monkeypatch.setattr(
        physical, "prepare_sharded_execution_v12", lambda **_kwargs: prepared
    )
    monkeypatch.setattr(
        physical,
        "require_external_sharded_output_v12",
        lambda _path, **_kwargs: output,
    )
    anchor = _anchor(tmp_path)
    monkeypatch.setattr(physical, "load_immutable_w1_anchor_v13", lambda _config: anchor)
    functional = SimpleNamespace(
        manifest_path=tmp_path / "functional" / "manifest.json",
        manifest_sha256=_digest("functional"),
    )
    monkeypatch.setattr(
        fragment_runtime,
        "build_immutable_w1_functional_projection_v13",
        lambda **_kwargs: functional,
    )
    monkeypatch.setattr(
        fragment_runtime,
        "load_immutable_w1_functional_projection_v13",
        lambda *_args, **_kwargs: functional,
    )
    io_projection = SimpleNamespace(
        manifest_path=tmp_path / "logical" / "manifest.json",
        manifest_sha256=_digest("logical"),
    )
    monkeypatch.setattr(
        logical_io,
        "build_immutable_w1_io_projection_v13",
        lambda **_kwargs: io_projection,
    )
    monkeypatch.setattr(
        logical_io,
        "load_immutable_w1_io_projection_v13",
        lambda *_args, **_kwargs: (io_projection, {}, {}, {}),
    )
    monkeypatch.setattr(
        physical,
        "v13_implementation_lineage",
        lambda _root: (_digest("implementation"), ("module.py",)),
    )
    real_fsync = physical.os.fsync
    calls = 0

    def fail_first_fsync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("official marker fsync failed")
        real_fsync(descriptor)

    monkeypatch.setattr(physical.os, "fsync", fail_first_fsync)
    call = dict(
        config_path=CONFIG_PATH,
        output_dir=output,
        workspace_root=ROOT,
        scene_ir=tmp_path / "scene",
        query_sample=tmp_path / "sample",
        labels=tmp_path / "labels",
        sidecars=tmp_path / "sidecars",
        mmap_sidecar=tmp_path / "mmap",
        reader_parity_manifest=tmp_path / "reader",
        parent_model_bundle=tmp_path / "parent",
        scientific_config=tmp_path / "science.json",
        executor_config=tmp_path / "executor.json",
        prefit_parity_manifest=tmp_path / "prefit.json",
        sharded_execution_plan=tmp_path / "plan.json",
        immutable_projection_root=tmp_path / "projections",
    )
    with pytest.raises(
        physical._OneShotAttemptConsumptionErrorV13,
        match="after O_EXCL",
    ):
        physical.run_physical_benchmark_v13(**call)
    assert marker.is_file()
    stop = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert stop["status"] == "STOP_V13_EXECUTION_EXCEPTION"
    assert stop["phase"] == "ATOMIC_ATTEMPT_MARKER_CREATION"
    assert stop["attempt_marker"]["sha256"] == sha256_file(marker)
    assert stop["execution_authorization_published"] is False
    with pytest.raises(
        physical.PhysicalBenchmarkInvariantErrorV13,
        match="noncanonical or already exists",
    ):
        physical.run_physical_benchmark_v13(**call)
