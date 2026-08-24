from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import pytest

from xgb_matcher.retrieval_fold_fused_oof_v12 import sha256_file
from xgb_matcher.retrieval_fold_fused_physical_benchmark_v12 import (
    ALLOWED_FOLDS,
    PhysicalBenchmarkInvariantError,
    PhysicalResourceMetricsV12,
    PhysicalTrialProductV12,
    MATERIALIZED_FAMILIES,
    REQUIRED_DIGESTS,
    load_sharded_oof_authorization_v12,
    physical_trial_product_from_sharded_v12,
    run_physical_worker_benchmark_v12,
)


ROOT = Path(__file__).resolve().parents[1]
SHARDED_PLAN_PATH = (
    ROOT / "config/retrieval_all_human_v12_sharded_execution_plan.json"
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _lineage() -> dict[str, str]:
    return {
        "scene_ir_sha256": _digest("scene-build"),
        "scene_ir_manifest_sha256": _digest("scene"),
        "source_candidate_sha256": _digest("source-candidate"),
        "source_manifest_sha256": _digest("source-manifest"),
        "label_ledger_sha256": _digest("ledger-data"),
        "label_ledger_manifest_sha256": _digest("ledger-manifest"),
        "scientific_config_sha256": _digest("scientific"),
        "executor_config_sha256": _digest("executor"),
        "official_sidecar_manifest_sha256": _digest("official"),
        "mmap_sidecar_manifest_sha256": _digest("mmap"),
        "reader_parity_evidence_sha256": _digest("reader"),
        "parent_model_bundle_sha256": _digest("parent"),
        "query_sample_sha256": _digest("sample"),
    }


def _prefit(path: Path, lineage: dict[str, str]) -> str:
    path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "phase": "PREFIT_NON_CIRCULAR",
                "query_count": 512,
                "fold0_opened": False,
                "fold1_opened": False,
                "positive_injection": False,
                "lineage": lineage,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return sha256_file(path)


def _sharded_plan(
    path: Path, lineage: dict[str, str], prefit_sha: str
) -> str:
    payload = json.loads(SHARDED_PLAN_PATH.read_text(encoding="utf-8"))
    payload["authorities"]["scientific_config"]["sha256"] = lineage[
        "scientific_config_sha256"
    ]
    payload["authorities"]["fold_fused_executor_config"]["sha256"] = lineage[
        "executor_config_sha256"
    ]
    scene = payload["input_lineage"]["scene_ir"]
    scene.update(
        {
            "build_id": lineage["scene_ir_sha256"],
            "manifest_sha256": lineage["scene_ir_manifest_sha256"],
            "source_candidate_sha256": lineage["source_candidate_sha256"],
            "source_manifest_sha256": lineage["source_manifest_sha256"],
        }
    )
    labels = payload["input_lineage"]["labels"]
    labels.update(
        {
            "ledger_sha256": lineage["label_ledger_sha256"],
            "manifest_sha256": lineage["label_ledger_manifest_sha256"],
            "runtime_query_sample_sha256": lineage["query_sample_sha256"],
        }
    )
    official = payload["input_lineage"]["official_sidecars"]
    official.update(
        {
            "sidecar_manifest_sha256": lineage[
                "official_sidecar_manifest_sha256"
            ],
            "mmap_manifest_sha256": lineage["mmap_sidecar_manifest_sha256"],
            "reader_parity_manifest_sha256": lineage[
                "reader_parity_evidence_sha256"
            ],
        }
    )
    payload["input_lineage"]["parent_model_bundle"]["manifest_sha256"] = lineage[
        "parent_model_bundle_sha256"
    ]
    payload["input_lineage"]["prefit_parity"]["manifest_sha256"] = prefit_sha
    payload["sharded_parity_gate"]["sample_sha256"] = lineage[
        "query_sample_sha256"
    ]
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return sha256_file(path)


class FixtureRunner:
    def __init__(
        self,
        lineage: dict[str, str],
        *,
        digest_drift_worker: int | None = None,
        folds: tuple[int, ...] = ALLOWED_FOLDS,
        materialized_io_bytes: int = 100_000,
        site_model_bytes: int = 1_000,
    ) -> None:
        self.lineage = lineage
        self.digest_drift_worker = digest_drift_worker
        self.folds = folds
        self.materialized_io_bytes = materialized_io_bytes
        self.site_model_bytes = site_model_bytes
        self.calls: list[int] = []

    def __call__(self, worker_count: int, output_dir: Path) -> PhysicalTrialProductV12:
        self.calls.append(worker_count)
        output_dir.mkdir(parents=True)
        manifest = output_dir / "manifest.json"
        manifest.write_text(
            json.dumps({"worker_count": worker_count}) + "\n", encoding="utf-8"
        )
        digests = {
            name: _digest(
                f"drift-{name}"
                if worker_count == self.digest_drift_worker
                else f"exact-{name}"
            )
            for name in REQUIRED_DIGESTS
        }
        scalable = max(0, self.materialized_io_bytes - self.site_model_bytes)
        per_family = scalable // (len(MATERIALIZED_FAMILIES) - 1)
        families = {
            name: (
                self.site_model_bytes if name == "site_models" else per_family
            )
            for name in MATERIALIZED_FAMILIES
        }
        return PhysicalTrialProductV12(
            worker_count=worker_count,
            manifest_path=manifest,
            query_count=512,
            folds=self.folds,
            lineage=self.lineage,
            digests=digests,
            materialized_io_bytes=self.materialized_io_bytes,
            site_model_bytes=self.site_model_bytes,
            materialized_families=families,
        )


def _resources(
    *,
    wall: float,
    io_bytes: int,
    rss_bytes: int = 8 * 1024**3,
    machine_used_bytes: int = 12 * 1024**3,
    swap: int = 0,
) -> PhysicalResourceMetricsV12:
    return PhysicalResourceMetricsV12(
        wall_seconds=wall,
        coordinator_cpu_seconds=1.0,
        children_cpu_seconds=2.0,
        machine_busy_cpu_seconds=3.0,
        peak_process_tree_rss_bytes=rss_bytes,
        peak_machine_used_memory_bytes=machine_used_bytes,
        minimum_machine_available_memory_bytes=6 * 1024**3,
        swap_out_delta_bytes=swap,
        process_read_bytes=io_bytes // 2,
        process_write_bytes=io_bytes - io_bytes // 2,
        machine_read_bytes=io_bytes,
        machine_write_bytes=io_bytes,
        sample_interval_seconds=0.05,
    )


def _measurer(
    resources_by_worker: dict[int, PhysicalResourceMetricsV12],
) -> Callable:
    def measure(callback: Callable):
        product = callback()
        return product, resources_by_worker[product.worker_count]

    return measure


def _run(
    tmp_path: Path,
    *,
    runner: FixtureRunner,
    resources: dict[int, PhysicalResourceMetricsV12],
) -> tuple[Path, str, str]:
    prefit_path = tmp_path / "prefit.json"
    prefit_sha = _prefit(prefit_path, runner.lineage)
    plan_path = tmp_path / "sharded_plan.json"
    plan_sha = _sharded_plan(plan_path, runner.lineage, prefit_sha)
    manifest = run_physical_worker_benchmark_v12(
        runner=runner,
        output_dir=tmp_path / "benchmark",
        expected_lineage=runner.lineage,
        prefit_parity_manifest_path=prefit_path,
        prefit_parity_manifest_sha256=prefit_sha,
        sharded_execution_plan_path=plan_path,
        sharded_execution_plan_sha256=plan_sha,
        measurer=_measurer(resources),
    )
    return manifest, prefit_sha, plan_sha


def _replace_nested(payload: dict, path: tuple[object, ...], value: object) -> None:
    current: object = payload
    for key in path[:-1]:
        current = current[key]  # type: ignore[index]
    current[path[-1]] = value  # type: ignore[index]


@pytest.mark.parametrize(
    ("field_path", "drifted_value"),
    [
        (("engine_id",), "WRONG_ENGINE"),
        (("physical_plan_id",), "WRONG_PHYSICAL_PLAN"),
        (("authorities", "scientific_contract", "path"), "docs/wrong.md"),
        (("authorities", "sharded_execution_addendum", "sha256"), "invalid"),
        (("fold_policy", "fold0_diagnostic_only"), False),
        (("shards", 1, "source_row_group_stop_exclusive"), 424),
        (("process_policy", "site_fit_order"), [4, 3, 2]),
        (("query_compilation", "label_blind_worker_cache_only"), False),
        (("train_merge", "outer3"), [4, 2]),
        (("heldout_replay", "fold_to_site_model_outer", "3"), 2),
        (("output_merge", "per_shard_metric_average_forbidden"), False),
        (("sharded_parity_gate", "required_checks"), ["incomplete"]),
        (
            ("sharded_parity_gate", "required_negative_tests"),
            ["incomplete"],
        ),
        (("scientific_gates_unchanged", "exact_recall_at_100"), 0.991),
        (("resources", "maximum_temporary_bytes"), 21 * 1024**3),
        (("final_decision_policy", "postfit_refit_allowed"), True),
    ],
    ids=[
        "engine",
        "physical-plan",
        "authority-path",
        "authority-sha",
        "fold-policy",
        "shard-range",
        "process-policy",
        "query-policy",
        "train-merge",
        "heldout-replay",
        "output-merge",
        "parity-checks",
        "negative-tests",
        "scientific-gate",
        "resources",
        "final-decision",
    ],
)
def test_sharded_plan_semantic_drift_is_rejected_before_any_trial(
    tmp_path: Path,
    field_path: tuple[object, ...],
    drifted_value: object,
) -> None:
    lineage = _lineage()
    runner = FixtureRunner(lineage)
    prefit_path = tmp_path / "prefit.json"
    prefit_sha = _prefit(prefit_path, lineage)
    plan_path = tmp_path / "sharded_plan.json"
    _sharded_plan(plan_path, lineage, prefit_sha)
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    _replace_nested(payload, field_path, drifted_value)
    plan_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(PhysicalBenchmarkInvariantError):
        run_physical_worker_benchmark_v12(
            runner=runner,
            output_dir=tmp_path / "benchmark",
            expected_lineage=lineage,
            prefit_parity_manifest_path=prefit_path,
            prefit_parity_manifest_sha256=prefit_sha,
            sharded_execution_plan_path=plan_path,
            sharded_execution_plan_sha256=sha256_file(plan_path),
            measurer=_measurer(
                {
                    1: _resources(wall=30.0, io_bytes=1_000),
                    3: _resources(wall=10.0, io_bytes=1_100),
                }
            ),
        )
    assert runner.calls == []


def test_selects_w3_only_after_exact_full_product_parity(tmp_path: Path) -> None:
    runner = FixtureRunner(_lineage())
    manifest_path, prefit_sha, plan_sha = _run(
        tmp_path,
        runner=runner,
        resources={
            1: _resources(wall=30.0, io_bytes=1_000),
            3: _resources(wall=10.0, io_bytes=1_100),
        },
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert runner.calls == [3, 1]
    assert manifest["selected_worker_count"] == 3
    assert manifest["candidate_decisions"]["3"] == {
        "worker_count": 3,
        "parity_exact": True,
        "speedup": 3.0,
        "io_ratio": 1.1,
        "io_measurement_source": "PROCESS_TREE_DISK_BYTES",
        "ram_pass": True,
        "swap_pass": True,
        "speed_pass": True,
        "io_pass": True,
        "eligible": True,
    }
    # The policy runner is intentionally generic, but an execution
    # authorization may only be loaded from the complete sharded core.  A
    # self-declared unit fixture must therefore be rejected even if the outer
    # benchmark copied its SHA correctly.
    with pytest.raises(PhysicalBenchmarkInvariantError):
        load_sharded_oof_authorization_v12(
            manifest_path,
            manifest_path.parent / "execution_authorization.json",
            expected_lineage=runner.lineage,
            expected_prefit_parity_manifest_sha256=prefit_sha,
            expected_sharded_execution_plan_sha256=plan_sha,
        )


def test_falls_back_to_w2_only_when_w3_is_exact_but_ineligible(
    tmp_path: Path,
) -> None:
    runner = FixtureRunner(_lineage())
    manifest_path, _, _ = _run(
        tmp_path,
        runner=runner,
        resources={
            1: _resources(wall=30.0, io_bytes=1_000),
            2: _resources(wall=15.0, io_bytes=1_100),
            3: _resources(wall=16.0, io_bytes=1_100),
        },
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert runner.calls == [3, 1, 2]
    assert manifest["execution_order"] == [3, 1, 2]
    assert manifest["selected_worker_count"] == 2


def test_stops_when_no_parallel_worker_count_passes_resource_gates(
    tmp_path: Path,
) -> None:
    runner = FixtureRunner(_lineage())
    manifest_path, _, _ = _run(
        tmp_path,
        runner=runner,
        resources={
            1: _resources(wall=30.0, io_bytes=1_000),
            2: _resources(wall=16.0, io_bytes=1_300),
            3: _resources(wall=10.0, io_bytes=1_100, swap=4096),
        },
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["selected_worker_count"] == 0
    assert manifest["status"] == "STOP_NO_PARALLEL_WORKER_AUTHORIZED"
    assert manifest["selection"] == "NONE"
    assert not (manifest_path.parent / "execution_authorization.json").exists()
    assert manifest["candidate_decisions"]["3"]["swap_pass"] is False
    assert manifest["candidate_decisions"]["2"]["io_pass"] is False


def test_machine_used_ram_not_only_worker_rss_controls_ram_gate(
    tmp_path: Path,
) -> None:
    runner = FixtureRunner(_lineage())
    manifest_path, _, _ = _run(
        tmp_path,
        runner=runner,
        resources={
            1: _resources(wall=30.0, io_bytes=1_000),
            2: _resources(wall=18.0, io_bytes=1_100),
            3: _resources(
                wall=10.0,
                io_bytes=1_100,
                rss_bytes=8 * 1024**3,
                machine_used_bytes=14 * 1024**3 + 1,
            ),
        },
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["candidate_decisions"]["3"]["ram_pass"] is False
    assert manifest["selected_worker_count"] == 0
    assert manifest["status"] == "STOP_NO_PARALLEL_WORKER_AUTHORIZED"


def test_national_storage_projection_blocks_authorization(tmp_path: Path) -> None:
    runner = FixtureRunner(
        _lineage(), materialized_io_bytes=1 * 1024**3, site_model_bytes=1_000
    )
    manifest_path, _, _ = _run(
        tmp_path,
        runner=runner,
        resources={
            1: _resources(wall=30.0, io_bytes=1_000),
            3: _resources(wall=10.0, io_bytes=1_100),
        },
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "STOP_RESOURCE_POLICY"
    assert manifest["national_storage_projection"]["pass"] is False
    assert not (manifest_path.parent / "execution_authorization.json").exists()


def test_io_gate_falls_back_to_exact_core_materialized_bytes(
    tmp_path: Path,
) -> None:
    runner = FixtureRunner(_lineage())
    manifest_path, _, _ = _run(
        tmp_path,
        runner=runner,
        resources={
            1: _resources(wall=30.0, io_bytes=0),
            3: _resources(wall=10.0, io_bytes=0),
        },
    )
    decision = json.loads(manifest_path.read_text(encoding="utf-8"))[
        "candidate_decisions"
    ]["3"]
    assert decision["io_measurement_source"] == (
        "CORE_MATERIALIZED_ARTIFACT_BYTES"
    )
    assert decision["io_ratio"] == 1.0


def test_digest_mismatch_stops_without_trying_w2(tmp_path: Path) -> None:
    runner = FixtureRunner(_lineage(), digest_drift_worker=3)
    prefit_path = tmp_path / "prefit.json"
    prefit_sha = _prefit(prefit_path, runner.lineage)
    plan_path = tmp_path / "sharded_plan.json"
    plan_sha = _sharded_plan(plan_path, runner.lineage, prefit_sha)
    with pytest.raises(PhysicalBenchmarkInvariantError, match="W3 diverged"):
        run_physical_worker_benchmark_v12(
            runner=runner,
            output_dir=tmp_path / "benchmark",
            expected_lineage=runner.lineage,
            prefit_parity_manifest_path=prefit_path,
            prefit_parity_manifest_sha256=prefit_sha,
            sharded_execution_plan_path=plan_path,
            sharded_execution_plan_sha256=plan_sha,
            measurer=_measurer(
                {
                    1: _resources(wall=30.0, io_bytes=1_000),
                    3: _resources(wall=10.0, io_bytes=1_100),
                }
            ),
        )
    assert runner.calls == [3, 1]
    assert not (tmp_path / "benchmark" / "execution_authorization.json").exists()


@pytest.mark.parametrize(
    ("prefit_patch", "runner_folds"),
    [
        ({"fold1_opened": True}, ALLOWED_FOLDS),
        ({"status": "FAIL"}, ALLOWED_FOLDS),
        ({"lineage": {"wrong": _digest("wrong")}}, ALLOWED_FOLDS),
        ({}, (1, 2, 3)),
    ],
)
def test_folds_lineage_and_prefit_fail_closed(
    tmp_path: Path,
    prefit_patch: dict,
    runner_folds: tuple[int, ...],
) -> None:
    lineage = _lineage()
    runner = FixtureRunner(lineage, folds=runner_folds)
    prefit_path = tmp_path / "prefit.json"
    _prefit(prefit_path, lineage)
    payload = json.loads(prefit_path.read_text(encoding="utf-8"))
    payload.update(prefit_patch)
    prefit_path.write_text(json.dumps(payload), encoding="utf-8")
    prefit_sha = sha256_file(prefit_path)
    plan_path = tmp_path / "sharded_plan.json"
    plan_sha = _sharded_plan(plan_path, lineage, prefit_sha)
    with pytest.raises((PhysicalBenchmarkInvariantError, ValueError)):
        run_physical_worker_benchmark_v12(
            runner=runner,
            output_dir=tmp_path / "benchmark",
            expected_lineage=lineage,
            prefit_parity_manifest_path=prefit_path,
            prefit_parity_manifest_sha256=prefit_sha,
            sharded_execution_plan_path=plan_path,
            sharded_execution_plan_sha256=plan_sha,
            measurer=_measurer(
                {
                    1: _resources(wall=30.0, io_bytes=1_000),
                    3: _resources(wall=10.0, io_bytes=1_100),
                }
            ),
        )


def test_generic_fixture_authorization_is_rejected_before_artifact_validation(
    tmp_path: Path,
) -> None:
    runner = FixtureRunner(_lineage())
    manifest_path, prefit_sha, plan_sha = _run(
        tmp_path,
        runner=runner,
        resources={
            1: _resources(wall=30.0, io_bytes=1_000),
            3: _resources(wall=10.0, io_bytes=1_100),
        },
    )
    # The generic runner has no strict sharded child manifests or execution
    # lineage.  It must fail at that stronger boundary; its self-declared
    # trial files are not evidence eligible for an artifact-tamper test.
    with pytest.raises(
        PhysicalBenchmarkInvariantError,
        match="physical authorization lineage/status drifted",
    ):
        load_sharded_oof_authorization_v12(
            manifest_path,
            manifest_path.parent / "execution_authorization.json",
            expected_lineage=runner.lineage,
            expected_prefit_parity_manifest_sha256=prefit_sha,
            expected_sharded_execution_plan_sha256=plan_sha,
        )


def test_resource_metric_validation_rejects_negative_values() -> None:
    valid = _resources(wall=1.0, io_bytes=100)
    with pytest.raises(PhysicalBenchmarkInvariantError):
        replace(valid, swap_out_delta_bytes=-1)


def test_core_adapter_seals_exact_payload_spool_and_all_products(
    tmp_path: Path,
) -> None:
    lineage = _lineage()
    train_sha = {str(fold): _digest(f"train-{fold}") for fold in ALLOWED_FOLDS}
    group_sha = {str(fold): _digest(f"group-{fold}") for fold in ALLOWED_FOLDS}
    model_sha = {str(fold): _digest(f"model-{fold}") for fold in ALLOWED_FOLDS}
    heldout_sha = _digest("complete-heldout-payload")
    bridge_sha = _digest("direct-prefit-bridge")
    output_sha = {
        "top500": _digest("top500"),
        "top100": _digest("top100"),
        "outcomes": _digest("outcomes"),
    }
    core_digests = {
        "prefit_bridge": bridge_sha,
        "train_matrix": train_sha,
        "group_sequence": group_sha,
        "heldout_spool": heldout_sha,
        "model_by_outer": model_sha,
        "prerank500": output_sha["top500"],
        "top100": output_sha["top100"],
        "outcomes": output_sha["outcomes"],
    }
    core_io = {"total_materialized": 10_000}
    model_paths = {}
    train_paths = {}
    for fold in ALLOWED_FOLDS:
        path = tmp_path / f"model-{fold}.json"
        path.write_text(f"model {fold}\n", encoding="utf-8")
        model_paths[fold] = path
        train_path = tmp_path / f"train-{fold}.parquet"
        train_path.write_text(f"train {fold}\n", encoding="utf-8")
        train_paths[fold] = train_path
    replay_paths = {}
    for name in output_sha:
        path = tmp_path / f"{name}.parquet"
        path.write_text(f"{name}\n", encoding="utf-8")
        replay_paths[name] = path
    manifest = tmp_path / "core.json"
    payload = {
        "status": "EXECUTED_NO_READY_CLAIM",
        "run_kind": "PHYSICAL_BENCHMARK_MINIATURE",
        "miniature_fit": True,
        "logical_recipe_changes": False,
        "query_count": 512,
        "folds": [2, 3, 4],
        "fold0_opened": False,
        "fold1_opened": False,
        "positive_injection": False,
        "worker_count": 3,
        "shard_count": 3,
        "maximum_concurrent_workers": 3,
        "lineage": lineage,
        "digests": core_digests,
        "io_bytes": core_io,
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    artifact = SimpleNamespace(
        manifest_path=manifest,
        manifest_sha256=sha256_file(manifest),
        query_count=512,
        folds=ALLOWED_FOLDS,
        lineage=lineage,
        specs=(object(), object(), object()),
        build_artifacts=(),
        train=SimpleNamespace(
            matrices={
                fold: SimpleNamespace(
                    sha256=train_sha[str(fold)], path=train_paths[fold]
                )
                for fold in ALLOWED_FOLDS
            },
            group_sequence_sha256={
                fold: group_sha[str(fold)] for fold in ALLOWED_FOLDS
            },
        ),
        heldout=SimpleNamespace(
            artifact=SimpleNamespace(sha256=heldout_sha, size_bytes=0)
        ),
        models={
            fold: SimpleNamespace(
                sha256=model_sha[str(fold)], path=model_paths[fold]
            )
            for fold in ALLOWED_FOLDS
        },
        replay=SimpleNamespace(
            outputs={
                name: SimpleNamespace(sha256=value, path=replay_paths[name])
                for name, value in output_sha.items()
            }
        ),
        replay_artifacts=(),
        prefit_bridge=SimpleNamespace(sha256=bridge_sha),
        digests=core_digests,
        io_bytes=core_io,
    )
    product = physical_trial_product_from_sharded_v12(artifact)
    assert product.worker_count == 3
    assert product.digests["heldout_spool"] == heldout_sha
    assert set(product.digests) == set(REQUIRED_DIGESTS)

    payload["digests"]["heldout_spool"] = _digest("metadata-only")
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    artifact.manifest_sha256 = sha256_file(manifest)
    artifact.digests = payload["digests"]
    with pytest.raises(PhysicalBenchmarkInvariantError, match="parity surface"):
        physical_trial_product_from_sharded_v12(artifact)
