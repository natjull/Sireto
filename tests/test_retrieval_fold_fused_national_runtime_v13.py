from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import xgb_matcher.retrieval_fold_fused_national_runtime_v13 as national

from xgb_matcher.retrieval_fold_fused_national_runtime_v13 import (
    _V12_GUARD_KEYS,
    _artifact_record,
    _resolve_artifact,
    _validate_runtime_resource_guard_v13,
)
from xgb_matcher.retrieval_fold_fused_oof_v12 import sha256_file
from xgb_matcher.retrieval_fold_fused_physical_benchmark_v13 import (
    PhysicalBenchmarkInvariantErrorV13,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _valid_runtime_guard() -> dict[str, object]:
    base: dict[str, object] = {
        "schema_version": "sireto-retrieval-v12-national-worker-guard-1",
        "status": "PASS_W3",
        "selection_reason": "W3_CURRENT_STATE_PASS",
        "physical_authorized_maximum_concurrent_workers": 3,
        "selected_maximum_concurrent_workers": 3,
        "fixed_shard_count": 3,
        "memory_metric": "psutil.virtual_memory().used",
        "memory_used_before_bytes": 8 * 1024**3,
        "memory_used_after_bytes": 9 * 1024**3,
        "memory_used_peak_bytes": 9 * 1024**3,
        "memory_threshold_bytes": 14 * 1024**3,
        "swapout_metric": "vm_stat.Swapouts",
        "swapout_source": "vm_stat",
        "swapout_page_size_bytes": 4096,
        "swapout_before_pages": 100,
        "swapout_after_pages": 100,
        "swapout_delta_pages": 0,
        "swapout_delta_bytes": 0,
        "observation_seconds": 5.0,
        "observation_started_at_unix_seconds": 1000.0,
        "observation_finished_at_unix_seconds": 1005.0,
        "measurement_valid": True,
        "measured_before_first_shard_open": True,
        "worker_count_change_after_measurement": False,
        "automatic_retry_allowed": False,
    }
    assert set(base) == _V12_GUARD_KEYS
    guard = {
        **base,
        "base_guard_sha256": _sha_json(base),
        "storage_free_bytes": 30 * 1024**3,
        "storage_required_bytes": 20 * 1024**3,
        "storage_free_pass": True,
        "fallback_worker_count_allowed": False,
    }
    guard["record_sha256"] = _sha_json(guard)
    return guard


def _valid_fit_guard() -> dict[str, object]:
    guard: dict[str, object] = {
        "schema_version": national.FIT_GUARD_SCHEMA_VERSION,
        "status": "PASS_PREFIT_RESOURCES",
        "phase": "AFTER_BUILD_ACTORS_JOIN_BEFORE_FIRST_SITE_FIT",
        "build_actors_joined": True,
        "live_build_actor_count": 0,
        "fit_parallelism": 1,
        "outer_fit_order": [2, 3, 4],
        "gc_between_outer_fits": True,
        "memory_metric": "psutil.virtual_memory().used_and_available",
        "memory_total_bytes": 24 * 1024**3,
        "memory_used_peak_bytes": 13 * 1024**3,
        "memory_used_maximum_bytes": 14 * 1024**3,
        "memory_available_before_bytes": 11 * 1024**3,
        "memory_available_after_bytes": 10 * 1024**3,
        "memory_available_minimum_bytes": 10 * 1024**3,
        "memory_available_required_bytes": 10 * 1024**3,
        "swapout_metric": "vm_stat.Swapouts",
        "swapout_source": "vm_stat",
        "swapout_page_size_bytes": 4096,
        "swapout_before_pages": 100,
        "swapout_after_pages": 100,
        "swapout_delta_pages": 0,
        "swapout_delta_bytes": 0,
        "observation_seconds": 5.0,
        "observation_started_at_unix_seconds": 1000.0,
        "observation_finished_at_unix_seconds": 1005.0,
        "base_guard_sha256": _digest("fit-base-guard"),
        "automatic_retry_allowed": False,
    }
    guard["record_sha256"] = _sha_json(guard)
    return guard


def test_national_internal_artifact_record_rejects_external_file(
    tmp_path: Path,
) -> None:
    output = tmp_path / "national"
    output.mkdir()
    external = tmp_path / "external.json"
    external.write_text("{}\n", encoding="utf-8")
    with pytest.raises(
        PhysicalBenchmarkInvariantErrorV13,
        match="escaped the output tree",
    ):
        _artifact_record(external, output)


def test_national_internal_artifact_is_relative_and_inventory_bound(
    tmp_path: Path,
) -> None:
    output = tmp_path / "national"
    artifact = output / "build" / "manifest.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}\n", encoding="utf-8")
    record = _artifact_record(artifact, output)
    inventory = {
        "build/manifest.json": {
            "role": "national_artifact",
            "family": "support",
            "path": "build/manifest.json",
            "sha256": sha256_file(artifact),
            "size_bytes": artifact.stat().st_size,
        }
    }
    assert record["path"] == "build/manifest.json"
    assert (
        _resolve_artifact(
            output,
            record,
            label="build dataset",
            inventory=inventory,
        )
        == artifact.resolve()
    )

    absolute_record = dict(record)
    absolute_record["path"] = str(artifact.resolve())
    with pytest.raises(
        PhysicalBenchmarkInvariantErrorV13,
        match="not output-relative",
    ):
        _resolve_artifact(
            output,
            absolute_record,
            label="build dataset",
            inventory=inventory,
        )

    with pytest.raises(
        PhysicalBenchmarkInvariantErrorV13,
        match="bytes drifted",
    ):
        _resolve_artifact(
            output,
            record,
            label="build dataset",
            inventory={},
        )


def test_national_internal_record_refuses_hash_or_size_substitution(
    tmp_path: Path,
) -> None:
    output = tmp_path / "national"
    artifact = output / "replay" / "manifest.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}\n", encoding="utf-8")
    record = dict(_artifact_record(artifact, output))
    inventory = {
        str(record["path"]): {
            "role": "national_artifact",
            "family": "support",
            "path": record["path"],
            "sha256": record["sha256"],
            "size_bytes": record["size_bytes"],
        }
    }
    record["sha256"] = _digest("substitution")
    with pytest.raises(
        PhysicalBenchmarkInvariantErrorV13,
        match="bytes drifted",
    ):
        _resolve_artifact(
            output,
            record,
            label="replay dataset",
            inventory=inventory,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fallback_worker_count_allowed", True),
        ("storage_free_bytes", 10 * 1024**3),
        ("swapout_after_pages", 101),
        ("memory_used_peak_bytes", 15 * 1024**3),
    ],
)
def test_national_runtime_guard_recomputes_rehashed_arithmetic(
    field: str, value: object
) -> None:
    guard = _valid_runtime_guard()
    assert _validate_runtime_resource_guard_v13(guard) == guard["record_sha256"]
    guard[field] = value
    guard["record_sha256"] = _sha_json(
        {key: item for key, item in guard.items() if key != "record_sha256"}
    )
    with pytest.raises(
        PhysicalBenchmarkInvariantErrorV13,
        match="runtime guard arithmetic",
    ):
        _validate_runtime_resource_guard_v13(guard)


def test_national_writer_schema_reopens_full_authority_guard_and_inventory_dag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    output = tmp_path / "national"
    output.mkdir()
    authority_root = tmp_path / "authorities"
    authority_root.mkdir()
    source = authority_root / "source.parquet"
    source.write_bytes(b"sealed-source")
    prefit = authority_root / "prefit.json"
    prefit.write_text("{}\n", encoding="utf-8")
    plan = authority_root / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "input_lineage": {
                    "prefit_parity": {
                        "manifest_path": str(prefit),
                        "manifest_sha256": sha256_file(prefit),
                    }
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    authorization_path = authority_root / "authorization.json"
    authorization_path.write_text("{}\n", encoding="utf-8")
    benchmark_path = authority_root / "benchmark.json"

    implementation_sha = _digest("implementation")
    scientific_lineage = {"scene_ir_sha256": _digest("scene")}
    authorized_w3_lineage = {
        **scientific_lineage,
        "sharded_implementation_sha256": _digest("sharded"),
        "runtime_environment_sha256": _digest("runtime"),
        "v13_implementation_sha256": implementation_sha,
    }
    authorization_sha = sha256_file(authorization_path)
    benchmark_path.write_text(
        json.dumps({"w3": {"lineage": authorized_w3_lineage}}) + "\n",
        encoding="utf-8",
    )
    benchmark_sha = sha256_file(benchmark_path)
    lineage = {
        **authorized_w3_lineage,
        "physical_benchmark_manifest_sha256": benchmark_sha,
        "execution_authorization_sha256": authorization_sha,
    }
    specs = tuple(
        SimpleNamespace(
            digest=_digest(f"spec-{fold}"),
            as_manifest=lambda fold=fold: {"fold": fold},
        )
        for fold in (2, 3, 4)
    )

    def artifact(name: str) -> SimpleNamespace:
        path = output / name / "manifest.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"kind": name}) + "\n", encoding="utf-8")
        return SimpleNamespace(
            manifest_path=path,
            manifest_sha256=sha256_file(path),
            query_count=national.NATIONAL_QUERY_COUNT,
            lineage=lineage,
            specs=specs,
        )

    build = artifact("build_fragment_dataset")
    replay = artifact("compact_replay_dataset")
    canonical = artifact("canonical_datasets")
    guard = _valid_runtime_guard()
    fit_guard = _valid_fit_guard()
    gates = {
        "denominator": "ALL_USABLE_HUMAN_SIRET_LABELS_FOLDS_2_3_4",
        "gate_eligible": False,
        "decision": "NO_READY_DECISION",
    }
    inventory = national._national_inventory_v13(output, output / "manifest.json")
    config = {
        "authorities": {
            "v12_sharded_plan": {
                "path": str(plan),
                "sha256": sha256_file(plan),
            }
        },
        "input_lineage": {
            "source_candidate_path": str(source),
            "source_candidate_sha256": _digest("source-authority"),
        },
        "scientific_invariants": {
            "master_recall_at_2000": 0.995,
            "prerank_recall_at_500": 0.994,
            "exact_recall_at_100": 0.992,
            "exact_recall_at_100_wilson95_lower": 0.990,
            "minimum_each_outer_fold_exact_recall_at_100": 0.988,
        },
        "cache_neutral_io_gate": {
            "source_extent": {
                "production_projection": {
                    "columns": ["query_id"],
                    "sha256": _digest("projection"),
                },
                "national_expected_unique_row_groups": 635,
                "national_expected_logical_compressed_bytes": 4_646_932_145,
            }
        },
        "national_execution_protocol": {
            "canonical_output_dir": str(output),
            "attempt_consumed_marker_path": str(
                tmp_path / ".national_attempt.json"
            ),
            "stage_directory_prefix": ".national-test-stage-",
            "stage_identity": "SHA256_CONFIG_AUTH_BENCHMARK_IMPLEMENTATION_SOURCE_PLAN_PREFIT_OUTPUT",
            "success_promotion": "ATOMIC_RENAME_SAME_FILESYSTEM",
            "automatic_retry_allowed": False,
            "stage_retained_on_failure": True,
            "exception_status": "STOP_NATIONAL_EXECUTION_EXCEPTION",
        },
    }
    authorization_identity = {
        "sharded_execution_plan_sha256": sha256_file(plan),
        "prefit_parity_manifest_sha256": sha256_file(prefit),
    }
    attempt_base = national._national_attempt_base_v13(
        config_sha256=national.CONFIG_SHA256,
        authorization_sha256=authorization_sha,
        benchmark_sha256=benchmark_sha,
        implementation_sha256=implementation_sha,
        source_candidate_sha256=config["input_lineage"][
            "source_candidate_sha256"
        ],
        sharded_execution_plan_sha256=sha256_file(plan),
        prefit_parity_manifest_sha256=sha256_file(prefit),
        canonical_output_dir=output,
    )
    attempt_id = _sha_json(attempt_base)
    _canonical, stage, marker = national._national_execution_paths_v13(
        config, requested_output=output, attempt_id=attempt_id
    )
    marker_sha = national._consume_national_attempt_v13(
        marker_path=marker,
        payload=national._national_attempt_payload_v13(
            base=attempt_base,
            attempt_id=attempt_id,
            stage_output_dir=stage,
        ),
    )
    identity = national._national_identity_v13(
        authorization_sha256=authorization_sha,
        physical_benchmark_manifest_sha256=benchmark_sha,
        implementation_sha256=implementation_sha,
        lineage=lineage,
        source_candidate_sha256=config["input_lineage"][
            "source_candidate_sha256"
        ],
        sharded_execution_plan_sha256=sha256_file(plan),
        prefit_parity_manifest_sha256=sha256_file(prefit),
        runtime_resource_guard_sha256=guard["record_sha256"],
        fit_resource_guard_sha256=fit_guard["record_sha256"],
        attempt_marker_sha256=marker_sha,
        attempt_id=attempt_id,
        canonical_output_dir=str(output),
        artifact_inventory_sha256=_sha_json(inventory),
        source_spec_sha256=[value.digest for value in specs],
        source_row_group_ledger_sha256=[
            _digest(f"row-group-ledger-{fold}") for fold in (2, 3, 4)
        ],
        source_row_group_union_sha256=_digest("row-group-union"),
        source_row_group_logical_compressed_bytes=4_646_932_145,
        build_fragment_dataset_sha256=build.manifest_sha256,
        compact_replay_dataset_sha256=replay.manifest_sha256,
        zero_copy_canonical_run_sha256=canonical.manifest_sha256,
        gates=gates,
    )
    manifest = {
        "schema_version": national.SCHEMA_VERSION,
        "status": "EXECUTED_NO_READY_CLAIM",
        "run_id": _sha_json(identity),
        "identity": identity,
        "run_kind": "NATIONAL_OOF_FOLDS_2_3_4_ZERO_COPY_V13",
        "config_sha256": national.CONFIG_SHA256,
        "authorization": {
            "path": str(authorization_path),
            "sha256": authorization_sha,
            "benchmark_manifest_path": str(benchmark_path),
            "benchmark_manifest_sha256": benchmark_sha,
        },
        "source_candidate": {
            "path": str(source),
            "sha256": config["input_lineage"]["source_candidate_sha256"],
        },
        "sharded_execution_plan": {
            "path": str(plan),
            "sha256": sha256_file(plan),
        },
        "prefit_parity": {
            "path": str(prefit),
            "sha256": sha256_file(prefit),
        },
        "runtime_resource_guard": guard,
        "fit_resource_guard": fit_guard,
        "national_attempt": {
            "path": str(marker),
            "sha256": marker_sha,
            "attempt_id": attempt_id,
            "stage_directory_name": stage.name,
        },
        "lineage": lineage,
        "query_count": national.NATIONAL_QUERY_COUNT,
        "worker_count": 3,
        "folds": [2, 3, 4],
        "fold0_opened": False,
        "fold1_opened": False,
        "positive_injection": False,
        "source_specs": [value.as_manifest() for value in specs],
        "source_row_group_ledgers": [],
        "build_fragment_dataset": _artifact_record(build.manifest_path, output),
        "compact_replay_dataset": _artifact_record(replay.manifest_path, output),
        "zero_copy_canonical_run": _artifact_record(
            canonical.manifest_path, output
        ),
        "canonical_datasets": list(national.CANONICAL_FAMILIES),
        "gates": gates,
        "no_merged_files": True,
        "top500_compact": True,
        "top100_full": True,
        "fold1_ready_claimed": False,
        "artifact_inventory": inventory,
        "materialized_output_bytes": 0,
    }
    manifest_path = output / "manifest.json"
    child_bytes = sum(int(value["size_bytes"]) for value in inventory)
    for _ in range(16):
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        observed = child_bytes + manifest_path.stat().st_size
        if manifest["materialized_output_bytes"] == observed:
            break
        manifest["materialized_output_bytes"] = observed
    else:  # pragma: no cover
        raise AssertionError("fixed point")

    auth_calls: list[Path] = []

    def load_auth(path: Path, **_kwargs):
        auth_calls.append(Path(path).resolve())
        return {"authorization_identity": authorization_identity}

    monkeypatch.setattr(national, "load_execution_authorization_v13", load_auth)
    monkeypatch.setattr(
        national,
        "v13_implementation_lineage",
        lambda _root: (implementation_sha, ()),
    )
    monkeypatch.setattr(national, "load_physical_config_v13", lambda _path: config)
    monkeypatch.setattr(
        national,
        "sharded_execution_lineage_v12",
        lambda observed, **_kwargs: {
            **observed,
            "sharded_implementation_sha256": authorized_w3_lineage[
                "sharded_implementation_sha256"
            ],
            "runtime_environment_sha256": authorized_w3_lineage[
                "runtime_environment_sha256"
            ],
        },
    )
    monkeypatch.setattr(national, "load_build_fragment_dataset_v13", lambda *_args, **_kwargs: build)
    monkeypatch.setattr(national, "load_compact_replay_dataset_v13", lambda *_args, **_kwargs: replay)
    monkeypatch.setattr(
        national,
        "load_zero_copy_canonical_run_v13",
        lambda *_args, **_kwargs: canonical,
    )
    fake_row_group_artifacts = tuple(
        SimpleNamespace(
            manifest_path=output / f"fake-ledger-{fold}.json",
            manifest_sha256=_digest(f"row-group-ledger-{fold}"),
        )
        for fold in (2, 3, 4)
    )
    fake_actual_reads = SimpleNamespace(
        artifacts=fake_row_group_artifacts,
        union_sha256=_digest("row-group-union"),
        logical_compressed_bytes=4_646_932_145,
    )
    monkeypatch.setattr(
        national,
        "load_actor_row_group_ledger_records_v13",
        lambda *_args, **_kwargs: fake_row_group_artifacts,
    )
    monkeypatch.setattr(
        national,
        "_national_projection_totals_v13",
        lambda *_args, **_kwargs: (635, 4_646_932_145),
    )
    monkeypatch.setattr(
        national,
        "validate_actual_row_group_ledger_set_v13",
        lambda *_args, **_kwargs: fake_actual_reads,
    )
    monkeypatch.setattr(
        national,
        "aggregate_compact_replay_gates_v13",
        lambda _replay, _thresholds: gates,
    )

    loaded = national.load_national_fragment_oof_v13(
        manifest_path,
        expected_manifest_sha256=sha256_file(manifest_path),
        workspace_root=workspace,
    )
    assert loaded.run_id == manifest["run_id"]
    assert auth_calls == [authorization_path.resolve()]

    forged = json.loads(manifest_path.read_text(encoding="utf-8"))
    forged["gates"]["decision"] = "FORGED_READY"
    forged["identity"]["gates_sha256"] = _sha_json(forged["gates"])
    forged["run_id"] = _sha_json(forged["identity"])
    for _ in range(16):
        manifest_path.write_text(
            json.dumps(forged, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        observed = child_bytes + manifest_path.stat().st_size
        if forged["materialized_output_bytes"] == observed:
            break
        forged["materialized_output_bytes"] = observed
    with pytest.raises(
        PhysicalBenchmarkInvariantErrorV13,
        match="prefit gates drifted",
    ):
        national.load_national_fragment_oof_v13(
            manifest_path,
            expected_manifest_sha256=sha256_file(manifest_path),
            workspace_root=workspace,
        )


def _test_national_protocol(tmp_path: Path) -> dict[str, object]:
    return {
        "national_execution_protocol": {
            "canonical_output_dir": str(tmp_path / "national"),
            "attempt_consumed_marker_path": str(tmp_path / ".attempt.json"),
            "stage_directory_prefix": ".national.stage-",
            "stage_identity": "SHA256_CONFIG_AUTH_BENCHMARK_IMPLEMENTATION_SOURCE_PLAN_PREFIT_OUTPUT",
            "success_promotion": "ATOMIC_RENAME_SAME_FILESYSTEM",
            "automatic_retry_allowed": False,
            "stage_retained_on_failure": True,
            "exception_status": "STOP_NATIONAL_EXECUTION_EXCEPTION",
        },
        "input_lineage": {
            "source_candidate_path": str(tmp_path / "source.parquet"),
            "source_candidate_sha256": _digest("source"),
        },
        "authorities": {
            "v12_sharded_plan": {
                "path": "plan.json",
                "sha256": _digest("plan"),
            }
        },
    }


def _test_prepared(tmp_path: Path) -> SimpleNamespace:
    source = tmp_path / "source.parquet"
    source.write_bytes(b"source")
    return SimpleNamespace(
        national_query_ids=range(national.NATIONAL_QUERY_COUNT),
        sharded_execution_plan_sha256=_digest("plan"),
        prefit_parity_manifest_sha256=_digest("prefit"),
        scene_cache=SimpleNamespace(source=source),
    )


def test_national_public_runner_atomically_promotes_one_stage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _test_national_protocol(tmp_path)
    prepared = _test_prepared(tmp_path)
    canonical = tmp_path / "national"
    authorization = tmp_path / "authorization.json"
    benchmark = tmp_path / "benchmark.json"
    authorization.write_text("{}\n", encoding="utf-8")
    benchmark.write_text("{}\n", encoding="utf-8")
    auth_sha = sha256_file(authorization)
    benchmark_sha = sha256_file(benchmark)
    implementation_sha = _digest("implementation")
    monkeypatch.setattr(national, "load_physical_config_v13", lambda _path: config)
    monkeypatch.setattr(
        national,
        "load_execution_authorization_v13",
        lambda *_args, **_kwargs: {
            "authorization_identity": {
                "sharded_execution_plan_sha256": _digest("plan"),
                "prefit_parity_manifest_sha256": _digest("prefit"),
            }
        },
    )
    monkeypatch.setattr(
        national,
        "v13_implementation_lineage",
        lambda _root: (
            implementation_sha,
            ({"path": "implementation.py", "sha256": implementation_sha, "size_bytes": 1},),
        ),
    )

    def fake_stage(**kwargs):
        stage = Path(kwargs["output_dir"])
        stage.mkdir()
        manifest = stage / "manifest.json"
        manifest.write_text("{}\n", encoding="utf-8")
        kwargs["phase_state"]["phase"] = "STAGE_SEALED"
        return national.NationalFragmentOOFV13(
            manifest_path=manifest,
            manifest_sha256=sha256_file(manifest),
            run_id=_digest("run"),
            query_count=national.NATIONAL_QUERY_COUNT,
            lineage={"scene": _digest("scene")},
            gates={},
        )

    monkeypatch.setattr(
        national, "_execute_national_fragment_oof_stage_v13", fake_stage
    )
    loaded_paths: list[Path] = []

    def fake_load(path: Path, **_kwargs):
        loaded_paths.append(Path(path).resolve())
        assert canonical.is_dir()
        return SimpleNamespace(manifest_path=Path(path).resolve())

    monkeypatch.setattr(national, "load_national_fragment_oof_v13", fake_load)
    result = national.execute_national_fragment_oof_v13(
        prepared=prepared,
        workspace_root=tmp_path,
        output_dir=canonical,
        physical_benchmark_manifest=benchmark,
        physical_benchmark_manifest_sha256=benchmark_sha,
        execution_authorization=authorization,
        execution_authorization_sha256=auth_sha,
    )
    assert result.manifest_path == canonical / "manifest.json"
    assert loaded_paths == [canonical / "manifest.json"]
    assert (tmp_path / ".attempt.json").is_file()
    assert not list(tmp_path.glob(".national.stage-*"))


def test_national_exception_keeps_stage_stop_and_forbids_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _test_national_protocol(tmp_path)
    prepared = _test_prepared(tmp_path)
    canonical = tmp_path / "national"
    authorization = tmp_path / "authorization.json"
    benchmark = tmp_path / "benchmark.json"
    authorization.write_text("{}\n", encoding="utf-8")
    benchmark.write_text("{}\n", encoding="utf-8")
    auth_sha = sha256_file(authorization)
    benchmark_sha = sha256_file(benchmark)
    implementation_sha = _digest("implementation")
    monkeypatch.setattr(national, "load_physical_config_v13", lambda _path: config)
    monkeypatch.setattr(
        national,
        "load_execution_authorization_v13",
        lambda *_args, **_kwargs: {
            "authorization_identity": {
                "sharded_execution_plan_sha256": _digest("plan"),
                "prefit_parity_manifest_sha256": _digest("prefit"),
            }
        },
    )
    monkeypatch.setattr(
        national,
        "v13_implementation_lineage",
        lambda _root: (
            implementation_sha,
            ({"path": "implementation.py", "sha256": implementation_sha, "size_bytes": 1},),
        ),
    )

    def fail_stage(**kwargs):
        stage = Path(kwargs["output_dir"])
        stage.mkdir()
        (stage / "partial.bin").write_bytes(b"partial")
        kwargs["phase_state"]["phase"] = "SEQUENTIAL_SITE_FITS"
        raise RuntimeError("fit failed\nwith unsafe control: \N{SNOWMAN}")

    monkeypatch.setattr(
        national, "_execute_national_fragment_oof_stage_v13", fail_stage
    )
    call = dict(
        prepared=prepared,
        workspace_root=tmp_path,
        output_dir=canonical,
        physical_benchmark_manifest=benchmark,
        physical_benchmark_manifest_sha256=benchmark_sha,
        execution_authorization=authorization,
        execution_authorization_sha256=auth_sha,
    )
    with pytest.raises(RuntimeError, match="fit failed"):
        national.execute_national_fragment_oof_v13(**call)
    assert not canonical.exists()
    stages = list(tmp_path.glob(".national.stage-*"))
    assert len(stages) == 1
    stop = json.loads((stages[0] / "manifest.json").read_text(encoding="utf-8"))
    assert stop["status"] == "STOP_NATIONAL_EXECUTION_EXCEPTION"
    assert stop["failed_phase"] == "SEQUENTIAL_SITE_FITS"
    assert "\n" not in stop["exception_message_sanitized"]
    assert stop["automatic_retry_allowed"] is False
    with pytest.raises(
        PhysicalBenchmarkInvariantErrorV13, match="unavailable"
    ):
        national.execute_national_fragment_oof_v13(**call)


def test_national_post_o_excl_marker_fault_publishes_stop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _test_national_protocol(tmp_path)
    prepared = _test_prepared(tmp_path)
    canonical = tmp_path / "national"
    authorization = tmp_path / "authorization.json"
    benchmark = tmp_path / "benchmark.json"
    authorization.write_text("{}\n", encoding="utf-8")
    benchmark.write_text("{}\n", encoding="utf-8")
    auth_sha = sha256_file(authorization)
    benchmark_sha = sha256_file(benchmark)
    implementation_sha = _digest("implementation")
    monkeypatch.setattr(national, "load_physical_config_v13", lambda _path: config)
    monkeypatch.setattr(
        national,
        "load_execution_authorization_v13",
        lambda *_args, **_kwargs: {
            "authorization_identity": {
                "sharded_execution_plan_sha256": _digest("plan"),
                "prefit_parity_manifest_sha256": _digest("prefit"),
            }
        },
    )
    monkeypatch.setattr(
        national,
        "v13_implementation_lineage",
        lambda _root: (
            implementation_sha,
            ({"path": "implementation.py", "sha256": implementation_sha, "size_bytes": 1},),
        ),
    )
    real_fsync_directory = national._fsync_directory_v13
    calls = 0

    def fail_first_directory_fsync(path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("marker parent fsync failed")
        real_fsync_directory(path)

    monkeypatch.setattr(
        national, "_fsync_directory_v13", fail_first_directory_fsync
    )
    call = dict(
        prepared=prepared,
        workspace_root=tmp_path,
        output_dir=canonical,
        physical_benchmark_manifest=benchmark,
        physical_benchmark_manifest_sha256=benchmark_sha,
        execution_authorization=authorization,
        execution_authorization_sha256=auth_sha,
    )
    with pytest.raises(
        national._NationalAttemptConsumptionErrorV13,
        match="after O_EXCL",
    ):
        national.execute_national_fragment_oof_v13(**call)

    marker = tmp_path / ".attempt.json"
    assert marker.is_file()
    assert not canonical.exists()
    stages = list(tmp_path.glob(".national.stage-*"))
    assert len(stages) == 1
    stop = json.loads((stages[0] / "manifest.json").read_text(encoding="utf-8"))
    assert stop["status"] == "STOP_NATIONAL_EXECUTION_EXCEPTION"
    assert stop["failed_phase"] == "ATOMIC_ATTEMPT_MARKER_CREATION"
    assert stop["attempt_marker"]["sha256"] == sha256_file(marker)
    assert stop["automatic_retry_allowed"] is False
    with pytest.raises(
        PhysicalBenchmarkInvariantErrorV13, match="unavailable"
    ):
        national.execute_national_fragment_oof_v13(**call)


def test_national_failed_strict_reload_rolls_promotion_back_to_stage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _test_national_protocol(tmp_path)
    prepared = _test_prepared(tmp_path)
    canonical = tmp_path / "national"
    authorization = tmp_path / "authorization.json"
    benchmark = tmp_path / "benchmark.json"
    authorization.write_text("{}\n", encoding="utf-8")
    benchmark.write_text("{}\n", encoding="utf-8")
    auth_sha = sha256_file(authorization)
    benchmark_sha = sha256_file(benchmark)
    implementation_sha = _digest("implementation")
    monkeypatch.setattr(national, "load_physical_config_v13", lambda _path: config)
    monkeypatch.setattr(
        national,
        "load_execution_authorization_v13",
        lambda *_args, **_kwargs: {
            "authorization_identity": {
                "sharded_execution_plan_sha256": _digest("plan"),
                "prefit_parity_manifest_sha256": _digest("prefit"),
            }
        },
    )
    monkeypatch.setattr(
        national,
        "v13_implementation_lineage",
        lambda _root: (
            implementation_sha,
            ({"path": "implementation.py", "sha256": implementation_sha, "size_bytes": 1},),
        ),
    )

    def fake_stage(**kwargs):
        stage = Path(kwargs["output_dir"])
        stage.mkdir()
        manifest = stage / "manifest.json"
        manifest.write_text("{}\n", encoding="utf-8")
        return national.NationalFragmentOOFV13(
            manifest_path=manifest,
            manifest_sha256=sha256_file(manifest),
            run_id=_digest("run"),
            query_count=national.NATIONAL_QUERY_COUNT,
            lineage={"scene": _digest("scene")},
            gates={},
        )

    monkeypatch.setattr(
        national, "_execute_national_fragment_oof_stage_v13", fake_stage
    )
    monkeypatch.setattr(
        national,
        "load_national_fragment_oof_v13",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("strict reload failed")
        ),
    )
    with pytest.raises(RuntimeError, match="strict reload failed"):
        national.execute_national_fragment_oof_v13(
            prepared=prepared,
            workspace_root=tmp_path,
            output_dir=canonical,
            physical_benchmark_manifest=benchmark,
            physical_benchmark_manifest_sha256=benchmark_sha,
            execution_authorization=authorization,
            execution_authorization_sha256=auth_sha,
        )
    assert not canonical.exists()
    stages = list(tmp_path.glob(".national.stage-*"))
    assert len(stages) == 1
    stop = json.loads((stages[0] / "manifest.json").read_text(encoding="utf-8"))
    assert stop["status"] == "STOP_NATIONAL_EXECUTION_EXCEPTION"
    assert stop["failed_phase"] == "STRICT_PROMOTED_RELOAD"


def test_national_prefit_guard_uses_frozen_ram_budget_and_swap(
    tmp_path: Path,
) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text("{}\n", encoding="utf-8")
    base = {
        key: value
        for key, value in _valid_runtime_guard().items()
        if key in national._V12_GUARD_KEYS
    }
    base["record_sha256"] = _sha_json(base)
    readings = iter(
        (
            SimpleNamespace(total=24 * 1024**3, used=12 * 1024**3, available=11 * 1024**3),
            SimpleNamespace(total=24 * 1024**3, used=13 * 1024**3, available=10 * 1024**3),
        )
    )
    guard = national._fit_resource_guard_v13(
        sharded_execution_plan_path=plan,
        memory_used_maximum_bytes=14 * 1024**3,
        memory_reader=lambda: next(readings),
        guard_reader=lambda *_args: base,
    )
    assert guard["status"] == "PASS_PREFIT_RESOURCES"
    assert national._validate_fit_resource_guard_v13(guard) == guard["record_sha256"]

    stopped = dict(guard)
    stopped["memory_available_after_bytes"] = 9 * 1024**3
    stopped["memory_available_minimum_bytes"] = 9 * 1024**3
    stopped["status"] = "STOP_PREFIT_RESOURCES"
    stopped["record_sha256"] = _sha_json(
        {key: value for key, value in stopped.items() if key != "record_sha256"}
    )
    with pytest.raises(
        PhysicalBenchmarkInvariantErrorV13, match="fit guard arithmetic"
    ):
        national._validate_fit_resource_guard_v13(stopped)


def test_national_internal_manifest_absolute_path_is_rejected(
    tmp_path: Path,
) -> None:
    root = tmp_path / "national"
    child = root / "child"
    child.mkdir(parents=True)
    (child / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "sireto-retrieval-test-1",
                "artifact": {"path": str(tmp_path / "external.parquet")},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        PhysicalBenchmarkInvariantErrorV13, match="absolute path"
    ):
        national._validate_relocatable_manifest_tree_v13(root)
