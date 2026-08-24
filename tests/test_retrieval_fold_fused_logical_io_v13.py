from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from xgb_matcher.retrieval_fold_fused_logical_io_v13 import (
    CANONICAL_DATASET_KEYS,
    LogicalIOInvariantError,
    REQUIRED_LOGICAL_DIGEST_KEYS,
    build_immutable_w1_io_projection_v13,
    compare_w3_to_immutable_w1_io_v13,
    load_immutable_w1_io_projection_v13,
    load_logical_io_gate_v13,
)
from xgb_matcher.retrieval_fold_fused_oof_v12 import sha256_file
from xgb_matcher.retrieval_fold_fused_row_group_ledger_v13 import (
    ActualRowGroupLedgerSetV13,
    ActualRowGroupLedgerSinkV13,
    actor_row_group_ledger_record_v13,
)
from xgb_matcher.retrieval_top100_cascade_v12 import production_projection
from xgb_matcher.retrieval_v12_scene_ir import SceneIRRowGroupReadEvent


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _logical_digests() -> dict[str, str]:
    return {name: _digest(name) for name in REQUIRED_LOGICAL_DIGEST_KEYS}


def _canonical_sha(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return sha256_file(path)


def _spec(
    *,
    ordinal: int,
    query_ids: list[str],
    query_folds: list[int],
    query_start: int,
    ranges: list[list[int]],
) -> dict[str, object]:
    sequence = [[query_id, fold] for query_id, fold in zip(query_ids, query_folds, strict=True)]
    return {
        "schema_version": "sireto-retrieval-v12-shard-spec-1",
        "shard_id": f"fold{ordinal + 2}",
        "shard_ordinal": ordinal,
        "sequence_start": query_start,
        "sequence_stop": query_start + len(query_ids),
        "query_ids": query_ids,
        "query_folds": query_folds,
        "source_ordinals": list(range(query_start, query_start + len(query_ids))),
        "source_ranges": ranges,
        "candidate_row_count": sum(stop - start for start, stop in ranges),
        "query_sequence_sha256": _canonical_sha(sequence),
        "fold0_opened": False,
        "fold1_opened": False,
    }


def _projected_bytes(source: Path, row_groups: list[int]) -> int:
    parquet = pq.ParquetFile(source)
    columns = production_projection(parquet.schema_arrow.names)
    indexes = {name: index for index, name in enumerate(parquet.schema_arrow.names)}
    return sum(
        int(parquet.metadata.row_group(row_group).column(indexes[name]).total_compressed_size)
        for row_group in row_groups
        for name in columns
    )


def _baseline_fixture(root: Path, *, baseline_materialized: int = 1_000_000) -> dict[str, object]:
    workspace = root / "workspace"
    config_dir = workspace / "config"
    config_dir.mkdir(parents=True)
    source = root / "sealed" / "source.parquet"
    source.parent.mkdir(parents=True)
    query_folds = [2] * 170 + [3] * 170 + [4] * 172
    pq.write_table(
        pa.table(
            {
                "query_id": [f"q{index:04d}" for index in range(512)],
                "fold": query_folds,
                "candidate": list(range(512)),
                "ground_truth_state": ["SECRET"] * 512,
            }
        ),
        source,
        compression="zstd",
        row_group_size=128,
    )
    source_sha = sha256_file(source)
    parquet = pq.ParquetFile(source)
    query_ids = [f"q{index:04d}" for index in range(512)]
    w1_spec = _spec(
        ordinal=0,
        query_ids=query_ids,
        query_folds=query_folds,
        query_start=0,
        ranges=[[0, 512]],
    )
    w1_manifest = root / "benchmark" / "trial_w1" / "manifest.json"
    w1_sha = _write_json(
        w1_manifest,
        {
            "schema_version": "sireto-retrieval-v12-sharded-oof-manifest-1",
            "run_kind": "PHYSICAL_BENCHMARK_MINIATURE",
            "status": "EXECUTED_NO_READY_CLAIM",
            "query_count": 512,
            "folds": [2, 3, 4],
            "worker_count": 1,
            "shard_count": 1,
            "fold0_opened": False,
            "fold1_opened": False,
            "positive_injection": False,
            "specs": [w1_spec],
        },
    )
    plan = config_dir / "plan.json"
    plan_sha = _write_json(
        plan,
        {
            "input_lineage": {
                "scene_ir": {
                    "source_candidate_path": str(source),
                    "source_candidate_sha256": source_sha,
                    "source_candidate_size_bytes": source.stat().st_size,
                    "candidate_rows": 512,
                    "source_row_groups": parquet.num_row_groups,
                }
            }
        },
    )
    digests = {name: _digest(name) for name in ("heldout", "fit", "top100")}
    benchmark = root / "benchmark" / "manifest.json"
    benchmark_sha = _write_json(
        benchmark,
        {
            "schema_version": "sireto-retrieval-v12-physical-worker-benchmark-1",
            "status": "STOP_NO_PARALLEL_WORKER_AUTHORIZED",
            "selected_worker_count": 0,
            "query_count": 512,
            "folds": [2, 3, 4],
            "fold0_opened": False,
            "fold1_opened": False,
            "positive_injection": False,
            "sharded_execution_plan": {"path": str(plan), "sha256": plan_sha},
            "trials": {
                "1": {
                    "artifact_manifest": {
                        "path": "trial_w1/manifest.json",
                        "sha256": w1_sha,
                    },
                    "materialized_io_bytes": baseline_materialized,
                    "digests": digests,
                }
            },
        },
    )
    projected_columns = list(production_projection(parquet.schema_arrow.names))
    projection_sha = _canonical_sha(projected_columns)
    logical_bytes = _projected_bytes(source, [0, 1, 2, 3])
    config = config_dir / "v13.json"
    config_sha = _write_json(
        config,
        {
            "schema_version": "sireto-retrieval-v13-physical-execution-1",
            "status": "PREREGISTERED_V13_PHYSICAL_BENCHMARK_NOT_RUN",
            "revision_scope": "PHYSICAL_ONLY",
            "logical_recipe_changes": False,
            "national_oof_allowed": False,
            "authorities": {
                "v12_sharded_plan": {
                    "path": "config/plan.json",
                    "sha256": plan_sha,
                }
            },
            "immutable_v12_stop": {
                "verdict": "STOP_NO_PARALLEL_WORKER_AUTHORIZED",
                "reinterpretation_forbidden": True,
                "replacement_forbidden": True,
                "execution_authorization_published": False,
                "benchmark_manifest_path": str(benchmark),
                "benchmark_manifest_sha256": benchmark_sha,
                "w1_trial": {
                    "relative_manifest_path": "trial_w1/manifest.json",
                    "manifest_sha256": w1_sha,
                    "query_count": 512,
                    "folds": [2, 3, 4],
                    "materialized_bytes": baseline_materialized,
                    "fold0_opened": False,
                    "fold1_opened": False,
                    "positive_injection": False,
                    "digests": digests,
                },
            },
            "input_lineage": {"source_candidate_sha256": source_sha},
            "cache_neutral_io_gate": {
                "source_extent": {
                    "production_projection": {
                        "source_schema_column_count": 4,
                        "projected_column_count": len(projected_columns),
                        "derivation": "retrieval_v12_scene_ir.production_projection(source_schema_names)",
                        "canonical_encoding": "JSON_UTF8_ENSURE_ASCII_FALSE_COMPACT_ARRAY_IN_SOURCE_ORDER",
                        "sha256": projection_sha,
                        "columns": projected_columns,
                        "excluded_source_columns": ["ground_truth_state"],
                        "drift_policy": "STOP_V13_IO_BEFORE_W3",
                    },
                    "logical_amplification_maximum": 1.0,
                    "sealed_sample_expected_unique_row_groups": 4,
                    "sealed_sample_expected_logical_compressed_bytes": logical_bytes,
                },
                "materialized": {
                    "baseline_w1_bytes": baseline_materialized,
                    "maximum_ratio_numerator": 115,
                    "maximum_ratio_denominator": 100,
                    "comparison": "W3_BYTES_TIMES_100_LE_W1_BYTES_TIMES_115",
                },
            },
        },
    )
    return {
        "source": source,
        "source_sha": source_sha,
        "projection_sha": projection_sha,
        "projection_columns": tuple(projected_columns),
        "plan_sha": plan_sha,
        "query_ids": query_ids,
        "query_folds": query_folds,
        "config": config,
        "config_sha": config_sha,
        "logical_bytes": logical_bytes,
        "baseline_materialized": baseline_materialized,
    }


def _actual_read_ledgers(
    root: Path,
    baseline: dict[str, object],
    specs: list[dict[str, object]],
    inventory: list[dict[str, object]],
) -> tuple[list[dict[str, object]], ActualRowGroupLedgerSetV13]:
    source = Path(baseline["source"])
    parquet = pq.ParquetFile(source)
    columns = tuple(baseline["projection_columns"])
    indexes = {
        name: ordinal for ordinal, name in enumerate(parquet.schema_arrow.names)
    }
    starts: list[int] = []
    stops: list[int] = []
    absolute = 0
    for group in range(parquet.num_row_groups):
        starts.append(absolute)
        absolute += int(parquet.metadata.row_group(group).num_rows)
        stops.append(absolute)
    lineage = {
        "source_candidate_sha256": str(baseline["source_sha"]),
        "source_manifest_sha256": _digest("source-manifest"),
        "scene_ir_sha256": _digest("scene-ir"),
        "v13_implementation_sha256": _digest("implementation"),
    }
    artifacts = []
    records: list[dict[str, object]] = []
    actor_sets: list[set[int]] = []
    for ordinal, spec in enumerate(specs):
        touched: set[int] = set()
        for start, stop in spec["source_ranges"]:
            for group, (row_start, row_stop) in enumerate(zip(starts, stops)):
                if row_start < int(stop) and row_stop > int(start):
                    touched.add(group)
        sink = ActualRowGroupLedgerSinkV13(
            shard_spec=spec,
            execution_lineage=lineage,
            sharded_execution_plan_sha256=str(baseline["plan_sha"]),
            expected_projection_sha256=str(baseline["projection_sha"]),
            expected_projection_column_count=len(columns),
        )
        for group in sorted(touched):
            metadata = parquet.metadata.row_group(group)
            sink(
                SceneIRRowGroupReadEvent(
                    source_path=str(source),
                    source_candidate_sha256=str(baseline["source_sha"]),
                    source_manifest_sha256=lineage["source_manifest_sha256"],
                    scene_ir_build_id=lineage["scene_ir_sha256"],
                    source_row_group_count=parquet.num_row_groups,
                    source_row_group=group,
                    row_start=starts[group],
                    row_stop=stops[group],
                    row_count=stops[group] - starts[group],
                    projected_columns=columns,
                    projection_sha256=str(baseline["projection_sha"]),
                    projected_compressed_bytes=sum(
                        int(
                            metadata.column(indexes[name]).total_compressed_size
                        )
                        for name in columns
                    ),
                )
            )
        artifact = sink.seal(
            root
            / "worker_build"
            / f"fold{ordinal + 2}"
            / "source_row_group_ledger"
        )
        record = actor_row_group_ledger_record_v13(artifact, root=root)
        records.append(record)
        artifacts.append(artifact)
        actor_sets.append(set(artifact.row_group_ordinals))
        for path_key, sha_key, size_key, role in (
            (
                "manifest_path",
                "manifest_sha256",
                "manifest_size_bytes",
                "actual_row_group_ledger_manifest",
            ),
            (
                "parquet_path",
                "parquet_sha256",
                "parquet_size_bytes",
                "actual_row_group_ledger_rows",
            ),
        ):
            inventory.append(
                {
                    "role": role,
                    "family": "source_read_ledgers",
                    "path": record[path_key],
                    "sha256": record[sha_key],
                    "size_bytes": record[size_key],
                }
            )
    union = tuple(sorted(set().union(*actor_sets)))
    overlap = sum(
        len(actor_sets[left] & actor_sets[right])
        for left in range(3)
        for right in range(left + 1, 3)
    )
    logical_bytes = sum(
        int(
            parquet.metadata.row_group(group)
            .column(indexes[name])
            .total_compressed_size
        )
        for group in union
        for name in columns
    )
    return records, ActualRowGroupLedgerSetV13(
        artifacts=tuple(artifacts),
        union_row_group_ordinals=union,
        unique_row_group_count=len(union),
        logical_compressed_bytes=logical_bytes,
        cross_actor_overlap_count=overlap,
        union_sha256=_canonical_sha(list(union)),
        per_actor=tuple(),
    )


def _project_w1(fixture: dict[str, object], output: Path):
    return build_immutable_w1_io_projection_v13(
        physical_config_path=fixture["config"],
        physical_config_sha256=fixture["config_sha"],
        output_root=output,
    )


def _stabilise_w3_manifest(path: Path, raw: dict[str, object]) -> str:
    child_bytes = sum(
        child.stat().st_size
        for child in path.parent.rglob("*")
        if child.is_file() and child != path
    )
    raw["materialized_output_bytes"] = 0
    for _ in range(10):
        _write_json(path, raw)
        total = child_bytes + path.stat().st_size
        if raw["materialized_output_bytes"] == total:
            return sha256_file(path)
        raw["materialized_output_bytes"] = total
    raise AssertionError("W3 root manifest size did not stabilise")


def _w3_fixture(
    root: Path,
    baseline: dict[str, object],
    *,
    ranges: list[list[list[int]]] | None = None,
    padding_bytes: int = 0,
) -> tuple[Path, str]:
    root.mkdir(parents=True)
    datasets: dict[str, object] = {}
    inventory: list[dict[str, object]] = []
    for family in CANONICAL_DATASET_KEYS:
        child = root / "datasets" / family / "manifest.json"
        _write_json(child, {"family": family, "leaves": [_digest(family)]})
        record = {
            "path": child.relative_to(root).as_posix(),
            "sha256": sha256_file(child),
            "size_bytes": child.stat().st_size,
            "merkle_root": _digest(f"merkle-{family}"),
            "rows": 512,
            "fragment_count": 1,
        }
        datasets[family] = record
        inventory.append(
            {
                "role": "dataset_manifest",
                "family": family,
                "path": record["path"],
                "sha256": record["sha256"],
                "size_bytes": record["size_bytes"],
            }
        )
    if padding_bytes:
        padding = root / "models" / "site_models.bin"
        padding.parent.mkdir(parents=True)
        padding.write_bytes(b"x" * padding_bytes)
        inventory.append(
            {
                "role": "site_models",
                "family": "site_models",
                "path": padding.relative_to(root).as_posix(),
                "sha256": sha256_file(padding),
                "size_bytes": padding.stat().st_size,
            }
        )
    query_ids = baseline["query_ids"]
    query_folds = baseline["query_folds"]
    partitions = ((0, 170), (170, 340), (340, 512))
    if ranges is None:
        ranges = [[[0, 128]], [[128, 256]], [[256, 512]]]
    specs = [
        _spec(
            ordinal=ordinal,
            query_ids=query_ids[start:stop],
            query_folds=query_folds[start:stop],
            query_start=start,
            ranges=ranges[ordinal],
        )
        for ordinal, (start, stop) in enumerate(partitions)
    ]
    ledger_records, _actual_reads = _actual_read_ledgers(
        root, baseline, specs, inventory
    )
    wrapper = root / "manifest.json"
    raw: dict[str, object] = {
        "schema_version": "sireto-retrieval-v13-sharded-run-1",
        "run_kind": "PHYSICAL_BENCHMARK_W3_V13",
        "query_count": 512,
        "worker_count": 3,
        "fold0_opened": False,
        "fold1_opened": False,
        "positive_injection": False,
        "source_candidate": {
            "path": str(baseline["source"]),
            "sha256": baseline["source_sha"],
            "projection_sha256": baseline["projection_sha"],
        },
        "source_specs": specs,
        "source_row_group_ledgers": ledger_records,
        "canonical_datasets": datasets,
        "artifact_inventory": inventory,
        "materialized_output_bytes": 0,
        "logical_digests": _logical_digests(),
        "no_merged_files": True,
    }
    return wrapper, _stabilise_w3_manifest(wrapper, raw)


def test_w1_projection_is_filtered_reproducible_and_content_addressed(tmp_path: Path) -> None:
    fixture = _baseline_fixture(tmp_path / "fixture")
    first = _project_w1(fixture, tmp_path / "w1_projection")
    second = _project_w1(fixture, tmp_path / "w1_projection")

    assert first == second
    assert first.unique_extent_count == 4
    assert first.logical_compressed_bytes == fixture["logical_bytes"]
    assert first.manifest_path.parent.name == first.build_id[:16]
    raw = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert raw["projection"]["columns"] == ["query_id", "fold", "candidate"]
    assert "ground_truth_state" not in raw["projection"]["columns"]
    assert raw["source_bytes_rehashed_before_w3"] is False
    assert raw["full_source_hash_required_post_w3"] is True
    loaded, _, rows, _ = load_immutable_w1_io_projection_v13(
        first.manifest_path,
        expected_manifest_sha256=first.manifest_sha256,
    )
    assert loaded == first
    assert len(rows) == 4


def test_pre_w3_projection_forbids_full_source_rehash(tmp_path: Path) -> None:
    fixture = _baseline_fixture(tmp_path / "fixture")
    with pytest.raises(LogicalIOInvariantError, match="would warm candidate pages"):
        build_immutable_w1_io_projection_v13(
            physical_config_path=fixture["config"],
            physical_config_sha256=fixture["config_sha"],
            output_root=tmp_path / "w1_projection",
            verify_source_hash=True,
        )


def test_pre_w3_projection_reload_does_not_hash_source_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixture = _baseline_fixture(tmp_path / "fixture")
    w1 = _project_w1(fixture, tmp_path / "w1_projection")
    import xgb_matcher.retrieval_fold_fused_logical_io_v13 as logical_io

    original = logical_io.sha256_file

    def guarded(path: Path) -> str:
        if Path(path).resolve() == Path(fixture["source"]).resolve():
            raise AssertionError("pre-W3 loader attempted a full source hash")
        return original(path)

    monkeypatch.setattr(logical_io, "sha256_file", guarded)
    loaded, _, rows, _ = load_immutable_w1_io_projection_v13(
        w1.manifest_path,
        expected_manifest_sha256=w1.manifest_sha256,
    )
    assert loaded == w1
    assert len(rows) == 4


def test_post_w3_comparison_requires_full_source_hash(tmp_path: Path) -> None:
    fixture = _baseline_fixture(tmp_path / "fixture")
    w1 = _project_w1(fixture, tmp_path / "w1_projection")
    w3_path, w3_sha = _w3_fixture(tmp_path / "w3", fixture)
    source = Path(fixture["source"])
    corrupted = bytearray(source.read_bytes())
    corrupted[0] ^= 0x01
    source.write_bytes(corrupted)

    with pytest.raises(LogicalIOInvariantError, match="source candidate bytes drifted"):
        compare_w3_to_immutable_w1_io_v13(
            w1_projection_manifest_path=w1.manifest_path,
            w1_projection_manifest_sha256=w1.manifest_sha256,
            w3_run_manifest_path=w3_path,
            w3_run_manifest_sha256=w3_sha,
            output_root=tmp_path / "io_gate",
        )


def test_fresh_w3_exact_extents_and_exhaustive_inventory_pass(tmp_path: Path) -> None:
    fixture = _baseline_fixture(tmp_path / "fixture")
    w1 = _project_w1(fixture, tmp_path / "w1_projection")
    w3_path, w3_sha = _w3_fixture(tmp_path / "w3", fixture)
    result = compare_w3_to_immutable_w1_io_v13(
        w1_projection_manifest_path=w1.manifest_path,
        w1_projection_manifest_sha256=w1.manifest_sha256,
        w3_run_manifest_path=w3_path,
        w3_run_manifest_sha256=w3_sha,
        output_root=tmp_path / "io_gate",
    )

    assert result.status == "PASS_V13_IO"
    assert result.gate_pass is True
    assert result.logical_io_ratio == 1.0
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["w3"]["per_actor_unique_extent_count_sum"] == 4
    assert manifest["w3"]["union_unique_extent_count"] == 4
    assert manifest["w3"]["inter_actor_duplicate_extent_references"] == 0
    assert (
        manifest["w3"]["declared_materialized_output_bytes"]
        == manifest["w3"]["actual_tree_materialized_bytes"]
        == manifest["w3"]["materialized_output_bytes"]
    )
    assert manifest["artifact"]["rows"] == 8
    assert manifest["process_or_machine_io_used_for_decision"] is False
    assert load_logical_io_gate_v13(
        result.manifest_path,
        expected_manifest_sha256=result.manifest_sha256,
    ) == result


def test_runtime_wrapper_is_consumed_without_schema_adaptation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from xgb_matcher import retrieval_fold_fused_fragment_runtime_v13 as runtime
    from xgb_matcher import retrieval_fold_fused_physical_benchmark_v13 as physical
    from xgb_matcher.retrieval_fold_fused_sharded_oof_v12 import ShardSpecV12

    assert tuple(REQUIRED_LOGICAL_DIGEST_KEYS) == tuple(
        physical.REQUIRED_LOGICAL_BRIDGE_DIGESTS
    )
    fixture = _baseline_fixture(tmp_path / "fixture")
    w1 = _project_w1(fixture, tmp_path / "w1_projection")
    output = tmp_path / "w3"
    output.mkdir()
    datasets: dict[str, SimpleNamespace] = {}
    for family in CANONICAL_DATASET_KEYS:
        path = output / "canonical_datasets" / family / "manifest.json"
        _write_json(path, {"family": family, "merkle": _digest(family)})
        datasets[family] = SimpleNamespace(
            path=path,
            sha256=sha256_file(path),
            size_bytes=path.stat().st_size,
            merkle_root=_digest(f"merkle-{family}"),
            rows=512,
            fragment_count=1,
        )
    build_path = output / "build_fragment_dataset" / "manifest.json"
    replay_path = output / "compact_replay_dataset" / "manifest.json"
    evidence_path = output / "evidence" / "trial_evidence.json"
    for path, role in (
        (build_path, "build"),
        (replay_path, "replay"),
        (evidence_path, "evidence"),
    ):
        _write_json(path, {"role": role})
    canonical_path = output / "canonical_datasets" / "manifest.json"
    _write_json(canonical_path, {"role": "canonical"})
    query_ids = fixture["query_ids"]
    query_folds = fixture["query_folds"]
    partitions = ((0, 170), (170, 340), (340, 512))
    source_ranges = (((0, 128),), ((128, 256),), ((256, 512),))
    specs = tuple(
        ShardSpecV12.from_manifest(
            _spec(
                ordinal=ordinal,
                query_ids=query_ids[start:stop],
                query_folds=query_folds[start:stop],
                query_start=start,
                ranges=[list(value) for value in source_ranges[ordinal]],
            )
        )
        for ordinal, (start, stop) in enumerate(partitions)
    )
    prepared = SimpleNamespace(
        runtime_payload={"paths": {"workspace_root": str(tmp_path)}},
        scene_cache=SimpleNamespace(
            source=fixture["source"],
            manifest={"source_candidate_sha256": fixture["source_sha"]},
        ),
    )
    monkeypatch.setattr(
        runtime,
        "load_physical_config_v13",
        lambda _path: {
            "cache_neutral_io_gate": {
                "source_extent": {
                    "production_projection": {"sha256": fixture["projection_sha"]}
                }
            }
        },
    )
    _records, actual_reads = _actual_read_ledgers(
        output,
        fixture,
        [value.as_manifest() for value in specs],
        [],
    )
    wrapper = runtime._write_w3_wrapper_v13(
        output=output,
        prepared=prepared,
        specs=specs,
        logical=_logical_digests(),
        canonical=SimpleNamespace(
            canonical_datasets=datasets,
            manifest_path=canonical_path,
        ),
        build=SimpleNamespace(manifest_path=build_path),
        replay=SimpleNamespace(manifest_path=replay_path),
        evidence_path=evidence_path,
        row_group_ledgers=actual_reads,
    )
    raw = json.loads(wrapper.read_text(encoding="utf-8"))
    children = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path != wrapper
    }
    assert set(raw["artifact_inventory"][0]) == {
        "role",
        "family",
        "path",
        "sha256",
        "size_bytes",
    }
    assert {record["path"] for record in raw["artifact_inventory"]} == children
    assert raw["materialized_output_bytes"] == sum(
        path.stat().st_size for path in output.rglob("*") if path.is_file()
    )
    result = compare_w3_to_immutable_w1_io_v13(
        w1_projection_manifest_path=w1.manifest_path,
        w1_projection_manifest_sha256=w1.manifest_sha256,
        w3_run_manifest_path=wrapper,
        w3_run_manifest_sha256=sha256_file(wrapper),
        output_root=tmp_path / "io_gate",
    )
    assert result.status == "PASS_V13_IO"


def test_cross_actor_row_group_overlap_is_counted_and_stops(tmp_path: Path) -> None:
    fixture = _baseline_fixture(tmp_path / "fixture")
    w1 = _project_w1(fixture, tmp_path / "w1_projection")
    w3_path, w3_sha = _w3_fixture(
        tmp_path / "w3",
        fixture,
        ranges=[[[0, 100]], [[100, 300]], [[300, 512]]],
    )
    result = compare_w3_to_immutable_w1_io_v13(
        w1_projection_manifest_path=w1.manifest_path,
        w1_projection_manifest_sha256=w1.manifest_sha256,
        w3_run_manifest_path=w3_path,
        w3_run_manifest_sha256=w3_sha,
        output_root=tmp_path / "io_gate",
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert result.status == "STOP_V13_IO"
    assert result.gate_pass is False
    assert manifest["checks"]["source_range_overlap_rows_zero"] is True
    assert manifest["checks"]["source_extents_pairwise_disjoint"] is False
    assert manifest["w3"]["per_actor_unique_extent_count_sum"] == 6
    assert manifest["w3"]["union_unique_extent_count"] == 4
    assert manifest["w3"]["inter_actor_duplicate_extent_references"] == 2


def test_cross_actor_source_range_overlap_is_a_sealed_stop(tmp_path: Path) -> None:
    fixture = _baseline_fixture(tmp_path / "fixture")
    w1 = _project_w1(fixture, tmp_path / "w1_projection")
    w3_path, w3_sha = _w3_fixture(
        tmp_path / "w3",
        fixture,
        ranges=[[[0, 160]], [[128, 320]], [[320, 512]]],
    )
    result = compare_w3_to_immutable_w1_io_v13(
        w1_projection_manifest_path=w1.manifest_path,
        w1_projection_manifest_sha256=w1.manifest_sha256,
        w3_run_manifest_path=w3_path,
        w3_run_manifest_sha256=w3_sha,
        output_root=tmp_path / "io_gate",
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert result.status == "STOP_V13_IO"
    assert manifest["checks"]["source_range_union_identical"] is True
    assert manifest["checks"]["source_range_overlap_rows_zero"] is False


def test_source_union_difference_is_a_sealed_stop(tmp_path: Path) -> None:
    fixture = _baseline_fixture(tmp_path / "fixture")
    w1 = _project_w1(fixture, tmp_path / "w1_projection")
    w3_path, w3_sha = _w3_fixture(
        tmp_path / "w3",
        fixture,
        ranges=[[[0, 128]], [[128, 256]], [[256, 384]]],
    )
    result = compare_w3_to_immutable_w1_io_v13(
        w1_projection_manifest_path=w1.manifest_path,
        w1_projection_manifest_sha256=w1.manifest_sha256,
        w3_run_manifest_path=w3_path,
        w3_run_manifest_sha256=w3_sha,
        output_root=tmp_path / "io_gate",
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert result.status == "STOP_V13_IO"
    assert manifest["checks"]["source_range_union_identical"] is False
    assert manifest["checks"]["source_extent_union_identical"] is False


def test_materialized_tree_ratio_above_115_percent_stops(tmp_path: Path) -> None:
    fixture = _baseline_fixture(
        tmp_path / "fixture", baseline_materialized=100_000
    )
    w1 = _project_w1(fixture, tmp_path / "w1_projection")
    w3_path, w3_sha = _w3_fixture(
        tmp_path / "w3", fixture, padding_bytes=120_000
    )
    result = compare_w3_to_immutable_w1_io_v13(
        w1_projection_manifest_path=w1.manifest_path,
        w1_projection_manifest_sha256=w1.manifest_sha256,
        w3_run_manifest_path=w3_path,
        w3_run_manifest_sha256=w3_sha,
        output_root=tmp_path / "io_gate",
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert result.status == "STOP_V13_IO"
    assert result.materialized_io_ratio > 1.15
    assert manifest["checks"]["materialized_ratio_within_maximum"] is False


def test_undeclared_child_file_is_rejected(tmp_path: Path) -> None:
    fixture = _baseline_fixture(tmp_path / "fixture")
    w1 = _project_w1(fixture, tmp_path / "w1_projection")
    w3_path, _ = _w3_fixture(tmp_path / "w3", fixture)
    (w3_path.parent / "undeclared.log").write_text("noise", encoding="utf-8")
    w3_sha = sha256_file(w3_path)

    with pytest.raises(LogicalIOInvariantError, match="not exhaustive"):
        compare_w3_to_immutable_w1_io_v13(
            w1_projection_manifest_path=w1.manifest_path,
            w1_projection_manifest_sha256=w1.manifest_sha256,
            w3_run_manifest_path=w3_path,
            w3_run_manifest_sha256=w3_sha,
            output_root=tmp_path / "io_gate",
        )


def test_two_inventory_paths_to_one_hardlink_inode_are_rejected(tmp_path: Path) -> None:
    fixture = _baseline_fixture(tmp_path / "fixture")
    w1 = _project_w1(fixture, tmp_path / "w1_projection")
    w3_path, _ = _w3_fixture(tmp_path / "w3", fixture)
    raw = json.loads(w3_path.read_text(encoding="utf-8"))
    original = w3_path.parent / raw["artifact_inventory"][0]["path"]
    duplicate = w3_path.parent / "models" / "hardlinked.bin"
    duplicate.parent.mkdir(parents=True)
    os.link(original, duplicate)
    raw["artifact_inventory"].append(
        {
            "role": "site_model",
            "family": "site_models",
            "path": duplicate.relative_to(w3_path.parent).as_posix(),
            "sha256": sha256_file(duplicate),
            "size_bytes": duplicate.stat().st_size,
        }
    )
    w3_sha = _stabilise_w3_manifest(w3_path, raw)

    with pytest.raises(LogicalIOInvariantError, match="forbidden hardlink"):
        compare_w3_to_immutable_w1_io_v13(
            w1_projection_manifest_path=w1.manifest_path,
            w1_projection_manifest_sha256=w1.manifest_sha256,
            w3_run_manifest_path=w3_path,
            w3_run_manifest_sha256=w3_sha,
            output_root=tmp_path / "io_gate",
        )


def test_single_inventory_path_hardlinked_outside_tree_is_rejected(tmp_path: Path) -> None:
    fixture = _baseline_fixture(tmp_path / "fixture")
    w1 = _project_w1(fixture, tmp_path / "w1_projection")
    w3_path, w3_sha = _w3_fixture(tmp_path / "w3", fixture)
    raw = json.loads(w3_path.read_text(encoding="utf-8"))
    child = w3_path.parent / raw["artifact_inventory"][0]["path"]
    os.link(child, tmp_path / "outside-hardlink.bin")

    with pytest.raises(LogicalIOInvariantError, match="forbidden hardlink"):
        compare_w3_to_immutable_w1_io_v13(
            w1_projection_manifest_path=w1.manifest_path,
            w1_projection_manifest_sha256=w1.manifest_sha256,
            w3_run_manifest_path=w3_path,
            w3_run_manifest_sha256=w3_sha,
            output_root=tmp_path / "io_gate",
        )


def test_projection_sha_drift_is_rejected_before_extent_comparison(tmp_path: Path) -> None:
    fixture = _baseline_fixture(tmp_path / "fixture")
    w1 = _project_w1(fixture, tmp_path / "w1_projection")
    w3_path, _ = _w3_fixture(tmp_path / "w3", fixture)
    raw = json.loads(w3_path.read_text(encoding="utf-8"))
    raw["source_candidate"]["projection_sha256"] = _digest("wrong")
    w3_sha = _stabilise_w3_manifest(w3_path, raw)

    with pytest.raises(LogicalIOInvariantError, match="wrapper contract"):
        compare_w3_to_immutable_w1_io_v13(
            w1_projection_manifest_path=w1.manifest_path,
            w1_projection_manifest_sha256=w1.manifest_sha256,
            w3_run_manifest_path=w3_path,
            w3_run_manifest_sha256=w3_sha,
            output_root=tmp_path / "io_gate",
        )


def test_w3_logical_digest_schema_drift_is_rejected(tmp_path: Path) -> None:
    fixture = _baseline_fixture(tmp_path / "fixture")
    w1 = _project_w1(fixture, tmp_path / "w1_projection")
    w3_path, _ = _w3_fixture(tmp_path / "w3", fixture)
    raw = json.loads(w3_path.read_text(encoding="utf-8"))
    raw["logical_digests"].pop("outcomes")
    w3_sha = _stabilise_w3_manifest(w3_path, raw)

    with pytest.raises(LogicalIOInvariantError, match="wrapper contract"):
        compare_w3_to_immutable_w1_io_v13(
            w1_projection_manifest_path=w1.manifest_path,
            w1_projection_manifest_sha256=w1.manifest_sha256,
            w3_run_manifest_path=w3_path,
            w3_run_manifest_sha256=w3_sha,
            output_root=tmp_path / "io_gate",
        )


def test_w3_prefit_bridge_logical_digest_is_mandatory(tmp_path: Path) -> None:
    fixture = _baseline_fixture(tmp_path / "fixture")
    w1 = _project_w1(fixture, tmp_path / "w1_projection")
    w3_path, _ = _w3_fixture(tmp_path / "w3", fixture)
    raw = json.loads(w3_path.read_text(encoding="utf-8"))
    raw["logical_digests"].pop("prefit_bridge_logical")
    w3_sha = _stabilise_w3_manifest(w3_path, raw)

    with pytest.raises(LogicalIOInvariantError, match="wrapper contract"):
        compare_w3_to_immutable_w1_io_v13(
            w1_projection_manifest_path=w1.manifest_path,
            w1_projection_manifest_sha256=w1.manifest_sha256,
            w3_run_manifest_path=w3_path,
            w3_run_manifest_sha256=w3_sha,
            output_root=tmp_path / "io_gate",
        )


def test_w3_site_feature_values_digest_is_mandatory(tmp_path: Path) -> None:
    fixture = _baseline_fixture(tmp_path / "fixture")
    w1 = _project_w1(fixture, tmp_path / "w1_projection")
    w3_path, _ = _w3_fixture(tmp_path / "w3", fixture)
    raw = json.loads(w3_path.read_text(encoding="utf-8"))
    raw["logical_digests"].pop("site_feature_values")
    w3_sha = _stabilise_w3_manifest(w3_path, raw)

    with pytest.raises(LogicalIOInvariantError, match="wrapper contract"):
        compare_w3_to_immutable_w1_io_v13(
            w1_projection_manifest_path=w1.manifest_path,
            w1_projection_manifest_sha256=w1.manifest_sha256,
            w3_run_manifest_path=w3_path,
            w3_run_manifest_sha256=w3_sha,
            output_root=tmp_path / "io_gate",
        )


def test_w3_prerank500_full_payload_digest_is_mandatory(tmp_path: Path) -> None:
    fixture = _baseline_fixture(tmp_path / "fixture")
    w1 = _project_w1(fixture, tmp_path / "w1_projection")
    w3_path, _ = _w3_fixture(tmp_path / "w3", fixture)
    raw = json.loads(w3_path.read_text(encoding="utf-8"))
    raw["logical_digests"].pop("prerank500_full_payload")
    w3_sha = _stabilise_w3_manifest(w3_path, raw)

    with pytest.raises(LogicalIOInvariantError, match="wrapper contract"):
        compare_w3_to_immutable_w1_io_v13(
            w1_projection_manifest_path=w1.manifest_path,
            w1_projection_manifest_sha256=w1.manifest_sha256,
            w3_run_manifest_path=w3_path,
            w3_run_manifest_sha256=w3_sha,
            output_root=tmp_path / "io_gate",
        )


@pytest.mark.parametrize(
    "external_key",
    (
        "attempt_marker_sha256",
        "sharded_implementation_sha256",
        "runtime_environment_sha256",
    ),
)
def test_external_authority_fields_are_forbidden_in_w3_wrapper(
    tmp_path: Path, external_key: str
) -> None:
    fixture = _baseline_fixture(tmp_path / "fixture")
    w1 = _project_w1(fixture, tmp_path / "w1_projection")
    w3_path, _ = _w3_fixture(tmp_path / "w3", fixture)
    raw = json.loads(w3_path.read_text(encoding="utf-8"))
    raw[external_key] = _digest(external_key)
    w3_sha = _stabilise_w3_manifest(w3_path, raw)

    with pytest.raises(LogicalIOInvariantError, match="wrapper contract"):
        compare_w3_to_immutable_w1_io_v13(
            w1_projection_manifest_path=w1.manifest_path,
            w1_projection_manifest_sha256=w1.manifest_sha256,
            w3_run_manifest_path=w3_path,
            w3_run_manifest_sha256=w3_sha,
            output_root=tmp_path / "io_gate",
        )


def test_w3_source_ordinal_sequence_drift_is_rejected(tmp_path: Path) -> None:
    fixture = _baseline_fixture(tmp_path / "fixture")
    w1 = _project_w1(fixture, tmp_path / "w1_projection")
    w3_path, _ = _w3_fixture(tmp_path / "w3", fixture)
    raw = json.loads(w3_path.read_text(encoding="utf-8"))
    raw["source_specs"][0]["source_ordinals"][-1] += 1_000
    w3_sha = _stabilise_w3_manifest(w3_path, raw)

    with pytest.raises(LogicalIOInvariantError, match="source ordinal sequence"):
        compare_w3_to_immutable_w1_io_v13(
            w1_projection_manifest_path=w1.manifest_path,
            w1_projection_manifest_sha256=w1.manifest_sha256,
            w3_run_manifest_path=w3_path,
            w3_run_manifest_sha256=w3_sha,
            output_root=tmp_path / "io_gate",
        )


def test_w1_projection_loader_rejects_ledger_tampering(tmp_path: Path) -> None:
    fixture = _baseline_fixture(tmp_path / "fixture")
    w1 = _project_w1(fixture, tmp_path / "w1_projection")
    ledger = w1.manifest_path.parent / "logical_extents.parquet"
    ledger.write_bytes(ledger.read_bytes() + b"tamper")

    with pytest.raises(LogicalIOInvariantError, match="ledger bytes drifted"):
        load_immutable_w1_io_projection_v13(
            w1.manifest_path,
            expected_manifest_sha256=w1.manifest_sha256,
        )


def test_final_gate_loader_rejects_extent_ledger_tampering(tmp_path: Path) -> None:
    fixture = _baseline_fixture(tmp_path / "fixture")
    w1 = _project_w1(fixture, tmp_path / "w1_projection")
    w3_path, w3_sha = _w3_fixture(tmp_path / "w3", fixture)
    result = compare_w3_to_immutable_w1_io_v13(
        w1_projection_manifest_path=w1.manifest_path,
        w1_projection_manifest_sha256=w1.manifest_sha256,
        w3_run_manifest_path=w3_path,
        w3_run_manifest_sha256=w3_sha,
        output_root=tmp_path / "io_gate",
    )
    ledger = result.manifest_path.parent / "logical_extents.parquet"
    ledger.write_bytes(ledger.read_bytes() + b"tamper")

    with pytest.raises(LogicalIOInvariantError, match="extent ledger bytes drifted"):
        load_logical_io_gate_v13(
            result.manifest_path,
            expected_manifest_sha256=result.manifest_sha256,
        )


def test_cli_exposes_separate_pre_w3_and_post_w3_phases() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "audit_retrieval_fold_fused_logical_io_v13.py"
    root_help = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    project_help = subprocess.run(
        [sys.executable, str(script), "project-w1", "--help"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    compare_help = subprocess.run(
        [sys.executable, str(script), "compare-w3", "--help"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "project-w1" in root_help and "compare-w3" in root_help
    assert "--physical-config-sha256" in project_help
    assert "--w3-run-manifest-sha256" in compare_help


def test_cli_defaults_are_bound_to_current_preregistered_config() -> None:
    from scripts.audit_retrieval_fold_fused_logical_io_v13 import (
        DEFAULT_PHYSICAL_CONFIG,
        DEFAULT_PHYSICAL_CONFIG_SHA256,
        parser,
    )

    args = parser().parse_args(["project-w1", "--output-root", "/tmp/io-v13"])
    assert args.physical_config == DEFAULT_PHYSICAL_CONFIG
    assert args.physical_config_sha256 == DEFAULT_PHYSICAL_CONFIG_SHA256
    assert sha256_file(DEFAULT_PHYSICAL_CONFIG) == DEFAULT_PHYSICAL_CONFIG_SHA256
