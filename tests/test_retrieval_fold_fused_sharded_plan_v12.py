from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib
import json

import pyarrow.parquet as pq
import pytest


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "config/retrieval_all_human_v12_sharded_execution_plan.json"
GIB = 1024**3


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plan() -> dict[str, Any]:
    return _load(PLAN_PATH)


def test_plan_is_prefit_pinned_and_cannot_authorize_national_oof() -> None:
    plan = _plan()
    assert plan["schema_version"] == (
        "sireto-retrieval-v12-fold-sharded-execution-plan-1"
    )
    assert plan["status"] == "PREREGISTERED_SHARDED_PARITY_PENDING"
    assert plan["engine_id"] == "FOLD_FUSED_CROSSFIT_COMPILER_V12_1"
    assert plan["logical_recipe_changes"] is False
    assert plan["national_oof_allowed"] is False
    assert plan["required_next_gate"] == "SEALED_W1_W3_PARITY_512_PASS"

    prefit = plan["input_lineage"]["prefit_parity"]
    assert prefit["manifest_sha256"] == (
        "0da156d88928c732d4e226dc7266159f713c8c5c9af131cfbe5cfdfe3c0a6ed0"
    )
    assert prefit["phase"] == "PREFIT_NON_CIRCULAR"
    assert prefit["status"] == "PASS"
    assert prefit["query_count"] == 512
    assert prefit["context_count"] == 1536
    assert set(prefit["checks"]) == {
        "parent_scores_bit_identical",
        "master_candidates_and_ranks_byte_identical",
        "site_features_float32_bit_identical",
        "fit_row_ids_and_order_byte_identical",
    }
    assert all(prefit["checks"].values())
    assert prefit["fold0_opened"] is False
    assert prefit["fold1_opened"] is False
    assert prefit["positive_injection"] is False


def test_repository_authorities_are_content_pinned() -> None:
    for record in _plan()["authorities"].values():
        path = ROOT / record["path"]
        assert path.is_file()
        assert _sha256(path) == record["sha256"]


def test_three_real_fold_ranges_are_exact_contiguous_and_exhaustive() -> None:
    plan = _plan()
    shards = plan["shards"]
    assert shards == [
        {
            "shard_id": "fold2",
            "fold": 2,
            "query_ordinal_start_inclusive": 0,
            "query_ordinal_stop_exclusive": 7767,
            "query_count": 7767,
            "source_row_group_start_inclusive": 0,
            "source_row_group_stop_exclusive": 210,
            "source_row_group_count": 210,
            "candidate_row_start_inclusive": 0,
            "candidate_row_stop_exclusive": 13750106,
            "candidate_row_count": 13750106,
        },
        {
            "shard_id": "fold3",
            "fold": 3,
            "query_ordinal_start_inclusive": 7767,
            "query_ordinal_stop_exclusive": 15734,
            "query_count": 7967,
            "source_row_group_start_inclusive": 210,
            "source_row_group_stop_exclusive": 423,
            "source_row_group_count": 213,
            "candidate_row_start_inclusive": 13750106,
            "candidate_row_stop_exclusive": 27705638,
            "candidate_row_count": 13955532,
        },
        {
            "shard_id": "fold4",
            "fold": 4,
            "query_ordinal_start_inclusive": 15734,
            "query_ordinal_stop_exclusive": 23571,
            "query_count": 7837,
            "source_row_group_start_inclusive": 423,
            "source_row_group_stop_exclusive": 635,
            "source_row_group_count": 212,
            "candidate_row_start_inclusive": 27705638,
            "candidate_row_stop_exclusive": 41590480,
            "candidate_row_count": 13884842,
        },
    ]
    for previous, current in zip(shards, shards[1:], strict=False):
        assert previous["query_ordinal_stop_exclusive"] == current[
            "query_ordinal_start_inclusive"
        ]
        assert previous["source_row_group_stop_exclusive"] == current[
            "source_row_group_start_inclusive"
        ]
        assert previous["candidate_row_stop_exclusive"] == current[
            "candidate_row_start_inclusive"
        ]
    assert sum(value["query_count"] for value in shards) == 23571
    assert sum(value["source_row_group_count"] for value in shards) == 635
    assert sum(value["candidate_row_count"] for value in shards) == 41590480
    assert [value["fold"] for value in shards] == [2, 3, 4]


def test_nested_oof_merge_and_replay_routes_are_exact() -> None:
    plan = _plan()
    merge = plan["train_merge"]
    assert merge["operator"] == "PHYSICAL_GROUP_CONCATENATION_NO_SORT"
    assert merge["outer2"] == [3, 4]
    assert merge["outer3"] == [2, 4]
    assert merge["outer4"] == [2, 3]
    for outer in (2, 3, 4):
        assert set(merge[f"outer{outer}"]) == {2, 3, 4} - {outer}
    assert merge["preserve_scene_ir_query_order"] is True
    assert merge["preserve_intra_query_candidate_order"] is True
    assert merge["preserve_float32_feature_bits"] is True
    assert merge["query_groups_must_remain_contiguous"] is True
    assert merge["distributed_or_incremental_fit"] is False

    replay = plan["heldout_replay"]
    assert replay["fold_to_site_model_outer"] == {"2": 2, "3": 3, "4": 4}
    assert replay["scene_ir_reopened_after_fit"] is False
    assert replay["official_sidecars_reopened_after_fit"] is False
    assert replay["parent_models_reopened_after_fit"] is False
    assert replay["validation_candidate_sampling"] is False
    assert replay["full_master_scored"] is True

    compilation = plan["query_compilation"]
    assert compilation["scene_compilations_per_query"] == 1
    assert compilation["outer_contexts_per_query"] == 3
    assert compilation["training_contexts_per_query"] == 2
    assert compilation["heldout_contexts_per_query"] == 1
    assert compilation["heldout_spooled_before_label_access"] is True
    assert compilation["full_master_statistics_before_fit_sampling"] is True
    assert compilation["fit_sampling_maximum_rows_per_training_context"] == 128
    assert compilation["positive_injection"] is False


def _selected_workers(
    policy: dict[str, Any], *, used_bytes: int, swapout_delta: int, valid: bool = True
) -> int:
    guard = policy["preferred_w3_requires_all"]
    if (
        valid
        and used_bytes <= guard["machine_used_ram_maximum_bytes"]
        and swapout_delta == guard["swapout_delta_pages_required"]
    ):
        return policy["preferred_maximum_concurrent_workers"]
    return policy["fallback_maximum_concurrent_workers"]


def test_spawn_w3_resource_guard_and_fail_closed_w2_fallback_are_frozen() -> None:
    policy = _plan()["process_policy"]
    assert policy["start_method"] == "spawn"
    assert policy["shard_count"] == 3
    assert policy["preferred_maximum_concurrent_workers"] == 3
    assert policy["fallback_maximum_concurrent_workers"] == 2
    guard = policy["preferred_w3_requires_all"]
    assert guard["machine_used_ram_metric"] == "psutil.virtual_memory().used"
    assert guard["machine_used_ram_maximum_bytes"] == 14 * GIB
    assert guard["machine_used_ram_maximum_gib"] == 14
    assert guard["swapout_metric"] == "vm_stat.Swapouts"
    assert guard["swapout_observation_seconds"] == 5
    assert guard["swapout_delta_pages_required"] == 0
    assert _selected_workers(policy, used_bytes=14 * GIB, swapout_delta=0) == 3
    assert _selected_workers(policy, used_bytes=14 * GIB + 1, swapout_delta=0) == 2
    assert _selected_workers(policy, used_bytes=10 * GIB, swapout_delta=1) == 2
    assert _selected_workers(
        policy, used_bytes=10 * GIB, swapout_delta=0, valid=False
    ) == 2
    assert policy["runtime_change_after_first_shard_open"] == "FORBIDDEN"
    assert policy["automatic_retry_with_different_worker_count"] is False
    assert policy["site_fit_parallelism"] == 1
    assert policy["site_fit_order"] == [2, 3, 4]
    assert policy["site_fit_uses_frozen_n_jobs_zero"] is True


def test_sharded_parity_is_a_blocking_physical_gate_not_a_quality_tuning_loop() -> None:
    gate = _plan()["sharded_parity_gate"]
    assert gate["status"] == "NOT_RUN"
    assert gate["sample_query_count"] == 512
    assert gate["reference_worker_count"] == 1
    assert gate["candidate_worker_count"] == 3
    assert gate["fixture_required_worker_counts"] == [1, 2, 3]
    assert gate["real_required_comparison"] == "W1_EQUALS_W3_BYTE_OR_BIT_EXACT"
    assert gate["quality_metrics_may_be_used_for_tuning"] is False
    assert gate["failure_decision"] == "STOP_SHARDED_PARITY"
    assert gate["national_oof_on_failure"] is False
    assert len(gate["required_checks"]) == 13
    assert len(gate["required_negative_tests"]) == 11
    assert "sample_site_model_sha256_identical" in gate["required_checks"]
    assert "per_shard_metric_average_rejected" in gate["required_negative_tests"]


def test_scientific_gates_and_caps_equal_the_frozen_v12_config() -> None:
    plan = _plan()
    scientific = _load(ROOT / plan["authorities"]["scientific_config"]["path"])
    source = scientific["gates"]
    frozen = plan["scientific_gates_unchanged"]
    assert frozen["master_recall_at_2000"] == source[
        "master_exact_recall_at_2000"
    ]["minimum"]
    assert frozen["master_decision_blocking"] == source[
        "master_exact_recall_at_2000"
    ]["decision_blocking"]
    assert frozen["prerank_recall_at_500"] == source[
        "prerank_exact_recall_at_500"
    ]["minimum"]
    product = source["product_exact_recall_at_100"]
    assert frozen["exact_recall_at_100"] == product["minimum"]
    assert frozen["exact_recall_at_100_wilson95_lower"] == product[
        "wilson95_lower_minimum"
    ]
    assert frozen["minimum_each_outer_fold_exact_recall_at_100"] == product[
        "minimum_each_outer_fold"
    ]
    ceilings = source["candidate_ceilings"]
    assert frozen["maximum_master_candidates"] == ceilings["master_references"]
    assert frozen["maximum_fit_candidates"] == scientific["admission"][
        "fit_candidate_sampling"
    ]["maximum_candidates_per_query"]
    assert frozen["maximum_prerank_candidates"] == ceilings[
        "prerank_and_hydrated_sirets"
    ]
    assert frozen["maximum_product_candidates"] == ceilings["published_sirets"]
    latency = source["warm_end_to_end_latency_ms"]
    assert frozen["warm_latency_p95_seconds"] * 1000 == latency["p95_maximum"]
    assert frozen["warm_latency_p99_seconds"] * 1000 == latency["p99_maximum"]
    assert plan["output_merge"]["per_shard_metric_average_forbidden"] is True
    assert plan["output_merge"]["global_gate_reduction"] == (
        "RECOMPUTE_FROM_ALL_ORDERED_QUERY_OUTCOMES"
    )


def test_real_scene_index_and_prefit_artifacts_match_plan_when_volume_is_mounted() -> None:
    plan = _plan()
    scene = plan["input_lineage"]["scene_ir"]
    index_path = Path(scene["index_path"])
    if not index_path.is_file():
        pytest.skip("content-addressed SIRETO volume is not mounted")

    scene_manifest_path = Path(scene["directory"]) / "manifest.json"
    assert _sha256(scene_manifest_path) == scene["manifest_sha256"]
    assert _sha256(index_path) == scene["index_sha256"]
    scene_manifest = _load(scene_manifest_path)
    assert scene_manifest["build_id"] == scene["build_id"]
    assert scene_manifest["query_count"] == scene["query_count"]
    assert scene_manifest["candidate_rows"] == scene["candidate_rows"]
    assert scene_manifest["fold_counts"] == {"2": 7767, "3": 7967, "4": 7837}

    index = pq.read_table(
        index_path, columns=["query_id", "fold", "row_start", "row_count"]
    )
    assert index.num_rows == 23571
    folds = index["fold"].to_pylist()
    starts = index["row_start"].to_pylist()
    counts = index["row_count"].to_pylist()
    assert set(folds) == {2, 3, 4}
    for shard in plan["shards"]:
        begin = shard["query_ordinal_start_inclusive"]
        end = shard["query_ordinal_stop_exclusive"]
        assert set(folds[begin:end]) == {shard["fold"]}
        assert starts[begin] == shard["candidate_row_start_inclusive"]
        assert starts[end - 1] + counts[end - 1] == shard[
            "candidate_row_stop_exclusive"
        ]

    source_path = Path(scene["source_candidate_path"])
    assert source_path.is_file()
    assert source_path.stat().st_size == scene["source_candidate_size_bytes"]
    parquet = pq.ParquetFile(source_path)
    assert parquet.num_row_groups == scene["source_row_groups"] == 635
    row_group_offsets = [0]
    for position in range(parquet.num_row_groups):
        row_group_offsets.append(
            row_group_offsets[-1] + parquet.metadata.row_group(position).num_rows
        )
    assert row_group_offsets[-1] == scene["candidate_rows"]
    for shard in plan["shards"]:
        assert row_group_offsets[
            shard["source_row_group_start_inclusive"]
        ] == shard["candidate_row_start_inclusive"]
        assert row_group_offsets[
            shard["source_row_group_stop_exclusive"]
        ] == shard["candidate_row_stop_exclusive"]

    small_pins = []
    labels = plan["input_lineage"]["labels"]
    small_pins.extend(
        [
            (labels["ledger_path"], labels["ledger_sha256"]),
            (labels["manifest_path"], labels["manifest_sha256"]),
            (
                labels["operational_sidecar_path"],
                labels["operational_sidecar_sha256"],
            ),
            (
                labels["runtime_query_sample_path"],
                labels["runtime_query_sample_sha256"],
            ),
        ]
    )
    official = plan["input_lineage"]["official_sidecars"]
    small_pins.extend(
        [
            (official["sidecar_manifest_path"], official["sidecar_manifest_sha256"]),
            (official["mmap_manifest_path"], official["mmap_manifest_sha256"]),
            (
                official["reader_parity_manifest_path"],
                official["reader_parity_manifest_sha256"],
            ),
        ]
    )
    parent = plan["input_lineage"]["parent_model_bundle"]
    small_pins.append((parent["manifest_path"], parent["manifest_sha256"]))
    prefit = plan["input_lineage"]["prefit_parity"]
    small_pins.append((prefit["manifest_path"], prefit["manifest_sha256"]))
    for raw_path, expected in small_pins:
        path = Path(raw_path)
        assert path.is_file()
        assert _sha256(path) == expected

    prefit_manifest = _load(Path(prefit["manifest_path"]))
    assert prefit_manifest["status"] == "PASS"
    assert prefit_manifest["phase"] == "PREFIT_NON_CIRCULAR"
    assert prefit_manifest["query_count"] == 512
    assert prefit_manifest["context_count"] == 1536
    assert prefit_manifest["lineage"]["scene_ir_sha256"] == scene["build_id"]
    assert prefit_manifest["lineage"]["scientific_config_sha256"] == plan[
        "authorities"
    ]["scientific_config"]["sha256"]
