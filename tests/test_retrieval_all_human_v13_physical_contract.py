from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from xgb_matcher.retrieval_fold_fused_fragment_dataset_v13 import merkle_root_v13
from xgb_matcher.retrieval_v12_scene_ir import production_projection


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "config/retrieval_all_human_v13_physical_execution.json"
ADDENDUM_PATH = (
    ROOT / "docs/retrieval_all_human_v13_physical_execution_addendum.md"
)
GIB = 1024**3


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plan() -> dict[str, Any]:
    return _load(PLAN_PATH)


def test_v13_is_a_preregistered_physical_only_revision() -> None:
    plan = _plan()
    assert plan["schema_version"] == "sireto-retrieval-v13-physical-execution-1"
    assert plan["status"] == "PREREGISTERED_V13_PHYSICAL_BENCHMARK_NOT_RUN"
    assert plan["revision_scope"] == "PHYSICAL_ONLY"
    assert plan["logical_recipe_changes"] is False
    assert plan["quality_metric_tuning_allowed"] is False
    assert plan["national_oof_allowed"] is False
    assert plan["required_next_gate"] == (
        "SEALED_V13_W3_ONLY_PHYSICAL_BENCHMARK_512"
    )
    assert plan["decision_policy"]["before_benchmark"] == "NO_READY"
    assert plan["decision_policy"]["benchmark_pass_without_separate_authorization"] == (
        "NO_READY"
    )


def test_repository_authorities_and_v13_addendum_are_content_pinned() -> None:
    plan = _plan()
    for record in plan["authorities"].values():
        path = ROOT / record["path"]
        assert path.is_file()
        assert _sha256(path) == record["sha256"]

    text = ADDENDUM_PATH.read_text(encoding="utf-8")
    assert "PRÉENREGISTRÉ — BENCHMARK V13 NON EXÉCUTÉ" in text
    assert "OOF NATIONAL\nINTERDIT" in text
    assert "bridge non circulaire v12 W1 → v13 W3" in text


def test_v12_stop_and_w1_baseline_are_immutable_and_exactly_anchored() -> None:
    stop = _plan()["immutable_v12_stop"]
    assert stop["verdict"] == "STOP_NO_PARALLEL_WORKER_AUTHORIZED"
    assert stop["reinterpretation_forbidden"] is True
    assert stop["replacement_forbidden"] is True
    assert stop["execution_authorization_published"] is False
    assert stop["benchmark_manifest_sha256"] == (
        "2269b8925a9ca5aa3b4215fc3f3ec30fccc6c59d8ace1d4165ba12daa0bb5242"
    )
    assert stop["code_commit"] == "ca4c4f3"

    w1 = stop["w1_trial"]
    assert w1["manifest_sha256"] == (
        "67d37417237d0e93b933c95759699e871d5b3a825cb4a03deeed470391a36a4"
    )
    assert w1["query_count"] == 512
    assert w1["folds"] == [2, 3, 4]
    assert w1["wall_seconds"] == 1918.9878759580315
    assert w1["materialized_bytes"] == 464378945
    assert w1["fold0_opened"] is False
    assert w1["fold1_opened"] is False
    assert w1["positive_injection"] is False
    assert set(w1["digests"]) == {
        "heldout_spool",
        "train_matrices",
        "query_groups",
        "miniature_models",
        "prefit_bridge",
        "prerank500",
        "top100",
        "outcomes",
    }
    assert all(len(value) == 64 for value in w1["digests"].values())


def test_fold_partitions_are_disjoint_contiguous_and_exhaustive() -> None:
    partitions = _plan()["partitions"]
    assert [value["actor_id"] for value in partitions] == [
        "fold2",
        "fold3",
        "fold4",
    ]
    assert [value["fold"] for value in partitions] == [2, 3, 4]
    for previous, current in zip(partitions, partitions[1:], strict=False):
        assert previous["query_stop_exclusive"] == current[
            "query_start_inclusive"
        ]
        assert previous["source_row_group_stop_exclusive"] == current[
            "source_row_group_start_inclusive"
        ]
        assert previous["candidate_row_stop_exclusive"] == current[
            "candidate_row_start_inclusive"
        ]
    assert partitions[0]["query_start_inclusive"] == 0
    assert partitions[-1]["query_stop_exclusive"] == 23571
    assert partitions[0]["source_row_group_start_inclusive"] == 0
    assert partitions[-1]["source_row_group_stop_exclusive"] == 635
    assert partitions[0]["candidate_row_start_inclusive"] == 0
    assert partitions[-1]["candidate_row_stop_exclusive"] == 41590480


def test_three_actors_persist_per_phase_but_build_rss_is_released_before_fit() -> None:
    actors = _plan()["actor_architecture"]
    assert actors["start_method"] == "spawn"
    assert actors["maximum_concurrent_actors"] == 3
    assert actors["dynamic_task_pool"] is False
    assert actors["work_stealing"] is False
    assert actors["process_per_query"] is False
    assert actors["query_migration"] is False
    assert actors["automatic_retry"] is False

    build = actors["build_phase"]
    assert build["actor_count"] == 3
    assert build["ownership"] == (
        "ONE_PERSISTENT_ACTOR_PER_FOLD_FOR_ENTIRE_BUILD_PHASE"
    )
    assert build["scene_compilations_per_query"] == 1
    assert build["outer_contexts_per_query"] == 3
    assert build["actors_must_exit_before_first_site_fit"] is True
    assert build["actors_must_be_waited_and_rss_released"] is True

    fit = actors["site_fit_phase"]
    assert fit["parallelism"] == 1
    assert fit["outer_order"] == [2, 3, 4]
    assert fit["requires_zero_live_build_actors"] is True
    assert fit["distributed_incremental_fit"] is False
    assert fit["frozen_n_jobs_zero"] is True

    replay = actors["replay_phase"]
    assert replay["actor_count"] == 3
    assert replay["ownership"] == (
        "ONE_NEW_PERSISTENT_ACTOR_PER_FOLD_FOR_ENTIRE_REPLAY_PHASE"
    )
    assert replay["fold_to_outer_model"] == {"2": 2, "3": 3, "4": 4}
    assert replay["opens_scene_ir"] is False
    assert replay["opens_official_sidecars"] is False
    assert replay["opens_parent_models"] is False
    assert replay["full_master_scored"] is True


def test_ordered_merkle_views_replace_every_physical_merge_copy() -> None:
    protocol = _plan()["ordered_fragment_protocol"]
    assert protocol["physical_merge_files_allowed"] is False
    assert protocol["forbidden_path_prefixes"] == [
        "merged_",
        "canonical_merged_",
    ]
    assert protocol["families"] == [
        "train_outer2",
        "train_outer3",
        "train_outer4",
        "heldout_spool",
        "prerank500_compact",
        "top100_full",
        "outcomes",
    ]
    assert protocol["query_groups"] == {
        "physical_family": False,
        "required_metadata_digest_on_each_train_view": True,
        "digest_covers_ordered_query_id_fold_and_group_size": True,
    }
    inventory = protocol["artifact_inventory"]
    assert inventory["train_fragment_may_be_referenced_by_two_views"] is True
    assert inventory[
        "physical_fragment_counted_once_even_if_referenced_by_two_views"
    ] is True
    assert inventory["hardlinks_forbidden"] is True
    assert protocol["fold_order"] == [2, 3, 4]
    assert protocol["fragment_order_key"] == ["fold_order", "fragment_ordinal"]
    assert protocol["train_views"] == {
        "outer2": [3, 4],
        "outer3": [2, 4],
        "outer4": [2, 3],
    }
    assert protocol["global_output_view"] == [2, 3, 4]
    assert protocol["per_fold_metric_average_forbidden"] is True

    parquet = protocol["parquet_encoding"]
    assert parquet["schema_and_column_order_frozen"] is True
    assert parquet["query_groups_contiguous"] is True
    assert parquet["scene_ir_and_candidate_order_preserved"] is True

    merkle = protocol["merkle"]
    assert merkle["hash"] == "sha256"
    assert merkle["leaf_prefix_hex"] == "00"
    assert merkle["node_prefix_hex"] == "01"
    assert merkle["empty_dataset_allowed"] is False
    assert merkle["odd_leaf_rule"] == "DUPLICATE_LAST"
    assert merkle["ordered_leaf_inventory_required"] is True
    assert merkle["root_without_inventory_allowed"] is False
    assert merkle["leaf_fields"][-3:] == [
        "schema_sha256",
        "size_bytes",
        "file_sha256",
    ]

    records = [{"fold": 2, "ordinal": 0}, {"fold": 3, "ordinal": 1}, {"fold": 4, "ordinal": 2}]
    leaves = [
        hashlib.sha256(
            bytes.fromhex(merkle["leaf_prefix_hex"])
            + json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).digest()
        for record in records
    ]
    while len(leaves) > 1:
        if len(leaves) % 2:
            leaves.append(leaves[-1])
        leaves = [
            hashlib.sha256(
                bytes.fromhex(merkle["node_prefix_hex"])
                + leaves[index]
                + leaves[index + 1]
            ).digest()
            for index in range(0, len(leaves), 2)
        ]
    assert merkle_root_v13(records) == leaves[0].hex()


def test_prerank500_is_compact_and_replayable_while_top100_keeps_full_payload() -> None:
    policy = _plan()["payload_policy"]
    assert policy["feature_computation_changed"] is False
    assert policy["ranking_changed"] is False
    compact = policy["prerank500"]
    assert compact["maximum_rows_per_query"] == 500
    assert compact["full_payload_persisted"] is False
    assert compact["json_payload_forbidden"] is True
    assert compact["feature_values_persisted"] is False
    assert compact["feature_values_logical_digest_sealed"] is True
    assert compact["feature_values_replayed_from"] == (
        "CONTENT_ADDRESSED_HELDOUT_SPOOL_JOIN_KEY_AND_FROZEN_CONFIG"
    )
    assert compact["required_compact_fields"] == [
        "query_ordinal",
        "scene_fold",
        "query_id",
        "candidate_siret",
        "candidate_parent",
        "selected_rank",
        "site_score_float32",
        "selection_reason",
        "heldout_spool_fragment_sha256",
        "heldout_spool_row_ordinal",
        "heldout_spool_join_key_sha256",
    ]
    assert all(
        "feature_values" not in field and "payload" not in field
        for field in compact["required_compact_fields"]
    )
    assert compact["audit_rehydration_must_be_exact"] is True

    full = policy["top100"]
    assert full["maximum_rows_per_query"] == 100
    assert full["full_payload_persisted"] is True
    assert full["official_evidence_and_provenance_required"] is True


def test_cache_neutral_io_gate_uses_extents_and_materialized_bytes_only() -> None:
    gate = _plan()["cache_neutral_io_gate"]
    assert gate["decision_metric"] == (
        "LOGICAL_SOURCE_EXTENTS_AND_DECLARED_MATERIALIZED_BYTES"
    )
    source = gate["source_extent"]
    assert source["extent_key"] == [
        "source_sha256",
        "row_group",
        "projection_sha256",
    ]
    assert source["unique_once_within_actor"] is True
    assert source["pairwise_disjoint_between_w3_actors"] is True
    assert source["w3_union_must_equal_immutable_w1_projection"] is True
    assert source["projection_must_equal"] is True
    assert source["logical_amplification_maximum"] == 1.0
    assert source["sealed_sample_expected_unique_row_groups"] == 368
    assert source["sealed_sample_expected_logical_compressed_bytes"] == 2696076099

    materialized = gate["materialized"]
    assert materialized["baseline_w1_bytes"] == 464378945
    assert materialized["maximum_ratio_numerator"] == 115
    assert materialized["maximum_ratio_denominator"] == 100
    assert materialized["maximum_w3_bytes"] == math.floor(464378945 * 1.15)
    assert materialized["includes_logs_and_undeclared_files"] is True
    assert gate["diagnostic_metrics_can_authorize_or_reject"] is False
    assert set(gate["diagnostic_only_metrics"]) >= {
        "process_tree_read_bytes",
        "machine_read_bytes",
        "machine_write_bytes",
    }


def test_source_extent_projection_is_the_exact_frozen_116_column_projection() -> None:
    source = _plan()["cache_neutral_io_gate"]["source_extent"]
    projection = source["production_projection"]
    columns = projection["columns"]
    assert projection["source_schema_column_count"] == 121
    assert projection["projected_column_count"] == 116 == len(columns)
    assert len(columns) == len(set(columns))
    assert tuple(columns) == production_projection(columns)
    assert projection["excluded_source_columns"] == [
        "ground_truth_state",
        "label_kind",
        "source_kind",
        "unseen_siren",
        "new_site_known_siren",
    ]
    encoded = json.dumps(
        columns,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    assert hashlib.sha256(encoded).hexdigest() == projection["sha256"]
    assert projection["sha256"] == (
        "c47aa0d64d96ad756df1a0b333328df69c13071d55861f85ba5b808e6813e9e8"
    )
    assert projection["drift_policy"] == "STOP_V13_IO_BEFORE_W3"

    source_path = Path(_plan()["input_lineage"]["source_candidate_path"])
    if source_path.is_file():
        source_names = pq.ParquetFile(source_path).schema_arrow.names
        assert len(source_names) == projection["source_schema_column_count"]
        assert production_projection(source_names) == tuple(columns)


def test_non_circular_bridge_is_against_pinned_v12_w1_not_self_parity() -> None:
    bridge = _plan()["immutable_w1_projection"]
    assert bridge["role"] == "NON_CIRCULAR_V12_W1_TO_V13_W3_BRIDGE"
    assert bridge["generated_from_pinned_w1_in_memory_before_w3"] is True
    assert bridge["persisted_unpinned_projection_allowed"] is False
    assert bridge["hash"] == "sha256"
    assert bridge["float32_comparison"] == "BITWISE"
    assert bridge["byte_or_bit_exact_required"] is True
    assert bridge["new_format_self_parity_alone_is_sufficient"] is False
    assert bridge["physical_parquet_file_sha_equality_to_v12_required"] is False
    assert bridge["projection_mismatch_decision"] == "STOP_V13_PARITY"
    assert set(bridge["ordered_projections"]) >= {
        "heldout_spool",
        "fit_row_ids_groups_features_labels_and_weights",
        "miniature_site_models",
        "prerank500_compact_columns",
        "top100_full_payload_and_reasons",
        "outcomes_and_global_aggregates",
    }


def test_benchmark_is_one_fresh_w3_against_immutable_w1_with_frozen_gates() -> None:
    benchmark = _plan()["benchmark_512"]
    assert benchmark["status"] == "NOT_RUN"
    assert benchmark["mode"] == "ONE_FRESH_W3_VS_IMMUTABLE_V12_W1"
    assert benchmark["fresh_trials_allowed"] == [3]
    assert benchmark["fresh_trial_count"] == 1
    assert benchmark["rerun_w1_allowed"] is False
    assert benchmark["w2_allowed"] is False
    assert benchmark["cache_purge_allowed"] is False
    assert benchmark["automatic_retry_allowed"] is False
    assert benchmark["sample_query_count"] == 512
    assert benchmark["sample_folds"] == [2, 3, 4]
    assert benchmark["minimum_speedup"] == 2.2
    assert benchmark["maximum_w3_wall_seconds"] == (
        benchmark["baseline_w1_wall_seconds"] / benchmark["minimum_speedup"]
    )
    assert benchmark["maximum_w3_peak_machine_used_memory_bytes"] == 14 * GIB
    assert benchmark["required_w3_swapout_delta_bytes"] == 0
    assert benchmark["maximum_projected_national_storage_bytes"] == 20 * GIB
    assert benchmark["parity"] == (
        "IMMUTABLE_W1_PROJECTION_EQUALS_FRESH_W3_BYTE_OR_BIT_EXACT"
    )
    assert benchmark["quality_metrics_decision_blocking"] is False
    assert benchmark["pass_can_directly_launch_national_oof"] is False
    assert benchmark["separate_content_addressed_national_authorization_required"] is True
    assert len(benchmark["required_negative_tests"]) == 23


def test_scientific_gates_caps_and_models_are_unchanged_from_v12() -> None:
    plan = _plan()
    scientific = _load(ROOT / plan["authorities"]["scientific_config"]["path"])
    gates = scientific["gates"]
    frozen = plan["scientific_invariants"]
    assert frozen["master_recall_at_2000"] == gates[
        "master_exact_recall_at_2000"
    ]["minimum"]
    assert frozen["master_decision_blocking"] == gates[
        "master_exact_recall_at_2000"
    ]["decision_blocking"]
    assert frozen["prerank_recall_at_500"] == gates[
        "prerank_exact_recall_at_500"
    ]["minimum"]
    product = gates["product_exact_recall_at_100"]
    assert frozen["exact_recall_at_100"] == product["minimum"]
    assert frozen["exact_recall_at_100_wilson95_lower"] == product[
        "wilson95_lower_minimum"
    ]
    assert frozen["minimum_each_outer_fold_exact_recall_at_100"] == product[
        "minimum_each_outer_fold"
    ]
    ceilings = gates["candidate_ceilings"]
    assert frozen["maximum_master_candidates"] == ceilings["master_references"]
    assert frozen["maximum_fit_candidates"] == scientific["admission"][
        "fit_candidate_sampling"
    ]["maximum_candidates_per_query"]
    assert frozen["maximum_prerank_candidates"] == ceilings[
        "prerank_and_hydrated_sirets"
    ]
    assert frozen["maximum_product_candidates"] == ceilings["published_sirets"]
    latency = gates["warm_end_to_end_latency_ms"]
    assert frozen["warm_latency_p95_seconds"] * 1000 == latency["p95_maximum"]
    assert frozen["warm_latency_p99_seconds"] * 1000 == latency["p99_maximum"]
    assert frozen["site_feature_count"] == 319
    for key in (
        "candidate_order_changed",
        "features_changed",
        "models_changed",
        "model_hyperparameters_changed",
        "labels_changed",
        "sampling_changed",
        "positive_injection",
    ):
        assert frozen[key] is False
