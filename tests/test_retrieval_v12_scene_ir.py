from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.xgb_matcher.retrieval_top100_cascade_v12 import (
    RetrievalTop100CascadeV12,
    RetrievalTop100CascadeV12Config,
    production_projection,
)
from src.xgb_matcher.retrieval_v12_scene_ir import (
    SCENE_IR_MANIFEST_VERSION,
    SceneIRCache,
    build_scene_ir_cache,
    sha256_file,
)


def _candidate_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    definitions = (
        ("q2", 2, "11111111100001", 1, "ALPHA"),
        ("q2", 2, "22222222200002", 2, "BETA"),
        ("q3", 3, "33333333300003", 1, "GAMMA"),
        ("q4", 4, "44444444400004", 1, "DELTA"),
        ("q4", 4, "55555555500005", 2, "EPSILON"),
    )
    for query_id, fold, siret, rank, name in definitions:
        rows.append(
            {
                "query_id": query_id,
                "fold": fold,
                "oof_fold": fold,
                "crm_name": name,
                "crm_address": "",
                "crm_insee": "",
                "crm_postcode": "",
                "siret": siret,
                "siren": siret[:9],
                "union_rank": rank,
                "retrieval_rank": rank,
                "sources": ["SIRENE"],
                "channels": ["legal_name_exact"],
                "signals": [
                    {
                        "source": "SIRENE",
                        "channel": "legal_name_exact",
                        "rank": rank,
                        "score": 1.0 / rank,
                        "parent_identifier": "",
                        "parent_rank": None,
                        "parent_support_count": None,
                        "site_rank": None,
                        "relation_direction": "",
                        "relation_date": "",
                    }
                ],
                "candidate_names": [name],
                "candidate_addresses": [],
                "candidate_insee": "",
                "candidate_postcode": "",
                "candidate_number": "",
                "candidate_state": "A",
                "candidate_is_siege": False,
                "temporal_valid_from": "",
                "temporal_valid_to": "",
                "legal_name_exact_rank": rank,
                "legal_name_exact_score": 1.0 / rank,
                # These fields prove that the scanner, not a downstream
                # consumer, removes human truth and qualification metadata.
                "ground_truth_siret": siret,
                "label_kind": "MATCH_EXACT",
            }
        )
    return rows


def _source(tmp_path: Path) -> tuple[Path, Path, list[dict[str, object]]]:
    rows = _candidate_rows()
    source = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist(rows), source, row_group_size=2)
    stable = {
        "schema_version": "sireto-retrieval-all-human-v11-candidates-manifest-1",
        "candidate_schema_version": "sireto-retrieval-all-human-v11-union-1",
        "source_input_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "official_config_sha256": "c" * 64,
        "sidecar_manifest_sha256": "d" * 64,
        "positive_injection": False,
        "human_labels_opened_by_retrieval": False,
    }
    source_shards: list[dict[str, str]] = []
    for fold in (2, 3, 4):
        shard_rows = [row for row in rows if int(row["fold"]) == fold]
        shard = tmp_path / f"source_fold{fold}.parquet"
        pq.write_table(pa.Table.from_pylist(shard_rows), shard, row_group_size=2)
        query_counts: dict[str, int] = {}
        for row in shard_rows:
            query_id = str(row["query_id"])
            query_counts[query_id] = query_counts.get(query_id, 0) + 1
        shard_manifest = {
            **stable,
            "run_kind": "LABEL_BLIND_RETRIEVAL",
            "candidate_parquet_sha256": sha256_file(shard),
            "source_manifest_sha256": "e" * 64,
            "folds": [fold],
            "queries": len(query_counts),
            "candidate_rows": len(shard_rows),
            "empty_queries": 0,
            "maximum_union_candidates": max(query_counts.values()),
        }
        shard_manifest_path = shard.with_suffix(shard.suffix + ".manifest.json")
        shard_manifest_path.write_text(
            json.dumps(shard_manifest, sort_keys=True), encoding="utf-8"
        )
        source_shards.append(
            {
                "path": str(shard.resolve()),
                "sha256": sha256_file(shard),
                "manifest_sha256": sha256_file(shard_manifest_path),
            }
        )
    manifest_path = tmp_path / "source.parquet.manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                **stable,
                "run_kind": "LABEL_BLIND_RETRIEVAL_COMBINED",
                "candidate_parquet_sha256": sha256_file(source),
                "candidate_rows": len(rows),
                "maximum_union_candidates": 2,
                "queries": 3,
                "empty_queries": 0,
                "folds": [2, 3, 4],
                "source_shards": source_shards,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return source, manifest_path, rows


def test_scene_ir_is_content_addressed_label_blind_and_query_complete(
    tmp_path: Path,
) -> None:
    source, source_manifest, rows = _source(tmp_path)
    output_root = tmp_path / "external-cache"
    first = build_scene_ir_cache(
        source=source,
        source_manifest=source_manifest,
        output_root=output_root,
        batch_size=2,
    )
    second = build_scene_ir_cache(
        source=source,
        source_manifest=source_manifest,
        output_root=output_root,
        batch_size=3,
    )
    # Batch size is an execution detail and does not change the cache identity.
    assert first.output_dir == second.output_dir
    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == SCENE_IR_MANIFEST_VERSION
    assert manifest["candidate_data_copied"] is False
    assert manifest["human_labels_opened"] is False
    assert manifest["folds"] == [2, 3, 4]
    assert manifest["strict_source_manifest"]["run_kind"] == (
        "LABEL_BLIND_RETRIEVAL_COMBINED"
    )
    assert len(manifest["strict_source_manifest"]["source_shards"]) == 3
    assert manifest["query_count"] == 3
    assert manifest["candidate_rows"] == 5

    cache = SceneIRCache(first.output_dir)
    scenes = list(cache.iter_scenes(batch_size=2))
    assert [(scene.query_id, scene.fold, scene.row_count) for scene in scenes] == [
        ("q2", 2, 2),
        ("q3", 3, 1),
        ("q4", 4, 2),
    ]
    for scene in scenes:
        assert "ground_truth_siret" not in scene.rows.column_names
        assert "label_kind" not in scene.rows.column_names
    expected_safe = set(production_projection(pa.Table.from_pylist(rows).column_names))
    assert set(scenes[0].rows.column_names) == expected_safe


def test_scene_ir_bridge_is_candidate_parity_with_reference_cascade(
    tmp_path: Path,
) -> None:
    source, source_manifest, rows = _source(tmp_path)
    build = build_scene_ir_cache(
        source=source,
        source_manifest=source_manifest,
        output_root=tmp_path / "external-cache",
        batch_size=2,
    )
    reference_groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        reference_groups.setdefault(str(row["query_id"]), []).append(row)
    cascade = RetrievalTop100CascadeV12(
        object(), RetrievalTop100CascadeV12Config()
    )
    for scene in SceneIRCache(build.output_dir).iter_scenes(batch_size=2):
        reference = cascade.cascade(
            reference_groups[scene.query_id], query_id=scene.query_id
        )
        replay = cascade.cascade(scene.as_mappings(), query_id=scene.query_id)
        assert replay.master_references == reference.master_references
        assert replay.hydration_shortlist == reference.hydration_shortlist
        assert replay.parent_candidates == reference.parent_candidates
        assert replay.parent_shortlist == reference.parent_shortlist


def test_scene_ir_selective_reader_returns_only_sealed_ranges_label_blind(
    tmp_path: Path,
) -> None:
    source, source_manifest, _rows = _source(tmp_path)
    build = build_scene_ir_cache(
        source=source,
        source_manifest=source_manifest,
        output_root=tmp_path / "external-cache",
        batch_size=2,
    )
    cache = SceneIRCache(build.output_dir)
    # Output remains in immutable source order, independently of request order.
    scenes = list(cache.iter_selected_scenes(["q4", "q2"]))
    assert [(scene.query_id, scene.fold, scene.row_count) for scene in scenes] == [
        ("q2", 2, 2),
        ("q4", 4, 2),
    ]
    assert all("ground_truth_siret" not in scene.rows.column_names for scene in scenes)
    assert all("label_kind" not in scene.rows.column_names for scene in scenes)
    with pytest.raises(ValueError, match="population differs"):
        list(cache.iter_selected_scenes(["missing-query"]))
    with pytest.raises(ValueError, match="truth-side"):
        list(
            cache.iter_selected_scenes(
                ["q2"], columns=["query_id", "ground_truth_siret"]
            )
        )


def test_scene_ir_selective_observer_reports_only_real_row_group_cache_misses(
    tmp_path: Path,
) -> None:
    source, source_manifest, rows = _source(tmp_path)
    build = build_scene_ir_cache(
        source=source,
        source_manifest=source_manifest,
        output_root=tmp_path / "external-cache",
        batch_size=2,
    )
    events = []
    scenes = list(
        SceneIRCache(build.output_dir).iter_selected_scenes(
            ["q3", "q4"], row_group_observer=events.append
        )
    )
    assert [scene.query_id for scene in scenes] == ["q3", "q4"]
    # q3 and the first half of q4 share row group 1.  It is decoded once;
    # q4 then opens row group 2.  The observer is not a range reconstruction.
    assert [event.source_row_group for event in events] == [1, 2]
    assert [(event.row_start, event.row_stop, event.row_count) for event in events] == [
        (2, 4, 2),
        (4, 5, 1),
    ]
    columns = tuple(production_projection(pa.Table.from_pylist(rows).column_names))
    projection_sha = hashlib.sha256(
        json.dumps(
            list(columns),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert all(event.projected_columns == columns for event in events)
    assert all(event.projection_sha256 == projection_sha for event in events)
    assert all(event.projected_compressed_bytes > 0 for event in events)


def test_scene_ir_refuses_forbidden_projection_and_workspace_output(
    tmp_path: Path,
) -> None:
    source, source_manifest, _rows = _source(tmp_path)
    build = build_scene_ir_cache(
        source=source,
        source_manifest=source_manifest,
        output_root=tmp_path / "external-cache",
    )
    cache = SceneIRCache(build.output_dir)
    with pytest.raises(ValueError, match="truth-side"):
        list(cache.iter_scenes(columns=["query_id", "ground_truth_siret"]))
    workspace_output = Path(__file__).resolve().parents[1] / ".scene-ir-forbidden"
    with pytest.raises(ValueError, match="outside the workspace"):
        build_scene_ir_cache(
            source=source,
            source_manifest=source_manifest,
            output_root=workspace_output,
        )


def test_scene_ir_refuses_fold1_even_when_manifest_claims_development_folds(
    tmp_path: Path,
) -> None:
    source, source_manifest, rows = _source(tmp_path)
    rows[0]["fold"] = 1
    rows[0]["oof_fold"] = 1
    pq.write_table(pa.Table.from_pylist(rows), source, row_group_size=2)
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    manifest["candidate_parquet_sha256"] = sha256_file(source)
    source_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="fold 0/1"):
        build_scene_ir_cache(
            source=source,
            source_manifest=source_manifest,
            output_root=tmp_path / "external-cache",
            batch_size=2,
        )


def test_scene_ir_rejects_minimal_forged_v11_manifest(tmp_path: Path) -> None:
    source, source_manifest, rows = _source(tmp_path)
    source_manifest.write_text(
        json.dumps(
            {
                "schema_version": "sireto-retrieval-all-human-v11-candidates-manifest-1",
                "candidate_schema_version": "sireto-retrieval-all-human-v11-union-1",
                "candidate_parquet_sha256": sha256_file(source),
                "candidate_rows": len(rows),
                "maximum_union_candidates": 2,
                "queries": 3,
                "empty_queries": 0,
                "folds": [2, 3, 4],
                "positive_injection": False,
                "human_labels_opened_by_retrieval": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="official label-blind"):
        build_scene_ir_cache(
            source=source,
            source_manifest=source_manifest,
            output_root=tmp_path / "external-cache",
        )
