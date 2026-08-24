from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from xgb_matcher.retrieval_fold_fused_oof_v12 import sha256_file
from xgb_matcher.retrieval_fold_fused_row_group_ledger_v13 import (
    ActualRowGroupLedgerSinkV13,
    RowGroupLedgerInvariantErrorV13,
    actor_row_group_ledger_record_v13,
    validate_actual_row_group_ledger_set_v13,
)
from xgb_matcher.retrieval_top100_cascade_v12 import production_projection
from xgb_matcher.retrieval_v12_scene_ir import SceneIRRowGroupReadEvent


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _fixture(tmp_path: Path) -> dict[str, object]:
    source = tmp_path / "source.parquet"
    folds = [2, 2, 3, 3, 4, 4]
    pq.write_table(
        pa.table(
            {
                "query_id": [f"q{index}" for index in range(6)],
                "fold": folds,
                "candidate_name": [f"name-{index}" for index in range(6)],
                "ground_truth_siret": [f"{index:014d}" for index in range(6)],
            }
        ),
        source,
        compression="zstd",
        row_group_size=1,
    )
    columns = tuple(production_projection(pq.read_schema(source).names))
    projection_sha = _canonical_sha(list(columns))
    source_sha = sha256_file(source)
    lineage = {
        "source_candidate_sha256": source_sha,
        "source_manifest_sha256": "b" * 64,
        "scene_ir_sha256": "c" * 64,
        "scientific_config_sha256": "d" * 64,
        "v13_implementation_sha256": "e" * 64,
    }
    plan_sha = "f" * 64
    specs = []
    for actor, fold in enumerate((2, 3, 4)):
        start = actor * 2
        query_ids = [f"q{start}", f"q{start + 1}"]
        query_folds = [fold, fold]
        specs.append(
            {
                "schema_version": "sireto-retrieval-v12-shard-spec-1",
                "shard_id": f"fold{fold}",
                "shard_ordinal": actor,
                "sequence_start": start,
                "sequence_stop": start + 2,
                "query_ids": query_ids,
                "query_folds": query_folds,
                "source_ordinals": [start, start + 1],
                "source_ranges": [[start, start + 2]],
                "candidate_row_count": 2,
                "query_sequence_sha256": _canonical_sha(
                    [[query_id, fold] for query_id in query_ids]
                ),
                "fold0_opened": False,
                "fold1_opened": False,
            }
        )
    parquet = pq.ParquetFile(source)
    indexes = {
        name: ordinal for ordinal, name in enumerate(parquet.schema_arrow.names)
    }
    row_starts = list(range(6))
    compressed = [
        sum(
            int(
                parquet.metadata.row_group(group)
                .column(indexes[name])
                .total_compressed_size
            )
            for name in columns
        )
        for group in range(parquet.num_row_groups)
    ]
    return {
        "source": source,
        "columns": columns,
        "projection_sha": projection_sha,
        "lineage": lineage,
        "plan_sha": plan_sha,
        "specs": specs,
        "row_starts": row_starts,
        "compressed": compressed,
    }


def _event(fixture: dict[str, object], row_group: int) -> SceneIRRowGroupReadEvent:
    lineage = fixture["lineage"]
    assert isinstance(lineage, dict)
    columns = fixture["columns"]
    assert isinstance(columns, tuple)
    compressed = fixture["compressed"]
    assert isinstance(compressed, list)
    source = fixture["source"]
    assert isinstance(source, Path)
    return SceneIRRowGroupReadEvent(
        source_path=str(source),
        source_candidate_sha256=str(lineage["source_candidate_sha256"]),
        source_manifest_sha256=str(lineage["source_manifest_sha256"]),
        scene_ir_build_id=str(lineage["scene_ir_sha256"]),
        source_row_group_count=6,
        source_row_group=row_group,
        row_start=row_group,
        row_stop=row_group + 1,
        row_count=1,
        projected_columns=columns,
        projection_sha256=str(fixture["projection_sha"]),
        projected_compressed_bytes=int(compressed[row_group]),
    )


def _sink(
    fixture: dict[str, object], actor: int
) -> ActualRowGroupLedgerSinkV13:
    specs = fixture["specs"]
    lineage = fixture["lineage"]
    columns = fixture["columns"]
    assert isinstance(specs, list)
    assert isinstance(lineage, dict)
    assert isinstance(columns, tuple)
    return ActualRowGroupLedgerSinkV13(
        shard_spec=specs[actor],
        execution_lineage=lineage,
        sharded_execution_plan_sha256=str(fixture["plan_sha"]),
        expected_projection_sha256=str(fixture["projection_sha"]),
        expected_projection_column_count=len(columns),
    )


def _seal_all(
    fixture: dict[str, object], tmp_path: Path
) -> tuple[list[tuple[Path, str]], list[object]]:
    records: list[tuple[Path, str]] = []
    artifacts = []
    for actor in range(3):
        sink = _sink(fixture, actor)
        for row_group in range(actor * 2, actor * 2 + 2):
            sink(_event(fixture, row_group))
        artifact = sink.seal(tmp_path / f"actor{actor}" / "source_row_group_ledger")
        records.append((artifact.manifest_path, artifact.manifest_sha256))
        artifacts.append(artifact)
    return records, artifacts


def test_actual_row_group_ledgers_seal_and_reconcile_footer_exactly(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    records, artifacts = _seal_all(fixture, tmp_path)
    specs = fixture["specs"]
    source = fixture["source"]
    lineage = fixture["lineage"]
    columns = fixture["columns"]
    compressed = fixture["compressed"]
    assert isinstance(specs, list)
    assert isinstance(source, Path)
    assert isinstance(lineage, dict)
    assert isinstance(columns, tuple)
    assert isinstance(compressed, list)
    validated = validate_actual_row_group_ledger_set_v13(
        records,
        expected_specs=specs,
        source_path=source,
        expected_source_candidate_sha256=str(
            lineage["source_candidate_sha256"]
        ),
        expected_projection_columns=columns,
        expected_projection_sha256=str(fixture["projection_sha"]),
        expected_sharded_execution_plan_sha256=str(fixture["plan_sha"]),
        expected_execution_lineage=lineage,
        expected_unique_row_groups=6,
        expected_logical_compressed_bytes=sum(int(value) for value in compressed),
        verify_source_hash=True,
    )
    assert validated.union_row_group_ordinals == (0, 1, 2, 3, 4, 5)
    assert validated.unique_row_group_count == 6
    assert validated.logical_compressed_bytes == sum(compressed)
    wrapper_record = actor_row_group_ledger_record_v13(
        artifacts[0], root=tmp_path
    )
    assert wrapper_record["actor_id"] == "fold2"
    assert wrapper_record["actual_cache_miss_count"] == 2
    assert not Path(wrapper_record["manifest_path"]).is_absolute()


def test_actual_row_group_ledger_rejects_an_extra_group(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    records, _artifacts = _seal_all(fixture, tmp_path)
    extra_sink = _sink(fixture, 0)
    for group in (0, 1, 2):
        extra_sink(_event(fixture, group))
    extra = extra_sink.seal(tmp_path / "extra" / "source_row_group_ledger")
    records[0] = (extra.manifest_path, extra.manifest_sha256)
    specs = fixture["specs"]
    source = fixture["source"]
    lineage = fixture["lineage"]
    columns = fixture["columns"]
    compressed = fixture["compressed"]
    assert isinstance(specs, list)
    assert isinstance(source, Path)
    assert isinstance(lineage, dict)
    assert isinstance(columns, tuple)
    assert isinstance(compressed, list)
    with pytest.raises(RowGroupLedgerInvariantErrorV13, match="actor 0"):
        validate_actual_row_group_ledger_set_v13(
            records,
            expected_specs=specs,
            source_path=source,
            expected_source_candidate_sha256=str(
                lineage["source_candidate_sha256"]
            ),
            expected_projection_columns=columns,
            expected_projection_sha256=str(fixture["projection_sha"]),
            expected_sharded_execution_plan_sha256=str(fixture["plan_sha"]),
            expected_execution_lineage=lineage,
            expected_unique_row_groups=6,
            expected_logical_compressed_bytes=sum(compressed),
        )


def test_actual_row_group_ledger_rejects_duplicate_and_out_of_order_reads(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    duplicate = _sink(fixture, 0)
    duplicate(_event(fixture, 0))
    with pytest.raises(RowGroupLedgerInvariantErrorV13, match="order"):
        duplicate(_event(fixture, 0))

    out_of_order = _sink(fixture, 0)
    out_of_order(_event(fixture, 1))
    with pytest.raises(RowGroupLedgerInvariantErrorV13, match="order"):
        out_of_order(_event(fixture, 0))


def test_actual_row_group_ledger_rejects_projection_drift(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    sink = _sink(fixture, 0)
    event = _event(fixture, 0)
    drifted = SceneIRRowGroupReadEvent(
        **{
            **event.__dict__,
            "projected_columns": (*event.projected_columns, "unexpected"),
            "projection_sha256": _canonical_sha(
                [*event.projected_columns, "unexpected"]
            ),
        }
    )
    with pytest.raises(RowGroupLedgerInvariantErrorV13, match="projection"):
        sink(drifted)
