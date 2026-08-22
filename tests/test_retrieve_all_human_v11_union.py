from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.retrieve_all_human_v11_union import (
    _candidate_schema,
    attach_labels,
    combine,
    sha256_file,
)


def test_candidate_schema_does_not_freeze_empty_channel_as_null() -> None:
    first = {
        "query_id": "q1",
        "trade_name_exact_rank": None,
        "trade_name_exact_score": None,
        "signals": [],
        "channels": [],
    }
    schema = _candidate_schema(first)
    assert str(schema.field("trade_name_exact_rank").type) == "int64"
    assert str(schema.field("trade_name_exact_score").type) == "double"

    # This is the transition that crashed the national fold jobs: null-only in
    # the first batch, then a populated channel in a later batch.
    import pyarrow as pa

    pa.Table.from_pylist(
        [
            {
                "query_id": "q2",
                "trade_name_exact_rank": 3,
                "trade_name_exact_score": 0.75,
                "signals": [],
                "channels": ["trade_name_exact"],
            }
        ],
        schema=schema,
    )


def test_label_attachment_is_post_retrieval_and_does_not_inject(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.parquet"
    pd.DataFrame(
        [
            {"query_id": "q1", "union_rank": 1, "siret": "11111111100001"},
            {"query_id": "q1", "union_rank": 2, "siret": "22222222200002"},
        ]
    ).to_parquet(candidates, index=False)
    candidates.with_suffix(".parquet.manifest.json").write_text(
        json.dumps(
            {
                "candidate_parquet_sha256": sha256_file(candidates),
                "positive_injection": False,
                "human_labels_opened_by_retrieval": False,
                "folds": [2],
                "queries": 1,
            }
        ),
        encoding="utf-8",
    )
    labels = tmp_path / "labels.parquet"
    pd.DataFrame(
        [
            {
                "query_id": "q1",
                "human_ground_truth_siret": "22222222200002",
                "human_ground_truth_source": "CURRENT",
                "oof_fold": 2,
            }
        ]
    ).to_parquet(labels, index=False)
    operational = tmp_path / "operational.parquet"
    pd.DataFrame(
        [
            {
                "query_id": "q1",
                "acceptable_sirets_operational_json": '["22222222200002"]',
            }
        ]
    ).to_parquet(operational, index=False)
    output = tmp_path / "labelled.parquet"

    attach_labels(
        argparse.Namespace(
            candidates=candidates,
            labels=labels,
            operational_labels=operational,
            output=output,
        )
    )

    result = pd.read_parquet(output)
    assert len(result) == 2
    assert set(result["human_ground_truth_siret"]) == {"22222222200002"}
    manifest = json.loads(
        output.with_suffix(".parquet.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["positive_injection"] is False
    assert manifest["labels_attached_after_retrieval"] is True
    assert manifest["maximum_union_candidates"] == 2


def test_combine_requires_disjoint_folds_and_preserves_cap(tmp_path: Path) -> None:
    shards: list[Path] = []
    for fold in (2, 3):
        path = tmp_path / f"fold{fold}.parquet"
        pd.DataFrame(
            [
                {
                    "query_id": f"q{fold}",
                    "fold": fold,
                    "union_rank": rank,
                    "siret": f"{fold:09d}{rank:05d}",
                }
                for rank in (1, 2)
            ]
        ).to_parquet(path, index=False)
        manifest = {
            "schema_version": "manifest",
            "candidate_schema_version": "candidates",
            "source_input_sha256": "source",
            "config_sha256": "config",
            "official_config_sha256": "official",
            "sidecar_manifest_sha256": "sidecar",
            "positive_injection": False,
            "human_labels_opened_by_retrieval": False,
            "candidate_parquet_sha256": sha256_file(path),
            "folds": [fold],
            "queries": 1,
            "empty_queries": 0,
        }
        path.with_suffix(".parquet.manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        shards.append(path)

    output = tmp_path / "combined.parquet"
    combine(argparse.Namespace(input=shards, output=output))

    combined = pd.read_parquet(output)
    assert list(combined["query_id"].drop_duplicates()) == ["q2", "q3"]
    manifest = json.loads(
        output.with_suffix(".parquet.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["folds"] == [2, 3]
    assert manifest["queries"] == 2
    assert manifest["maximum_union_candidates"] == 2
