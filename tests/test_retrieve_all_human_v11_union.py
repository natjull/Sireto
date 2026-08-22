from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.retrieve_all_human_v11_union import attach_labels, sha256_file


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
