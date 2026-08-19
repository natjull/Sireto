from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.evaluate_retrieval_union_oracle import evaluate


def test_evaluates_fold_zero_raw_union_without_calling_it_admission(tmp_path: Path):
    path = tmp_path / "union.parquet"
    rows = [
        {
            "query_id": "q1", "fold": 0, "ground_truth_siret": "12345678900011",
            "identifiable_exact": True, "siret": "12345678900011", "union_rank": 120,
            "retrieval_latency_ms": 10.0, "candidate_present": True,
            "acceptable_sirets_operational": ["12345678900011"],
            "raw_union_size": 300,
        },
        {
            "query_id": "q2", "fold": 0, "ground_truth_siret": "98765432100029",
            "identifiable_exact": False, "siret": "98765432100029", "union_rank": 1,
            "retrieval_latency_ms": 20.0, "candidate_present": True,
            "acceptable_sirets_operational": ["98765432100029"],
            "raw_union_size": 20,
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows), path)
    report = evaluate(path)
    assert report["opened_folds"] == [0]
    assert report["test_opened"] is False
    assert report["exact_oracle"]["hits"] == 1
    assert report["raw_union_rank_top100"]["hits"] == 0
    assert report["raw_union_rank_top100"]["is_trained_admission"] is False
