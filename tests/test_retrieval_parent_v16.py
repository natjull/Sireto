from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery
from src.xgb_matcher.retrieval_parent_v16 import (
    ParentChannelSpec,
    fuse_parent_signals,
    generate_query_signals,
    load_query_selection,
    search_native_parents,
    sha256_file,
    write_generation,
)
from src.xgb_matcher.retrieval_parent_v16_evaluation import evaluate_parent_generation


class FakeBackend:
    def __init__(self, rows_by_channel):
        self.rows_by_channel = rows_by_channel

    def search_native_parent_documents(self, *, query, spec):
        assert isinstance(query, RetrievalQuery)
        return self.rows_by_channel.get(spec.name, [])

    @staticmethod
    def _first(document, key):
        return document.get(key, "")


def spec(name="name_word", weight=1.0):
    return ParentChannelSpec(
        name=name,
        backend="base",
        family="BASE",
        mode="word",
        fields=("names",),
        query_value="name",
        limit=128,
        weight=weight,
    )


def test_native_parent_search_deduplicates_sirens_and_rejects_non_sirens():
    backend = FakeBackend(
        {
            "name_word": [
                (2.0, {"siren": "123456789"}),
                (3.0, {"siren": "123456789"}),
                (9.0, {"siren": "12345678900001"}),
                (1.0, {"siren": "987654321"}),
            ]
        }
    )
    result = search_native_parents(backend, RetrievalQuery(name="ACME", insee="75056"), spec())
    assert result == (("123456789", 3.0), ("987654321", 1.0))


def test_generation_is_deterministic_and_contains_no_truth_fields():
    specs = (spec("one"), spec("two"))
    backend = FakeBackend(
        {
            "one": [(1.0, {"siren": "111111111"})],
            "two": [(2.0, {"siren": "222222222"})],
        }
    )
    rows = generate_query_signals(
        query_id="q", fold=2, query=RetrievalQuery(name="ACME", postcode="75001"),
        backends={"base": backend}, specs=specs, workers=2,
    )
    assert [row["channel"] for row in rows] == ["one", "two"]
    assert not any("truth" in key or "label" in key for row in rows for key in row)


def test_rrf_fusion_respects_channel_weights_and_cap():
    rows = [
        {"channel": "a", "channel_rank": 1, "siren": "111111111"},
        {"channel": "b", "channel_rank": 1, "siren": "222222222"},
        {"channel": "b", "channel_rank": 2, "siren": "111111111"},
    ]
    fused = fuse_parent_signals(
        rows, allowed_channels={"a", "b"}, channel_weights={"a": 1.0, "b": 2.0}, cap=1,
    )
    assert fused[0]["siren"] == "111111111"
    assert fused[0]["channel_support"] == 2


def test_query_loader_refuses_truth_and_closed_folds(tmp_path: Path):
    path = tmp_path / "queries.parquet"
    pq.write_table(pa.table({"query_id": ["q"], "fold": pa.array([1], type=pa.int8())}), path)
    with pytest.raises(ValueError, match="forbidden"):
        load_query_selection(path)
    pq.write_table(pa.table({"query_id": ["q"], "fold": pa.array([2], type=pa.int8()), "truth": ["x"]}), path)
    with pytest.raises(ValueError, match="only query_id and fold"):
        load_query_selection(path)


def test_truth_side_evaluation_compares_to_sealed_baseline(tmp_path: Path):
    cohort = tmp_path / "cohort"
    cohort.mkdir()
    query_ids = [f"q{i:04d}" for i in range(1034)]
    folds = [2 + (i % 3) for i in range(1034)]
    kinds = ["BASE_TOP1750_MISS" if i < 522 else "BASE_TOP1750_CONTROL" for i in range(1034)]
    truths = [f"{i:09d}00001" for i in range(1034)]
    queries_path = cohort / "queries.parquet"
    labels_path = cohort / "cohort_labels.parquet"
    pq.write_table(pa.table({"query_id": query_ids, "fold": pa.array(folds, type=pa.int8())}), queries_path)
    pq.write_table(
        pa.table({"query_id": query_ids, "fold": pa.array(folds, type=pa.int8()), "human_ground_truth_siret": truths, "cohort_kind": kinds}),
        labels_path,
    )
    cohort_manifest = {
        "files": {
            "queries.parquet": {"sha256": sha256_file(queries_path)},
            "cohort_labels.parquet": {"sha256": sha256_file(labels_path)},
        }
    }
    (cohort / "manifest.json").write_text(json.dumps(cohort_manifest))

    generation = tmp_path / "generation"
    signal_rows = [
        {"query_id": query_ids[i], "fold": folds[i], "backend": "base", "family": "BASE", "channel": "c", "channel_rank": 1, "siren": truths[i][:9], "score": 1.0}
        for i in range(465)
    ]
    write_generation(
        generation, signal_rows,
        {"truth_columns_opened": False, "fold0_opened": False, "fold1_opened": False},
    )
    baseline = tmp_path / "baseline.parquet"
    pq.write_table(
        pa.table({"query_id": query_ids[:103], "parent_rank": pa.array([1] * 103, type=pa.int16()), "siren": [value[:9] for value in truths[:103]]}),
        baseline,
    )
    config = {
        "channels": [{"name": "c", "weight": 1.0}],
        "ablations": [{"name": "PRIMARY", "channels": ["c"]}],
        "evaluation": {"cutoffs": [250, 512], "primary_ablation": "PRIMARY"},
        "gates": {
            "hard_kill_parent_hits_at_512_by_fold": {"2": 0, "3": 0, "4": 0},
            "confidence_parent_hits_at_512": 465,
            "confidence_parent_hits_at_512_by_fold": {"2": 0, "3": 0, "4": 0},
            "minimum_promising_absolute_gain": 100,
        },
        "authorities": {"v15_parents": {"sha256": sha256_file(baseline)}},
    }
    result = evaluate_parent_generation(
        generation_dir=generation, cohort_dir=cohort, baseline_parents_path=baseline,
        output_dir=tmp_path / "evaluation", config=config, config_sha256="cfg",
    )
    payload = json.loads(result.read_text())
    assert payload["decision"] == "PASS_PARENT_V16_FEASIBILITY"
    assert payload["primary_gain_vs_v15"] == 362
