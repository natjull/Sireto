from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.evaluate_v41_retrieval import (
    assert_dev_only,
    evaluate,
    join_crm_input_siret,
    select_variant,
)
from src.xgb_matcher.retrieval import CandidatePoolResult
from src.xgb_matcher.v41_retrieval import (
    InputSiretQualification,
    InputSiretState,
    V41CandidatePoolResult,
)
from src.xgb_matcher.v9_dataset import file_sha256


def _benchmark() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "query_id": "q1",
                "crm_record_id": "service-1",
                "crm_name": "ALPHA",
                "crm_address": "1 RUE TEST",
                "crm_city": "TEST",
                "postcode": "01000",
                "insee": "01001",
                "ground_truth_siret": "11111111100001",
                "split": "dev",
            },
            {
                "query_id": "q2",
                "crm_record_id": "service-2",
                "crm_name": "BETA",
                "crm_address": "2 RUE TEST",
                "crm_city": "TEST",
                "postcode": "01000",
                "insee": "01001",
                "ground_truth_siret": "22222222200002",
                "split": "dev",
            },
        ]
    )


class _FakeGlobalStore:
    def get_active_siblings(self, sirens, *, max_per_siren, **_kwargs):
        return {
            str(siren): [
                {"siret": f"{siren}00001", "etat_admin": "A"},
                {"siret": f"{siren}00002", "etat_admin": "A"},
            ][:max_per_siren]
            for siren in sirens
        }


class _FakeRetriever:
    def __init__(self, variant: str):
        self.variant = variant

    def build(self, *, crm_row, input_siret, gt_siret, **_kwargs):
        assert gt_siret is None
        truth_by_query = {
            "q1": "11111111100001",
            "q2": "22222222200002",
        }
        truth = truth_by_query[crm_row["query_id"]]
        candidate = {
            "siret": truth,
            "siren": truth[:9],
            "etat_admin": "A",
            "denomination": "DETAIL GLOBAL",
        }
        qualification = InputSiretQualification(
            raw_value=str(input_siret),
            normalized_siret=str(input_siret),
            siren=str(input_siret)[:9],
            state=InputSiretState.ACTIVE,
            candidate=candidate,
        )
        return V41CandidatePoolResult(
            candidates=[candidate],
            input_siret=qualification,
            channels={
                "input_siret_active": [truth],
                "input_siren_active_sites": [truth],
                "closed_alias_name": [],
                "closed_alias_address": [],
            },
            sparse_result=CandidatePoolResult(candidates=[candidate]),
        )


def test_refuses_non_dev_and_mixed_benchmark(tmp_path: Path) -> None:
    benchmark = _benchmark()
    benchmark.loc[1, "split"] = "test"
    with pytest.raises(ValueError, match="dev rows only"):
        assert_dev_only(
            benchmark,
            benchmark_path=tmp_path / "benchmark_dev.parquet",
            split="dev",
        )
    with pytest.raises(ValueError, match="restricted"):
        assert_dev_only(
            _benchmark(),
            benchmark_path=tmp_path / "benchmark_dev.parquet",
            split="test",
        )
    with pytest.raises(ValueError, match="path marked"):
        assert_dev_only(
            _benchmark(),
            benchmark_path=tmp_path / "holdout" / "benchmark_dev.parquet",
            split="dev",
        )


def test_join_uses_crm_record_id_and_service_id(tmp_path: Path) -> None:
    crm_path = tmp_path / "crm.csv"
    pd.DataFrame(
        {
            "SERVICE ID": ["service-1", "service-2"],
            "SIRET": ["11111111100001", "99999999900009"],
        }
    ).to_csv(crm_path, sep=";", index=False)

    joined = join_crm_input_siret(_benchmark(), crm_path)

    assert joined["input_siret"].tolist() == [
        "11111111100001",
        "99999999900009",
    ]
    assert joined["input_equals_truth"].tolist() == [True, False]


def test_join_keeps_present_service_id_with_missing_siret_as_invalid(
    tmp_path: Path,
) -> None:
    crm_path = tmp_path / "crm.csv"
    pd.DataFrame(
        {
            "SERVICE ID": ["service-1", "service-2"],
            "SIRET": ["11111111100001", None],
        }
    ).to_csv(crm_path, sep=";", index=False)

    joined = join_crm_input_siret(_benchmark(), crm_path)

    assert joined.loc[1, "crm_record_id"] == "service-2"
    assert pd.isna(joined.loc[1, "input_siret"])


def test_selection_applies_gates_and_simplicity_tie_break() -> None:
    def values(recall, p95=10.0, segment=1.0):
        return {
            "recall_at_100": {"rate": recall},
            "segments": {"multi_site=true": {"rate": segment}},
            "candidate_counts": {"max": 100, "over_100": 0},
            "closed_candidate_count": 0,
            "latency_ms": {"p95": p95},
        }

    selection = select_variant(
        {
            "A": values(0.990),
            "B": values(0.995),
            "C": values(1.000),
        }
    )
    assert selection["selected_variant"] == "B"

    failed = select_variant(
        {
            "A": values(0.980),
            "B": values(0.995, segment=0.97),
            "C": values(0.995, p95=21.0),
        }
    )
    assert failed["selected_variant"] is None
    assert failed["verdict"] == "PIVOT"


def test_dry_evaluation_writes_hashed_artifacts_without_injection(
    tmp_path: Path,
) -> None:
    benchmark_path = tmp_path / "benchmark_dev.parquet"
    crm_path = tmp_path / "crm.csv"
    partitions = tmp_path / "partitions"
    global_store = tmp_path / "global_store"
    cache = tmp_path / "ssd_cache"
    output = tmp_path / "output"
    _benchmark().to_parquet(benchmark_path, index=False)
    pd.DataFrame(
        {
            "SERVICE ID": ["service-1", "service-2"],
            "SIRET": ["11111111100001", "22222222200002"],
        }
    ).to_csv(crm_path, sep=";", index=False)
    (partitions / "manifest").mkdir(parents=True)
    pd.DataFrame({"key": ["fixture"]}).to_parquet(
        partitions / "manifest" / "fixture.parquet",
        index=False,
    )
    global_store.mkdir()
    (global_store / "manifest.json").write_text(
        json.dumps({"database_sha256": "fixture"}),
        encoding="utf-8",
    )

    evaluate(
        benchmark_path=benchmark_path,
        crm_source_path=crm_path,
        partitions_dir=partitions,
        global_store_path=global_store,
        cache_dir=cache,
        output_dir=output,
        partitioned_store=object(),
        global_store=_FakeGlobalStore(),
        retrievers={
            variant: _FakeRetriever(variant) for variant in ("A", "B", "C")
        },
    )

    raw = pd.read_parquet(output / "raw_results.parquet")
    summary = json.loads((output / "summary.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(raw) == 6
    assert raw["hit_at_100"].all()
    assert not raw["positive_injected"].any()
    assert raw["candidate_count"].max() == 1
    assert raw["closed_candidate_count"].sum() == 0
    assert summary["variants"]["A"]["recall_at_100"]["rate"] == 1.0
    assert summary["selection"]["selected_variant"] == "A"
    assert manifest["split"] == "dev"
    assert manifest["positive_injection"] is False
    assert manifest["outputs"]["raw_results.parquet"] == file_sha256(
        output / "raw_results.parquet"
    )
    assert manifest["outputs"]["summary.json"] == file_sha256(
        output / "summary.json"
    )


def test_evaluation_validates_dev_relative_hash_from_manifest(
    tmp_path: Path,
) -> None:
    dev_dir = tmp_path / "dev"
    dev_dir.mkdir()
    benchmark_path = dev_dir / "benchmark.parquet"
    crm_path = tmp_path / "crm.csv"
    manifest_path = tmp_path / "manifest.json"
    _benchmark().to_parquet(benchmark_path, index=False)
    pd.DataFrame(
        {
            "SERVICE ID": ["service-1", "service-2"],
            "SIRET": ["11111111100001", "22222222200002"],
        }
    ).to_csv(crm_path, sep=";", index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "split": "dev",
                "outputs": {"dev/benchmark.parquet": "wrong-hash"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="hash does not match"):
        evaluate(
            benchmark_path=benchmark_path,
            benchmark_manifest_path=manifest_path,
            crm_source_path=crm_path,
            partitions_dir=tmp_path / "unused-partitions",
            global_store_path=tmp_path / "unused-store",
            cache_dir=tmp_path / "cache",
            output_dir=tmp_path / "output",
            partitioned_store=object(),
            global_store=_FakeGlobalStore(),
            retrievers={
                variant: _FakeRetriever(variant)
                for variant in ("A", "B", "C")
            },
        )
