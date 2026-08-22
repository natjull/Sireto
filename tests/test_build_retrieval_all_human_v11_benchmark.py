import json
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

from scripts.build_retrieval_all_human_v11_benchmark import build, file_sha256


ROOT = Path(__file__).resolve().parents[1]


def _source(tmp_path: Path, *, cross_siren_operational: bool = False) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    rows = pd.DataFrame(
        {
            "query_id": ["q0", "q1", "q2", "q3", "q4", "q1-missing"],
            "crm_name": ["A", "B", "C", "D", "E", "F"],
            "crm_address": ["1 RUE A"] * 6,
            "crm_city": ["PARIS"] * 6,
            "postcode": ["75001"] * 6,
            "insee": ["75101"] * 6,
            "reference_date": [""] * 6,
            "split": ["dev", "test", "train", "train", "train", "test"],
            "oof_fold": [0, 1, 2, 3, 4, 1],
            "ground_truth_siret": [
                "11111111100001",
                "",
                None,
                "bad-current",
                "44444444400004",
                "",
            ],
            "historical_ground_truth_siret": [
                "99999999900009",
                "22222222200002",
                "33333333300003",
                "",
                None,
                "",
            ],
            "ground_truth_state": ["A", "F", "A", "", "A", ""],
            "label_kind": [
                "MATCH_EXACT",
                "UNRESOLVED",
                "AMBIGUOUS",
                "UNRESOLVED",
                "MATCH_EXACT",
                "AMBIGUOUS",
            ],
            "source_kind": ["REAL_HUMAN_CRM"] * 6,
            "data_origin": ["HISTORICAL"] * 6,
            "is_synthetic": [False] * 6,
            "acceptable_sirets_operational": [
                json.dumps(
                    [
                        "11111111100001",
                        "88888888800008" if cross_siren_operational else "11111111100019",
                    ]
                ),
                "[]",
                None,
                "[]",
                '["44444444400004"]',
                "[]",
            ],
        }
    )
    benchmark = source / "benchmark.parquet"
    rows.to_parquet(benchmark, index=False)
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "fixture",
                "build_identity": {"population_scope": "all_human"},
                "positive_injection": False,
                "outputs": {"benchmark.parquet": file_sha256(benchmark)},
            }
        ),
        encoding="utf-8",
    )
    return source


def _args(tmp_path: Path, source: Path) -> Namespace:
    return Namespace(
        source=source,
        contract=ROOT / "docs/retrieval_all_human_v11_contract.md",
        config=ROOT / "config/retrieval_all_human_v11.json",
        output_root=tmp_path / "output",
    )


def test_build_uses_current_then_historical_and_excludes_only_missing(tmp_path):
    destination = build(_args(tmp_path, _source(tmp_path)))
    development = pd.read_parquet(destination / "development.parquet")
    test_locked = pd.read_parquet(destination / "test_locked.parquet")
    missing_dev = pd.read_parquet(
        destination / "missing_human_labels_development.parquet"
    )
    missing_test = pd.read_parquet(
        destination / "missing_human_labels_test_locked.parquet"
    )
    manifest = json.loads((destination / "manifest.json").read_text())

    assert set(development["query_id"]) == {"q0", "q2", "q4"}
    assert set(development["oof_fold"]) == {0, 2, 4}
    assert set(test_locked["query_id"]) == {"q1"}
    assert development.set_index("query_id").at["q0", "human_ground_truth_siret"] == (
        "11111111100001"
    )
    assert development.set_index("query_id").at["q2", "human_ground_truth_siret"] == (
        "33333333300003"
    )
    assert development.set_index("query_id").at["q2", "human_ground_truth_source"] == (
        "HISTORICAL_FALLBACK"
    )
    assert set(missing_dev["query_id"]) == {"q3"}
    assert set(missing_test["query_id"]) == {"q1-missing"}
    assert manifest["population"] == {
        "excluded_missing_human_labels": 2,
        "human_ground_truth_source_counts": {"CURRENT": 2, "HISTORICAL_FALLBACK": 2},
        "missing_by_fold": {"0": 0, "1": 1, "2": 0, "3": 1, "4": 0},
        "source_rows": 6,
        "usable_by_fold": {"0": 1, "1": 1, "2": 1, "3": 0, "4": 1},
        "usable_human_labels": 4,
    }
    assert manifest["fold_lock"]["fold1_locked"] is True
    assert manifest["fold_lock"]["fold1_opened"] is False
    assert manifest["fold_lock"]["development_contains_fold1"] is False


def test_operational_labels_are_separate_same_siren_and_hashed(tmp_path):
    destination = build(_args(tmp_path, _source(tmp_path)))
    exact = pd.read_parquet(destination / "development.parquet")
    operational = pd.read_parquet(
        destination / "operational_labels_development.parquet"
    ).set_index("query_id")
    manifest = json.loads((destination / "manifest.json").read_text())

    assert "acceptable_sirets_operational" not in exact.columns
    assert "acceptable_sirets_operational_json" not in exact.columns
    assert json.loads(operational.at["q0", "acceptable_sirets_operational_json"]) == [
        "11111111100001",
        "11111111100019",
    ]
    assert json.loads(operational.at["q2", "acceptable_sirets_operational_json"]) == [
        "33333333300003"
    ]
    for metadata in manifest["outputs"].values():
        path = destination / metadata["path"]
        assert file_sha256(path) == metadata["sha256"]
    assert manifest["outputs"]["test_locked"]["locked"] is True
    assert manifest["outputs"]["development"]["locked"] is False
    assert build(_args(tmp_path, tmp_path / "source")) == destination


def test_cross_siren_operational_alternative_fails_closed(tmp_path):
    with pytest.raises(ValueError, match="cross SIREN"):
        build(_args(tmp_path, _source(tmp_path, cross_siren_operational=True)))


def test_source_hash_and_retrieval_output_columns_fail_closed(tmp_path):
    source = _source(tmp_path)
    frame = pd.read_parquet(source / "benchmark.parquet")
    frame["union_rank"] = 1
    frame.to_parquet(source / "benchmark.parquet", index=False)
    manifest = json.loads((source / "manifest.json").read_text())
    manifest["outputs"]["benchmark.parquet"] = file_sha256(source / "benchmark.parquet")
    (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Retrieval/model output"):
        build(_args(tmp_path, source))

    manifest["outputs"]["benchmark.parquet"] = "0" * 64
    (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        build(_args(tmp_path, source))
