from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.freeze_v9_closed_benchmark import (
    assign_legacy_v7_splits,
    load_closed_queries,
    split_audit,
)


def _source_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "crm_name": [f"CRM {index}" for index in range(8)],
            "crm_cp": ["01000"] * 8,
            "crm_insee": ["01001"] * 8,
            "crm_id": [f"id-{index}" for index in range(8)],
            "crm_commune": ["TEST"] * 8,
            "gt_siret": [
                "10000000100001",
                "10000000100002",
                "20000000200001",
                "30000000300001",
                "40000000400001",
                "50000000500001",
                "60000000600001",
                "70000000700001",
            ],
            "crm_adresse": ["1 RUE TEST"] * 8,
        }
    )


def test_closed_benchmark_split_is_siren_disjoint_and_deterministic(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.csv"
    _source_rows().to_csv(source, sep=";", index=False)
    loaded = load_closed_queries(source)

    first = assign_legacy_v7_splits(loaded, seed=42)
    second = assign_legacy_v7_splits(loaded, seed=42)
    audit = split_audit(first, None)

    assert first["split"].tolist() == second["split"].tolist()
    assert audit["siren_overlaps"] == {
        "train_dev": 0,
        "train_test": 0,
        "dev_test": 0,
    }
    same_siren = first[first["ground_truth_siren"].eq("100000001")]
    assert same_siren["split"].nunique() == 1


def test_closed_benchmark_rejects_invalid_or_duplicate_siret(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.csv"
    rows = _source_rows()
    rows.loc[1, "gt_siret"] = rows.loc[0, "gt_siret"]
    rows.to_csv(source, sep=";", index=False)

    try:
        load_closed_queries(source)
    except ValueError as error:
        assert "unique ground-truth SIRET" in str(error)
    else:
        raise AssertionError("Duplicate exact SIRET was accepted")
