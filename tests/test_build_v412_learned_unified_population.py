from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.build_v412_learned_unified_population import (
    SCHEMA_VERSION,
    _stable_fold,
    build,
)
from src.xgb_matcher.v9_dataset import file_sha256


def _siren_for_fold(fold: int) -> str:
    for value in range(100_000_000, 999_999_999):
        siren = str(value)
        if _stable_fold(siren, 42, 5) == fold:
            return siren
    raise AssertionError("Unable to construct a fold fixture")


def test_build_unified_population_is_input_blind_and_siren_grouped(tmp_path: Path) -> None:
    sirens = [_siren_for_fold(fold) for fold in range(5)]
    historical_rows = []
    crm_rows = []
    for query_id in range(10):
        siren = sirens[query_id % 5]
        siret = f"{siren}{query_id:05d}"
        historical_rows.append(
            {
                "query_id": str(query_id),
                "crm_record_id": f"CRM{query_id}",
                "crm_name": f"ENTITY {query_id}",
                "crm_address": f"{query_id} RUE TEST",
                "crm_city": "LYON",
                "postcode": "69000",
                "insee": "69123",
                "reference_date": None,
                "split": "train" if query_id < 7 else "dev",
                "label_kind": "MATCH_EXACT",
                "ground_truth_siret": siret,
                "historical_ground_truth_siret": siret,
            }
        )
        crm_rows.append(
            {
                "crm_name": f"ENTITY {query_id}",
                "crm_cp": "69000",
                "crm_insee": "69123",
                "crm_id": f"CRM{query_id}",
                "crm_commune": "LYON",
                "gt_siret": siret,
                "crm_adresse": f"{query_id} RUE TEST",
            }
        )

    qualification_dir = tmp_path / "qualification"
    qualification_dir.mkdir()
    benchmark_path = qualification_dir / "benchmark.parquet"
    pd.DataFrame(historical_rows).to_parquet(benchmark_path, index=False)
    (qualification_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "sireto-benchmark-v3-evidence-1",
                "build_id": "fixture-v3",
                "establishment_snapshot_sha256": "snapshot",
                "legal_unit_snapshot_sha256": "legal-unit",
                "outputs": {"benchmark.parquet": file_sha256(benchmark_path)},
            }
        ),
        encoding="utf-8",
    )
    crm_path = tmp_path / "crm.csv"
    pd.DataFrame(crm_rows).to_csv(crm_path, sep=";", index=False)

    fresh_id = "fresh:ONE"
    fresh_siren = "987654321"
    fresh_siret = f"{fresh_siren}00001"
    fresh_queries_path = tmp_path / "fresh.parquet"
    pd.DataFrame(
        [
            {
                "query_id": fresh_id,
                "crm_record_id": "FRESH1",
                "crm_name": "FRESH ENTITY",
                "crm_address": "1 RUE FRAICHE",
                "crm_postcode": "75001",
                "crm_city": "PARIS",
                "crm_insee": "75056",
            },
            {
                "query_id": "fresh:CONTROL",
                "crm_record_id": "CONTROL",
                "crm_name": "CONTROL ENTITY",
                "crm_address": "2 RUE FRAICHE",
                "crm_postcode": "75001",
                "crm_city": "PARIS",
                "crm_insee": "75056",
            },
        ]
    ).to_parquet(fresh_queries_path, index=False)

    audited_path = tmp_path / "audited.csv"
    pd.DataFrame(
        [
            {
                "query_id": "0",
                "label_kind": "AMBIGUOUS",
                "ground_truth_siret": "",
                "reliability": "HIGH",
                "error_family": "COLOCATION",
                "evidence_reference": "human:0",
                "cohort": "AUDIT",
                "source_file": "fixture",
            },
            {
                "query_id": fresh_id,
                "label_kind": "MATCH_EXACT",
                "ground_truth_siret": fresh_siret,
                "reliability": "HIGH",
                "error_family": "EXACT",
                "evidence_reference": "human:fresh",
                "cohort": "AUDIT",
                "source_file": "fixture",
            },
        ]
    ).to_csv(audited_path, index=False)
    audited_canonical_path = tmp_path / "audited_canonical.csv"
    pd.DataFrame(
        [
            {
                "query_id": "0",
                "label_kind": "MATCH_EXACT",
                "ground_truth_siret": historical_rows[0]["ground_truth_siret"],
                "reliability": "HIGH",
                "error_family": "OLD",
                "evidence_reference": "old:0",
                "cohort": "AUDIT",
                "source_file": "fixture",
            },
            {
                "query_id": fresh_id,
                "label_kind": "MATCH_EXACT",
                "ground_truth_siret": fresh_siret,
                "reliability": "HIGH",
                "error_family": "EXACT",
                "evidence_reference": "human:fresh",
                "cohort": "AUDIT",
                "source_file": "fixture",
            },
        ]
    ).to_csv(audited_canonical_path, index=False)

    corrected_siren = "876543210"
    corrected_siret = f"{corrected_siren}00001"
    controls_path = tmp_path / "controls.csv"
    pd.DataFrame(
        [
            {
                "query_id": "1",
                "historical_ground_truth_siret": historical_rows[1]["ground_truth_siret"],
                "corrected_ground_truth_siret": corrected_siret,
                "reliability": "HIGH",
                "evidence_reference": "human:control",
            },
            {
                "query_id": "fresh:CONTROL",
                "historical_ground_truth_siret": "11111111100001",
                "corrected_ground_truth_siret": "22222222200001",
                "reliability": "HIGH",
                "evidence_reference": "human:external-control",
            },
        ]
    ).to_csv(controls_path, index=False)

    final_sirets = [row["ground_truth_siret"] for row in historical_rows[1:]]
    final_sirets[0] = corrected_siret
    final_sirets.append(fresh_siret)
    snapshot_path = tmp_path / "snapshot.parquet"
    pd.DataFrame(
        {
            "siret": final_sirets,
            "etatAdministratifEtablissement": [
                "F" if siret == historical_rows[2]["ground_truth_siret"] else "A"
                for siret in final_sirets
            ],
        }
    ).to_parquet(snapshot_path, index=False)

    args = argparse.Namespace(
        crm_source=crm_path,
        qualification_dir=[qualification_dir],
        audited_labels=audited_path,
        audited_canonical=audited_canonical_path,
        control_corrections=controls_path,
        fresh_queries=fresh_queries_path,
        establishment_snapshot=snapshot_path,
        output_root=tmp_path / "outputs",
        seed=42,
        expected_historical_count=10,
        expected_audited_count=2,
        expected_fresh_count=1,
        expected_population_count=11,
    )
    output = build(args)
    assert build(args) == output

    labels = pd.read_parquet(output / "labels.parquet").set_index("query_id")
    queries = pd.read_parquet(output / "queries.parquet")
    folds = pd.read_parquet(output / "fold_assignments.parquet")
    external = pd.read_parquet(output / "external_regression_controls.parquet")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["qualification"]["retrieval_inputs_used"] is False
    assert manifest["development_contract"]["independent_test_available"] is False
    assert len(labels) == len(queries) == len(folds) == 11
    assert set(folds["oof_fold"]) == set(range(5))
    assert folds.groupby("siren_component_id")["oof_fold"].nunique().max() == 1
    assert not any("retrieval" in column for column in labels.columns)
    assert {"retrieval_rank", "ranker_score", "candidate_siret"}.isdisjoint(
        labels.columns
    )

    assert labels.loc["0", "label_kind"] == "AMBIGUOUS"
    assert labels.loc["0", "acceptor_weight"] == 4.0
    assert labels.loc[fresh_id, "ranker_weight"] == 4.0
    assert labels.loc["1", "ground_truth_siret"] == corrected_siret
    assert (
        labels.loc["1", "historical_ground_truth_siret"]
        == historical_rows[1]["ground_truth_siret"]
    )
    assert labels.loc["2", "ranker_weight"] == 0.5
    assert external["query_id"].tolist() == ["fresh:CONTROL"]
    assert manifest["counts"]["states_exact"] == {"A": 9, "F": 1}
