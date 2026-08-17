import json
from argparse import Namespace

import pandas as pd

from scripts.prepare_crm_gt_v2_retrieval_input import build
from src.xgb_matcher.v9_dataset import file_sha256


def test_build_binds_commercial_population_contract_and_partitions(tmp_path):
    population = tmp_path / "population"
    population.mkdir()
    pd.DataFrame(
        {
            "query_id": ["q1"], "crm_name": ["ACME"], "crm_address": ["1 RUE A"],
            "crm_city": ["PARIS"], "crm_postcode": ["75001"], "crm_insee": ["75056"],
            "reference_date": [""], "split_role": ["PROSPECTIVE_DEV"], "oof_fold": [0],
        }
    ).to_parquet(population / "queries.parquet", index=False)
    pd.DataFrame(
        {
            "query_id": ["q1"], "data_origin": ["REAL_CRM_20260817"],
            "exact_metric_eligible": [True], "ground_truth_siret": ["12345678900001"],
            "ground_truth_siren": ["123456789"], "ground_truth_state": ["A"],
        }
    ).to_parquet(population / "labels.parquet", index=False)
    outputs = {
        name: file_sha256(population / name)
        for name in ("queries.parquet", "labels.parquet")
    }
    (population / "manifest.json").write_text(
        json.dumps({"audit_gate": {"status": "PASS"}, "outputs": outputs})
    )
    partitions = tmp_path / "partitions"
    partitions.mkdir()
    (partitions / "part.bin").write_bytes(b"candidate-store")
    contract = tmp_path / "contract.md"
    contract.write_text("frozen retrieval policy")

    destination = build(
        Namespace(
            population=population,
            retrieval_contract=contract,
            partitions_dir=partitions,
            output_root=tmp_path / "output",
        )
    )
    benchmark = pd.read_parquet(destination / "benchmark.parquet")
    manifest = json.loads((destination / "manifest.json").read_text())
    assert len(benchmark) == 1
    assert benchmark.loc[0, "split"] == "crm_prospective_dev"
    assert manifest["query_count"] == 1
    assert manifest["positive_injection"] is False
    assert manifest["build_identity"]["retrieval_contract_sha256"] == file_sha256(contract)
