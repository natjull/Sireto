from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import pytest

from scripts.build_v48_acceptor_partitions import build_partitions


def _write_fixture(tmp_path: Path) -> dict[str, Path]:
    scenes = pd.DataFrame(
        [
            {
                "query_id": "h-fit",
                "predicted_siren": "222222222",
                "ground_truth_siren": "111111111",
                "is_exact_siret_correct": 0,
                "ranker_prediction_is_out_of_sample": True,
                "acceptor_eligible": True,
            },
            {
                "query_id": "h-dev",
                "predicted_siren": "333333333",
                "ground_truth_siren": "333333333",
                "is_exact_siret_correct": 1,
                "ranker_prediction_is_out_of_sample": True,
                "acceptor_eligible": True,
            },
        ]
    )
    assignments = pd.DataFrame(
        [
            {
                "query_id": "h-fit",
                "siren_component_id": "legacy-fit",
                "split": "fit",
                "oof_fold": 0,
            },
            {
                "query_id": "h-dev",
                "siren_component_id": "legacy-dev",
                "split": "dev",
                "oof_fold": 1,
            },
        ]
    )
    queries = pd.DataFrame(
        [
            {
                "query_id": "h-fit",
                "input_siret": "11111111100001",
                "input_siren": "111111111",
            },
            {
                "query_id": "h-dev",
                "input_siret": "33333333300001",
                "input_siren": "333333333",
            },
        ]
    )
    labels = pd.DataFrame(
        [
            {
                "query_id": "h-fit",
                "ground_truth_siret": "11111111100001",
                "ground_truth_siren": "111111111",
            },
            {
                "query_id": "h-dev",
                "ground_truth_siret": "33333333300001",
                "ground_truth_siren": "333333333",
            },
        ]
    )
    current = pd.DataFrame(
        [
            {
                "audit_case_id": "a-hard",
                "query_id": "c-hard",
                "sampling_stratum": "AUTO_NEAR_THRESHOLD",
                "input_siret": "44444444400001",
                "current_top1_siret": "55555555500001",
                "current_top1_siren": "555555555",
                "replayed_top1_siret": "55555555500001",
                "candidate_pool_sirens_json": '["999999999"]',
                "current_adjudication_label": "TOP1_WRONG",
                "current_evidence_validated": True,
                "current_training_eligible": True,
                "current_acceptor_target": 0,
            },
            {
                "audit_case_id": "a-random",
                "query_id": "c-random",
                "sampling_stratum": "RANDOM_POPULATION",
                "input_siret": "66666666600001",
                "current_top1_siret": "77777777700001",
                "current_top1_siren": "777777777",
                "replayed_top1_siret": "77777777700001",
                "candidate_pool_sirens_json": '["999999999", "555555555"]',
                "current_adjudication_label": "TOP1_CORRECT",
                "current_evidence_validated": True,
                "current_training_eligible": True,
                "current_acceptor_target": 1,
            },
        ]
    )
    feature_order = [f"feature_{index}" for index in range(80)]
    for index, feature in enumerate(feature_order):
        current[feature] = float(index)
    adjudications = pd.DataFrame(
        [
            {"query_id": "c-hard", "validated_correct_siret": ""},
            {"query_id": "c-random", "validated_correct_siret": ""},
        ]
    )
    contract = tmp_path / "contract.md"
    contract.write_text("fixture\n", encoding="utf-8")
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps({"feature_order": feature_order}),
        encoding="utf-8",
    )
    frames = {
        "current_labels": current,
        "adjudications": adjudications,
        "historical_scenes": scenes,
        "historical_assignments": assignments,
        "historical_queries": queries,
        "historical_labels": labels,
    }
    paths: dict[str, Path] = {
        "contract": contract,
        "acceptor_metadata": metadata,
    }
    for name, frame in frames.items():
        path = tmp_path / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        paths[name] = path
    return paths


def _build(tmp_path: Path) -> Path:
    paths = _write_fixture(tmp_path)
    return build_partitions(
        current_labels_path=paths["current_labels"],
        adjudications_path=paths["adjudications"],
        historical_scenes_path=paths["historical_scenes"],
        historical_assignments_path=paths["historical_assignments"],
        historical_queries_path=paths["historical_queries"],
        historical_labels_path=paths["historical_labels"],
        acceptor_metadata_path=paths["acceptor_metadata"],
        contract_path=paths["contract"],
        output_root=tmp_path / "out",
        enforce_canonical=False,
    )


def test_candidate_pool_does_not_join_hard_to_random(tmp_path: Path) -> None:
    target = _build(tmp_path)
    rows = pd.read_parquet(target / "partition_assignments.parquet")
    hard = rows.loc[rows["query_id"].eq("c-hard")].iloc[0]
    random = rows.loc[rows["query_id"].eq("c-random")].iloc[0]
    assert hard["component_id"] != random["component_id"]
    assert hard["role"] == "hard_oof"
    assert random["role"] == "random_sealed"


def test_random_targets_are_masked_and_components_sealed(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    current = pd.read_parquet(paths["current_labels"])
    current.loc[current["query_id"].eq("c-random"), "input_siret"] = "11111111100001"
    current.to_parquet(paths["current_labels"], index=False)
    target = build_partitions(
        current_labels_path=paths["current_labels"],
        adjudications_path=paths["adjudications"],
        historical_scenes_path=paths["historical_scenes"],
        historical_assignments_path=paths["historical_assignments"],
        historical_queries_path=paths["historical_queries"],
        historical_labels_path=paths["historical_labels"],
        acceptor_metadata_path=paths["acceptor_metadata"],
        contract_path=paths["contract"],
        output_root=tmp_path / "out",
        enforce_canonical=False,
    )
    rows = pd.read_parquet(target / "partition_assignments.parquet")
    random = rows.loc[rows["query_id"].eq("c-random")].iloc[0]
    historical = rows.loc[rows["query_id"].eq("h-fit")].iloc[0]
    assert random["partition"] == historical["partition"] == "random_sealed"
    assert pd.isna(random["acceptor_target"])
    assert not bool(random["label_visible"])


def test_dev_component_locks_targeted_case(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    current = pd.read_parquet(paths["current_labels"])
    current.loc[current["query_id"].eq("c-hard"), "input_siret"] = "33333333300001"
    current.to_parquet(paths["current_labels"], index=False)
    target = build_partitions(
        current_labels_path=paths["current_labels"],
        adjudications_path=paths["adjudications"],
        historical_scenes_path=paths["historical_scenes"],
        historical_assignments_path=paths["historical_assignments"],
        historical_queries_path=paths["historical_queries"],
        historical_labels_path=paths["historical_labels"],
        acceptor_metadata_path=paths["acceptor_metadata"],
        contract_path=paths["contract"],
        output_root=tmp_path / "out",
        enforce_canonical=False,
    )
    rows = pd.read_parquet(target / "partition_assignments.parquet")
    hard = rows.loc[rows["query_id"].eq("c-hard")].iloc[0]
    assert hard["partition"] == "historical_dev"
    assert hard["role"] == "hard_dev_locked"


def test_rejects_query_overlap(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    current = pd.read_parquet(paths["current_labels"])
    current.loc[0, "query_id"] = "h-fit"
    current.to_parquet(paths["current_labels"], index=False)
    with pytest.raises(ValueError, match="overlap"):
        build_partitions(
            current_labels_path=paths["current_labels"],
            adjudications_path=paths["adjudications"],
            historical_scenes_path=paths["historical_scenes"],
            historical_assignments_path=paths["historical_assignments"],
            historical_queries_path=paths["historical_queries"],
            historical_labels_path=paths["historical_labels"],
            acceptor_metadata_path=paths["acceptor_metadata"],
            contract_path=paths["contract"],
            output_root=tmp_path / "out",
            enforce_canonical=False,
        )
