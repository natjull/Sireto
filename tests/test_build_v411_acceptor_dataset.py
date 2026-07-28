from __future__ import annotations

from pathlib import Path
import hashlib
import json

import pandas as pd
import pytest

from scripts.build_v411_acceptor_dataset import (
    RANKER_PREDICTION_COLUMNS,
    RANKER_REQUIRED_INVARIANTS,
    _dev_partition,
    _join_ranker_scores,
    _validate_ranker_artifact_link,
    _validate_population,
    build_scene_frame,
)
from src.xgb_matcher.v9_dataset import file_sha256
from src.xgb_matcher.v411_scene import (
    V411_ACCEPTOR_FEATURE_NAMES,
    V411_ROLE_SOURCE_COLUMNS,
)
from src.xgb_matcher.v49_site_function import SiteFunctionTaxonomy


TAXONOMY = SiteFunctionTaxonomy.load(
    Path("config/v4_9_site_function_taxonomy.json")
)


def _population_inputs() -> tuple[pd.DataFrame, ...]:
    queries = pd.DataFrame(
        [
            {
                "query_id": "fit",
                "crm_record_id": "a",
                "crm_name": "ECOLE DES FLEURS",
                "crm_address": "1 RUE A",
                "crm_city": "LYON",
            },
            {
                "query_id": "dev",
                "crm_record_id": "b",
                "crm_name": "SOCIETE B",
                "crm_address": "2 RUE B",
                "crm_city": "PARIS",
            },
            {
                "query_id": "unresolved",
                "crm_record_id": "c",
                "crm_name": "INCONNU",
                "crm_address": "",
                "crm_city": "",
            },
        ]
    )
    audit = pd.DataFrame(
        {
            "query_id": ["fit", "dev", "unresolved"],
            "input_siret_state": ["ACTIVE", "CLOSED", "NOT_FOUND"],
            "source_segment": ["train", "dev", "dev"],
        }
    )
    labels = pd.DataFrame(
        {
            "query_id": ["fit", "dev", "unresolved"],
            "label_kind": ["MATCH_EXACT", "AMBIGUOUS", "UNRESOLVED"],
            "ground_truth_siret": ["11111111100001", None, None],
            "ground_truth_siren": ["111111111", None, None],
        }
    )
    assignments = pd.DataFrame(
        {
            "query_id": ["fit", "dev", "unresolved"],
            "siren_component_id": ["c1", "c2", "c3"],
            "split": ["fit", "dev", "dev"],
            "oof_fold": [0, 1, 2],
        }
    )
    return queries, audit, labels, assignments


def _candidate(query_id: str, siret: str, rank: int = 1) -> dict[str, object]:
    return {
        "query_id": query_id,
        "candidate_siret": siret,
        "candidate_siren": siret[:9],
        "retrieval_rank": rank,
        "is_ground_truth": query_id == "fit",
        "is_crm_school": query_id == "fit",
        **{name: None for name in V411_ROLE_SOURCE_COLUMNS},
    }


def test_dev_partition_is_component_deterministic() -> None:
    assert _dev_partition("same") == _dev_partition("same")
    assert _dev_partition("same") in {"threshold_dev", "comparison_dev"}


def test_population_validation_excludes_unresolved_from_targets_later() -> None:
    population = _validate_population(
        *_population_inputs(),
        enforce_canonical=False,
    )
    assert len(population) == 3
    assert population.loc[population["split"].eq("fit"), "dev_partition"].eq("").all()
    assert set(population.loc[population["split"].eq("dev"), "dev_partition"]).issubset(
        {"threshold_dev", "comparison_dev"}
    )


def test_population_rejects_component_shared_by_fit_and_dev() -> None:
    queries, audit, labels, assignments = _population_inputs()
    assignments.loc[assignments["query_id"].eq("dev"), "siren_component_id"] = "c1"
    with pytest.raises(ValueError, match="component crosses fit/dev"):
        _validate_population(
            queries,
            audit,
            labels,
            assignments,
            enforce_canonical=False,
        )


def test_population_rejects_component_shared_by_oof_folds() -> None:
    queries, audit, labels, assignments = _population_inputs()
    duplicate_query = queries.iloc[[0]].copy()
    duplicate_query["query_id"] = "fit-2"
    queries = pd.concat([queries, duplicate_query], ignore_index=True)
    duplicate_audit = audit.iloc[[0]].copy()
    duplicate_audit["query_id"] = "fit-2"
    audit = pd.concat([audit, duplicate_audit], ignore_index=True)
    duplicate_label = labels.iloc[[0]].copy()
    duplicate_label["query_id"] = "fit-2"
    labels = pd.concat([labels, duplicate_label], ignore_index=True)
    assignments = pd.concat(
        [
            assignments,
            pd.DataFrame(
                [
                    {
                        "query_id": "fit-2",
                        "siren_component_id": "c1",
                        "split": "fit",
                        "oof_fold": 4,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="component crosses OOF folds"):
        _validate_population(
            queries,
            audit,
            labels,
            assignments,
            enforce_canonical=False,
        )


def test_ranker_artifact_link_requires_go_and_exact_dataset(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    dataset_manifest = {"build_id": "dataset-build"}
    (dataset_dir / "manifest.json").write_text(
        json.dumps(dataset_manifest), encoding="utf-8"
    )
    identity = {"frozen": "identity"}
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    ranker_dir = tmp_path / build_id
    (ranker_dir / "ranker_c").mkdir(parents=True)
    model_path = ranker_dir / "ranker_c/full_fit.json"
    model_path.write_text("model", encoding="utf-8")
    ranker_manifest = {
        "schema_version": "sireto-v4.11-input-blind-ranker-c-development-1",
        "run_id": "V411_INPUT_BLIND_RANKER_C",
        "build_identity": identity,
        "build_id": build_id,
        "dataset_manifest_sha256": file_sha256(dataset_dir / "manifest.json"),
        "dataset_build_id": "dataset-build",
        "verdict": "GO_RANKER_C",
        "checks": {"all_frozen_gates": True},
        "invariants": dict(RANKER_REQUIRED_INVARIANTS),
        "outputs": {
            "ranker_c/full_fit.json": {"sha256": file_sha256(model_path)}
        },
    }
    _validate_ranker_artifact_link(
        dataset_dir, dataset_manifest, ranker_dir, ranker_manifest
    )
    ranker_manifest["verdict"] = "PIVOT_INPUT_BLIND_RANKER"
    with pytest.raises(ValueError, match="did not pass frozen gates"):
        _validate_ranker_artifact_link(
            dataset_dir, dataset_manifest, ranker_dir, ranker_manifest
        )
    ranker_manifest["verdict"] = "GO_RANKER_C"
    ranker_manifest["dataset_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="not trained on"):
        _validate_ranker_artifact_link(
            dataset_dir, dataset_manifest, ranker_dir, ranker_manifest
        )


def test_ranker_join_requires_exact_candidate_population_and_oof_origin(
    tmp_path: Path,
) -> None:
    population = _validate_population(
        *_population_inputs(),
        enforce_canonical=False,
    )
    candidates = pd.DataFrame(
        [
            _candidate("fit", "11111111100001"),
            _candidate("dev", "22222222200002"),
        ]
    )
    predictions = pd.DataFrame(
        [
            {
                "query_id": "fit",
                "candidate_siret": "11111111100001",
                "candidate_siren": "111111111",
                "retrieval_rank": 1,
                "is_ground_truth": True,
                "ranker_score": 2.0,
                "prediction_origin": "ranker_c_oof",
                "oof_fold": 0,
                "ranker_rank": 1,
            },
            {
                "query_id": "dev",
                "candidate_siret": "22222222200002",
                "candidate_siren": "222222222",
                "retrieval_rank": 1,
                "is_ground_truth": False,
                "ranker_score": 1.0,
                "prediction_origin": "ranker_c_dev",
                "oof_fold": 1,
                "ranker_rank": 1,
            },
        ],
        columns=RANKER_PREDICTION_COLUMNS,
    )
    path = tmp_path / "predictions.parquet"
    predictions.to_parquet(path, index=False)
    joined = _join_ranker_scores(candidates, path, population)
    assert joined["ranker_score"].tolist() == [2.0, 1.0]
    broken = predictions.copy()
    broken.loc[0, "prediction_origin"] = "ranker_c_dev"
    broken.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="not ranker-C OOF"):
        _join_ranker_scores(candidates, path, population)


def test_scene_dataset_has_one_row_per_query_and_nullable_unresolved_target() -> None:
    population = _validate_population(
        *_population_inputs(),
        enforce_canonical=False,
    )
    candidates = pd.DataFrame(
        [
            {
                **_candidate("fit", "11111111100001"),
                "ranker_score": 2.0,
                "prediction_origin": "ranker_c_oof",
                "oof_fold": 0,
            },
            {
                **_candidate("dev", "22222222200002"),
                "ranker_score": 1.0,
                "prediction_origin": "ranker_c_dev",
                "oof_fold": 1,
            },
        ]
    )
    scenes = build_scene_frame(population, candidates, TAXONOMY)
    assert len(scenes) == 3
    assert set(V411_ACCEPTOR_FEATURE_NAMES).issubset(scenes)
    by_id = scenes.set_index("query_id")
    assert by_id.loc["fit", "acceptor_target"] == 1
    assert by_id.loc["dev", "acceptor_target"] == 0
    assert pd.isna(by_id.loc["unresolved", "acceptor_target"])
    assert by_id.loc["unresolved", "predicted_siret"] is None
    assert by_id.loc["unresolved", "candidate_count"] == 0
