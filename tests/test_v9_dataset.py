import json

import pandas as pd
import pytest

from src.xgb_matcher.contracts import (
    MatchDecision,
    ReviewReason,
    V9MatchResult,
)
from src.xgb_matcher.retrieval_config import RetrievalConfigV1
from src.xgb_matcher.v9_dataset import (
    V9_CANDIDATE_FEATURE_NAMES,
    V9DatasetManifest,
    assert_entity_disjoint,
    build_canonical_dataset,
    canonicalize_labels,
    stable_split,
)


def test_stable_split_is_entity_deterministic():
    assert stable_split("123456789", 42) == stable_split("123456789", 42)
    labels = pd.DataFrame(
        {
            "ground_truth_siren": ["123456789", "123456789"],
            "split": ["train", "test"],
        }
    )
    with pytest.raises(ValueError, match="leakage"):
        assert_entity_disjoint(labels)


def test_canonical_dataset_build_and_manifest_validation(tmp_path):
    queries = pd.DataFrame(
        {
            "crm_id": ["q1", "q2"],
            "crm_name": ["École Saint-Joseph", "Entreprise absente"],
            "crm_address": ["12 rue de l'Église", "1 rue inconnue"],
            "postcode": ["69001", "75001"],
            "crm_city": ["Lyon", "Paris"],
            "insee": ["69123", "75056"],
        }
    )
    labels = pd.DataFrame(
        {
            "crm_id": ["q1", "q2"],
            "label_kind": ["MATCH_EXACT", "NO_MATCH"],
            "ground_truth_siret": ["12345678900011", None],
            "validator": ["alice", "alice"],
        }
    )
    candidates = pd.DataFrame(
        {
            "crm_id": ["q1", "q1", "q2"],
            "siret_candidate": [
                "12345678900011",
                "99999999900011",
                "88888888800011",
            ],
            "name_jaro_max": [1.0, 0.2, 0.1],
        }
    )
    query_path = tmp_path / "queries.csv"
    label_path = tmp_path / "labels.csv"
    candidate_path = tmp_path / "candidates.csv"
    queries.to_csv(query_path, sep=";", index=False)
    labels.to_csv(label_path, sep=";", index=False)
    candidates.to_csv(candidate_path, sep=";", index=False)

    output = build_canonical_dataset(
        query_source_path=query_path,
        label_source_path=label_path,
        candidate_source_path=candidate_path,
        output_root=tmp_path / "v9",
        sirene_snapshot_id="sirene-2026-07",
    )

    manifest = V9DatasetManifest.load(output / "manifest.json")
    manifest.validate(retrieval_config=RetrievalConfigV1())
    assert manifest.feature_order == V9_CANDIDATE_FEATURE_NAMES
    assert manifest.row_counts == {"queries": 2, "labels": 2, "candidates": 3}
    built_labels = pd.read_parquet(output / "labels.parquet")
    built_candidates = pd.read_parquet(output / "candidates.parquet")
    assert built_labels.set_index("query_id").at["q2", "label_kind"] == "NO_MATCH"
    assert built_candidates["is_ground_truth"].sum() == 1

    payload = json.loads((output / "manifest.json").read_text())
    payload["feature_order"] = ["incompatible"]
    broken = output / "broken-manifest.json"
    broken.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="feature order"):
        V9DatasetManifest.load(broken).validate()


def test_match_result_keeps_legacy_routing_status():
    auto = V9MatchResult(
        crm_id="q1",
        decision=MatchDecision.AUTO_MATCH,
        predicted_siret="12345678900011",
        predicted_siren="123456789",
        confidence=0.999,
        review_reason=None,
        model_bundle_id="model-1",
        dataset_manifest_id="data-1",
    )
    assert auto.to_dict()["routing_status"] == "AUTO"

    review = V9MatchResult(
        crm_id="q2",
        decision=MatchDecision.REVIEW,
        predicted_siret=None,
        predicted_siren=None,
        confidence=0.2,
        review_reason=ReviewReason.NO_CANDIDATE,
        model_bundle_id="model-1",
        dataset_manifest_id="data-1",
    )
    assert review.to_dict()["routing_status"] == "REVIEW"


def test_match_exact_requires_siret():
    source = pd.DataFrame(
        {"crm_id": ["q1"], "label_kind": ["MATCH_EXACT"]}
    )
    with pytest.raises(ValueError, match="require ground_truth_siret"):
        canonicalize_labels(
            source,
            snapshot_id="snapshot",
            seed=42,
            default_source="test",
        )
