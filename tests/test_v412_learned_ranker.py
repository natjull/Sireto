from __future__ import annotations

import pandas as pd

from scripts.evaluate_v412_ranker_business_features import (
    RELATIONAL_FEATURES,
    _relational_features,
)
from scripts.train_v412_learned_oof_rankers import _training_targets


def test_relational_features_describe_competition_without_selecting() -> None:
    frame = pd.DataFrame(
        {
            "query_id": ["q", "q", "q"],
            "candidate_siret": ["1", "2", "3"],
            "candidate_siren": ["a", "a", "b"],
            "raw_street_number": ["1", "1", "1"],
            "raw_street_type": ["RUE", "RUE", "RUE"],
            "raw_street_name": ["TEST", "TEST", "TEST"],
            "raw_postcode": ["75001", "75001", "75001"],
            "address_id": ["x", "x", "x"],
            "name_sim_max_etab": [0.8, 1.0, 0.7],
            "name_sim_max_ul": [1.0, 1.0, 0.8],
            "addr_jaro": [1.0, 0.9, 1.0],
            "business_role_match": [0.0, 1.0, 0.0],
            "business_role_conflict": [0.0, 0.0, 1.0],
            "establishment_start_year": [2000.0, 2010.0, 2020.0],
            "source_name_score": [1.0, 1.0, 0.8],
            "source_name_address_consistency": [1.0, 0.9, 0.8],
            "operating_evidence": [0.1, 2.0, -0.9],
            "source_name_exact": [1.0, 1.0, 0.0],
            "is_employer": [0.0, 1.0, 0.0],
            "is_siege": [1.0, 0.0, 1.0],
        }
    )

    enriched = _relational_features(frame)

    assert set(RELATIONAL_FEATURES).issubset(enriched.columns)
    assert enriched.loc[1, "unique_role_match_same_siren"] == 1.0
    assert enriched.loc[0, "unique_seat_same_siren"] == 1.0
    assert enriched.loc[1, "best_operating_evidence_query"] == 1.0
    assert "predicted_siret" not in enriched.columns


def test_human_multiplier_changes_weights_not_targets() -> None:
    candidates = pd.DataFrame(
        {
            "query_id": ["human", "human", "ordinary", "ordinary"],
            "candidate_siret": ["00000000000001", "00000000000002"] * 2,
        }
    )
    labels = pd.DataFrame(
        {
            "query_id": ["human", "ordinary"],
            "label_kind": ["MATCH_EXACT", "MATCH_EXACT"],
            "ground_truth_siret": ["00000000000001", "00000000000001"],
            "historical_ground_truth_siret": [None, None],
            "label_is_human_validated": [True, False],
            "ranker_weight": [4.0, 1.0],
            "oof_fold": [0, 1],
        }
    )

    rows, _ = _training_targets(
        candidates,
        labels,
        include_weak_open_labels=False,
        human_weight_multiplier=2.0,
    )

    weights = rows.drop_duplicates("query_id").set_index("query_id")["query_weight"]
    assert weights["human"] == 8.0
    assert weights["ordinary"] == 1.0
    assert rows.groupby("query_id")["training_positive"].sum().eq(1).all()
