from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from src.xgb_matcher.v411_scene import (
    V411_ACCEPTOR_FEATURE_NAMES,
    V411_BINARY_FEATURE_NAMES,
    V411_EVIDENCE_BASE_FEATURE_NAMES,
    V411_MONOTONIC_CONSTRAINTS,
    V411_SCALED_FEATURE_NAMES,
    assert_v411_train_serve_parity,
    build_v411_compact_scene,
    build_v411_compact_scene_features,
    rank_v411_candidates,
    validate_v411_feature_mapping,
    validate_v411_feature_order,
    validate_v411_scene_frame,
)
from src.xgb_matcher.v49_site_function import SiteFunctionTaxonomy


TAXONOMY = SiteFunctionTaxonomy.load(
    Path("config/v4_9_site_function_taxonomy.json")
)


def _candidate(
    siret: str,
    *,
    score: float,
    retrieval_rank: int,
    **values: object,
) -> dict[str, object]:
    return {
        "candidate_siret": siret,
        "candidate_siren": siret[:9],
        "ranker_score": score,
        "retrieval_rank": retrieval_rank,
        "enseigne1": None,
        "enseigne2": None,
        "enseigne3": None,
        "denomination_usuelle": None,
        "activity_code": None,
        **values,
    }


def _query(name: str = "ENTREPRISE") -> dict[str, str]:
    return {
        "crm_name": name,
        "crm_address": "1 RUE DE PARIS",
        "crm_city": "LYON",
    }


def test_feature_contract_is_exact_unique_and_partitioned() -> None:
    assert len(V411_ACCEPTOR_FEATURE_NAMES) == 80
    assert len(set(V411_ACCEPTOR_FEATURE_NAMES)) == 80
    assert len(V411_EVIDENCE_BASE_FEATURE_NAMES) == 30
    assert len(V411_BINARY_FEATURE_NAMES) + len(V411_SCALED_FEATURE_NAMES) == 80
    assert not set(V411_BINARY_FEATURE_NAMES) & set(V411_SCALED_FEATURE_NAMES)
    assert len(V411_MONOTONIC_CONSTRAINTS) == 80
    assert set(V411_MONOTONIC_CONSTRAINTS) == {-1, 0, 1}
    assert not any(name.startswith("top2_") for name in V411_ACCEPTOR_FEATURE_NAMES)
    assert "top1_name_is_city_like_max" in V411_BINARY_FEATURE_NAMES
    assert "delta_name_is_city_like_max" in V411_BINARY_FEATURE_NAMES
    validate_v411_feature_order(V411_ACCEPTOR_FEATURE_NAMES)


def test_ranker_tie_break_is_score_then_retrieval_rank() -> None:
    candidates = pd.DataFrame(
        [
            _candidate("22222222200002", score=0.8, retrieval_rank=4),
            _candidate("33333333300003", score=0.9, retrieval_rank=3),
            _candidate("11111111100001", score=0.8, retrieval_rank=2),
            _candidate("44444444400004", score=0.8, retrieval_rank=1),
        ]
    )
    ranked = rank_v411_candidates(candidates)
    assert ranked["candidate_siret"].tolist() == [
        "33333333300003",
        "44444444400004",
        "11111111100001",
        "22222222200002",
    ]


def test_query_normalised_scene_formulas_and_top1_top2_evidence() -> None:
    candidates = pd.DataFrame(
        [
            _candidate(
                "11111111100001",
                score=10.0,
                retrieval_rank=2,
                name_jaro_max=0.9,
                postcode_match=1,
                is_crm_school=1,
            ),
            _candidate(
                "11111111100002",
                score=8.0,
                retrieval_rank=1,
                name_jaro_max=0.4,
                postcode_match=0,
                is_crm_school=1,
            ),
            _candidate(
                "22222222200003",
                score=0.0,
                retrieval_rank=3,
                name_jaro_max=0.2,
                postcode_match=1,
                is_crm_school=1,
            ),
        ]
    )
    scene = build_v411_compact_scene_features(_query(), candidates, TAXONOMY)

    assert list(scene) == V411_ACCEPTOR_FEATURE_NAMES
    assert scene["candidate_count"] == 3
    assert scene["ranker_gap_fraction"] == pytest.approx(0.2)
    assert scene["ranker_top3_gap_fraction"] == pytest.approx(0.6)
    assert scene["ranker_score_std_fraction"] == pytest.approx(
        pd.Series([10.0, 8.0, 0.0]).std(ddof=0) / 10.0
    )
    assert 0.0 < scene["ranker_score_entropy"] < 1.0
    assert scene["unique_siren_count"] == 2
    assert scene["top1_siren_candidate_count"] == 2
    assert scene["same_siren_top2"] == 1
    assert scene["siren_gap_fraction"] == 1.0
    assert scene["same_siren_best_sibling_gap_fraction"] == pytest.approx(0.2)
    assert scene["retrieval_rank_top1_recip"] == 0.5
    assert scene["retrieval_rank_gap_recip"] == -0.5
    assert scene["crm_is_school"] == 1
    assert scene["top1_name_jaro_max"] == pytest.approx(0.9)
    assert scene["delta_name_jaro_max"] == pytest.approx(0.5)
    assert scene["top1_postcode_match"] == 1
    assert scene["delta_postcode_match"] == 1
    # A historical feature absent from both candidates is imputed to zero.
    assert scene["top1_idf_name"] == 0
    assert scene["delta_idf_name"] == 0


def test_single_candidate_and_no_candidate_are_defined_without_fake_top2() -> None:
    one = pd.DataFrame(
        [
            _candidate(
                "11111111100001",
                score=-3.0,
                retrieval_rank=1,
                name_jaro_max=0.7,
                postcode_match=1,
            )
        ]
    )
    single = build_v411_compact_scene_features(_query(), one, TAXONOMY)
    assert single["ranker_gap_fraction"] == 1
    assert single["ranker_top3_gap_fraction"] == 1
    assert single["siren_gap_fraction"] == 1
    assert single["same_siren_best_sibling_gap_fraction"] == 1
    assert single["ranker_score_std_fraction"] == 0
    assert single["ranker_score_entropy"] == 0
    assert single["retrieval_rank_gap_recip"] == 1.0
    assert single["top1_name_jaro_max"] == pytest.approx(0.7)
    assert single["delta_name_jaro_max"] == pytest.approx(0.7)
    assert single["role_top1_top2_conflict"] == 0
    assert single["naf_top1_top2_division_equal"] == 0

    empty = build_v411_compact_scene_features(
        _query("MAIRIE DE LYON"),
        one.iloc[0:0],
        TAXONOMY,
    )
    assert all(empty[name] == 0 for name in V411_ACCEPTOR_FEATURE_NAMES[:13])
    assert all(
        empty[f"top1_{base}"] == 0 and empty[f"delta_{base}"] == 0
        for base in V411_EVIDENCE_BASE_FEATURE_NAMES
    )
    # CRM-only information remains observable, but cannot create a candidate.
    assert empty["role_crm_count"] == 1
    assert empty["role_top1_count"] == 0


def test_equal_scores_force_all_normalised_score_gaps_to_zero() -> None:
    candidates = pd.DataFrame(
        [
            _candidate("11111111100001", score=2.0, retrieval_rank=1),
            _candidate("11111111100002", score=2.0, retrieval_rank=2),
            _candidate("22222222200003", score=2.0, retrieval_rank=3),
        ]
    )
    scene = build_v411_compact_scene_features(_query(), candidates, TAXONOMY)
    assert scene["ranker_gap_fraction"] == 0
    assert scene["ranker_top3_gap_fraction"] == 0
    assert scene["ranker_score_std_fraction"] == 0
    assert scene["siren_gap_fraction"] == 0
    assert scene["same_siren_best_sibling_gap_fraction"] == 0
    assert scene["ranker_score_entropy"] == 1
    assert scene["naf_top1_top2_division_equal"] == 0


def test_compact_role_features_use_top1_top2_and_same_siren_constellation() -> None:
    candidates = pd.DataFrame(
        [
            _candidate(
                "11111111100001",
                score=0.9,
                retrieval_rank=1,
                enseigne1="ECOLE PRIMAIRE DES FLEURS",
                activity_code="85.20Z",
            ),
            _candidate(
                "11111111100002",
                score=0.8,
                retrieval_rank=2,
                enseigne1="MAIRIE ANNEXE",
                activity_code="84.11Z",
            ),
            _candidate(
                "22222222200003",
                score=0.1,
                retrieval_rank=3,
                enseigne1="ECOLE PRIMAIRE",
                activity_code="85.20Z",
            ),
        ]
    )
    scene = build_v411_compact_scene_features(
        _query("MAIRIE DE LYON"),
        candidates,
        TAXONOMY,
    )
    assert scene["role_crm_count"] == 1
    assert scene["role_top1_count"] == 1
    assert scene["role_crm_top1_conflict"] == 1
    assert scene["role_top1_top2_conflict"] == 1
    assert scene["same_siren_distinct_role_count"] == 2
    assert scene["same_siren_role_plurality"] == 1
    assert scene["naf_top1_top2_division_equal"] == 0


def test_prediction_metadata_uses_the_same_tie_break_as_features() -> None:
    candidates = pd.DataFrame(
        [
            _candidate("22222222200002", score=0.8, retrieval_rank=1),
            _candidate("11111111100001", score=0.8, retrieval_rank=2),
        ]
    )
    output = build_v411_compact_scene(_query(), candidates, TAXONOMY)
    assert output["predicted_siret"] == "22222222200002"
    assert output["predicted_siren"] == "222222222"
    assert output["candidate_count"] == 2


def test_integrity_guards_fail_closed() -> None:
    duplicate = pd.DataFrame(
        [
            _candidate("11111111100001", score=0.9, retrieval_rank=1),
            _candidate("11111111100001", score=0.8, retrieval_rank=2),
        ]
    )
    with pytest.raises(ValueError, match="duplicate SIRET"):
        rank_v411_candidates(duplicate)

    invalid_score = pd.DataFrame(
        [_candidate("11111111100001", score=math.inf, retrieval_rank=1)]
    )
    with pytest.raises(ValueError, match="non-finite"):
        build_v411_compact_scene_features(_query(), invalid_score, TAXONOMY)

    invalid_feature = pd.DataFrame(
        [
            _candidate(
                "11111111100001",
                score=0.9,
                retrieval_rank=1,
                addr_jaro=math.inf,
            )
        ]
    )
    with pytest.raises(ValueError, match="candidate feature addr_jaro"):
        build_v411_compact_scene_features(_query(), invalid_feature, TAXONOMY)

    missing_role_sources = pd.DataFrame(
        [
            {
                "candidate_siret": "11111111100001",
                "candidate_siren": "111111111",
                "ranker_score": 0.9,
                "retrieval_rank": 1,
            }
        ]
    )
    with pytest.raises(ValueError, match="missing candidate scene columns"):
        rank_v411_candidates(missing_role_sources)

    non_contiguous = pd.DataFrame(
        [
            _candidate("11111111100001", score=0.9, retrieval_rank=1),
            _candidate("22222222200002", score=0.8, retrieval_rank=3),
        ]
    )
    with pytest.raises(ValueError, match="unique and contiguous"):
        rank_v411_candidates(non_contiguous)

    malformed_siret = pd.DataFrame(
        [_candidate("111 111 111 00001", score=0.9, retrieval_rank=1)]
    )
    with pytest.raises(ValueError, match="invalid candidate SIRET"):
        rank_v411_candidates(malformed_siret)

    with pytest.raises(ValueError, match="feature order drift"):
        validate_v411_feature_order(V411_ACCEPTOR_FEATURE_NAMES[::-1])


def test_train_serve_parity_and_materialised_frame_validation() -> None:
    candidates = pd.DataFrame(
        [
            _candidate(
                "11111111100001",
                score=0.9,
                retrieval_rank=1,
                name_jaro_max=0.8,
            ),
            _candidate(
                "22222222200002",
                score=0.5,
                retrieval_rank=2,
                name_jaro_max=0.3,
            ),
        ]
    )
    train = build_v411_compact_scene_features(_query(), candidates, TAXONOMY)
    serve = build_v411_compact_scene_features(_query(), candidates, TAXONOMY)
    assert_v411_train_serve_parity(train, serve)
    validate_v411_feature_mapping(train)
    validate_v411_scene_frame(pd.DataFrame([{**train, "query_id": "q1"}]))

    changed = dict(serve)
    changed["candidate_count"] = 3.0
    with pytest.raises(ValueError, match="train/serve scene drift"):
        assert_v411_train_serve_parity(train, changed)

    broken = pd.DataFrame([{**train, "same_siren_top2": 0.5}])
    with pytest.raises(ValueError, match="invalid binary feature domain"):
        validate_v411_scene_frame(broken)
