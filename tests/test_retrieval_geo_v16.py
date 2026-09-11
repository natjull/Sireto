from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.evaluate_retrieval_geo_v16_workbench import (
    FEATURE_NAMES,
    FORBIDDEN_MODEL_FEATURES,
    _metric_block,
)
from src.xgb_matcher.retrieval_geo_v16 import (
    BanCandidate,
    ban_declared_weak,
    ban_uncertainty_m,
    choose_ban_match,
    l93_distance_m,
    lonlat_to_l93,
    normalize_street,
    validate_label_blind_queries,
)


def test_lambert93_projection_matches_epsg_reference_points() -> None:
    assert lonlat_to_l93(3.0, 46.5) == pytest.approx((700000.0, 6600000.0), abs=1e-6)
    assert lonlat_to_l93(2.331, 48.868) == pytest.approx(
        (650924.2931549175, 6863315.813614038), abs=1e-5
    )


def test_lambert_distance_is_euclidean_metres() -> None:
    assert l93_distance_m(0, 0, 300, 400) == 500.0


def test_normalize_street_removes_leading_house_number() -> None:
    assert normalize_street("10 bis, rue de la Paix", "10 BIS") == "RUE DE LA PAIX"


def test_ban_uncertainty_exposes_weak_metadata() -> None:
    strong = ban_uncertainty_m(
        type_position="entrée", source_position="commune", certified=True, street_score=0.98
    )
    weak = ban_uncertainty_m(
        type_position="inconnue", source_position="inconnue", certified=False, street_score=0.6
    )
    assert strong < weak
    assert not ban_declared_weak(
        type_position="entrée", source_position="commune", certified=True, street_score=0.98
    )
    assert ban_declared_weak(
        type_position="segment", source_position="inconnue", certified=False, street_score=0.98
    )


def test_choose_ban_match_prefers_insee_then_street_without_truth() -> None:
    postcode = BanCandidate(
        2.33, 48.86, "RUE DE LA PAIX", "75002", "75102", "entrée", "commune", True, "POSTCODE"
    )
    insee = BanCandidate(
        2.331, 48.861, "RUE PAIX", "75002", "75102", "segment", "inconnue", False, "INSEE"
    )
    match = choose_ban_match(crm_street="RUE DE LA PAIX", candidates=[postcode, insee])
    assert match is not None
    assert match["ban_geography_match"] == "INSEE"
    assert match["ban_declared_weak"] is True
    assert match["ban_match_count"] == 2


def test_ban_equal_priority_distinct_metadata_is_order_independent() -> None:
    from dataclasses import replace
    first = BanCandidate(2.33,48.86,'RUE TEST','75001','75101','entrée','commune',True,'INSEE')
    second = replace(first,postcode='75002',type_position='ENTREE')
    forward=choose_ban_match(crm_street='RUE TEST',candidates=[first,second])
    assert forward==choose_ban_match(crm_street='RUE TEST',candidates=[second,first])
    assert forward['ban_match_count']==2


def test_ban_exact_duplicates_preserve_match_count_and_no_crash() -> None:
    first = BanCandidate(2.33,48.86,'RUE TEST','75001','75101','entrée','commune',True,'INSEE')
    result=choose_ban_match(crm_street='RUE TEST',candidates=[first,first])
    assert result['ban_match_count']==2
    assert result['ban_resolved']


def test_generator_selection_rejects_truth_and_forbidden_folds(tmp_path: Path) -> None:
    truth = tmp_path / "truth.parquet"
    pq.write_table(
        pa.table({"query_id": ["q"], "fold": pa.array([2], type=pa.int8()), "human_ground_truth_siret": ["123"]}),
        truth,
    )
    with pytest.raises(ValueError, match="only query_id and fold"):
        validate_label_blind_queries(truth)

    fold1 = tmp_path / "fold1.parquet"
    pq.write_table(pa.table({"query_id": ["q"], "fold": pa.array([1], type=pa.int8())}), fold1)
    with pytest.raises(ValueError, match="folds 2/3/4 only"):
        validate_label_blind_queries(fold1)


def test_oof_feature_contract_excludes_all_identifiers() -> None:
    assert not FORBIDDEN_MODEL_FEATURES.intersection(FEATURE_NAMES)


def test_metric_denominator_keeps_queries_without_ban_or_candidates() -> None:
    block = _metric_block([], {"resolved", "unresolved"})
    assert block["queries"] == 2
    assert block["exact_oracle_at_250"] == 0


def test_geo_module_has_no_pyproj_dependency() -> None:
    source = (Path(__file__).parents[1] / "src/xgb_matcher/retrieval_geo_v16.py").read_text()
    assert "import pyproj" not in source
    assert "from pyproj" not in source
