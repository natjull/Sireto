from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.xgb_matcher.retrieval_site_inventory_v17 import (
    balanced_parent_order,
    build_parent_rows,
    require_complete_site_expansion,
    rrf_parent_order,
    same_site_evidence,
    split_street_number,
    validate_query_selection,
)


def test_split_street_number_normalizes_suffix_and_missing_zero() -> None:
    assert split_street_number("0012 b") == ("12", "BIS")
    assert split_street_number("12 TER") == ("12", "TER")
    assert split_street_number("0") == ("", "")
    assert split_street_number("") == ("", "")


def test_operational_same_site_requires_strong_address() -> None:
    base = {
        "query_number": "12 BIS",
        "query_street": "RUE DE LA PAIX",
        "query_insee": "75001",
        "query_postcode": "75001",
        "site_number": "12",
        "site_suffix": "BIS",
        "site_street": "RUE DE LA PAIX",
        "site_insee": "75001",
        "site_postcode": "75001",
    }
    assert same_site_evidence(**base)
    assert not same_site_evidence(**{**base, "site_suffix": "TER"})
    assert not same_site_evidence(**{**base, "site_street": "RUE DE LYON"})
    assert not same_site_evidence(**{**base, "site_insee": "69001", "site_postcode": "69001"})


def test_operational_same_site_refuses_missing_number_and_ban_like_proximity() -> None:
    assert not same_site_evidence(
        query_number="",
        query_street="RUE DE LA PAIX",
        query_insee="75001",
        query_postcode="75001",
        site_number="12",
        site_suffix="",
        site_street="RUE DE LA PAIX",
        site_insee="75001",
        site_postcode="75001",
    )


def test_rrf_parent_order_is_deterministic_and_deduplicated() -> None:
    sources = {
        "TEXT": ["111111111", "222222222"],
        "ADDRESS": ["222222222", "333333333"],
        "GEO": ["333333333", "111111111"],
    }
    first = rrf_parent_order(sources, weights={"TEXT": 1, "ADDRESS": 1.5, "GEO": 0.75}, cap=3)
    second = rrf_parent_order(sources, weights={"TEXT": 1, "ADDRESS": 1.5, "GEO": 0.75}, cap=3)
    assert first == second
    assert set(first) == {"111111111", "222222222", "333333333"}


def test_balanced_parent_order_reserves_sources_then_fills_duplicates() -> None:
    sources = {
        "TEXT": ["111111111", "222222222", "333333333"],
        "ADDRESS": ["111111111", "444444444", "555555555"],
        "GEO": ["222222222", "666666666", "777777777"],
    }
    result = balanced_parent_order(
        sources,
        quotas={"TEXT": 2, "ADDRESS": 2, "GEO": 2},
        cap=6,
    )
    assert len(result) == 6
    assert len(set(result)) == 6
    assert "444444444" in result and "666666666" in result


def test_build_parent_rows_has_no_truth_and_exact_primary_cap() -> None:
    candidates, presence = build_parent_rows(
        query_id="q",
        fold=2,
        source_orders={
            "TEXT": ["111111111", "222222222"],
            "ADDRESS": ["333333333"],
            "GEO": ["444444444"],
        },
        variants={
            "PRIMARY": {"text_quota": 2, "address_quota": 1, "geo_quota": 1},
            "RRF": {"text_weight": 1.0, "address_weight": 1.5, "geo_weight": 0.75},
        },
        primary_variant="PRIMARY",
        cap=4,
    )
    assert len([row for row in candidates if row["is_primary"]]) == 4
    assert len(presence) == 4
    assert not ({"human_ground_truth_siret", "label", "target"} & set(candidates[0]))


def test_site_expansion_proof_rejects_any_cap() -> None:
    complete = [
        {"query_id": "q", "siren": "111111111", "emitted_site_count": 2, "source_site_count": 2},
        {"query_id": "q", "siren": "111111111", "emitted_site_count": 2, "source_site_count": 2},
    ]
    require_complete_site_expansion(complete)
    with pytest.raises(ValueError, match="not exhaustive"):
        require_complete_site_expansion(
            [{"query_id": "q", "siren": "111111111", "emitted_site_count": 1, "source_site_count": 2}]
        )


def test_query_selection_is_label_blind_and_fold_closed(tmp_path: Path) -> None:
    path = tmp_path / "queries.parquet"
    pq.write_table(
        pa.table({"query_id": ["q2", "q3"], "fold": pa.array([2, 3], type=pa.int8())}),
        path,
    )
    assert validate_query_selection(path, expected_count=2) == (("q2", 2), ("q3", 3))
    pq.write_table(
        pa.table({"query_id": ["q1"], "fold": pa.array([1], type=pa.int8()), "label": [1]}),
        path,
    )
    with pytest.raises(ValueError):
        validate_query_selection(path)


def test_frozen_config_declares_no_site_cap_and_separate_operational_metric() -> None:
    config = json.loads(
        (Path(__file__).parents[1] / "config/retrieval_all_human_v17_site_inventory_oracle.json").read_text()
    )
    assert config["scope"] == "BOUNDED_STEPS_1_TO_3_ONLY"
    assert config["recipe"]["all_sites_per_selected_parent"] is True
    assert config["recipe"]["pre_score_site_cap"] is None
    assert "operational_metric_substitutes_exact" in config["forbidden"]
    assert "national_run" in config["forbidden"]
