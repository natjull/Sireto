from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.xgb_matcher.retrieval_ltr_admission import (
    AdmissionConfig,
    build_internal_union,
    evaluate_outcomes,
    feature_order,
    prepare_training_rows,
    protected_top100,
    score_and_select,
    train_ranker,
    validate_candidate_input,
)


@pytest.fixture(scope="module")
def config() -> AdmissionConfig:
    return AdmissionConfig.load()


def _siret(siren_seed: int, nic: int = 1) -> str:
    return f"{siren_seed:09d}{nic:05d}"


def _candidate(
    *,
    query_id: str,
    fold: int,
    candidate_siret: str,
    gt_siret: str,
    source: str,
    rank: int,
    score: float = 0.0,
    address: str = "10 RUE DES TESTS",
    site_key: str = "",
    operational: list[str] | None = None,
) -> dict[str, object]:
    return {
        "query_id": query_id,
        "siret": candidate_siret,
        "siren": candidate_siret[:9],
        "fold": fold,
        "gt_siret": gt_siret,
        "crm_name": "ALPHA SERVICES",
        "crm_address": "10 RUE DES TESTS 75001 PARIS",
        "crm_number": "10",
        "crm_insee": "75056",
        "crm_postcode": "75001",
        "names": ["ALPHA SERVICES" if rank == 1 else f"SOCIETE {rank}"],
        "addresses": [address],
        "number": "10",
        "insee": "75056",
        "postcode": "75001",
        "state": "A",
        "is_siege": False,
        "retrieval_source": source,
        "retrieval_rank": rank,
        "retrieval_score": score,
        "retrieval_latency_ms": 10.0,
        "site_key": site_key,
        "source_kind": "HUMAN_CRM",
        "v2_exact": True,
        "v3_exact": True,
        "ground_truth_state": "A",
        "pool_size": 500,
        "unseen_siren": True,
        "acceptable_sirets_operational_json": json.dumps(
            operational or [gt_siret]
        ),
    }


def test_union_cap_is_label_blind_and_protects_exact_and_consensus(
    config: AdmissionConfig,
) -> None:
    gt = _siret(999999999, 1)
    rows = [
        _candidate(
            query_id="q-cap",
            fold=0,
            candidate_siret=_siret(100000000 + index, 1),
            gt_siret=gt,
            source="name_word",
            rank=index,
        )
        for index in range(1, 2006)
    ]
    exact_siret = rows[-1]["siret"]
    rows[-1]["retrieval_source"] = "name_exact"
    rows[-2]["retrieval_source"] = "name_char+address_word"

    union, diagnostics = build_internal_union(
        pd.DataFrame(rows), config, allowed_folds=[0]
    )

    assert len(union) == 2000
    assert diagnostics["queries_capped_at_2000"] == 1
    assert exact_siret in set(union["candidate_siret"])
    assert rows[-2]["siret"] in set(union["candidate_siret"])
    assert gt not in set(union["candidate_siret"])
    assert diagnostics["positive_injection"] is False


def test_feature_contract_has_no_identifiers_and_includes_overlay(
    config: AdmissionConfig,
) -> None:
    features = feature_order(config)
    assert not any(
        "siret" in feature or "siren" in feature or "query_id" in feature
        for feature in features
    )
    for channel in (
        "rne_name",
        "rne_address",
        "bodacc_name",
        "bodacc_address",
        "bodacc_relation",
    ):
        assert f"source_{channel}" in features
    assert "bodacc_name" not in config.exact_channels
    assert "bodacc_address" not in config.exact_channels


def test_typed_dossier_contract_uses_single_union_and_source_features() -> None:
    typed = AdmissionConfig.load("config/retrieval_ltr_admission_dossier_v2.json")
    assert typed.internal_union_cap == 2000
    assert typed.max_candidates == 100
    assert "fielded_name_bm25" in typed.channels
    assert "number_exact" in typed.channels
    assert "number_exact" not in typed.exact_channels
    assert "bodacc_relation" in typed.channels
    features = feature_order(typed)
    assert "official_source_registry_current" in features
    assert "official_source_rne" in features
    assert "official_source_bodacc" in features


def test_per_channel_columns_can_supply_overlay_provenance(
    config: AdmissionConfig,
) -> None:
    truth = _siret(876543210, 1)
    row = _candidate(
        query_id="q-overlay",
        fold=0,
        candidate_siret=truth,
        gt_siret=truth,
        source="",
        rank=7,
    )
    row["retrieval_rank_rne_name"] = 2
    row["retrieval_score_rne_name"] = 8.5
    row["bodacc_relation_rank"] = 3
    row["bodacc_relation_score"] = 4.0

    union, _ = build_internal_union(pd.DataFrame([row]), config, allowed_folds=[0])

    assert union.iloc[0]["source_rne_name"] == 1
    assert union.iloc[0]["source_bodacc_relation"] == 1
    assert union.iloc[0]["is_consensus_protected"] == 1
    assert union.iloc[0]["rank_reciprocal_rne_name"] == 0.5


def test_same_siren_same_site_is_removed_from_training_negatives(
    config: AdmissionConfig,
) -> None:
    truth = _siret(111111111, 1)
    sibling = _siret(111111111, 2)
    real_negative = _siret(222222222, 1)
    raw = pd.DataFrame(
        [
            _candidate(
                query_id="q-site",
                fold=2,
                candidate_siret=truth,
                gt_siret=truth,
                source="name_exact+address_exact",
                rank=1,
                site_key="SITE-A",
                operational=[truth, sibling],
            ),
            _candidate(
                query_id="q-site",
                fold=2,
                candidate_siret=sibling,
                gt_siret=truth,
                source="address_exact",
                rank=2,
                site_key="SITE-A",
                operational=[truth, sibling],
            ),
            _candidate(
                query_id="q-site",
                fold=2,
                candidate_siret=real_negative,
                gt_siret=truth,
                source="name_char",
                rank=3,
                site_key="SITE-B",
                operational=[truth, sibling],
            ),
        ]
    )
    union, _ = build_internal_union(raw, config, allowed_folds=[2])

    training, diagnostics = prepare_training_rows(union, config)

    assert sibling not in set(training["candidate_siret"])
    assert set(training["candidate_siret"]) == {truth, real_negative}
    assert diagnostics["same_siren_same_site_negatives_excluded"] == 1
    assert diagnostics["positive_injection"] is False


def test_protected_top100_retains_low_scoring_exact_and_consensus(
    config: AdmissionConfig,
) -> None:
    frame = pd.DataFrame(
        {
            "candidate_siret": [_siret(300000000 + index, 1) for index in range(120)],
            "ltr_score": np.arange(120, dtype=float)[::-1],
            "retrieval_channel_count": 1,
            "union_rank": np.arange(1, 121),
            "is_exact_protected": 0,
            "exact_signal_count": 0,
            "is_consensus_protected": 0,
        }
    )
    frame.loc[118, ["is_exact_protected", "exact_signal_count"]] = [1, 1]
    frame.loc[119, ["is_consensus_protected", "retrieval_channel_count"]] = [1, 2]

    selected = protected_top100(frame, config)

    assert len(selected) == 100
    assert frame.loc[118, "candidate_siret"] in set(selected["candidate_siret"])
    assert frame.loc[119, "candidate_siret"] in set(selected["candidate_siret"])
    assert selected["candidate_siret"].is_unique


def _ranker_fixture() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold in (0, 2, 3, 4):
        for query_number in range(2):
            siren = 400000000 + fold * 100 + query_number
            truth = _siret(siren, 1)
            query_id = f"q-{fold}-{query_number}"
            for position in range(1, 6):
                candidate = truth if position == 1 else _siret(siren + position, 1)
                rows.append(
                    _candidate(
                        query_id=query_id,
                        fold=fold,
                        candidate_siret=candidate,
                        gt_siret=truth,
                        source=("name_exact+address_word" if position == 1 else "name_char"),
                        rank=position,
                        score=1.0 / position,
                        site_key=f"SITE-{query_id}-{position}",
                    )
                )
    return pd.DataFrame(rows)


def test_fixed_ranker_trains_only_234_and_scores_dev(
    config: AdmissionConfig,
) -> None:
    raw = _ranker_fixture()
    train_raw = raw[raw["fold"].isin(config.train_folds)]
    dev_raw = raw[raw["fold"].eq(config.dev_fold)]
    train_union, _ = build_internal_union(
        train_raw, config, allowed_folds=config.train_folds
    )
    dev_union, _ = build_internal_union(
        dev_raw, config, allowed_folds=[config.dev_fold]
    )

    model, diagnostics = train_ranker(train_union, config)
    selected, outcomes, latency = score_and_select(model, dev_union, config)

    assert diagnostics["train_folds"] == [2, 3, 4]
    assert diagnostics["test_fold_read"] is False
    assert selected.groupby("query_id").size().max() <= 100
    assert outcomes["exact_oracle_hit"].all()
    assert latency["score_and_select_ms"]["p95"] is not None
    assert latency["retrieval_latency_complete"] is True
    assert latency["retrieval_ms"]["p50"] == 10.0
    assert latency["end_to_end_admission_ms"]["p50"] >= 10.0


def test_metrics_keep_exact_and_operational_separate_and_verdicts(
    config: AdmissionConfig,
) -> None:
    outcomes = pd.DataFrame(
        {
            "identifiable_exact": [True, True, True, True],
            "historical_gt_siret": ["1", "2", "3", "4"],
            "historical_oracle_hit": [True, True, True, True],
            "historical_hit_at_100": [True, True, True, True],
            "v2_exact_available": [True, True, True, True],
            "v2_exact": [True, True, True, True],
            "v3_exact_available": [True, True, True, True],
            "v3_exact": [True, True, True, True],
            "exact_oracle_hit": [True, True, True, True],
            "operational_oracle_hit": [True, True, True, True],
            "exact_hit_at_100": [True, True, True, True],
            "operational_hit_at_100": [True, True, True, True],
            "union_candidate_count": [100, 100, 100, 100],
            "selected_candidate_count": [100, 100, 100, 100],
        }
    )
    latency = {
        "retrieval_latency_complete": True,
        "end_to_end_admission_ms": {"p95": 100.0, "p99": 150.0},
    }
    go = evaluate_outcomes(outcomes, config, latency=latency)
    assert go["verdict"] == "GO"
    outcomes.loc[0, "exact_hit_at_100"] = False
    pivot = evaluate_outcomes(outcomes, config, latency=latency)
    assert pivot["verdict"] == "PIVOT"
    assert pivot["metrics"]["operational_recall_at_100"]["rate"] == 1.0
    slow = evaluate_outcomes(
        outcomes,
        config,
        latency={
            "retrieval_latency_complete": True,
            "end_to_end_admission_ms": {"p95": 1000.1, "p99": 1500.0},
        },
    )
    assert slow["verdict"] == "PIVOT"
    assert slow["gates"]["end_to_end_p95_ms"]["passed"] is False
    stopped = evaluate_outcomes(
        outcomes,
        config,
        integrity={"authorization": False},
        latency=latency,
    )
    assert stopped["verdict"] == "STOP"


def test_missing_latency_or_qualification_is_an_explicit_pivot(
    config: AdmissionConfig,
) -> None:
    outcomes = pd.DataFrame(
        {
            "identifiable_exact": [True],
            "exact_oracle_hit": [True],
            "operational_oracle_hit": [True],
            "exact_hit_at_100": [True],
            "operational_hit_at_100": [True],
            "union_candidate_count": [100],
            "selected_candidate_count": [100],
        }
    )
    result = evaluate_outcomes(outcomes, config)

    assert result["verdict"] == "PIVOT"
    assert result["gates"]["latency_measured"]["reason"] == "latency_not_measured"
    assert result["qualification_views"]["v2"]["status"] == "NOT_AVAILABLE"
    assert result["qualification_views"]["v3"]["status"] == "NOT_AVAILABLE"
    assert result["segments"] == {}


def test_dense_and_synthetic_inputs_are_rejected(config: AdmissionConfig) -> None:
    truth = _siret(555555555, 1)
    dense = pd.DataFrame(
        [
            {
                **_candidate(
                    query_id="q-dense",
                    fold=0,
                    candidate_siret=truth,
                    gt_siret=truth,
                    source="dense",
                    rank=1,
                ),
                "dense_score": 1.0,
            }
        ]
    )
    with pytest.raises(ValueError, match="Dense"):
        validate_candidate_input(dense, config, allowed_folds=[0])
    synthetic = dense.drop(columns=["dense_score"]).copy()
    synthetic["retrieval_source"] = "name_word"
    synthetic["source_kind"] = "SYNTHETIC_GT"
    with pytest.raises(ValueError, match="Synthetic"):
        validate_candidate_input(synthetic, config, allowed_folds=[0])
