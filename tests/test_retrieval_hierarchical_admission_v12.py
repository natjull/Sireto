from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.run_retrieval_hierarchical_admission_v12 as admission_cli
import src.xgb_matcher.retrieval_hierarchical_admission_v12 as v12
from src.xgb_matcher.retrieval_top100_cascade_v12 import CASCADE_PARENT_FEATURES


TEST_CHANNELS = (
    "legal_name_exact",
    "legal_name_word",
    "trade_name_word",
    "address_exact",
    "address_word",
    "geo_exact_cell",
    "contrastive_siren_rescue",
)


def tiny_config(*, sample_cap: int = 4) -> v12.AdmissionConfigV12:
    raw = deepcopy(v12.AdmissionConfigV12.default().raw)
    admission = raw["admission_v12"]
    admission["channels"] = list(TEST_CHANNELS)
    admission["channel_families"] = {
        "current_name": ["legal_name_exact", "legal_name_word", "trade_name_word"],
        "current_address": ["address_exact", "address_word"],
        "geographic_cell": ["geo_exact_cell"],
        "parent_sparse": ["contrastive_siren_rescue"],
    }
    admission["fit_candidate_sampling"] = {
        "maximum_candidates_per_query": sample_cap,
        "union_head": min(2, sample_cap),
        "channel_head": 1,
        "retain_natural_positive_if_present": True,
    }
    for level in ("parent", "site"):
        admission["xgboost"][level].update(
            {
                "n_estimators": 3,
                "learning_rate": 0.2,
                "max_depth": 2,
                "min_child_weight": 1.0,
                "reg_lambda": 1.0,
                "n_jobs": 1,
                "lambdarank_num_pair_per_sample": 2,
            }
        )
    raw["gates"] = {
        "master_exact_recall_at_2000": {"minimum": 0.0},
        "prerank_exact_recall_at_500": {"minimum": 0.0},
        "product_exact_recall_at_100": {
            "minimum": 0.0,
            "wilson95_lower_minimum": 0.0,
            "minimum_each_outer_fold": 0.0,
        },
    }
    return v12.AdmissionConfigV12.from_mapping(raw)


def _siret(parent: int, site: int) -> str:
    return f"{parent:09d}{site:05d}"


def make_union(
    *, queries_per_fold: int = 1, candidates_per_query: int = 8
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold in (2, 3, 4):
        for query_index in range(queries_per_fold):
            truth_parent = 100_000_000 + fold * 10_000 + query_index * 100
            truth = _siret(truth_parent, 1)
            query_id = f"f{fold}-q{query_index}"
            for index in range(candidates_per_query):
                parent = truth_parent if index < 2 else truth_parent + index
                candidate = _siret(parent, index + 1 if index < 2 else 1)
                is_truth = candidate == truth
                channels = (
                    ["legal_name_exact", "legal_name_word", "address_exact"]
                    if is_truth
                    else ["legal_name_word", "trade_name_word"]
                    if index == 1
                    else ["address_word"]
                )
                row: dict[str, object] = {
                    "query_id": query_id,
                    "fold": fold,
                    "human_ground_truth_siret": truth,
                    "acceptable_sirets_operational_json": json.dumps([truth]),
                    "union_rank": index + 1,
                    "siret": candidate,
                    "siren": f"{parent:09d}",
                    "crm_name": f"Entreprise {truth_parent}",
                    "crm_address": "10 rue de la Paix 75001 Paris",
                    "crm_number": "10",
                    "crm_postcode": "75001",
                    "crm_insee": "75056",
                    "candidate_names": [
                        f"Entreprise {truth_parent}" if is_truth else f"Concurrent {index}"
                    ],
                    "candidate_addresses": [
                        "10 rue de la Paix 75001 Paris"
                        if is_truth
                        else f"{index + 20} rue Autre 75001 Paris"
                    ],
                    "candidate_number": "10" if is_truth else str(index + 20),
                    "candidate_postcode": "75001",
                    "candidate_insee": "75056",
                    "candidate_state": "A",
                    "channels": channels,
                    "official_evidence_sources": ["SIRENE"],
                    "retrieval_latency_ms": 1.0,
                    "official_catalog_present": True,
                }
                for rank, channel in enumerate(channels, start=1):
                    row[f"{channel}_rank"] = rank
                    row[f"{channel}_score"] = 10.0 - rank
                rows.append(row)
    return pd.DataFrame(rows)


def test_config_and_parent_model_use_exact_cascade_schema() -> None:
    config = v12.AdmissionConfigV12.load("config/retrieval_all_human_v12.json")
    assert config.train_folds == (2, 3, 4)
    assert config.prerank_cap == 500
    assert v12.parent_feature_names(config) == CASCADE_PARENT_FEATURES
    assert len(v12.site_feature_names(config)) == len(
        set(v12.site_feature_names(config))
    )
    assert {"v12_parent_local_expansion", "v12_rescue_local_expansion"}.issubset(
        config.channels
    )
    assert set(config._admission) >= {
        "channels",
        "channel_families",
        "fit_candidate_sampling",
        "selection",
        "xgboost",
    }


def test_production_rejects_any_label_column_before_model_use() -> None:
    config = tiny_config()
    labelled = make_union()
    with pytest.raises(ValueError, match="forbidden label columns"):
        v12.predict_top100(None, labelled, config)  # type: ignore[arg-type]
    empty_label = labelled.drop(columns=["human_ground_truth_siret"])
    empty_label["ground_truth_siret"] = ""
    with pytest.raises(ValueError, match="forbidden label columns"):
        v12.canonicalize_union(empty_label, config, labels_allowed=False)


def test_complete_union_features_are_unchanged_by_fit_sampling() -> None:
    config = tiny_config(sample_cap=4)
    raw = make_union(candidates_per_query=10)
    # Move the positive beyond the sample head; it remains natural and is only
    # retained for fitting after all full-union features exist.
    raw.loc[raw["siret"].eq(raw["human_ground_truth_siret"]), "union_rank"] = 10
    for query_id, group in raw.groupby("query_id"):
        positive_index = group.index[group["siret"].eq(group["human_ground_truth_siret"])][0]
        collision = group.index[group["union_rank"].eq(10) & group.index.to_series().ne(positive_index)]
        if len(collision):
            raw.loc[collision[0], "union_rank"] = 1
    complete = v12.canonicalize_union(raw, config, labels_allowed=True, require_labels=True)
    sampled = v12.sample_precomputed_candidates(complete, config)
    features = [
        "union_rank_percentile_inverse",
        "within_parent_candidate_count_log",
        "channel__legal_name_exact__rank_percentile_inverse",
    ]
    shared = sampled[["query_id", "candidate_siret", *features]].merge(
        complete[["query_id", "candidate_siret", *features]],
        on=["query_id", "candidate_siret"],
        suffixes=("_sample", "_full"),
        validate="one_to_one",
    )
    for feature in features:
        np.testing.assert_allclose(shared[f"{feature}_sample"], shared[f"{feature}_full"])
    assert sampled.groupby("query_id").size().eq(4).all()
    assert sampled["candidate_siret"].eq(sampled["human_siret"]).groupby(sampled["query_id"]).any().all()

    unsampled_config = tiny_config(sample_cap=20)
    unsampled = v12.sample_precomputed_candidates(complete, unsampled_config)
    assert len(unsampled) == len(complete)


def test_consensus_counts_independent_families_not_correlated_channels() -> None:
    config = tiny_config()
    raw = make_union()
    row = raw.index[0]
    raw.at[row, "channels"] = ["legal_name_exact", "legal_name_word", "trade_name_word"]
    for channel in TEST_CHANNELS:
        raw.at[row, f"{channel}_rank"] = np.nan
        raw.at[row, f"{channel}_score"] = np.nan
    raw.at[row, "legal_name_exact_rank"] = 1
    raw.at[row, "legal_name_word_rank"] = 2
    raw.at[row, "trade_name_word_rank"] = 3
    prepared = v12.canonicalize_union(raw, config, labels_allowed=True)
    candidate = prepared[
        prepared["query_id"].eq(raw.at[row, "query_id"])
        & prepared["candidate_siret"].eq(raw.at[row, "siret"])
    ].iloc[0]
    assert candidate["evidence_family_count"] == 1
    assert candidate["independent_consensus"] == 0

    raw.at[row, "channels"] = [
        "legal_name_exact",
        "legal_name_word",
        "trade_name_word",
        "geo_exact_cell",
    ]
    raw.at[row, "geo_exact_cell_rank"] = 1
    raw.at[row, "geo_exact_cell_score"] = 1.0
    prepared = v12.canonicalize_union(raw, config, labels_allowed=True)
    candidate = prepared[
        prepared["query_id"].eq(raw.at[row, "query_id"])
        & prepared["candidate_siret"].eq(raw.at[row, "siret"])
    ].iloc[0]
    assert candidate["evidence_family_count"] == 2
    assert candidate["independent_consensus"] == 1


def test_cascade_signals_supply_channel_rank_and_score_without_flat_columns() -> None:
    config = tiny_config()
    raw = make_union().drop(columns=["legal_name_exact_rank", "legal_name_exact_score"], errors="ignore")
    raw["signals"] = None
    index = raw.index[0]
    raw.at[index, "channels"] = ["legal_name_exact", "address_exact"]
    raw.at[index, "signals"] = [
        {"channel": "legal_name_exact", "rank": 7, "score": 4.5},
        {"channel": "address_exact", "rank": 2, "score": 8.0},
    ]
    prepared = v12.canonicalize_union(raw, config, labels_allowed=True)
    candidate = prepared[
        prepared["query_id"].eq(raw.at[index, "query_id"])
        & prepared["candidate_siret"].eq(raw.at[index, "siret"])
    ].iloc[0]
    assert candidate["channel__legal_name_exact__present"] == 1
    assert candidate["__raw_rank__legal_name_exact"] == 7
    assert candidate["__raw_score__legal_name_exact"] == 4.5


def test_number_only_is_not_geography_and_bridge_maps_headquarters() -> None:
    raw_config = deepcopy(tiny_config().raw)
    raw_config["admission_v12"]["channels"].append("number_exact")
    raw_config["admission_v12"]["channel_families"]["current_address"].append(
        "number_exact"
    )
    config = v12.AdmissionConfigV12.from_mapping(raw_config)
    raw = make_union()
    index = raw.index[0]
    for channel in config.channels:
        raw.at[index, f"{channel}_rank"] = np.nan
        raw.at[index, f"{channel}_score"] = np.nan
    raw.at[index, "channels"] = ["legal_name_exact", "number_exact"]
    raw.at[index, "candidate_insee"] = "13055"
    raw.at[index, "candidate_postcode"] = "13001"
    raw.at[index, "candidate_number"] = "10"
    raw.at[index, "candidate_is_headquarters"] = True
    raw.at[index, "legal_name_exact_rank"] = 1
    raw.at[index, "legal_name_exact_score"] = 1.0
    raw.at[index, "number_exact_rank"] = 1
    raw.at[index, "number_exact_score"] = 1.0
    prepared = v12.canonicalize_union(raw, config, labels_allowed=True)
    candidate = prepared[
        prepared["query_id"].eq(raw.at[index, "query_id"])
        & prepared["candidate_siret"].eq(raw.at[index, "siret"])
    ].iloc[0]
    assert candidate["street_number_match"] == 1
    assert candidate["exact_name_present"] == 1
    assert candidate["exact_geo_present"] == 0
    assert candidate["joint_name_geo"] == 0
    assert candidate["candidate_is_siege"] == 1


def test_adaptive_selection_has_no_hard_three_site_parent_cap() -> None:
    config = tiny_config()
    rows: list[dict[str, object]] = []
    dominant_parent = "123456789"
    for index in range(150):
        parent = dominant_parent if index < 12 else f"{200_000_000 + index:09d}"
        rows.append(
            {
                "query_id": "q",
                "candidate_siret": parent + f"{index + 1:05d}",
                "candidate_parent": parent,
                "joint_name_geo": 0,
                "site_score": float(200 - index),
                "parent_score_probability": 0.9 if parent == dominant_parent else 0.001,
            }
        )
    scored = pd.DataFrame(rows)
    scored["ranked_rank"] = scored["site_score"].rank(
        method="first", ascending=False
    )
    scored["prerank_rank"] = scored["ranked_rank"].where(
        scored["ranked_rank"].le(config.prerank_cap)
    )
    selected = v12.select_top100(scored, config)
    assert len(selected) == 100
    assert int(selected["candidate_parent"].eq(dominant_parent).sum()) > 3
    assert set(selected["selection_reason"]) == {"adaptive_marginal"}


def test_tied_site_scores_are_invariant_to_master_row_permutation() -> None:
    config = tiny_config()
    complete = v12.canonicalize_union(
        make_union(candidates_per_query=150), config, labels_allowed=True
    )

    class TiedModel:
        def predict(self, matrix: pd.DataFrame) -> np.ndarray:
            return np.zeros(len(matrix), dtype=float)

    first = v12.select_top100(
        v12.attach_site_scores(TiedModel(), complete, config),  # type: ignore[arg-type]
        config,
    )
    permuted = complete.sample(frac=1.0, random_state=42).reset_index(drop=True)
    second = v12.select_top100(
        v12.attach_site_scores(TiedModel(), permuted, config),  # type: ignore[arg-type]
        config,
    )
    columns = ["query_id", "selected_rank", "candidate_siret", "selection_reason"]
    pd.testing.assert_frame_equal(
        first[columns].sort_values(["query_id", "selected_rank"]).reset_index(drop=True),
        second[columns].sort_values(["query_id", "selected_rank"]).reset_index(drop=True),
    )


def test_fold_zero_and_one_are_rejected_before_any_fit() -> None:
    config = tiny_config()
    for forbidden_fold in (0, 1):
        raw = make_union()
        raw["fold"] = forbidden_fold
        complete = v12.canonicalize_union(raw, config, labels_allowed=True, require_labels=True)
        with pytest.raises(ValueError, match="configured training folds"):
            v12.assert_ground_truth_parent_disjoint(complete, config.train_folds)


def test_cascade_parent_adapter_rejects_schema_drift() -> None:
    values = [float(index) for index in range(len(CASCADE_PARENT_FEATURES))]
    parents = pd.DataFrame(
        {
            "query_id": ["q"],
            "siren": ["123456789"],
            "feature_names": [list(CASCADE_PARENT_FEATURES)],
            "feature_values": [values],
        }
    )
    labels = pd.DataFrame(
        {"query_id": ["q"], "fold": [2], "human_siret": ["12345678900001"]}
    )
    adapted = v12.adapt_cascade_parent_rows(parents, labels)
    assert adapted[list(CASCADE_PARENT_FEATURES)].iloc[0].tolist() == values
    parents.at[0, "feature_names"] = list(reversed(CASCADE_PARENT_FEATURES))
    with pytest.raises(ValueError, match="schema drift"):
        v12.adapt_cascade_parent_rows(parents, labels)


def test_parent_ranker_refuses_post_expansion_candidate_aggregates() -> None:
    config = tiny_config()
    complete = v12.canonicalize_union(
        make_union(), config, labels_allowed=True, require_labels=True
    )
    with pytest.raises(ValueError, match="pre-expansion cascade"):
        v12.fit_parent_ranker(complete, config)


def test_sealed_cascade_parent_scores_are_used_without_parent_refit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tiny_config()
    raw = make_union()
    raw["parent_feature_names"] = [
        list(CASCADE_PARENT_FEATURES) for _ in range(len(raw))
    ]
    raw["parent_feature_values"] = [
        [0.0] * len(CASCADE_PARENT_FEATURES) for _ in range(len(raw))
    ]
    raw["parent_rank"] = raw.groupby("query_id")["siren"].transform(
        lambda values: pd.factorize(values)[0] + 1
    )
    raw["parent_score"] = 100.0 - raw["parent_rank"]
    complete = v12.canonicalize_union(
        raw, config, labels_allowed=True, require_labels=True
    )
    monkeypatch.setattr(
        v12,
        "fit_parent_ranker",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("pass-B parent refit is forbidden")
        ),
    )
    attached = v12.attach_sealed_parent_scores(complete, config)
    assert attached["_parent_model_score"].notna().all()
    assert attached["parent_rank"].ge(1).all()


def test_outer_oof_rejects_one_shared_candidate_population() -> None:
    config = tiny_config()
    complete = v12.canonicalize_union(
        make_union(), config, labels_allowed=True, require_labels=True
    )
    with pytest.raises(ValueError, match="split-specific"):
        v12.fit_outer_oof_models(  # type: ignore[arg-type]
            complete, config
        )


def test_master_and_prerank_ranks_are_distinct_and_missing_queries_count() -> None:
    config = tiny_config()
    raw = make_union(candidates_per_query=8)
    raw["master_rank"] = raw["union_rank"]
    raw["hydration_rank"] = raw["union_rank"].where(raw["union_rank"].le(4))
    truth_mask = raw["siret"].eq(raw["human_ground_truth_siret"])
    raw.loc[truth_mask, "hydration_rank"] = np.nan
    complete = v12.canonicalize_union(
        raw, config, labels_allowed=True, require_labels=True
    )
    outcomes, _ = v12.evaluate_all_human(
        complete, scored=None, selected=None, config=config
    )
    assert outcomes["truth_master_rank"].eq(1).all()
    assert outcomes["truth_heuristic_hydration_rank"].isna().all()
    assert outcomes["truth_prerank_rank"].isna().all()
    assert outcomes["loss_reason"].eq("EVICTED_MASTER_TO_500").all()

    labels = pd.DataFrame(
        {
            "query_id": ["absent"],
            "fold": [2],
            "human_siret": ["12345678900001"],
            "acceptable_sirets_operational_json": ['["12345678900001"]'],
        }
    )
    missing = v12.missing_label_outcomes(labels, (), config)
    assert len(missing) == 1
    assert missing.iloc[0]["loss_reason"] == "OFFICIAL_CATALOG_STATUS_UNKNOWN"
    assert missing.iloc[0]["union_candidate_count"] == 0

    unknown_catalog = complete.copy()
    unknown_catalog["official_catalog_present"] = pd.NA
    unknown_catalog = unknown_catalog[
        ~unknown_catalog["candidate_siret"].eq(unknown_catalog["human_siret"])
    ]
    unknown_outcomes, unknown_summary = v12.evaluate_all_human(
        unknown_catalog, scored=None, selected=None, config=config
    )
    assert unknown_outcomes["loss_reason"].eq(
        "OFFICIAL_CATALOG_STATUS_UNKNOWN"
    ).all()
    assert unknown_summary["verdict"] == "STOP"


def test_oof_population_reconciliation_rejects_duplicates_gaps_and_fold_drift() -> None:
    labels = pd.DataFrame(
        {"query_id": ["q2", "q3", "q4"], "fold": [2, 3, 4]}
    )
    admission_cli._reconcile_oof_population(labels.copy(), labels)

    duplicate = pd.DataFrame(
        {"query_id": ["q2", "q2", "q4"], "fold": [2, 2, 4]}
    )
    with pytest.raises(ValueError, match="non-empty and unique"):
        admission_cli._reconcile_oof_population(duplicate, labels)

    missing = pd.DataFrame({"query_id": ["q2", "q3"], "fold": [2, 3]})
    with pytest.raises(ValueError, match="sealed label denominator"):
        admission_cli._reconcile_oof_population(missing, labels)

    wrong_fold = labels.copy()
    wrong_fold.loc[wrong_fold["query_id"].eq("q3"), "fold"] = 2
    with pytest.raises(ValueError, match="folds do not match"):
        admission_cli._reconcile_oof_population(wrong_fold, labels)


def test_site_lambdamart_scores_full_master_and_rescues_outside_heuristic_500() -> None:
    config = tiny_config()
    raw = make_union(candidates_per_query=520)
    raw["hydration_rank"] = np.nan
    for _query_id, group in raw.groupby("query_id", sort=False):
        nontruth = group[~group["siret"].eq(group["human_ground_truth_siret"])]
        raw.loc[nontruth.index[:500], "hydration_rank"] = np.arange(1, 501)
    raw["parent_feature_names"] = [
        list(CASCADE_PARENT_FEATURES) for _ in range(len(raw))
    ]
    raw["parent_feature_values"] = [
        [0.0] * len(CASCADE_PARENT_FEATURES) for _ in range(len(raw))
    ]
    raw["parent_rank"] = raw.groupby("query_id")["siren"].transform(
        lambda values: pd.factorize(values)[0] + 1
    )
    raw["parent_score"] = 1000.0 - raw["parent_rank"]
    complete = v12.canonicalize_union(
        raw, config, labels_allowed=True, require_labels=True
    )
    assert complete.loc[
        complete["candidate_siret"].eq(complete["human_siret"]),
        "heuristic_hydration_rank",
    ].isna().all()

    class ExactEvidenceModel:
        def predict(self, matrix: pd.DataFrame) -> np.ndarray:
            return matrix["exact_name_present"].to_numpy(dtype=float) * 100.0

    blind = complete.copy()
    blind["human_siret"] = ""
    scored = v12.score_canonical_candidates(
        v12.HierarchicalModelsV12(None, ExactEvidenceModel(), {}),  # type: ignore[arg-type]
        blind,
        config,
    )
    assert scored.groupby("query_id").size().eq(520).all()
    learned_truth = scored[scored["candidate_siret"].isin(complete["human_siret"])]
    assert learned_truth["prerank_rank"].eq(1).all()
    selected = v12.select_top100(scored, config)
    outcomes, _ = v12.evaluate_all_human(
        complete, scored=scored, selected=selected, config=config
    )
    assert outcomes["truth_heuristic_hydration_rank"].isna().all()
    assert outcomes["truth_prerank_rank"].eq(1).all()
    assert outcomes["selected_hit_at_100"].all()


def test_official_source_temporal_and_relation_features_are_numeric() -> None:
    config = tiny_config()
    raw = make_union()
    row = raw.index[0]
    raw.at[row, "official_evidence_sources"] = ["SIRENE_HISTORY", "INPI_RNE"]
    raw.at[row, "signals"] = [
        {
            "source": "BODACC",
            "channel": "legal_name_exact",
            "rank": 1,
            "score": 9.0,
            "relation_direction": "SUCCESSOR",
            "relation_date": "2020-01-15",
        }
    ]
    raw.at[row, "temporal_valid_from"] = "2018-02-01"
    raw.at[row, "temporal_valid_to"] = "2022-05-31"
    complete = v12.canonicalize_union(raw, config, labels_allowed=True)
    candidate = complete.loc[row]
    assert candidate["site_source_registry_present"] == 1
    assert candidate["site_source_rne_present"] == 1
    assert candidate["site_source_bodacc_present"] == 1
    assert candidate["site_source_sector_present"] == 0
    assert candidate["official_source_count_log"] == pytest.approx(np.log1p(3))
    assert candidate["temporal_ended_before_reference"] == 1
    assert candidate["temporal_valid_at_reference"] == 0
    assert candidate["temporal_end_distance_days_log1p"] > 0
    assert candidate["relation_successor_present"] == 1
    assert candidate["relation_dated_present"] == 1
    assert candidate["relation_nearest_age_days_log1p"] > 0


def test_metrics_publish_curve_wilson_folds_and_exclusive_losses() -> None:
    config = tiny_config()
    complete = v12.canonicalize_union(
        make_union(candidates_per_query=8), config, labels_allowed=True, require_labels=True
    )
    # Label-blind perfect scores for the fixture.
    blind = complete.copy()
    blind["human_siret"] = ""
    blind["site_score"] = 1.0 / blind["union_rank"]
    blind["final_score"] = blind["site_score"]
    blind["ranked_rank"] = blind.groupby("query_id")["site_score"].rank(method="first", ascending=False)
    blind["prerank_rank"] = blind["ranked_rank"].where(
        blind["ranked_rank"].le(config.prerank_cap)
    )
    blind["parent_rank"] = 1.0
    blind["parent_score_probability"] = 1.0 / blind.groupby("query_id")["candidate_siret"].transform("size")
    blind["site_rank"] = blind.groupby(["query_id", "candidate_parent"])["site_score"].rank(method="first", ascending=False)
    blind["admission_latency_ms"] = 1.0
    selected = v12.select_top100(blind, config)
    selected["admission_latency_ms"] = 1.0
    outcomes, summary = v12.evaluate_all_human(
        complete, scored=blind, selected=selected, config=config
    )
    assert set(summary["metrics"]["union_recall"]) == {str(value) for value in v12.METRIC_CUTOFFS}
    assert summary["master_exact_recall_at_2000"]["wilson_95"] is not None
    assert set(summary["outer_folds"]) == {"2", "3", "4"}
    assert summary["verdict"] == "READY_FOR_FOLD1"
    assert summary["product_operational_recall_at_100"]["rate"] == 1.0
    assert (
        summary["operational_equivalence_mode"]
        == "EXACT_ONLY_PENDING_SEALED_SAME_SITE_EVIDENCE"
    )
    assert (
        summary["integrity"]["operational_equivalence_mode"]
        == summary["operational_equivalence_mode"]
    )
    assert summary["latency"]["p99_ms"] == 2.0
    assert outcomes["loss_reason"].eq("HIT").all()
    assert summary["gates"]["oracle_at_2000"]["decision_blocking"] is False
    assert summary["loss_taxonomy"] == summary["metrics"]["loss_taxonomy"]
    assert summary["loss_taxonomy"]["exact_failures"] == 0
    assert summary["loss_taxonomy"]["exclusive_assignment_complete"] is True

    one_failure = outcomes.copy()
    failed_index = one_failure.index[0]
    failed_fold = str(int(one_failure.at[failed_index, "fold"]))
    one_failure.at[failed_index, "selected_hit_at_100"] = False
    one_failure.at[failed_index, "loss_reason"] = "EVICTED_500_TO_100"
    failure_summary = v12.summarize_outcomes(one_failure, config)
    failure_taxonomy = failure_summary["loss_taxonomy"]
    assert failure_taxonomy["denominator_queries"] == len(one_failure)
    assert failure_taxonomy["exact_failures"] == 1
    assert failure_taxonomy["classified_failures"] == 1
    assert failure_taxonomy["unattributed_failures"] == 0
    assert failure_taxonomy["exclusive_assignment_complete"] is True
    assert failure_taxonomy["by_reason"]["EVICTED_500_TO_100"] == {
        "count": 1,
        "rate_all_queries": pytest.approx(1.0 / len(one_failure)),
        "rate_exact_failures": 1.0,
    }
    assert (
        failure_summary["outer_folds"][failed_fold]["loss_taxonomy"]
        ["by_reason"]["EVICTED_500_TO_100"]["count"]
        == 1
    )
    segment_value = str(one_failure.at[failed_index, "segment_origin"] or "UNKNOWN")
    assert (
        failure_summary["segments"]["crm_origin"][segment_value]
        ["loss_taxonomy"]["by_reason"]["EVICTED_500_TO_100"]["count"]
        == 1
    )
    unknown_catalog = outcomes.copy()
    unknown_catalog["official_catalog_present"] = None
    unknown_summary = v12.summarize_outcomes(unknown_catalog, config)
    assert unknown_summary["verdict"] == "STOP"
    assert (
        unknown_summary["stop_reason"]
        == "INTEGRITY_OFFICIAL_CATALOG_UNAVAILABLE"
    )
    assert (
        unknown_summary["gates"]["official_catalog_loss_taxonomy"]["passed"]
        is False
    )


def test_missing_latency_fails_closed_after_recall_passes() -> None:
    config = tiny_config()
    complete = v12.canonicalize_union(
        make_union(candidates_per_query=8),
        config,
        labels_allowed=True,
        require_labels=True,
    )
    blind = complete.copy()
    blind["human_siret"] = ""
    blind["site_score"] = 1.0 / blind["union_rank"]
    blind["final_score"] = blind["site_score"]
    blind["ranked_rank"] = blind.groupby("query_id")["site_score"].rank(
        method="first", ascending=False
    )
    blind["prerank_rank"] = blind["ranked_rank"].where(
        blind["ranked_rank"].le(config.prerank_cap)
    )
    blind["parent_rank"] = 1.0
    blind["parent_score_probability"] = 0.5
    blind["site_rank"] = blind.groupby(
        ["query_id", "candidate_parent"]
    )["site_score"].rank(method="first", ascending=False)
    selected = v12.select_top100(blind, config)
    _, summary = v12.evaluate_all_human(
        complete, scored=blind, selected=selected, config=config
    )
    assert summary["verdict"] == "PIVOT_LATENCY"
    assert summary["gates"]["latency_p95_ms"]["passed"] is False


def test_parent_fit_streaming_sample_caps_rows_and_retains_natural_positive(
    tmp_path: Path,
) -> None:
    config = tiny_config(sample_cap=4)
    rows: list[dict[str, object]] = []
    labels: list[dict[str, object]] = []
    for fold in (2, 3, 4):
        query_id = f"q{fold}"
        truth_parent = f"{fold}99999999"
        labels.append(
            {
                "query_id": query_id,
                "fold": fold,
                "human_siret": truth_parent + "00001",
            }
        )
        for rank in range(1, 7):
            parent = truth_parent if rank == 6 else f"{fold}{rank:08d}"
            rows.append(
                {
                    "query_id": query_id,
                    "fold": fold,
                    "siren": parent,
                    "feature_names": list(CASCADE_PARENT_FEATURES),
                    "feature_values": [float(rank)] * len(CASCADE_PARENT_FEATURES),
                    "parent_rank": rank,
                    "parent_score": float(7 - rank),
                }
            )
    parent_path = tmp_path / "parents.parquet"
    pd.DataFrame(rows).to_parquet(parent_path, index=False)
    sampled, diagnostics = admission_cli._load_parent_fit_sample(
        parent_path, pd.DataFrame(labels), config
    )
    assert sampled.groupby("query_id").size().eq(4).all()
    for label in labels:
        query = sampled[sampled["query_id"].eq(label["query_id"])]
        assert str(label["human_siret"])[:9] in set(query["candidate_parent"])
    assert diagnostics["rows_before_sampling"] == 18
    assert diagnostics["rows_after_sampling"] == 12
    assert diagnostics["natural_parent_positive_queries"] == 3


def test_standalone_oracle_is_diagnostic_and_counts_query_without_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tiny_config()
    raw = make_union().drop(
        columns=["human_ground_truth_siret", "acceptable_sirets_operational_json"]
    )
    truth_by_query = (
        make_union()
        .groupby("query_id", sort=False)
        .first()["human_ground_truth_siret"]
        .to_dict()
    )
    labels = pd.DataFrame(
        [
            {
                "query_id": query_id,
                "fold": int(query_id[1]),
                "human_siret": truth,
                "acceptable_sirets_operational_json": json.dumps([truth]),
            }
            for query_id, truth in truth_by_query.items()
        ]
        + [
            {
                "query_id": "absent",
                "fold": 2,
                "human_siret": "99999999900001",
                "acceptable_sirets_operational_json": '["99999999900001"]',
            }
        ]
    )
    groups = [group.copy() for _, group in raw.groupby("query_id", sort=False)]
    monkeypatch.setattr(admission_cli.AdmissionConfigV12, "load", lambda _path: config)
    monkeypatch.setattr(
        admission_cli,
        "verify_candidate_manifest",
        lambda *_args, **_kwargs: (Path("source.manifest.json"), {}),
    )
    monkeypatch.setattr(
        admission_cli,
        "_load_sealed_query_labels",
        lambda _path, **_kwargs: (labels, Path("labels.manifest.json")),
    )
    monkeypatch.setattr(
        admission_cli,
        "iter_complete_query_frames",
        lambda *_args, **_kwargs: iter(groups),
    )
    monkeypatch.setattr(admission_cli, "sha256_file", lambda _path: "sealed-hash")
    output_dir = tmp_path / "oracle"
    admission_cli.run_oracle(
        argparse.Namespace(
            config=Path("config.json"),
            candidates=Path("master.parquet"),
            labels=Path("labels.parquet"),
            output_dir=output_dir,
        )
    )
    summary = json.loads((output_dir / "summary.json").read_text())
    outcomes = pd.read_parquet(output_dir / "query_outcomes.parquet")
    assert summary["verdict"] == "DIAGNOSTIC_ONLY"
    assert summary["decision_eligible"] is False
    assert summary["query_count"] == 4
    absent = outcomes[outcomes["query_id"].eq("absent")].iloc[0]
    assert absent["loss_reason"] == "OFFICIAL_CATALOG_STATUS_UNKNOWN"
    assert absent["union_candidate_count"] == 0


def test_production_predict_is_closed_until_frozen_refit() -> None:
    with pytest.raises(RuntimeError, match="AFTER_READY_ONLY"):
        admission_cli.run_predict(argparse.Namespace())


def test_oof_rejects_parent_bundle_fitted_on_another_label_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tiny_config()
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    labels_path = tmp_path / "label_ledger.parquet"
    labels_path.write_bytes(b"current-label-ledger")
    labels_manifest = tmp_path / "label_manifest.json"
    labels_manifest.write_text("{}", encoding="utf-8")
    current_label_sha = admission_cli.sha256_file(labels_path)
    current_label_manifest_sha = admission_cli.sha256_file(labels_manifest)
    source_lineage = {
        "source_candidate_sha256": "1" * 64,
        "source_manifest_sha256": "2" * 64,
        "sidecar_manifest_sha256": "3" * 64,
    }
    cascade_manifest = {
        "config_sha256": admission_cli.sha256_file(config_path),
        **source_lineage,
        "parent_model_bundle_sha256": "4" * 64,
        "nested_parent_oof": {
            "training_lineage": {
                "source_retrieval": source_lineage,
                # Source snapshot is correct, but the parent models came from
                # another human-label build and must not enter this OOF run.
                "label_ledger_sha256": "5" * 64,
                "label_ledger_manifest_sha256": current_label_manifest_sha,
            }
        },
    }
    monkeypatch.setattr(admission_cli.AdmissionConfigV12, "load", lambda _path: config)
    monkeypatch.setattr(
        admission_cli,
        "_load_sealed_query_labels",
        lambda _path, **_kwargs: (pd.DataFrame(), labels_manifest),
    )
    monkeypatch.setattr(
        admission_cli,
        "verify_candidate_manifest",
        lambda *_args, **_kwargs: (tmp_path / "cascade_manifest.json", cascade_manifest),
    )
    monkeypatch.setattr(
        admission_cli,
        "verify_parent_candidates",
        lambda *_args, **_kwargs: (tmp_path / "cascade_manifest.json", cascade_manifest),
    )

    with pytest.raises(ValueError, match="another label ledger"):
        admission_cli.run_oof(
            argparse.Namespace(
                config=config_path,
                labels=labels_path,
                outer2_cascade=tmp_path / "outer2",
                outer3_cascade=tmp_path / "outer3",
                outer4_cascade=tmp_path / "outer4",
                output_dir=tmp_path / "oof",
                runtime_profile_manifest=None,
            )
        )
    assert current_label_sha != "5" * 64


def test_oof_without_sealed_runtime_profile_cannot_be_ready() -> None:
    config = tiny_config()
    summary = {
        "latency": {"measured_queries": 3, "p95_ms": 1.0, "p99_ms": 2.0},
        "gates": {
            "prerank_at_500": {"passed": True},
            "selected_recall_at_100": {"passed": True},
            "wilson95_lower_at_100": {"passed": True},
            "worst_fold_at_100": {"passed": True},
            "latency_p95_ms": {"passed": True},
            "latency_p99_ms": {"passed": True},
            "official_catalog_loss_taxonomy": {"passed": True},
        },
        "verdict": "READY_FOR_FOLD1",
    }
    gated = admission_cli._apply_runtime_profile_gate(
        summary, config=config, profile=None, manifest_path=None
    )
    assert gated["verdict"] == "PIVOT_LATENCY"
    assert gated["decision_eligible"] is False
    assert gated["runtime_profile"]["gate_eligible"] is False
    assert gated["gates"]["latency_p95_ms"]["passed"] is False
    assert gated["embedded_latency_diagnostic"]["p95_ms"] == 1.0
    failed_generation = deepcopy(summary)
    failed_generation["gates"]["prerank_at_500"]["passed"] = False
    failed_generation = admission_cli._apply_runtime_profile_gate(
        failed_generation, config=config, profile=None, manifest_path=None
    )
    assert failed_generation["verdict"] == "PIVOT_GENERATION"
    assert failed_generation["decision_eligible"] is False


def test_runtime_profile_requires_exact_512_query_sealed_replay(
    tmp_path: Path,
) -> None:
    config_path = admission_cli.DEFAULT_CONFIG
    labels = pd.DataFrame(
        {
            "query_id": [f"q{index:04d}" for index in range(513)],
            "fold": [2 + (index % 3) for index in range(513)],
        }
    )
    sample = labels.assign(
        _sample_key=labels["query_id"].map(
            lambda value: hashlib.sha256(value.encode()).hexdigest()
        )
    ).sort_values(["_sample_key", "query_id"]).head(512)[["query_id", "fold"]]
    sample_path = tmp_path / "runtime_sample.parquet"
    sample.to_parquet(sample_path, index=False)
    label_manifest_path = tmp_path / "label-ledger-manifest.json"
    label_manifest_path.write_text(
        json.dumps(
            {"runtime_query_sample_sha256": admission_cli.sha256_file(sample_path)}
        ),
        encoding="utf-8",
    )
    parity_rows = sample.copy()
    parity_rows["reference_sha256"] = parity_rows["query_id"].map(
        lambda value: hashlib.sha256(f"selected:{value}".encode()).hexdigest()
    )
    parity_rows["replay_sha256"] = parity_rows["reference_sha256"]
    parity_rows["byte_identical"] = True
    parity_path = tmp_path / "runtime_parity.parquet"
    parity_rows.to_parquet(parity_path, index=False)
    parity_digest = admission_cli._parity_aggregate_digest(
        parity_rows, "reference_sha256"
    )
    cascade_manifests: dict[int, Path] = {}
    candidate_paths: dict[int, Path] = {}
    model_paths: dict[int, Path] = {}
    selected_paths: dict[int, Path] = {}
    for fold in (2, 3, 4):
        for paths, stem in (
            (cascade_manifests, "cascade"),
            (candidate_paths, "master"),
            (model_paths, "model"),
            (selected_paths, "selected"),
        ):
            artifact = tmp_path / f"{stem}-{fold}.bin"
            artifact.write_bytes(f"{stem}-{fold}".encode())
            paths[fold] = artifact
    config_value = json.loads(config_path.read_text(encoding="utf-8"))
    contract_path = admission_cli.ROOT / config_value["contract_path"]
    profile = {
        "schema_version": admission_cli._RUNTIME_PROFILE_SCHEMA,
        "run_kind": "WARM_END_TO_END_OOF_REPLAY",
        "config_sha256": admission_cli.sha256_file(config_path),
        "contract_sha256": admission_cli.sha256_file(contract_path),
        "code_sha256": {
            "admission_cli": admission_cli.sha256_file(Path(admission_cli.__file__)),
            "admission_module": admission_cli.sha256_file(
                admission_cli.ROOT
                / "src/xgb_matcher/retrieval_hierarchical_admission_v12.py"
            ),
            "cascade_module": admission_cli.sha256_file(
                admission_cli.ROOT / "src/xgb_matcher/retrieval_top100_cascade_v12.py"
            ),
            "runtime_profiler": admission_cli.sha256_file(
                admission_cli.ROOT / "scripts/profile_retrieval_all_human_v12.py"
            ),
        },
        "folds": [2, 3, 4],
        "fold0_opened": False,
        "fold1_opened": False,
        "positive_injection": False,
        "gate_eligible": True,
        "lineage": {
            "outer_cascades": {
                str(fold): {
                    "manifest_sha256": admission_cli.sha256_file(
                        cascade_manifests[fold]
                    ),
                    "master_sha256": admission_cli.sha256_file(candidate_paths[fold]),
                }
                for fold in (2, 3, 4)
            },
            "site_models": {
                str(fold): {
                    "site_model_sha256": admission_cli.sha256_file(model_paths[fold])
                }
                for fold in (2, 3, 4)
            },
            "selected_outputs": {
                str(fold): {
                    "selected_output_sha256": admission_cli.sha256_file(
                        selected_paths[fold]
                    )
                }
                for fold in (2, 3, 4)
            },
        },
        "producer_lineage": {
            "query_source_sha256": "1" * 64,
            "query_source_manifest_sha256": "2" * 64,
            "query_sample_source_sha256": admission_cli.sha256_file(sample_path),
            "label_ledger_manifest_sha256": admission_cli.sha256_file(
                label_manifest_path
            ),
            "v11_config_sha256": "3" * 64,
            "official_config_sha256": "4" * 64,
            "v11_candidate_manifest_sha256": "5" * 64,
            "sidecar_manifest_sha256": "6" * 64,
            "nested_parent_bundle_sha256": "7" * 64,
            "index_manifest_sha256": {
                "base": ["8" * 64],
                "overlay": [],
                "temporal_overlay": [],
                "legal_entity_role": [],
                "sector_official": [],
            },
            "code_sha256": {
                "profiler": admission_cli.sha256_file(
                    admission_cli.ROOT / "scripts/profile_retrieval_all_human_v12.py"
                ),
                "query_projection_builder": admission_cli.sha256_file(
                    admission_cli.ROOT
                    / "scripts/build_retrieval_all_human_v12_runtime_queries.py"
                ),
                "v11_retriever": admission_cli.sha256_file(
                    admission_cli.ROOT / "src/xgb_matcher/retrieval_all_human_v11.py"
                ),
                "official_retriever": admission_cli.sha256_file(
                    admission_cli.ROOT
                    / "src/xgb_matcher/official_evidence_retrieval.py"
                ),
                "cascade_runner": admission_cli.sha256_file(
                    admission_cli.ROOT / "scripts/run_retrieval_top100_cascade_v12.py"
                ),
            },
        },
        "query_sample": {
            "path": sample_path.name,
            "sha256": admission_cli.sha256_file(sample_path),
            "rows": 512,
            "fold_counts": {
                str(int(key)): int(value)
                for key, value in sample["fold"].value_counts().sort_index().items()
            },
            "selection_rule": admission_cli._RUNTIME_SAMPLE_RULE,
            "labels_present": False,
        },
        "hardware": {
            "computer": "Mac",
            "chip": "Apple M4 Pro",
            "os": "macOS-test",
            "python": "3.test",
            "xgboost": "test",
        },
        "protocol": {
            "cache_state": "WARM",
            "warm": True,
            "warmup_queries": 1,
            "repetitions": 1,
            "concurrency": 1,
            "end_to_end": True,
            "measured_queries": 512,
            "missing_queries": 0,
            "stages": [
                "candidate_generation",
                "cascade",
                "canonicalization",
                "feature_computation",
                "site_lambdamart",
                "top100_selection",
            ],
        },
        "parity": {
            "path": parity_path.name,
            "sha256": admission_cli.sha256_file(parity_path),
            "columns": ["query_id", "candidate_siret", "selected_rank"],
            "rows": 512,
            "candidates_and_order_byte_identical": True,
            "reference_sha256": parity_digest,
            "replay_sha256": parity_digest,
        },
        "latency_ms": {"p50": 100.0, "p95": 500.0, "p99": 900.0},
    }
    profile_path = tmp_path / "runtime-profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    loaded = admission_cli._validate_runtime_profile_manifest(
        profile_path,
        config_path=config_path,
        labels=labels,
        label_manifest_path=label_manifest_path,
        sealed_lineage=profile["lineage"],
    )
    assert loaded["latency_ms"]["p99"] == 900.0
    omitted = labels.loc[~labels["query_id"].isin(sample["query_id"])].iloc[[0]]
    forged_sample = pd.concat([sample.iloc[:-1], omitted], ignore_index=True)
    forged_sample.to_parquet(sample_path, index=False)
    profile["query_sample"]["sha256"] = admission_cli.sha256_file(sample_path)
    profile["query_sample"]["fold_counts"] = {
        str(int(key)): int(value)
        for key, value in forged_sample["fold"].value_counts().sort_index().items()
    }
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    with pytest.raises(ValueError, match="deterministic rule"):
        admission_cli._validate_runtime_profile_manifest(
            profile_path,
            config_path=config_path,
            labels=labels,
            label_manifest_path=label_manifest_path,
            sealed_lineage=profile["lineage"],
        )
    profile["query_sample"]["rows"] = 511
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    with pytest.raises(ValueError, match="query sample is not sealed"):
        admission_cli._validate_runtime_profile_manifest(
            profile_path,
            config_path=config_path,
            labels=labels,
            label_manifest_path=label_manifest_path,
            sealed_lineage=profile["lineage"],
        )


def test_apply_runtime_profile_updates_only_decision_without_refit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tiny_config()
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    labels_path = tmp_path / "labels.parquet"
    pd.DataFrame({"query_id": ["q2"], "fold": [2]}).to_parquet(
        labels_path, index=False
    )
    labels_manifest = tmp_path / "labels-manifest.json"
    labels_manifest.write_text("{}", encoding="utf-8")
    oof_dir = tmp_path / "oof"
    oof_dir.mkdir()
    summary = {
        "latency": {"p95_ms": 1.0, "p99_ms": 2.0},
        "gates": {
            "prerank_at_500": {"passed": True},
            "selected_recall_at_100": {"passed": True},
            "wilson95_lower_at_100": {"passed": True},
            "worst_fold_at_100": {"passed": True},
            "latency_p95_ms": {"passed": False},
            "latency_p99_ms": {"passed": False},
            "official_catalog_loss_taxonomy": {"passed": True},
        },
        "verdict": "PIVOT_LATENCY",
    }
    (oof_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    required_names = ["query_outcomes.parquet"] + [
        name
        for fold in (2, 3, 4)
        for name in (
            f"fold{fold}_site_ranker.json",
            f"fold{fold}_selected_candidates.parquet",
        )
    ]
    for name in required_names:
        (oof_dir / name).write_bytes(name.encode())
    outputs = {
        path.name: admission_cli.sha256_file(path)
        for path in oof_dir.iterdir()
        if path.is_file()
    }
    outer = {
        str(fold): {
            "master_sha256": f"{fold}" * 64,
            "parent_candidates_sha256": "b" * 64,
            "manifest_sha256": "c" * 64,
        }
        for fold in (2, 3, 4)
    }
    oof_manifest = {
        "schema_version": admission_cli.SCHEMA_VERSION,
        "run_kind": "NESTED_OUTER_OOF",
        "config_sha256": admission_cli.sha256_file(config_path),
        "positive_injection": False,
        "fold0_opened": False,
        "fold1_opened": False,
        "opened_folds": [2, 3, 4],
        "source_labels_sha256": admission_cli.sha256_file(labels_path),
        "source_labels_manifest_sha256": admission_cli.sha256_file(labels_manifest),
        "outputs": outputs,
        "outer_cascades": outer,
    }
    oof_manifest_path = oof_dir / "manifest.json"
    oof_manifest_path.write_text(json.dumps(oof_manifest), encoding="utf-8")
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(admission_cli.AdmissionConfigV12, "load", lambda _path: config)
    monkeypatch.setattr(
        admission_cli,
        "_load_sealed_query_labels",
        lambda _path, **_kwargs: (
            pd.DataFrame({"query_id": ["q2"], "fold": [2]}),
            labels_manifest,
        ),
    )

    def fake_profile(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {
            "schema_version": admission_cli._RUNTIME_PROFILE_SCHEMA,
            "lineage": {
                "oof_manifest_sha256": admission_cli.sha256_file(oof_manifest_path),
                "query_outcomes_sha256": outputs["query_outcomes.parquet"],
            },
            "latency_ms": {"p50": 100.0, "p95": 500.0, "p99": 900.0},
            "hardware": {"chip": "Apple M4 Pro"},
            "protocol": {"cache_state": "WARM"},
        }

    monkeypatch.setattr(
        admission_cli, "_validate_runtime_profile_manifest", fake_profile
    )
    monkeypatch.setattr(
        admission_cli,
        "fit_one_outer_site_model",
        lambda *_args, **_kwargs: pytest.fail("runtime application must not refit"),
    )
    output_dir = tmp_path / "decision"
    admission_cli.run_apply_runtime_profile(
        argparse.Namespace(
            oof_dir=oof_dir,
            labels=labels_path,
            runtime_profile_manifest=runtime_path,
            output_dir=output_dir,
            config=config_path,
        )
    )
    decision = json.loads((output_dir / "manifest.json").read_text())
    updated = json.loads((output_dir / "summary.json").read_text())
    assert decision["model_refit"] is False
    assert decision["candidate_rescore"] is False
    assert updated["verdict"] == "READY_FOR_FOLD1"
