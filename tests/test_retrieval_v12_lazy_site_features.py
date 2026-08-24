from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.xgb_matcher.retrieval_hierarchical_admission_v12 import (
    AdmissionConfigV12,
    CASCADE_PARENT_FEATURES,
    canonicalize_union,
    sample_precomputed_candidates,
    site_feature_names,
)
from src.xgb_matcher.retrieval_v12_lazy_site_features import (
    FoldFusedLazySiteFeatureAdapterV12,
    HumanSiteLabelV12,
    LazySiteFeatureError,
    compute_lazy_site_statistics,
    materialize_labelled_master_training_sample,
    materialize_training_sample,
)


def _siret(parent: int, site: int) -> str:
    return f"{parent:09d}{site:05d}"


def _master_fixture() -> pd.DataFrame:
    """Varied >128-row unions exercising every complete-population aggregate."""

    rows: list[dict[str, object]] = []
    query_specs = (("q-fold2", 2, 173, 161), ("q-fold3", 3, 149, 137))
    for query_offset, (query_id, fold, count, truth_index) in enumerate(query_specs):
        parent_base = 310_000_000 + query_offset * 10_000
        truth_parent = parent_base + truth_index // 4
        truth = _siret(truth_parent, truth_index % 4 + 1)
        operational_sibling = _siret(truth_parent, 99)
        for index in range(count):
            parent_index = index // 4
            parent = parent_base + parent_index
            site = index % 4 + 1
            candidate = _siret(parent, site)
            channels: list[str] = []
            if index % 11 == 0 or index == truth_index:
                channels.append("legal_name_exact")
            if index % 3 == 0:
                channels.append("name_word")
            if index % 7 == 0 or index == truth_index:
                channels.append("address_exact")
            if index % 5 == 0:
                channels.append("current_name_char")
            if index % 13 == 0:
                channels.append("rne_name")
            if index % 17 == 0:
                channels.append("bodacc_address")
            if index % 19 == 0:
                channels.append("historical_name_word")
            if index % 23 == 0:
                channels.append("geo_exact_cell")
            if index % 29 == 0:
                channels.append("official_successor")
            if index % 31 == 0:
                channels.append("v12_parent_local_expansion")

            same_name = index == truth_index or index % 11 == 0
            same_address = index == truth_index or index % 7 == 0
            row: dict[str, object] = {
                "query_id": query_id,
                "fold": fold,
                "human_ground_truth_siret": truth,
                "acceptable_sirets_operational_json": json.dumps(
                    [operational_sibling, truth]
                ),
                "master_rank": index + 1,
                # Only part of Master belongs to the heuristic PreRank.  The
                # positive deliberately lives outside that shortlist.
                "hydration_rank": index + 1 if index < 111 else np.nan,
                "candidate_siret": candidate,
                "candidate_parent": f"{parent:09d}",
                "crm_name": f"Atelier principal {query_offset}",
                "crm_address": "17 avenue des Tests 75001 Paris",
                "crm_number": "17",
                "crm_postcode": "75001",
                "crm_insee": "75056",
                "candidate_names": [
                    f"Atelier principal {query_offset}"
                    if same_name
                    else f"Concurrent {parent_index} site {site}",
                    f"Nom historique {parent_index}" if index % 19 == 0 else "",
                ],
                "candidate_addresses": [
                    "17 avenue des Tests 75001 Paris"
                    if same_address
                    else f"{20 + index} rue secondaire 75001 Paris"
                ],
                "candidate_number": "17" if same_address else str(20 + index),
                "candidate_postcode": "75001" if index % 9 else "75002",
                "candidate_insee": "75056" if index % 10 else "75101",
                "candidate_state": "F" if index % 8 == 0 else "A",
                "candidate_is_siege": index % 4 == 0,
                "channels": channels,
                "official_evidence_sources": [
                    "SIRENE",
                    *( ["RNE"] if index % 13 == 0 else [] ),
                    *( ["BODACC"] if index % 17 == 0 else [] ),
                ],
                "official_evidence_ids": [f"preuve-{query_offset}-{index}"],
                "signals": [
                    {
                        "channel": "official_successor",
                        "rank": 1,
                        "score": 0.75,
                        "source": "SIRENE",
                        "relation_direction": "SUCCESSOR",
                        "relation_date": "2021-03-14",
                    }
                ]
                if index % 29 == 0
                else [],
                "temporal_valid_from": "2012-06-01" if index % 2 else "",
                "temporal_valid_to": "2020-04-03" if index % 8 == 0 else "",
                "parent_rank": parent_index + 1,
                # Wide range plus ties across queries, with a stable score per
                # query/parent as required by the sealed cascade contract.
                "parent_score": float((parent_index % 9) - 12.5),
                "parent_feature_names": list(CASCADE_PARENT_FEATURES),
                "parent_feature_values": [
                    float(parent_index % (feature_index + 2)) / (feature_index + 1)
                    for feature_index, _name in enumerate(CASCADE_PARENT_FEATURES)
                ],
                "retrieval_latency_ms": float(20 + query_offset),
                "ground_truth_state": "FERME" if query_offset else "ACTIF",
                "data_origin": "CRM_NEW" if query_offset else "CRM_OLD",
                "seen_unseen_siren": "UNSEEN" if query_offset else "SEEN",
                "known_new_site": "NEW_SITE" if query_offset else "KNOWN_SITE",
                "megapolis": "YES",
                "qualification": "AMBIGUOUS" if query_offset else "MATCH_EXACT",
                "qualification_v2": "V2-A",
                "qualification_v3": "V3-B",
                "official_catalog_present": True,
            }
            # Ranks contain ties and scores contain both ties and negatives so
            # average percentiles/max-relative values are actually exercised.
            for channel_position, channel in enumerate(channels):
                row[f"{channel}_rank"] = float(index % (channel_position + 5) + 1)
                row[f"{channel}_score"] = float((index % 7) - 3) / 2.0
            if index % 37 == 0:
                # This channel deliberately exists only in the structured
                # signal payload.  The canonical parser accepts both physical
                # encodings, so a lazy summary may not inspect just the wide
                # ``<channel>_rank`` columns.
                row["signals"] = [
                    *list(row["signals"]),
                    {
                        "channel": "bodacc_relation",
                        "rank": float(index % 4 + 1),
                        "score": float((index % 5) - 2),
                        "source": "BODACC",
                        "relation_direction": "PREDECESSOR",
                        "event_date": "2019-11-02",
                    },
                ]
            rows.append(row)
    frame = pd.DataFrame(rows)
    # The canonical implementation computes parent softmax before its final
    # stable sort.  A shuffle ensures the lazy implementation preserves that
    # exact encounter-order behavior as well.
    return frame.sample(frac=1.0, random_state=90210).reset_index(drop=True)


def _reference_fit_projection(
    raw: pd.DataFrame, config: AdmissionConfigV12
) -> pd.DataFrame:
    complete = canonicalize_union(
        raw, config, labels_allowed=True, require_labels=True
    )
    sampled = sample_precomputed_candidates(complete, config)
    features = list(site_feature_names(config))
    fit_columns = list(
        dict.fromkeys(
            [
                "query_id",
                "fold",
                "human_siret",
                "operational_sirets_json",
                "candidate_siret",
                "candidate_parent",
                "heuristic_hydration_rank",
                "features_computed_on_complete_union",
                "sealed_parent_features_present",
                "sealed_parent_rank",
                "sealed_parent_score",
                "parent_preexpansion_missing",
                *features,
            ]
        )
    )
    sampled = sampled[fit_columns].copy()
    sampled[features] = sampled[features].fillna(0.0).astype(np.float32)
    return sampled.reset_index(drop=True)


def _labels(raw: pd.DataFrame) -> list[HumanSiteLabelV12]:
    output: list[HumanSiteLabelV12] = []
    for query_id, group in raw.groupby("query_id", sort=True):
        output.append(
            HumanSiteLabelV12(
                query_id=str(query_id),
                human_siret=str(group.iloc[0]["human_ground_truth_siret"]),
                operational_sirets=tuple(
                    json.loads(
                        str(group.iloc[0]["acceptable_sirets_operational_json"])
                    )
                ),
            )
        )
    return output


def test_lazy_labelled_wrapper_is_exactly_equal_to_full_canonical_chain() -> None:
    config = AdmissionConfigV12.default()
    raw = _master_fixture()
    expected = _reference_fit_projection(raw, config)
    result = materialize_labelled_master_training_sample(raw, config)

    assert len(result.feature_names) == 319
    assert result.complete_rows == len(raw)
    assert result.selected_rows == 256
    assert result.maximum_selected_per_query == 128
    assert result.materialized_feature_cells == 256 * 319
    assert result.avoided_dense_feature_cells == (len(raw) - 256) * 319
    pd.testing.assert_frame_equal(result.frame, expected, check_exact=True)


def test_split_label_blind_adapter_matches_reference_and_keeps_positive() -> None:
    config = AdmissionConfigV12.default()
    raw = _master_fixture()
    blind = raw.drop(
        columns=[
            "human_ground_truth_siret",
            "acceptable_sirets_operational_json",
        ]
    )
    adapter = FoldFusedLazySiteFeatureAdapterV12(config)
    statistics = adapter.compute_full_master_statistics(blind)
    result = adapter.materialize_training_sample(statistics, _labels(raw))
    expected = _reference_fit_projection(raw, config)

    assert statistics.dense_complete_feature_cells == 0
    assert statistics.complete_rows == len(raw)
    assert statistics.folds == (2, 3)
    assert "name_char_dice" not in statistics._base
    assert "temporal_valid_at_reference" not in statistics._base
    assert statistics._base["__raw_rank__bodacc_relation"].notna().any()
    assert len(statistics.statistics_sha256) == 64
    assert len(statistics.candidate_order_sha256) == 64
    assert result.source_statistics_sha256 == statistics.statistics_sha256
    assert result.frame.groupby("query_id").size().eq(128).all()
    assert (
        result.frame["candidate_siret"]
        .eq(result.frame["human_siret"])
        .groupby(result.frame["query_id"])
        .any()
        .all()
    )
    pd.testing.assert_frame_equal(result.frame, expected, check_exact=True)


def test_absent_human_label_is_not_injected_and_still_matches_reference() -> None:
    config = AdmissionConfigV12.default()
    raw = _master_fixture()
    absent_by_query = {
        "q-fold2": "99800000100001",
        "q-fold3": "99800000200001",
    }
    for query_id, absent in absent_by_query.items():
        mask = raw["query_id"].eq(query_id)
        raw.loc[mask, "human_ground_truth_siret"] = absent
        raw.loc[mask, "acceptable_sirets_operational_json"] = json.dumps([absent])
    expected = _reference_fit_projection(raw, config)
    result = materialize_labelled_master_training_sample(raw, config)
    assert not result.frame["candidate_siret"].eq(result.frame["human_siret"]).any()
    assert len(result.frame) == 256
    pd.testing.assert_frame_equal(result.frame, expected, check_exact=True)


def test_small_union_materializes_all_rows_with_complete_population_values() -> None:
    config = AdmissionConfigV12.default()
    raw = _master_fixture()
    raw = pd.concat(
        [group.nsmallest(31, "master_rank") for _, group in raw.groupby("query_id")],
        ignore_index=True,
    )
    # Ensure each now-small fixture has a naturally present truth.
    for query_id, group in raw.groupby("query_id"):
        truth = str(group.iloc[0]["human_ground_truth_siret"])
        first = group.index[0]
        raw.loc[first, "human_ground_truth_siret"] = str(raw.loc[first, "candidate_siret"])
        raw.loc[group.index, "human_ground_truth_siret"] = str(
            raw.loc[first, "candidate_siret"]
        )
        raw.loc[group.index, "acceptable_sirets_operational_json"] = json.dumps(
            [str(raw.loc[first, "candidate_siret"])]
        )
        assert truth != ""
    expected = _reference_fit_projection(raw, config)
    result = materialize_labelled_master_training_sample(raw, config)
    assert result.selected_rows == len(raw)
    assert result.avoided_dense_feature_cells == 0
    pd.testing.assert_frame_equal(result.frame, expected, check_exact=True)


def test_lazy_path_rejects_labels_in_statistics_and_forbidden_folds() -> None:
    config = AdmissionConfigV12.default()
    raw = _master_fixture()
    with pytest.raises(ValueError, match="forbidden label columns"):
        compute_lazy_site_statistics(raw, config)

    blind = raw.drop(
        columns=[
            "human_ground_truth_siret",
            "acceptable_sirets_operational_json",
        ]
    )
    blind["fold"] = 1
    with pytest.raises(LazySiteFeatureError, match="refuse fold 0/1"):
        compute_lazy_site_statistics(blind, config)


def test_lazy_path_rejects_label_query_mismatch_and_parent_drift() -> None:
    config = AdmissionConfigV12.default()
    raw = _master_fixture()
    blind = raw.drop(
        columns=[
            "human_ground_truth_siret",
            "acceptable_sirets_operational_json",
        ]
    )
    statistics = compute_lazy_site_statistics(blind, config)
    with pytest.raises(LazySiteFeatureError, match="label/Master query mismatch"):
        materialize_training_sample(statistics, _labels(raw)[:1], config)

    drift = blind.copy()
    parent = str(drift.iloc[0]["candidate_parent"])
    query = str(drift.iloc[0]["query_id"])
    same_parent = drift.index[
        drift["candidate_parent"].astype(str).eq(parent)
        & drift["query_id"].astype(str).eq(query)
    ]
    if len(same_parent) < 2:
        pytest.skip("fixture shuffle unexpectedly produced no repeated parent")
    drift.at[same_parent[1], "parent_score"] = float(
        drift.at[same_parent[1], "parent_score"]
    ) + 1.0
    with pytest.raises(LazySiteFeatureError, match="sealed parent field varies"):
        compute_lazy_site_statistics(drift, config)
