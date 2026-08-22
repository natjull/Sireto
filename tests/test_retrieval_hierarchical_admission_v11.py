from __future__ import annotations

from copy import deepcopy
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.xgb_matcher.retrieval_hierarchical_admission_v11 as v11
from scripts.run_retrieval_hierarchical_admission_v11 import (
    _fit_sample,
    run_oracle,
    run_oof,
    sha256_file,
)


def tiny_config() -> v11.HierarchicalAdmissionConfig:
    raw = deepcopy(v11.HierarchicalAdmissionConfig.default().raw)
    for name in ("parent", "siret"):
        raw["admission"]["xgboost"][name].update(
            {
                "n_estimators": 4,
                "learning_rate": 0.2,
                "max_depth": 2,
                "min_child_weight": 1.0,
                "reg_lambda": 1.0,
                "n_jobs": 1,
                "lambdarank_num_pair_per_sample": 4,
            }
        )
    raw["gates"] = {
        "minimum_oof_oracle_recall_at_2000": 0.0,
        "minimum_oof_exact_recall_at_100": 0.0,
        "minimum_oof_exact_recall_at_100_wilson95_lower": 0.0,
        "minimum_each_outer_fold_exact_recall_at_100": 0.0,
        "maximum_warm_p95_ms": 1000,
        "maximum_warm_p99_ms": 2000,
    }
    return v11.HierarchicalAdmissionConfig.from_mapping(raw)


def _siret(parent: int, nic: int) -> str:
    return f"{parent:09d}{nic:05d}"


def make_union(*, queries_per_fold: int = 2, candidates_per_query: int = 7) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold in (2, 3, 4):
        for query_index in range(queries_per_fold):
            truth_parent = 100_000_000 + fold * 10_000 + query_index * 100
            truth = _siret(truth_parent, 1)
            query_id = f"f{fold}-q{query_index}"
            for candidate_index in range(candidates_per_query):
                if candidate_index < 2:
                    parent = truth_parent
                    nic = candidate_index + 1
                    site_key = "truth site"
                else:
                    parent = truth_parent + candidate_index
                    nic = 1
                    site_key = f"other site {candidate_index}"
                candidate = _siret(parent, nic)
                is_truth = candidate == truth
                channels = (
                    ["legal_name_exact", "address_word", "current_name_char"]
                    if is_truth
                    else ["address_word", "current_name_char"]
                    if candidate_index == 1
                    else ["current_name_char"]
                )
                rows.append(
                    {
                        "query_id": query_id,
                        "fold": fold,
                        "historical_ground_truth_siret": truth,
                        "acceptable_sirets_operational_json": (
                            f'["{truth}","{_siret(truth_parent, 2)}"]'
                        ),
                        "union_rank": candidate_index + 1,
                        "siret": candidate,
                        "siren": f"{parent:09d}",
                        "crm_name": f"Entreprise {truth_parent}",
                        "crm_address": "10 rue de la Paix 75001 Paris",
                        "crm_number": "10",
                        "crm_postcode": "75001",
                        "crm_insee": "75056",
                        "candidate_names": [
                            f"Entreprise {truth_parent}" if is_truth else f"Concurrent {candidate_index}"
                        ],
                        "candidate_addresses": [
                            "10 rue de la Paix 75001 Paris"
                            if candidate_index < 2
                            else f"{candidate_index} avenue Autre 75001 Paris"
                        ],
                        "candidate_number": "10" if candidate_index < 2 else str(candidate_index),
                        "candidate_postcode": "75001",
                        "candidate_insee": "75056",
                        "candidate_site_key": site_key,
                        "candidate_state": "A" if candidate_index != 1 else "F",
                        "candidate_is_siege": is_truth,
                        "channels": channels,
                        "retrieval_rank": candidate_index + 1,
                        "retrieval_score": float(candidates_per_query - candidate_index),
                        "official_evidence_sources": ["SIRENE"],
                        "retrieval_latency_ms": 10.0,
                    }
                )
    return pd.DataFrame(rows)


def test_contract_has_identifier_free_features_and_fixed_limits() -> None:
    config = tiny_config()
    v11.assert_identifier_free_features(v11.PARENT_FEATURES)
    v11.assert_identifier_free_features(v11.SIRET_FEATURES)
    assert config.train_folds == (2, 3, 4)
    assert config.internal_union_cap == 2000
    assert config.max_candidates == 100
    assert (config.anchor_slots, config.consensus_slots) == (20, 20)
    assert config.max_unprotected_per_parent == 3


def test_fit_sampling_retains_only_a_natural_positive_and_never_samples_validation() -> None:
    config = tiny_config()
    truth = "99999999900001"
    rows = pd.DataFrame(
        [
            {
                "query_id": "q",
                "union_rank": rank,
                "siret": truth if rank == 200 else f"{rank:09d}00001",
                "human_ground_truth_siret": truth,
                "name_word_rank": rank,
            }
            for rank in range(1, 201)
        ]
    )
    sampled = _fit_sample(rows, config)
    assert len(sampled) == 128
    assert truth in set(sampled["siret"])
    assert set(config.fit_candidate_sampling) == {
        "maximum_candidates_per_query",
        "union_head",
        "channel_head",
        "retain_natural_positive_if_present",
        "validation_sampling",
    }
    assert config.fit_candidate_sampling["validation_sampling"] is False


def test_streaming_oof_scores_full_candidate_groups(tmp_path: Path) -> None:
    config = tiny_config()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config.raw), encoding="utf-8")
    candidates = tmp_path / "candidates.parquet"
    make_union(queries_per_fold=2, candidates_per_query=7).sort_values(
        ["query_id", "union_rank"]
    ).to_parquet(candidates, index=False)
    candidates.with_suffix(".parquet.manifest.json").write_text(
        json.dumps(
            {
                "candidate_parquet_sha256": sha256_file(candidates),
                "positive_injection": False,
                "maximum_union_candidates": 7,
                "labels_attached_after_retrieval": True,
                "all_label_queries_have_natural_candidates": True,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "oof"
    run_oof(
        argparse.Namespace(
            candidates=candidates,
            output_dir=output,
            config=config_path,
            fit_query_batch_size=2,
            validation_query_batch_size=2,
        )
    )

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["query_count"] == 6
    assert summary["nested_oof"]["stream_validation"][
        "maximum_validation_union"
    ] == 7
    selected = pd.read_parquet(output / "selected_candidates.parquet")
    assert selected.groupby("query_id").size().eq(7).all()


def test_oracle_gate_runs_before_any_admission_fit(tmp_path: Path) -> None:
    config = tiny_config()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config.raw), encoding="utf-8")
    candidates = tmp_path / "candidates.parquet"
    make_union(queries_per_fold=1, candidates_per_query=7).sort_values(
        ["query_id", "union_rank"]
    ).to_parquet(candidates, index=False)
    candidates.with_suffix(".parquet.manifest.json").write_text(
        json.dumps(
            {
                "candidate_parquet_sha256": sha256_file(candidates),
                "positive_injection": False,
                "maximum_union_candidates": 7,
                "labels_attached_after_retrieval": True,
                "all_label_queries_have_natural_candidates": True,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "oracle"
    run_oracle(
        argparse.Namespace(
            candidates=candidates,
            output_dir=output,
            config=config_path,
            query_batch_size=2,
        )
    )

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["run_kind"] == "OOF_ORACLE_BEFORE_ADMISSION"
    assert summary["admission_model_fit"] is False
    assert summary["query_count"] == 3
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["fold0_opened"] is False
    assert manifest["fold1_opened"] is False


def test_label_blind_preparation_rejects_human_labels() -> None:
    config = tiny_config()
    labelled = make_union(queries_per_fold=1)
    with pytest.raises(ValueError, match="cannot receive human labels"):
        v11.canonicalize_union(labelled, config, labels_allowed=False)

    unlabeled = labelled.drop(columns=["historical_ground_truth_siret"])
    prepared = v11.canonicalize_union(unlabeled, config, labels_allowed=False)
    assert prepared["human_siret"].eq("").all()
    assert len(prepared) == len(unlabeled)
    assert prepared[["query_id", "candidate_siret"]].duplicated().sum() == 0


def test_address_exact_alone_is_never_a_protected_anchor() -> None:
    config = tiny_config()
    frame = make_union(queries_per_fold=1)
    frame = frame.drop(columns=["historical_ground_truth_siret"])
    first = frame.index[0]
    frame.at[first, "channels"] = ["address_exact"]
    prepared = v11.canonicalize_union(frame, config, labels_allowed=False)
    candidate = prepared[
        (prepared["query_id"].eq(frame.at[first, "query_id"]))
        & (prepared["candidate_siret"].eq(frame.at[first, "siret"]))
    ].iloc[0]
    assert candidate["is_anchor"] == 0


def test_same_parent_same_site_sibling_is_not_an_exact_negative() -> None:
    config = tiny_config()
    raw = make_union(queries_per_fold=1)
    prepared = v11.canonicalize_union(raw, config, labels_allowed=True, require_labels=True)
    one_query = prepared[prepared["query_id"].eq("f2-q0")]
    training, diagnostics = v11._site_training_rows(one_query)
    truth = one_query.iloc[0]["human_siret"]
    sibling = one_query[
        one_query["candidate_parent"].eq(str(truth)[:9])
        & ~one_query["candidate_siret"].eq(truth)
    ].iloc[0]["candidate_siret"]
    assert sibling not in set(training["candidate_siret"])
    assert diagnostics["same_parent_same_site_negatives_excluded"] == 1
    assert int(training["siret_label"].sum()) == 1


def test_portfolio_selection_has_fixed_lanes_and_strict_ceiling() -> None:
    config = tiny_config()
    records: list[dict[str, object]] = []
    for index in range(150):
        parent = f"{100_000_000 + index // 3:09d}"
        records.append(
            {
                "query_id": "q",
                "candidate_siret": parent + f"{index % 3 + 1:05d}",
                "candidate_parent": parent,
                "is_anchor": int(index < 25),
                "is_consensus": int(25 <= index < 60),
                "siret_score": float(150 - index),
                "_parent_model_score": float(150 - index // 3),
                "union_rank": index + 1,
            }
        )
    selected = v11.select_top100(pd.DataFrame(records), config)
    assert len(selected) == 100
    assert int(selected["selected_rank"].max()) == 100
    assert int(selected["selection_reason"].eq("anchor").sum()) == 20
    consensus = selected[selected["selection_reason"].eq("consensus")]
    assert len(consensus) == 12
    assert consensus["candidate_parent"].is_unique
    diverse = selected[selected["selection_reason"].eq("ranked_parent_diverse")]
    assert int(diverse.groupby("candidate_parent").size().max()) <= 3


def test_all_human_metric_keeps_oracle_misses_in_denominator() -> None:
    config = tiny_config()
    raw = make_union(queries_per_fold=1)
    # Remove one natural truth without adding any replacement.
    query = "f4-q0"
    human = raw.loc[raw["query_id"].eq(query), "historical_ground_truth_siret"].iloc[0]
    raw = raw[~(raw["query_id"].eq(query) & raw["siret"].eq(human))].copy()
    candidates = v11.canonicalize_union(raw, config, labels_allowed=True, require_labels=True)
    selected = candidates.copy()
    selected["selected_rank"] = selected.groupby("query_id").cumcount() + 1
    outcomes, summary = v11.evaluate_all_human(candidates, selected, config)
    assert len(outcomes) == 3
    assert summary["metrics"]["exact_oracle_at_2000"]["total"] == 3
    assert summary["metrics"]["exact_oracle_at_2000"]["successes"] == 2
    assert summary["metrics"]["exact_recall_at_100"]["successes"] == 2


def test_nested_outer_oof_scores_complete_unions_without_injection() -> None:
    config = tiny_config()
    raw = make_union(queries_per_fold=2, candidates_per_query=7)
    selected, outcomes, summary = v11.outer_oof_evaluate(raw, config)
    assert len(outcomes) == 6
    assert outcomes["exact_oracle_hit"].all()
    assert outcomes["exact_hit_at_100"].all()
    assert selected.groupby("query_id").size().eq(7).all()
    assert selected["human_siret"].eq("").all()
    assert {
        "selected_rank",
        "final_score",
        "parent_rank",
        "site_rank",
        "selection_reason",
        "channels_json",
        "official_evidence_sources_json",
    }.issubset(selected.columns)
    assert summary["nested_oof"]["validation_candidate_sampling"] is False
    assert summary["nested_oof"]["fold0_opened"] is False
    assert summary["nested_oof"]["fold1_opened"] is False
    assert summary["integrity"]["positive_injection"] is False
    assert summary["population"] == "ALL_USABLE_HUMAN_LABELS"
    assert set(summary["outer_folds"]) == {"2", "3", "4"}
    assert set(summary["oracle_explanatory"]) == {
        "baseline_current",
        "parent_complete",
        "parent_plus_geography",
        "official_stack_complete",
    }
    assert summary["latency_ms"] == {"p95": 10.0, "p99": 10.0}


def test_outer_oof_excludes_only_queries_without_any_usable_human_label() -> None:
    config = tiny_config()
    raw = make_union(queries_per_fold=2, candidates_per_query=7)
    raw.loc[raw["query_id"].eq("f2-q1"), "historical_ground_truth_siret"] = ""
    _selected, outcomes, summary = v11.outer_oof_evaluate(raw, config)
    assert len(outcomes) == 5
    assert summary["nested_oof"]["input_query_count"] == 6
    assert summary["nested_oof"]["usable_human_query_count"] == 5
    assert summary["nested_oof"]["unusable_human_queries_excluded"] == 1


def test_no_identifier_value_can_change_label_blind_features() -> None:
    config = tiny_config()
    raw = make_union(queries_per_fold=1).drop(columns=["historical_ground_truth_siret"])
    first = v11.canonicalize_union(raw, config, labels_allowed=False)
    changed = raw.copy()
    changed["query_id"] = "renamed-" + changed["query_id"].astype(str)
    second = v11.canonicalize_union(changed, config, labels_allowed=False)
    numeric = list(v11.PARENT_FEATURES)  # parent values are compared after aggregation.
    parent_first = v11.build_parent_rows(first)[numeric].to_numpy(dtype=float)
    parent_second = v11.build_parent_rows(second)[numeric].to_numpy(dtype=float)
    np.testing.assert_allclose(parent_first, parent_second)
