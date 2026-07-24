from __future__ import annotations

import json

import pandas as pd

from scripts.audit_retrieval_channels import (
    ALL_CHANNELS,
    _current_sparse_indices,
    _siren_sibling_channels,
    summarize_channel_audit,
)


def test_current_sparse_padding_is_stable_below_trigger() -> None:
    candidates = [
        {"siret": "00000000000001"},
        {"siret": "00000000000002"},
    ]
    indices = _current_sparse_indices(
        candidates=candidates,
        crm_name="ALPHA",
        crm_address="1 RUE A",
        name_vectorizer=None,
        name_matrix=None,
        char_vectorizer=None,
        char_matrix=None,
        address_vectorizer=None,
        address_matrix=None,
        rescue_indices=[1],
        per_channel_k=500,
        budget=500,
        rrf_k=60,
        prefilter_trigger_size=50,
    )
    assert indices == [0, 1]


def test_channel_summary_counts_paired_recoveries() -> None:
    rows = []
    for query_index in range(3):
        row = {
            "query_id": str(query_index),
            "ground_truth_state": "A" if query_index < 2 else "F",
            "mega_base_pool": query_index == 2,
            "multi_site_siren": False,
            "location_match_type": "insee",
            "ground_truth_in_base": query_index != 2,
            "ground_truth_siren_in_base": True,
            "current_sparse_siren_rank": 1,
            "latency_ms": 1.0,
        }
        for channel in ALL_CHANNELS:
            row[f"{channel}_rank"] = None
            row[f"{channel}_count"] = 0
            row[f"{channel}_sirets_json"] = json.dumps([])
        rows.append(row)
    rows[0]["current_sparse_rank"] = 1
    rows[0]["current_sparse_count"] = 1
    rows[1]["name_word_rank"] = 80
    rows[1]["name_word_count"] = 100
    rows[2]["address_exact_rank"] = 1
    rows[2]["address_exact_count"] = 1

    summary = summarize_channel_audit(pd.DataFrame(rows), cutoffs=[50, 100])

    assert summary["channels"]["current_sparse"]["recall"]["100"]["successes"] == 1
    assert summary["channels"]["name_word"]["recall"]["100"]["successes"] == 1
    assert (
        summary["paired_at_100"]["name_word"]["recovers_current_sparse_misses"]
        == 1
    )
    assert (
        summary["diagnostic_oracle"]["100"]["any_individual_channel"][
            "successes"
        ]
        == 2
    )


def test_siren_channels_prioritize_exact_address_then_head_office() -> None:
    candidates = [
        {"siret": "11111111100001", "siren": "111111111", "is_siege": True},
        {"siret": "11111111100002", "siren": "111111111", "is_siege": False},
        {"siret": "22222222200001", "siren": "222222222", "is_siege": True},
    ]
    heads, sites = _siren_sibling_channels(
        candidates=candidates,
        word_indices=[0, 2],
        char_indices=[0, 2],
        current_indices=[0, 2],
        address_indices=[1],
        address_exact_indices=[1],
        max_output=10,
    )
    assert heads == [1, 2]
    assert sites == [1, 2, 0]
