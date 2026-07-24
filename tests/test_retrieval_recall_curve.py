from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.run_retrieval_recall_curve import (
    _loss_bucket,
    build_recall_curve,
    build_stage_audit,
)


def _row(**overrides):
    row = {
        "ground_truth_siret": "12345678900001",
        "ground_truth_siren": "123456789",
        "ground_truth_rank": 1,
        "candidate_sirets_json": json.dumps(
            ["12345678900001", "99999999900001"]
        ),
        "ground_truth_in_base": True,
        "ground_truth_in_filtered_pre_dedupe": True,
        "ground_truth_in_deduped": True,
        "ground_truth_state": "A",
        "mega_base_pool": False,
        "multi_site_siren": False,
        "location_match_type": "insee",
    }
    row.update(overrides)
    return row


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"ground_truth_in_base": False}, "PARTITION_MISS"),
        (
            {"ground_truth_in_filtered_pre_dedupe": False},
            "FILTER_MISS",
        ),
        ({"ground_truth_in_deduped": False}, "DEDUPE_MISS"),
        ({"ground_truth_rank": 1}, "HIT_AT_50"),
        (
            {"ground_truth_rank": 75},
            "PRUNED_AT_50_RECOVERED_BY_100",
        ),
        (
            {"ground_truth_rank": 150},
            "PRUNED_AT_100_RECOVERED_BY_200",
        ),
        (
            {"ground_truth_rank": 350},
            "PRUNED_AT_200_RECOVERED_BY_500",
        ),
        ({"ground_truth_rank": None}, "PRUNED_AT_500"),
    ],
)
def test_loss_bucket_is_mutually_exclusive(overrides, expected) -> None:
    assert _loss_bucket(pd.Series(_row(**overrides)), max_cutoff=500) == expected


def test_recall_curve_uses_prefixes_and_exact_siret() -> None:
    rows = pd.DataFrame(
        [
            _row(),
            _row(
                ground_truth_siret="22222222200002",
                ground_truth_siren="222222222",
                ground_truth_rank=2,
                candidate_sirets_json=json.dumps(
                    ["99999999900001", "22222222200002"]
                ),
            ),
        ]
    )

    curve = build_recall_curve(rows, cutoffs=[1, 2])

    assert curve["1"]["siret"]["rate"] == pytest.approx(0.5)
    assert curve["2"]["siret"]["rate"] == pytest.approx(1.0)
    assert curve["2"]["siren"]["rate"] == pytest.approx(1.0)


def test_stage_audit_counts_first_loss() -> None:
    rows = pd.DataFrame(
        [
            _row(),
            _row(ground_truth_in_base=False, ground_truth_rank=None),
        ]
    )

    audit = build_stage_audit(rows, max_cutoff=500)

    assert audit["stage_recall"]["partition"]["successes"] == 1
    assert audit["loss_buckets"] == {
        "HIT_AT_50": 1,
        "PARTITION_MISS": 1,
    }
