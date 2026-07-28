from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.xgb_matcher import v412_direct_evidence as subject


def _raw(siret: str, *, name: bool = True, address: bool = False) -> dict:
    return {
        "candidate_siret": siret,
        "candidate_siren": siret[:9],
        "candidate_state": "A",
        "exact_name_anchor": name,
        "exact_address_anchor": address,
        "direct_evidence_class": "NAME_AND_ADDRESS",
    }


def _frames(raw_candidates: list[dict]):
    candidates = [
        subject.candidate_evidence_record("q1", row)
        for row in raw_candidates
    ]
    query = subject.query_evidence_record(
        query_id="q1",
        partition_key="insee:75056",
        active_universe_count=500,
        candidates=candidates,
    )
    return subject.build_evidence_frames([query], candidates)


@pytest.mark.parametrize(
    ("raw_candidates", "count", "sirens", "cross", "multisite"),
    [
        ([], 0, 0, False, False),
        ([_raw("11111111100011")], 1, 1, False, False),
        (
            [_raw("11111111100011"), _raw("11111111100029")],
            2,
            1,
            False,
            True,
        ),
        (
            [_raw("11111111100011"), _raw("22222222200022")],
            2,
            2,
            True,
            False,
        ),
    ],
)
def test_aggregate_cardinality_and_collision_flags(
    raw_candidates, count, sirens, cross, multisite
):
    queries, candidates = _frames(raw_candidates)
    row = queries.iloc[0]
    assert row["direct_candidate_count"] == count
    assert row["direct_siren_count"] == sirens
    assert bool(row["cross_siren_direct_collision"]) is cross
    assert bool(row["same_siren_direct_multisite"]) is multisite
    refs = json.loads(row["evidence_refs_json"])
    assert refs == sorted(candidates["evidence_ref"].tolist())
    if count == 1:
        assert row["sole_direct_siret"] == raw_candidates[0]["candidate_siret"]
    else:
        assert pd.isna(row["sole_direct_siret"])


def test_more_than_100_direct_candidates_is_valid_and_not_truncated():
    raw = [
        _raw(f"123456789{index:05d}")
        for index in range(137)
    ]
    queries, candidates = _frames(raw)
    assert len(candidates) == 137
    assert queries.loc[0, "direct_candidate_count"] == 137


def test_candidate_projection_requires_frozen_direct_policy():
    assert (
        subject.candidate_evidence_record(
            "q1", _raw("11111111100011", name=True, address=True)
        )["direct_match_rule"]
        == "EXACT_NAME_AND_ADDRESS"
    )
    with pytest.raises(ValueError, match="lacks exact anchor"):
        subject.candidate_evidence_record(
            "q1", _raw("11111111100011", name=False, address=False)
        )
    not_direct = _raw("11111111100011")
    not_direct["direct_evidence_class"] = "NAME_ONLY"
    with pytest.raises(ValueError, match="non-direct"):
        subject.candidate_evidence_record("q1", not_direct)
    assert subject.direct_match_rule(
        {
            "exact_name_anchor": np.bool_(True),
            "exact_address_anchor": np.bool_(False),
        }
    ) == "EXACT_NAME_STRONG_ADDRESS"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("candidate_state", "F", "inactive"),
        ("candidate_siren", "999999999", "SIRET/SIREN"),
        ("evidence_ref", "forged", "reference"),
        ("strong_address_evidence", False, "weak"),
    ],
)
def test_validation_fails_closed_on_candidate_integrity(field, value, message):
    queries, candidates = _frames([_raw("11111111100011")])
    candidates.loc[0, field] = value
    with pytest.raises(ValueError, match=message):
        subject.validate_evidence(queries, candidates)


def test_validation_fails_closed_on_aggregate_count():
    queries, candidates = _frames([_raw("11111111100011")])
    queries.loc[0, "direct_candidate_count"] = 2
    with pytest.raises(ValueError, match="aggregate mismatch"):
        subject.validate_evidence(queries, candidates)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("active_universe_count", -1, "invalid aggregate count"),
        ("active_universe_count", 0, "exceeds active universe"),
        ("query_id", "", "empty query ID"),
        ("partition_key", "", "empty partition key"),
        ("cross_siren_direct_collision", 1, "aggregate boolean"),
    ],
)
def test_validation_fails_closed_on_query_integrity(field, value, message):
    queries, candidates = _frames([_raw("11111111100011")])
    if field in {
        "cross_siren_direct_collision",
        "same_siren_direct_multisite",
    }:
        queries[field] = queries[field].astype(object)
    queries.loc[0, field] = value
    with pytest.raises(ValueError, match=message):
        subject.validate_evidence(queries, candidates)


def test_validation_recomputes_direct_rule_from_anchors():
    queries, candidates = _frames([_raw("11111111100011")])
    candidates.loc[0, "direct_match_rule"] = "EXACT_ADDRESS_STRONG_NAME"
    with pytest.raises(ValueError, match="direct match rule changed"):
        subject.validate_evidence(queries, candidates)


@pytest.mark.parametrize(
    ("decision", "reason", "predicted", "count", "sole", "expected"),
    [
        ("REVIEW", "LOW_CONFIDENCE", None, 1, "11111111100011",
         ("REVIEW", "LOW_CONFIDENCE")),
        ("AUTO_MATCH", None, "11111111100011", 0, None,
         ("REVIEW", "NO_DIRECT_EVIDENCE")),
        ("AUTO_MATCH", None, "11111111100011", 2, None,
         ("REVIEW", "MULTIPLE_STRONG_DIRECT_CANDIDATES")),
        ("AUTO_MATCH", None, "11111111100011", 1, "22222222200022",
         ("REVIEW", "DIRECT_EVIDENCE_DISAGREES_TOP1")),
        ("AUTO_MATCH", None, "11111111100011", 1, "11111111100011",
         ("AUTO_MATCH", None)),
    ],
)
def test_guard_is_a_pure_veto(decision, reason, predicted, count, sole, expected):
    assert subject.apply_guard(
        decision_v411=decision,
        review_reason_v411=reason,
        predicted_siret=predicted,
        direct_candidate_count=count,
        sole_direct_siret=sole,
    ) == expected


@pytest.mark.parametrize("invalid_count", [float("nan"), -1, 1.5, True])
def test_guard_rejects_malformed_cardinality(invalid_count):
    with pytest.raises(ValueError, match="invalid direct candidate count"):
        subject.apply_guard(
            decision_v411="AUTO_MATCH",
            review_reason_v411=None,
            predicted_siret="11111111100011",
            direct_candidate_count=invalid_count,
            sole_direct_siret="11111111100011",
        )


@pytest.mark.parametrize(
    ("predicted", "sole"),
    [(None, None), ("11111111100011", None), (None, "11111111100011")],
)
def test_guard_never_accepts_missing_unique_identifiers(predicted, sole):
    assert subject.apply_guard(
        decision_v411="AUTO_MATCH",
        review_reason_v411=None,
        predicted_siret=predicted,
        direct_candidate_count=1,
        sole_direct_siret=sole,
    ) == ("REVIEW", "DIRECT_EVIDENCE_DISAGREES_TOP1")
