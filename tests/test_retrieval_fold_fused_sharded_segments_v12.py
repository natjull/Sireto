from __future__ import annotations

from src.xgb_matcher.retrieval_fold_fused_oof_v12 import QueryOutcome
from src.xgb_matcher.retrieval_fold_fused_sharded_oof_v12 import (
    aggregate_sharded_gates_v12,
)


def _outcome(
    query_id: str,
    fold: int,
    *,
    master: bool,
    prerank: bool,
    exact: bool,
    operational: bool,
    loss: str,
    audit: dict[str, object],
) -> QueryOutcome:
    return QueryOutcome(
        query_id=query_id,
        fold=fold,
        human_siret=f"{fold:014d}",
        master_candidate_count=12,
        prerank_candidate_count=8,
        product_candidate_count=3,
        truth_master_rank=1 if master else None,
        truth_prerank_rank=1 if prerank else None,
        truth_product_rank=1 if exact else None,
        master_hit_at_2000=master,
        prerank_hit_at_500=prerank,
        exact_hit_at_100=exact,
        operational_hit_at_100=operational,
        loss_reason=loss,
        audit_metadata=audit,
    )


def _population() -> tuple[QueryOutcome, ...]:
    return (
        _outcome(
            "q2-active",
            2,
            master=True,
            prerank=True,
            exact=True,
            operational=True,
            loss="HIT",
            audit={
                "active_closed": "ACTIVE",
                "crm_origin": "NEW_CRM",
                "seen_unseen_siren": "SEEN",
                "known_new_site": "KNOWN_SITE",
                "megapolis": "NON_MEGAPOLIS",
                "qualification": "EXACT_IDENTIFIABLE",
                "label_kind": "MATCH_EXACT",
                "qualification_v2": "MATCH_EXACT",
                "qualification_v3": "MATCH_EXACT",
            },
        ),
        _outcome(
            "q2-closed",
            2,
            master=False,
            prerank=False,
            exact=False,
            operational=False,
            loss="PARENT_NOT_RETRIEVED",
            audit={
                "active_closed": "CLOSED",
                "crm_origin": "HISTORICAL_CRM",
                "seen_unseen_siren": "UNSEEN",
                "known_new_site": "NEW_SITE",
                "megapolis": "MEGAPOLIS",
                "qualification": "AMBIGUOUS",
                "label_kind": "AMBIGUOUS",
                "qualification_v2": "AMBIGUOUS",
                "qualification_v3": "AMBIGUOUS",
            },
        ),
        _outcome(
            "q3-unknown-metadata",
            3,
            master=True,
            prerank=False,
            exact=False,
            operational=False,
            loss="EVICTED_MASTER_TO_500",
            audit={"label_kind": "UNRESOLVED"},
        ),
        _outcome(
            "q4-aliases",
            4,
            master=True,
            prerank=True,
            exact=False,
            operational=True,
            loss="EVICTED_500_TO_100",
            audit={
                "ground_truth_state": "ACTIVE",
                "source_kind": "LEGACY_CRM",
                "unseen_siren": False,
                "new_site_known_siren": True,
                "is_megapolis": True,
                "qualification_status": "UNRESOLVED",
                "label_kind": "MATCH_EXACT",
                "v2_label_kind": "UNRESOLVED",
                "v3_label_kind": "MATCH_EXACT",
            },
        ),
    )


def test_all_required_audit_segments_are_complete_and_non_filtering() -> None:
    outcomes = _population()
    result = aggregate_sharded_gates_v12(outcomes)

    # Existing global metrics retain the all-human denominator and semantics.
    assert result["master_recall_at_2000"]["successes"] == 3
    assert result["master_recall_at_2000"]["total"] == 4
    assert result["prerank_recall_at_500"]["successes"] == 2
    assert result["exact_recall_at_100"]["successes"] == 1
    assert result["operational_recall_at_100"]["successes"] == 2

    segments = result["segments"]
    assert tuple(segments) == (
        "active_closed",
        "crm_origin",
        "seen_unseen_siren",
        "known_new_site",
        "megapolis",
        "qualification",
        "label_kind",
        "qualification_v2",
        "qualification_v3",
    )
    for dimension in segments.values():
        assert "UNKNOWN" in dimension
        assert sum(bucket["query_count"] for bucket in dimension.values()) == 4
        assert all(
            bucket["exact_recall_at_100"]["total"] == bucket["query_count"]
            for bucket in dimension.values()
        )

    active = segments["active_closed"]["ACTIVE"]
    assert active["query_count"] == 2
    assert active["master_recall_at_2000"]["successes"] == 2
    assert active["prerank_recall_at_500"]["successes"] == 2
    assert active["exact_recall_at_100"]["successes"] == 1
    assert active["operational_recall_at_100"]["successes"] == 2
    assert active["loss_counts"] == {
        "EVICTED_500_TO_100": 1,
        "HIT": 1,
    }

    assert segments["active_closed"]["UNKNOWN"]["query_count"] == 1
    assert segments["seen_unseen_siren"]["SEEN"]["query_count"] == 2
    assert segments["known_new_site"]["NEW_SITE"]["query_count"] == 2
    assert segments["megapolis"]["MEGAPOLIS"]["query_count"] == 2
    # UNKNOWN remains explicit even when every outcome has this dimension.
    assert segments["label_kind"]["UNKNOWN"]["query_count"] == 0


def test_segment_reporting_is_deterministic_under_outcome_reordering() -> None:
    outcomes = _population()
    forward = aggregate_sharded_gates_v12(outcomes)
    reverse = aggregate_sharded_gates_v12(tuple(reversed(outcomes)))

    assert reverse == forward
    for dimension in forward["segments"].values():
        assert tuple(dimension) == tuple(sorted(dimension))
        for bucket in dimension.values():
            assert tuple(bucket["loss_counts"]) == tuple(
                sorted(bucket["loss_counts"])
            )
