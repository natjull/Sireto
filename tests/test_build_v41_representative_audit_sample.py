import pandas as pd

from scripts.build_v41_representative_audit_sample import (
    QUOTAS,
    build_blind_cases,
    select_sample,
)


def _decisions():
    rows = []
    for index in range(1_200):
        if index < 550:
            decision = "AUTO_MATCH"
            confidence = 0.5 + index / 2_000
            candidate_count = 100
            reason = None
        else:
            decision = "REVIEW"
            confidence = (index - 550) / 2_000
            candidate_count = 0 if index < 650 else 100
            reason = (
                "REVIEW_NO_ACTIVE_CANDIDATE"
                if candidate_count == 0
                else "REVIEW_LOW_CONFIDENCE"
            )
        rows.append(
            {
                "service_id": f"S{index:05d}",
                "decision": decision,
                "confidence": confidence,
                "candidate_count": candidate_count,
                "predicted_siret": None if candidate_count == 0 else "1" * 14,
                "review_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def test_sample_is_deterministic_disjoint_and_respects_quotas():
    first = select_sample(_decisions())
    second = select_sample(_decisions().sample(frac=1, random_state=7))
    assert len(first) == 800
    assert first["service_id"].nunique() == 800
    assert first.sort_values("service_id")["service_id"].tolist() == second.sort_values(
        "service_id"
    )["service_id"].tolist()
    assert first["sampling_stratum"].value_counts().to_dict() == QUOTAS


def test_blind_cases_never_expose_model_outputs():
    sample = select_sample(_decisions())
    inventory = pd.DataFrame(
        {
            "service_id": _decisions()["service_id"],
            "eligible_for_shadow": True,
            "SITE": "EXEMPLE",
            "input_siret_state": "ACTIVE",
        }
    )
    blind = build_blind_cases(sample, inventory)
    forbidden = {
        "decision",
        "confidence",
        "predicted_siret",
        "review_reason",
        "sampling_stratum",
    }
    assert not forbidden & set(blind)
    assert len(blind) == 800
