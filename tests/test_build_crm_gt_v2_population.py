from __future__ import annotations

import pandas as pd

from scripts.build_crm_gt_v2_population import assign_unseen, audit_sample


def _rows(count: int = 900) -> pd.DataFrame:
    rows = []
    for i in range(count):
        rows.append({
            "query_id": f"Q{i}", "target_siren": f"{300000000+i:09d}",
            "existing_component_relation": "UNSEEN_SIREN_NEEDS_ASSIGNMENT",
            "sirene_etat": "F" if i % 5 == 0 else "A",
            "loc_match_type": "CP_FALLBACK_INSEE_MISSING" if i < 3 else (
                "INSEE_ONLY" if i % 7 == 0 else "INSEE_AND_CP"
            ),
        })
    return pd.DataFrame(rows)


def test_unseen_assignment_is_grouped_stable_and_near_70_15_15() -> None:
    rows = _rows()
    first = assign_unseen(rows, 42)
    assert first == assign_unseen(rows.sample(frac=1, random_state=7), 42)
    roles = pd.Series({siren: role for siren, (role, _fold) in first.items()})
    counts = roles.value_counts()
    assert abs(counts["TRAIN"] / len(rows) - 0.70) < 0.03
    assert abs(counts["PROSPECTIVE_DEV"] / len(rows) - 0.15) < 0.02
    assert abs(counts["PROSPECTIVE_TEST"] / len(rows) - 0.15) < 0.02
    assert all(fold in {2, 3, 4} for role, fold in first.values() if role == "TRAIN")
    assert all(fold == 0 for role, fold in first.values() if role == "PROSPECTIVE_DEV")
    assert all(fold == 1 for role, fold in first.values() if role == "PROSPECTIVE_TEST")


def test_audit_sample_has_fixed_roles_unique_sirens_and_strata_coverage() -> None:
    rows = _rows()
    assignment = assign_unseen(rows, 42)
    rows["split_role"] = rows["target_siren"].map(lambda value: assignment[value][0])
    sample = audit_sample(rows, 42)
    assert len(sample) == sample["target_siren"].nunique() == 400
    assert sample["split_role"].value_counts().to_dict() == {
        "TRAIN": 200, "PROSPECTIVE_DEV": 100, "PROSPECTIVE_TEST": 100,
    }
    assert set(rows["loc_match_type"]).issubset(set(sample["loc_match_type"]))
    assert set(rows["sirene_etat"]).issubset(set(sample["sirene_etat"]))
    assert sample["audit_verdict"].eq("PENDING_INDEPENDENT_REVIEW").all()


def test_audit_sample_can_exclude_a_prior_review() -> None:
    rows = _rows(3000)
    assignment = assign_unseen(rows, 42)
    rows["split_role"] = rows["target_siren"].map(lambda value: assignment[value][0])
    first = audit_sample(rows, 42)
    second = audit_sample(rows, 20260818, set(first["query_id"]))
    assert len(second) == 400
    assert set(first["query_id"]).isdisjoint(second["query_id"])
