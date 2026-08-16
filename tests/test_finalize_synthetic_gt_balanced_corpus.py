from copy import deepcopy
from collections import Counter

import pytest

from scripts.finalize_synthetic_gt_balanced_corpus import (
    cap_limits,
    final_state_audit,
    quota_audit,
)
from scripts.select_synthetic_gt_balanced_production import exact_counts


def plan() -> dict:
    return {
        "objective": {
            "promoted_variant_target": 10,
            "maximum_unique_targets": 8,
            "maximum_variants_per_target": 3,
        },
        "corpus_balance": {
            "state_variants": {"A": 0.5, "F": 0.5},
            "target_identity_active_share_bounds": [0.545, 0.59],
            "difficulty": {"EASY": 0.2, "MEDIUM": 0.5, "HARD": 0.3},
            "augmentation_strata": {
                "FAIL_BOTH_MODELS": 0.2,
                "FAIL_XGB_ONLY": 0.15,
                "FAIL_BGE_ONLY": 0.15,
                "TRAIN_DISTRIBUTION": 0.4,
                "NEAR_CLEAN_CONTROL": 0.1,
            },
        },
        "global_caps": {
            "inspiration_ref_uses": 4,
            "exact_operator_share": 0.2,
            "relation_pair_share": 0.4,
            "official_name_alias_share": 0.6,
            "name_token_subset_share": 0.3,
            "name_token_subset_signature_share_global": 0.2,
            "name_token_subset_signature_share_within_family": 0.5,
        },
    }


def summary(promoted: int) -> dict:
    difficulty = exact_counts(promoted, plan()["corpus_balance"]["difficulty"])
    strata = exact_counts(
        promoted, plan()["corpus_balance"]["augmentation_strata"]
    )
    return {
        "promoted_variants": promoted,
        "difficulty_counts": difficulty,
        "augmentation_stratum_counts": strata,
        "inspiration_ref_counts": {},
        "exact_operator_counts": {},
        "relation_pair_counts": {},
        "name_token_subset_signature_counts": {},
        "distinct_target_sirets": min(promoted, 4),
    }


def test_quota_audit_accepts_exact_final_target_without_group_padding() -> None:
    result = quota_audit(summary(10), plan(), require_complete=True)
    assert result["promoted"] == 10
    assert result["remaining"] == 0
    assert all(
        value == 0
        for values in result["deficits"].values()
        for value in values.values()
    )


def test_quota_audit_reports_consistent_residual() -> None:
    current = summary(4)
    # Prefix production is globally steered, not expected to be a miniature
    # exact 20/50/30 corpus. Keep every cell below its final ceiling.
    current["difficulty_counts"] = {"EASY": 1, "MEDIUM": 2, "HARD": 1}
    current["augmentation_stratum_counts"] = {
        "FAIL_BOTH_MODELS": 1,
        "FAIL_XGB_ONLY": 1,
        "FAIL_BGE_ONLY": 0,
        "TRAIN_DISTRIBUTION": 2,
        "NEAR_CLEAN_CONTROL": 0,
    }
    result = quota_audit(current, plan(), require_complete=False)
    assert result["remaining"] == 6
    assert sum(result["deficits"]["difficulty"].values()) == 6
    assert sum(result["deficits"]["augmentation_strata"].values()) == 6


def test_quota_audit_rejects_cap_overrun() -> None:
    current = deepcopy(summary(10))
    current["inspiration_ref_counts"] = {"ref": 5}
    with pytest.raises(ValueError, match="cumulative cap"):
        quota_audit(current, plan(), require_complete=True)


def test_cap_limits_materialize_absolute_operator_and_ref_avenant() -> None:
    value = plan()
    value["objective"]["promoted_variant_target"] = 20_000
    value["global_caps"]["inspiration_ref_uses"] = 4_100
    value["global_caps"]["exact_operator_share"] = 0.105
    limits = cap_limits(value)
    assert limits["inspiration_ref"] == 4_100
    assert limits["exact_operator"] == 4_200


def test_quota_audit_rejects_official_alias_family_overrun() -> None:
    current = deepcopy(summary(10))
    current["relation_pair_counts"] = {
        "name:OFFICIAL_NAME_ALIAS+address:ADDRESS_ABBREVIATE": 4,
        "name:OFFICIAL_NAME_ALIAS+city:PUNCTUATION_REMOVED": 3,
    }
    with pytest.raises(ValueError, match="cumulative cap"):
        quota_audit(current, plan(), require_complete=True)


def test_final_state_audit_requires_exact_variants_and_identity_envelope() -> None:
    result = final_state_audit(
        Counter({"A": 5, "F": 5}), Counter({"A": 4, "F": 3}), plan()
    )
    assert result["active_target_identity_share"] == 4 / 7

    with pytest.raises(ValueError, match="state variant quotas"):
        final_state_audit(
            Counter({"A": 6, "F": 4}), Counter({"A": 4, "F": 3}), plan()
        )
    with pytest.raises(ValueError, match="identity share"):
        final_state_audit(
            Counter({"A": 5, "F": 5}), Counter({"A": 3, "F": 3}), plan()
        )


def test_final_state_audit_accepts_certified_545_identity_avenant() -> None:
    result = final_state_audit(
        Counter({"A": 5, "F": 5}),
        Counter({"A": 5_377, "F": 4_489}),
        plan(),
    )
    assert result["active_target_identity_share"] == 5_377 / 9_866
    assert result["active_target_identity_share_integer_margins"]["lower"] >= 0

    with pytest.raises(ValueError, match="identity share"):
        final_state_audit(
            Counter({"A": 5, "F": 5}),
            Counter({"A": 5_449, "F": 4_551}),
            plan(),
        )
