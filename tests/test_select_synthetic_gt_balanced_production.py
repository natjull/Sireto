from scripts.select_synthetic_gt_balanced_production import (
    difficulty,
    exact_counts,
    pair_signature,
)


def context(state="A", internal=None):
    return {"target": {"state": state}, "internal_context": internal or []}


def test_exact_counts_preserve_total_and_balance() -> None:
    assert exact_counts(450, {"EASY": 0.2, "MEDIUM": 0.5, "HARD": 0.3}) == {
        "EASY": 90,
        "MEDIUM": 225,
        "HARD": 135,
    }
    values = exact_counts(450, {"A": 0.15, "B": 0.15, "C": 0.7})
    assert sum(values.values()) == 450
    assert abs(values["A"] - values["B"]) <= 1


def test_difficulty_combines_scene_and_operator_strength() -> None:
    easy_pair = ("LEGAL_FORM_REMOVE", ("address", "ADDRESS_ABBREVIATE"))
    hard_pair = ("TOKEN_ORDER", ("address", "ADDRESS_TOKEN_SUBSET"))
    assert difficulty(context(), easy_pair) == "EASY"
    assert difficulty(context(state="F", internal=[{
        "relation_tags": ["SAME_SIREN", "SAME_OFFICIAL_ADDRESS"]
    }]), hard_pair) == "HARD"
    assert pair_signature(hard_pair) == "name:TOKEN_ORDER+address:ADDRESS_TOKEN_SUBSET"
