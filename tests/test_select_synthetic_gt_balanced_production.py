import json

from scripts.select_synthetic_gt_balanced_production import (
    difficulty,
    exact_counts,
    pair_signature,
    production_usage,
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


def test_production_usage_reserves_cumulative_caps(tmp_path) -> None:
    seed_input = tmp_path / "prior.jsonl"
    fragments = {
        "name": {
            "field": "name",
            "relation": "LEGAL_FORM_REMOVE",
            "inspiration_ref": "name-ref",
            "operation_parameters": {"removed_legal_forms": ["sarl"]},
        },
        "address": {
            "field": "address",
            "relation": "ADDRESS_ABBREVIATE",
            "inspiration_ref": "address-ref",
            "operation_parameters": {"pairs": [{"source": "rue", "target": "r"}]},
        },
    }
    row = {
        "seed_card": {"composite_contracts": [{
            "field_relations": {
                "name": "LEGAL_FORM_REMOVE",
                "address": "ADDRESS_ABBREVIATE",
            },
            "field_inspirations": fragments,
        }]}
    }
    seed_input.write_text(json.dumps(row) + "\n", encoding="utf-8")

    variants, refs, operators, pairs = production_usage([seed_input])

    assert variants == 1
    assert refs == {"name-ref": 1, "address-ref": 1}
    assert sum(operators.values()) == 2
    assert pairs == {"name:LEGAL_FORM_REMOVE+address:ADDRESS_ABBREVIATE": 1}
