import json

from scripts.select_synthetic_gt_balanced_production import (
    difficulty,
    exact_counts,
    pair_signature,
    production_usage,
    remaining_quota_counts,
    official_name_alias_fragments,
    strict_token_subset_anchor,
    eligible_for_official_alias,
    adapt_strata_to_easy_capacity,
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


def test_remaining_quota_counts_compensate_prior_promotions() -> None:
    result = remaining_quota_counts(
        100, 1000, {"EASY": 0.2, "MEDIUM": 0.5, "HARD": 0.3},
        {"EASY": 100, "MEDIUM": 100, "HARD": 100},
    )
    assert sum(result.values()) == 100
    # MEDIUM has the largest remaining deficit; this is deliberately not a
    # fresh per-batch 20/50/30 allocation.
    assert result["MEDIUM"] > 50
    assert result["EASY"] < 20


def test_official_alias_fragments_are_authoritative_and_distinct() -> None:
    value = {
        "target_siret": "12345678900012",
        "context_sha256": "a" * 64,
        "target": {
            "state": "A",
            "names": [
                {"kind": "OFFICIAL_NAME", "value": "ALPHA SERVICES"},
                {"kind": "ENSEIGNE", "value": "ALPHA SHOP"},
                {"kind": "ENSEIGNE", "value": "alpha shop"},
            ],
            "address": {
                "number": "1", "street_type": "RUE", "street": "DU TEST",
                "postcode": "75001", "city": "PARIS", "insee": "75056",
            },
        },
    }
    aliases = official_name_alias_fragments(value)
    assert len(aliases) == 1
    fragment = aliases[0]
    assert fragment["observed_crm_value"] == "ALPHA SHOP"
    assert fragment["source_fold"] == -1
    assert fragment["evidence_source_type"] == "SIRENE_OFFICIAL_NAME_OPTION"
    assert fragment["operation_parameters"]["official_alias_value"] == "ALPHA SHOP"


def test_strict_token_subset_protects_rare_retained_anchor() -> None:
    value = {
        "target_siret": "12345678900012", "context_sha256": "b" * 64,
        "target": {
            "state": "A",
            "names": [{"kind": "OFFICIAL_NAME", "value": "SAS ALPHA CONSEIL ZYGOMA"}],
            "address": {"number": "1", "street_type": "RUE", "street": "DU TEST",
                        "postcode": "75001", "city": "PARIS", "insee": "75056"},
        },
    }
    fragment = {
        "field": "name", "relation": "TOKEN_SUBSET",
        "operation_parameters": {
            "source_token_count": 4, "retained_positions": [0, 1, 3],
        },
    }
    assert strict_token_subset_anchor(value, fragment, {"alpha": 20, "zygoma": 1}) == "zygoma"

    # No legal form, enseigne or organisation descriptor: conservative person proxy.
    value["target"]["names"][0]["value"] = "ALPHA BETA GAMMA ZYGOMA"
    assert strict_token_subset_anchor(value, fragment, {"zygoma": 1}) is None


def test_official_alias_eligibility_does_not_require_long_baseline_name() -> None:
    value = {
        "target_siret": "12345678900012", "context_sha256": "c" * 64,
        "qualification": {
            "pre_generation_exact_eligible": True, "siblings_complete": True,
            "same_address_complete": True, "same_name_geography_complete": True,
        },
        "target": {
            "state": "A",
            "names": [
                {"kind": "OFFICIAL_NAME", "value": "ALPHA SAS"},
                {"kind": "ENSEIGNE", "value": "BOUTIQUE ALPHA"},
            ],
            "address": {"number": "1", "street_type": "RUE", "street": "DU TEST",
                        "postcode": "75001", "city": "PARIS", "insee": "75056"},
        },
    }
    assert eligible_for_official_alias(value)


def test_strata_defer_control_without_exceeding_final_remainders() -> None:
    ideal = {
        "FAIL_BOTH_MODELS": 120, "FAIL_XGB_ONLY": 90,
        "FAIL_BGE_ONLY": 90, "TRAIN_DISTRIBUTION": 240,
        "NEAR_CLEAN_CONTROL": 60,
    }
    maximum = {
        "FAIL_BOTH_MODELS": 1000, "FAIL_XGB_ONLY": 800,
        "FAIL_BGE_ONLY": 800, "TRAIN_DISTRIBUTION": 2000,
        "NEAR_CLEAN_CONTROL": 900,
    }
    result = adapt_strata_to_easy_capacity(ideal, maximum, easy_capacity=37)
    assert sum(result.values()) == 600
    assert result["NEAR_CLEAN_CONTROL"] == 37
    assert all(result[key] <= maximum[key] for key in result)
