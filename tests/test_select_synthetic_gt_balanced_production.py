import json
from collections import Counter

from scripts import select_synthetic_gt_balanced_production as selector
from scripts.select_synthetic_gt_balanced_production import (
    difficulty,
    exact_counts,
    identity_share_coefficients,
    pair_signature,
    production_usage,
    remaining_quota_counts,
    official_name_alias_fragments,
    strict_token_subset_anchor,
    eligible_for_official_alias,
    adapt_strata_to_easy_capacity,
    candidate_bundles,
    maximum_batch_target_additions,
    materializable_relation_pair_count,
    stratified_context_pool,
    build_context_index,
    indexed_context_stub,
    read_indexed_contexts,
    runtime_stable_punctuation_fragment,
    safe_capabilities,
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


def test_identity_share_coefficients_preserve_decimal_bounds_exactly() -> None:
    assert identity_share_coefficients((0.5499, 0.59)) == (
        (4501, -5499), (41, -59),
    )
    assert identity_share_coefficients((0.55, 0.59)) == (
        (9, -11), (41, -59),
    )


def test_difficulty_combines_scene_and_operator_strength() -> None:
    easy_pair = ("LEGAL_FORM_REMOVE", ("address", "ADDRESS_ABBREVIATE"))
    hard_pair = ("TOKEN_ORDER", ("address", "ADDRESS_TOKEN_SUBSET"))
    assert difficulty(context(), easy_pair) == "EASY"
    assert difficulty(context(state="F", internal=[{
        "relation_tags": ["SAME_SIREN", "SAME_OFFICIAL_ADDRESS"]
    }]), hard_pair) == "HARD"
    assert pair_signature(hard_pair) == "name:TOKEN_ORDER+address:ADDRESS_TOKEN_SUBSET"


def test_selector_rejects_only_runtime_unstable_no_space_punctuation_joins() -> None:
    unsafe = {
        "field": "name", "relation": "PUNCTUATION_REMOVED", "source_fold": 2,
        "inspiration_ref": "unsafe",
        "operation_parameters": {
            "edits": [{"after_token_index": 0, "mark": "-", "replacement": ""}],
        },
    }
    final_pair = {
        **unsafe, "inspiration_ref": "final-pair",
        "operation_parameters": {
            "edits": [{"after_token_index": 1, "mark": "'", "replacement": ""}],
        },
    }
    space_preserving = {
        **unsafe, "inspiration_ref": "space",
        "operation_parameters": {
            "edits": [{"after_token_index": 0, "mark": "-", "replacement": " "}],
        },
    }
    assert not runtime_stable_punctuation_fragment("JEAN-MARC MERMET", unsafe)
    assert runtime_stable_punctuation_fragment("BUREAUX D'ORSAY", final_pair)
    assert runtime_stable_punctuation_fragment("JEAN-MARC MERMET", space_preserving)
    assert not runtime_stable_punctuation_fragment(
        "JEAN-MARC MERMET",
        {**unsafe, "operation_parameters": {"edits": "not-a-list"}},
    )

    value = {
        "target_siret": "12345678900012",
        "context_sha256": "a" * 64,
        "target": {
            "state": "A",
            "names": [{"kind": "OFFICIAL_NAME", "value": "JEAN-MARC MERMET"}],
            "address": {
                "number": "1", "repetition_index": "", "street_type": "RUE",
                "street": "DU TEST", "postcode": "75001", "city": "PARIS",
                "insee": "75056",
            },
        },
        "qualification": {},
        "internal_context": [],
    }
    names, _locations = safe_capabilities(
        value,
        {("name", "PUNCTUATION_REMOVED"): [unsafe]},
        Counter({"jean": 1, "marc": 1, "mermet": 1}),
        allowed_name_relations=("PUNCTUATION_REMOVED",),
    )
    assert names == {"PUNCTUATION_REMOVED": []}


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


def test_context_pool_reserves_state_and_non_alias_easy_capacity() -> None:
    values = []
    aliases = {}
    easy = {}
    for state in ("A", "F"):
        for is_alias in (True, False):
            for index in range(10):
                siret = f"{state}{int(is_alias)}{index:02d}"
                values.append({"target_siret": siret, "target": {"state": state}})
                aliases[siret] = is_alias
                easy[siret] = 3 if state == "A" and not is_alias else 0
    selected = stratified_context_pool(values, 20, "seed", aliases, easy)
    counts = Counter(
        (value["target"]["state"], aliases[value["target_siret"]])
        for value in selected
    )
    assert counts == {("A", True): 5, ("A", False): 5,
                      ("F", True): 5, ("F", False): 5}
    assert sum(easy[value["target_siret"]] for value in selected) == 15


def test_context_pool_can_oversample_active_candidates_without_losing_alias_mix() -> None:
    values = []
    aliases = {}
    for state in ("A", "F"):
        for is_alias in (True, False):
            for index in range(20):
                siret = f"{state}{int(is_alias)}{index:02d}"
                values.append({"target_siret": siret, "target": {"state": state}})
                aliases[siret] = is_alias
    selected = stratified_context_pool(
        values, 30, "seed", aliases, active_share=2 / 3,
    )
    counts = Counter(
        (value["target"]["state"], aliases[value["target_siret"]])
        for value in selected
    )
    assert counts == {("A", True): 10, ("A", False): 10,
                      ("F", True): 5, ("F", False): 5}


def test_candidate_bundles_cover_one_to_three_runtime_contracts() -> None:
    value = {
        "target_siret": "12345678900000",
        "target": {"state": "A"},
        "internal_context": [],
    }
    names = {"OFFICIAL_NAME_ALIAS": [{"inspiration_ref": "alias"}]}
    locations = {
        ("address", "ADDRESS_ABBREVIATE"): [{"inspiration_ref": "addr"}],
        ("address", "ADDRESS_ALIAS_EXPAND"): [{"inspiration_ref": "alias-addr"}],
        ("city", "PUNCTUATION_REMOVED"): [{"inspiration_ref": "city"}],
    }
    bundles = candidate_bundles(value, names, locations, "seed")
    assert {len(bundle) for bundle in bundles} == {1, 2, 3}


def test_candidate_bundles_keep_a_single_safe_runtime_contract() -> None:
    value = {
        "target_siret": "12345678900000",
        "target": {"state": "A"},
        "internal_context": [],
    }
    bundles = candidate_bundles(
        value,
        {"OFFICIAL_NAME_ALIAS": [{"inspiration_ref": "alias"}]},
        {("address", "ADDRESS_ABBREVIATE"): [{"inspiration_ref": "addr"}]},
        "seed",
    )
    assert len(bundles) == 1
    assert len(bundles[0]) == 1


def test_candidate_bundles_prune_saturated_global_relation_pairs() -> None:
    value = {
        "target_siret": "12345678900000",
        "target": {"state": "A"},
        "internal_context": [],
    }
    saturated = ("OFFICIAL_NAME_ALIAS", ("address", "ADDRESS_ABBREVIATE"))
    residual = ("OFFICIAL_NAME_ALIAS", ("city", "PUNCTUATION_REMOVED"))
    bundles = candidate_bundles(
        value,
        {"OFFICIAL_NAME_ALIAS": [{"inspiration_ref": "alias"}]},
        {
            saturated[1]: [{"inspiration_ref": "addr"}],
            residual[1]: [
                {"inspiration_ref": "city-1"}, {"inspiration_ref": "city-2"}
            ],
        },
        "seed",
        pair_remaining={saturated: 0, residual: 2},
    )
    assert bundles
    assert all(saturated not in bundle for bundle in bundles)
    assert max(Counter(bundle)[residual] for bundle in bundles) == 2


def test_candidate_bundle_thinning_preserves_alias_and_exact_pair_marginals() -> None:
    value = {
        "target_siret": "12345678900000",
        "target": {"state": "A"},
        "internal_context": [],
    }
    names = {
        relation: [{"inspiration_ref": f"name-{relation}"}]
        for relation in (
            "LEGAL_FORM_REMOVE", "PUNCTUATION_REMOVED", "OFFICIAL_NAME_ALIAS",
        )
    }
    locations = {
        relation: [{"inspiration_ref": f"location-{index}"}]
        for index, relation in enumerate((
            ("address", "ADDRESS_ABBREVIATE"),
            ("address", "ADDRESS_ALIAS_EXPAND"),
            ("address", "PUNCTUATION_REMOVED"),
            ("city", "PUNCTUATION_REMOVED"),
        ))
    }
    bundles = candidate_bundles(value, names, locations, "seed")
    singleton_pairs = {bundle[0] for bundle in bundles if len(bundle) == 1}
    assert singleton_pairs == {
        (name_relation, location_relation)
        for name_relation in names for location_relation in locations
    }
    assert {sum(pair[0] == "OFFICIAL_NAME_ALIAS" for pair in bundle)
            for bundle in bundles} >= {0, 1, 2, 3}


def test_unique_target_budget_reserves_three_per_future_target() -> None:
    assert maximum_batch_target_additions(
        3_723, 1_335, 600, 20_000, 8_000, 3,
    ) == 1_439
    assert maximum_batch_target_additions(
        19_873, 7_900, 127, 20_000, 8_000, 3,
    ) == 100


def test_one_exact_relation_pair_is_a_materializable_capability() -> None:
    assert materializable_relation_pair_count(
        {"TOKEN_ORDER": [{"op": 1}], "TOKEN_SUBSET": []},
        {("address", "ADDRESS_ABBREVIATE"): [{"op": 2}]},
    ) == 1


def test_context_index_is_sealed_reusable_and_restores_requested_order(
    tmp_path, monkeypatch,
) -> None:
    def official_context(suffix: str, name: str, state: str) -> dict:
        return {
            "target_siret": f"123456789000{suffix}",
            "target_siren": f"12345678{suffix}",
            "context_sha256": suffix * 64,
            "target": {
                "state": state,
                "names": [{"kind": "OFFICIAL_NAME", "value": name}],
                "address": {
                    "number": "1", "repetition_index": "",
                    "street_type": "RUE", "street": "DU TEST",
                    "postcode": "75001", "city": "PARIS", "insee": "75056",
                },
            },
            "qualification": {},
            "internal_context": [],
        }

    expected = [
        official_context("1", "ALPHA BETA", "A"),
        official_context("2", "ALPHA GAMMA", "F"),
    ]
    source = tmp_path / "official.jsonl"
    source.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in expected),
        encoding="utf-8",
    )
    monkeypatch.setattr(selector.fragments, "eligible", lambda _value: True)
    entries, frequencies = build_context_index(
        source, selector.sha256(source), tmp_path / "cache",
    )
    assert [value["target_siret"] for value in entries] == [
        value["target_siret"] for value in expected
    ]
    assert frequencies == {"alpha": 2, "beta": 1, "gamma": 1}

    # A valid hit must not re-run eligibility over the large official source.
    monkeypatch.setattr(
        selector.fragments, "eligible",
        lambda _value: (_ for _ in ()).throw(AssertionError("cache miss")),
    )
    cached_entries, cached_frequencies = build_context_index(
        source, selector.sha256(source), tmp_path / "cache",
    )
    assert cached_entries == entries
    assert cached_frequencies == frequencies
    reversed_contexts = read_indexed_contexts(
        source,
        [indexed_context_stub(value) for value in reversed(cached_entries)],
    )
    assert reversed_contexts == list(reversed(expected))
