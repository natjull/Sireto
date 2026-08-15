from __future__ import annotations

from scripts import select_synthetic_gt_fragment_pilot as selector


def fragment(field: str, relation: str, parameters: dict) -> dict:
    return {
        "field": field, "relation": relation, "operation_parameters": parameters,
        "source_fold": 2,
    }


def test_distinctive_tokens_exclude_competitor_and_generic_words() -> None:
    context = {
        "target": {"names": [{"kind": "OFFICIAL_NAME", "value": "SAS ALPHA UNIQUE PARIS"}]},
        "internal_context": [{"name_values": ["ALPHA SERVICES"]}],
    }
    assert selector.distinctive_name_tokens(context) == ["unique", "paris"]


def test_distinctive_tokens_exclude_all_legal_forms() -> None:
    context = {
        "target": {"names": [{"kind": "OFFICIAL_NAME", "value": "EURL ANTONIO COSTA"}]},
        "internal_context": [],
    }
    assert selector.distinctive_name_tokens(context) == ["antonio", "costa"]


def test_subset_requires_retained_distinctive_anchor() -> None:
    value = fragment(
        "name", "TOKEN_SUBSET", {"source_token_count": 4, "retained_positions": [0, 1, 2]}
    )
    assert selector.fragment_supports("name", "ALPHA BETA GAMMA DELTA", value, ["alpha"])
    assert not selector.fragment_supports("name", "ALPHA BETA GAMMA DELTA", value, ["delta"])


def test_subset_preserves_linked_compounds_and_rejects_function_word_endings() -> None:
    cut_compound = fragment(
        "name", "TOKEN_SUBSET", {"source_token_count": 3, "retained_positions": [0, 1]}
    )
    assert not selector.fragment_supports(
        "name", "GARAGE PRUD'HOMME", cut_compound, ["prud"]
    )
    ends_with_de = fragment(
        "name", "TOKEN_SUBSET", {"source_token_count": 4, "retained_positions": [0, 1]}
    )
    assert not selector.fragment_supports(
        "name", "ARTISANALE DE MACONNERIE GENERALE", ends_with_de, ["artisanale"]
    )
    keep_compound = fragment(
        "name", "TOKEN_SUBSET", {"source_token_count": 4, "retained_positions": [0, 1, 2]}
    )
    assert selector.fragment_supports(
        "name", "SARL PRUD'HOMME SERVICES", keep_compound, ["prud"]
    )
    legal_only_removal = fragment(
        "name", "TOKEN_SUBSET", {"source_token_count": 3, "retained_positions": [1, 2]}
    )
    assert not selector.fragment_supports(
        "name", "SARL ALPHA BETA", legal_only_removal, ["alpha", "beta"]
    )


def test_subset_requires_last_distinctive_anchor() -> None:
    drop_last_anchor = fragment(
        "name", "TOKEN_SUBSET", {"source_token_count": 3, "retained_positions": [0, 1]}
    )
    assert not selector.fragment_supports(
        "name", "JEAN MICHEL COLOMBIER", drop_last_anchor,
        ["jean", "michel", "colombier"],
    )
    ends_with_da = fragment(
        "name", "TOKEN_SUBSET", {"source_token_count": 4, "retained_positions": [0, 1, 2]}
    )
    assert not selector.fragment_supports(
        "name", "EURL ANTONIO DA COSTA", ends_with_da, ["antonio", "costa"]
    )


def test_protected_subset_anchor_can_be_selected_from_retained_positions() -> None:
    words = ["alpha", "beta", "gamma", "delta"]
    retained = [0, 2]
    anchors = ["delta", "gamma"]
    assert [value for value in anchors if value in {words[index] for index in retained}][:1] == ["gamma"]


def test_address_subset_preserves_number_type_and_content_anchor() -> None:
    remove_des = fragment(
        "address", "ADDRESS_TOKEN_SUBSET",
        {"source_token_count": 4, "retained_positions": [0, 1, 3]},
    )
    assert selector.fragment_supports(
        "address", "26 RUE DES PETUNIAS", remove_des, []
    )
    assert not selector.fragment_supports(
        "address", "26 RUE MAURICE THOREZ", remove_des, []
    )
    remove_type = fragment(
        "address", "ADDRESS_TOKEN_SUBSET",
        {"source_token_count": 4, "retained_positions": [0, 2, 3]},
    )
    assert not selector.fragment_supports(
        "address", "26 RUE DES PETUNIAS", remove_type, []
    )


def test_added_marks_and_wrong_punctuation_boundary_are_rejected() -> None:
    added = fragment("city", "DIACRITIC_ADDED", {})
    assert not selector.fragment_supports("city", "SAINT-DENIS", added, [])
    removed = fragment(
        "city", "PUNCTUATION_REMOVED",
        {"edits": [{"after_token_index": 1, "mark": "-"}]},
    )
    assert not selector.fragment_supports("city", "SAINT-DENIS", removed, [])


def test_token_order_allows_only_legal_form_end_move() -> None:
    local_swap = fragment(
        "name", "TOKEN_ORDER", {"source_token_count": 3, "permutation": [1, 0, 2]}
    )
    arbitrary_cycle = fragment(
        "name", "TOKEN_ORDER", {"source_token_count": 3, "permutation": [1, 2, 0]}
    )
    legal_form_move = fragment(
        "name", "TOKEN_ORDER", {"source_token_count": 3, "permutation": [1, 2, 0]}
    )
    assert not selector.fragment_supports("name", "ALPHA BETA GAMMA", local_swap, [])
    assert not selector.fragment_supports("name", "ALPHA BETA GAMMA", arbitrary_cycle, [])
    assert selector.fragment_supports("name", "SARL ALPHA BETA", legal_form_move, [])


def test_canary_name_plan_suspends_join_split() -> None:
    assert "JOIN_SPLIT" not in selector.NAME_QUOTAS
    assert sum(selector.NAME_QUOTAS.values()) == 30


def test_relation_flow_honours_quota_and_three_distinct_per_target() -> None:
    targets = [{"target_siret": str(index)} for index in range(2)]
    caps = {
        str(index): {relation: [{}] for relation in ("A", "B", "C")}
        for index in range(2)
    }
    result = selector.relation_assignment(
        targets, {"A": 2, "B": 2, "C": 2}, caps, distinct_per_target=True
    )
    assert result is not None
    assert all(set(values) == {"A", "B", "C"} for values in result.values())
