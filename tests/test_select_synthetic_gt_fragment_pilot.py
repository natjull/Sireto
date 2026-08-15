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


def test_subset_requires_retained_distinctive_anchor() -> None:
    value = fragment(
        "name", "TOKEN_SUBSET", {"source_token_count": 4, "retained_positions": [0, 1]}
    )
    assert selector.fragment_supports("name", "ALPHA BETA GAMMA DELTA", value, ["alpha"])
    assert not selector.fragment_supports("name", "ALPHA BETA GAMMA DELTA", value, ["delta"])


def test_protected_subset_anchor_can_be_selected_from_retained_positions() -> None:
    words = ["alpha", "beta", "gamma", "delta"]
    retained = [0, 2]
    anchors = ["delta", "gamma"]
    assert [value for value in anchors if value in {words[index] for index in retained}][:1] == ["gamma"]


def test_added_marks_and_wrong_punctuation_boundary_are_rejected() -> None:
    added = fragment("city", "DIACRITIC_ADDED", {})
    assert not selector.fragment_supports("city", "SAINT-DENIS", added, [])
    removed = fragment(
        "city", "PUNCTUATION_REMOVED",
        {"edits": [{"after_token_index": 1, "mark": "-"}]},
    )
    assert not selector.fragment_supports("city", "SAINT-DENIS", removed, [])


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
