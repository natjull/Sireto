from __future__ import annotations

from scripts import build_synthetic_gt_field_inspiration_bank as bank


def test_subset_operator_is_exact_and_not_too_destructive() -> None:
    assert bank.operation_parameters(
        "name", "TOKEN_SUBSET", "MAISON DES FLEURS PARIS", "MAISON FLEURS"
    ) == {"source_token_count": 4, "retained_positions": [0, 2]}
    assert bank.operation_parameters(
        "name", "TOKEN_SUBSET", "MAISON DES FLEURS PARIS", "FLEURS"
    ) is None


def test_added_marks_are_never_safe_relations() -> None:
    assert "DIACRITIC_ADDED" not in bank.SAFE_RELATIONS["name"]
    assert "DIACRITIC_ADDED" not in bank.SAFE_RELATIONS["address"]
    assert "PUNCTUATION_ADDED" not in bank.SAFE_RELATIONS["city"]


def test_removed_mark_and_abbreviation_parameters_are_exact() -> None:
    assert bank.operation_parameters(
        "city", "PUNCTUATION_REMOVED", "SAINT-DENIS", "SAINT DENIS"
    ) == {"edits": [{"after_token_index": 0, "mark": "-"}]}
    assert bank.operation_parameters(
        "address", "ADDRESS_ABBREVIATE", "12 RUE DES LILAS", "12 R DES LILAS"
    ) == {"pairs": [{"source": "RUE", "target": "R"}]}


def test_token_permutation_is_position_locked() -> None:
    assert bank.operation_parameters(
        "name", "TOKEN_ORDER", "MAISON DES FLEURS", "FLEURS MAISON DES"
    ) == {"source_token_count": 3, "permutation": [2, 0, 1]}


def test_join_operator_records_exact_contiguous_groups() -> None:
    assert bank.operation_parameters(
        "name", "JOIN_SPLIT", "ALPHA BETA GAMMA", "ALPHABETA GAMMA"
    ) == {"source_token_count": 3, "groups": [[0, 1], [2]]}


def test_address_subset_preserves_number_and_exact_positions() -> None:
    assert bank.operation_parameters(
        "address", "ADDRESS_TOKEN_SUBSET", "12 RUE DES LILAS", "12 RUE LILAS"
    ) == {"source_token_count": 4, "retained_positions": [0, 1, 3]}


def test_subset_inspiration_cannot_hide_a_punctuation_edit() -> None:
    parameters = bank.operation_parameters(
        "name", "TOKEN_SUBSET", "GARAGE PRUD'HOMME", "PRUD HOMME"
    )
    assert parameters == {"source_token_count": 3, "retained_positions": [1, 2]}
    assert not bank.loop.composite_preserves_non_target_punctuation(
        "TOKEN_SUBSET", "GARAGE PRUD'HOMME", "PRUD HOMME", parameters
    )
