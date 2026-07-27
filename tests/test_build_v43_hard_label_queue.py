from __future__ import annotations

from scripts.build_v43_hard_label_queue import (
    address_overlap,
    entity_types,
    priority,
    risk_signals,
    token_overlap,
)


def test_readable_matching_signals_detect_address_only_entity_conflict() -> None:
    signals = risk_signals(
        crm_name="Médiathèque Jacques Prévert",
        crm_address="12 rue Victor Hugo",
        predicted_name="Angeline Beauté - Les Reflets du Miroir",
        predicted_address="12 RUE VICTOR HUGO 75000 PARIS",
        input_siret_state="CLOSED",
        input_siret="11111111100001",
        predicted_siret="22222222200002",
    )

    assert signals["name_token_overlap"] == 0.0
    assert signals["address_token_coverage"] == 1.0
    assert signals["entity_type_conflict"] is True
    assert signals["address_only_weak_name"] is True
    assert set(entity_types("Médiathèque municipale")) == {
        "CULTURE",
        "PUBLIC_ADMIN",
    }


def test_name_and_address_scores_are_plain_and_bounded() -> None:
    assert token_overlap("GH HOLDING", "G.H. HOLDING") == 1.0
    assert token_overlap("A.H.A.M", "AHAM | ASSOCIATION HAVRAISE") == 1.0
    assert token_overlap("Hyper Buro", "HYPERBURO | ZWILLER") == 1.0
    assert token_overlap("MAIRIE DE TEST", "ECOLE DE TEST") == 0.5
    assert address_overlap("1 rue de Paris", "1 RUE DE PARIS 75001") == 1.0


def test_priority_never_turns_a_signal_into_training_label() -> None:
    score, reason = priority(
        decision="AUTO_MATCH",
        sampling_stratum="RANDOM_POPULATION",
        known_contradiction=False,
        entity_type_conflict=True,
        address_only_weak_name=True,
        prediction_differs_active_input=True,
    )

    assert score == 90
    assert reason == "P0_AUTO_ENTITY_TYPE_CONFLICT"
