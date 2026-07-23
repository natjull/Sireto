from src.xgb_matcher.v9_cross_encoder import (
    cross_encoder_gate,
    serialize_cross_encoder_pair,
)


def test_cross_encoder_serialization_is_explicit():
    crm, candidate = serialize_cross_encoder_pair(
        {
            "crm_name": "École Saint Joseph",
            "crm_address": "12 rue de l'Église",
            "crm_postcode": "69001",
            "crm_city": "Lyon",
            "denomination": "OGEC Saint Joseph",
            "enseigne1": "École Saint-Joseph",
            "address": "12 RUE DE L EGLISE",
            "postcode": "69001",
            "city": "LYON",
            "forme_juridique": "Association",
        }
    )
    assert "[NOM]" in crm and "[COMMUNE] Lyon" in crm
    assert "[ETABLISSEMENT]" in candidate and "[FORME] Association" in candidate


def test_cross_encoder_promotion_gate_is_conditional():
    accepted = cross_encoder_gate(
        baseline_coverage=0.74,
        variant_coverage=0.755,
        max_segment_regression=0.01,
        baseline_latency_p95_ms=100,
        variant_latency_p95_ms=190,
    )
    rejected = cross_encoder_gate(
        baseline_coverage=0.74,
        variant_coverage=0.745,
        max_segment_regression=0.03,
        baseline_latency_p95_ms=100,
        variant_latency_p95_ms=210,
    )
    assert accepted["promote"] is True
    assert rejected["promote"] is False
