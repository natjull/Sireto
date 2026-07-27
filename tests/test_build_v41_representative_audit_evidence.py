from scripts.build_v41_representative_audit_evidence import (
    provisional_adjudication,
)


def test_provisional_adjudication_prefers_direct_geographic_evidence():
    result = provisional_adjudication(
        audit_case_id="case",
        direct_sirets=["11111111100011"],
        lineage_strong_sirets=["22222222200022"],
    )
    assert result["label_kind"] == "MATCH_EXACT"
    assert result["ground_truth_siret"] == "11111111100011"
    assert result["rule_code"] == "UNIQUE_DIRECT_GEO"


def test_provisional_adjudication_preserves_ambiguity_and_unknowns():
    ambiguous = provisional_adjudication(
        audit_case_id="case",
        direct_sirets=[],
        lineage_strong_sirets=["11111111100011", "11111111100029"],
    )
    unresolved = provisional_adjudication(
        audit_case_id="other",
        direct_sirets=[],
        lineage_strong_sirets=[],
    )
    assert ambiguous["label_kind"] == "AMBIGUOUS"
    assert ambiguous["ground_truth_siret"] is None
    assert unresolved["label_kind"] == "UNRESOLVED"
    assert unresolved["ground_truth_siret"] is None
    assert unresolved["adjudication_status"] == "PROVISIONAL"
