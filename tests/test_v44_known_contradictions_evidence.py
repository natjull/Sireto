import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "reports" / "v9" / "v4_4_known_contradictions_evidence.json"
EXPECTED_CASE_IDS = {
    "0107123ac3ab0732",
    "00ebcafaaa0a8bf5",
    "007d8c6b8f26962b",
    "003c6fdad046a903",
    "026ff9f27001bebd",
}


def _load():
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_exactly_the_five_frozen_contradictions_are_present():
    artifact = _load()
    cases = artifact["cases"]

    assert len(cases) == 5
    assert {case["audit_case_id"] for case in cases} == EXPECTED_CASE_IDS
    assert len({case["audit_case_id"] for case in cases}) == len(cases)


def test_validated_decisions_have_two_independent_source_families():
    for case in _load()["cases"]:
        counted_families = {
            source["source_family"]
            for source in case["sources"]
            if source["counts_for_independence"]
        }

        assert counted_families == set(case["independent_source_families"])
        if case["evidence_validated"]:
            assert case["adjudication_label"] in {
                "TOP1_CORRECT",
                "TOP1_WRONG",
                "AMBIGUOUS",
            }
            assert len(counted_families) >= 2
            assert case["training_eligible"] is True
        else:
            assert case["adjudication_label"] == "UNRESOLVED"
            assert case["training_eligible"] is False


def test_sources_are_traceable_full_documents_not_snippets():
    for case in _load()["cases"]:
        assert case["sources"]
        for source in case["sources"]:
            assert source["producer"].strip()
            assert source["url"].startswith("https://")
            assert source["collected_at"]
            assert source["document_type"].strip()
            assert source["archived_facts"]
            assert all(fact.strip() for fact in source["archived_facts"])
            assert "snippet" not in source["document_type"].lower()
            assert "search_result" not in source
            assert "search_snippet" not in source


def test_same_registry_views_never_create_false_independence():
    welcoop = next(
        case
        for case in _load()["cases"]
        if case["audit_case_id"] == "0107123ac3ab0732"
    )
    registry_sources = [
        source
        for source in welcoop["sources"]
        if source["source_family"] == "REGISTRY_CORE_SIRENE"
    ]

    assert len(registry_sources) == 2
    assert sum(source["counts_for_independence"] for source in registry_sources) == 1


def test_no_unproven_alternative_siret_is_created():
    artifact = _load()

    assert artifact["summary"]["validated_correct_siret_count"] == 0
    assert all(case["validated_correct_siret"] is None for case in artifact["cases"])


def test_conservative_outcome_is_four_wrong_and_one_unresolved():
    cases = _load()["cases"]
    labels = [case["adjudication_label"] for case in cases]

    assert labels.count("TOP1_WRONG") == 4
    assert labels.count("UNRESOLVED") == 1
    unresolved = next(case for case in cases if case["adjudication_label"] == "UNRESOLVED")
    assert unresolved["audit_case_id"] == "007d8c6b8f26962b"
    assert len(unresolved["independent_source_families"]) == 1
