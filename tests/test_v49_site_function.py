from __future__ import annotations

from pathlib import Path

from src.xgb_matcher.v49_site_function import SiteFunctionTaxonomy


TAXONOMY = Path("config/v4_9_site_function_taxonomy.json")


def test_mairie_school_conflict() -> None:
    taxonomy = SiteFunctionTaxonomy.load(TAXONOMY)
    crm = taxonomy.detect(["COMMUNE EXEMPLE MAIRIE"])
    candidate = taxonomy.detect(["ECOLE DE LA COMMUNE"], activity_code="85.20Z")
    decision = taxonomy.guard(crm, candidate)
    assert crm.roles == ("ADMIN_MAIRIE",)
    assert candidate.roles == ("EDU_PRIMAIRE",)
    assert decision.review
    assert decision.reason == "SITE_FUNCTION_CONFLICT"


def test_maternelle_primary_conflict() -> None:
    taxonomy = SiteFunctionTaxonomy.load(TAXONOMY)
    crm = taxonomy.detect(["Ecole maternelle André Philip"])
    candidate = taxonomy.detect(
        ["ECOLE PRIMAIRE ANDRE PHILIP"],
        activity_code="85.20Z",
    )
    assert taxonomy.guard(crm, candidate).reason == "SITE_FUNCTION_CONFLICT"


def test_fam_mas_multi_role_is_ambiguous() -> None:
    taxonomy = SiteFunctionTaxonomy.load(TAXONOMY)
    crm = taxonomy.detect(["AFAPEI 15 FAM MAS"])
    candidate = taxonomy.detect(["FOYER ARC EN CIEL"], activity_code="87.10C")
    decision = taxonomy.guard(crm, candidate)
    assert set(crm.roles) == {"MED_FAM", "MED_MAS"}
    assert candidate.roles == ("MED_FOYER",)
    assert decision.reason == "SITE_FUNCTION_AMBIGUOUS"


def test_unknown_does_not_force_review() -> None:
    taxonomy = SiteFunctionTaxonomy.load(TAXONOMY)
    crm = taxonomy.detect(["SOCIETE EXEMPLE"])
    candidate = taxonomy.detect(["ETABLISSEMENT EXEMPLE"])
    assert not taxonomy.guard(crm, candidate).review


def test_same_role_remains_compatible() -> None:
    taxonomy = SiteFunctionTaxonomy.load(TAXONOMY)
    crm = taxonomy.detect(["Pharmacie centrale"])
    candidate = taxonomy.detect(["Officine centrale"], activity_code="47.73Z")
    decision = taxonomy.guard(crm, candidate)
    assert crm.roles == candidate.roles == ("HEALTH_PHARMACY",)
    assert not decision.review
