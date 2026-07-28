import json

import pandas as pd

from scripts.evaluate_v49_guard_retrospective import build_predictions, summarize
from src.xgb_matcher.v49_site_function import SiteFunctionTaxonomy


def _taxonomy() -> SiteFunctionTaxonomy:
    return SiteFunctionTaxonomy(
        {
            "schema_version": "sireto-v4.9-site-function-taxonomy-1",
            "roles": [
                {"role": "ADMIN_MAIRIE", "patterns": [r"\bmairie\b"]},
                {
                    "role": "EDU_PRIMAIRE",
                    "patterns": [r"\becole primaire\b"],
                    "activity_codes": ["85.20Z"],
                },
            ],
            "incompatible_families": [["ADMIN_MAIRIE", "EDU_PRIMAIRE"]],
        }
    )


def test_build_predictions_uses_crm_and_sirene_fields():
    labels = pd.DataFrame(
        [
            {
                "audit_case_id": f"{index:016d}",
                "query_id": f"{index:016d}",
                "service_id": f"S{index}",
                "sampling_stratum": "RANDOM_POPULATION",
                "current_label_origin": "V4.7_CURRENT_TOP1",
                "current_adjudication_label": (
                    "TOP1_WRONG" if index == 0 else "UNRESOLVED"
                ),
                "current_top1_siret": f"{index:014d}",
            }
            for index in range(172)
        ]
    )
    crm = pd.DataFrame(
        {
            "audit_case_id": labels["audit_case_id"],
            "SITE": ["MAIRIE"] + ["INCONNU"] * 171,
        }
    )
    sirene = pd.DataFrame(
        {
            "siret": labels["current_top1_siret"],
            "enseigne1Etablissement": [None] * 172,
            "enseigne2Etablissement": [None] * 172,
            "enseigne3Etablissement": [None] * 172,
            "denominationUsuelleEtablissement": ["ECOLE PRIMAIRE"]
            + ["INCONNU"] * 171,
            "activitePrincipaleEtablissement": ["85.20Z"] + ["00.00Z"] * 171,
        }
    )
    result = build_predictions(labels, crm, sirene, _taxonomy())
    first = result[result["audit_case_id"].eq("0000000000000000")].iloc[0]
    assert bool(first["guard_review"]) is True
    assert first["guard_reason"] == "SITE_FUNCTION_CONFLICT"
    assert json.loads(first["crm_roles_json"]) == ["ADMIN_MAIRIE"]
    assert json.loads(first["candidate_roles_json"]) == ["EDU_PRIMAIRE"]


def test_summarize_applies_preregistered_gate():
    rows = []
    for index in range(100):
        rows.append(
            {
                "audit_case_id": f"c{index}",
                "current_label_origin": "V4.4_TRANSPORT_EXACT_TOP1",
                "sampling_stratum": "HARD",
                "current_adjudication_label": "TOP1_CORRECT",
                "reliable_label": True,
                "is_negative_or_ambiguous": False,
                "is_v48_random_error": False,
                "guard_review": index < 5,
            }
        )
    for index in range(5):
        rows.append(
            {
                "audit_case_id": f"w{index}",
                "current_label_origin": "V4.7_CURRENT_TOP1",
                "sampling_stratum": "RANDOM_POPULATION",
                "current_adjudication_label": "TOP1_WRONG",
                "reliable_label": True,
                "is_negative_or_ambiguous": True,
                "is_v48_random_error": index == 0,
                "guard_review": True,
            }
        )
    report = summarize(pd.DataFrame(rows))
    assert report["verdict"] == "GO_FRESH_V49"
    assert report["counts"]["negative_or_ambiguous_rejected"] == 5
    assert report["rates"]["correct_rejection_rate"] == 0.05


def test_summarize_fails_when_signal_is_too_small():
    frame = pd.DataFrame(
        [
            {
                "audit_case_id": "x",
                "current_label_origin": "V4.7_CURRENT_TOP1",
                "sampling_stratum": "HARD",
                "current_adjudication_label": "TOP1_WRONG",
                "reliable_label": True,
                "is_negative_or_ambiguous": True,
                "is_v48_random_error": True,
                "guard_review": True,
            }
        ]
    )
    assert summarize(frame)["verdict"] == "STOP_SITE_FUNCTION_GUARD"
