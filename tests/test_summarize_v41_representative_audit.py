import pandas as pd

from scripts.summarize_v41_representative_audit import summarize


def test_conservative_precision_is_an_upper_bound_not_a_claim():
    registry = pd.DataFrame(
        [
            {
                "audit_case_id": "a",
                "service_id": "A",
                "sampling_stratum": "RANDOM_POPULATION",
                "decision": "AUTO_MATCH",
                "predicted_siret": "11111111100011",
            },
            {
                "audit_case_id": "b",
                "service_id": "B",
                "sampling_stratum": "RANDOM_POPULATION",
                "decision": "AUTO_MATCH",
                "predicted_siret": "22222222200022",
            },
        ]
    )
    adjudications = pd.DataFrame(
        [
            {
                "audit_case_id": "a",
                "service_id": "A",
                "label_kind": "MATCH_EXACT",
                "ground_truth_siret": "11111111100011",
            },
            {
                "audit_case_id": "b",
                "service_id": "B",
                "label_kind": "UNRESOLVED",
                "ground_truth_siret": None,
            },
        ]
    )
    top10 = pd.DataFrame(
        [
            {
                "service_id": "A",
                "candidate_siret": "11111111100011",
                "rank": 1,
            }
        ]
    )
    contradictions = pd.DataFrame(
        [
            {
                "audit_case_id": "b",
                "service_id": "B",
                "predicted_siret": "22222222200022",
                "contradiction_code": "WRONG_ENTITY",
                "review_status": "AI_PROVISIONAL",
            }
        ]
    )
    _, result = summarize(
        registry=registry,
        adjudications=adjudications,
        top10=top10,
        contradictions=contradictions,
    )
    assert result["random_population"][
        "conservative_auto_precision_upper_bound"
    ] == 0.5
    assert result["interpretation"]["precision_claim_allowed"] is False
