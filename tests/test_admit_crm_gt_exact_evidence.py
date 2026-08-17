from __future__ import annotations

import pandas as pd

from scripts.admit_crm_gt_exact_evidence import crm_number_and_repeat, strict_admission


def _row(**overrides: object) -> dict[str, object]:
    row = {
        "name_norm_exact": 1.0,
        "name_jaro_max": 1.0,
        "name_token_overlap_max": 1.0,
        "name_contains_crm_max": 0.0,
        "name_crm_contains_cand_max": 0.0,
        "acronym_match_max": 0.0,
        "crm_address": "1 B RUE DES FLEURS",
        "numeroVoie": "1",
        "indiceRepetition": "BIS",
        "postcode_match": 1.0,
        "insee_match": 1.0,
        "street_name_jaro": 1.0,
    }
    row.update(overrides)
    return row


def test_number_parser_preserves_repetition() -> None:
    assert crm_number_and_repeat("1B RUE DES FLEURS") == ("1", "B")
    assert crm_number_and_repeat("12 TER AVENUE A") == ("12", "TER")
    assert crm_number_and_repeat("RUE SANS NUMERO") == ("", "")


def test_admission_requires_identity_and_exact_site() -> None:
    frame = pd.DataFrame(
        [
            _row(),
            _row(name_norm_exact=0.0, name_jaro_max=0.86, name_token_overlap_max=0.25),
            _row(indiceRepetition=""),
            _row(insee_match=0.0),
        ]
    )
    result = strict_admission(frame)
    assert result["admission_status"].tolist() == [
        "ADMIT_EXACT_SIRET",
        "QUARANTINE_INSUFFICIENT_DIRECT_EVIDENCE",
        "QUARANTINE_INSUFFICIENT_DIRECT_EVIDENCE",
        "QUARANTINE_INSUFFICIENT_DIRECT_EVIDENCE",
    ]
