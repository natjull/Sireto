import pandas as pd
import pytest

from src.xgb_matcher.v9_adjudication import validate_adjudications


def valid_row(**changes):
    row = {
        "query_id": "q1",
        "label_kind": "NO_MATCH",
        "ground_truth_siret": None,
        "validator": "Alice",
        "validated_at": "2026-07-23",
        "evidence_refs": "SIRENE snapshot query log #123",
        "sirene_snapshot_id": "sirene-2026-07",
        "reference_date": "2026-07-01",
    }
    row.update(changes)
    return row


def test_no_match_is_snapshot_and_human_bound():
    validated = validate_adjudications(pd.DataFrame([valid_row()]))
    assert validated.iloc[0]["label_kind"] == "NO_MATCH"
    with pytest.raises(ValueError, match="human validator"):
        validate_adjudications(pd.DataFrame([valid_row(validator="")]))
    with pytest.raises(ValueError, match="reference_date"):
        validate_adjudications(pd.DataFrame([valid_row(reference_date="")]))


def test_llm_cannot_replace_evidence_or_exact_siret():
    with pytest.raises(ValueError, match="evidence"):
        validate_adjudications(pd.DataFrame([valid_row(evidence_refs="")]))
    with pytest.raises(ValueError, match="require a SIRET"):
        validate_adjudications(
            pd.DataFrame([valid_row(label_kind="MATCH_EXACT")])
        )
