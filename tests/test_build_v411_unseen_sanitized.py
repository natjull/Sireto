from __future__ import annotations

import pandas as pd
import pytest

from scripts.build_v411_unseen_sanitized import (
    QUERY_COLUMNS,
    SOURCE_COLUMNS,
    _query_id,
    validate_query_schema,
    validate_sealed_mapping,
)


def _query_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "query_id": _query_id(1),
                "crm_record_id": _query_id(1),
                "crm_name": "ENTREPRISE",
                "crm_address": "1 RUE A",
                "crm_postcode": "75001",
                "crm_city": "PARIS",
                "crm_insee": "75056",
            }
        ],
        columns=QUERY_COLUMNS,
    )


def test_physical_projection_never_requests_identifier_columns() -> None:
    lowered = {name.lower() for name in SOURCE_COLUMNS}
    assert "siret" not in lowered
    assert "input_siret_norm" not in lowered
    assert "row_fingerprint_sha256" not in lowered
    assert "service id" not in lowered


def test_query_id_is_deterministic_and_contains_no_source_value() -> None:
    assert _query_id(42) == _query_id(42)
    assert _query_id(42) != _query_id(43)
    assert "42" not in _query_id(42)


def test_sanitized_schema_rejects_forbidden_or_empty_fields() -> None:
    frame = _query_frame()
    validate_query_schema(frame, canonical=False)
    forbidden = frame.assign(candidate_score=0.5)
    with pytest.raises(ValueError, match="schema changed"):
        validate_query_schema(forbidden, canonical=False)
    empty = frame.copy()
    empty.loc[0, "crm_address"] = ""
    with pytest.raises(ValueError, match="required CRM field empty"):
        validate_query_schema(empty, canonical=False)


def test_sealed_mapping_binds_ids_rows_and_exposed_cohort() -> None:
    queries = pd.DataFrame(
        [
            {
                **_query_frame().iloc[0].to_dict(),
                "query_id": _query_id(row),
                "crm_record_id": _query_id(row),
            }
            for row in (1102, 2000)
        ],
        columns=QUERY_COLUMNS,
    )
    mapping = pd.DataFrame(
        [
            {
                "query_id": _query_id(1102),
                "source_row_number": 1102,
                "cohort": "EXPOSED_3",
            },
            {
                "query_id": _query_id(2000),
                "source_row_number": 2000,
                "cohort": "DESCRIPTIVE_UNSEEN_BLIND_222",
            },
        ]
    )
    validate_sealed_mapping(mapping, queries, canonical=False)
    mapping.loc[0, "cohort"] = "DESCRIPTIVE_UNSEEN_BLIND_222"
    with pytest.raises(ValueError, match="cohort mapping changed"):
        validate_sealed_mapping(mapping, queries, canonical=False)
