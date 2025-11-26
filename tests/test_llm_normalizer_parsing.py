"""Unit tests for the JSON parser of the CRM LLM normalizer (task 3.2)."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipe_v6.llm_normalizer import (  # noqa: E402
    NormalizationParseError,
    NormalizedCRMEntry,
    parse_llm_response,
    parse_normalizer_output,
)
from pipe_v6.llm_utils import LLMResponse  # noqa: E402


def _payload(**overrides):
    base = {
        "normalized_name": "ECOLE ELEMENTAIRE JULES FERRY",
        "normalized_address": "12 RUE DE PARIS 75011",
        "category": "PUBLIC",
    }
    base.update(overrides)
    return base


def test_parse_normalizer_output_success():
    result = parse_normalizer_output(_payload())

    assert isinstance(result, NormalizedCRMEntry)
    assert result.normalized_name == "ECOLE ELEMENTAIRE JULES FERRY"
    assert result.normalized_address == "12 RUE DE PARIS 75011"
    assert result.category == "PUBLIC"


def test_parse_normalizer_output_rejects_lowercase_name():
    with pytest.raises(NormalizationParseError):
        parse_normalizer_output(_payload(normalized_name="Ecole Test"))


def test_parse_normalizer_output_requires_postcode():
    with pytest.raises(NormalizationParseError):
        parse_normalizer_output(_payload(normalized_address="12 RUE DE PARIS"))


def test_parse_normalizer_output_rejects_unknown_category():
    with pytest.raises(NormalizationParseError):
        parse_normalizer_output(_payload(category="AUTRE"))


def test_parse_normalizer_output_accepts_alphanumeric_number_suffix():
    result = parse_normalizer_output(
        _payload(normalized_address="2BIS RUE DE CATALOGNE 69150")
    )

    assert result.normalized_address == "2BIS RUE DE CATALOGNE 69150"


def test_parse_normalizer_output_accepts_spaced_suffix():
    result = parse_normalizer_output(
        _payload(normalized_address="12 TER RUE DES FLEURS 75011")
    )

    assert result.normalized_address == "12 TER RUE DES FLEURS 75011"


def test_parse_normalizer_output_cleans_tokens_and_city():
    payload = _payload(
        normalized_name="SAS CABLAGE FIBRE AGENCE LYON",
    )
    result = parse_normalizer_output(payload, expected_city="Lyon")

    assert result.normalized_name == "CABLAGE FIBRE"


def test_parse_llm_response_without_json_raises():
    response = LLMResponse(
        text="{}",
        model="gpt-oss:20b",
        raw={"response": "{}"},
        parsed_json=None,
    )

    with pytest.raises(NormalizationParseError):
        parse_llm_response(response)
