"""Unit tests for the CRM normalizer orchestrator (task 3.3)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import sys

import pandas as pd
import pytest
from unittest.mock import Mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipe_v6.config import PipelineConfig
from pipe_v6.llm_normalizer import (
    NormalizedCRMEntry,
    build_normalizer_prompt,
    normalize_crm_entry,
)


def _dummy_config() -> PipelineConfig:
    return PipelineConfig(
        crm_path=Path("crm.csv"),
        output_path=Path("out.csv"),
        sqlite_path=Path("cache.sqlite"),
        log_path=Path("logs.log"),
        sirene_api_url="https://example",
        sirene_token="token",
        rne_api_url="https://example",
        rne_client_id="id",
        rne_client_secret="secret",
        qwant_base_url="https://example",
    )


def _sample_row(**overrides: Any) -> pd.Series:
    base = {
        "crm_id": "AC001651",
        "crm_name": "ITAFRAN Site 2",
        "street_number": "21",
        "street_name": "AVENUE DE L INDUSTRIE",
        "postcode": "69960",
        "city": "CORBAS",
        "insee_code": "69066",
    }
    base.update(overrides)
    return pd.Series(base)


def test_build_normalizer_prompt_contains_core_fields():
    row = _sample_row()
    prompt = build_normalizer_prompt(row)

    assert "ITAFRAN Site 2" in prompt
    assert "21 AVENUE DE L INDUSTRIE" in prompt
    assert "69960" in prompt
    assert "CORBAS" in prompt
    assert "JSON" in prompt  # ensures the JSON instruction is present


def test_build_normalizer_prompt_mentions_missing_parts():
    row = _sample_row(street_number=pd.NA, postcode=pd.NA)
    prompt = build_normalizer_prompt(row)

    assert "CHAMPS MANQUANTS" in prompt
    assert "00000" in prompt  # placeholder for missing postcode
    assert "0" in prompt  # placeholder for missing street number


def test_normalize_crm_entry_passes_expected_city_and_uses_llm(mocked_normalize_with_llm):
    row = _sample_row(city="CORBAS")
    config = _dummy_config()

    result = normalize_crm_entry(row, config)

    # The mock returns a predictable NormalizedCRMEntry
    assert isinstance(result, NormalizedCRMEntry)
    assert result.category == "PRIVE"

    # Check that expected_city was forwarded to normalize_with_llm
    kwargs = mocked_normalize_with_llm.call_args.kwargs
    assert kwargs["expected_city"] == "CORBAS"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mocked_normalize_with_llm(monkeypatch):
    """Patch normalize_with_llm to avoid real LLM calls."""

    from pipe_v6 import llm_normalizer as module

    def _fake(prompt: str, config, logger=None, client=None, expected_city=None):
        return NormalizedCRMEntry(
            normalized_name="ITAFRAN",
            normalized_address="21 AVENUE DE L INDUSTRIE 69960",
            category="PRIVE",
        )

    mock = Mock(side_effect=_fake)
    monkeypatch.setattr(module, "normalize_with_llm", mock)
    return mock
