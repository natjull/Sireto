"""Tests for the RNE client and mapping (task 4.2)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipe_v6.config import PipelineConfig
from pipe_v6.rne_client import RneApiError, RneClient, _map_company_to_candidates
from pipe_v6.external_sources import search_rne


def _dummy_config() -> PipelineConfig:
    return PipelineConfig(
        crm_path=Path("crm.csv"),
        output_path=Path("out.csv"),
        sqlite_path=Path("cache.sqlite"),
        log_path=Path("logs.log"),
        sirene_api_url="https://example",
        sirene_token="token",
        rne_api_url="https://api.inpi.fr/api/rne/",
        rne_token_url="https://api.inpi.fr/api/sso/oauth2/token",
        rne_client_id="id",
        rne_client_secret="secret",
        qwant_base_url="https://example",
    )


def _load_fixture(name: str) -> dict[str, Any]:
    path = PROJECT_ROOT / "tests" / "fixtures" / name
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


class DummyLogger:
    def __init__(self):
        self.messages = []

    def info(self, *args, **kwargs):
        self.messages.append(("INFO", args))

    def debug(self, *args, **kwargs):
        self.messages.append(("DEBUG", args))

    def warning(self, *args, **kwargs):
        self.messages.append(("WARN", args))

    def error(self, *args, **kwargs):
        self.messages.append(("ERROR", args))


def test_map_company_to_candidates_from_fixture():
    data = _load_fixture("rne_sample.json")
    company = data["results"][0]

    cands = _map_company_to_candidates(
        company,
        query="ITAFRAN 69960",
        rank_offset=0,
        store_raw=False,
    )

    assert len(cands) == 3
    siret_values = {c.siret for c in cands}
    assert "12345678900012" in siret_values
    assert "12345678900020" in siret_values
    assert "12345678900038" in siret_values
    assert all(c.source == "RNE" for c in cands)
    assert all(c.siren == "123456789" for c in cands)


def test_search_rne_limits_results(monkeypatch):
    data = _load_fixture("rne_sample.json")
    company = data["results"][0]

    # Mock token retrieval
    def fake_get_token(self):
        return "TOKEN"

    # Mock HTTP GET to always return the fixture (2 pages max)
    call_count = {"value": 0}

    def fake_search(self, name, postcode, max_results):
        call_count["value"] += 1
        # Simulate multiple pages by duplicating the company
        return [company] * 5

    monkeypatch.setattr(RneClient, "_get_token", fake_get_token)
    monkeypatch.setattr(RneClient, "search_companies", fake_search)

    config = _dummy_config()
    config.max_candidates_per_source = 2
    logger = DummyLogger()

    results = search_rne(
        normalized_name="ITAFRAN",
        city="CORBAS",
        config=config,
        logger=logger,
        postcode="69960",
    )

    assert len(results) == 2  # limited by max_candidates_per_source
    assert call_count["value"] == 1


def test_token_cache_reuse(monkeypatch):
    token_calls = {"count": 0}

    def fake_post(url, data, timeout=15.0, max_retries=3, logger=None):
        token_calls["count"] += 1
        return {"access_token": "TOKEN", "expires_in": 3600}

    def fake_get_json(url, headers, timeout=20.0, max_retries=3, logger=None):
        return {"results": []}

    monkeypatch.setattr("pipe_v6.rne_client._http_post_form", fake_post)
    monkeypatch.setattr("pipe_v6.rne_client._http_get_json", fake_get_json)

    config = _dummy_config()
    client = RneClient(config=config, logger=DummyLogger())

    client.search_companies("A", None, max_results=5)
    client.search_companies("B", None, max_results=5)

    assert token_calls["count"] == 1  # token reused


def test_raises_on_invalid_results(monkeypatch):
    def fake_post(url, data, timeout=15.0, max_retries=3, logger=None):
        return {"access_token": "TOKEN", "expires_in": 3600}

    def fake_get_json(url, headers, timeout=20.0, max_retries=3, logger=None):
        return {"unexpected": []}

    monkeypatch.setattr("pipe_v6.rne_client._http_post_form", fake_post)
    monkeypatch.setattr("pipe_v6.rne_client._http_get_json", fake_get_json)

    config = _dummy_config()
    client = RneClient(config=config, logger=DummyLogger())

    with pytest.raises(RneApiError):
        client.search_companies("A", None, max_results=5)

