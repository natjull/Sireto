"""Tests for the DataGouv (annuaire-entreprises) client mapping and search."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
import sys

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipe_v6.candidate_store import create_raw_candidate
from pipe_v6.config import PipelineConfig
from pipe_v6.external_sources import (
    DataGouvApiError,
    _http_get_datagouv,
    _map_datagouv_result_to_candidates,
    search_datagouv,
)


def _load_fixture(name: str) -> dict[str, Any]:
    path = PROJECT_ROOT / "tests" / "fixtures" / name
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


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
        qwant_base_url="https://api.qwant.com/api/search/web",
        datagouv_api_url="https://recherche-entreprises.api.gouv.fr",
    )


def test_map_datagouv_result_to_candidates_from_fixture():
    fixture = _load_fixture("datagouv_sample.json")
    result = fixture["results"][0]

    candidates = _map_datagouv_result_to_candidates(
        result,
        query="ACME 75002",
        rank_offset=0,
        store_raw=True,
    )

    assert len(candidates) == 3  # siège + 2 matching
    assert all(c.source == "DATAGOUV" for c in candidates)
    assert all(c.siren == "123456789" for c in candidates)
    assert candidates[0].siret == "12345678900012"
    assert candidates[0].extra["rank"] == 0
    assert "adresse" in candidates[0].extra


def test_search_datagouv_limits_results(monkeypatch):
    fixture = _load_fixture("datagouv_sample.json")

    def fake_get(url, params, **kwargs):
        return fixture

    monkeypatch.setattr("pipe_v6.external_sources._http_get_datagouv", fake_get)

    config = _dummy_config()
    config.max_candidates_per_source = 2

    candidates = search_datagouv(
        "ACME",
        "PARIS",
        config,
        postcode="75002",
    )

    assert len(candidates) == 2


def test_search_datagouv_deduplicates_candidates(monkeypatch):
    response = {
        "results": [
            {
                "siren": "123456789",
                "nom_complet": "ACME ENERGIE",
                "etat_administratif": "A",
                "nature_juridique": "5710",
                "activite_principale": "43.21A",
                "siege": {"siret": "12345678900012", "adresse": "1 RUE"},
                "matching_etablissements": [
                    {"siret": "12345678900012", "adresse": "1 RUE"},
                    {"siret": "12345678900020", "adresse": "2 AVE"},
                ],
            }
        ],
        "total_results": 1,
        "page": 1,
        "per_page": 10,
        "total_pages": 1,
    }

    def fake_get(url, params, **kwargs):
        return response

    monkeypatch.setattr("pipe_v6.external_sources._http_get_datagouv", fake_get)

    config = _dummy_config()
    config.max_candidates_per_source = 5

    candidates = search_datagouv("ACME", "PARIS", config)

    assert len(candidates) == 2
    assert {c.siret for c in candidates} == {"12345678900012", "12345678900020"}


def test_search_datagouv_handles_empty_response(monkeypatch):
    def fake_get(url, params, **kwargs):
        return {"results": [], "total_results": 0, "page": 1, "per_page": 10}

    monkeypatch.setattr("pipe_v6.external_sources._http_get_datagouv", fake_get)

    config = _dummy_config()
    candidates = search_datagouv("NONE", "PARIS", config)

    assert candidates == []


def test_search_datagouv_raises_on_invalid_response(monkeypatch):
    def fake_get(url, params, **kwargs):
        return {"error": "bad"}

    monkeypatch.setattr("pipe_v6.external_sources._http_get_datagouv", fake_get)

    config = _dummy_config()
    with pytest.raises(DataGouvApiError, match="missing 'results'"):
        search_datagouv("ACME", "PARIS", config)


def test_http_get_datagouv_retries_on_429(monkeypatch):
    payload = {"results": [], "total_results": 0}
    calls = {"count": 0}

    def fake_urlopen(request, timeout=0):
        calls["count"] += 1
        if calls["count"] == 1:
            fp = io.BytesIO(b"rate limited")
            raise HTTPError(request.full_url, 429, "Too Many Requests", {}, fp)
        class FakeResp:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps(payload).encode("utf-8")

        return FakeResp()

    monkeypatch.setattr("pipe_v6.external_sources.urlopen", fake_urlopen)
    monkeypatch.setattr("pipe_v6.external_sources.time.sleep", lambda *_args, **_kwargs: None)

    result = _http_get_datagouv(
        "https://recherche-entreprises.api.gouv.fr/search",
        {"q": "ACME"},
        max_retries=2,
        timeout=5.0,
        connect_timeout=1.0,
    )

    assert result == payload
    assert calls["count"] == 2
