import json
from pathlib import Path

import pandas as pd

from pipe_v6.config import PipelineConfig
from pipe_v6.external_sources import (
    QwantApiError,
    qwant_search,
    search_qwant_sites,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_fixture(name: str) -> dict:
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


def test_qwant_search_returns_items(monkeypatch):
    fixture = _load_fixture("qwant_pappers_sample.json")

    def fake_get(url, params, **kwargs):
        return fixture

    monkeypatch.setattr("pipe_v6.external_sources._http_get_qwant", fake_get)

    config = _dummy_config()
    items = qwant_search("test query", config)

    assert len(items) == 2
    assert items[0]["url"].startswith("https://www.pappers.fr")


def test_search_qwant_sites_three_sources(monkeypatch):
    fixtures = {
        "pappers": _load_fixture("qwant_pappers_sample.json"),
        "annuaire": _load_fixture("qwant_annuaire_sample.json"),
        "societe": _load_fixture("qwant_societe_sample.json"),
    }

    def fake_qwant_search(query, config, logger=None):
        if "pappers" in query:
            return fixtures["pappers"]["data"]["result"]["items"]
        if "annuaire-entreprises" in query:
            return fixtures["annuaire"]["data"]["result"]["items"]
        if "societe.com" in query:
            return fixtures["societe"]["data"]["result"]["items"]
        raise QwantApiError("unexpected query")

    monkeypatch.setattr("pipe_v6.external_sources.qwant_search", fake_qwant_search)

    config = _dummy_config()
    row = pd.Series(
        {
            "crm_name": "ACME ENERGIE",
            "street_number": "12",
            "street_name": "RUE DE TEST",
            "postcode": "75001",
        }
    )

    candidates = search_qwant_sites(row, config)

    assert len(candidates) == 6  # 2 per source
    sources = {c.source for c in candidates}
    assert sources == {"QWANT_PAPPERS", "QWANT_ANNUAIRE", "QWANT_SOCIETE"}


def test_search_qwant_sites_limits_per_source(monkeypatch):
    def fake_qwant_search(query, config, logger=None):
        items = []
        for i in range(20):
            siret = f"1000000000{i:02d}{i:02d}"
            items.append(
                {
                    "url": f"https://example.com/{siret}",
                    "title": f"Item {i}",
                }
            )
        return items

    monkeypatch.setattr("pipe_v6.external_sources.qwant_search", fake_qwant_search)

    config = _dummy_config()
    config.max_candidates_per_source = 3

    row = pd.Series(
        {
            "crm_name": "ACME",
            "street_number": "1",
            "street_name": "RUE",
            "postcode": "75002",
        }
    )

    candidates = search_qwant_sites(row, config)

    assert len(candidates) == 9  # 3 per source * 3 sources
    assert all(c.siret is not None for c in candidates)


def test_search_qwant_sites_skips_no_siren(monkeypatch):
    def fake_qwant_search(query, config, logger=None):
        return [
            {"url": "https://example.com/no-id", "title": "No id"},
            {"url": "https://example.com/also-no-id", "title": "No id 2"},
        ]

    monkeypatch.setattr("pipe_v6.external_sources.qwant_search", fake_qwant_search)

    config = _dummy_config()
    row = pd.Series({"crm_name": "ACME"})

    candidates = search_qwant_sites(row, config)

    assert candidates == []
