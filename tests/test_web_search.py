from pathlib import Path

import pandas as pd

from pipe_v6.config import PipelineConfig
from pipe_v6.external_sources import WebSearchResult, web_search, search_web_sites


def _dummy_config(tmp_path: Path) -> PipelineConfig:
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
        qwant_base_url="https://api.qwant.com/v3/search/web",
        datagouv_api_url="https://recherche-entreprises.api.gouv.fr",
        google_api_key="test-google",
        google_cse_id="test-cse",
        brave_api_key="",
        web_cache_path=tmp_path / "web_cache.sqlite",
        web_search_enabled=True,
    )


def test_web_search_returns_items(monkeypatch, tmp_path):
    def fake_google_search(query, config, logger):
        return 200, [
            {
                "url": "https://www.pappers.fr/entreprise/12345678900011",
                "title": "ACME",
                "snippet": "ACME 12345678900011",
                "rank": 0,
            }
        ]

    monkeypatch.setattr("pipe_v6.external_sources._google_search", fake_google_search)

    config = _dummy_config(tmp_path)
    result = web_search("test query", config)

    assert result.provider == "google"
    assert len(result.items) == 1
    assert result.items[0]["url"].startswith("https://www.pappers.fr")


def test_search_web_sites_multisite(monkeypatch, tmp_path):
    items = [
        {"url": "https://www.pappers.fr/entreprise/12345678900011", "title": "A", "snippet": "A"},
        {"url": "https://annuaire-entreprises.data.gouv.fr/etablissement/12345678900022", "title": "B", "snippet": "B"},
        {"url": "https://www.societe.com/societe/acme-12345678900033.html", "title": "C", "snippet": "C"},
        {"url": "https://entreprises.lefigaro.fr/acme-12345678900044", "title": "D", "snippet": "D"},
    ]

    def fake_web_search(query, config, logger=None):
        return WebSearchResult(provider="google", items=items, from_cache=True)

    monkeypatch.setattr("pipe_v6.external_sources.web_search", fake_web_search)

    config = _dummy_config(tmp_path)
    row = pd.Series(
        {
            "crm_name": "ACME ENERGIE",
            "street_number": "12",
            "street_name": "RUE DE TEST",
            "postcode": "75001",
        }
    )

    candidates = search_web_sites(row, config)
    sources = {c.source for c in candidates}

    assert sources == {"WEB_PAPPERS", "WEB_ANNUAIRE", "WEB_SOCIETE", "WEB_LEFIGARO"}


def test_search_web_sites_limits_per_source(monkeypatch, tmp_path):
    items = []
    for i in range(10):
        siret_p = f"1000000000{i:02d}{i:02d}"
        siret_a = f"2000000000{i:02d}{i:02d}"
        items.append({"url": f"https://www.pappers.fr/entreprise/{siret_p}", "title": f"Item {i}"})
        items.append({"url": f"https://annuaire-entreprises.data.gouv.fr/etablissement/{siret_a}", "title": f"Item {i}"})

    def fake_web_search(query, config, logger=None):
        return WebSearchResult(provider="google", items=items, from_cache=True)

    monkeypatch.setattr("pipe_v6.external_sources.web_search", fake_web_search)

    config = _dummy_config(tmp_path)
    config.max_candidates_per_source = 2

    row = pd.Series(
        {
            "crm_name": "ACME",
            "street_number": "1",
            "street_name": "RUE",
            "postcode": "75002",
        }
    )

    candidates = search_web_sites(row, config)

    assert len(candidates) == 4  # 2 per source * 2 sources used here
    assert all(c.siret is not None for c in candidates)


def test_search_web_sites_skips_no_siren(monkeypatch, tmp_path):
    items = [
        {"url": "https://www.pappers.fr/entreprise/no-id", "title": "No id", "snippet": "none"},
        {"url": "https://societe.com/no-id", "title": "No id 2", "snippet": "none"},
    ]

    def fake_web_search(query, config, logger=None):
        return WebSearchResult(provider="google", items=items, from_cache=True)

    monkeypatch.setattr("pipe_v6.external_sources.web_search", fake_web_search)

    config = _dummy_config(tmp_path)
    row = pd.Series({"crm_name": "ACME"})

    candidates = search_web_sites(row, config)

    assert candidates == []
