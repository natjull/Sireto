from __future__ import annotations

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipe_v6.candidate_store import NormalizedCandidate
from pipe_v6.config import PipelineConfig
from pipe_v6.llm_matcher import filter_candidates_by_category


def _candidate(siret: str, category: str, sources: list[str]) -> NormalizedCandidate:
    return NormalizedCandidate(
        siren=siret[:9],
        siret=siret,
        name="ACME",
        address="1 RUE",
        postcode="75001",
        city="PARIS",
        insee_code="75056",
        legal_nature="5710",
        sources=sources,
        raw_candidates=[],
        category=category,
    )


def _config(**overrides) -> PipelineConfig:
    base = dict(
        crm_path=Path("crm.csv"),
        output_path=Path("out.csv"),
        sqlite_path=Path("cache.sqlite"),
        log_path=Path("log.txt"),
        sirene_api_url="https://api",
        sirene_token="token",
        rne_api_url="https://rne",
        rne_client_id="id",
        rne_client_secret="secret",
        rne_token_url="https://rne/token",
        qwant_base_url="https://qwant",
    )
    base.update(overrides)
    return PipelineConfig(**base)  # type: ignore[arg-type]


class TestFilterCandidatesByCategory:
    def test_public_filters_public_only(self):
        candidates = [
            _candidate("12345678900001", "PUBLIC", ["RNE"]),
            _candidate("12345678900002", "PRIVE", ["RNE"]),
        ]
        cfg = _config()

        res = filter_candidates_by_category("PUBLIC", candidates, cfg)

        assert len(res) == 1
        assert res[0].category == "PUBLIC"

    def test_public_fallback_to_all_when_empty(self):
        candidates = [
            _candidate("12345678900002", "PRIVE", ["RNE"]),
        ]
        cfg = _config(category_filter_fallback=True)

        res = filter_candidates_by_category("PUBLIC", candidates, cfg)

        assert res == candidates  # fallback to all

    def test_public_no_fallback_when_disabled(self):
        candidates = [
            _candidate("12345678900002", "PRIVE", ["RNE"]),
        ]
        cfg = _config(category_filter_fallback=False)

        res = filter_candidates_by_category("PUBLIC", candidates, cfg)

        assert res == []  # no fallback

    def test_prive_filters_prive_only(self):
        candidates = [
            _candidate("12345678900001", "PUBLIC", ["RNE"]),
            _candidate("12345678900002", "PRIVE", ["RNE", "DATAGOUV"]),
        ]
        cfg = _config()

        res = filter_candidates_by_category("PRIVE", candidates, cfg)

        assert len(res) == 1
        assert res[0].category == "PRIVE"

    def test_inconnu_keeps_all(self):
        candidates = [
            _candidate("12345678900001", "PUBLIC", ["RNE"]),
            _candidate("12345678900002", "PRIVE", ["RNE", "DATAGOUV"]),
        ]
        cfg = _config()

        res = filter_candidates_by_category("INCONNU", candidates, cfg)

        assert res == candidates

    def test_equipement_urbain_returns_empty(self):
        candidates = [
            _candidate("12345678900001", "PUBLIC", ["RNE"]),
        ]
        cfg = _config()

        res = filter_candidates_by_category("EQUIPEMENT_URBAIN", candidates, cfg)

        assert res == []

    def test_filter_disabled_returns_all(self):
        candidates = [
            _candidate("12345678900001", "PUBLIC", ["RNE"]),
            _candidate("12345678900002", "PRIVE", ["RNE"]),
        ]
        cfg = _config(category_filter_enabled=False)

        res = filter_candidates_by_category("PUBLIC", candidates, cfg)

        assert res == candidates

    def test_limit_and_sort_by_sources(self):
        candidates = [
            _candidate("12345678900001", "PRIVE", ["RNE"]),
            _candidate("12345678900002", "PRIVE", ["RNE", "DATAGOUV", "QWANT"]),
            _candidate("12345678900003", "PRIVE", ["RNE", "DATAGOUV"]),
        ]
        cfg = _config(max_candidates_llm_matcher=2)

        res = filter_candidates_by_category("PRIVE", candidates, cfg)

        assert len(res) == 2
        # Sorted by number of sources desc: 00002 (3 sources) then 00003 (2 sources)
        assert [c.siret for c in res] == ["12345678900002", "12345678900003"]
