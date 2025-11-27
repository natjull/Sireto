from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipe_v6.candidate_store import NormalizedCandidate
from pipe_v6.config import PipelineConfig
from pipe_v6.llm_matcher import LLMMatchDecision, decide_match
from pipe_v6.llm_normalizer import NormalizedCRMEntry


class DummyResponse:
    def __init__(self, parsed_json):
        self.parsed_json = parsed_json


class DummyClient:
    def __init__(self, parsed_json):
        self.parsed_json = parsed_json
        self.called = False

    def call_json(self, prompt: str, **_kwargs):
        self.called = True
        return DummyResponse(self.parsed_json)

    def close(self):
        pass


def _cand(num: int, category: str, sources: list[str]) -> NormalizedCandidate:
    siret = f"1234567890{num:04d}"
    return NormalizedCandidate(
        siren=siret[:9],
        siret=siret,
        name=f"ACME-{num}",
        address=f"{num} RUE",
        postcode="75001",
        city="PARIS",
        insee_code="75056",
        legal_nature="5710",
        sources=sources,
        raw_candidates=[],
        category=category,
    )


def _config() -> PipelineConfig:
    return PipelineConfig(
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


def _row():
    return {
        "crm_name": "ACME",
        "street_number": "1",
        "street_name": "RUE",
        "postcode": "75001",
        "city": "PARIS",
    }


def _norm():
    return NormalizedCRMEntry(
        normalized_name="ACME",
        normalized_address="1 RUE 75001",
        category="PRIVE",
    )


class TestDecideMatch:
    def test_no_candidates_short_circuit(self):
        cfg = _config()
        res = decide_match(_row(), _norm(), [], cfg)

        assert res.decision == "NO_MATCH"
        assert res.chosen_siret is None
        assert res.confidence == 0.0

    def test_valid_llm_response(self):
        cfg = _config()
        candidates = [_cand(1, "PRIVE", ["RNE"])]
        client = DummyClient(
            {
                "decision": "BEST_MATCH",
                "chosen_siret": candidates[0].siret,
                "confidence": 0.9,
                "reason": "OK",
            }
        )

        res = decide_match(_row(), _norm(), candidates, cfg, client=client)

        assert client.called is True
        assert res.decision == "BEST_MATCH"
        assert res.chosen_siret == candidates[0].siret
        assert res.confidence == 0.9

    def test_invalid_json_fallback(self):
        cfg = _config()
        candidates = [_cand(1, "PRIVE", ["RNE"])]
        client = DummyClient(parsed_json=None)

        res = decide_match(_row(), _norm(), candidates, cfg, client=client)

        assert res.decision == "NO_MATCH"
        assert res.chosen_siret is None
        assert res.confidence == 0.0
        assert res.reason.startswith("LLM_ERROR")

    def test_exception_fallback(self):
        class FailingClient(DummyClient):
            def call_json(self, prompt: str, **_kwargs):
                raise RuntimeError("boom")

        cfg = _config()
        candidates = [_cand(1, "PRIVE", ["RNE"])]
        client = FailingClient(parsed_json={})

        res = decide_match(_row(), _norm(), candidates, cfg, client=client)

        assert res.decision == "NO_MATCH"
        assert res.chosen_siret is None
        assert res.confidence == 0.0
        assert res.reason.startswith("LLM_ERROR")
