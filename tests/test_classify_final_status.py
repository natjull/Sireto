from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipe_v6.config import PipelineConfig
from pipe_v6.llm_matcher import LLMMatchDecision, classify_final_status


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


class TestClassifyFinalStatus:
    def test_best_match_high_confidence(self):
        cfg = _config(confidence_auto_match=0.85, confidence_review_min=0.6)
        decision = LLMMatchDecision("BEST_MATCH", "12345678901234", 0.9, "OK")
        assert classify_final_status(decision, cfg) == "MATCH"

    def test_best_match_medium_confidence(self):
        cfg = _config(confidence_auto_match=0.85, confidence_review_min=0.6)
        decision = LLMMatchDecision("BEST_MATCH", "12345678901234", 0.7, "OK")
        assert classify_final_status(decision, cfg) == "REVIEW"

    def test_best_match_low_confidence(self):
        cfg = _config(confidence_auto_match=0.85, confidence_review_min=0.6)
        decision = LLMMatchDecision("BEST_MATCH", "12345678901234", 0.5, "OK")
        assert classify_final_status(decision, cfg) == "NO_MATCH"

    def test_no_match_always_no_match(self):
        cfg = _config(confidence_auto_match=0.85, confidence_review_min=0.6)
        decision = LLMMatchDecision("NO_MATCH", None, 0.9, "No good match")
        assert classify_final_status(decision, cfg) == "NO_MATCH"
