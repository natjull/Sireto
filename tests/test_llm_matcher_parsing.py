from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipe_v6.candidate_store import NormalizedCandidate
from pipe_v6.llm_matcher import (
    LLMMatchDecision,
    MatchDecisionParseError,
    parse_match_decision,
)


def _candidate(siret: str) -> NormalizedCandidate:
    return NormalizedCandidate(
        siren="123456789",
        siret=siret,
        name="ACME",
        address="1 RUE",
        postcode="75001",
        city="PARIS",
        insee_code="75056",
        legal_nature="5710",
        sources=["RNE"],
        raw_candidates=[],
        category="PRIVE",
    )


class TestParseMatchDecision:
    def test_valid_best_match(self):
        candidates = [_candidate("12345678901234")]
        data = {
            "decision": "BEST_MATCH",
            "chosen_siret": "12345678901234",
            "confidence": 0.9,
            "reason": "Bonne correspondance",
        }

        with caplog.at_level(logging.WARNING, logger="pipe_v6.llm_matcher"):
            res = parse_match_decision(data, candidates)

        assert isinstance(res, LLMMatchDecision)
        assert res.decision == "BEST_MATCH"
        assert res.chosen_siret == "12345678901234"
        assert res.confidence == 0.9
        assert res.reason == "Bonne correspondance"

    def test_confidence_clamped(self):
        candidates = [_candidate("12345678901234")]
        data = {
            "decision": "BEST_MATCH",
            "chosen_siret": "12345678901234",
            "confidence": 1.5,  # will be clamped to 1.0
        }

        res = parse_match_decision(data, candidates)
        assert res.confidence == 1.0

    def test_missing_chosen_siret_raises(self):
        candidates = [_candidate("12345678901234")]
        data = {
            "decision": "BEST_MATCH",
            "confidence": 0.8,
        }

        with pytest.raises(MatchDecisionParseError, match="requires chosen_siret"):
            parse_match_decision(data, candidates)

    def test_invalid_decision_raises(self):
        candidates = [_candidate("12345678901234")]
        data = {
            "decision": "MAYBE",
            "confidence": 0.5,
        }

        with pytest.raises(MatchDecisionParseError, match="Invalid decision"):
            parse_match_decision(data, candidates)

    def test_invalid_siret_forces_no_match(self, caplog):
        candidates = [_candidate("12345678901234")]
        data = {
            "decision": "BEST_MATCH",
            "chosen_siret": "00000000000000",
            "confidence": 0.9,
        }

        res = parse_match_decision(data, candidates)

        assert res.decision == "NO_MATCH"
        assert res.chosen_siret is None
        assert res.confidence == 0.0
        assert "not in candidates" in (res.reason or "")

        assert "Invalid SIRET from LLM" in caplog.text

    def test_no_match_forces_chosen_siret_none(self):
        candidates = [_candidate("12345678901234")]
        data = {
            "decision": "NO_MATCH",
            "chosen_siret": "12345678901234",  # should be ignored
            "confidence": 0.2,
        }

        res = parse_match_decision(data, candidates)

        assert res.decision == "NO_MATCH"
        assert res.chosen_siret is None
        assert res.confidence == 0.2
