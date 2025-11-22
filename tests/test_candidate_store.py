"""Tests for candidate_store (task 4.1)."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipe_v6.candidate_store import (
    RAW_CANDIDATE_SOURCES,
    CandidateSource,
    InvalidCandidateError,
    RawCandidate,
    candidate_key,
    create_raw_candidate,
    NormalizedCandidate,
)


class TestRawCandidateCreation:
    def test_create_valid_candidate_with_siret(self):
        cand = create_raw_candidate(
            source="RNE",
            siret="12345678901234",
            siren="123456789",
            label="ACME SAS",
            url="https://example.com",
        )

        assert cand.source == "RNE"
        assert cand.siret == "12345678901234"
        assert cand.siren == "123456789"
        assert cand.label == "ACME SAS"

    def test_normalize_siret_with_spaces(self):
        cand = create_raw_candidate(
            source="DATAGOUV",
            siret=" 123 456 789 01234 ",
        )

        assert cand.siret == "12345678901234"

    def test_invalid_source_raises(self):
        with pytest.raises(InvalidCandidateError, match="Unknown source"):
            create_raw_candidate(source="UNKNOWN", siren="123456789")

    def test_invalid_siret_length_raises(self):
        with pytest.raises(InvalidCandidateError, match="Invalid identifier"):
            create_raw_candidate(source="RNE", siret="12345")

    def test_candidate_without_identifiers_allowed(self):
        cand = create_raw_candidate(
            source="QWANT_ANNUAIRE",
            label="Some business",
            url="https://example.com",
        )

        assert cand.siren is None
        assert cand.siret is None
        assert cand.label == "Some business"


class TestCandidateKey:
    def test_key_with_siret(self):
        cand = create_raw_candidate(
            source="RNE",
            siret="12345678901234",
            siren="123456789",
        )

        assert candidate_key(cand) == ("siret", "12345678901234")

    def test_key_with_siren_only(self):
        cand = create_raw_candidate(
            source="RNE",
            siren="123456789",
        )

        assert candidate_key(cand) == ("siren", "123456789")

    def test_key_without_identifiers(self):
        cand = create_raw_candidate(
            source="QWANT_ANNUAIRE",
            label="Unknown business",
        )

        assert candidate_key(cand) is None


class TestCandidateSerialization:
    def test_to_dict_includes_all_fields(self):
        cand = create_raw_candidate(
            source="RNE",
            siret="12345678901234",
            label="ACME",
            extra={"score": 0.95, "rank": 1},
        )

        data = cand.to_dict()

        assert data["source"] == "RNE"
        assert data["siret"] == "12345678901234"
        assert data["label"] == "ACME"
        assert data["extra"]["score"] == 0.95


class TestNormalizedCandidate:
    def test_create_normalized_candidate_all_fields(self):
        raw = create_raw_candidate(source="RNE", siret="12345678901234", label="ACME")

        norm = NormalizedCandidate(
            siren="123456789",
            siret="12345678901234",
            name="ACME",
            address="1 RUE DE PARIS",
            postcode="75001",
            city="PARIS",
            insee_code="75056",
            legal_nature="5710",
            sources=["RNE", "DATAGOUV"],
            raw_candidates=[raw],
        )

        assert norm.siren == "123456789"
        assert norm.siret == "12345678901234"
        assert norm.sources == ["RNE", "DATAGOUV"]
        assert norm.raw_candidates[0].label == "ACME"

    def test_normalized_candidate_is_frozen(self):
        norm = NormalizedCandidate(
            siren="123456789",
            siret=None,
            name="ACME",
            address="1 RUE",
            postcode="75001",
            city="PARIS",
            insee_code=None,
            legal_nature=None,
            sources=["RNE"],
            raw_candidates=[],
        )

        with pytest.raises(AttributeError):
            # frozen dataclass should prevent mutation
            norm.name = "NEW"

    def test_normalized_candidate_to_dict(self):
        raw = create_raw_candidate(source="DATAGOUV", siret="12345678901234", label="ACME")

        norm = NormalizedCandidate(
            siren="123456789",
            siret="12345678901234",
            name="ACME",
            address="1 RUE",
            postcode="75001",
            city="PARIS",
            insee_code="75056",
            legal_nature=None,
            sources=["DATAGOUV"],
            raw_candidates=[raw],
        )

        data = norm.to_dict()
        assert data["siren"] == "123456789"
        assert data["siret"] == "12345678901234"
        assert data["sources"] == ["DATAGOUV"]
        assert data["raw_candidates"][0]["source"] == "DATAGOUV"
