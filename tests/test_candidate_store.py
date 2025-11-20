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

