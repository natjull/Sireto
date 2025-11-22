"""Tests for candidate_store (task 4.1)."""

from __future__ import annotations

import sqlite3
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
    enrich_candidates_from_sirene,
)
from pipe_v6 import sirene_cache


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
            category="PRIVE",
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
            category="INCONNU",
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
            category="PRIVE",
        )

        data = norm.to_dict()
        assert data["siren"] == "123456789"
        assert data["siret"] == "12345678901234"
        assert data["sources"] == ["DATAGOUV"]
        assert data["raw_candidates"][0]["source"] == "DATAGOUV"
        assert data["category"] == "PRIVE"


class TestEnrichCandidatesFromSirene:
    def _conn(self, tmp_path: Path) -> sqlite3.Connection:
        return sirene_cache.init_db(tmp_path / "cache.sqlite")

    def _insert_establishment(
        self,
        conn: sqlite3.Connection,
        *,
        siret: str,
        siren: str,
        denomination: str,
        address_full: str,
        postcode: str,
        city: str,
        insee_code: str | None = None,
        legal_nature: str | None = None,
        etat: str = "A",
    ) -> None:
        conn.execute(
            """
            INSERT OR REPLACE INTO establishments (
                siret, siren, nic, etablissement_siege,
                denomination, address_full, postcode, city, insee_code,
                legal_nature, etat_administratif
            ) VALUES (?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                siret,
                siren,
                siret[-5:],
                denomination,
                address_full,
                postcode,
                city,
                insee_code,
                legal_nature,
                etat,
            ),
        )
        conn.commit()

    def test_siret_known_single_establishment(self, tmp_path: Path):
        conn = self._conn(tmp_path)
        self._insert_establishment(
            conn,
            siret="12345678901234",
            siren="123456789",
            denomination="ACME",
            address_full="1 RUE",
            postcode="75001",
            city="PARIS",
            insee_code="75056",
            legal_nature="5710",
        )

        groups = {
            ("siret", "12345678901234"): [
                create_raw_candidate(source="RNE", siret="12345678901234", label="ACME"),
            ]
        }

        res = enrich_candidates_from_sirene(groups, conn)

        assert len(res) == 1
        cand = res[0]
        assert cand.siren == "123456789"
        assert cand.name == "ACME"
        assert cand.address == "1 RUE"
        assert cand.sources == ["RNE"]
        assert cand.category == "PRIVE"

    def test_siret_not_in_cache_fallback(self, tmp_path: Path):
        conn = self._conn(tmp_path)

        groups = {
            ("siret", "99999999999999"): [
                create_raw_candidate(source="DATAGOUV", siret="99999999999999", label="MISSING"),
            ]
        }

        res = enrich_candidates_from_sirene(groups, conn)

        assert len(res) == 1
        cand = res[0]
        assert cand.name == "MISSING"
        assert cand.siret == "99999999999999"
        assert cand.category == "INCONNU"

    def test_siren_known_multiple_establishments(self, tmp_path: Path):
        conn = self._conn(tmp_path)
        for idx in range(3):
            self._insert_establishment(
                conn,
                siret=f"123456789000{10+idx}",
                siren="123456789",
                denomination=f"ACME-{idx}",
                address_full=f"{idx} RUE",
                postcode="75001",
                city="PARIS",
                insee_code="75056",
            )

        groups = {
            ("siren", "123456789"): [
                create_raw_candidate(source="RNE", siren="123456789", label="ACME"),
            ]
        }

        res = enrich_candidates_from_sirene(groups, conn)

        assert len(res) == 3
        siret_list = sorted(c.siret for c in res)
        assert siret_list == ["12345678900010", "12345678900011", "12345678900012"]
        # raw_candidates shared
        assert all(len(c.raw_candidates) == 1 for c in res)
        assert all(c.category == "INCONNU" for c in res)

    def test_siren_no_active_includes_closed(self, tmp_path: Path):
        conn = self._conn(tmp_path)
        self._insert_establishment(
            conn,
            siret="12345678900099",
            siren="123456789",
            denomination="ACME CLOSED",
            address_full="9 RUE",
            postcode="69000",
            city="LYON",
            etat="F",
        )

        groups = {
            ("siren", "123456789"): [
                create_raw_candidate(source="QWANT_PAPPERS", siren="123456789", label="ACME"),
            ]
        }

        res = enrich_candidates_from_sirene(groups, conn)

        assert len(res) == 1
        assert res[0].name == "ACME CLOSED"
        assert res[0].category == "INCONNU"

    def test_sources_deduplication_order(self, tmp_path: Path):
        conn = self._conn(tmp_path)
        self._insert_establishment(
            conn,
            siret="12345678900055",
            siren="123456789",
            denomination="ACME",
            address_full="1 RUE",
            postcode="75001",
            city="PARIS",
        )

        groups = {
            ("siret", "12345678900055"): [
                create_raw_candidate(source="RNE", siret="12345678900055"),
                create_raw_candidate(source="DATAGOUV", siret="12345678900055"),
                create_raw_candidate(source="RNE", siret="12345678900055"),
                create_raw_candidate(source="QWANT_PAPPERS", siret="12345678900055"),
            ]
        }

        res = enrich_candidates_from_sirene(groups, conn)

        assert res[0].sources == ["RNE", "DATAGOUV", "QWANT_PAPPERS"]

    def test_siren_respects_limit_20(self, tmp_path: Path):
        conn = self._conn(tmp_path)
        for idx in range(25):
            self._insert_establishment(
                conn,
                siret=f"1234567890{idx:04d}",
                siren="123456789",
                denomination=f"ACME-{idx}",
                address_full=f"{idx} RUE",
                postcode="75001",
                city="PARIS",
            )

        groups = {
            ("siren", "123456789"): [
                create_raw_candidate(source="RNE", siren="123456789"),
            ]
        }

        res = enrich_candidates_from_sirene(groups, conn)

        assert len(res) == 20

    def test_empty_groups_returns_empty_list(self, tmp_path: Path):
        conn = self._conn(tmp_path)
        res = enrich_candidates_from_sirene({}, conn)
        assert res == []

    def test_enrichment_assigns_public_category(self, tmp_path: Path):
        conn = self._conn(tmp_path)
        self._insert_establishment(
            conn,
            siret="12345678900001",
            siren="123456789",
            denomination="HOPITAL PUBLIC",
            address_full="1 RUE DE LA SANTE",
            postcode="75014",
            city="PARIS",
            legal_nature="7210",
        )

        groups = {
            ("siret", "12345678900001"): [
                create_raw_candidate(source="RNE", siret="12345678900001"),
            ]
        }

        res = enrich_candidates_from_sirene(groups, conn)

        assert len(res) == 1
        assert res[0].category == "PUBLIC"

    def test_enrichment_assigns_prive_category(self, tmp_path: Path):
        conn = self._conn(tmp_path)
        self._insert_establishment(
            conn,
            siret="98765432100010",
            siren="987654321",
            denomination="ACME SAS",
            address_full="10 AVENUE",
            postcode="69000",
            city="LYON",
            legal_nature="5710",
        )

        groups = {
            ("siret", "98765432100010"): [
                create_raw_candidate(source="DATAGOUV", siret="98765432100010"),
            ]
        }

        res = enrich_candidates_from_sirene(groups, conn)

        assert len(res) == 1
        assert res[0].category == "PRIVE"

    def test_enrichment_handles_null_legal_nature(self, tmp_path: Path):
        conn = self._conn(tmp_path)
        self._insert_establishment(
            conn,
            siret="22222222200022",
            siren="222222222",
            denomination="UNKNOWN ENTITY",
            address_full="2 RUE",
            postcode="31000",
            city="TOULOUSE",
            legal_nature=None,
        )

        groups = {
            ("siret", "22222222200022"): [
                create_raw_candidate(source="QWANT_PAPPERS", siret="22222222200022"),
            ]
        }

        res = enrich_candidates_from_sirene(groups, conn)

        assert len(res) == 1
        assert res[0].category == "INCONNU"
