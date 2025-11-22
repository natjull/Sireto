from __future__ import annotations

from pathlib import Path
import sys

import pytest
import sqlite3

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipe_v6 import sirene_cache
from pipe_v6.config import PipelineConfig
from pipe_v6.commune_detection import CommuneKey


def _make_config(tmp_path: Path) -> PipelineConfig:
    dummy = {
        "crm_path": tmp_path / "crm.csv",
        "output_path": tmp_path / "out.csv",
        "sqlite_path": tmp_path / "cache.sqlite",
        "log_path": tmp_path / "logs.log",
        "sirene_api_url": "https://api.insee.fr/entreprises/sirene/V3",
        "sirene_token": "token",
        "rne_api_url": "https://api.inpi.fr/api/rne/",
        "rne_token_url": "https://api.inpi.fr/api/sso/oauth2/token",
        "rne_client_id": "client",
        "rne_client_secret": "secret",
        "qwant_base_url": "https://api.qwant.com/api/search/web",
        "model_name": "gpt-oss:20b",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 256,
        "max_candidates_per_source": 10,
        "confidence_auto_match": 0.85,
        "confidence_review_min": 0.6,
        "log_level": "INFO",
        "store_source_raw": True,
        "extra": {},
    }
    return PipelineConfig(**dummy)


def test_init_db_creates_expected_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "sirene_cache.sqlite"
    conn = sirene_cache.init_db(db_path)

    try:
        establishments = conn.execute("PRAGMA table_info(establishments);").fetchall()
        table_columns = {row[1] for row in establishments}
        assert {
            "siret",
            "siren",
            "nic",
            "denomination",
            "postcode",
            "city",
            "legal_nature",
        }.issubset(table_columns)

        status_table = conn.execute(
            "PRAGMA table_info(commune_cache_status);"
        ).fetchall()
        status_columns = {row[1] for row in status_table}
        assert {
            "insee_code",
            "postcode",
            "city",
            "fetched_at",
            "status",
        }.issubset(status_columns)

        index_rows = conn.execute(
            "PRAGMA index_list(establishments);"
        ).fetchall()
        index_names = {row[1] for row in index_rows}
        assert "idx_establishments_siren" in index_names
    finally:
        conn.close()


def test_get_cache_connection_reuses_existing_schema(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    conn = sirene_cache.get_cache_connection(config)

    try:
        pragma_value = conn.execute("PRAGMA journal_mode;").fetchone()[0].lower()
        assert pragma_value == "wal"

        # Insert a minimal establishment to ensure the schema accepts data with mandatory fields
        conn.execute(
            """
            INSERT INTO establishments (
                siret, siren, nic, etablissement_siege, postcode, city
            ) VALUES (?, ?, ?, 0, '69000', 'LYON')
            """,
            ("12345678901234", "123456789", "01234"),
        )
        conn.commit()

        rows = conn.execute(
            "SELECT siret, siren FROM establishments WHERE siret = ?",
            ("12345678901234",),
        ).fetchone()
        assert rows == ("12345678901234", "123456789")
    finally:
        conn.close()


def test_init_db_creates_file_on_disk(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "sirene_cache.sqlite"
    conn = sirene_cache.init_db(db_path)
    conn.close()

    assert db_path.exists(), "Database file should be created on disk"


def _sample_record() -> dict:
    return {
        "siret": "12345678901234",
        "siren": "123456789",
        "nic": "01234",
        "etablissementSiege": True,
        "enseigne1Etablissement": "Enseigne One",
        "etatAdministratifEtablissement": "A",
        "dateCreationEtablissement": "2020-01-01",
        "dateDernierTraitementEtablissement": "2024-01-01",
        "activitePrincipaleEtablissement": "61.10Z",
        "nomenclatureActivitePrincipaleEtablissement": "NAFRev2",
        "trancheEffectifsEtablissement": "11",
        "anneeEffectifsEtablissement": "2023",
        "adresseEtablissement": {
            "numeroVoieEtablissement": "12",
            "indiceRepetitionEtablissement": "BIS",
            "typeVoieEtablissement": "RUE",
            "libelleVoieEtablissement": "DU TEST",
            "complementAdresseEtablissement": "COMPLEMENT",
            "codePostalEtablissement": "6907",
            "libelleCommuneEtablissement": "Lyon",
            "codeCommuneEtablissement": "69387",
        },
        "uniteLegale": {
            "denominationUniteLegale": "Covage Grand Lyon",
            "nomUniteLegale": "DOE",
            "prenom1UniteLegale": "JOHN",
            "categorieJuridiqueUniteLegale": "5710",
            "activitePrincipaleUniteLegale": "61.90Z",
            "nomenclatureActivitePrincipaleUniteLegale": "NAFRev2",
            "trancheEffectifsUniteLegale": "12",
            "anneeEffectifsUniteLegale": "2022",
        },
    }


def test_upsert_establishments_projects_and_inserts(tmp_path: Path) -> None:
    conn = sirene_cache.init_db(tmp_path / "cache.sqlite")
    conn.row_factory = sqlite3.Row
    record = _sample_record()

    inserted = sirene_cache.upsert_establishments([record], conn, store_source_raw=False)
    assert inserted == 1

    row = conn.execute(
        "SELECT * FROM establishments WHERE siret = ?",
        ("12345678901234",),
    ).fetchone()

    assert row["denomination"] == "Covage Grand Lyon"
    assert row["denomination_ci"] == "COVAGE GRAND LYON"
    assert row["street_number"] == "12 BIS"
    assert row["address_full"] == "COMPLEMENT 12 BIS RUE DU TEST"
    assert row["postcode"] == "06907"
    assert row["city"] == "LYON"
    assert row["commune_libelle_raw"] == "Lyon"
    assert row["etat_administratif"] == "A"
    assert row["date_debut"] == "2020-01-01"
    assert row["activite_principale"] == "61.10Z"
    assert row["tranche_effectifs"] == "11"
    assert row["legal_nature"] == "5710"
    assert row["source_raw"] is None


def test_upsert_establishments_deduplicates_and_fallbacks(tmp_path: Path) -> None:
    conn = sirene_cache.init_db(tmp_path / "cache.sqlite")
    conn.row_factory = sqlite3.Row
    record = _sample_record()
    record["uniteLegale"]["denominationUniteLegale"] = None
    record["enseigne1Etablissement"] = None
    record["enseigne2Etablissement"] = "Second"
    record["etablissementSiege"] = False

    second = _sample_record()
    second["siret"] = "12345678901234"
    second["etatAdministratifEtablissement"] = "F"
    second["uniteLegale"]["denominationUniteLegale"] = None
    second["enseigne1Etablissement"] = None
    second["enseigne2Etablissement"] = None
    second["enseigne3Etablissement"] = None

    inserted = sirene_cache.upsert_establishments(
        [record, second], conn, store_source_raw=True
    )
    assert inserted == 2  # both attempted inserts

    row = conn.execute(
        "SELECT denomination, etablissement_siege, etat_administratif, source_raw FROM establishments WHERE siret = ?",
        ("12345678901234",),
    ).fetchone()

    # Last insert replaces previous row (etat F) and fallback to person name
    assert row["denomination"] == "DOE JOHN"
    assert row["etablissement_siege"] == 1
    assert row["etat_administratif"] == "F"
    assert row["source_raw"] is not None


def test_exists_in_cache_with_insee_and_postcode(tmp_path: Path) -> None:
    conn = sirene_cache.init_db(tmp_path / "cache.sqlite")
    conn.execute(
        """INSERT INTO establishments (siret, siren, nic, etablissement_siege, postcode, city, insee_code)
        VALUES ('11111111111111','111111111','11111',0,'69007','LYON','69381')"""
    )
    conn.commit()

    cached, count = sirene_cache.exists_in_cache(
        CommuneKey("69381", None, None), conn
    )
    assert cached is True and count == 1

    cached_cp, count_cp = sirene_cache.exists_in_cache(
        CommuneKey(None, "69007", None), conn
    )
    assert cached_cp is True and count_cp == 1


def test_get_or_fetch_commune_returns_cached(tmp_path: Path) -> None:
    conn = sirene_cache.init_db(tmp_path / "cache.sqlite")
    conn.row_factory = sqlite3.Row
    sirene_cache.upsert_establishments([_sample_record()], conn, store_source_raw=False)

    config = _make_config(tmp_path)
    result = sirene_cache.get_or_fetch_commune(
        CommuneKey("69387", None, None), config, conn=conn
    )
    assert result["status"] == "CACHED"
    assert result["count"] == 1
    assert result["inserted"] == 0


def test_get_or_fetch_commune_downloads_when_missing(tmp_path: Path, monkeypatch) -> None:
    conn = sirene_cache.init_db(tmp_path / "cache.sqlite")
    config = _make_config(tmp_path)
    commune = CommuneKey(None, "69007", None)

    sample = _sample_record()
    sample["adresseEtablissement"]["codePostalEtablissement"] = "69007"
    sample["adresseEtablissement"]["codeCommuneEtablissement"] = "69007"

    calls = {}

    def fake_fetch(commune_arg, config_arg, logger=None):  # type: ignore[override]
        calls["commune"] = commune_arg
        return [sample]

    monkeypatch.setattr(
        "pipe_v6.sirene_cache.sirene_client.fetch_establishments_for_commune",
        fake_fetch,
    )

    result = sirene_cache.get_or_fetch_commune(commune, config, conn=conn)

    assert result["status"] == "DOWNLOADED"
    assert result["count"] == 1
    assert result["inserted"] == 1
    assert "commune" in calls
