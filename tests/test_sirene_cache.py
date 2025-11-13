from __future__ import annotations

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipe_v6 import sirene_cache
from pipe_v6.config import PipelineConfig


def _make_config(tmp_path: Path) -> PipelineConfig:
    dummy = {
        "crm_path": tmp_path / "crm.csv",
        "output_path": tmp_path / "out.csv",
        "sqlite_path": tmp_path / "cache.sqlite",
        "log_path": tmp_path / "logs.log",
        "sirene_api_url": "https://api.insee.fr/entreprises/sirene/V3",
        "sirene_token": "token",
        "rne_api_url": "https://api.inpi.fr/api/rne/",
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
