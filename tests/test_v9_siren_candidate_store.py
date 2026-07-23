from __future__ import annotations

import json

import duckdb

from src.xgb_matcher.siren_retrieval import SirenCandidateStore


def test_siren_candidate_store_batches_indexed_lookups(tmp_path) -> None:
    database = tmp_path / "siren_candidates.duckdb"
    connection = duckdb.connect(str(database))
    connection.execute(
        """
        CREATE TABLE candidates (
            siret VARCHAR,
            siren VARCHAR,
            denomination VARCHAR,
            insee VARCHAR,
            postcode VARCHAR,
            siren_geo_count INTEGER
        )
        """
    )
    connection.execute(
        """
        INSERT INTO candidates VALUES
        ('00000000100001', '000000001', 'ONE', '75056', '75001', 4),
        ('00000000100002', '000000001', 'TWO', '92050', '92000', 1),
        ('00000000300001', '000000003', 'THREE', '13055', '13001', 2)
        """
    )
    connection.execute(
        "CREATE INDEX candidates_siren_idx ON candidates(siren)"
    )
    connection.close()
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {"schema_version": "v9-siren-candidate-duckdb-2"}
        ),
        encoding="utf-8",
    )

    store = SirenCandidateStore(tmp_path)
    try:
        candidates = store.get_candidates(
            ["1", "000000003"],
            crm_insee="92050",
            crm_postcode="92000",
            max_per_siren=2,
        )
    finally:
        store.close()

    assert [row["siret"] for row in candidates["000000001"]] == [
        "00000000100002",
        "00000000100001",
    ]
    assert candidates["000000003"][0]["denomination"] == "THREE"
