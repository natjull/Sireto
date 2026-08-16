from pathlib import Path

import duckdb
import pytest

from scripts import audit_synthetic_gt_full_sirene_exact as audit


def candidate(siret: str, name: str, address: str, *, insee: str = "93066", postcode: str = "75001"):
    words = address.split()
    return {
        "siret": siret, "siren": siret[:9], "insee": insee, "postcode": postcode,
        "number": words[0], "repetition_index": "", "street_type": words[1],
        "street": " ".join(words[2:]), "establishment_usual": name,
        "enseigne1": "", "enseigne2": "", "enseigne3": "",
        "legal_denomination": name, "legal_sigle": "", "legal_usual1": "",
        "legal_usual2": "", "legal_usual3": "", "legal_last_name": "",
        "legal_usage_name": "", "legal_usual_first": "", "legal_first": "",
    }


def crm(name="FLEURS MAISON", address="12 R DES LILAS"):
    return {"name": name, "address": address, "postcode": "75001", "city": "SAINT DENIS", "insee": "93066"}


def test_exact_address_naturally_returns_target_without_injection():
    target = candidate("12345678900012", "MAISON DES FLEURS", "12 RUE DES LILAS")
    other = candidate("98765432100019", "AUTRE SOCIETE", "8 RUE DES LILAS")
    result = audit.qualify_variant(target["siret"], crm(), [target, other])
    assert result["decision"] == "EXACT_IDENTIFIABLE"
    assert result["exact_witness"] in {"G_N_A", "G_A", "G_N"}
    assert result["target_naturally_returned"] is True


def test_missing_target_is_not_injected():
    other = candidate("98765432100019", "AUTRE SOCIETE", "8 RUE DES LILAS")
    result = audit.qualify_variant("12345678900012", crm(), [other])
    assert result["decision"] == "TARGET_NOT_NATURALLY_MATCHED"
    assert result["target_naturally_returned"] is False


def test_same_name_and_address_is_ambiguous_not_exact():
    first = candidate("12345678900012", "MAISON DES FLEURS", "12 RUE DES LILAS")
    second = candidate("98765432100019", "MAISON DES FLEURS", "12 RUE DES LILAS")
    result = audit.qualify_variant(first["siret"], crm(), [first, second])
    assert result["decision"] == "AMBIGUOUS_OFFICIAL"
    assert result["candidate_counts"]["G_N_A"] == 2


def test_same_siren_same_site_is_reported_operationally_but_not_exact():
    first = candidate("12345678900012", "MAISON DES FLEURS", "12 RUE DES LILAS")
    second = candidate("12345678900020", "MAISON DES FLEURS", "12 RUE DES LILAS")
    result = audit.qualify_variant(first["siret"], crm(), [first, second])
    assert result["decision"] == "AMBIGUOUS_OFFICIAL"
    assert result["operational_equivalence"] is True
    assert result["operational_equivalent_sirets"] == [second["siret"]]


def test_finite_language_allows_whole_token_delete_reorder_and_join_not_letter_anagram():
    assert audit.whole_token_language("FLEURS MAISON", "MAISON DES FLEURS")
    assert audit.whole_token_language("SAO", "S A O REYMOND")
    assert not audit.whole_token_language("NOSA", "S A O REYMOND")
    assert not audit.whole_token_language("FLEUSR", "MAISON DES FLEURS")


def tiny_official_sources(tmp_path: Path) -> tuple[Path, Path]:
    establishments = tmp_path / "establishments.parquet"
    legal_units = tmp_path / "legal_units.parquet"
    connection = duckdb.connect()
    try:
        connection.execute(
            """CREATE TABLE establishments AS SELECT * FROM (VALUES
                ('55210055400013','552100554','93066','75001','12',NULL,'RUE','DES LILAS',
                 'MAISON DES FLEURS',NULL,NULL,NULL,'A'),
                ('73282932000074','732829320','75101','75002','8',NULL,'RUE','DE LA PAIX',
                 NULL,'AUTRE ENSEIGNE',NULL,NULL,'F')
            ) t(siret,siren,codeCommuneEtablissement,codePostalEtablissement,
                numeroVoieEtablissement,indiceRepetitionEtablissement,
                typeVoieEtablissement,libelleVoieEtablissement,
                denominationUsuelleEtablissement,enseigne1Etablissement,
                enseigne2Etablissement,enseigne3Etablissement,
                etatAdministratifEtablissement)"""
        )
        connection.execute(
            """CREATE TABLE legal_units AS SELECT * FROM (VALUES
                ('552100554','MAISON DES FLEURS SAS','MDF',NULL,NULL,NULL,NULL,NULL,NULL,NULL),
                ('732829320','AUTRE SOCIETE',NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL)
            ) t(siren,denominationUniteLegale,sigleUniteLegale,
                denominationUsuelle1UniteLegale,denominationUsuelle2UniteLegale,
                denominationUsuelle3UniteLegale,nomUniteLegale,nomUsageUniteLegale,
                prenomUsuelUniteLegale,prenom1UniteLegale)"""
        )
        connection.execute("COPY establishments TO ? (FORMAT PARQUET)", [str(establishments)])
        connection.execute("COPY legal_units TO ? (FORMAT PARQUET)", [str(legal_units)])
    finally:
        connection.close()
    return establishments, legal_units


def test_official_cache_is_content_addressed_read_only_and_reused(tmp_path: Path):
    establishments, legal_units = tiny_official_sources(tmp_path)
    source_hashes = {
        "sirene_establishments": audit.sha256(establishments),
        "sirene_legal_units": audit.sha256(legal_units),
    }
    cache_directory = tmp_path / "cache"
    temp_directory = tmp_path / "duckdb-temp"
    cache, metadata = audit.ensure_official_cache(
        cache_directory, establishments, legal_units, source_hashes, 2, temp_directory,
    )
    first_stat = cache.stat()

    reused, reused_metadata = audit.ensure_official_cache(
        cache_directory, establishments, legal_units, source_hashes, 2, temp_directory,
    )

    assert reused == cache
    assert reused.stat().st_mtime_ns == first_stat.st_mtime_ns
    assert reused_metadata == metadata
    assert cache.stat().st_mode & 0o222 == 0
    assert cache.name == f"official_{metadata['cache_key']}.duckdb"


def test_cached_query_contains_only_official_projection_and_preserves_values(tmp_path: Path):
    establishments, legal_units = tiny_official_sources(tmp_path)
    source_hashes = {
        "sirene_establishments": audit.sha256(establishments),
        "sirene_legal_units": audit.sha256(legal_units),
    }
    cache, _ = audit.ensure_official_cache(
        tmp_path / "cache", establishments, legal_units, source_hashes, 2,
        tmp_path / "duckdb-temp",
    )

    candidates = audit.query_candidates(cache, ["93066"], tmp_path / "query-temp")

    assert set(candidates) == {"93066"}
    assert len(candidates["93066"]) == 1
    row = candidates["93066"][0]
    assert row["siret"] == "55210055400013"
    assert row["legal_denomination"] == "MAISON DES FLEURS SAS"
    assert row["enseigne1"] == ""
    connection = duckdb.connect(str(cache), read_only=True)
    try:
        tables = {value[0] for value in connection.execute("SHOW TABLES").fetchall()}
        columns = {
            value[1]
            for value in connection.execute("PRAGMA table_info('official_candidates')").fetchall()
        }
    finally:
        connection.close()
    assert tables == {"cache_metadata", "official_candidates"}
    assert not any(
        forbidden in value
        for value in tables | columns
        for forbidden in ("crm", "target", "decision", "score", "rank")
    )


def test_official_cache_rejects_wrong_source_seal(tmp_path: Path):
    establishments, legal_units = tiny_official_sources(tmp_path)
    source_hashes = {
        "sirene_establishments": audit.sha256(establishments),
        "sirene_legal_units": audit.sha256(legal_units),
    }
    cache, _ = audit.ensure_official_cache(
        tmp_path / "cache", establishments, legal_units, source_hashes, 2,
        tmp_path / "duckdb-temp",
    )
    wrong = {**source_hashes, "sirene_legal_units": "0" * 64}

    with pytest.raises(ValueError, match="mismatch"):
        audit.validate_official_cache(cache, wrong, 2)
