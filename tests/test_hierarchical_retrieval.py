from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.build_hierarchical_retrieval_index import build_index
from src.xgb_matcher.hierarchical_retrieval import (
    BackendHit,
    HierarchicalRetrievalConfig,
    HierarchicalSiretRetriever,
    InMemoryBackend,
    IndexedEstablishment,
    RetrievalQuery,
    TantivyBackend,
    exact_and_operational_hits,
    normalize_insee,
    operational_acceptable_sirets,
)


def _record(siret: str, *, name: str = "ALPHA SERVICES",
            address: str = "10 RUE DES FLEURS", number: str = "10",
            insee: str = "75056", postcode: str = "75001",
            is_siege: bool = False, linked_sirets: tuple[str, ...] = ()):
    return IndexedEstablishment(
        siret=siret, siren=siret[:9], insee=insee, postcode=postcode,
        names=(name,), addresses=(address,), number=number, active=True,
        is_siege=is_siege, linked_sirets=linked_sirets, payload={"siret": siret},
    )


def _config(**changes):
    values = dict(enabled=True, backend="in_memory", direct_slots=60,
                  hierarchical_slots=30, character_rescue_slots=10)
    values.update(changes)
    return HierarchicalRetrievalConfig(**values)


def test_insee_primary_cp_fallback_and_corsica():
    assert normalize_insee("2a-004") == "2A004"
    a = _record("11111111100001", insee="2A004", postcode="20000")
    b = _record("22222222200001", insee="2B033", postcode="20000")
    backend = InMemoryBackend([b, a])
    hits = backend.search(RetrievalQuery(name="ALPHA SERVICES", insee="2A004", postcode="20000"), "name_word", 10)
    assert [hit.record.siret for hit in hits] == [a.siret]
    fallback = backend.search(RetrievalQuery(name="ALPHA SERVICES", insee="99999", postcode="20000"), "name_word", 10)
    assert {hit.record.siret for hit in fallback} == {a.siret, b.siret}


def test_certified_crm_query_column_names_are_consumed():
    query = RetrievalQuery.from_mapping(
        {
            "crm_name": "Alpha Services",
            "crm_address": "10 rue des Fleurs",
            "crm_postcode": "75001",
            "crm_insee": "75056",
        }
    )
    assert query.name == "ALPHA SERVICES"
    assert query.address == "10 RUE DES FLEURS"
    assert query.number == "10"
    assert query.postcode == "75001"
    assert query.insee == "75056"


def test_official_history_and_bidirectional_one_hop():
    old = _record("11111111100001", name="ANCIEN NOM OFFICIEL", linked_sirets=("22222222200002",))
    new = _record("22222222200002", name="NOUVEAU NOM OFFICIEL", linked_sirets=(old.siret,))
    retriever = HierarchicalSiretRetriever(InMemoryBackend([old, new]), _config())
    forward = retriever.retrieve(RetrievalQuery(name="ANCIEN NOM OFFICIEL", address="10 RUE DES FLEURS", insee="75056"))
    assert new.siret in {candidate.siret for candidate in forward}
    assert "official_successor" in next(candidate.sources for candidate in forward if candidate.siret == new.siret)
    reverse = retriever.retrieve(RetrievalQuery(name="NOUVEAU NOM OFFICIEL", insee="75056"))
    assert old.siret in {candidate.siret for candidate in reverse}


def test_expansion_cap_and_address_before_headquarters():
    records = [_record(f"123456789{index:05d}", name="GROUPE NATIONAL",
                       address="20 RUE CIBLE" if index == 39 else f"{index} RUE AUTRE",
                       number="20" if index == 39 else str(index), is_siege=index == 0)
               for index in range(40)]
    query = RetrievalQuery(name="GROUPE NATIONAL", address="20 RUE CIBLE", number="20", insee="75056")
    sites = InMemoryBackend(records).sites_for_siren("123456789", query, 32)
    assert len(sites) == 32
    assert sites[0].siret == "12345678900039"
    assert records[0].is_siege is True


def test_deterministic_deduplicated_max100():
    records = [_record(f"{100000000 + index:09d}00001", name="COMMUNE SERVICE") for index in range(130)]
    retriever = HierarchicalSiretRetriever(InMemoryBackend(records + [records[0]]), _config())
    query = RetrievalQuery(name="COMMUNE SERVICE", insee="75056")
    first, second = retriever.retrieve(query), retriever.retrieve(query)
    assert [x.siret for x in first] == [x.siret for x in second]
    assert len(first) == len({x.siret for x in first}) == 100
    assert [x.rank for x in first] == list(range(1, 101))


def test_union_cap_reserves_all_three_fusion_families():
    direct = [_record(f"{110000000 + i:09d}00001", name="DIRECT") for i in range(120)]
    hierarchical = [_record(f"123456789{i:05d}", name="HIER") for i in range(40)]
    character = [_record(f"{330000000 + i:09d}00001", name="CHAR") for i in range(30)]

    class Backend:
        def search(self, query, channel, limit):
            rows = direct if channel == "name_word" else character if channel == "name_char" else []
            return [BackendHit(row, 10_000 - rank) for rank, row in enumerate(rows[:limit])]

        def search_sirens(self, query, limit):
            return [("123456789", 1.0)]

        def sites_for_siren(self, siren, query, limit):
            return hierarchical[:limit]

        def by_siret(self, siret):
            return None

    output = HierarchicalSiretRetriever(Backend(), _config(union_cap=100)).retrieve(
        RetrievalQuery(name="QUERY", insee="75056")
    )
    sources = [set(candidate.sources) for candidate in output]
    assert sum("name_word" in value for value in sources) == 60
    assert sum("hierarchical" in value for value in sources) == 30
    assert sum("name_char" in value for value in sources) == 10


def test_exact_and_operational_views_are_separate_and_same_site_strict():
    gt = {"siret": "12345678900001", "siren": "123456789", "postcode": "75001",
          "numeroVoie": "10", "indiceRepetition": "BIS", "typeVoie": "RUE", "libelleVoie": "DES FLEURS"}
    same = {**gt, "siret": "12345678900002"}
    wrong_road = {**same, "siret": "12345678900003", "libelleVoie": "DES LILAS"}
    wrong_siren = {**same, "siret": "98765432100002", "siren": "987654321"}
    acceptable = operational_acceptable_sirets(gt, [same, wrong_road, wrong_siren])
    assert acceptable == ("12345678900001", "12345678900002")
    candidate = type("Candidate", (), {"siret": "12345678900002"})()
    assert exact_and_operational_hits([candidate], exact_siret=gt["siret"], operational_sirets=acceptable) == {
        "exact_hit": False, "operational_hit": True}


def _write_tiny_sources(root: Path):
    establishments, legal_units = root / "etab.parquet", root / "ul.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"siret": "12345678900001", "siren": "123456789",
         "codeCommuneEtablissement": "75056", "codePostalEtablissement": "75001",
         "numeroVoieEtablissement": "10", "indiceRepetitionEtablissement": None,
         "etatAdministratifEtablissement": "A", "etablissementSiege": False,
         "enseigne1Etablissement": "ALPHA SERVICES", "enseigne2Etablissement": None,
         "enseigne3Etablissement": None, "denominationUsuelleEtablissement": None,
         "typeVoieEtablissement": "RUE", "libelleVoieEtablissement": "DES FLEURS",
         "libelleCommuneEtablissement": "PARIS",
         "complementAdresseEtablissement": None},
        {"siret": "12345678900002", "siren": "123456789",
         "codeCommuneEtablissement": "92050", "codePostalEtablissement": "75001",
         "numeroVoieEtablissement": "20", "indiceRepetitionEtablissement": None,
         "etatAdministratifEtablissement": "A", "etablissementSiege": True,
         "enseigne1Etablissement": "ALPHA SIEGE", "enseigne2Etablissement": None,
         "enseigne3Etablissement": None, "denominationUsuelleEtablissement": None,
         "typeVoieEtablissement": "AVENUE", "libelleVoieEtablissement": "DU CENTRE",
         "libelleCommuneEtablissement": "NANTERRE",
         "complementAdresseEtablissement": None},
    ]), establishments)
    pq.write_table(pa.Table.from_pylist([{
        "siren": "123456789", "denominationUniteLegale": "ALPHA GROUPE",
        "denominationUsuelle1UniteLegale": None, "denominationUsuelle2UniteLegale": None,
        "denominationUsuelle3UniteLegale": None, "sigleUniteLegale": None,
        "prenomUsuelUniteLegale": None, "nomUniteLegale": None,
        "nomUsageUniteLegale": None,
    }]), legal_units)
    return establishments, legal_units


def test_real_tantivy_current_only_smoke(tmp_path: Path):
    establishments, legal_units = _write_tiny_sources(tmp_path)
    temp_directory = tmp_path / "duckdb_tmp"
    temp_directory.mkdir()
    output = build_index(Namespace(
        establishments=establishments, legal_units=legal_units,
        retrieval_config=Path("config/retrieval_hierarchical_v1.json"),
        establishments_history=None, legal_units_history=None, successions=None,
        output_root=tmp_path / "indices", temp_directory=temp_directory,
        batch_size=1, commit_every=10, writer_heap_bytes=30_000_000,
        writer_threads=1, duckdb_memory_limit="1GB", duckdb_threads=1,
        smoke_limit=2,
    ))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["num_establishment_documents"] == 2
    assert manifest["num_siren_documents"] == 2
    assert manifest["temporal_complete"] is False
    assert manifest["missing_optional_roles"] == [
        "establishments_history", "legal_units_history", "successions"]
    assert manifest["contains_crm_labels"] is False

    backend = TantivyBackend(output)
    hits = backend.search(RetrievalQuery(name="ALPHA SERVICES", address="10 RUE DES FLEURS",
                                         number="10", insee="75056"), "name_word", 10)
    assert hits[0].record.siret == "12345678900001"
    assert backend.search(
        RetrievalQuery(name="SIEGE", insee="75056", postcode="75001"),
        "name_word",
        10,
    ) == []
    sirens = backend.search_sirens(RetrievalQuery(name="ALPHA GROUPE", insee="75056"), 5)
    assert [siren for siren, _score in sirens] == ["123456789"]


def test_builder_indexes_official_history_and_successions(tmp_path: Path):
    establishments, legal_units = _write_tiny_sources(tmp_path)
    establishment_history = tmp_path / "etab_history.parquet"
    legal_history = tmp_path / "ul_history.parquet"
    successions = tmp_path / "successions.parquet"
    pq.write_table(pa.Table.from_pylist([{
        "siret": "12345678900001", "enseigne1Etablissement": "ANCIENNE ALPHA",
        "numeroVoieEtablissement": "10", "indiceRepetitionEtablissement": None,
        "typeVoieEtablissement": "RUE", "libelleVoieEtablissement": "DES FLEURS",
        "complementAdresseEtablissement": None,
    }]), establishment_history)
    pq.write_table(pa.Table.from_pylist([{
        "siren": "123456789", "denominationUniteLegale": "ALPHA HISTORIQUE",
    }]), legal_history)
    pq.write_table(pa.Table.from_pylist([{
        "siretEtablissementPredecesseur": "12345678900001",
        "siretEtablissementSuccesseur": "12345678900002",
    }]), successions)
    temp_directory = tmp_path / "duckdb_tmp"
    temp_directory.mkdir()
    output = build_index(Namespace(
        establishments=establishments, legal_units=legal_units,
        retrieval_config=Path("config/retrieval_hierarchical_v1.json"),
        establishments_history=establishment_history,
        legal_units_history=legal_history, successions=successions,
        output_root=tmp_path / "indices", temp_directory=temp_directory,
        batch_size=2, commit_every=10, writer_heap_bytes=30_000_000,
        writer_threads=1, duckdb_memory_limit="1GB", duckdb_threads=1,
        smoke_limit=2,
    ))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["temporal_complete"] is True
    backend = TantivyBackend(output)
    old_name_hits = backend.search(
        RetrievalQuery(name="ANCIENNE ALPHA", insee="75056"), "name_exact", 10)
    assert [hit.record.siret for hit in old_name_hits] == ["12345678900001"]
    assert backend.by_siret("12345678900001").linked_sirets == ("12345678900002",)
    assert backend.by_siret("12345678900002").linked_sirets == ("12345678900001",)


def test_typed_dossier_portfolio_keeps_history_as_rescue_only(tmp_path: Path):
    documents = tmp_path / "documents"
    documents.mkdir()
    common = {
        "document_id": "12345678900001", "siren": "123456789",
        "document_kind": "SIRET", "insee": "75056", "postcode": "75001",
        "number": "10", "number_suffix": "", "administrative_state": "A",
        "is_headquarters": True, "legal_current_names": "ALPHA GROUPE",
        "trade_current_names": "ALPHA", "site_current_names": "ALPHA PARIS",
        "historical_names": "ANCIENNE ALPHA", "supporting_names": "ALPHA BODACC",
        "current_address_text": "10 RUE DES FLEURS",
        "historical_address_text": "8 RUE DES FLEURS",
        "supporting_address_text": "10 R DES FLEURS",
        "linked_sirets": "", "linked_sirens": "",
    }
    pq.write_table(pa.Table.from_pylist([common]), documents / "retrieval_siret_documents.parquet")
    pq.write_table(
        pa.Table.from_pylist([{**common, "document_id": "123456789", "document_kind": "SIREN",
                               "number": "", "number_suffix": "", "is_headquarters": False,
                               "current_address_text": "", "historical_address_text": "",
                               "supporting_address_text": ""}]),
        documents / "retrieval_siren_documents.parquet",
    )
    pq.write_table(pa.Table.from_pylist([{"siren": "123456789"}]),
                   documents / "retrieval_name_portfolio.parquet")
    (documents / "manifest.json").write_text(json.dumps({
        "schema_version": "sireto-siren-dossier-retrieval-documents-v2",
        "temporal_complete": True,
    }))
    temp_directory = tmp_path / "duckdb_tmp"
    temp_directory.mkdir()
    output = build_index(Namespace(
        dossier_documents=documents, establishments=None, legal_units=None,
        retrieval_config=Path("config/retrieval_hierarchical_v2.json"),
        establishments_history=None, legal_units_history=None, successions=None,
        output_root=tmp_path / "indices", temp_directory=temp_directory,
        batch_size=2, commit_every=10, writer_heap_bytes=30_000_000,
        writer_threads=1, duckdb_memory_limit="1GB", duckdb_threads=1, smoke_limit=None,
    ))
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["typed_name_portfolio"] is True
    backend = TantivyBackend(output)
    current = backend.search(RetrievalQuery(name="ALPHA PARIS", insee="75056"),
                             "site_name_exact", 10)
    assert [hit.record.siret for hit in current] == ["12345678900001"]
    assert backend.search(RetrievalQuery(name="ANCIENNE ALPHA", insee="75056"),
                          "site_name_exact", 10) == []
    historical = backend.search(RetrievalQuery(name="ANCIENNE ALPHA", insee="75056"),
                                "historical_name_word", 10)
    assert [hit.record.siret for hit in historical] == ["12345678900001"]
    assert backend.search(
        RetrievalQuery(name="ALPHA PARIS", insee="75056"),
        "fielded_name_bm25",
        10,
    )[0].record.siret == "12345678900001"
    assert backend.search(
        RetrievalQuery(name="ALPHA", number="10", insee="75056"),
        "number_exact",
        10,
    )[0].record.siret == "12345678900001"
    retriever = HierarchicalSiretRetriever(
        backend, HierarchicalRetrievalConfig.load("config/retrieval_hierarchical_v2.json")
    )
    candidates = retriever.retrieve(RetrievalQuery(name="ANCIENNE ALPHA", insee="75056"))
    assert candidates[0].siret == "12345678900001"
    assert "historical_name_word" in candidates[0].sources
