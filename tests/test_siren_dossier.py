from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from src.xgb_matcher.official_evidence import (
    OfficialAddress,
    OfficialEvidence,
    OfficialName,
    OfficialRelation,
    OfficialRelationType,
    OfficialSource,
    OfficialSubjectKind,
    official_evidence_arrow_schema,
    official_relation_arrow_schema,
)
from src.xgb_matcher.siren_dossier import (
    SirenDossierInputs,
    build_siren_dossier,
    materialize_dossier_retrieval_documents,
    open_siren_dossier,
    project_dossier_candidate_features,
    project_dossier_fusion_text,
)
from src.xgb_matcher.rne_accounts import ACCOUNT_DEPOSIT_SCHEMA


def _write_fixture(tmp_path: Path) -> SirenDossierInputs:
    establishments = pa.Table.from_pylist(
        [
            {
                "siren": "123456789", "nic": "00011", "siret": "12345678900011",
                "etatAdministratifEtablissement": "A", "etablissementSiege": True,
                "dateCreationEtablissement": "2020-01-01", "dateDebut": "2020-01-01",
                "codeCommuneEtablissement": "75056", "codePostalEtablissement": "75001",
                "numeroVoieEtablissement": "10", "indiceRepetitionEtablissement": "",
                "typeVoieEtablissement": "RUE", "libelleVoieEtablissement": "DE LA PAIX",
                "complementAdresseEtablissement": "", "activitePrincipaleEtablissement": "62.01Z",
                "enseigne1Etablissement": "ALPHA SHOP", "enseigne2Etablissement": None,
                "enseigne3Etablissement": None, "denominationUsuelleEtablissement": "ALPHA",
            },
            {
                "siren": "123456789", "nic": "00029", "siret": "12345678900029",
                "etatAdministratifEtablissement": "F", "etablissementSiege": False,
                "dateCreationEtablissement": "2010-01-01", "dateDebut": "2018-01-01",
                "codeCommuneEtablissement": "75056", "codePostalEtablissement": "75002",
                "numeroVoieEtablissement": "5", "indiceRepetitionEtablissement": "",
                "typeVoieEtablissement": "AV", "libelleVoieEtablissement": "OPERA",
                "complementAdresseEtablissement": "", "activitePrincipaleEtablissement": "62.01Z",
                "enseigne1Etablissement": None, "enseigne2Etablissement": None,
                "enseigne3Etablissement": None, "denominationUsuelleEtablissement": None,
            },
        ]
    )
    legal_units = pa.Table.from_pylist(
        [{
            "siren": "123456789", "etatAdministratifUniteLegale": "A",
            "dateCreationUniteLegale": "2010-01-01", "dateDebut": "2020-01-01",
            "categorieJuridiqueUniteLegale": 5710, "activitePrincipaleUniteLegale": "62.01Z",
            "categorieEntreprise": "PME", "nicSiegeUniteLegale": "00011",
            "denominationUniteLegale": "ALPHA SAS", "denominationUsuelle1UniteLegale": None,
            "denominationUsuelle2UniteLegale": None, "denominationUsuelle3UniteLegale": None,
            "sigleUniteLegale": "ALPHA", "nomUsageUniteLegale": None, "nomUniteLegale": None,
        }]
    )
    establishments_path = tmp_path / "establishments.parquet"
    legal_units_path = tmp_path / "legal_units.parquet"
    pq.write_table(establishments, establishments_path)
    pq.write_table(legal_units, legal_units_path)
    evidence = [
        OfficialEvidence(
            source=OfficialSource.RNE, source_record_id="rne-1",
            subject_kind=OfficialSubjectKind.SIREN, siren="123456789",
            names=(OfficialName("ALPHA TECHNOLOGIES"),),
            addresses=(OfficialAddress("10 RUE DE LA PAIX", postcode="75001", insee="75056"),),
            valid_from="2019-01-01", is_current=True,
        ),
        OfficialEvidence(
            source=OfficialSource.BODACC, source_record_id="bodacc-1",
            subject_kind=OfficialSubjectKind.SIREN, siren="123456789",
            names=(OfficialName("OLD ALPHA"),),
            addresses=(OfficialAddress("5 AV OPERA", postcode="75002", insee="75056"),),
            valid_from="2018-01-01", is_current=False,
        ),
    ]
    relation = OfficialRelation(
        source=OfficialSource.BODACC, source_record_id="bodacc-rel",
        relation_type=OfficialRelationType.ASSET_TRANSFER,
        from_kind=OfficialSubjectKind.SIREN, from_identifier="987654321",
        to_kind=OfficialSubjectKind.SIREN, to_identifier="123456789",
        effective_date="2018-01-01",
    )
    evidence_path = tmp_path / "evidence.parquet"
    relation_path = tmp_path / "relations.parquet"
    pq.write_table(pa.Table.from_pylist([item.to_dict() for item in evidence], schema=official_evidence_arrow_schema()), evidence_path)
    pq.write_table(pa.Table.from_pylist([relation.to_dict()], schema=official_relation_arrow_schema()), relation_path)
    return SirenDossierInputs(establishments_path, legal_units_path, (evidence_path,), (relation_path,))


def test_builds_dossier_resolves_only_unique_exact_sites_and_catalog(tmp_path: Path):
    result = build_siren_dossier(_write_fixture(tmp_path), output_root=tmp_path / "out", memory_limit="1GB")
    assert result.counts["legal_units"] == 1
    assert result.counts["establishments"] == 2
    assert result.counts["entity_evidence"] == 2
    assert result.counts["address_site_resolution"] == 2
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["policy"]["bodacc_siren_address_auto_identity"] is False
    connection = open_siren_dossier(result.output_dir)
    assert connection.execute("select count(*) from establishments").fetchone()[0] == 2
    resolved = connection.execute(
        "select source,resolved_siret from address_site_resolution order by source"
    ).fetchall()
    assert resolved == [("BODACC", "12345678900029"), ("RNE", "12345678900011")]
    connection.close()


def test_projects_shared_model_features_without_ids_as_feature_values(tmp_path: Path):
    result = build_siren_dossier(_write_fixture(tmp_path), output_root=tmp_path / "out", memory_limit="1GB")
    candidates = pa.Table.from_pylist(
        [{
            "query_id": "q1", "candidate_siret": "12345678900011",
            "crm_name_normalized": "ALPHA TECHNOLOGIES",
            "crm_address_normalized": "10 RUE DE LA PAIX",
            "crm_insee": "75056", "crm_postcode": "75001",
        }]
    )
    candidates_path = tmp_path / "candidates.parquet"
    output = tmp_path / "features.parquet"
    pq.write_table(candidates, candidates_path)
    assert project_dossier_candidate_features(
        dossier_dir=result.output_dir, candidates_path=candidates_path, output_path=output
    ) == 1
    row = pq.read_table(output).to_pylist()[0]
    assert row["exact_official_name"] == 1
    assert row["exact_official_address"] == 1
    assert row["official_insee_agreement"] == 1
    assert row["exact_external_site_resolution_count"] == 1
    assert row["candidate_relation_count"] == 1
    assert row["max_current_legal_name_jw"] > 0
    assert row["max_rne_name_jw"] == 1
    assert row["current_entity_source_count"] == 1


def test_materializes_hierarchical_retrieval_and_source_separated_fusion(tmp_path: Path):
    result = build_siren_dossier(_write_fixture(tmp_path), output_root=tmp_path / "out", memory_limit="1GB")
    documents = tmp_path / "documents"
    counts = materialize_dossier_retrieval_documents(
        dossier_dir=result.output_dir, output_dir=documents
    )
    assert counts == {"name_portfolio": 6, "siret_documents": 2, "siren_documents": 2}
    rows = pq.read_table(documents / "retrieval_siret_documents.parquet").to_pylist()
    assert any("ALPHA TECHNOLOGIES" in row["trade_current_names"] for row in rows)
    assert all("name_text" not in row for row in rows)
    portfolio = pq.read_table(documents / "retrieval_name_portfolio.parquet").to_pylist()
    assert {row["name_role"] for row in portfolio} == {
        "LEGAL_CURRENT", "TRADE_CURRENT", "SITE_CURRENT", "SUPPORTING"
    }
    manifest = json.loads((documents / "manifest.json").read_text())
    assert manifest["schema_version"] == "sireto-siren-dossier-retrieval-documents-v3"
    assert manifest["sources_separate"] is True
    assert manifest["blind_name_concatenation"] is False
    assert manifest["current_exact_only"] is True
    assert all(row["linked_sirens"] == "987654321" for row in rows)
    assert materialize_dossier_retrieval_documents(
        dossier_dir=result.output_dir, output_dir=documents
    ) == counts
    candidates_path = tmp_path / "candidate_ids.parquet"
    fusion_path = tmp_path / "fusion.parquet"
    pq.write_table(
        pa.Table.from_pylist([{"query_id": "q1", "candidate_siret": "12345678900011"}]),
        candidates_path,
    )
    count = project_dossier_fusion_text(
        dossier_dir=result.output_dir, candidates_path=candidates_path, output_path=fusion_path
    )
    fusion = pq.read_table(fusion_path).to_pylist()
    assert count == len(fusion) > 2
    assert {row["field"] for row in fusion} == {"NAME", "ADDRESS"}
    assert {row["source"] for row in fusion}.issuperset({"SIRENE_CURRENT", "RNE"})
    assert {row["evidence_role"] for row in fusion}.issuperset(
        {"LEGAL_CURRENT", "TRADE_CURRENT", "SITE_CURRENT"}
    )


def test_materialization_supports_a_bounded_real_smoke_scope(tmp_path: Path):
    result = build_siren_dossier(
        _write_fixture(tmp_path), output_root=tmp_path / "out", memory_limit="1GB"
    )
    documents = tmp_path / "smoke-documents"
    counts = materialize_dossier_retrieval_documents(
        dossier_dir=result.output_dir, output_dir=documents, document_limit=1
    )
    assert counts["siret_documents"] == 1
    assert counts["siren_documents"] == 1
    manifest = json.loads((documents / "manifest.json").read_text())
    assert manifest["document_limit"] == 1


def test_dossier_v2_keeps_rne_accounts_siren_level_and_held_out(tmp_path: Path):
    base = _write_fixture(tmp_path)
    accounts_path = tmp_path / "account_deposits.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "source_record_uid": "u1", "snapshot_id": "s1",
                    "archive_member": "stock.json", "source_record_ordinal": 1,
                    "filing_id": "f1", "siren": "123456789",
                    "denomination": "ALPHA", "filing_date": date(2026, 3, 20),
                    "closing_date": date(2025, 12, 31), "previous_closing_date": date(2024, 12, 31),
                    "updated_at": "2026-03-20", "chronology_number": "1",
                    "confidentiality": "Public", "is_public": True,
                    "is_deleted": False, "account_type": "C", "currency": "EUR",
                    "duration_months": 12, "activity_code": "6201Z",
                    "structured_accounts_present": True,
                }
            ],
            schema=ACCOUNT_DEPOSIT_SCHEMA,
        ),
        accounts_path,
    )
    inputs = SirenDossierInputs(
        base.sirene_establishments,
        base.sirene_legal_units,
        base.official_evidence,
        base.official_relations,
        (accounts_path,),
    )
    result = build_siren_dossier(inputs, output_root=tmp_path / "out", memory_limit="1GB")
    connection = open_siren_dossier(result.output_dir)
    assert connection.execute("select count(*) from rne_account_deposits").fetchone()[0] == 1
    assert connection.execute(
        "select rne_public_account_period_count from siren_summary"
    ).fetchone()[0] == 1
    connection.close()
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["policy"]["rne_account_model_use_enabled"] is False
    assert manifest["consumer_contract"]["held_out_structured"] == ["rne_account_deposits"]
