from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.xgb_matcher.hierarchical_retrieval import (
    HierarchicalRetrievalConfig,
    InMemoryBackend,
    IndexedEstablishment,
    RetrievalQuery,
)
from src.xgb_matcher.official_evidence import (
    OfficialAddress,
    OfficialEvidence,
    OfficialName,
    OfficialNameKind,
    OfficialRelation,
    OfficialRelationType,
    OfficialSource,
    OfficialSubjectKind,
    QuarantineReason,
    official_evidence_arrow_schema,
    official_relation_arrow_schema,
)
from src.xgb_matcher.official_evidence_builder import (
    SnapshotRole,
    SnapshotSpec,
    build_official_evidence_layer,
    canonicalize_snapshot_record,
    resolve_evidence_precedence,
)
from src.xgb_matcher.official_evidence_retrieval import (
    LTR_CHANNELS,
    OfficialEvidenceRetrievalConfig,
    OfficialEvidenceRetriever,
    retrieve_official_evidence_union_to_parquet,
)
from src.xgb_matcher.official_evidence_tantivy import (
    OfficialEvidenceTantivyBackend,
    build_official_evidence_tantivy_overlay,
)
from src.xgb_matcher.retrieval_ltr_admission import (
    AdmissionConfig,
    build_internal_union,
)


def _spec(path: Path, source: OfficialSource, role: SnapshotRole) -> SnapshotSpec:
    return SnapshotSpec(path=path, source=source, role=role, observed_at="2026-08-18")


def _write_parquet(path: Path, rows: list[dict]) -> Path:
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def _sirene_establishment(
    *,
    siret: str = "12345678900011",
    name: str = "Société Étoile",
    insee: str = "75056",
    postcode: str = "75001",
    street: str = "DES FLEURS",
) -> dict:
    return {
        "siret": siret,
        "siren": siret[:9],
        "enseigne1Etablissement": name,
        "denominationUsuelleEtablissement": None,
        "numeroVoieEtablissement": "10",
        "indiceRepetitionEtablissement": None,
        "typeVoieEtablissement": "RUE",
        "libelleVoieEtablissement": street,
        "complementAdresseEtablissement": None,
        "codePostalEtablissement": postcode,
        "codeCommuneEtablissement": insee,
        "etatAdministratifEtablissement": "A",
        "etablissementSiege": True,
        "dateCreationEtablissement": "2020-01-01",
    }


def test_canonical_schema_keeps_raw_and_normalized_but_has_no_sensitive_fields():
    name = OfficialName("  Société Étoile  ", OfficialNameKind.LEGAL)
    address = OfficialAddress(
        "10, rue de l'Été", postcode="75001", insee="75056", number="10"
    )
    assert name.raw_value == "Société Étoile"
    assert name.normalized_value == "SOCIETE ETOILE"
    assert address.raw_value == "10, rue de l'Été"
    assert address.normalized_value == "10 RUE DE L ETE"

    evidence = OfficialEvidence(
        source=OfficialSource.RNE,
        source_record_id="rne-1",
        subject_kind=OfficialSubjectKind.SIRET,
        siren="123456789",
        siret="12345678900011",
        names=(name,),
        addresses=(address,),
    )
    round_trip = OfficialEvidence.from_dict(evidence.to_dict())
    assert round_trip == evidence
    fields = set(official_evidence_arrow_schema().names)
    fields.update(official_relation_arrow_schema().names)
    assert not fields & {
        "dirigeants",
        "beneficial_owners",
        "beneficiaires_effectifs",
        "bodacc_text",
        "texte",
    }


def test_rne_formalites_v4_content_paths_and_reuse_opposition(tmp_path: Path):
    formality = {
        "diffusionCommerciale": True,
        "diffusionINSEE": "O",
        "content": {
            "personneMorale": {
                "identite": {
                    "entreprise": {
                        "siren": "123456789",
                        "denomination": "Institut National Exemple",
                        "nomCommercial": "INPI EXEMPLE",
                    },
                    "description": {"sigle": "INE"},
                },
                "adresseEntreprise": {
                    "adresse": {
                        "numVoie": "15",
                        "typeVoie": "RUE",
                        "voie": "DES MINIMES",
                        "codePostal": "92400",
                        "codeInseeCommune": "92026",
                    }
                },
            }
        },
    }
    record = {
        "company": {
            "id": "company-123456789",
            "siren": "123456789",
            "formality": formality,
        }
    }
    spec = _spec(
        tmp_path / "rne-api.jsonl",
        OfficialSource.RNE,
        SnapshotRole.RNE_RECORDS,
    )
    accepted = canonicalize_snapshot_record(spec, record, ordinal=1)
    assert len(accepted.evidence) == 1
    evidence = accepted.evidence[0]
    assert {item.normalized_value for item in evidence.names} == {
        "INSTITUT NATIONAL EXEMPLE",
        "INPI EXEMPLE",
        "INE",
    }
    assert evidence.addresses[0].normalized_value == "15 RUE DES MINIMES"
    assert evidence.addresses[0].insee == "92026"

    opposed = canonicalize_snapshot_record(
        spec,
        {
            "company": {
                **record["company"],
                "formality": {**formality, "diffusionCommerciale": "false"},
            }
        },
        ordinal=2,
    )
    assert opposed.evidence == ()
    assert opposed.quarantine[0].reason is QuarantineReason.OFFICIAL_REUSE_OPPOSITION


def test_duplicate_precedence_keeps_latest_observation():
    common = dict(
        source=OfficialSource.RNE,
        subject_kind=OfficialSubjectKind.SIRET,
        siren="123456789",
        siret="12345678900011",
        names=(OfficialName("Étoile", OfficialNameKind.LEGAL),),
    )
    old = OfficialEvidence(
        **common, source_record_id="old", observed_at="2026-08-01"
    )
    new = OfficialEvidence(
        **common, source_record_id="new", observed_at="2026-08-18"
    )
    accepted, quarantined = resolve_evidence_precedence([old, new])
    assert [item.source_record_id for item in accepted] == ["new"]
    assert quarantined[0].reason is QuarantineReason.DUPLICATE_LOWER_PRECEDENCE


def test_bodacc_ods_nested_json_is_allowlisted_and_structured(tmp_path: Path):
    record = {
        "numeroAnnonce": "A-42",
        "dateparution": "2026-08-18",
        "listepersonnes": json.dumps(
            [
                {
                    "personne": {
                        "numeroImmatriculation": {
                            "numeroIdentification": "123 456 789"
                        },
                        "denomination": "Société Étoile",
                        "adresseSiegeSocial": {
                            "numeroVoie": "10",
                            "typeVoie": "RUE",
                            "nomVoie": "DE L'ÉTÉ",
                            "codePostal": "75001",
                            "codeCommune": "75056",
                        },
                        "dirigeants": [{"nom": "NE DOIT PAS SORTIR"}],
                    }
                }
            ]
        ),
        "listeetablissements": [
            {
                "etablissement": {
                    "siret": "12345678900011",
                    "enseigne": "Étoile Boutique",
                    "adresse": {
                        "numeroVoie": "10",
                        "typeVoie": "RUE",
                        "nomVoie": "DE L'ÉTÉ",
                        "codePostal": "75001",
                        "codeCommune": "75056",
                    },
                }
            }
        ],
        "listeprecedentproprietaire": json.dumps(
            [
                {
                    "precedentProprietairePM": {
                        "numeroImmatriculation": {
                            "numeroIdentification": "987654321"
                        },
                        "siret": "98765432100019",
                    }
                }
            ]
        ),
        "texte": "FAUX IDENTIFIANT 11111111100011 ET TEXTE COMPLET INTERDIT",
    }
    result = canonicalize_snapshot_record(
        _spec(
            tmp_path / "incremental.jsonl",
            OfficialSource.BODACC,
            SnapshotRole.BODACC_ANNOUNCEMENTS,
        ),
        record,
        ordinal=1,
    )
    assert {item.siren for item in result.evidence} == {"123456789"}
    flattened = json.dumps(
        [item.to_dict() for item in result.evidence], ensure_ascii=False
    )
    assert "Société Étoile" in flattened
    assert "SOCIETE ETOILE" in flattened
    assert "NE DOIT PAS SORTIR" not in flattened
    assert "FAUX IDENTIFIANT" not in flattened
    assert {
        (item.relation_type, item.from_identifier, item.to_identifier)
        for item in result.relations
    } == {
        (
            OfficialRelationType.ASSET_TRANSFER,
            "987654321",
            "123456789",
        ),
        (
            OfficialRelationType.ESTABLISHMENT_SUCCESSION,
            "98765432100019",
            "12345678900011",
        ),
    }


def test_bodacc_text_never_creates_a_relation(tmp_path: Path):
    result = canonicalize_snapshot_record(
        _spec(
            tmp_path / "incremental.jsonl",
            OfficialSource.BODACC,
            SnapshotRole.BODACC_ANNOUNCEMENTS,
        ),
        {
            "numeroAnnonce": "A-43",
            "texte": "cession de 123456789 à 987654321",
        },
        ordinal=1,
    )
    assert result.evidence == ()
    assert result.relations == ()
    assert result.quarantine[0].reason is QuarantineReason.UNSTRUCTURED_RELATION


def test_streaming_builder_applies_precedence_and_quarantines_geo(tmp_path: Path):
    sirene = _write_parquet(
        tmp_path / "sirene.parquet", [_sirene_establishment()]
    )
    rne = tmp_path / "rne.jsonl"
    rne.write_text(
        json.dumps(
            {
                "id": "rne-conflict",
                "siren": "123456789",
                "siret": "12345678900011",
                "denomination": "Nouveau Nom RNE",
                "adresse": {
                    "numeroVoie": "20",
                    "typeVoie": "RUE",
                    "libelleVoie": "DES LILAS",
                    "codePostal": "69001",
                    "codeCommune": "69123",
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    result = build_official_evidence_layer(
        [
            _spec(sirene, OfficialSource.SIRENE_CURRENT, SnapshotRole.SIRENE_ESTABLISHMENTS),
            _spec(rne, OfficialSource.RNE, SnapshotRole.RNE_RECORDS),
        ],
        tmp_path / "canonical",
        batch_size=1,
    )
    evidence = pq.read_table(result.evidence_path).to_pylist()
    rne_row = next(item for item in evidence if item["source"] == "RNE")
    assert rne_row["names"][0]["raw_value"] == "Nouveau Nom RNE"
    assert rne_row["names"][0]["normalized_value"] == "NOUVEAU NOM RNE"
    assert rne_row["addresses"] == []
    quarantine = pq.read_table(result.quarantine_path).to_pylist()
    assert {
        item["reason"] for item in quarantine
    } == {"LOWER_PRECEDENCE_CURRENT_GEO_CONFLICT"}
    assert "raw_record" not in pq.ParquetFile(result.quarantine_path).schema.names


def _build_complete_fixture(tmp_path: Path):
    sirene = _write_parquet(
        tmp_path / "sirene.parquet",
        [
            _sirene_establishment(),
            _sirene_establishment(
                siret="12345678900029",
                name="AUTRE SITE",
                street="DES ROSES",
            ),
        ],
    )
    legal = _write_parquet(
        tmp_path / "legal.parquet",
        [
            {
                "siren": "123456789",
                "denominationUniteLegale": "Étoile Groupe",
                "etatAdministratifUniteLegale": "A",
            }
        ],
    )
    succession = _write_parquet(
        tmp_path / "succession.parquet",
        [
            {
                "siretEtablissementPredecesseur": "12345678900011",
                "siretEtablissementSuccesseur": "12345678900029",
            }
        ],
    )
    rne = tmp_path / "rne.jsonl"
    rne.write_text(
        json.dumps(
            {
                "id": "rne-name",
                "siren": "123456789",
                "siret": "12345678900011",
                "denomination": "Nouveau Nom RNE",
                "adresse": {
                    "numeroVoie": "10",
                    "typeVoie": "RUE",
                    "libelleVoie": "DES FLEURS",
                    "codePostal": "75001",
                    "codeCommune": "75056",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    canonical = build_official_evidence_layer(
        [
            _spec(sirene, OfficialSource.SIRENE_CURRENT, SnapshotRole.SIRENE_ESTABLISHMENTS),
            _spec(legal, OfficialSource.SIRENE_CURRENT, SnapshotRole.SIRENE_LEGAL_UNITS),
            _spec(succession, OfficialSource.SIRENE_SUCCESSION, SnapshotRole.SIRENE_SUCCESSIONS),
            _spec(rne, OfficialSource.RNE, SnapshotRole.RNE_RECORDS),
        ],
        tmp_path / "canonical",
        batch_size=2,
    )
    overlay = build_official_evidence_tantivy_overlay(
        canonical.evidence_path,
        canonical.relation_path,
        tmp_path / "overlay_indices",
        writer_heap_bytes=30_000_000,
        writer_threads=1,
        commit_every=10,
        batch_size=2,
        duckdb_temp_directory=tmp_path / "duckdb_spill",
        duckdb_memory_limit="256MB",
    )
    return canonical, overlay


def test_overlay_is_content_addressed_and_base_backend_compatible(tmp_path: Path):
    canonical, overlay = _build_complete_fixture(tmp_path)
    assert overlay.parent.name == "overlay_indices"
    assert len(overlay.name) == 16
    assert (overlay / "manifest.sha256").is_file()
    assert build_official_evidence_tantivy_overlay(
        canonical.evidence_path,
        canonical.relation_path,
        tmp_path / "overlay_indices",
        writer_heap_bytes=30_000_000,
        writer_threads=1,
        commit_every=10,
        batch_size=2,
        duckdb_temp_directory=tmp_path / "duckdb_spill",
        duckdb_memory_limit="256MB",
    ) == overlay

    backend = OfficialEvidenceTantivyBackend(overlay)
    hits = backend.search(
        RetrievalQuery(
            name="NOUVEAU NOM RNE",
            address="10 RUE DES FLEURS",
            insee="75056",
            postcode="75001",
        ),
        "name_exact",
        10,
    )
    assert hits[0].record.siret == "12345678900011"
    assert hits[0].record.names == (
        "ETOILE GROUPE",
        "NOUVEAU NOM RNE",
        "SOCIETE ETOILE",
    )
    assert backend.by_siret("12345678900011").linked_sirets == (
        "12345678900029",
    )


def _record(
    siret: str,
    *,
    insee: str = "75056",
    postcode: str = "75001",
    linked_sirets: tuple[str, ...] = (),
    payload: dict | None = None,
) -> IndexedEstablishment:
    return IndexedEstablishment(
        siret=siret,
        siren=siret[:9],
        insee=insee,
        postcode=postcode,
        names=("NOUVEAU NOM RNE",),
        addresses=("10 RUE DES FLEURS",),
        number="10",
        active=True,
        linked_sirets=linked_sirets,
        payload=payload or {},
    )


def test_union_maps_channels_and_geo_guards_siret_relations():
    source = _record(
        "12345678900011",
        linked_sirets=("12345678900029", "12345678900037"),
    )
    same_geo = _record("12345678900029")
    wrong_geo = _record("12345678900037", insee="69123", postcode="69001")
    overlay_record = _record(
        "12345678900011",
        payload={"official_evidence_sources": ["RNE", "BODACC"]},
    )
    retriever = OfficialEvidenceRetriever(
        InMemoryBackend([source, same_geo, wrong_geo]),
        InMemoryBackend([overlay_record]),
        OfficialEvidenceRetrievalConfig(
            max_union_candidates=20,
            exact_limit=10,
            word_limit=10,
            character_limit=10,
            siren_limit=3,
            sites_per_siren=3,
            relation_seed_limit=10,
            relation_limit=10,
            search_workers=2,
        ),
    )
    result = retriever.retrieve(
        RetrievalQuery(
            name="NOUVEAU NOM RNE",
            address="10 RUE DES FLEURS",
            insee="75056",
            postcode="75001",
        )
    )
    candidates = {item.siret: item for item in result.candidates}
    assert "12345678900029" in candidates
    assert "official_successor" in candidates["12345678900029"].channels
    assert "12345678900037" not in candidates
    source_channels = candidates["12345678900011"].channels
    assert "rne_name" in source_channels
    assert "bodacc_name" in source_channels
    assert "hierarchical" in source_channels


def test_builder_overlay_union_parquet_is_ltr_consumable(tmp_path: Path):
    _canonical, overlay = _build_complete_fixture(tmp_path)
    retriever = OfficialEvidenceRetriever(
        InMemoryBackend([]),
        OfficialEvidenceTantivyBackend(overlay),
        OfficialEvidenceRetrievalConfig(
            max_union_candidates=20,
            exact_limit=10,
            word_limit=10,
            character_limit=10,
            siren_limit=3,
            sites_per_siren=3,
            relation_seed_limit=10,
            relation_limit=10,
            search_workers=2,
        ),
    )
    output = retrieve_official_evidence_union_to_parquet(
        retriever,
        [
            {
                "query_id": "q-1",
                "fold": 0,
                "ground_truth_siret": "12345678900011",
                "ground_truth_state": "A",
                "crm_name": "Nouveau Nom RNE",
                "crm_address": "10 rue des Fleurs",
                "crm_postcode": "75001",
                "crm_insee": "75056",
                "identifiable": True,
                "acceptable_sirets_operational": [
                    "12345678900011",
                    "12345678900029",
                ],
            }
        ],
        tmp_path / "union.parquet",
        batch_size=2,
    )
    table = pq.read_table(output)
    columns = set(table.column_names)
    required = {
        "fold",
        "ground_truth_siret",
        "crm_name",
        "identifiable",
        "acceptable_sirets_operational",
        "candidate_names",
        "candidate_addresses",
        "candidate_insee",
        "candidate_postcode",
        "retrieval_source",
        "retrieval_rank",
        "retrieval_score",
        "retrieval_latency_ms",
    }
    for channel in LTR_CHANNELS:
        required.update({f"{channel}_rank", f"{channel}_score"})
    assert required <= columns
    rows = table.to_pylist()
    match = next(item for item in rows if item["siret"] == "12345678900011")
    assert match["fold"] == 0
    assert match["ground_truth_siret"] == "12345678900011"
    assert match["rne_name_rank"] is not None
    assert match["candidate_insee"] == "75056"
    assert match["retrieval_latency_ms"] >= 0.0
    union, diagnostics = build_internal_union(
        pd.read_parquet(output), AdmissionConfig.load(), allowed_folds={0}
    )
    assert diagnostics["query_count"] == 1
    assert union.iloc[0]["gt_siret"] == "12345678900011"
    assert bool(union.iloc[0]["identifiable_exact"]) is True
    assert union.iloc[0]["source_rne_name"] == 1
