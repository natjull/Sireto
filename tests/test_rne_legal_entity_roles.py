from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.xgb_matcher.hierarchical_retrieval import InMemoryBackend, RetrievalQuery
from src.xgb_matcher.official_evidence_retrieval import (
    OfficialEvidenceRetrievalConfig,
    OfficialEvidenceRetriever,
)
from src.xgb_matcher.rne_legal_entity_roles import (
    LegalEntityRoleTantivyBackend,
    build_legal_entity_role_index,
    extract_legal_entity_roles,
    write_legal_entity_roles,
)


def _record() -> dict:
    return {
        "id": "record-1",
        "siren": "123456789",
        "updatedAt": "2026-03-04T14:00:00Z",
        "formality": {
            "siren": "123456789",
            "content": {
                "personneMorale": {
                    "composition": {
                        "pouvoirs": [
                            {
                                "typeDePersonne": "ENTREPRISE",
                                "roleEntreprise": "71",
                                "entreprise": {
                                    "siren": "987654321",
                                    "denomination": "CONSEIL AUDIT ET SYNTHESE",
                                },
                                "actif": True,
                            },
                            {
                                "typeDePersonne": "ENTREPRISE",
                                "roleEntreprise": "99",
                                "entreprise": {"denomination": "FIDUCIAIRE DE LA SEINE"},
                                "actif": True,
                            },
                            {
                                "typeDePersonne": "INDIVIDU",
                                "individu": {"descriptionPersonne": {"nom": "DUPONT"}},
                            },
                        ]
                    }
                }
            },
        },
    }


def test_extracts_only_legal_entities_and_preserves_structured_siren() -> None:
    roles = extract_legal_entity_roles(_record())
    assert len(roles) == 2
    assert {role.related_denomination for role in roles} == {
        "CONSEIL AUDIT ET SYNTHESE",
        "FIDUCIAIRE DE LA SEINE",
    }
    assert {role.related_siren for role in roles} == {"", "987654321"}
    assert all("DUPONT" not in role.related_denomination for role in roles)


def test_writes_private_free_role_table_and_builds_siren_overlay(tmp_path: Path) -> None:
    roles = tmp_path / "roles.parquet"
    counts = write_legal_entity_roles([_record()], roles, batch_size=1)
    assert counts == {
        "source_records": 1,
        "legal_entity_roles": 2,
        "structured_related_siren_roles": 1,
    }
    table = pq.read_table(roles)
    assert "related_denomination" in table.column_names
    assert not any("birth" in name or "address" in name for name in table.column_names)

    documents = tmp_path / "siren_documents.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "document_id": "123456789",
                    "siren": "123456789",
                    "document_kind": "SIREN",
                    "insee": "75056",
                    "postcode": "75002",
                }
            ]
        ),
        documents,
    )
    output = build_legal_entity_role_index(
        roles_path=roles, siren_documents_path=documents, output_root=tmp_path / "index"
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["num_documents"] == 1
    assert manifest["privacy"]["natural_persons_indexed"] is False
    backend = LegalEntityRoleTantivyBackend(output)
    result = backend.search_sirens(
        RetrievalQuery(
            name="CONSEIL AUDIT ET SYNTHESE",
            address="",
            postcode="75002",
            insee="75056",
            number="",
        ),
        10,
    )
    assert result[0][0] == "123456789"

    base = InMemoryBackend(
        [
            {
                "siret": "12345678900011",
                "siren": "123456789",
                "insee": "75056",
                "postcode": "75002",
                "names": ["SOCIETE REPRESENTEE"],
                "addresses": ["10 RUE DE LA PAIX"],
                "number": "10",
                "state": "A",
            }
        ]
    )
    retriever = OfficialEvidenceRetriever(
        base,
        config=OfficialEvidenceRetrievalConfig(
            max_union_candidates=20,
            exact_limit=10,
            word_limit=10,
            character_limit=10,
            historical_limit=10,
            supporting_limit=10,
            siren_limit=5,
            parent_siren_limit=5,
            sites_per_siren=5,
            relation_seed_limit=10,
            relation_limit=10,
            search_workers=2,
        ),
        legal_entity_role_backend=backend,
    )
    retrieved = retriever.retrieve(
        {
            "crm_name": "CONSEIL AUDIT ET SYNTHESE",
            "crm_address": "10 rue de la Paix",
            "crm_postcode": "75002",
            "crm_insee": "75056",
        }
    )
    candidate = next(item for item in retrieved.candidates if item.siret == "12345678900011")
    assert "legal_entity_role_name" in candidate.channels
