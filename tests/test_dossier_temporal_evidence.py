from __future__ import annotations

from datetime import date
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.xgb_matcher.dossier_temporal_evidence import (
    materialize_dossier_temporal_layers,
)
from src.xgb_matcher.official_evidence_tantivy import (
    iter_official_evidence_documents,
)


def _write_fixture(root: Path) -> None:
    root.mkdir()
    names = []
    for index in range(20):
        names.append(
            {
                "siren": "123456789",
                "siret": "12345678900011",
                "subject_kind": "SIRET",
                "source": "RNE",
                "name_kind": "TRADE_NAME",
                "raw_value": f"Nom {index}",
                "normalized_value": f"NOM {index:02d}",
                "valid_from": date(2020, 1, 1),
                "valid_to": None,
                "is_current": index >= 10,
                "evidence_id": f"rne-name-{index}",
                "source_record_id": "rne-record",
                "observed_at": date(2026, 3, 4),
                "source_priority": 80,
            }
        )
    names.extend(
        [
            {
                **names[0],
                "subject_kind": "SIREN",
                "siret": "",
                "normalized_value": "SOCIETE RNE",
                "name_kind": "LEGAL_NAME",
                "evidence_id": "rne-legal",
            },
            {
                **names[0],
                "siren": "987654321",
                "siret": "98765432100029",
                "source": "BODACC",
                "normalized_value": "ANCIENNE ENSEIGNE",
                "evidence_id": "bodacc-name",
            },
        ]
    )
    pq.write_table(pa.Table.from_pylist(names), root / "name_evidence.parquet")

    address_template = {
        "subject_kind": "SIRET",
        "raw_value": "10 rue des Fleurs",
        "normalized_value": "10 RUE DES FLEURS",
        "postcode": "75001",
        "insee": "75056",
        "street_number": "10",
        "street_number_suffix": "",
        "valid_from": date(2019, 1, 1),
        "valid_to": None,
        "is_current": False,
        "source_record_id": "record",
        "observed_at": date(2026, 3, 4),
        "source_priority": 80,
    }
    addresses = [
        {
            **address_template,
            "siren": "123456789",
            "siret": "12345678900011",
            "source": "RNE",
            "evidence_id": "rne-address",
        },
        {
            **address_template,
            "siren": "987654321",
            "siret": "98765432100029",
            "source": "BODACC",
            "evidence_id": "bodacc-address",
        },
    ]
    pq.write_table(pa.Table.from_pylist(addresses), root / "address_evidence.parquet")

    relation_schema = pa.schema(
        [
            ("relation_id", pa.string()),
            ("source", pa.string()),
            ("source_record_id", pa.string()),
            ("relation_type", pa.string()),
            ("from_kind", pa.string()),
            ("from_identifier", pa.string()),
            ("to_kind", pa.string()),
            ("to_identifier", pa.string()),
            ("effective_date", pa.date32()),
            ("observed_at", pa.date32()),
            ("source_priority", pa.int16()),
        ]
    )
    relations = pa.Table.from_pylist(
        [
            {
                "relation_id": "relation-1",
                "source": "BODACC",
                "source_record_id": "bodacc-record",
                "relation_type": "LEGAL_UNIT_SUCCESSION",
                "from_kind": "SIREN",
                "from_identifier": "987654321",
                "to_kind": "SIREN",
                "to_identifier": "123456789",
                "effective_date": date(2021, 1, 1),
                "observed_at": date(2026, 3, 4),
                "source_priority": 70,
            }
        ],
        schema=relation_schema,
    )
    pq.write_table(relations, root / "relations.parquet")
    (root / "manifest.json").write_text(json.dumps({"build": "fixture"}))


def test_temporal_projection_is_source_separated_bounded_and_dated(tmp_path: Path):
    dossier = tmp_path / "dossier"
    _write_fixture(dossier)

    output = materialize_dossier_temporal_layers(
        dossier,
        tmp_path / "layers",
        duckdb_temp_directory=tmp_path / "duckdb",
        duckdb_memory_limit="1GB",
        duckdb_threads=1,
    )

    manifest = json.loads((output / "manifest.json").read_text())
    assert {layer["source"] for layer in manifest["layers"]} == {"RNE", "BODACC"}
    rne_documents = list(
        iter_official_evidence_documents(
            output / "rne/official_evidence.parquet",
            output / "rne/official_relation.parquet",
            duckdb_temp_directory=tmp_path / "iterator-duckdb",
            duckdb_memory_limit="1GB",
            duckdb_threads=1,
        )
    )
    rne_site = next(document for document in rne_documents if document.siret)
    assert rne_site.evidence_sources == ("RNE",)
    # Sixteen site-level names plus the independently capped parent legal name.
    assert len(rne_site.names) == 17
    assert "SOCIETE RNE" in rne_site.names
    assert rne_site.temporal_valid_from == "2019-01-01"
    assert rne_site.temporal_observed_at == "2026-03-04"
