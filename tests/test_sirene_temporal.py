from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.xgb_matcher.hierarchical_retrieval import InMemoryBackend, IndexedEstablishment
from src.xgb_matcher.official_evidence_retrieval import (
    OfficialEvidenceRetrievalConfig,
    OfficialEvidenceRetriever,
)
from src.xgb_matcher.official_evidence_tantivy import OfficialEvidenceTantivyBackend
from src.xgb_matcher.sirene_temporal import (
    SireneTemporalInputs,
    build_temporal_projection,
    project_temporal_candidate_features,
    temporal_preflight,
)
from src.xgb_matcher.sirene_temporal_tantivy import build_temporal_tantivy_overlay


def _write(path: Path, rows: list[dict]) -> Path:
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def _inputs(tmp_path: Path) -> SireneTemporalInputs:
    siren = "123456789"
    siret = siren + "00011"
    successor = siren + "00029"
    return SireneTemporalInputs(
        establishment_history=_write(tmp_path / "eh.parquet", [{
            "siren": siren, "siret": siret, "dateDebut": "2010-01-01", "dateFin": "2018-01-01",
            "etatAdministratifEtablissement": "A", "enseigne1Etablissement": "ANCIENNE ENSEIGNE",
            "enseigne2Etablissement": None, "enseigne3Etablissement": None,
            "denominationUsuelleEtablissement": None,
        }]),
        legal_unit_history=_write(tmp_path / "uh.parquet", [{
            "siren": siren, "dateDebut": "2010-01-01", "dateFin": "2018-01-01",
            "etatAdministratifUniteLegale": "A", "nomUniteLegale": None,
            "nomUsageUniteLegale": None, "denominationUniteLegale": "ANCIEN GROUPE",
            "denominationUsuelle1UniteLegale": None, "denominationUsuelle2UniteLegale": None,
            "denominationUsuelle3UniteLegale": None,
        }]),
        successions=_write(tmp_path / "succ.parquet", [{
            "siretEtablissementPredecesseur": siret,
            "siretEtablissementSuccesseur": successor,
            "dateLienSuccession": "2019-01-01", "transfertSiege": True,
            "continuiteEconomique": True, "dateDernierTraitementLienSuccession": "2020-01-01",
        }]),
        establishments=_write(tmp_path / "est.parquet", [
            {
                "siren": siren, "siret": siret, "etatAdministratifEtablissement": "F",
                "etablissementSiege": True, "numeroVoieEtablissement": "12",
                "indiceRepetitionEtablissement": None, "typeVoieEtablissement": "RUE",
                "libelleVoieEtablissement": "DES FLEURS", "codePostalEtablissement": "75001",
                "codeCommuneEtablissement": "75101", "libelleCommuneEtablissement": "PARIS",
                "enseigne1Etablissement": "NOUVELLE ENSEIGNE", "enseigne2Etablissement": None,
                "enseigne3Etablissement": None, "denominationUsuelleEtablissement": None,
            },
            {
                "siren": siren, "siret": successor, "etatAdministratifEtablissement": "A",
                "etablissementSiege": False, "numeroVoieEtablissement": "12",
                "indiceRepetitionEtablissement": None, "typeVoieEtablissement": "RUE",
                "libelleVoieEtablissement": "DES FLEURS", "codePostalEtablissement": "75001",
                "codeCommuneEtablissement": "75101", "libelleCommuneEtablissement": "PARIS",
                "enseigne1Etablissement": "NOUVELLE ENSEIGNE", "enseigne2Etablissement": None,
                "enseigne3Etablissement": None, "denominationUsuelleEtablissement": None,
            },
        ]),
        legal_units=_write(tmp_path / "ul.parquet", [{
            "siren": siren, "etatAdministratifUniteLegale": "A", "nomUniteLegale": None,
            "nomUsageUniteLegale": None, "denominationUniteLegale": "NOUVEAU GROUPE",
            "denominationUsuelle1UniteLegale": None, "denominationUsuelle2UniteLegale": None,
            "denominationUsuelle3UniteLegale": None,
        }]),
    )


def test_real_schema_preflight_contract_and_projection(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    report = temporal_preflight(inputs, siren_prefix="12")
    assert report["gate_pass"] is True
    assert report["counts"]["succession_rows"] == 1
    build = build_temporal_projection(inputs, tmp_path / "projection", siren_prefix="12")
    assert all(value > 0 for value in build.counts.values())
    siret = pq.read_table(build.output_dir / "historical_siret_identity.parquet").to_pylist()[0]
    assert siret["normalized_name"] == "ANCIENNE ENSEIGNE"
    assert siret["geo_scaffold"] is True
    assert siret["geo_scaffold_source"] == "SIRENE_CURRENT_ESTABLISHMENT"
    manifest = json.loads(build.manifest_path.read_text())
    assert manifest["semantics"]["historical_addresses_claimed"] is False


def test_missing_real_field_fails_before_build(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    _write(tmp_path / "eh-bad.parquet", [{"siren": "123456789", "siret": "12345678900011"}])
    bad = SireneTemporalInputs(
        tmp_path / "eh-bad.parquet", inputs.legal_unit_history, inputs.successions,
        inputs.establishments, inputs.legal_units,
    )
    with pytest.raises(ValueError, match="missing required columns"):
        temporal_preflight(bad, siren_prefix="12")


def test_partition_may_have_no_relation_but_global_smoke_remains_strict(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    # Preserve the empty Parquet schema, unlike an empty from_pylist table.
    pq.write_table(pq.read_table(inputs.successions).slice(0, 0), tmp_path / "succ-empty.parquet")
    partition_inputs = SireneTemporalInputs(
        inputs.establishment_history, inputs.legal_unit_history, tmp_path / "succ-empty.parquet",
        inputs.establishments, inputs.legal_units,
    )
    report = temporal_preflight(inputs, siren_prefix="12")
    build = build_temporal_projection(
        partition_inputs, tmp_path / "projection-empty-relation", siren_prefix="12",
        preflight_report=report, require_all_families=False,
    )
    assert build.counts["succession_edges"] == 0


def test_temporal_overlay_is_rescue_not_exact_and_exports_features(tmp_path: Path) -> None:
    build = build_temporal_projection(_inputs(tmp_path), tmp_path / "projection", siren_prefix="12")
    index_dir = build_temporal_tantivy_overlay([build.output_dir], tmp_path / "index")
    temporal = OfficialEvidenceTantivyBackend(index_dir)
    assert temporal.linked_sirets("12345678900011") == ("12345678900029",)
    assert temporal.linked_sirets("12345678900029") == ("12345678900011",)
    base = InMemoryBackend([
        IndexedEstablishment(
            siret="12345678900011", siren="123456789", insee="75101", postcode="75001",
            names=("NOUVELLE ENSEIGNE",), addresses=("12 RUE DES FLEURS",), number="12",
        ),
        IndexedEstablishment(
            siret="12345678900029", siren="123456789", insee="75101", postcode="75001",
            names=("NOUVELLE ENSEIGNE",), addresses=("12 RUE DES FLEURS",), number="12",
        ),
    ])
    retriever = OfficialEvidenceRetriever(
        base, config=OfficialEvidenceRetrievalConfig(require_temporal_complete=False),
        temporal_overlay_backend=temporal,
    )
    result = retriever.retrieve({"name": "ANCIENNE ENSEIGNE", "address": "12 RUE DES FLEURS", "insee": "75101"})
    candidate = next(item for item in result.candidates if item.siret == "12345678900011")
    assert "historical_name_word" in candidate.channels
    assert "name_exact" not in candidate.channels
    assert "official_successor" in {signal.channel for item in result.candidates for signal in item.signals}

    candidates = _write(tmp_path / "candidates.parquet", [{
        "query_id": "q1", "candidate_siret": "12345678900011", "candidate_siren": "123456789",
        "candidate_insee": "75101", "crm_name": "ANCIENNE ENSEIGNE",
    }])
    output = project_temporal_candidate_features(candidates, [build.output_dir], tmp_path / "features.parquet")
    row = pq.read_table(output).to_pylist()[0]
    assert row["temporal_siret_name_exact"] is True
    assert row["temporal_siren_name_count"] == 1
    assert row["succession_out_count"] == 1
    assert "[SIRENE_HISTORY_SIRET]" in row["temporal_fusion_text"]
