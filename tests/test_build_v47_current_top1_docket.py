from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.build_v47_current_top1_docket import build_docket


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    scenes = pd.DataFrame(
        [
            {
                "audit_case_id": "case-1",
                "query_id": "q-1",
                "service_id": "svc-1",
                "sampling_stratum": "RANDOM_POPULATION",
                "frozen_top1_siret": "11111111100001",
                "replayed_top1_siret": "22222222200002",
                "input_siret": "22222222200001",
                "scene_status": "SCENE_DRIFT",
                "frozen_adjudication_label": "TOP1_WRONG",
            }
        ]
    )
    scenes_path = tmp_path / "scenes.parquet"
    scenes.to_parquet(scenes_path, index=False)
    crm_path = tmp_path / "crm.csv"
    pd.DataFrame(
        [
            {
                "SERVICE ID": "svc-1",
                "SITE": "EXEMPLE",
                "CODE_POSTAL": "75001",
                "CODE_INSEE": "75101",
                "COMMUNE": "PARIS",
                "SIRET": "22222222200001",
                "SITE_CLI_ADRESSE": "1 RUE TEST",
                "SITE_CLI_COMMUNE": "PARIS",
            }
        ]
    ).to_csv(crm_path, sep=";", index=False)
    sirene_path = tmp_path / "sirene.parquet"
    pd.DataFrame(
        [
            {
                "siret": "22222222200002",
                "siren": "222222222",
                "etatAdministratifEtablissement": "A",
                "etablissementSiege": True,
                "enseigne1Etablissement": "",
                "enseigne2Etablissement": "",
                "enseigne3Etablissement": "",
                "denominationUsuelleEtablissement": "EXEMPLE",
                "complementAdresseEtablissement": "",
                "numeroVoieEtablissement": "2",
                "indiceRepetitionEtablissement": "",
                "typeVoieEtablissement": "RUE",
                "libelleVoieEtablissement": "TEST",
                "codePostalEtablissement": "75001",
                "libelleCommuneEtablissement": "PARIS",
                "codeCommuneEtablissement": "75101",
                "activitePrincipaleEtablissement": "00.00Z",
            }
        ]
    ).to_parquet(sirene_path, index=False)
    contract_path = tmp_path / "contract.md"
    contract_path.write_text("frozen\n", encoding="utf-8")
    return scenes_path, crm_path, sirene_path, contract_path


def test_build_docket_binds_only_current_top1(tmp_path: Path) -> None:
    scenes, crm, sirene, contract = _write_inputs(tmp_path)
    target = build_docket(
        scenes_path=scenes,
        crm_path=crm,
        sirene_path=sirene,
        contract_path=contract,
        output_root=tmp_path / "out",
        enforce_canonical=False,
    )
    docket = pd.read_parquet(target / "docket.parquet")
    assert docket["siret_to_adjudicate"].tolist() == ["22222222200002"]
    assert docket["evidence_partition"].tolist() == ["random_sealed"]
    assert docket["old_label_is_search_lead_only"].all()
    assert (target / "manifest.json").exists()


def test_build_docket_rejects_unchanged_top1(tmp_path: Path) -> None:
    scenes, crm, sirene, contract = _write_inputs(tmp_path)
    frame = pd.read_parquet(scenes)
    frame["frozen_top1_siret"] = frame["replayed_top1_siret"]
    frame.to_parquet(scenes, index=False)
    with pytest.raises(ValueError, match="unchanged"):
        build_docket(
            scenes_path=scenes,
            crm_path=crm,
            sirene_path=sirene,
            contract_path=contract,
            output_root=tmp_path / "out",
            enforce_canonical=False,
        )
