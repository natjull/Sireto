#!/usr/bin/env python3
"""Freeze the 37 V4.7 current-top1 dossiers before evidence collection."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import pandas as pd
import pyarrow.dataset as pads

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402


SCHEMA_VERSION = "sireto-v4.7-current-top1-docket-1"
EXPECTED_SCENES_SHA256 = (
    "72540dcdba6f33da0eb1875ef4bcdc8c44a2cd10083589b5e1683098cd954a08"
)
EXPECTED_DRIFT_COUNT = 37
EXPECTED_RANDOM_COUNT = 8
SIRENE_COLUMNS = [
    "siret",
    "siren",
    "etatAdministratifEtablissement",
    "etablissementSiege",
    "enseigne1Etablissement",
    "enseigne2Etablissement",
    "enseigne3Etablissement",
    "denominationUsuelleEtablissement",
    "complementAdresseEtablissement",
    "numeroVoieEtablissement",
    "indiceRepetitionEtablissement",
    "typeVoieEtablissement",
    "libelleVoieEtablissement",
    "codePostalEtablissement",
    "libelleCommuneEtablissement",
    "codeCommuneEtablissement",
    "activitePrincipaleEtablissement",
]


def _normalise_siret(value: Any) -> str:
    digits = "".join(character for character in str(value or "") if character.isdigit())
    return digits.zfill(14) if digits else ""


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_docket(
    *,
    scenes_path: Path,
    crm_path: Path,
    sirene_path: Path,
    contract_path: Path,
    output_root: Path,
    enforce_canonical: bool = True,
) -> Path:
    scenes_path = Path(scenes_path).resolve()
    crm_path = Path(crm_path).resolve()
    sirene_path = Path(sirene_path).resolve()
    contract_path = Path(contract_path).resolve()
    input_hash = file_sha256(scenes_path)
    if enforce_canonical and input_hash != EXPECTED_SCENES_SHA256:
        raise ValueError("V4.7 scene compatibility hash mismatch")

    scenes = pd.read_parquet(scenes_path)
    required = {
        "audit_case_id",
        "query_id",
        "service_id",
        "sampling_stratum",
        "frozen_top1_siret",
        "replayed_top1_siret",
        "input_siret",
        "scene_status",
        "frozen_adjudication_label",
    }
    missing = required - set(scenes.columns)
    if missing:
        raise ValueError(f"V4.7 scenes missing columns: {sorted(missing)}")
    drift = scenes.loc[scenes["scene_status"].astype(str).eq("SCENE_DRIFT")].copy()
    if enforce_canonical and len(drift) != EXPECTED_DRIFT_COUNT:
        raise ValueError("V4.7 requires exactly 37 SCENE_DRIFT dossiers")
    if drift["audit_case_id"].astype(str).duplicated().any():
        raise ValueError("V4.7 audit_case_id values are not unique")
    for column in ("frozen_top1_siret", "replayed_top1_siret", "input_siret"):
        drift[column] = drift[column].map(_normalise_siret)
    if drift["replayed_top1_siret"].str.len().ne(14).any():
        raise ValueError("V4.7 contains an invalid current top1 SIRET")
    if drift["replayed_top1_siret"].eq(drift["frozen_top1_siret"]).any():
        raise ValueError("V4.7 drift population contains an unchanged top1")
    random_count = int(
        drift["sampling_stratum"].astype(str).eq("RANDOM_POPULATION").sum()
    )
    if enforce_canonical and random_count != EXPECTED_RANDOM_COUNT:
        raise ValueError("V4.7 requires exactly eight random drift dossiers")

    crm = pd.read_csv(crm_path, sep=";", dtype=str).fillna("")
    crm_columns = [
        "SERVICE ID",
        "SITE",
        "CODE_POSTAL",
        "CODE_INSEE",
        "COMMUNE",
        "SIRET",
        "SITE_CLI_ADRESSE",
        "SITE_CLI_COMMUNE",
    ]
    crm_missing = set(crm_columns) - set(crm.columns)
    if crm_missing:
        raise ValueError(f"CRM source missing columns: {sorted(crm_missing)}")
    wanted_service_ids = set(drift["service_id"].astype(str))
    crm = crm.loc[
        crm["SERVICE ID"].astype(str).isin(wanted_service_ids),
        crm_columns,
    ].drop_duplicates()
    if crm["SERVICE ID"].duplicated().any():
        raise ValueError("CRM source has ambiguous SERVICE ID rows")
    docket = drift.merge(
        crm,
        left_on="service_id",
        right_on="SERVICE ID",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not docket["_merge"].eq("both").all():
        raise ValueError("V4.7 CRM join is incomplete")
    docket = docket.drop(columns=["_merge", "SERVICE ID"])

    wanted = sorted(set(docket["replayed_top1_siret"]))
    dataset = pads.dataset(sirene_path, format="parquet")
    current = dataset.to_table(
        columns=SIRENE_COLUMNS,
        filter=pads.field("siret").isin(wanted),
    ).to_pandas()
    current["siret"] = current["siret"].map(_normalise_siret)
    if current["siret"].duplicated().any() or set(current["siret"]) != set(wanted):
        raise ValueError("V4.7 current top1 lookup is incomplete or duplicated")
    current = current.rename(
        columns={column: f"current_{column}" for column in current.columns}
    )
    docket = docket.merge(
        current,
        left_on="replayed_top1_siret",
        right_on="current_siret",
        how="left",
        validate="one_to_one",
    )
    docket["siret_to_adjudicate"] = docket["replayed_top1_siret"]
    docket["siren_to_adjudicate"] = docket["replayed_top1_siret"].str[:9]
    docket["evidence_partition"] = docket["sampling_stratum"].map(
        lambda value: (
            "random_sealed"
            if str(value) == "RANDOM_POPULATION"
            else "targeted"
        )
    )
    docket["old_label_is_search_lead_only"] = True
    docket = docket.sort_values("audit_case_id").reset_index(drop=True)

    identity = {
        "schema_version": SCHEMA_VERSION,
        "scenes_sha256": input_hash,
        "crm_sha256": file_sha256(crm_path),
        "sirene_sha256": file_sha256(sirene_path),
        "contract_sha256": file_sha256(contract_path),
        "case_ids": docket["audit_case_id"].astype(str).tolist(),
        "sirets_to_adjudicate": docket["siret_to_adjudicate"].tolist(),
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    target = Path(output_root).resolve() / build_id
    if target.exists():
        raise FileExistsError(f"Immutable V4.7 docket already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{build_id}.tmp-", dir=target.parent))
    try:
        docket_path = staging / "docket.parquet"
        docket.to_parquet(docket_path, index=False)
        summary = {
            "case_count": int(len(docket)),
            "random_sealed_count": random_count,
            "targeted_count": int(len(docket) - random_count),
            "distinct_current_top1_count": int(
                docket["siret_to_adjudicate"].nunique()
            ),
            "old_labels_transported": 0,
            "model_training_performed": False,
            "test_opened": False,
        }
        summary_path = staging / "summary.json"
        _json_dump(summary_path, summary)
        manifest = {
            **identity,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "inputs": {
                "scenes": {"path": str(scenes_path), "sha256": input_hash},
                "crm": {"path": str(crm_path), "sha256": file_sha256(crm_path)},
                "sirene": {
                    "path": str(sirene_path),
                    "sha256": file_sha256(sirene_path),
                },
                "contract": {
                    "path": str(contract_path),
                    "sha256": file_sha256(contract_path),
                },
            },
            "outputs": {
                docket_path.name: file_sha256(docket_path),
                summary_path.name: file_sha256(summary_path),
            },
            "summary": summary,
        }
        _json_dump(staging / "manifest.json", manifest)
        staging.rename(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--crm", type=Path, required=True)
    parser.add_argument("--sirene", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        build_docket(
            scenes_path=args.scenes,
            crm_path=args.crm,
            sirene_path=args.sirene,
            contract_path=args.contract,
            output_root=args.output_root,
        )
    )


if __name__ == "__main__":
    main()
