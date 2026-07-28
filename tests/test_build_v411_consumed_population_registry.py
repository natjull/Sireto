from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from scripts.build_v411_consumed_population_registry import (
    CRM_COLUMNS,
    build_registry,
    canonical_text,
    normalize_siret,
    row_fingerprint,
    write_build,
)


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    crm = pd.DataFrame(
        [
            {
                "SITE": "École  A",
                "CODE_POSTAL": "75001",
                "CODE_INSEE": "75101",
                "SERVICE ID": " S-1 ",
                "COMMUNE": "Paris",
                "SIRET": "123 456 789 00012",
                "SITE_CLI_ADRESSE": "1 rue A",
                "SITE_CLI_COMMUNE": "Paris",
            },
            {
                "SITE": "Mairie",
                "CODE_POSTAL": "69001",
                "CODE_INSEE": "69381",
                "SERVICE ID": "",
                "COMMUNE": "Lyon",
                "SIRET": "98765432100019",
                "SITE_CLI_ADRESSE": "2 rue B",
                "SITE_CLI_COMMUNE": "Lyon",
            },
            {
                "SITE": "Inédit",
                "CODE_POSTAL": "13001",
                "CODE_INSEE": "13201",
                "SERVICE ID": "",
                "COMMUNE": "Marseille",
                "SIRET": "11111111111111",
                "SITE_CLI_ADRESSE": "3 rue C",
                "SITE_CLI_COMMUNE": "Marseille",
            },
        ],
        columns=CRM_COLUMNS,
    )
    closed = pd.DataFrame(
        [{"crm_record_id": "S-1", "ground_truth_siret": "12345678900012"}]
    )
    fresh = pd.DataFrame(
        [
            {
                "crm_record_id": "",
                "historical_ground_truth_siret": "98765432100019",
            }
        ]
    )
    crm_path = tmp_path / "crm.csv"
    closed_path = tmp_path / "closed.parquet"
    fresh_path = tmp_path / "fresh.parquet"
    crm.to_csv(crm_path, sep=";", index=False)
    closed.to_parquet(closed_path, index=False)
    fresh.to_parquet(fresh_path, index=False)
    return crm_path, closed_path, fresh_path


def test_normalization_and_fingerprint_are_canonical() -> None:
    assert canonical_text("  école   a ") == "ÉCOLE A"
    assert normalize_siret("123 456 789 00012") == "12345678900012"
    row = pd.Series({column: "" for column in CRM_COLUMNS})
    row["SITE"] = "  école  "
    first = row_fingerprint(row)
    row["SITE"] = "ÉCOLE"
    assert row_fingerprint(row) == first
    assert len(first) == 64


def test_registry_uses_siret_when_service_id_is_missing(tmp_path: Path) -> None:
    crm_path, closed_path, fresh_path = _write_inputs(tmp_path)
    registry, audit = build_registry(
        crm_path, closed_path, fresh_path, verify_expected=False
    )
    assert registry["population_status"].tolist() == [
        "CONSUMED",
        "CONSUMED",
        "UNSEEN",
    ]
    assert registry["source_key"].is_unique
    assert registry.loc[1, "matched_fresh_by_siret"]
    assert audit["integrity"]["closed_fresh_source_overlap"] == 0
    assert audit["integrity"]["unseen_rows"] == 1


def test_write_build_is_immutable_and_hashed(tmp_path: Path) -> None:
    crm_path, closed_path, fresh_path = _write_inputs(tmp_path)
    registry, audit = build_registry(
        crm_path, closed_path, fresh_path, verify_expected=False
    )
    script = tmp_path / "builder.py"
    script.write_text("fixture\n", encoding="utf-8")
    output = write_build(registry, audit, tmp_path / "out", script)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["verdict"] == "PASS_REGISTRY"
    assert manifest["outputs"]["source_registry.parquet"]["rows"] == 3
    assert (
        manifest["outputs"]["unseen.parquet"]["sha256"]
        == hashlib.sha256((output / "unseen.parquet").read_bytes()).hexdigest()
    )
    try:
        write_build(registry, audit, tmp_path / "out", script)
    except FileExistsError:
        pass
    else:
        raise AssertionError("immutable output must refuse overwrite")

