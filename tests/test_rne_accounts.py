from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import pyarrow.parquet as pq

from src.xgb_matcher.rne_accounts import build_rne_account_deposits


def test_rne_account_deposits_are_siren_level_and_privacy_minimal(tmp_path: Path):
    archive = tmp_path / "accounts.zip"
    record = {
        "id": "filing-1",
        "siren": "123456789",
        "denomination": "SOCIETE EXEMPLE",
        "dateDepot": "2026-03-20",
        "dateCloture": "2025-12-31",
        "confidentiality": "Public",
        "deleted": False,
        "bilanSaisi": {
            "bilan": {
                "identite": {"codeTypeBilan": "C", "codeDevise": "EUR", "dureeExerciceN": "12"},
                "detail": {"pages": [{"liasses": [{"code": "FL", "m1": "999999"}]}]},
            }
        },
    }
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        output.writestr("stock_000001.json", json.dumps([record]))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest = {
        "build_id": "snapshot-fixture",
        "payload": [{"name": archive.name, "size_bytes": archive.stat().st_size, "sha256": digest}],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = build_rne_account_deposits(
        manifest_path=manifest_path,
        payload_name=archive.name,
        output_root=tmp_path / "output",
        batch_size=1,
    )
    rows = pq.read_table(result.deposits_path).to_pylist()
    assert rows[0]["siren"] == "123456789"
    assert rows[0]["is_public"] is True
    assert rows[0]["structured_accounts_present"] is True
    assert "siret" not in rows[0]
    assert "pages" not in rows[0]
    assert "turnover" not in rows[0]
    saved = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert saved["policy"]["model_use_enabled"] is False


def test_rne_account_deposits_quarantine_invalid_json_member(tmp_path: Path):
    archive = tmp_path / "accounts.zip"
    valid = {"id": "f1", "siren": "123456789", "confidentiality": "Confidentiel"}
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        output.writestr("stock_000001.json", json.dumps([valid]))
        output.writestr("stock_000002.json", '[{"siren":"987654321"}')
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest = {"build_id": "s", "payload": [{"name": archive.name, "sha256": digest}]}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = build_rne_account_deposits(
        manifest_path=manifest_path,
        payload_name=archive.name,
        output_root=tmp_path / "output",
        batch_size=1,
    )
    saved = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert saved["count"] == 1
    assert saved["invalid_member_count"] == 1
    assert saved["invalid_members"][0]["archive_member"] == "stock_000002.json"
