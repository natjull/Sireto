from __future__ import annotations

from pathlib import Path
import json
from unittest.mock import patch
import sys

import pandas as pd

# Ensure project modules are importable when running tests from repo root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_pipe_v6 import main
from pipe_v6.pipeline import RESULT_COLUMNS
from pipe_v6.exporter import EXPORT_COLUMNS


def _write_sample_crm(path: Path) -> None:
    content = (
        "N° Service;Client final;Adresse;Commune;Code Postal;Code INSEE\n"
        "ID001;COMPANY A;12 RUE TEST;VILLEURBANNE;69100;69266\n"
        "ID002;COMPANY B;5 AV DEMO;LYON;69003;69123\n"
        "ID003;COMPANY C;8 BD EXAMPLE;LYON;69007;69387\n"
        "ID004;COMPANY D;15 RUE AUTRE;VILLEURBANNE;69100;69266\n"
        "ID005;COMPANY E;20 AV FINAL;LYON;69003;69123\n"
    )
    path.write_text(content, encoding="utf-8")


def _mock_results() -> pd.DataFrame:
    rows = []
    for i, status in enumerate(["MATCH", "REVIEW", "NO_MATCH", "MATCH", "NO_MATCH"], start=1):
        rows.append(
            {
                "crm_id": f"ID00{i}",
                "status": status,
                "chosen_siret": f"1234567890{i:03d}" if status != "NO_MATCH" else None,
                "confidence": 0.9 - i * 0.05,
                "reason": "",
                "crm_category": "PRIVE" if i % 2 else "PUBLIC",
                "normalized_name": f"COMPANY {chr(64+i)}",
                "normalized_address": "ADDR",
                "candidate_count_total": i,
                "candidate_count_used": max(0, i - 1),
                "sources": ["RNE", "DATAGOUV"] if status == "MATCH" else [],
            }
        )
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def test_cli_end_to_end_with_mocked_pipeline(tmp_path, monkeypatch) -> None:
    crm_csv = tmp_path / "crm.csv"
    out_csv = tmp_path / "out.csv"
    stats_json = tmp_path / "stats.json"
    _write_sample_crm(crm_csv)

    mock_df = _mock_results()

    # Prepare argv for the CLI
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_pipe_v6.py",
            "--crm-path",
            str(crm_csv),
            "--output-path",
            str(out_csv),
            "--stats-json",
            str(stats_json),
        ],
    )

    with patch("scripts.run_pipe_v6.run_pipeline", return_value=mock_df) as mocked:
        exit_code = main()

    assert exit_code == 0
    mocked.assert_called_once()

    assert out_csv.exists()
    df_export = pd.read_csv(out_csv, sep=";", encoding="utf-8")
    assert len(df_export) == 5
    assert list(df_export.columns) == EXPORT_COLUMNS

    assert stats_json.exists()
    stats = json.loads(stats_json.read_text(encoding="utf-8"))
    assert stats.get("total_rows") == 5
    assert "status" in stats and "candidates" in stats
