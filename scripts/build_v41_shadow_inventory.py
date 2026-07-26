#!/usr/bin/env python3
"""Build the V4.1 shadow inventory and frozen pre-prediction panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xgb_matcher.v41_shadow import (  # noqa: E402
    DenylistSpec,
    build_pre_prediction_panel,
    build_shadow_inventory,
    enrich_inventory_siret_state,
    inventory_summary,
)
from src.xgb_matcher.v9_dataset import file_sha256, read_table  # noqa: E402


def _read_relevant_sirene(
    snapshot: Path,
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    """Read only establishments whose SIREN occurs in the CRM inventory."""

    input_sirens = pd.DataFrame(
        {
            "siren": sorted(
                {
                    str(value)
                    for value in inventory["input_siren"].dropna()
                    if str(value)
                }
            )
        }
    )
    connection = duckdb.connect()
    try:
        connection.register("input_sirens", input_sirens)
        return connection.execute(
            """
            SELECT
                CAST(est.siret AS VARCHAR) AS siret,
                CAST(est.siren AS VARCHAR) AS siren,
                CAST(est.etatAdministratifEtablissement AS VARCHAR)
                    AS etatAdministratifEtablissement
            FROM read_parquet(?) AS est
            INNER JOIN input_sirens AS wanted
                ON CAST(est.siren AS VARCHAR) = wanted.siren
            """,
            [str(snapshot)],
        ).fetchdf()
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--old-test", type=Path, required=True)
    parser.add_argument("--old-test-manifest", type=Path)
    parser.add_argument("--fresh-holdout", type=Path, required=True)
    parser.add_argument("--fresh-holdout-manifest", type=Path)
    parser.add_argument("--sirene-snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--panel-seed", type=int, default=42)
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--expected-eligible-count", type=int)
    args = parser.parse_args()

    if (args.expected_source_sha256 is None) != (
        args.expected_eligible_count is None
    ):
        raise ValueError(
            "--expected-source-sha256 and --expected-eligible-count "
            "must be supplied together"
        )
    observed_source_hash = file_sha256(args.source)
    if (
        args.expected_source_sha256 is not None
        and observed_source_hash != args.expected_source_sha256
    ):
        raise ValueError("CRM source hash differs from the expected source")

    specs = [
        DenylistSpec(
            "OLD_TEST",
            args.old_test,
            split_column="split",
            split_value="test",
            manifest_path=args.old_test_manifest,
        ),
        DenylistSpec(
            "FRESH_HOLDOUT",
            args.fresh_holdout,
            manifest_path=args.fresh_holdout_manifest,
        ),
    ]
    source = read_table(args.source)
    inventory = build_shadow_inventory(
        source,
        denylists={spec.name: spec.load_ids() for spec in specs},
    )
    registry = _read_relevant_sirene(args.sirene_snapshot, inventory)
    inventory = enrich_inventory_siret_state(inventory, registry)
    panel = build_pre_prediction_panel(inventory, seed=args.panel_seed)
    summary = inventory_summary(inventory)
    if (
        args.expected_eligible_count is not None
        and summary["eligible_row_count"] != args.expected_eligible_count
    ):
        raise ValueError(
            "Eligible count failed source-integrity assertion: "
            f"{summary['eligible_row_count']} != {args.expected_eligible_count}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    inventory.to_parquet(args.output_dir / "inventory.parquet", index=False)
    panel.to_parquet(args.output_dir / "panel500.parquet", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "sireto-shadow-v4.1-inventory-1",
        "source": {
            "path": str(args.source),
            "sha256": observed_source_hash,
        },
        "denylists": [spec.provenance() for spec in specs],
        "sirene_snapshot": {
            "path": str(args.sirene_snapshot),
            "sha256": file_sha256(args.sirene_snapshot),
        },
        "panel_seed": args.panel_seed,
        "expected_eligible_count_asserted": args.expected_eligible_count,
        "outputs": {
            name: file_sha256(args.output_dir / name)
            for name in ("inventory.parquet", "panel500.parquet", "summary.json")
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
