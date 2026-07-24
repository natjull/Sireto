#!/usr/bin/env python3
"""Audit exact-SIRET label ambiguity among establishments of the same SIREN."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import sys
from typing import Any

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_v9_retrieval_experiment import _git_commit  # noqa: E402
from src.xgb_matcher.blocking import (  # noqa: E402
    address_hash,
    candidate_address_hash,
)
from src.xgb_matcher.features import preprocess_crm_row  # noqa: E402
from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402


SCHEMA_VERSION = "sireto-benchmark-site-label-audit-1"


def classify_site_label(
    *,
    crm_address_usable: bool,
    gt_state: str,
    date_reference_present: bool,
    exact_sibling_count: int,
    active_exact_sibling_count: int,
) -> str:
    """Classify an exact-SIRET label without changing it."""
    if not crm_address_usable:
        return "NO_USABLE_CRM_ADDRESS"
    if date_reference_present:
        return "HISTORICAL_REFERENCE_DATE_PRESENT"
    if active_exact_sibling_count > 1:
        return "MULTIPLE_ACTIVE_EXACT_SIBLINGS"
    if active_exact_sibling_count == 1 and gt_state == "F":
        return "CLOSED_GT_UNIQUE_ACTIVE_EXACT_SIBLING"
    if active_exact_sibling_count == 1:
        return "ACTIVE_GT_HAS_ACTIVE_EXACT_SIBLING"
    if exact_sibling_count:
        return "INACTIVE_EXACT_SIBLING_ONLY"
    return "NO_EXACT_SIBLING"


def _candidate_hash(row: pd.Series) -> str | None:
    def clean(value: Any) -> str | None:
        if value is None or pd.isna(value):
            return None
        text = str(value).strip()
        return text or None

    return candidate_address_hash(
        {
            "numeroVoie": clean(row.get("numeroVoie")),
            "typeVoie": clean(row.get("typeVoie")),
            "libelleVoie": clean(row.get("libelleVoie")),
        }
    )


def audit_rows(joined: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for query_id, group in joined.groupby("query_id", sort=False):
        query = group.iloc[0]
        crm = preprocess_crm_row(
            {
                "crm_name": query.get("crm_name") or "",
                "crm_address": query.get("crm_address") or "",
                "crm_city": query.get("crm_city") or "",
                "crm_city_addr": query.get("crm_city") or "",
                "postcode": query.get("postcode") or "",
                "insee": query.get("insee") or "",
            }
        )
        crm_hash = address_hash(
            crm.get("crm_street_num"),
            crm.get("crm_street_name"),
        )
        ground_truth = str(query["ground_truth_siret"])
        exact_siblings: list[str] = []
        active_exact_siblings: list[str] = []
        for _, candidate in group.iterrows():
            siret = str(candidate["siret"])
            if siret == ground_truth or not crm_hash:
                continue
            if _candidate_hash(candidate) != crm_hash:
                continue
            exact_siblings.append(siret)
            if str(candidate.get("candidate_state") or "") == "A":
                active_exact_siblings.append(siret)

        date_reference = query.get("date_reference")
        date_reference_present = bool(
            date_reference is not None and not pd.isna(date_reference)
        )
        classification = classify_site_label(
            crm_address_usable=bool(crm_hash),
            gt_state=str(query.get("ground_truth_state") or ""),
            date_reference_present=date_reference_present,
            exact_sibling_count=len(exact_siblings),
            active_exact_sibling_count=len(active_exact_siblings),
        )
        records.append(
            {
                "query_id": str(query_id),
                "split": str(query["split"]),
                "crm_name": query.get("crm_name"),
                "crm_address": query.get("crm_address"),
                "ground_truth_siret": ground_truth,
                "ground_truth_siren": str(query["ground_truth_siren"]),
                "ground_truth_state": query.get("ground_truth_state"),
                "date_reference": date_reference,
                "crm_address_hash": crm_hash,
                "exact_sibling_count": len(exact_siblings),
                "active_exact_sibling_count": len(active_exact_siblings),
                "exact_sibling_sirets_json": json.dumps(
                    exact_siblings,
                    separators=(",", ":"),
                ),
                "active_exact_sibling_sirets_json": json.dumps(
                    active_exact_siblings,
                    separators=(",", ":"),
                ),
                "site_label_class": classification,
            }
        )
    return pd.DataFrame(records)


def summarize(raw: pd.DataFrame) -> dict[str, Any]:
    classes = raw["site_label_class"].value_counts().sort_index()
    return {
        "query_count": int(len(raw)),
        "classes": {str(key): int(value) for key, value in classes.items()},
        "queries_with_any_exact_sibling": int(
            raw["exact_sibling_count"].gt(0).sum()
        ),
        "queries_with_active_exact_sibling": int(
            raw["active_exact_sibling_count"].gt(0).sum()
        ),
        "closed_gt_with_active_exact_sibling": int(
            (
                raw["ground_truth_state"].eq("F")
                & raw["active_exact_sibling_count"].gt(0)
            ).sum()
        ),
        "queries_with_multiple_active_exact_siblings": int(
            raw["active_exact_sibling_count"].gt(1).sum()
        ),
        "warning": (
            "An exact-address sibling is evidence of non-uniqueness or stale "
            "site identity, not an automatic replacement label."
        ),
    }


def _markdown(summary: dict[str, Any], manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Audit de cohérence des labels SIRET exacts",
            "",
            f"- Benchmark : `{manifest['benchmark_build_id']}`",
            f"- Split : `{manifest['split']}`",
            f"- Requêtes : {summary['query_count']}",
            "",
            "## Résultats",
            "",
            f"- autre SIRET du même SIREN à l'adresse exacte : "
            f"{summary['queries_with_any_exact_sibling']} ;",
            f"- autre SIRET actif du même SIREN à l'adresse exacte : "
            f"{summary['queries_with_active_exact_sibling']} ;",
            f"- label fermé avec sibling actif à l'adresse exacte : "
            f"{summary['closed_gt_with_active_exact_sibling']} ;",
            f"- plusieurs siblings actifs à l'adresse exacte : "
            f"{summary['queries_with_multiple_active_exact_siblings']}.",
            "",
            "Un sibling à la même adresse ne constitue pas automatiquement une "
            "nouvelle vérité. Il prouve en revanche que le SIRET exact peut être "
            "non identifiable avec les seuls champs CRM disponibles.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument("--establishment-snapshot", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "dev"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(
            f"Immutable output already exists: {args.output_dir}"
        )

    benchmark_manifest = json.loads(
        args.benchmark_manifest.read_text(encoding="utf-8")
    )
    expected_benchmark_hash = benchmark_manifest.get(
        "output_sha256",
        {},
    ).get(args.benchmark.name)
    if file_sha256(args.benchmark) != expected_benchmark_hash:
        raise ValueError("Benchmark hash does not match frozen manifest")
    if file_sha256(args.establishment_snapshot) != benchmark_manifest.get(
        "establishment_snapshot_sha256"
    ):
        raise ValueError("Establishment snapshot hash mismatch")

    connection = duckdb.connect()
    joined = connection.execute(
        """
        SELECT
            b.query_id,
            b.split,
            b.crm_name,
            b.crm_address,
            b.crm_city,
            b.postcode,
            b.insee,
            b.ground_truth_siret,
            b.ground_truth_siren,
            b.ground_truth_state,
            b.date_reference,
            CAST(e.siret AS VARCHAR) AS siret,
            e.etatAdministratifEtablissement AS candidate_state,
            CAST(e.numeroVoieEtablissement AS VARCHAR) AS numeroVoie,
            e.typeVoieEtablissement AS typeVoie,
            e.libelleVoieEtablissement AS libelleVoie
        FROM read_parquet(?) b
        JOIN read_parquet(?) e
          ON b.ground_truth_siren = CAST(e.siren AS VARCHAR)
        WHERE b.split = ?
        ORDER BY CAST(b.query_id AS BIGINT), CAST(e.siret AS VARCHAR)
        """,
        [
            args.benchmark.resolve().as_posix(),
            args.establishment_snapshot.resolve().as_posix(),
            args.split,
        ],
    ).df()
    raw = audit_rows(joined)
    summary = summarize(raw)

    args.output_dir.mkdir(parents=True)
    raw_path = args.output_dir / "raw_results.parquet"
    summary_path = args.output_dir / "summary.json"
    report_path = args.output_dir / "report.md"
    raw.to_parquet(raw_path, index=False)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "command": [sys.executable, *sys.argv],
        "benchmark_build_id": benchmark_manifest["build_id"],
        "split": args.split,
        "inputs": {
            "benchmark_sha256": file_sha256(args.benchmark),
            "benchmark_manifest_sha256": file_sha256(
                args.benchmark_manifest
            ),
            "establishment_snapshot_sha256": file_sha256(
                args.establishment_snapshot
            ),
        },
    }
    report_path.write_text(_markdown(summary, manifest), encoding="utf-8")
    manifest["outputs"] = {
        "raw_results.parquet": file_sha256(raw_path),
        "summary.json": file_sha256(summary_path),
        "report.md": file_sha256(report_path),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
