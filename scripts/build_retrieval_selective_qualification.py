#!/usr/bin/env python3
"""Build retrieval-independent V4 qualification from CRM and dossier Parquet."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xgb_matcher.retrieval_selective_qualification import (  # noqa: E402
    QualificationPolicy,
    qualify_direct_evidence,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sql(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def build(args: argparse.Namespace) -> Path:
    queries = Path(args.queries)
    dossier = Path(args.dossier_siret_documents)
    policy_path = Path(args.policy)
    for path in (queries, dossier, policy_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    policy = QualificationPolicy.load(policy_path)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    name_fields = policy.payload["allowed_official_name_fields"]
    address_fields = policy.payload["allowed_official_address_fields"]
    projections = ",".join(
        [f"d.{field}" for field in name_fields + address_fields]
    )
    frame = connection.execute(
        f"""
        SELECT q.query_id, q.oof_fold AS fold, q.crm_name, q.crm_address,
               regexp_extract(q.crm_address, '^[ ]*([0-9]+)', 1) AS crm_number,
               q.ground_truth_siret, d.number AS candidate_number,
               {projections}
        FROM read_parquet('{_sql(queries)}') q
        LEFT JOIN read_parquet('{_sql(dossier)}') d
          ON d.document_id = q.ground_truth_siret
        ORDER BY q.query_id
        """
    ).fetchdf()
    allowed = set(policy.payload["train_folds"]) | {policy.payload["development_fold"]}
    observed = set(int(value) for value in frame["fold"].unique())
    if not observed.issubset(allowed) or 1 in observed:
        raise ValueError(f"Qualification input opens forbidden folds: {observed}")
    records = []
    for row in frame.to_dict(orient="records"):
        result = qualify_direct_evidence(row, policy)
        records.append(
            {
                "query_id": str(row["query_id"]),
                "fold": int(row["fold"]),
                "ground_truth_siret": str(row["ground_truth_siret"] or ""),
                **result,
            }
        )
    qualified = pd.DataFrame(records).sort_values("query_id", kind="stable")
    if qualified["query_id"].duplicated().any():
        raise ValueError("Qualification query_id is not unique")
    train = qualified[qualified["fold"].isin(policy.payload["train_folds"])]
    train_coverage = float(train["identifiable_exact_v4"].mean())
    if train_coverage < float(policy.payload["minimum_train_coverage"]):
        raise ValueError(f"Training coverage below contract: {train_coverage}")
    qualified.to_parquet(output, index=False)
    manifest = {
        "schema_version": "sireto-retrieval-selective-qualification-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy_id": policy.payload["policy_id"],
        "policy_path": str(policy_path.resolve()),
        "policy_sha256": sha256_file(policy_path),
        "queries_path": str(queries.resolve()),
        "queries_sha256": sha256_file(queries),
        "dossier_siret_documents_path": str(dossier.resolve()),
        "dossier_siret_documents_sha256": sha256_file(dossier),
        "qualification_path": str(output.resolve()),
        "qualification_sha256": sha256_file(output),
        "query_count": int(len(qualified)),
        "opened_folds": sorted(observed),
        "fold1_closed": True,
        "retrieval_inputs_used": False,
        "retrieval_hits_ranks_scores_used": False,
        "train_coverage": train_coverage,
        "counts_by_fold": {
            str(fold): {
                "total": int(len(group)),
                "identifiable": int(group["identifiable_exact_v4"].sum()),
            }
            for fold, group in qualified.groupby("fold", sort=True)
        },
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", required=True)
    parser.add_argument("--dossier-siret-documents", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output", required=True)
    print(build(parser.parse_args()))


if __name__ == "__main__":
    main()
