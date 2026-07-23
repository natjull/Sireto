#!/usr/bin/env python3
"""Sample a stratified 500-query template for human open-set adjudication."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xgb_matcher.v9_dataset import read_table


STRATA = [
    "no_candidate",
    "homonym",
    "multi_site",
    "megacity",
    "closed",
    "historical_review",
    "positive_control",
    "other",
]


def assign_stratum(row: pd.Series) -> str:
    if float(row.get("pool_size") or row.get("candidate_count") or 0) == 0:
        return "no_candidate"
    if float(row.get("same_name_count") or 0) > 1:
        return "homonym"
    if float(row.get("same_siren_count") or row.get("siren_candidate_count") or 0) > 1:
        return "multi_site"
    if float(row.get("partition_size") or 0) >= 100_000:
        return "megacity"
    if str(row.get("candidate_state") or "").upper() in {"FERME", "F"}:
        return "closed"
    if str(row.get("routing_status") or "").upper() == "REVIEW":
        return "historical_review"
    if row.get("ground_truth_siret") or row.get("siret_gt"):
        return "positive_control"
    return "other"


def sample_template(source: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    rows = source.copy()
    if "query_id" not in rows.columns:
        if "crm_id" not in rows.columns:
            raise ValueError("Source requires query_id or crm_id")
        rows = rows.rename(columns={"crm_id": "query_id"})
    rows = rows.drop_duplicates("query_id").copy()
    rows["sampling_stratum"] = rows.apply(assign_stratum, axis=1)
    quota = max(1, count // len(STRATA))
    sampled_parts = []
    selected: set[str] = set()
    for stratum in STRATA:
        pool = rows[rows["sampling_stratum"].eq(stratum)]
        chosen = pool.sample(n=min(quota, len(pool)), random_state=seed)
        sampled_parts.append(chosen)
        selected.update(chosen["query_id"].astype(str))
    sampled = pd.concat(sampled_parts, ignore_index=True) if sampled_parts else rows.iloc[0:0]
    remaining_count = count - len(sampled)
    if remaining_count > 0:
        remainder = rows[~rows["query_id"].astype(str).isin(selected)]
        sampled = pd.concat(
            [
                sampled,
                remainder.sample(
                    n=min(remaining_count, len(remainder)),
                    random_state=seed + 1,
                ),
            ],
            ignore_index=True,
        )
    sampled = sampled.head(count).copy()
    sampled["label_kind"] = "UNRESOLVED"
    sampled["ground_truth_siret"] = None
    sampled["validator"] = ""
    sampled["validated_at"] = ""
    sampled["evidence_refs"] = ""
    sampled["sirene_snapshot_id"] = ""
    sampled["reference_date"] = ""
    sampled["llm_preannotation"] = ""
    sampled["llm_evidence_summary"] = ""
    return sampled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    template = sample_template(read_table(args.source), args.count, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(args.output, index=False)
    print(template["sampling_stratum"].value_counts().to_string())
    if len(template) < args.count:
        print(f"WARNING: only {len(template)} unique queries available")


if __name__ == "__main__":
    main()
