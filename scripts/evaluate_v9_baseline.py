#!/usr/bin/env python3
"""Evaluate a legacy V7 top-k export under strict V9 SIRET/open-set metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xgb_matcher.contracts import GroundTruthKind
from src.xgb_matcher.selective import certified_precision_lower, risk_coverage_curve
from src.xgb_matcher.v9_dataset import normalize_siret, read_table


def evaluate_baseline(topk: pd.DataFrame, labels: pd.DataFrame) -> dict:
    rows = topk.copy()
    if "query_id" not in rows.columns:
        rows = rows.rename(columns={"crm_id": "query_id"})
    if "candidate_siret" not in rows.columns:
        rows = rows.rename(columns={"siret_candidate": "candidate_siret"})
    rows["candidate_siret"] = rows["candidate_siret"].map(normalize_siret)
    rows["candidate_siren"] = rows["candidate_siret"].str[:9]
    if "rank" in rows.columns:
        rows = rows.sort_values(["query_id", "rank"])
    else:
        rows = rows.sort_values(["query_id", "score"], ascending=[True, False])
    top1 = rows.groupby("query_id", as_index=False).first()
    evaluated = labels.merge(top1, on="query_id", how="left")
    evaluated["is_exact_siret_correct"] = (
        evaluated["label_kind"].eq(GroundTruthKind.MATCH_EXACT.value)
        & evaluated["candidate_siret"].eq(evaluated["ground_truth_siret"])
    ).astype(int)

    exact = evaluated[
        evaluated["label_kind"].eq(GroundTruthKind.MATCH_EXACT.value)
    ].copy()
    pool_hits = labels[["query_id", "ground_truth_siret"]].merge(
        rows[["query_id", "candidate_siret"]],
        on="query_id",
        how="left",
    )
    pool_hits["hit"] = pool_hits["candidate_siret"].eq(
        pool_hits["ground_truth_siret"]
    )
    recall = pool_hits.groupby("query_id")["hit"].max().reindex(exact["query_id"])

    auto_mask = (
        evaluated.get("routing_status", pd.Series("", index=evaluated.index))
        .fillna("")
        .astype(str)
        .str.startswith("AUTO")
    )
    auto_count = int(auto_mask.sum())
    auto_correct = int(evaluated.loc[auto_mask, "is_exact_siret_correct"].sum())
    report = {
        "query_count": int(len(evaluated)),
        "exact_match_query_count": int(len(exact)),
        "candidate_recall": float(recall.fillna(False).mean()) if len(exact) else 0.0,
        "hit_at_1_siret": (
            float(exact["is_exact_siret_correct"].mean()) if len(exact) else 0.0
        ),
        "hit_at_1_siren": (
            float(
                exact["candidate_siren"]
                .eq(exact["ground_truth_siren"])
                .mean()
            )
            if len(exact)
            else 0.0
        ),
        "auto_count": auto_count,
        "auto_coverage": auto_count / len(evaluated) if len(evaluated) else 0.0,
        "auto_exact_siret_precision": (
            auto_correct / auto_count if auto_count else 0.0
        ),
        "auto_error_count": auto_count - auto_correct,
        "auto_precision_lower_99": (
            certified_precision_lower(auto_count - auto_correct, auto_count)
            if auto_count
            else 0.0
        ),
    }
    if "routing_confidence" in evaluated.columns:
        report["risk_coverage_curve"] = risk_coverage_curve(
            pd.to_numeric(
                evaluated["routing_confidence"],
                errors="coerce",
            ).fillna(0.0),
            evaluated["is_exact_siret_correct"],
        ).to_dict("records")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topk", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_baseline(read_table(args.topk), read_table(args.labels))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
