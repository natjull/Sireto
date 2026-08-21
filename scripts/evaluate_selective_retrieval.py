#!/usr/bin/env python3
"""Evaluate a completed retrieval run on an independent selective scope."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def metric(values: pd.Series) -> dict[str, float | int | None]:
    total = int(len(values))
    successes = int(values.fillna(False).astype(bool).sum())
    return {
        "successes": successes,
        "total": total,
        "rate": successes / total if total else None,
    }


def evaluate(args: argparse.Namespace) -> dict:
    outcomes_path = Path(args.outcomes)
    qualification_path = Path(args.qualification)
    qualification_manifest_path = qualification_path.with_suffix(
        qualification_path.suffix + ".manifest.json"
    )
    manifest = json.loads(qualification_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("retrieval_inputs_used") is not False or manifest.get(
        "retrieval_hits_ranks_scores_used"
    ) is not False:
        raise ValueError("Qualification is not retrieval-independent")
    if manifest.get("fold1_closed") is not True:
        raise ValueError("Qualification opened fold1")
    outcomes = pd.read_parquet(outcomes_path)
    qualification = pd.read_parquet(qualification_path)
    development = qualification[qualification["fold"].eq(0)][
        ["query_id", "identifiable_exact_v4", "qualification_reason_v4"]
    ]
    joined = outcomes.drop(columns=["identifiable_exact"]).merge(
        development, on="query_id", how="left", validate="one_to_one"
    )
    if joined["identifiable_exact_v4"].isna().any():
        raise ValueError("Qualification is incomplete for retrieval outcomes")
    scope = joined["identifiable_exact_v4"].astype(bool)
    scoped = joined[scope]
    coverage = metric(scope)
    exact = metric(scoped["exact_hit_at_100"])
    oracle = metric(scoped["exact_oracle_hit"])
    operational = metric(scoped["operational_hit_at_100"])
    max_candidates = int(joined["selected_candidate_count"].max())
    gates = {
        "identifiable_coverage": coverage["rate"] >= 0.8,
        "exact_recall_at_100": exact["rate"] >= 0.99,
        "candidate_ceiling": max_candidates <= 100,
    }
    result = {
        "schema_version": "sireto-selective-retrieval-evaluation-v1",
        "policy_id": manifest["policy_id"],
        "fold": 0,
        "fold1_closed": True,
        "query_count": int(len(joined)),
        "metrics": {
            "identifiable_coverage": coverage,
            "exact_oracle_recall": oracle,
            "exact_recall_at_100": exact,
            "operational_recall_at_100": operational,
        },
        "maximum_candidates": max_candidates,
        "gates": gates,
        "recall_verdict": "GO" if all(gates.values()) else "PIVOT",
        "segments": {
            state: {
                "count": int(len(group)),
                "exact_recall_at_100": metric(group["exact_hit_at_100"]),
                "operational_recall_at_100": metric(
                    group["operational_hit_at_100"]
                ),
            }
            for state, group in scoped.groupby("ground_truth_state", sort=True)
        },
        "qualification_reasons": {
            str(key): int(value)
            for key, value in scoped["qualification_reason_v4"]
            .value_counts()
            .sort_index()
            .items()
        },
        "qualification_manifest": str(qualification_manifest_path.resolve()),
        "outcomes": str(outcomes_path.resolve()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcomes", required=True)
    parser.add_argument("--qualification", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(evaluate(parser.parse_args()), ensure_ascii=False))


if __name__ == "__main__":
    main()
