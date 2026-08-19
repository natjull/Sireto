#!/usr/bin/env python3
"""Evaluate a raw official retrieval union without training or opening test."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate(path: Path) -> dict[str, object]:
    path = path.resolve()
    connection = duckdb.connect()
    description = {
        str(row[0])
        for row in connection.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]
        ).fetchall()
    }
    required = {
        "query_id",
        "fold",
        "ground_truth_siret",
        "identifiable_exact",
        "siret",
        "union_rank",
        "retrieval_latency_ms",
        "candidate_present",
    }
    missing = sorted(required - description)
    if missing:
        raise ValueError(f"retrieval union is missing columns: {missing}")
    folds = [
        int(row[0])
        for row in connection.execute(
            "SELECT DISTINCT fold FROM read_parquet(?) ORDER BY fold", [str(path)]
        ).fetchall()
    ]
    if folds != [0]:
        raise ValueError(f"dev oracle evaluation requires fold 0 only, observed {folds}")
    operational = (
        "coalesce(list_contains(acceptable_sirets_operational, siret), false)"
        if "acceptable_sirets_operational" in description
        else "false"
    )
    query = f"""
        WITH per_query AS (
          SELECT query_id,
                 bool_or(coalesce(identifiable_exact, false)) identifiable_exact,
                 bool_or(coalesce(candidate_present, false)
                         AND siret=ground_truth_siret) exact_oracle_hit,
                 bool_or(coalesce(candidate_present, false)
                         AND siret=ground_truth_siret AND union_rank<=100) raw_top100_hit,
                 bool_or(coalesce(candidate_present, false)
                         AND {operational}) operational_oracle_hit,
                 max(retrieval_latency_ms) retrieval_latency_ms,
                 max(raw_union_size) raw_union_size,
                 max(union_rank) emitted_candidate_count
          FROM read_parquet(?) GROUP BY query_id
        )
        SELECT count(*) query_count,
               count(*) FILTER (WHERE identifiable_exact) identifiable_count,
               count(*) FILTER (WHERE identifiable_exact AND exact_oracle_hit) exact_oracle_hits,
               count(*) FILTER (WHERE identifiable_exact AND raw_top100_hit) raw_top100_hits,
               count(*) FILTER (WHERE identifiable_exact AND operational_oracle_hit)
                   operational_oracle_hits,
               max(emitted_candidate_count) max_emitted_candidates,
               max(raw_union_size) max_raw_union_size,
               quantile_cont(retrieval_latency_ms, 0.5) latency_p50_ms,
               quantile_cont(retrieval_latency_ms, 0.95) latency_p95_ms,
               quantile_cont(retrieval_latency_ms, 0.99) latency_p99_ms
        FROM per_query
    """
    row = connection.execute(query, [str(path)]).fetchone()
    columns = [item[0] for item in connection.description]
    values = dict(zip(columns, row, strict=True))
    connection.close()
    identifiable = int(values["identifiable_count"])
    total = int(values["query_count"])
    exact_hits = int(values["exact_oracle_hits"])
    raw_hits = int(values["raw_top100_hits"])
    operational_hits = int(values["operational_oracle_hits"])
    return {
        "schema_version": "sireto-retrieval-raw-union-oracle-v1",
        "input": str(path),
        "input_sha256": _sha256(path),
        "opened_folds": folds,
        "test_opened": False,
        "positive_injection": False,
        "query_count": total,
        "identifiable_count": identifiable,
        "identifiable_coverage": identifiable / total if total else None,
        "exact_oracle": {
            "hits": exact_hits,
            "total": identifiable,
            "recall": exact_hits / identifiable if identifiable else None,
            "minimum": 0.993,
            "passed": bool(identifiable and exact_hits / identifiable >= 0.993),
        },
        "raw_union_rank_top100": {
            "hits": raw_hits,
            "total": identifiable,
            "recall": raw_hits / identifiable if identifiable else None,
            "is_trained_admission": False,
        },
        "operational_oracle": {
            "hits": operational_hits,
            "total": identifiable,
            "recall": operational_hits / identifiable if identifiable else None,
        },
        "maximum_emitted_candidates": int(values["max_emitted_candidates"] or 0),
        "maximum_raw_union_size": int(values["max_raw_union_size"] or 0),
        "retrieval_latency_ms": {
            "p50": float(values["latency_p50_ms"] or 0.0),
            "p95": float(values["latency_p95_ms"] or 0.0),
            "p99": float(values["latency_p99_ms"] or 0.0),
        },
        "verdict": (
            "CONTINUE_TO_LAMBDAMART"
            if identifiable and exact_hits / identifiable >= 0.993
            else "PIVOT_ORACLE_BELOW_99_3"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = evaluate(args.candidates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
