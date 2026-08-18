#!/usr/bin/env python3
"""Run one bounded hierarchical Recall@100 and latency evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.xgb_matcher.hierarchical_retrieval import (  # noqa: E402
    load_production_retriever,
    normalize_code,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def _first(row: dict[str, Any], names: list[str]) -> Any:
    for name in names:
        value = row.get(name)
        if value is not None and not (isinstance(value, float) and np.isnan(value)):
            return value
    return None


def _operational_values(value: Any, exact: str) -> set[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {exact}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = [part.strip() for part in value.split(",") if part.strip()]
    if not isinstance(value, (list, tuple, set)):
        value = [value]
    return {exact, *(normalize_code(item, 14) for item in value if item)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "config" / "retrieval_hierarchical_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = _load(args.input)
    retriever = load_production_retriever(
        config_path=args.config, index_path=args.index
    )
    exact_hits = 0
    operational_hits = 0
    evaluated = 0
    candidate_counts: list[int] = []
    latencies_ms: list[float] = []
    for row_number, row in enumerate(frame.to_dict(orient="records"), start=1):
        exact = normalize_code(
            _first(row, ["gt_siret", "ground_truth_siret", "siret_gt"]), 14
        )
        if not exact:
            continue
        started = time.perf_counter()
        candidates = retriever.retrieve(row)
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
        candidate_sirets = {candidate.siret for candidate in candidates}
        acceptable = _operational_values(
            _first(
                row,
                [
                    "acceptable_sirets_operational",
                    "acceptable_sirets_operational_json",
                ],
            ),
            exact,
        )
        exact_hits += int(exact in candidate_sirets)
        operational_hits += int(bool(candidate_sirets & acceptable))
        candidate_counts.append(len(candidates))
        evaluated += 1
        if row_number % 100 == 0:
            elapsed_s = sum(latencies_ms) / 1000.0
            print(
                f"progress={row_number}/{len(frame)} exact_hits={exact_hits} "
                f"query_time_s={elapsed_s:.1f}",
                flush=True,
            )

    if not evaluated:
        raise RuntimeError("input contains no usable exact GT SIRET")
    if max(candidate_counts) > 100:
        raise RuntimeError("absolute 100-candidate retrieval contract violated")
    latency = np.asarray(latencies_ms, dtype=np.float64)
    index_manifest = args.index / "manifest.json"
    payload = {
        "schema_version": "sireto-hierarchical-evaluation-v1",
        "input": {"path": str(args.input.resolve()), "sha256": _sha256(args.input)},
        "config": {"path": str(args.config.resolve()), "sha256": _sha256(args.config)},
        "index": {
            "path": str(args.index.resolve()),
            "manifest_sha256": _sha256(index_manifest),
        },
        "evaluated_rows": evaluated,
        "exact": {
            "hits_at_100": exact_hits,
            "recall_at_100": exact_hits / evaluated,
        },
        "operational_secondary": {
            "hits_at_100": operational_hits,
            "recall_at_100": operational_hits / evaluated,
        },
        "candidate_count": {
            "maximum_observed": max(candidate_counts),
            "mean": float(np.mean(candidate_counts)),
        },
        "latency_ms": {
            "p50": float(np.percentile(latency, 50)),
            "p95": float(np.percentile(latency, 95)),
            "p99": float(np.percentile(latency, 99)),
        },
        "positive_injection": False,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["seal_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
