#!/usr/bin/env python3
"""Profile 512 label-blind v11 queries and prove candidate-order parity."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping

import duckdb
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.retrieve_all_human_v11_union import _rows, sha256_file  # noqa: E402
from src.xgb_matcher.official_evidence_retrieval import (  # noqa: E402
    OfficialEvidenceRetrievalConfig,
    OfficialEvidenceRetriever,
)
from src.xgb_matcher.retrieval_all_human_v11 import (  # noqa: E402
    AllHumanV11Retriever,
    RetrievalV11Config,
)
from src.xgb_matcher.retrieval_sidecars_v11 import RetrievalSidecarReader  # noqa: E402


SCHEMA_VERSION = "sireto-retrieval-all-human-v11-runtime-profile-1"
DEFAULT_CONFIG = ROOT / "config/retrieval_all_human_v11.json"
DEFAULT_SALT = "RETRIEVAL_ALL_HUMAN_V11_RUNTIME_512"


def deterministic_query_ids(
    rows: Iterable[Mapping[str, Any]], *, sample_size: int, salt: str
) -> list[str]:
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    identifiers = {str(row.get("query_id") or "") for row in rows}
    if "" in identifiers:
        raise ValueError("profile query_id cannot be empty")
    return sorted(
        identifiers,
        key=lambda value: (hashlib.sha256(f"{salt}|{value}".encode()).hexdigest(), value),
    )[:sample_size]


def candidate_order_digest(values: Mapping[str, list[str]]) -> str:
    digest = hashlib.sha256()
    for query_id in sorted(values):
        digest.update(query_id.encode())
        digest.update(b"\0")
        for siret in values[query_id]:
            digest.update(str(siret).encode())
            digest.update(b"\0")
    return digest.hexdigest()


def _reference_candidates(path: Path, query_ids: list[str]) -> dict[str, list[str]]:
    connection = duckdb.connect()
    try:
        connection.execute("CREATE TEMP TABLE wanted(query_id VARCHAR)")
        connection.executemany("INSERT INTO wanted VALUES (?)", [(value,) for value in query_ids])
        rows = connection.execute(
            "SELECT c.query_id, list(c.siret ORDER BY c.union_rank) "
            "FROM read_parquet(?) c JOIN wanted w USING(query_id) "
            "GROUP BY c.query_id ORDER BY c.query_id",
            [str(path.resolve())],
        ).fetchall()
    finally:
        connection.close()
    output = {str(query_id): [str(value) for value in sirets] for query_id, sirets in rows}
    missing = sorted(set(query_ids) - set(output))
    if missing:
        raise ValueError(f"reference union is missing {len(missing)} profile queries")
    return output


def _quantile(values: list[float], probability: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability))


def profile(args: argparse.Namespace) -> Path:
    if args.output.exists():
        raise FileExistsError(args.output)
    folds = {int(value) for value in args.folds.split(",") if value.strip()}
    if not folds or not folds.issubset({2, 3, 4}):
        raise ValueError("runtime profile may open only folds 2/3/4")
    all_rows = list(_rows(args.input, folds, args.batch_size))
    query_ids = deterministic_query_ids(
        all_rows, sample_size=args.sample_size, salt=args.salt
    )
    selected = set(query_ids)
    query_rows = {
        str(row["query_id"]): row
        for row in all_rows
        if str(row.get("query_id") or "") in selected
    }
    if len(query_rows) != len(query_ids):
        raise ValueError("profile source query identifiers are not unique")
    reference = _reference_candidates(args.reference_candidates, query_ids)

    config = RetrievalV11Config.load(args.config)
    official_config = OfficialEvidenceRetrievalConfig.load(args.official_config)
    official_config = replace(
        official_config,
        enable_direct_parent_expansion=False,
        enable_consensus_parent_expansion=False,
        relation_seed_limit=config.relation_seed_limit,
        search_workers=int(args.search_workers),
    )
    official = OfficialEvidenceRetriever.from_index_paths(
        args.base_index,
        args.overlay_index,
        official_config,
        temporal_overlay_index_path=args.temporal_overlay_index,
        legal_entity_role_index_path=args.legal_entity_role_index,
        sector_official_index_path=args.sector_official_index,
    )
    observed: dict[str, list[str]] = {}
    latencies: list[float] = []
    mismatch_queries: list[str] = []
    with RetrievalSidecarReader(args.sidecars) as sidecars:
        retriever = AllHumanV11Retriever(official, sidecars, config)
        for index, query_id in enumerate(query_ids):
            started = time.perf_counter()
            result = retriever.retrieve(query_rows[query_id])
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            values = [candidate.siret for candidate in result.candidates]
            observed[query_id] = values
            if values != reference[query_id]:
                mismatch_queries.append(query_id)
            if index >= int(args.warmup_queries):
                latencies.append(elapsed_ms)
    if mismatch_queries:
        raise AssertionError(
            f"runtime profile changed candidate order for {len(mismatch_queries)} queries"
        )
    if not latencies:
        raise ValueError("warmup_queries consumed the complete profile sample")
    report = {
        "schema_version": SCHEMA_VERSION,
        "sample_salt": args.salt,
        "sample_size": len(query_ids),
        "warmup_queries": int(args.warmup_queries),
        "measured_queries": len(latencies),
        "folds_opened": sorted(folds),
        "fold0_opened": False,
        "fold1_opened": False,
        "human_labels_opened": False,
        "candidate_order_byte_parity": True,
        "candidate_order_sha256": candidate_order_digest(observed),
        "reference_candidate_order_sha256": candidate_order_digest(reference),
        "latency_ms": {
            "p50": _quantile(latencies, 0.50),
            "p95": _quantile(latencies, 0.95),
            "p99": _quantile(latencies, 0.99),
            "maximum": max(latencies),
        },
        "gates": {
            "warm_p95_ms": {
                "maximum": 1000.0,
                "passed": _quantile(latencies, 0.95) <= 1000.0,
            },
            "warm_p99_ms": {
                "maximum": 2000.0,
                "passed": _quantile(latencies, 0.99) <= 2000.0,
            },
        },
        "inputs": {
            "queries_sha256": sha256_file(args.input),
            "reference_candidates_sha256": sha256_file(args.reference_candidates),
            "config_sha256": sha256_file(args.config),
            "official_config_sha256": sha256_file(args.official_config),
            "sidecar_manifest_sha256": sha256_file(args.sidecars / "manifest.json"),
        },
    }
    report["verdict"] = (
        "RUNTIME_GATE_PASS"
        if all(value["passed"] for value in report["gates"].values())
        else "PIVOT_LATENCY"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output)
    return args.output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference-candidates", type=Path, required=True)
    parser.add_argument("--base-index", type=Path, required=True)
    parser.add_argument("--overlay-index", type=Path)
    parser.add_argument("--temporal-overlay-index", type=Path)
    parser.add_argument("--legal-entity-role-index", type=Path)
    parser.add_argument("--sector-official-index", type=Path)
    parser.add_argument("--sidecars", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--official-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", default="2,3,4")
    parser.add_argument("--sample-size", type=int, default=512)
    parser.add_argument("--warmup-queries", type=int, default=20)
    parser.add_argument("--search-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--salt", default=DEFAULT_SALT)
    return parser.parse_args()


def main() -> None:
    print(profile(parse_args()))


if __name__ == "__main__":
    main()
