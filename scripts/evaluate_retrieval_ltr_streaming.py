#!/usr/bin/env python3
"""Evaluate the frozen dev admission query-by-query without loading 12M rows.

This command accepts fold 0 only.  It preserves complete query groups while
streaming Parquet batches, builds the same label-blind internal union, scores
the frozen LambdaMART model and writes a compact top-100 audit artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterator

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_retrieval_ltr_admission import _load_model  # noqa: E402
from src.xgb_matcher.retrieval_ltr_admission import (  # noqa: E402
    AdmissionConfig,
    build_internal_union,
    evaluate_outcomes,
    score_and_select,
)


SCHEMA_VERSION = "sireto-retrieval-ltr-streaming-dev-v1"
SELECTED_SCHEMA = pa.schema(
    [
        ("query_id", pa.string()),
        ("fold", pa.int16()),
        ("gt_siret", pa.string()),
        ("candidate_siret", pa.string()),
        ("candidate_siren", pa.string()),
        ("selected_rank", pa.int16()),
        ("ltr_score", pa.float32()),
        ("union_rank", pa.int32()),
        ("retrieval_channel_count", pa.int16()),
        ("is_exact_protected", pa.int8()),
        ("is_consensus_protected", pa.int8()),
    ]
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_complete_query_frames(
    path: Path, *, batch_size: int = 131_072
) -> Iterator[pd.DataFrame]:
    """Yield contiguous query groups even when an Arrow batch splits one."""
    carry: pd.DataFrame | None = None
    seen: set[str] = set()
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
        frame = batch.to_pandas()
        if carry is not None:
            frame = pd.concat([carry, frame], ignore_index=True)
        if frame.empty:
            continue
        frame["query_id"] = frame["query_id"].astype(str)
        last_query = str(frame.iloc[-1]["query_id"])
        complete = frame[frame["query_id"].ne(last_query)]
        carry = frame[frame["query_id"].eq(last_query)].copy()
        for query_id, group in complete.groupby("query_id", sort=False):
            if query_id in seen:
                raise ValueError(f"Non-contiguous query rows: {query_id}")
            seen.add(str(query_id))
            yield group.reset_index(drop=True)
    if carry is not None and not carry.empty:
        query_id = str(carry.iloc[0]["query_id"])
        if carry["query_id"].astype(str).nunique() != 1 or query_id in seen:
            raise ValueError(f"Non-contiguous final query rows: {query_id}")
        yield carry.reset_index(drop=True)


def _latency(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p95": None, "p99": None, "mean": None, "max": None}
    raw = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.quantile(raw, 0.50)),
        "p95": float(np.quantile(raw, 0.95)),
        "p99": float(np.quantile(raw, 0.99)),
        "mean": float(raw.mean()),
        "max": float(raw.max()),
    }


def _selected_table(frame: pd.DataFrame) -> pa.Table:
    compact = pd.DataFrame(
        {
            "query_id": frame["query_id"].astype(str),
            "fold": frame["fold"].astype(np.int16),
            "gt_siret": frame["gt_siret"].astype(str),
            "candidate_siret": frame["candidate_siret"].astype(str),
            "candidate_siren": frame["candidate_siren"].astype(str),
            "selected_rank": frame["selected_rank"].astype(np.int16),
            "ltr_score": frame["ltr_score"].astype(np.float32),
            "union_rank": frame["union_rank"].astype(np.int32),
            "retrieval_channel_count": frame["retrieval_channel_count"].astype(np.int16),
            "is_exact_protected": frame["is_exact_protected"].astype(np.int8),
            "is_consensus_protected": frame["is_consensus_protected"].astype(np.int8),
        }
    )
    return pa.Table.from_pandas(compact, schema=SELECTED_SCHEMA, preserve_index=False)


def evaluate(args: argparse.Namespace) -> Path:
    config = AdmissionConfig.load(args.config)
    if config.dev_fold != 0 or config.test_fold != 1:
        raise ValueError("Frozen fold policy changed")
    manifest_path = args.candidates.with_suffix(args.candidates.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("candidate_parquet_sha256") != sha256_file(args.candidates):
        raise ValueError("Candidate manifest hash mismatch")
    if manifest.get("positive_injection") is not False:
        raise ValueError("Candidate source does not prove zero injection")
    model = _load_model(args.model_dir, args.config, config)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    selected_path = args.output_dir / "selected_candidates.parquet"
    outcomes_path = args.output_dir / "query_outcomes.parquet"
    summary_path = args.output_dir / "summary.json"
    writer = pq.ParquetWriter(selected_path, SELECTED_SCHEMA, compression="zstd")
    outcomes: list[pd.DataFrame] = []
    query_count = 0
    raw_groups: list[pd.DataFrame] = []

    def flush() -> None:
        nonlocal query_count
        if not raw_groups:
            return
        raw_batch = pd.concat(raw_groups, ignore_index=True)
        union, _diagnostics = build_internal_union(
            raw_batch, config, allowed_folds=[config.dev_fold]
        )
        selected, outcome, _query_latency = score_and_select(model, union, config)
        writer.write_table(_selected_table(selected))
        outcomes.append(outcome)
        query_count += int(raw_batch["query_id"].astype(str).nunique())
        raw_groups.clear()

    try:
        for raw_group in iter_complete_query_frames(
            args.candidates, batch_size=args.batch_size
        ):
            if set(pd.to_numeric(raw_group["fold"]).astype(int)) != {config.dev_fold}:
                raise ValueError("Streaming development input opened a non-dev fold")
            raw_groups.append(raw_group)
            if len(raw_groups) >= args.query_batch_size:
                flush()
        flush()
    finally:
        writer.close()
    outcome_frame = pd.concat(outcomes, ignore_index=True)
    outcome_frame.to_parquet(outcomes_path, index=False, compression="zstd")
    measured = outcome_frame["retrieval_latency_measured"].astype(bool)
    end_to_end = (
        outcome_frame.loc[measured, "retrieval_latency_ms"]
        + outcome_frame.loc[measured, "union_latency_ms"]
        + outcome_frame.loc[measured, "admission_total_ms"]
    ).tolist()
    latency = {
        "query_count": int(len(outcome_frame)),
        "retrieval_measured_query_count": int(measured.sum()),
        "retrieval_latency_complete": bool(measured.all()),
        "retrieval_ms": _latency(
            outcome_frame.loc[measured, "retrieval_latency_ms"].tolist()
        ),
        "scoring_ms": _latency(outcome_frame["admission_scoring_ms"].tolist()),
        "score_and_select_ms": _latency(outcome_frame["admission_total_ms"].tolist()),
        "union_ms": _latency(outcome_frame["union_latency_ms"].tolist()),
        "end_to_end_admission_ms": _latency(end_to_end),
    }
    summary = evaluate_outcomes(
        outcome_frame,
        config,
        integrity={
            "positive_injection_absent": True,
            "synthetic_rows_absent": True,
            "fold1_closed": True,
        },
        latency=latency,
    )
    summary.update(
        {
            "schema_version_run": SCHEMA_VERSION,
            "query_count_streamed": query_count,
            "opened_folds": [0],
            "test_opened": False,
        }
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lists_path = args.output_dir / "candidate_lists.parquet"
    connection = duckdb.connect()
    connection.execute(
        f"""
        COPY (
          SELECT query_id, count(*) candidate_count,
                 to_json(list(candidate_siret ORDER BY selected_rank)) candidate_sirets_json
          FROM read_parquet('{selected_path}') GROUP BY query_id ORDER BY query_id
        ) TO '{lists_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    connection.close()
    output_manifest = {
        "schema_version": SCHEMA_VERSION,
        "config_sha256": sha256_file(args.config),
        "candidate_sha256": sha256_file(args.candidates),
        "candidate_manifest_sha256": sha256_file(manifest_path),
        "model_sha256": sha256_file(args.model_dir / "ranker.json"),
        "model_metadata_sha256": sha256_file(args.model_dir / "metadata.json"),
        "opened_folds": [0],
        "test_opened": False,
        "positive_injection": False,
        "outputs": {
            path.name: sha256_file(path)
            for path in (selected_path, outcomes_path, lists_path, summary_path)
        },
        "verdict": summary["verdict"],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(output_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return args.output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/retrieval_ltr_admission_official_v6.json",
    )
    parser.add_argument("--batch-size", type=int, default=131_072)
    parser.add_argument("--query-batch-size", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    print(evaluate(parse_args()))
