#!/usr/bin/env python3
"""Prepare numeric retrieval features by query, then fit frozen LambdaMART."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_retrieval_ltr_streaming import (  # noqa: E402
    iter_complete_query_frames,
)
from src.xgb_matcher.retrieval_ltr_admission import (  # noqa: E402
    AdmissionConfig,
    assert_feature_order_is_identifier_free,
    build_internal_union,
    feature_order,
    prepare_training_rows,
)


SCHEMA_VERSION = "sireto-retrieval-ltr-streaming-train-v1"


def presample_hard_candidates(
    group: pd.DataFrame,
    config: AdmissionConfig,
    *,
    cap: int,
    union_head: int,
    channel_head: int,
) -> pd.DataFrame:
    """Keep a label-blind cross-channel hard-negative pool plus the natural truth.

    The positive is never injected: it is retained only when it already exists
    in the source union.  Every negative is selected without consulting labels.
    """
    if cap <= 0 or len(group) <= cap:
        return group.reset_index(drop=True)
    if union_head < 1 or channel_head < 1 or cap < union_head:
        raise ValueError("Invalid hard-negative presample limits")
    working = group.reset_index(drop=True).copy()
    selected: set[int] = set()
    union_rank = pd.to_numeric(working.get("union_rank"), errors="coerce")
    selected.update(union_rank.nsmallest(union_head).index.tolist())
    channel_best = pd.Series(np.inf, index=working.index, dtype=np.float64)
    channel_count = pd.Series(0, index=working.index, dtype=np.int16)
    exact_count = pd.Series(0, index=working.index, dtype=np.int16)
    for channel in config.channels:
        column = f"{channel}_rank"
        if column not in working.columns:
            continue
        ranks = pd.to_numeric(working[column], errors="coerce")
        present = ranks.notna() & ranks.ge(1)
        if not present.any():
            continue
        channel_count = channel_count.add(present.astype(np.int16), fill_value=0)
        channel_best = pd.concat([channel_best, ranks], axis=1).min(axis=1)
        selected.update(ranks[present].nsmallest(channel_head).index.tolist())
        if channel in config.exact_channels:
            exact_count = exact_count.add(present.astype(np.int16), fill_value=0)
    gt_siret = str(working.iloc[0].get("gt_siret") or "")
    natural_positive = working.index[working["siret"].astype(str).eq(gt_siret)]
    selected.update(natural_positive.tolist())
    candidates = working.loc[sorted(selected)].copy()
    candidates["__exact_count"] = exact_count.loc[candidates.index]
    candidates["__channel_count"] = channel_count.loc[candidates.index]
    candidates["__best_channel_rank"] = channel_best.loc[candidates.index]
    candidates["__union_rank"] = union_rank.loc[candidates.index]
    candidates["__natural_positive"] = candidates["siret"].astype(str).eq(gt_siret)
    candidates = candidates.sort_values(
        [
            "__natural_positive",
            "__exact_count",
            "__channel_count",
            "__best_channel_rank",
            "__union_rank",
            "siret",
        ],
        ascending=[False, False, False, True, True, True],
        kind="stable",
    ).head(cap)
    return candidates.drop(
        columns=[
            "__exact_count",
            "__channel_count",
            "__best_channel_rank",
            "__union_rank",
            "__natural_positive",
        ]
    ).sort_index().reset_index(drop=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _feature_schema(config: AdmissionConfig) -> pa.Schema:
    return pa.schema(
        [("query_id", pa.string()), ("fold", pa.int16()), ("ltr_label", pa.int8())]
        + [(name, pa.float32()) for name in feature_order(config)]
    )


def _feature_table(frame: pd.DataFrame, config: AdmissionConfig) -> pa.Table:
    features = feature_order(config)
    compact = pd.DataFrame(
        {
            "query_id": frame["query_id"].astype(str),
            "fold": frame["fold"].astype(np.int16),
            "ltr_label": frame["ltr_label"].astype(np.int8),
            **{
                name: pd.to_numeric(frame[name], errors="coerce")
                .fillna(0.0)
                .astype(np.float32)
                for name in features
            },
        }
    )
    return pa.Table.from_pandas(
        compact, schema=_feature_schema(config), preserve_index=False
    )


def prepare(args: argparse.Namespace) -> Path:
    config = AdmissionConfig.load(args.config)
    manifest_path = args.candidates.with_suffix(args.candidates.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("candidate_parquet_sha256") != sha256_file(args.candidates):
        raise ValueError("Candidate manifest hash mismatch")
    if manifest.get("positive_injection") is not False:
        raise ValueError("Training source does not prove zero injection")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(
        args.output, _feature_schema(config), compression="zstd"
    )
    queries = rows = positives = missing_positive = same_site_excluded = 0
    skipped_non_train_queries = 0
    presample_rows_before = presample_rows_after = 0
    raw_groups: list[pd.DataFrame] = []

    def flush() -> None:
        nonlocal queries, rows, positives, missing_positive, same_site_excluded
        if not raw_groups:
            return
        raw_batch = pd.concat(raw_groups, ignore_index=True)
        union, _union_diagnostics = build_internal_union(
            raw_batch, config, allowed_folds=config.train_folds
        )
        training, diagnostics = prepare_training_rows(
            union, config, allowed_folds=config.train_folds
        )
        writer.write_table(_feature_table(training, config))
        queries += int(raw_batch["query_id"].astype(str).nunique())
        rows += len(training)
        positives += int(training["ltr_label"].sum())
        missing_positive += int(diagnostics["queries_missing_natural_positive"])
        same_site_excluded += int(
            diagnostics["same_siren_same_site_negatives_excluded"]
        )
        raw_groups.clear()

    try:
        for raw_group in iter_complete_query_frames(
            args.candidates, batch_size=args.batch_size
        ):
            folds = set(pd.to_numeric(raw_group["fold"]).astype(int))
            if len(folds) != 1:
                raise ValueError(f"Forbidden training fold group: {sorted(folds)}")
            if not folds.issubset(set(config.train_folds)):
                if args.skip_non_train_folds:
                    skipped_non_train_queries += 1
                    continue
                raise ValueError(f"Forbidden training fold group: {sorted(folds)}")
            presample_rows_before += len(raw_group)
            raw_group = presample_hard_candidates(
                raw_group,
                config,
                cap=args.hard_negative_presample_cap,
                union_head=args.hard_negative_union_head,
                channel_head=args.hard_negative_channel_head,
            )
            presample_rows_after += len(raw_group)
            raw_groups.append(raw_group)
            if len(raw_groups) >= args.query_batch_size:
                flush()
        flush()
    finally:
        writer.close()
    output_manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_candidates": str(args.candidates.resolve()),
        "source_candidates_sha256": sha256_file(args.candidates),
        "source_manifest_sha256": sha256_file(manifest_path),
        "candidate_parquet_sha256": sha256_file(args.output),
        "config_sha256": sha256_file(args.config),
        "folds": list(config.train_folds),
        "query_count_seen": queries,
        "training_rows": rows,
        "natural_positives": positives,
        "queries_missing_natural_positive": missing_positive,
        "same_site_negatives_excluded": same_site_excluded,
        "skipped_non_train_queries": skipped_non_train_queries,
        "hard_negative_presample": {
            "cap": args.hard_negative_presample_cap,
            "union_head": args.hard_negative_union_head,
            "channel_head": args.hard_negative_channel_head,
            "rows_before": presample_rows_before,
            "rows_after": presample_rows_after,
            "label_blind_negatives": True,
            "natural_positive_only_if_present": True,
        },
        "feature_order": feature_order(config),
        "positive_injection": False,
        "synthetic_rows": 0,
        "test_opened": False,
    }
    Path(str(args.output) + ".manifest.json").write_text(
        json.dumps(output_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return args.output


def _contiguous_qid(values: pd.Series) -> np.ndarray:
    raw = values.astype(str).to_numpy()
    if len(raw) == 0:
        raise ValueError("Empty training matrix")
    starts = np.empty(len(raw), dtype=bool)
    starts[0] = True
    starts[1:] = raw[1:] != raw[:-1]
    qid = np.cumsum(starts, dtype=np.int64) - 1
    # A query must occupy exactly one contiguous block.
    if len(set(raw)) != int(qid[-1]) + 1:
        raise ValueError("Training query groups are not contiguous")
    return qid


def train(args: argparse.Namespace) -> Path:
    config = AdmissionConfig.load(args.config)
    feature_manifest_path = Path(str(args.features) + ".manifest.json")
    feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    if feature_manifest.get("candidate_parquet_sha256") != sha256_file(args.features):
        raise ValueError("Feature manifest hash mismatch")
    if feature_manifest.get("folds") != list(config.train_folds):
        raise ValueError("Feature folds differ from frozen train folds")
    if feature_manifest.get("positive_injection") is not False:
        raise ValueError("Feature source does not prove zero injection")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    features = feature_order(config)
    assert_feature_order_is_identifier_free(features)
    frame = pd.read_parquet(
        args.features, columns=["query_id", "ltr_label", *features]
    )
    qid = _contiguous_qid(frame["query_id"])
    matrix = frame[features].to_numpy(dtype=np.float32, copy=False)
    labels = frame["ltr_label"].to_numpy(dtype=np.int8, copy=False)
    model = xgb.XGBRanker(**config.xgb_params)
    started_at = datetime.now(timezone.utc)
    model.fit(matrix, labels, qid=qid, verbose=False)
    finished_at = datetime.now(timezone.utc)
    args.output_dir.mkdir(parents=True)
    model_path = args.output_dir / "ranker.json"
    metadata_path = args.output_dir / "metadata.json"
    model.save_model(model_path)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "policy_id": config.policy_id,
        "feature_order": features,
        "train_folds": list(config.train_folds),
        "training_rows": int(len(frame)),
        "training_groups": int(qid[-1]) + 1,
        "natural_positives": int(labels.sum()),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "dev_fold_read": False,
        "test_opened": False,
        "positive_injection": False,
        "source_union_sha256": sha256_file(args.features),
        "config_sha256": sha256_file(args.config),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "config_sha256": sha256_file(args.config),
        "feature_sha256": sha256_file(args.features),
        "feature_manifest_sha256": sha256_file(feature_manifest_path),
        "opened_folds": list(config.train_folds),
        "dev_fold_read": False,
        "test_opened": False,
        "positive_injection": False,
        "outputs": {
            model_path.name: sha256_file(model_path),
            metadata_path.name: sha256_file(metadata_path),
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return args.output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--candidates", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/retrieval_ltr_admission_official_v6.json",
    )
    prepare_parser.add_argument("--hard-negative-presample-cap", type=int, default=0)
    prepare_parser.add_argument("--hard-negative-union-head", type=int, default=64)
    prepare_parser.add_argument("--hard-negative-channel-head", type=int, default=16)
    prepare_parser.add_argument("--batch-size", type=int, default=131_072)
    prepare_parser.add_argument("--query-batch-size", type=int, default=128)
    prepare_parser.add_argument(
        "--skip-non-train-folds",
        action="store_true",
        help="Read a mixed development artifact but exclude every non-2/3/4 group before materialization.",
    )
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--features", type=Path, required=True)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/retrieval_ltr_admission_official_v6.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(prepare(arguments) if arguments.command == "prepare" else train(arguments))
