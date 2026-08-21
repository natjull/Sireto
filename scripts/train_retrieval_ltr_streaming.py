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
    try:
        for raw_group in iter_complete_query_frames(
            args.candidates, batch_size=args.batch_size
        ):
            folds = set(pd.to_numeric(raw_group["fold"]).astype(int))
            if len(folds) != 1 or not folds.issubset(set(config.train_folds)):
                raise ValueError(f"Forbidden training fold group: {sorted(folds)}")
            union, _union_diagnostics = build_internal_union(
                raw_group, config, allowed_folds=config.train_folds
            )
            try:
                training, diagnostics = prepare_training_rows(
                    union, config, allowed_folds=config.train_folds
                )
            except ValueError as error:
                if "No trainable ranking groups" not in str(error):
                    raise
                missing_positive += 1
                queries += 1
                continue
            writer.write_table(_feature_table(training, config))
            queries += 1
            rows += len(training)
            positives += int(training["ltr_label"].sum())
            missing_positive += int(diagnostics["queries_missing_natural_positive"])
            same_site_excluded += int(
                diagnostics["same_siren_same_site_negatives_excluded"]
            )
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
    prepare_parser.add_argument("--batch-size", type=int, default=131_072)
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
