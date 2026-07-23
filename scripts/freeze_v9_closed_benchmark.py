#!/usr/bin/env python3
"""Freeze the SIREN-disjoint closed-set benchmark used by V9 gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xgb_matcher.retrieval_config import RetrievalConfigV1
from src.xgb_matcher.v9_dataset import file_sha256


SCHEMA_VERSION = "v9-closed-benchmark-1"


def directory_tree_sha256(root: Path) -> dict:
    """Hash relative paths and bytes for every file in a directory tree."""
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        file_count += 1
        total_bytes += size
    return {
        "sha256": digest.hexdigest(),
        "file_count": file_count,
        "total_bytes": total_bytes,
    }


def load_closed_queries(source: Path) -> pd.DataFrame:
    frame = pd.read_csv(source, sep=";", dtype=str, encoding="utf-8-sig")
    required = {
        "crm_name",
        "crm_cp",
        "crm_insee",
        "crm_id",
        "crm_commune",
        "gt_siret",
        "crm_adresse",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Closed benchmark source is missing columns: {missing}")
    frame = frame.copy()
    frame["gt_siret"] = (
        frame["gt_siret"].fillna("").str.replace(r"\D", "", regex=True)
    )
    invalid = ~frame["gt_siret"].str.fullmatch(r"\d{14}")
    if invalid.any():
        raise ValueError(f"Invalid exact SIRET labels: {int(invalid.sum())}")
    if frame["gt_siret"].duplicated().any():
        raise ValueError("Closed benchmark requires one unique ground-truth SIRET per row")
    frame["ground_truth_siren"] = frame["gt_siret"].str[:9]
    frame["query_id"] = [str(index) for index in range(len(frame))]
    return frame


def assign_legacy_v7_splits(
    frame: pd.DataFrame,
    *,
    seed: int,
    train_ratio: float = 0.70,
    dev_ratio: float = 0.15,
) -> pd.DataFrame:
    """Reproduce the historical V7 ordered-SIREN split, then freeze it."""
    sirens = frame["ground_truth_siren"].drop_duplicates().tolist()
    random.Random(seed).shuffle(sirens)
    train_end = int(len(sirens) * train_ratio)
    dev_end = int(len(sirens) * (train_ratio + dev_ratio))
    split_by_siren = {
        siren: "train"
        for siren in sirens[:train_end]
    }
    split_by_siren.update(
        {siren: "dev" for siren in sirens[train_end:dev_end]}
    )
    split_by_siren.update(
        {siren: "test" for siren in sirens[dev_end:]}
    )
    output = frame.copy()
    output["split"] = output["ground_truth_siren"].map(split_by_siren)
    return output


def split_audit(
    frame: pd.DataFrame,
    historical_samples: Path | None,
) -> dict:
    split_sirens = {
        split: set(group["ground_truth_siren"])
        for split, group in frame.groupby("split")
    }
    overlaps = {
        f"{left}_{right}": len(split_sirens[left] & split_sirens[right])
        for left, right in (("train", "dev"), ("train", "test"), ("dev", "test"))
    }
    if any(overlaps.values()):
        raise ValueError(f"SIREN leakage across splits: {overlaps}")

    audit = {
        "split_counts": {
            split: {
                "queries": int(len(group)),
                "sirets": int(group["gt_siret"].nunique()),
                "sirens": int(group["ground_truth_siren"].nunique()),
            }
            for split, group in frame.groupby("split")
        },
        "siren_overlaps": overlaps,
    }
    if historical_samples is None:
        return audit

    positives = pq.read_table(
        historical_samples,
        columns=["query_id", "siret", "label", "split"],
    ).to_pandas()
    positives = positives[pd.to_numeric(positives["label"], errors="coerce").eq(1)]
    positives["query_id"] = positives["query_id"].astype(str)
    reference = frame[["query_id", "gt_siret", "split"]]
    checked = positives.merge(
        reference,
        on="query_id",
        how="left",
        suffixes=("_historical", "_frozen"),
    )
    missing_reference = int(checked["gt_siret"].isna().sum())
    split_mismatches = int(
        checked["split_historical"].ne(checked["split_frozen"]).sum()
    )
    siret_mismatches = int(
        checked["siret"].astype(str).ne(checked["gt_siret"].astype(str)).sum()
    )
    if missing_reference or split_mismatches or siret_mismatches:
        raise ValueError(
            "Historical V7 split validation failed: "
            f"missing={missing_reference}, split={split_mismatches}, "
            f"siret={siret_mismatches}"
        )
    historical_query_ids = set(positives["query_id"])
    audit["historical_v7_samples"] = {
        "positive_queries": int(positives["query_id"].nunique()),
        "frozen_queries_absent_from_historical_candidate_scenes": int(
            (~frame["query_id"].isin(historical_query_ids)).sum()
        ),
        "absent_by_split": {
            split: int(
                (~group["query_id"].isin(historical_query_ids)).sum()
            )
            for split, group in frame.groupby("split")
        },
        "split_mismatches": split_mismatches,
        "siret_mismatches": siret_mismatches,
    }
    return audit


def freeze_benchmark(
    *,
    source: Path,
    historical_samples: Path | None,
    establishment_snapshot: Path,
    legal_unit_snapshot: Path,
    partitions_dir: Path,
    output_root: Path,
    seed: int,
) -> Path:
    source_hash = file_sha256(source)
    sample_hash = (
        file_sha256(historical_samples)
        if historical_samples is not None
        else None
    )
    establishment_hash = file_sha256(establishment_snapshot)
    legal_unit_hash = file_sha256(legal_unit_snapshot)
    partition_fingerprint = directory_tree_sha256(partitions_dir)
    retrieval_config = RetrievalConfigV1(
        sparse_retrieval_enabled=True,
        dense_retrieval_enabled=False,
        prefilter_k=500,
        fusion_mode="rrf",
        retrieval_budget=50,
        min_candidates=50,
        mega_insee_policy="full_insee",
    )
    identity = {
        "schema_version": SCHEMA_VERSION,
        "source_sha256": source_hash,
        "historical_samples_sha256": sample_hash,
        "establishment_snapshot_sha256": establishment_hash,
        "legal_unit_snapshot_sha256": legal_unit_hash,
        "partitions_sha256": partition_fingerprint["sha256"],
        "seed": seed,
        "split": "ordered unique SIREN; 70/15/15; Python random shuffle",
        "retrieval_config": retrieval_config.to_dict(),
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    output_dir = output_root / build_id
    if output_dir.exists():
        raise FileExistsError(f"Immutable benchmark already exists: {output_dir}")

    frame = assign_legacy_v7_splits(load_closed_queries(source), seed=seed)
    audit = split_audit(frame, historical_samples)
    queries = pd.DataFrame(
        {
            "query_id": frame["query_id"],
            "crm_record_id": frame["crm_id"],
            "crm_name": frame["crm_name"],
            "crm_address": frame["crm_adresse"],
            "crm_city": frame["crm_commune"],
            "postcode": frame["crm_cp"],
            "insee": frame["crm_insee"],
            "split": frame["split"],
            "date_reference": pd.Series([None] * len(frame), dtype="object"),
        }
    )
    labels = pd.DataFrame(
        {
            "query_id": frame["query_id"],
            "label_kind": "MATCH_EXACT",
            "ground_truth_siret": frame["gt_siret"],
            "ground_truth_siren": frame["ground_truth_siren"],
            "source": "historical_crm_ground_truth",
            "validator": "historical_unreviewed",
            "reference_date": pd.Series([None] * len(frame), dtype="object"),
        }
    )
    benchmark = queries.merge(labels, on="query_id", how="inner")

    output_dir.mkdir(parents=True, exist_ok=False)
    queries.to_parquet(output_dir / "queries.parquet", index=False)
    labels.to_parquet(output_dir / "labels.parquet", index=False)
    benchmark.to_parquet(output_dir / "benchmark.parquet", index=False)
    benchmark.to_csv(output_dir / "benchmark.csv", sep=";", index=False)
    (output_dir / "split_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    manifest = {
        **identity,
        "build_id": build_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "paths": {
            "source": str(source.resolve()),
            "historical_samples": (
                str(historical_samples.resolve())
                if historical_samples is not None
                else None
            ),
            "establishment_snapshot": str(establishment_snapshot.resolve()),
            "legal_unit_snapshot": str(legal_unit_snapshot.resolve()),
            "partitions": str(partitions_dir.resolve()),
        },
        "partitions": partition_fingerprint,
        "counts": audit["split_counts"],
        "limitations": [
            "MATCH_EXACT labels are historical CRM ground truth, not a fresh human audit.",
            "The SIRENE semantic snapshot date is unknown; file hashes are authoritative.",
            "The local fine-tuned dense model has seen or tuned on these SIRENs and is not valid for final quality claims.",
        ],
        "output_sha256": {
            name: file_sha256(output_dir / name)
            for name in (
                "queries.parquet",
                "labels.parquet",
                "benchmark.parquet",
                "benchmark.csv",
                "split_audit.json",
            )
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/crm_ok_gt.csv"))
    parser.add_argument(
        "--historical-samples",
        type=Path,
        default=Path("data/samples_v7_ranker.parquet"),
    )
    parser.add_argument(
        "--establishment-snapshot",
        type=Path,
        default=Path("data/StockEtablissement_utf8.parquet"),
    )
    parser.add_argument(
        "--legal-unit-snapshot",
        type=Path,
        default=Path("data/StockUniteLegale_utf8.parquet"),
    )
    parser.add_argument(
        "--partitions-dir",
        type=Path,
        default=Path("data/candidates_v7_all"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = freeze_benchmark(
        source=args.source,
        historical_samples=args.historical_samples,
        establishment_snapshot=args.establishment_snapshot,
        legal_unit_snapshot=args.legal_unit_snapshot,
        partitions_dir=args.partitions_dir,
        output_root=args.output_root,
        seed=args.seed,
    )
    print(output)


if __name__ == "__main__":
    main()
