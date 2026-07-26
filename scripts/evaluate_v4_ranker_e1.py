#!/usr/bin/env python3
"""Compare raw, pinned-old and new V4 ranker orders on dev_new."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_v9_retrieval_experiment import (  # noqa: E402
    _git_commit,
    wilson_interval,
)
from src.xgb_matcher.v9_dataset import (  # noqa: E402
    V9DatasetManifest,
    file_sha256,
)


SCHEMA_VERSION = "sireto-v4-ranker-e1-evaluation-1"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _top1(frame: pd.DataFrame, order_column: str) -> pd.DataFrame:
    return (
        frame.sort_values(
            ["query_id", order_column, "candidate_siret"],
            ascending=[True, order_column == "retrieval_rank", True],
            kind="stable",
        )
        .groupby("query_id", as_index=False)
        .first()
    )


def ranking_summary(top1: pd.DataFrame, labels: pd.DataFrame) -> dict[str, Any]:
    evaluated = labels.merge(
        top1[["query_id", "candidate_siret", "candidate_siren"]],
        on="query_id",
        how="left",
        validate="one_to_one",
    )
    siret_correct = (
        evaluated["candidate_siret"].fillna("")
        == evaluated["ground_truth_siret"].fillna("")
    )
    siren_correct = (
        evaluated["candidate_siren"].fillna("")
        == evaluated["ground_truth_siren"].fillna("")
    )
    successes = int(siret_correct.sum())
    total = int(len(evaluated))
    low, high = wilson_interval(successes, total, confidence=0.95)
    return {
        "siret_successes": successes,
        "siren_successes": int(siren_correct.sum()),
        "total": total,
        "hit_at_1_siret": successes / total if total else 0.0,
        "hit_at_1_siren": float(siren_correct.mean()) if total else 0.0,
        "siret_wilson_95": [low, high],
        "missing_predictions": int(
            evaluated["candidate_siret"].isna().sum()
        ),
    }


def verdict(raw_rate: float, old_rate: float, new_rate: float) -> str:
    if new_rate > raw_rate and new_rate >= old_rate:
        return "GO_ACCEPTEUR_V4"
    if old_rate > new_rate:
        return "KEEP_OLD_RANKER"
    return "PIVOT_RANKER_V4"


def evaluate(
    *,
    dataset_dir: Path,
    old_model_dir: Path,
    new_model_dir: Path,
    contract_path: Path,
    output_dir: Path,
) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"Immutable output already exists: {output_dir}")
    dataset_manifest = V9DatasetManifest.load(
        dataset_dir / "manifest.json"
    )
    dataset_report = _read_json(dataset_dir / "build_report.json")
    if dataset_report.get("holdout_read") is not False:
        raise ValueError("Dataset builder did not preserve holdout sealing")
    for name in ("labels.parquet", "candidates.parquet"):
        path = dataset_dir / name
        if file_sha256(path) != dataset_report["outputs"][name]:
            raise ValueError(f"Dataset output hash mismatch: {path}")

    labels = pd.read_parquet(dataset_dir / "labels.parquet")
    labels["query_id"] = labels["query_id"].astype(str)
    dev_labels = labels[labels["split"].eq("dev")].copy()
    candidates = pd.read_parquet(dataset_dir / "candidates.parquet")
    candidates["query_id"] = candidates["query_id"].astype(str)
    dev = candidates[candidates["split"].eq("dev")].copy()
    if len(dev_labels) != 305:
        raise ValueError("E1 contract expects 305 dev queries")

    raw_top1 = _top1(dev, "retrieval_rank")
    old_metadata = _read_json(old_model_dir / "metadata.json")
    if old_metadata["feature_order"] != dataset_manifest.feature_order:
        raise ValueError("Pinned old ranker feature order is incompatible")
    old_model = xgb.XGBRanker()
    old_model.load_model(old_model_dir / "ranker.json")
    old_scored = dev[
        ["query_id", "candidate_siret", "candidate_siren"]
    ].copy()
    old_scored["score"] = old_model.predict(
        dev[dataset_manifest.feature_order].astype(float).to_numpy()
    )
    old_top1 = _top1(old_scored, "score")

    new_metadata = _read_json(new_model_dir / "metadata.json")
    if new_metadata["dataset_manifest_id"] != dataset_manifest.build_id:
        raise ValueError("New ranker was trained on a different dataset")
    if new_metadata["feature_order"] != dataset_manifest.feature_order:
        raise ValueError("New ranker feature order is incompatible")
    new_predictions = pd.read_parquet(
        new_model_dir / "ranker_predictions.parquet"
    )
    new_predictions["query_id"] = new_predictions["query_id"].astype(str)
    new_top1 = new_predictions[
        new_predictions["prediction_origin"].eq("holdout")
        & new_predictions["rank"].eq(1)
    ].copy()

    summaries = {
        "raw_admission_order": ranking_summary(raw_top1, dev_labels),
        "pinned_old_ranker": ranking_summary(old_top1, dev_labels),
        "new_v4_ranker": ranking_summary(new_top1, dev_labels),
    }
    selected_verdict = verdict(
        summaries["raw_admission_order"]["hit_at_1_siret"],
        summaries["pinned_old_ranker"]["hit_at_1_siret"],
        summaries["new_v4_ranker"]["hit_at_1_siret"],
    )

    truth = dev_labels[
        ["query_id", "ground_truth_siret", "ground_truth_siren"]
    ]
    comparison = (
        truth.merge(
            raw_top1[["query_id", "candidate_siret"]].rename(
                columns={"candidate_siret": "raw_siret"}
            ),
            on="query_id",
            how="left",
        )
        .merge(
            old_top1[["query_id", "candidate_siret"]].rename(
                columns={"candidate_siret": "old_siret"}
            ),
            on="query_id",
            how="left",
        )
        .merge(
            new_top1[["query_id", "candidate_siret"]].rename(
                columns={"candidate_siret": "new_siret"}
            ),
            on="query_id",
            how="left",
        )
    )
    for name in ("raw", "old", "new"):
        comparison[f"{name}_correct"] = (
            comparison[f"{name}_siret"]
            == comparison["ground_truth_siret"]
        )
    paired = {
        "old_wrong_new_right": int(
            (~comparison["old_correct"] & comparison["new_correct"]).sum()
        ),
        "old_right_new_wrong": int(
            (comparison["old_correct"] & ~comparison["new_correct"]).sum()
        ),
        "both_right": int(
            (comparison["old_correct"] & comparison["new_correct"]).sum()
        ),
        "both_wrong": int(
            (~comparison["old_correct"] & ~comparison["new_correct"]).sum()
        ),
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "verdict": selected_verdict,
        "dev_query_count": len(dev_labels),
        "candidate_recall": 1.0,
        "metrics": summaries,
        "paired_old_vs_new": paired,
        "controls": {
            "feature_order_compatible": True,
            "all_dev_queries_scored": all(
                metric["missing_predictions"] == 0
                for metric in summaries.values()
            ),
            "holdout_read": False,
            "old_test_read": False,
            "positive_injection": False,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    comparison_path = output_dir / "top1_comparison.parquet"
    summary_path = output_dir / "summary.json"
    comparison.to_parquet(comparison_path, index=False)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "verdict": selected_verdict,
        "holdout_read": False,
        "old_test_read": False,
        "inputs": {
            "dataset_manifest_sha256": file_sha256(
                dataset_dir / "manifest.json"
            ),
            "old_ranker_sha256": file_sha256(
                old_model_dir / "ranker.json"
            ),
            "old_metadata_sha256": file_sha256(
                old_model_dir / "metadata.json"
            ),
            "new_ranker_sha256": file_sha256(
                new_model_dir / "ranker.json"
            ),
            "new_predictions_sha256": file_sha256(
                new_model_dir / "ranker_predictions.parquet"
            ),
            "new_metadata_sha256": file_sha256(
                new_model_dir / "metadata.json"
            ),
            "contract_sha256": file_sha256(contract_path),
        },
        "outputs": {
            comparison_path.name: file_sha256(comparison_path),
            summary_path.name: file_sha256(summary_path),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--old-model-dir", type=Path, required=True)
    parser.add_argument("--new-model-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        evaluate(
            dataset_dir=args.dataset,
            old_model_dir=args.old_model_dir,
            new_model_dir=args.new_model_dir,
            contract_path=args.contract,
            output_dir=args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
