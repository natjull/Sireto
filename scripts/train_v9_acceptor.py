#!/usr/bin/env python3
"""Train the V9 query-level acceptor on OOF ranker scenes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xgb_matcher.v9_acceptor import train_selective_acceptor
from src.xgb_matcher.v9_dataset import V9DatasetManifest, read_table
from src.xgb_matcher.v9_scene import (
    assert_oof_training_scenes,
    build_query_scenes,
    split_dev_roles,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-precision", type=float, default=0.998)
    parser.add_argument("--min-auto-count", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = V9DatasetManifest.load(args.dataset / "manifest.json")
    manifest.validate(feature_order=manifest.feature_order)
    labels = pd.read_parquet(args.dataset / "labels.parquet")
    predictions = read_table(args.predictions)
    scenes = build_query_scenes(predictions, labels)
    assert_oof_training_scenes(scenes)
    scenes["dev_role"] = ""
    dev_mask = scenes["split"].eq("dev")
    scenes.loc[dev_mask, "dev_role"] = split_dev_roles(
        scenes.loc[dev_mask, "query_id"],
        seed=args.seed,
    ).to_numpy()

    bundle, report = train_selective_acceptor(
        scenes,
        dataset_manifest_id=manifest.build_id,
        target_precision=args.target_precision,
        min_auto_count=args.min_auto_count,
        seed=args.seed,
    )
    bundle.save(args.output_dir, {"training_report": report})
    scenes.to_parquet(args.output_dir / "scenes.parquet", index=False)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report["test"], indent=2))


if __name__ == "__main__":
    main()
