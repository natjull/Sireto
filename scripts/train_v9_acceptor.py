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

E2B_CONTRACT_COMMIT = "cf91432"


def validate_final_holdout_authorization(
    path: Path,
    *,
    dataset_manifest_id: str,
) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("purpose") != "downstream_final_holdout":
        raise ValueError("Final holdout authorization has an invalid purpose")
    if payload.get("dataset_manifest_id") != dataset_manifest_id:
        raise ValueError("Final holdout authorization targets another dataset")
    if payload.get("current_selective_test") is not False:
        raise ValueError("The consumed selective test cannot be authorized")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-precision", type=float, default=0.998)
    parser.add_argument("--min-auto-count", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--experiment",
        choices=("e2", "e2b"),
        default="e2",
        help="E2 uses isotonic only; E2b runs the pre-registered score transforms.",
    )
    parser.add_argument(
        "--evaluate-final-holdout",
        action="store_true",
        help="Evaluate split=test once, with an explicit new-holdout authorization.",
    )
    parser.add_argument(
        "--final-holdout-authorization",
        type=Path,
        help="Pre-frozen JSON authorization for a new independent holdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.experiment == "e2b":
        expected = {
            "target_precision": (args.target_precision, 0.998),
            "min_auto_count": (args.min_auto_count, 25),
            "seed": (args.seed, 42),
        }
        changed = {
            name: observed
            for name, (observed, frozen) in expected.items()
            if observed != frozen
        }
        if changed:
            raise ValueError(
                "E2b parameters are frozen by contract "
                f"{E2B_CONTRACT_COMMIT}: {changed}"
            )
    manifest = V9DatasetManifest.load(args.dataset / "manifest.json")
    manifest.validate(feature_order=manifest.feature_order)
    if args.evaluate_final_holdout:
        if args.final_holdout_authorization is None:
            raise ValueError(
                "--evaluate-final-holdout requires "
                "--final-holdout-authorization"
            )
        validate_final_holdout_authorization(
            args.final_holdout_authorization,
            dataset_manifest_id=manifest.build_id,
        )
    elif args.final_holdout_authorization is not None:
        raise ValueError(
            "--final-holdout-authorization requires --evaluate-final-holdout"
        )
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
        calibration_methods=(
            ("raw", "sigmoid", "isotonic")
            if args.experiment == "e2b"
            else ("isotonic",)
        ),
        minimum_gate_coverage=0.25,
        evaluate_test=args.evaluate_final_holdout,
    )
    report["experiment"] = args.experiment
    report["contract_commit"] = (
        E2B_CONTRACT_COMMIT if args.experiment == "e2b" else None
    )
    bundle.save(args.output_dir, {"training_report": report})
    scenes.to_parquet(args.output_dir / "scenes.parquet", index=False)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
