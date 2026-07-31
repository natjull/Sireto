#!/usr/bin/env python3
"""Train the V4.12 acceptor after removing unadjudicated mechanical ambiguity."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate_v412_acceptor_hard_weight import _fit, _scores
from scripts.run_v411_acceptor_development import decision_metrics, select_threshold
from src.xgb_matcher.v411_acceptor import V411_ACCEPTOR_FAMILIES
from src.xgb_matcher.v411_scene import V411_ACCEPTOR_FEATURE_NAMES


BASE = Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100")
DEFAULT_STACK = (
    BASE
    / "experiments/v4_12_corrected_label_stack/aae2ad5814ecfb5b"
)
DEFAULT_OUTPUT_ROOT = BASE / "experiments/v4_12_acceptor_clean_target"
SCHEMA_VERSION = "sireto-v4.12-acceptor-clean-target-development-1"
WEIGHTS = (1.0, 5.0, 10.0, 20.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hard_ids(args: argparse.Namespace) -> set[str]:
    ids: set[str] = set()
    for path in (args.r30_labels, args.r53_labels, args.corrected_overlay):
        ids |= set(pd.read_csv(path, dtype=str)["query_id"].astype(str))
    if len(ids) != 143:
        raise ValueError("Expected 143 adjudicated hard development queries")
    return ids


def _apply_labels(scenes: pd.DataFrame, labels_path: Path) -> pd.DataFrame:
    labels = pd.read_csv(labels_path, dtype=str).fillna("")
    if len(labels) != 30 or labels["label_kind"].value_counts().to_dict() != {
        "MATCH_EXACT": 26,
        "AMBIGUOUS": 4,
    }:
        raise ValueError("Consumed comparison labels changed")
    output = scenes[scenes["query_id"].isin(set(labels["query_id"]))].drop(
        columns=[
            "label_kind",
            "ground_truth_siret",
            "ground_truth_siren",
            "acceptor_target",
        ]
    )
    output = output.merge(
        labels[["query_id", "label_kind", "ground_truth_siret"]],
        on="query_id",
        validate="one_to_one",
    )
    output["ground_truth_siren"] = output["ground_truth_siret"].map(
        lambda value: value[:9] if value else None
    )
    output["acceptor_target"] = (
        output["label_kind"].eq("MATCH_EXACT")
        & output["predicted_siret"].astype(str).eq(
            output["ground_truth_siret"].astype(str)
        )
    ).astype(np.int8)
    return output


def run(args: argparse.Namespace) -> Path:
    stack = args.stack.resolve()
    scenes = pd.read_parquet(stack / "acceptor_scenes.parquet")
    scenes["query_id"] = scenes["query_id"].astype(str)
    hard_ids = _hard_ids(args)
    base_exact = scenes[
        scenes["split"].eq("fit") & scenes["label_kind"].eq("MATCH_EXACT")
    ].copy()
    hard = scenes[scenes["query_id"].isin(hard_ids)].copy()
    consumed = _apply_labels(scenes, args.consumed_comparison_labels)
    if (
        len(base_exact) != 4666
        or base_exact["acceptor_target"].value_counts().to_dict() != {1: 4657, 0: 9}
        or len(hard) != 143
        or hard["acceptor_target"].value_counts().to_dict() != {1: 111, 0: 32}
        or consumed["acceptor_target"].value_counts().to_dict() != {1: 24, 0: 6}
    ):
        raise ValueError("Clean-target populations changed")

    variants: list[dict[str, Any]] = []
    oof_by_variant: dict[tuple[str, float], tuple[pd.DataFrame, np.ndarray]] = {}
    for family in V411_ACCEPTOR_FAMILIES:
        for weight in WEIGHTS:
            held_out_parts: list[pd.DataFrame] = []
            score_parts: list[np.ndarray] = []
            for fold in range(5):
                train_hard = hard[hard["oof_fold"].astype(int).ne(fold)]
                fit = pd.concat([base_exact, train_hard], ignore_index=True)
                model = _fit(
                    fit,
                    set(train_hard["query_id"]),
                    weight,
                    family,
                )
                held_out = hard[hard["oof_fold"].astype(int).eq(fold)].copy()
                held_out_parts.append(held_out)
                score_parts.append(_scores(model, held_out))
            hard_oof = pd.concat(held_out_parts, ignore_index=True)
            hard_scores = np.concatenate(score_parts)
            selected = select_threshold(
                hard_scores,
                hard_oof["acceptor_target"].astype(int).to_numpy(),
                hard_oof["label_kind"].astype(str).to_numpy(),
            )
            if selected is None:
                variants.append(
                    {
                        "family": family,
                        "hard_weight": weight,
                        "eligible": False,
                        "reason": "NO_SAFE_HARD_OOF_THRESHOLD",
                    }
                )
                continue
            cutoff, metrics, _ = selected
            eligible = (
                metrics["auto_count"] > 0
                and metrics["error_auto"] == 0
                and metrics["ambiguous_auto"] == 0
            )
            variants.append(
                {
                    "family": family,
                    "hard_weight": weight,
                    "threshold": cutoff,
                    "hard_oof_metrics": metrics,
                    "eligible": eligible,
                }
            )
            oof_by_variant[(family, weight)] = (hard_oof, hard_scores)

    eligible = [variant for variant in variants if variant.get("eligible")]
    winner = (
        sorted(
            eligible,
            key=lambda variant: (
                -int(variant["hard_oof_metrics"]["auto_count"]),
                str(variant["family"]),
                float(variant["hard_weight"]),
            ),
        )[0]
        if eligible
        else None
    )
    consumed_metrics = None
    consumed_detail = consumed[
        [
            "query_id",
            "label_kind",
            "ground_truth_siret",
            "predicted_siret",
            "acceptor_target",
        ]
    ].copy()
    model = None
    hard_oof_detail = pd.DataFrame()
    if winner is not None:
        family = str(winner["family"])
        weight = float(winner["hard_weight"])
        threshold = float(winner["threshold"])
        full_fit = pd.concat([base_exact, hard], ignore_index=True)
        model = _fit(full_fit, hard_ids, weight, family)
        consumed_scores = _scores(model, consumed)
        consumed_metrics = decision_metrics(
            consumed_scores,
            consumed["acceptor_target"].astype(int).to_numpy(),
            consumed["label_kind"].astype(str).to_numpy(),
            threshold,
        )
        consumed_detail["acceptor_score"] = consumed_scores
        consumed_detail["decision"] = np.where(
            consumed_scores >= threshold, "AUTO_MATCH", "REVIEW"
        )
        hard_oof, hard_scores = oof_by_variant[(family, weight)]
        hard_oof_detail = hard_oof[
            [
                "query_id",
                "label_kind",
                "ground_truth_siret",
                "predicted_siret",
                "acceptor_target",
                "oof_fold",
            ]
        ].copy()
        hard_oof_detail["acceptor_score"] = hard_scores
        hard_oof_detail["decision"] = np.where(
            hard_scores >= threshold, "AUTO_MATCH", "REVIEW"
        )

    comparison_safe = (
        consumed_metrics is not None
        and consumed_metrics["auto_count"] > 0
        and consumed_metrics["error_auto"] == 0
        and consumed_metrics["ambiguous_auto"] == 0
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "CONSUMED_DEVELOPMENT_ONLY",
        "final_test_opened": False,
        "production_promotion_authorized": False,
        "target_policy": {
            "base_fit_match_exact_retained": len(base_exact),
            "base_fit_unadjudicated_ambiguous_excluded": int(
                (scenes["split"].eq("fit") & scenes["label_kind"].eq("AMBIGUOUS")).sum()
            ),
            "adjudicated_hard_retained": len(hard),
        },
        "variants": variants,
        "selected": winner,
        "consumed_comparison_metrics": consumed_metrics,
        "verdict": "GO_NEXT_BLIND_DOCKET" if comparison_safe else "PIVOT_CLEAN_TARGET",
    }
    identity = hashlib.sha256(
        json.dumps(
            {
                "schema": SCHEMA_VERSION,
                "scenes": _sha256(stack / "acceptor_scenes.parquet"),
                "r30": _sha256(args.r30_labels),
                "r53": _sha256(args.r53_labels),
                "corrected": _sha256(args.corrected_overlay),
                "comparison": _sha256(args.consumed_comparison_labels),
                "weights": WEIGHTS,
                "families": V411_ACCEPTOR_FAMILIES,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    output = args.output_root.resolve() / identity
    output.mkdir(parents=True, exist_ok=False)
    (output / "evaluation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    consumed_detail.to_parquet(output / "consumed_comparison.parquet", index=False)
    hard_oof_detail.to_parquet(output / "hard_oof.parquet", index=False)
    if model is not None:
        joblib.dump(model, output / "acceptor_candidate.joblib")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack", type=Path, default=DEFAULT_STACK)
    parser.add_argument(
        "--r30-labels",
        type=Path,
        default=Path("reports/v412_review_adjudication_labels.csv"),
    )
    parser.add_argument(
        "--r53-labels",
        type=Path,
        default=Path("reports/v412_review_rerank_counteraudit_53.csv"),
    )
    parser.add_argument(
        "--corrected-overlay",
        type=Path,
        default=Path("reports/v412_corrected_review_overlay_60.csv"),
    )
    parser.add_argument(
        "--consumed-comparison-labels",
        type=Path,
        default=Path("reports/v412_corrected_acceptor_independent_labels_30.csv"),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
