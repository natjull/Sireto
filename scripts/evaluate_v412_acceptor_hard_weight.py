#!/usr/bin/env python3
"""Tune hard-scene weight for the V4.12 monotonic selective acceptor.

Selection uses only consumed development data.  Hard scenes are evaluated
out of fold at acceptor level; the seven-query independent docket is excluded
entirely because it has already been consumed by the preceding experiment.
"""

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

from scripts.run_v411_acceptor_development import (  # noqa: E402
    EXPECTED_MODEL_CONFIGS,
    decision_metrics,
    select_threshold,
)
from src.xgb_matcher.v411_acceptor import (  # noqa: E402
    COMPACT_LOGIT,
    MONOTONIC_XGB,
    V411_ACCEPTOR_FAMILIES,
    build_v411_acceptor,
)
from src.xgb_matcher.v411_scene import V411_ACCEPTOR_FEATURE_NAMES  # noqa: E402


BASE = Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100")
DEFAULT_STACK = BASE / "experiments/v4_12_ranker_acceptor_stack/f6d3c21bd8a8359e"
DEFAULT_OUTPUT_ROOT = BASE / "experiments/v4_12_acceptor_hard_weight"
SCHEMA_VERSION = "sireto-v4.12-acceptor-hard-weight-development-2"
WEIGHTS = (1.0, 5.0, 10.0, 20.0, 50.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scores(model: Any, frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(
        model.predict_proba(
            frame[V411_ACCEPTOR_FEATURE_NAMES].to_numpy(dtype=np.float64)
        )[:, 1],
        dtype=np.float64,
    )


def _fit(
    frame: pd.DataFrame,
    hard_ids: set[str],
    weight: float,
    family: str = MONOTONIC_XGB,
) -> Any:
    model = build_v411_acceptor(family, EXPECTED_MODEL_CONFIGS[family])
    sample_weight = np.where(frame["query_id"].isin(hard_ids), weight, 1.0).astype(
        np.float32
    )
    fit_kwargs = (
        {"model__sample_weight": sample_weight}
        if family == COMPACT_LOGIT
        else {"sample_weight": sample_weight}
    )
    model.fit(
        frame[V411_ACCEPTOR_FEATURE_NAMES].to_numpy(dtype=np.float64),
        frame["acceptor_target"].astype(int).to_numpy(),
        **fit_kwargs,
    )
    return model


def run(args: argparse.Namespace) -> Path:
    stack = args.stack.resolve()
    scenes = pd.read_parquet(stack / "acceptor_scenes.parquet")
    hard_ids = set(pd.read_csv(args.r30_labels, dtype=str)["query_id"]) | set(
        pd.read_csv(args.r53_labels, dtype=str)["query_id"]
    )
    if args.corrected_overlay is not None:
        hard_ids |= set(
            pd.read_csv(args.corrected_overlay, dtype=str)["query_id"].astype(str)
        )
    independent_ids = set(
        pd.read_csv(args.independent_labels, dtype=str)["query_id"]
    )
    expected_hard_count = 143 if args.corrected_overlay is not None else 83
    if (
        len(hard_ids) != expected_hard_count
        or len(independent_ids) != 7
        or hard_ids & independent_ids
    ):
        raise ValueError("Adjudicated populations changed")

    base_fit = scenes[scenes["split"].eq("fit")].copy()
    hard = scenes[scenes["query_id"].isin(hard_ids)].copy()
    fit = pd.concat([base_fit, hard], ignore_index=True)
    development = scenes[
        scenes["split"].eq("dev")
        & ~scenes["query_id"].isin(hard_ids | independent_ids)
        & scenes["label_kind"].isin(["MATCH_EXACT", "AMBIGUOUS"])
    ].copy()
    threshold = development[development["dev_partition"].eq("threshold_dev")]
    comparison = development[development["dev_partition"].eq("comparison_dev")]
    expected_development = 1306 if args.corrected_overlay is not None else 1366
    if (
        len(base_fit) != 5547
        or len(hard) != expected_hard_count
        or len(threshold) + len(comparison) != expected_development
        or threshold.empty
        or comparison.empty
    ):
        raise ValueError("Hard-weight populations changed")

    variants: list[dict[str, Any]] = []
    models: dict[tuple[str, float], Any] = {}
    families = (
        V411_ACCEPTOR_FAMILIES
        if args.corrected_overlay is not None
        else (MONOTONIC_XGB,)
    )
    for family in families:
        for weight in WEIGHTS:
            model = _fit(fit, hard_ids, weight, family)
            threshold_scores = _scores(model, threshold)
            selected = select_threshold(
                threshold_scores,
                threshold["acceptor_target"].astype(int).to_numpy(),
                threshold["label_kind"].astype(str).to_numpy(),
            )
            if selected is None:
                variants.append(
                    {
                        "family": family,
                        "hard_weight": weight,
                        "eligible": False,
                        "reason": "NO_THRESHOLD",
                    }
                )
                continue
            cutoff, threshold_metrics, _ = selected
            comparison_metrics = decision_metrics(
                _scores(model, comparison),
                comparison["acceptor_target"].astype(int).to_numpy(),
                comparison["label_kind"].astype(str).to_numpy(),
                cutoff,
            )

            hard_parts: list[pd.DataFrame] = []
            hard_scores: list[np.ndarray] = []
            for fold in range(5):
                train_hard = hard[hard["oof_fold"].astype(int).ne(fold)]
                fold_fit = pd.concat([base_fit, train_hard], ignore_index=True)
                fold_model = _fit(
                    fold_fit,
                    set(train_hard["query_id"]),
                    weight,
                    family,
                )
                held_out = hard[hard["oof_fold"].astype(int).eq(fold)]
                hard_parts.append(held_out)
                hard_scores.append(_scores(fold_model, held_out))
            hard_oof = pd.concat(hard_parts, ignore_index=True)
            hard_oof_scores = np.concatenate(hard_scores)
            hard_metrics = decision_metrics(
                hard_oof_scores,
                hard_oof["acceptor_target"].astype(int).to_numpy(),
                hard_oof["label_kind"].astype(str).to_numpy(),
                cutoff,
            )
            comparison_safe = (
                comparison_metrics["auto_count"] > 0
                and 1000 * comparison_metrics["correct_auto"]
                >= 998 * comparison_metrics["auto_count"]
                and comparison_metrics["ambiguous_auto"] == 0
            )
            hard_safe_and_useful = (
                hard_metrics["auto_count"] > 0
                and hard_metrics["error_auto"] == 0
                and hard_metrics["ambiguous_auto"] == 0
            )
            variants.append(
                {
                    "family": family,
                    "hard_weight": weight,
                    "eligible": comparison_safe and hard_safe_and_useful,
                    "threshold": cutoff,
                    "threshold_metrics": threshold_metrics,
                    "comparison_metrics": comparison_metrics,
                    "hard_oof_metrics": hard_metrics,
                }
            )
            models[(family, weight)] = model

    # In the historical mode, weight 1 remains a comparator.  With the
    # corrected overlay, both frozen model families and every weight are
    # eligible because the question is now model selection on corrected data.
    baseline_hard_auto = {
        family: next(
            int(variant["hard_oof_metrics"]["auto_count"])
            for variant in variants
            if variant.get("family") == family
            and variant.get("hard_weight") == 1.0
            and "hard_oof_metrics" in variant
        )
        for family in families
    }
    if args.corrected_overlay is not None:
        eligible = [
            variant
            for variant in variants
            if variant.get("eligible")
            and int(variant["hard_oof_metrics"]["auto_count"]) > 0
        ]
    else:
        eligible = [
            variant
            for variant in variants
            if variant.get("eligible")
            and float(variant["hard_weight"]) > 1.0
            and int(variant["hard_oof_metrics"]["auto_count"])
            > baseline_hard_auto[MONOTONIC_XGB]
        ]
    winner = None
    if eligible:
        winner = sorted(
            eligible,
            key=lambda variant: (
                -int(variant["hard_oof_metrics"]["auto_count"]),
                -int(variant["comparison_metrics"]["auto_count"]),
                float(variant["hard_weight"]),
                str(variant["family"]),
            ),
        )[0]

    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "CONSUMED_DEVELOPMENT_ONLY",
        "independent_docket_used_for_selection": False,
        "final_test_opened": False,
        "production_promotion_authorized": False,
        "weights": list(WEIGHTS),
        "weight_one_hard_oof_auto": baseline_hard_auto,
        "variants": variants,
        "selected": winner,
        "verdict": (
            "GO_NEW_INDEPENDENT_ACCEPTOR_DOCKET"
            if winner is not None
            else "PIVOT_ACCEPTOR_FEATURES"
        ),
    }
    identity = hashlib.sha256(
        json.dumps(
            {
                "schema": SCHEMA_VERSION,
                "stack_scenes": _sha256(stack / "acceptor_scenes.parquet"),
                "corrected_overlay": (
                    _sha256(args.corrected_overlay)
                    if args.corrected_overlay is not None
                    else None
                ),
                "weights": WEIGHTS,
                "families": families,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    output = args.output_root.resolve() / identity
    output.mkdir(parents=True, exist_ok=False)
    (output / "evaluation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    if winner is not None:
        joblib.dump(
            models[(str(winner["family"]), float(winner["hard_weight"]))],
            output / "acceptor_candidate.joblib",
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack", type=Path, default=DEFAULT_STACK)
    parser.add_argument("--r30-labels", type=Path, default=Path("reports/v412_review_adjudication_labels.csv"))
    parser.add_argument("--r53-labels", type=Path, default=Path("reports/v412_review_rerank_counteraudit_53.csv"))
    parser.add_argument("--independent-labels", type=Path, default=Path("reports/v412_ranker_independent_validation_labels.csv"))
    parser.add_argument("--corrected-overlay", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
