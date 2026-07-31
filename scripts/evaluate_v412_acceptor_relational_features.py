#!/usr/bin/env python3
"""Ablate explicit relational evidence on the consumed V4.12 development set."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_v411_acceptor_development import (  # noqa: E402
    EXPECTED_MODEL_CONFIGS,
    decision_metrics,
    select_threshold,
)
from src.xgb_matcher.v411_acceptor import MONOTONIC_XGB  # noqa: E402
from src.xgb_matcher.v411_scene import (  # noqa: E402
    V411_ACCEPTOR_FEATURE_NAMES,
    V411_MONOTONIC_CONSTRAINTS,
)


BASE = Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100")
DEFAULT_STACK = BASE / "experiments/v4_12_ranker_acceptor_stack/f6d3c21bd8a8359e"
DEFAULT_OUTPUT_ROOT = BASE / "experiments/v4_12_acceptor_relational_features"
SCHEMA_VERSION = "sireto-v4.12-acceptor-relational-development-1"
HARD_WEIGHT = 10.0
RELATIONAL_FEATURES = {
    "legal_over_etab_top1": lambda frame: (
        frame["top1_name_sim_max_ul"] - frame["top1_name_sim_max_etab"]
    ),
    "legal_over_etab_delta": lambda frame: (
        frame["delta_name_sim_max_ul"] - frame["delta_name_sim_max_etab"]
    ),
    "legal_gap_support": lambda frame: (
        (frame["top1_name_sim_max_ul"] - frame["top1_name_sim_max_etab"])
        * frame["ranker_gap_fraction"]
    ),
    "site_competition_legal": lambda frame: (
        (frame["top1_name_sim_max_ul"] - frame["top1_name_sim_max_etab"])
        * frame["top1_siren_candidate_count"]
    ),
}
VARIANTS = {
    "BASE_WEIGHT_10": (),
    "LEGAL_RELATION_2": (
        "legal_over_etab_top1",
        "legal_over_etab_delta",
    ),
    "LEGAL_RELATION_4": tuple(RELATIONAL_FEATURES),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _add_relations(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for name, function in RELATIONAL_FEATURES.items():
        output[name] = function(output).astype(np.float64)
    return output


def _fit(frame: pd.DataFrame, hard_ids: set[str], columns: list[str], extra: tuple[str, ...]) -> Any:
    config = dict(EXPECTED_MODEL_CONFIGS[MONOTONIC_XGB])
    config["monotone_constraints"] = tuple(V411_MONOTONIC_CONSTRAINTS) + tuple(
        0 for _ in extra
    )
    model = xgb.XGBClassifier(**config)
    weights = np.where(frame["query_id"].isin(hard_ids), HARD_WEIGHT, 1.0).astype(
        np.float32
    )
    model.fit(
        frame[columns].to_numpy(dtype=np.float64),
        frame["acceptor_target"].astype(int).to_numpy(),
        sample_weight=weights,
    )
    return model


def _scores(model: Any, frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return np.asarray(
        model.predict_proba(frame[columns].to_numpy(dtype=np.float64))[:, 1],
        dtype=np.float64,
    )


def run(args: argparse.Namespace) -> Path:
    stack = args.stack.resolve()
    scenes = _add_relations(pd.read_parquet(stack / "acceptor_scenes.parquet"))
    hard_ids = set(pd.read_csv(args.r30_labels, dtype=str)["query_id"]) | set(
        pd.read_csv(args.r53_labels, dtype=str)["query_id"]
    )
    independent_ids = set(pd.read_csv(args.independent_labels, dtype=str)["query_id"])
    if len(hard_ids) != 83 or len(independent_ids) != 7 or hard_ids & independent_ids:
        raise ValueError("Adjudicated populations changed")

    base_fit = scenes[scenes["split"].eq("fit")].copy()
    hard = scenes[scenes["query_id"].isin(hard_ids)].copy()
    development = scenes[
        scenes["split"].eq("dev")
        & ~scenes["query_id"].isin(hard_ids | independent_ids)
        & scenes["label_kind"].isin(["MATCH_EXACT", "AMBIGUOUS"])
    ].copy()
    threshold = development[development["dev_partition"].eq("threshold_dev")]
    comparison = development[development["dev_partition"].eq("comparison_dev")]
    if (len(base_fit), len(hard), len(threshold), len(comparison)) != (5547, 83, 665, 701):
        raise ValueError("Relational-ablation populations changed")

    results: list[dict[str, Any]] = []
    oof_tables: list[pd.DataFrame] = []
    for variant_name, extra in VARIANTS.items():
        columns = list(V411_ACCEPTOR_FEATURE_NAMES) + list(extra)
        model = _fit(
            pd.concat([base_fit, hard], ignore_index=True), hard_ids, columns, extra
        )
        selected = select_threshold(
            _scores(model, threshold, columns),
            threshold["acceptor_target"].astype(int).to_numpy(),
            threshold["label_kind"].astype(str).to_numpy(),
        )
        if selected is None:
            results.append({"variant": variant_name, "eligible": False, "reason": "NO_THRESHOLD"})
            continue
        cutoff, threshold_metrics, _ = selected
        comparison_metrics = decision_metrics(
            _scores(model, comparison, columns),
            comparison["acceptor_target"].astype(int).to_numpy(),
            comparison["label_kind"].astype(str).to_numpy(),
            cutoff,
        )

        held_out_parts: list[pd.DataFrame] = []
        for fold in range(5):
            train_hard = hard[hard["oof_fold"].astype(int).ne(fold)]
            held_out = hard[hard["oof_fold"].astype(int).eq(fold)].copy()
            fold_model = _fit(
                pd.concat([base_fit, train_hard], ignore_index=True),
                set(train_hard["query_id"]),
                columns,
                extra,
            )
            held_out["score"] = _scores(fold_model, held_out, columns)
            held_out_parts.append(held_out)
        hard_oof = pd.concat(held_out_parts, ignore_index=True)
        hard_metrics = decision_metrics(
            hard_oof["score"].to_numpy(dtype=np.float64),
            hard_oof["acceptor_target"].astype(int).to_numpy(),
            hard_oof["label_kind"].astype(str).to_numpy(),
            cutoff,
        )
        safe = (
            comparison_metrics["error_auto"] == 0
            and comparison_metrics["ambiguous_auto"] == 0
            and hard_metrics["error_auto"] == 0
            and hard_metrics["ambiguous_auto"] == 0
        )
        results.append(
            {
                "variant": variant_name,
                "extra_features": list(extra),
                "safe": safe,
                "threshold": cutoff,
                "threshold_metrics": threshold_metrics,
                "comparison_metrics": comparison_metrics,
                "hard_oof_metrics": hard_metrics,
            }
        )
        oof_tables.append(
            hard_oof[["query_id", "label_kind", "acceptor_target", "score"]]
            .assign(variant=variant_name, threshold=cutoff)
        )

    baseline_auto = int(results[0]["hard_oof_metrics"]["auto_count"])
    improving = [
        result
        for result in results[1:]
        if result.get("safe")
        and int(result["hard_oof_metrics"]["auto_count"]) > baseline_auto
    ]
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "CONSUMED_DEVELOPMENT_ONLY",
        "hard_weight": HARD_WEIGHT,
        "independent_docket_used_for_selection": False,
        "final_test_opened": False,
        "production_promotion_authorized": False,
        "variants": results,
        "selected": max(
            improving,
            key=lambda item: (
                int(item["hard_oof_metrics"]["auto_count"]),
                int(item["comparison_metrics"]["auto_count"]),
            ),
            default=None,
        ),
        "verdict": "GO_RELATIONAL_FEATURES" if improving else "PIVOT_ACCEPTOR_FEATURES",
    }
    identity = hashlib.sha256(
        json.dumps(
            {
                "schema": SCHEMA_VERSION,
                "scenes": _sha256(stack / "acceptor_scenes.parquet"),
                "variants": {key: list(value) for key, value in VARIANTS.items()},
                "hard_weight": HARD_WEIGHT,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    output = args.output_root.resolve() / identity
    output.mkdir(parents=True, exist_ok=False)
    (output / "evaluation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    pd.concat(oof_tables, ignore_index=True).to_parquet(
        output / "hard_oof_scores.parquet", index=False
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack", type=Path, default=DEFAULT_STACK)
    parser.add_argument("--r30-labels", type=Path, default=Path("reports/v412_review_adjudication_labels.csv"))
    parser.add_argument("--r53-labels", type=Path, default=Path("reports/v412_review_rerank_counteraudit_53.csv"))
    parser.add_argument("--independent-labels", type=Path, default=Path("reports/v412_ranker_independent_validation_labels.csv"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
