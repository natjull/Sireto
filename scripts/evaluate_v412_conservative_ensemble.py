#!/usr/bin/env python3
"""Build a conservative V4.12 ranker ensemble on consumed development data.

The ensemble keeps the trusted-label ranker as its default and only swaps its
top two candidates under one of two explicit gates:

1. the small multilingual CE blend selects the stage-1 runner-up;
2. otherwise BGE and the targeted business ranker agree on that runner-up and
   BGE beats the stage-1 top-1 by a minimum raw-score margin.

The four counter-audited control-label corrections are applied only to the
reported corrected view.  The canonical dataset is never modified.
"""

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

from scripts.evaluate_v412_ranker_business_features import (
    BASE,
    DEFAULT_DATASET,
    DEFAULT_ETABLISSEMENTS,
    DEFAULT_TRUSTED,
    DEFAULT_UNITES_LEGALES,
    _read_enriched_sources,
    _relational_features,
    _source_features,
)


DEFAULT_SMALL = (
    BASE / "experiments/v4_12_cross_encoder_reranker/8d93c540ffcc3c04"
)
DEFAULT_BGE = (
    BASE / "experiments/v4_12_cross_encoder_reranker/d19079d68fc0940b"
)
DEFAULT_BUSINESS = (
    BASE / "experiments/v4_12_ranker_business_features/825f8266f658a093"
)
DEFAULT_OVERLAY = Path("reports/v412_control_label_counteraudit_4.csv")
DEFAULT_OUTPUT_ROOT = BASE / "experiments/v4_12_conservative_ensemble"

SMALL_ALPHA = 0.75
BGE_ALPHA = 10.0
BGE_RAW_MARGIN = 0.004


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _zscore(frame: pd.DataFrame, column: str) -> pd.Series:
    grouped = frame.groupby("query_id", sort=False)[column]
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0.0, np.nan)
    return ((frame[column] - mean) / std).fillna(0.0).astype("float32")


def _blend_top1(
    frame: pd.DataFrame, score_column: str, alpha: float, prefix: str
) -> pd.DataFrame:
    scored = frame.copy()
    scored["_blend"] = scored["stage1_z"] + alpha * scored[f"{score_column}_z"]
    top = (
        scored.sort_values(
            ["query_id", "_blend", "stage1_rank", "candidate_siret"],
            ascending=[True, False, True, True],
            kind="mergesort",
        )
        .drop_duplicates("query_id")
        [["query_id", "candidate_siret", "stage1_rank", score_column]]
    )
    return top.rename(
        columns={
            "candidate_siret": f"{prefix}_siret",
            "stage1_rank": f"{prefix}_stage1_rank",
            score_column: f"{prefix}_raw_score",
        }
    )


def _business_control_top1(
    dataset: Path,
    query_ids: set[str],
    business_dir: Path,
    etablissements: Path,
    unites_legales: Path,
) -> pd.Series:
    enriched = _read_enriched_sources(dataset, etablissements, unites_legales)
    enriched = enriched[enriched["query_id"].isin(query_ids)].copy()
    enriched = _relational_features(_source_features(enriched))
    evaluation = json.loads((business_dir / "evaluation.json").read_text())
    features = evaluation["variants"]["targeted"]["features"]
    model = xgb.XGBRanker()
    model.load_model(business_dir / "targeted_ranker.json")
    enriched["_business_score"] = model.predict(
        enriched[features].to_numpy(dtype=np.float32)
    ).astype("float32")
    return (
        enriched.sort_values(
            ["query_id", "_business_score", "retrieval_rank", "candidate_siret"],
            ascending=[True, False, True, True],
            kind="mergesort",
        )
        .drop_duplicates("query_id")
        .set_index("query_id")["candidate_siret"]
    )


def _metrics(decisions: pd.DataFrame, prediction: str, truth: str) -> dict[str, Any]:
    def one(frame: pd.DataFrame) -> dict[str, Any]:
        correct = frame[prediction].eq(frame[truth])
        return {
            "query_count": int(len(frame)),
            "top1_correct_count": int(correct.sum()),
            "hit_at_1": float(correct.mean()) if len(frame) else None,
        }

    return {
        "trusted": one(decisions[decisions["scope"].eq("TRUSTED")]),
        "control": one(decisions[decisions["scope"].eq("CONTROL")]),
        "combined": one(decisions),
    }


def _swap_ranked_candidates(
    candidates: pd.DataFrame, decisions: pd.DataFrame
) -> pd.DataFrame:
    output = candidates.merge(
        decisions[["query_id", "base_siret", "predicted_siret", "ensemble_rule"]],
        on="query_id",
        validate="many_to_one",
    )
    output["original_ranker_score"] = output["stage1_score"].astype("float32")
    output["original_ranker_rank"] = output["stage1_rank"].astype("int16")
    output["ranker_score"] = output["original_ranker_score"].copy()

    swaps = decisions[decisions["predicted_siret"].ne(decisions["base_siret"])]
    for row in swaps.itertuples(index=False):
        query_mask = output["query_id"].eq(row.query_id)
        base_mask = query_mask & output["candidate_siret"].eq(row.base_siret)
        selected_mask = query_mask & output["candidate_siret"].eq(row.predicted_siret)
        if int(base_mask.sum()) != 1 or int(selected_mask.sum()) != 1:
            raise ValueError(f"Incomplete top-two swap for {row.query_id}")
        base_score = float(output.loc[base_mask, "ranker_score"].iloc[0])
        selected_score = float(output.loc[selected_mask, "ranker_score"].iloc[0])
        output.loc[base_mask, "ranker_score"] = selected_score
        output.loc[selected_mask, "ranker_score"] = base_score

    output = output.sort_values(
        ["query_id", "ranker_score", "retrieval_rank", "candidate_siret"],
        ascending=[True, False, True, True],
        kind="mergesort",
    )
    output["ranker_rank"] = (
        output.groupby("query_id", sort=False).cumcount().add(1).astype("int16")
    )
    top1 = output[output["ranker_rank"].eq(1)].set_index("query_id")["candidate_siret"]
    expected = decisions.set_index("query_id")["predicted_siret"]
    if not top1.sort_index().equals(expected.sort_index()):
        raise ValueError("Ranked-candidate top-1 does not match ensemble decisions")
    output["candidate_siren"] = output["candidate_siret"].str[:9]
    return output[
        [
            "query_id",
            "candidate_siret",
            "candidate_siren",
            "retrieval_rank",
            "ranker_score",
            "ranker_rank",
            "original_ranker_score",
            "original_ranker_rank",
            "ensemble_rule",
        ]
    ].reset_index(drop=True)


def run(args: argparse.Namespace) -> Path:
    dataset = args.dataset.resolve()
    small = pd.read_parquet(args.small.resolve() / "top20_scores.parquet").rename(
        columns={"cross_encoder_score": "small_score"}
    )
    bge = pd.read_parquet(args.bge.resolve() / "top20_scores.parquet").rename(
        columns={"cross_encoder_score": "bge_score"}
    )
    for frame in (small, bge):
        frame["query_id"] = frame["query_id"].astype(str)
        frame["candidate_siret"] = frame["candidate_siret"].astype(str)
    candidates = small.merge(
        bge[["query_id", "candidate_siret", "stage1_rank", "stage1_score", "bge_score"]],
        on=["query_id", "candidate_siret"],
        suffixes=("", "_bge"),
        validate="one_to_one",
    )
    if not candidates["stage1_rank"].eq(candidates["stage1_rank_bge"]).all():
        raise ValueError("Small CE and BGE do not share the same stage-1 ranking")
    if not np.allclose(candidates["stage1_score"], candidates["stage1_score_bge"]):
        raise ValueError("Small CE and BGE do not share the same stage-1 scores")
    candidates = candidates.drop(columns=["stage1_rank_bge", "stage1_score_bge"])
    candidates["stage1_z"] = _zscore(candidates, "stage1_score")
    candidates["small_score_z"] = _zscore(candidates, "small_score")
    candidates["bge_score_z"] = _zscore(candidates, "bge_score")

    base = candidates[candidates["stage1_rank"].eq(1)][
        ["query_id", "candidate_siret", "bge_score"]
    ].rename(
        columns={"candidate_siret": "base_siret", "bge_score": "base_bge_score"}
    )
    decisions = base.merge(
        _blend_top1(candidates, "small_score", args.small_alpha, "small"),
        on="query_id",
        validate="one_to_one",
    ).merge(
        _blend_top1(candidates, "bge_score", args.bge_alpha, "bge"),
        on="query_id",
        validate="one_to_one",
    )

    trusted = pd.read_csv(args.trusted_labels, dtype=str).fillna("")
    trusted_exact = trusted[trusted["label_kind"].eq("MATCH_EXACT")][
        ["query_id", "ground_truth_siret"]
    ].copy()
    trusted_exact["scope"] = "TRUSTED"
    all_ids = set(decisions["query_id"])
    trusted_ids = set(trusted["query_id"])
    control_ids = all_ids - trusted_ids
    labels = pd.read_parquet(dataset / "labels.parquet")
    labels["query_id"] = labels["query_id"].astype(str)
    control_truth = labels[labels["query_id"].isin(control_ids)][
        ["query_id", "ground_truth_siret"]
    ].copy()
    control_truth["scope"] = "CONTROL"
    truth = pd.concat([trusted_exact, control_truth], ignore_index=True)
    if len(truth) != 1_381 or truth["scope"].value_counts().to_dict() != {
        "CONTROL": 1_127,
        "TRUSTED": 254,
    }:
        raise ValueError("Expected 254 trusted and 1,127 control queries")
    decisions = decisions.merge(truth, on="query_id", validate="one_to_one")
    decisions = decisions.rename(columns={"ground_truth_siret": "historical_truth_siret"})

    business_trusted = pd.read_parquet(
        args.business.resolve() / "targeted_trusted_oof_comparison.parquet"
    )
    business_trusted["query_id"] = business_trusted["query_id"].astype(str)
    business = business_trusted.set_index("query_id")["predicted_siret_candidate"]
    business_control = _business_control_top1(
        dataset,
        control_ids,
        args.business.resolve(),
        args.etablissements,
        args.unites_legales,
    )
    business = pd.concat([business, business_control])
    decisions["business_siret"] = decisions["query_id"].map(business)
    if decisions["business_siret"].isna().any():
        raise ValueError("Business predictions are incomplete")

    decisions["bge_margin_over_base"] = (
        decisions["bge_raw_score"] - decisions["base_bge_score"]
    )
    small_gate = decisions["small_siret"].ne(decisions["base_siret"]) & decisions[
        "small_stage1_rank"
    ].eq(2)
    bge_business_gate = (
        ~small_gate
        & decisions["bge_siret"].eq(decisions["business_siret"])
        & decisions["bge_siret"].ne(decisions["base_siret"])
        & decisions["bge_stage1_rank"].eq(2)
        & decisions["bge_margin_over_base"].ge(args.bge_raw_margin)
    )
    decisions["predicted_siret"] = decisions["base_siret"]
    decisions.loc[small_gate, "predicted_siret"] = decisions.loc[
        small_gate, "small_siret"
    ]
    decisions.loc[bge_business_gate, "predicted_siret"] = decisions.loc[
        bge_business_gate, "bge_siret"
    ]
    decisions["ensemble_rule"] = "KEEP_STAGE1"
    decisions.loc[small_gate, "ensemble_rule"] = "SMALL_CE_RUNNER_UP"
    decisions.loc[bge_business_gate, "ensemble_rule"] = "BGE_BUSINESS_AGREEMENT"

    overlay = pd.read_csv(args.control_overlay, dtype=str)
    correction = overlay.set_index("query_id")["corrected_ground_truth_siret"]
    decisions["corrected_truth_siret"] = decisions["query_id"].map(correction).fillna(
        decisions["historical_truth_siret"]
    )
    decisions["historical_top1_correct"] = decisions["predicted_siret"].eq(
        decisions["historical_truth_siret"]
    )
    decisions["corrected_top1_correct"] = decisions["predicted_siret"].eq(
        decisions["corrected_truth_siret"]
    )
    decisions["base_historical_top1_correct"] = decisions["base_siret"].eq(
        decisions["historical_truth_siret"]
    )
    decisions["base_corrected_top1_correct"] = decisions["base_siret"].eq(
        decisions["corrected_truth_siret"]
    )

    retrieval = pd.read_parquet(dataset / "candidates_sparse_top100.parquet")[
        ["query_id", "candidate_siret", "retrieval_rank"]
    ]
    retrieval["query_id"] = retrieval["query_id"].astype(str)
    retrieval["candidate_siret"] = retrieval["candidate_siret"].astype(str)
    candidates = candidates.merge(
        retrieval, on=["query_id", "candidate_siret"], validate="one_to_one"
    )
    ranked = _swap_ranked_candidates(candidates, decisions)

    payload = {
        "schema_version": "sireto-v4.12-conservative-ranker-ensemble-development-1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "CONSUMED_DEVELOPMENT_ONLY",
        "final_test_opened": False,
        "configuration": {
            "small_alpha": args.small_alpha,
            "bge_alpha": args.bge_alpha,
            "bge_raw_margin": args.bge_raw_margin,
            "candidate_scope": "stage1_top20",
            "ranking_mutation": "swap_selected_runner_up_with_stage1_top1_scores",
        },
        "historical": {
            "base": _metrics(decisions, "base_siret", "historical_truth_siret"),
            "ensemble": _metrics(
                decisions, "predicted_siret", "historical_truth_siret"
            ),
        },
        "corrected_identifiable": {
            "base": _metrics(decisions, "base_siret", "corrected_truth_siret"),
            "ensemble": _metrics(
                decisions, "predicted_siret", "corrected_truth_siret"
            ),
        },
        "rule_counts": {
            str(key): int(value)
            for key, value in decisions["ensemble_rule"].value_counts().items()
        },
        "observed_corrected_regressions": int(
            (
                decisions["base_corrected_top1_correct"]
                & ~decisions["corrected_top1_correct"]
            ).sum()
        ),
        "observed_corrected_fixes": int(
            (
                ~decisions["base_corrected_top1_correct"]
                & decisions["corrected_top1_correct"]
            ).sum()
        ),
        "inputs": {
            "small_evaluation_sha256": _sha256(args.small / "evaluation.json"),
            "bge_evaluation_sha256": _sha256(args.bge / "evaluation.json"),
            "business_evaluation_sha256": _sha256(args.business / "evaluation.json"),
            "control_overlay_sha256": _sha256(args.control_overlay),
        },
    }
    identity = hashlib.sha256(
        json.dumps(
            {
                "schema": payload["schema_version"],
                "configuration": payload["configuration"],
                "inputs": payload["inputs"],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    output = args.output_root.resolve() / identity
    output.mkdir(parents=True, exist_ok=False)
    (output / "evaluation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    decisions.sort_values(["scope", "query_id"], kind="mergesort").to_parquet(
        output / "decisions.parquet", index=False
    )
    ranked.to_parquet(output / "ranked_candidates.parquet", index=False)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--trusted-labels", type=Path, default=DEFAULT_TRUSTED)
    parser.add_argument("--control-overlay", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument("--small", type=Path, default=DEFAULT_SMALL)
    parser.add_argument("--bge", type=Path, default=DEFAULT_BGE)
    parser.add_argument("--business", type=Path, default=DEFAULT_BUSINESS)
    parser.add_argument("--etablissements", type=Path, default=DEFAULT_ETABLISSEMENTS)
    parser.add_argument("--unites-legales", type=Path, default=DEFAULT_UNITES_LEGALES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--small-alpha", type=float, default=SMALL_ALPHA)
    parser.add_argument("--bge-alpha", type=float, default=BGE_ALPHA)
    parser.add_argument("--bge-raw-margin", type=float, default=BGE_RAW_MARGIN)
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
