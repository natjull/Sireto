#!/usr/bin/env python3
"""Train the preregistered deterministic XGBoost stack on cross-fitted BGE scores."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_v412_learned_business_features import BUSINESS_FEATURE_ORDER  # noqa: E402
from scripts.train_v412_learned_oof_rankers import RANKER_PARAMS, _fit  # noqa: E402
from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402


BASE = Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100")
DEFAULT_CORPUS = BASE / "datasets/v4_12_neural_text_corpus/02b8668f8050c5e9"
DEFAULT_BUSINESS = BASE / "datasets/v4_12_learned_business_features/8800ef53f6927215"
DEFAULT_RANKER = BASE / "experiments/v4_12_learned_oof_rankers/839ef55308d5077e"
DEFAULT_OUTPUT_ROOT = BASE / "experiments/v4_12_bge_xgb_stack"
SCHEMA_VERSION = "sireto-v4.12-bge-xgb-stack-1"
TRAIN_FOLDS = (2, 3, 4)
TARGET_FOLD = 0
TOP_K = 10

STACK_SIGNAL_FEATURES = [
    "business_ranker_score",
    "business_rank_recip",
    "business_gap_to_best",
    "business_top1_top2_gap",
    "bge_score",
    "bge_rank",
    "bge_rank_recip",
    "bge_gap_to_best",
    "bge_top1_top2_gap",
    "business_bge_top1_agreement",
    "candidate_is_business_top1",
    "candidate_is_bge_top1",
    "business_bge_rank_difference",
]


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validate_bge_artifact(path: Path, expected_target: int) -> dict[str, Any]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    evaluation = json.loads((path / "evaluation.json").read_text(encoding="utf-8"))
    if manifest.get("outputs", {}).get("target_scores.parquet") != file_sha256(
        path / "target_scores.parquet"
    ):
        raise ValueError(f"BGE score hash mismatch: {path}")
    identity = manifest.get("build_identity", {})
    if int(identity.get("target_fold", -1)) != expected_target:
        raise ValueError(f"BGE target fold mismatch: {path}")
    train_folds = {int(value) for value in identity.get("train_folds", [])}
    if expected_target in train_folds:
        raise ValueError(f"BGE score is in-sample for fold {expected_target}")
    if evaluation.get("scope") != "OOF_TARGET":
        raise ValueError(f"BGE artifact is not a full OOF target: {path}")
    if evaluation.get("positive_injection") is not False:
        raise ValueError(f"BGE artifact suggests positive injection: {path}")
    return manifest


def _add_scene_signals(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(
        ["query_id", "bge_score", "ranker_rank", "retrieval_rank", "candidate_siret"],
        ascending=[True, False, True, True, True],
        kind="mergesort",
    ).copy()
    ordered["bge_rank"] = (
        ordered.groupby("query_id", sort=False).cumcount() + 1
    ).astype(np.float32)
    grouped = ordered.groupby("query_id", sort=False)
    ordered["bge_best_score"] = grouped["bge_score"].transform("max")
    bge_second = grouped["bge_score"].transform(
        lambda values: values.nlargest(2).iloc[-1] if len(values) > 1 else values.iloc[0]
    )
    ordered["bge_gap_to_best"] = ordered["bge_best_score"] - ordered["bge_score"]
    ordered["bge_top1_top2_gap"] = ordered["bge_best_score"] - bge_second
    ordered["bge_rank_recip"] = 1.0 / ordered["bge_rank"]
    ordered["business_ranker_score"] = ordered["ranker_score"].astype(np.float32)
    ordered["business_rank_recip"] = 1.0 / ordered["ranker_rank"].astype(np.float32)
    business_best = grouped["ranker_score"].transform("max")
    business_second = grouped["ranker_score"].transform(
        lambda values: values.nlargest(2).iloc[-1] if len(values) > 1 else values.iloc[0]
    )
    ordered["business_gap_to_best"] = business_best - ordered["ranker_score"]
    ordered["business_top1_top2_gap"] = business_best - business_second
    ordered["candidate_is_business_top1"] = ordered["ranker_rank"].astype(int).eq(1).astype(np.float32)
    ordered["candidate_is_bge_top1"] = ordered["bge_rank"].astype(int).eq(1).astype(np.float32)
    business_top1 = (
        ordered.sort_values(
            ["query_id", "ranker_rank", "retrieval_rank", "candidate_siret"],
            kind="mergesort",
        )
        .drop_duplicates("query_id")
        .set_index("query_id")["candidate_siret"]
    )
    bge_top1 = (
        ordered[ordered["bge_rank"].astype(int).eq(1)]
        .set_index("query_id")["candidate_siret"]
    )
    agreement = business_top1.eq(bge_top1).astype(np.float32)
    ordered["business_bge_top1_agreement"] = ordered["query_id"].map(agreement).astype(np.float32)
    ordered["business_bge_rank_difference"] = (
        ordered["ranker_rank"].astype(np.float32) - ordered["bge_rank"]
    )
    for name in STACK_SIGNAL_FEATURES:
        ordered[name] = ordered[name].astype(np.float32)
    if not np.isfinite(ordered[STACK_SIGNAL_FEATURES].to_numpy(dtype=np.float32)).all():
        raise ValueError("Stack signals contain non-finite values")
    return ordered


def _evaluate_ranked(
    corpus: Path,
    fold: int,
    ranked: pd.DataFrame,
    query_ids: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = pd.read_parquet(corpus / "labels.parquet")
    labels["query_id"] = labels["query_id"].astype(str)
    truth = labels[
        labels["query_id"].isin(query_ids)
        & labels["label_kind"].eq("MATCH_EXACT")
        & labels["oof_fold"].astype(int).eq(fold)
    ].copy()
    top1 = ranked[ranked["stack_rank"].astype(int).eq(1)][
        ["query_id", "candidate_siret", "candidate_siren", "stack_score", "retrieval_rank"]
    ]
    detail = truth.merge(top1, on="query_id", how="left", validate="one_to_one")
    detail["correct"] = detail["candidate_siret"].astype("string").eq(
        detail["ground_truth_siret"].astype("string")
    ).fillna(False)
    masks = {
        "exact": pd.Series(True, index=detail.index),
        "difficult": detail["label_is_human_validated"].astype(bool),
        "active": detail["ground_truth_state"].eq("A"),
        "closed": detail["ground_truth_state"].eq("F"),
    }
    metrics = []
    for segment, mask in masks.items():
        selected = detail[mask]
        metrics.append(
            {
                "fold": fold,
                "segment": segment,
                "correct": int(selected["correct"].sum()),
                "total": len(selected),
                "hit_at_1": float(selected["correct"].mean()) if len(selected) else None,
            }
        )
    return pd.DataFrame(metrics), detail


def _load_bge_scores(paths: list[Path]) -> tuple[pd.DataFrame, dict[str, str]]:
    expected = {0, 2, 3, 4}
    by_fold: dict[int, Path] = {}
    manifests: dict[str, str] = {}
    for path in paths:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        fold = int(manifest.get("build_identity", {}).get("target_fold", -1))
        if fold in by_fold:
            raise ValueError(f"Duplicate BGE target fold {fold}")
        _validate_bge_artifact(path, fold)
        by_fold[fold] = path
        manifests[str(fold)] = file_sha256(path / "manifest.json")
    if set(by_fold) != expected:
        raise ValueError(f"Expected BGE target folds {sorted(expected)}, got {sorted(by_fold)}")
    parts = []
    for fold, path in sorted(by_fold.items()):
        part = pd.read_parquet(path / "target_scores.parquet")
        part["oof_fold"] = np.int8(fold)
        part = part[part["ranker_rank"].astype(int).le(TOP_K)].copy()
        parts.append(part)
    scores = pd.concat(parts, ignore_index=True)
    if scores.duplicated(["query_id", "candidate_siret"]).any():
        raise ValueError("BGE target scores contain duplicate candidates")
    return scores, manifests


def run(args: argparse.Namespace) -> Path:
    business_manifest = json.loads((args.business / "manifest.json").read_text(encoding="utf-8"))
    corpus_manifest = json.loads((args.corpus / "manifest.json").read_text(encoding="utf-8"))
    ranker_manifest = json.loads((args.ranker / "manifest.json").read_text(encoding="utf-8"))
    for root, manifest, name in (
        (args.business, business_manifest, "candidates_business.parquet"),
        (args.business, business_manifest, "labels.parquet"),
        (args.corpus, corpus_manifest, "labels.parquet"),
        (args.ranker, ranker_manifest, "business_learned_oof_candidates.parquet"),
    ):
        if manifest.get("outputs", {}).get(name) != file_sha256(root / name):
            raise ValueError(f"Input hash mismatch: {root / name}")
    if business_manifest.get("positive_injection") is not False:
        raise ValueError("Business dataset suggests positive injection")
    if list(business_manifest.get("business_feature_order", [])) != BUSINESS_FEATURE_ORDER:
        raise ValueError("Business feature order changed")

    bge_scores, bge_manifests = _load_bge_scores(args.bge_artifact)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "runner_sha256": file_sha256(Path(__file__)),
        "business_manifest_sha256": file_sha256(args.business / "manifest.json"),
        "corpus_manifest_sha256": file_sha256(args.corpus / "manifest.json"),
        "ranker_manifest_sha256": file_sha256(args.ranker / "manifest.json"),
        "bge_manifests_by_target_fold": bge_manifests,
        "train_folds": list(TRAIN_FOLDS),
        "target_fold": TARGET_FOLD,
        "business_top_k": TOP_K,
        "business_features": BUSINESS_FEATURE_ORDER,
        "stack_signal_features": STACK_SIGNAL_FEATURES,
        "ranker_params": RANKER_PARAMS,
        "positive_injection": False,
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    destination = args.output_root / build_id
    if destination.exists():
        existing = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        if existing.get("build_identity") != identity:
            raise FileExistsError(f"Conflicting immutable output: {destination}")
        return destination

    candidates = pd.read_parquet(args.business / "candidates_business.parquet")
    labels = pd.read_parquet(args.business / "labels.parquet")
    for frame in (candidates, labels, bge_scores):
        frame["query_id"] = frame["query_id"].astype(str)
    candidates["candidate_siret"] = candidates["candidate_siret"].astype(str)
    bge_scores["candidate_siret"] = bge_scores["candidate_siret"].astype(str)
    labels = labels[labels["oof_fold"].astype(int).isin({0, 2, 3, 4})].copy()
    rows = candidates.merge(
        labels[["query_id", "label_kind", "ground_truth_siret", "ranker_weight", "oof_fold"]],
        on="query_id",
        validate="many_to_one",
    ).merge(
        bge_scores[
            [
                "query_id",
                "candidate_siret",
                "ranker_score",
                "ranker_rank",
                "bge_score",
                "oof_fold",
            ]
        ],
        on=["query_id", "candidate_siret", "oof_fold"],
        validate="one_to_one",
    )
    rows = _add_scene_signals(rows)
    features = BUSINESS_FEATURE_ORDER + STACK_SIGNAL_FEATURES
    if not np.isfinite(rows[features].to_numpy(dtype=np.float32)).all():
        raise ValueError("Stack model matrix contains non-finite values")
    rows["training_positive"] = rows["candidate_siret"].eq(rows["ground_truth_siret"]).astype(np.int8)
    rows["query_weight"] = rows["ranker_weight"].astype(np.float32)
    train = rows[
        rows["oof_fold"].astype(int).isin(TRAIN_FOLDS)
        & rows["label_kind"].eq("MATCH_EXACT")
    ].copy()
    positive_counts = train.groupby("query_id", sort=False)["training_positive"].sum()
    eligible = set(positive_counts[positive_counts.eq(1)].index.astype(str))
    train = train[train["query_id"].isin(eligible)].copy()
    target = rows[
        rows["oof_fold"].astype(int).eq(TARGET_FOLD)
        & rows["label_kind"].eq("MATCH_EXACT")
    ].copy()
    target_query_ids = set(
        labels[
            labels["oof_fold"].astype(int).eq(TARGET_FOLD)
            & labels["label_kind"].eq("MATCH_EXACT")
        ]["query_id"].astype(str)
    )
    if target["query_id"].nunique() != len(target_query_ids):
        raise ValueError("A fold-0 exact query has no top-10 stack candidates")

    started = time.perf_counter()
    model, training = _fit(train, features, negative_limit=0)
    fit_seconds = time.perf_counter() - started
    target["stack_score"] = model.predict(target[features].to_numpy(dtype=np.float32)).astype(np.float32)
    ranked = target.sort_values(
        ["query_id", "stack_score", "ranker_score", "retrieval_rank", "candidate_siret"],
        ascending=[True, False, False, True, True],
        kind="mergesort",
    ).copy()
    ranked["stack_rank"] = (
        ranked.groupby("query_id", sort=False).cumcount() + 1
    ).astype(np.int16)
    stack_scores = ranked[
        [
            "query_id",
            "candidate_siret",
            "candidate_siren",
            "retrieval_rank",
            "ranker_score",
            "ranker_rank",
            "bge_score",
            "bge_rank",
            "stack_score",
            "stack_rank",
        ]
    ].copy()
    metrics, detail = _evaluate_ranked(
        args.corpus,
        TARGET_FOLD,
        stack_scores,
        target_query_ids,
    )
    baseline_top1 = pd.read_parquet(args.ranker / "business_learned_oof_top1.parquet")
    baseline_top1["query_id"] = baseline_top1["query_id"].astype(str)
    baseline_top1 = baseline_top1[baseline_top1["oof_fold"].astype(int).eq(TARGET_FOLD)][
        ["query_id", "predicted_siret", "top1_correct"]
    ].rename(
        columns={
            "predicted_siret": "business_predicted_siret",
            "top1_correct": "business_correct",
        }
    )
    comparison = detail.merge(baseline_top1, on="query_id", validate="one_to_one")
    comparison["stack_correct"] = comparison["correct"].astype(bool)
    matrix = {
        "both_correct": int((comparison["business_correct"] & comparison["stack_correct"]).sum()),
        "xgb_only_correct": int((comparison["business_correct"] & ~comparison["stack_correct"]).sum()),
        "stack_only_correct": int((~comparison["business_correct"] & comparison["stack_correct"]).sum()),
        "both_wrong": int((~comparison["business_correct"] & ~comparison["stack_correct"]).sum()),
    }
    metric_map = {str(row.segment): row for row in metrics.itertuples(index=False)}
    gate = {
        "exact_2452_of_2797": int(metric_map["exact"].correct) >= 2452,
        "difficult_33_of_38": int(metric_map["difficult"].correct) >= 33,
        "active_2164_of_2391": int(metric_map["active"].correct) >= 2164,
        "closed_246_of_406": int(metric_map["closed"].correct) >= 246,
        "all_queries_scored": len(comparison) == 2797,
        "candidate_ceiling_10": int(ranked.groupby("query_id").size().max()) <= TOP_K,
        "bge_scores_cross_fitted": True,
        "positive_injection": False,
    }
    passed = all(gate.values())

    args.output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{build_id}.", dir=args.output_root))
    try:
        model.save_model(temporary / "stack_model.json")
        stack_scores.to_parquet(temporary / "fold0_ranked_candidates.parquet", index=False)
        comparison.to_parquet(temporary / "fold0_top1_comparison.parquet", index=False)
        metrics.to_csv(temporary / "fold0_metrics.csv", index=False)
        evaluation = {
            "schema_version": SCHEMA_VERSION,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "scope": "SELECTION_FOLD_0",
            "candidate_ceiling": TOP_K,
            "train_folds": list(TRAIN_FOLDS),
            "target_fold": TARGET_FOLD,
            "training": training,
            "fit_seconds": fit_seconds,
            "metrics": metrics.to_dict("records"),
            "correction_matrix": matrix,
            "gate": gate,
            "gate_passed": passed,
            "verdict": "GO_OPEN_CONFIRMATION_FOLD_1" if passed else "STOP_RANKER_GATE",
            "positive_injection": False,
            "confirmation_fold_opened": False,
            "final_test_opened": False,
        }
        _json_dump(temporary / "evaluation.json", evaluation)
        output_names = [
            str(path.relative_to(temporary))
            for path in temporary.rglob("*")
            if path.is_file()
        ]
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "build_id": build_id,
            "build_identity": identity,
            "outputs": {
                name: file_sha256(temporary / name) for name in sorted(output_names)
            },
        }
        _json_dump(temporary / "manifest.json", manifest)
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--business", type=Path, default=DEFAULT_BUSINESS)
    parser.add_argument("--ranker", type=Path, default=DEFAULT_RANKER)
    parser.add_argument("--bge-artifact", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
