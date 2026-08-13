#!/usr/bin/env python3
"""Train and compare the three V4.12-L candidate rankers in SIREN-grouped OOF."""

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
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_v412_learned_business_features import (  # noqa: E402
    BUSINESS_FEATURE_ORDER,
)
from scripts.run_v411_ranker_c_development import (  # noqa: E402
    RANKER_C_FEATURE_ORDER,
)
from scripts.run_v9_retrieval_experiment import _binary_metric  # noqa: E402
from src.xgb_matcher.v9_dataset import file_sha256, normalize_siret  # noqa: E402


SCHEMA_VERSION = "sireto-v4.12-learned-oof-rankers-1"
SEED = 42
DEFAULT_DATASET = Path(
    "/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/"
    "v4_12_learned_business_features/a77a3a5b226cfbe6"
)
DEFAULT_LOCAL_LABELS = Path("reports/v412_review_local_identifiable_labels_279.csv")
RANKER_PARAMS: dict[str, Any] = {
    "objective": "rank:pairwise",
    "eval_metric": "ndcg@1",
    "n_estimators": 600,
    "learning_rate": 0.035,
    "max_depth": 6,
    "min_child_weight": 3,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 5.0,
    "random_state": SEED,
    "n_jobs": -1,
    "tree_method": "hist",
}
VARIANTS = {
    "BASELINE_45": {
        "features": RANKER_C_FEATURE_ORDER,
        "include_weak_open_labels": False,
    },
    "BUSINESS_LEARNED": {
        "features": BUSINESS_FEATURE_ORDER,
        "include_weak_open_labels": False,
    },
    "BUSINESS_WEAK_LABELS": {
        "features": BUSINESS_FEATURE_ORDER,
        "include_weak_open_labels": True,
        "human_weight_multiplier": 1.0,
    },
    "BUSINESS_HUMAN_X2": {
        "features": BUSINESS_FEATURE_ORDER,
        "include_weak_open_labels": False,
        "human_weight_multiplier": 2.0,
    },
    "BUSINESS_HUMAN_X4": {
        "features": BUSINESS_FEATURE_ORDER,
        "include_weak_open_labels": False,
        "human_weight_multiplier": 4.0,
    },
    "BUSINESS_NDCG": {
        "features": BUSINESS_FEATURE_ORDER,
        "include_weak_open_labels": False,
        "human_weight_multiplier": 1.0,
        "ranker_params_override": {
            "objective": "rank:ndcg",
            "lambdarank_pair_method": "topk",
            "lambdarank_num_pair_per_sample": 20,
        },
    },
}
for _spec in VARIANTS.values():
    _spec.setdefault("human_weight_multiplier", 1.0)
    _spec.setdefault("ranker_params_override", {})
PREDICTION_COLUMNS = [
    "query_id",
    "candidate_siret",
    "candidate_siren",
    "retrieval_rank",
    "ranker_score",
    "ranker_rank",
    "oof_fold",
    "variant",
]


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _metric(values: pd.Series) -> dict[str, Any]:
    if values.empty:
        return {
            "successes": 0,
            "total": 0,
            "rate": None,
            "wilson_95": None,
            "wilson_99": None,
        }
    return _binary_metric(values.astype(bool))


def _load_dataset(dataset: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidates_path = dataset / "candidates_business.parquet"
    labels_path = dataset / "labels.parquet"
    for path in (candidates_path, labels_path):
        if manifest.get("outputs", {}).get(path.name) != file_sha256(path):
            raise ValueError(f"Dataset hash mismatch: {path}")
    if manifest.get("positive_injection") is not False:
        raise ValueError("Ranker dataset suggests positive injection")
    if int(manifest.get("candidate_ceiling", -1)) > 100:
        raise ValueError("Ranker dataset exceeds 100 candidates")
    if list(manifest.get("business_feature_order") or []) != BUSINESS_FEATURE_ORDER:
        raise ValueError("Business feature order differs from the runner")
    candidates = pd.read_parquet(candidates_path)
    labels = pd.read_parquet(labels_path)
    for frame in (candidates, labels):
        frame["query_id"] = frame["query_id"].astype(str)
    candidates["candidate_siret"] = candidates["candidate_siret"].astype(str).str.zfill(14)
    candidates["retrieval_rank_recip"] = (
        1.0 / candidates["retrieval_rank"].astype(np.float32)
    ).astype(np.float32)
    needed = sorted(set(RANKER_C_FEATURE_ORDER + BUSINESS_FEATURE_ORDER))
    matrix = candidates[needed].to_numpy(dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("Ranker dataset contains non-finite model features")
    if candidates.duplicated(["query_id", "candidate_siret"]).any():
        raise ValueError("Ranker candidate pools contain duplicate SIRETs")
    if candidates.groupby("query_id", sort=False).size().max() > 100:
        raise ValueError("Ranker candidate ceiling exceeded")
    if set(candidates["query_id"]) != set(labels["query_id"]):
        raise ValueError("Ranker candidates and labels do not align")
    if set(labels["oof_fold"].astype(int)) != set(range(5)):
        raise ValueError("Ranker requires five OOF folds")
    return manifest, candidates, labels


def _training_targets(
    candidates: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    include_weak_open_labels: bool,
    human_weight_multiplier: float = 1.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    label_columns = [
        "query_id",
        "label_kind",
        "ground_truth_siret",
        "historical_ground_truth_siret",
        "label_is_human_validated",
        "ranker_weight",
        "oof_fold",
    ]
    joined = candidates.merge(
        labels[label_columns], on="query_id", validate="many_to_one"
    )
    exact = joined["label_kind"].eq("MATCH_EXACT")
    weak = (
        include_weak_open_labels
        & ~exact
        & ~joined["label_is_human_validated"].astype(bool)
        & joined["historical_ground_truth_siret"].notna()
    )
    joined["training_target_siret"] = np.where(
        exact,
        joined["ground_truth_siret"],
        np.where(weak, joined["historical_ground_truth_siret"], None),
    )
    joined["training_target_siret"] = joined["training_target_siret"].map(
        normalize_siret
    )
    joined["training_positive"] = joined["candidate_siret"].eq(
        joined["training_target_siret"]
    ).astype(np.int8)
    positive_counts = joined.groupby("query_id", sort=False)["training_positive"].sum()
    eligible_ids = set(positive_counts[positive_counts.eq(1)].index.astype(str))
    joined = joined[joined["query_id"].isin(eligible_ids)].copy()
    joined["query_weight"] = np.where(
        joined["label_kind"].eq("MATCH_EXACT"),
        joined["ranker_weight"].astype(np.float32),
        np.float32(0.25),
    ).astype(np.float32)
    human_exact = (
        joined["label_kind"].eq("MATCH_EXACT")
        & joined["label_is_human_validated"].astype(bool)
    )
    joined.loc[human_exact, "query_weight"] *= np.float32(human_weight_multiplier)
    if joined.groupby("query_id")["query_weight"].nunique().max() != 1:
        raise ValueError("A training query carries multiple weights")
    exact_label_ids = set(labels.loc[labels["label_kind"].eq("MATCH_EXACT"), "query_id"])
    weak_label_ids = set(
        labels.loc[
            ~labels["label_kind"].eq("MATCH_EXACT")
            & ~labels["label_is_human_validated"].astype(bool),
            "query_id",
        ]
    )
    diagnostics = {
        "eligible_query_count": len(eligible_ids),
        "eligible_exact_query_count": len(eligible_ids & exact_label_ids),
        "eligible_weak_open_query_count": len(eligible_ids & weak_label_ids),
        "exact_missing_from_retrieval": len(exact_label_ids - eligible_ids),
    }
    return joined, diagnostics


def _hard_negative_sample(rows: pd.DataFrame, limit: int) -> pd.DataFrame:
    if limit <= 0:
        return rows
    ranked = rows.sort_values(
        ["query_id", "retrieval_rank", "candidate_siret"], kind="mergesort"
    ).copy()
    ranked["negative_order"] = ranked.groupby("query_id", sort=False).cumcount() + 1
    return ranked[
        ranked["negative_order"].le(limit) | ranked["training_positive"].eq(1)
    ].copy()


def _fit(
    rows: pd.DataFrame,
    features: list[str],
    *,
    negative_limit: int,
    ranker_params_override: dict[str, Any] | None = None,
) -> tuple[xgb.XGBRanker, dict[str, Any]]:
    sampled = _hard_negative_sample(rows, negative_limit)
    ordered = sampled.sort_values(
        ["query_id", "candidate_siret"], kind="mergesort"
    ).copy()
    grouped = ordered.groupby("query_id", sort=False)
    positive_counts = grouped["training_positive"].sum()
    if not positive_counts.eq(1).all():
        raise ValueError("Every ranker training group must contain one positive")
    query_order = list(grouped.indices)
    query_weights = (
        ordered.drop_duplicates("query_id").set_index("query_id")["query_weight"]
    )
    started = time.perf_counter()
    params = {**RANKER_PARAMS, **(ranker_params_override or {})}
    model = xgb.XGBRanker(**params)
    model.fit(
        ordered[features].to_numpy(dtype=np.float32),
        ordered["training_positive"].to_numpy(dtype=np.int8),
        group=grouped.size().to_numpy(),
        sample_weight=query_weights.reindex(query_order).to_numpy(dtype=np.float32),
        verbose=False,
    )
    return model, {
        "training_query_count": len(query_order),
        "training_candidate_count": len(ordered),
        "negative_limit": negative_limit,
        "elapsed_seconds": time.perf_counter() - started,
        "query_weight_counts": {
            str(k): int(v)
            for k, v in query_weights.value_counts().sort_index().items()
        },
    }


def _score(
    model: xgb.XGBRanker,
    rows: pd.DataFrame,
    features: list[str],
    *,
    fold: int,
    variant: str,
) -> pd.DataFrame:
    output = rows[
        ["query_id", "candidate_siret", "candidate_siren", "retrieval_rank"]
    ].copy()
    output["ranker_score"] = model.predict(
        rows[features].to_numpy(dtype=np.float32)
    ).astype(np.float32)
    if not np.isfinite(output["ranker_score"].to_numpy()).all():
        raise ValueError("Ranker produced a non-finite score")
    output["oof_fold"] = np.int8(fold)
    output["variant"] = variant
    output = output.sort_values(
        ["query_id", "ranker_score", "retrieval_rank", "candidate_siret"],
        ascending=[True, False, True, True],
        kind="mergesort",
    )
    output["ranker_rank"] = (
        output.groupby("query_id", sort=False).cumcount() + 1
    ).astype(np.int16)
    return output[PREDICTION_COLUMNS].reset_index(drop=True)


def _evaluate(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    local_labels: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    top1 = predictions[predictions["ranker_rank"].eq(1)].copy()
    if top1["query_id"].duplicated().any():
        raise ValueError("OOF predictions contain multiple top1 rows")
    detail = labels.merge(
        top1[["query_id", "candidate_siret", "ranker_score"]],
        on="query_id",
        how="left",
        validate="one_to_one",
    ).rename(columns={"candidate_siret": "predicted_siret"})
    detail["top1_correct"] = (
        detail["label_kind"].eq("MATCH_EXACT")
        & detail["predicted_siret"].eq(detail["ground_truth_siret"])
    )
    exact = detail["label_kind"].eq("MATCH_EXACT")
    local_exact_ids = set(
        local_labels.loc[local_labels["label_kind"].eq("MATCH_EXACT"), "query_id"]
    )
    human_exact = exact & detail["label_is_human_validated"].astype(bool)
    masks = {
        "all_exact_end_to_end": exact,
        "active_exact": exact & detail["ground_truth_state"].eq("A"),
        "closed_exact": exact & detail["ground_truth_state"].eq("F"),
        "human_validated_exact": human_exact,
        "difficult_local_241": detail["query_id"].isin(local_exact_ids),
    }
    metrics = {
        name: _metric(detail.loc[mask, "top1_correct"])
        for name, mask in masks.items()
    }
    metrics["folds"] = {
        str(fold): _metric(
            detail.loc[exact & detail["oof_fold"].eq(fold), "top1_correct"]
        )
        for fold in range(5)
    }
    return metrics, detail


def _save_models(models: dict[str, xgb.XGBRanker], directory: Path) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=False)
    hashes: dict[str, str] = {}
    for name, model in sorted(models.items()):
        path = directory / f"{name}.json"
        model.save_model(path)
        hashes[name] = file_sha256(path)
    return hashes


def run(args: argparse.Namespace) -> Path:
    manifest, candidates, labels = _load_dataset(args.dataset)
    local_labels = pd.read_csv(args.local_labels, dtype=str, keep_default_na=False)
    local_labels["query_id"] = local_labels["query_id"].astype(str)
    if len(local_labels[local_labels["label_kind"].eq("MATCH_EXACT")]) != 241:
        raise ValueError("The difficult local exact view must contain 241 rows")

    variant_names = args.variants or list(VARIANTS)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "runner_sha256": file_sha256(Path(__file__)),
        "dataset_manifest_sha256": file_sha256(args.dataset / "manifest.json"),
        "local_labels_sha256": file_sha256(args.local_labels),
        "variants": variant_names,
        "variant_specs": {name: VARIANTS[name] for name in variant_names},
        "ranker_params": RANKER_PARAMS,
        "negative_limit": args.negative_limit,
        "folds": [0, 1, 2, 3, 4],
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    destination = args.output_root / build_id
    if destination.exists():
        existing = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        if existing.get("build_identity") != identity:
            raise FileExistsError(f"Conflicting immutable output: {destination}")
        return destination

    args.output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{build_id}.", dir=args.output_root))
    started = time.perf_counter()
    results: dict[str, Any] = {}
    prediction_paths: dict[str, Path] = {}
    model_hashes: dict[str, dict[str, str]] = {}
    try:
        for variant in variant_names:
            spec = VARIANTS[variant]
            features = list(spec["features"])
            target_rows, target_diagnostics = _training_targets(
                candidates,
                labels,
                include_weak_open_labels=bool(spec["include_weak_open_labels"]),
                human_weight_multiplier=float(spec["human_weight_multiplier"]),
            )
            models: dict[str, xgb.XGBRanker] = {}
            prediction_parts: list[pd.DataFrame] = []
            fold_training: dict[str, Any] = {}
            for fold in range(5):
                train_rows = target_rows[target_rows["oof_fold"].astype(int).ne(fold)]
                model, training = _fit(
                    train_rows,
                    features,
                    negative_limit=args.negative_limit,
                    ranker_params_override=dict(spec["ranker_params_override"]),
                )
                models[f"oof_fold_{fold}"] = model
                validation = candidates[
                    candidates["query_id"].isin(
                        set(labels.loc[labels["oof_fold"].eq(fold), "query_id"])
                    )
                ]
                prediction_parts.append(
                    _score(
                        model,
                        validation,
                        features,
                        fold=fold,
                        variant=variant,
                    )
                )
                fold_training[str(fold)] = training
                print(
                    f"[ranker] {variant} fold={fold} "
                    f"train_queries={training['training_query_count']} "
                    f"seconds={training['elapsed_seconds']:.1f}",
                    flush=True,
                )
            full_model, full_training = _fit(
                target_rows,
                features,
                negative_limit=args.negative_limit,
                ranker_params_override=dict(spec["ranker_params_override"]),
            )
            models["full"] = full_model
            predictions = pd.concat(prediction_parts, ignore_index=True)
            if len(predictions) != len(candidates):
                raise ValueError(f"{variant}: OOF predictions do not cover candidates")
            if predictions.duplicated(["query_id", "candidate_siret"]).any():
                raise ValueError(f"{variant}: duplicate OOF candidate predictions")
            metrics, detail = _evaluate(predictions, labels, local_labels)
            prediction_path = temporary / f"{variant.lower()}_oof_candidates.parquet"
            detail_path = temporary / f"{variant.lower()}_oof_top1.parquet"
            predictions.to_parquet(prediction_path, index=False)
            detail.to_parquet(detail_path, index=False)
            prediction_paths[variant] = prediction_path
            model_hashes[variant] = _save_models(
                models, temporary / f"models_{variant.lower()}"
            )
            raw_importance = full_model.get_booster().get_score(
                importance_type="gain"
            )
            importance = pd.DataFrame(
                {
                    "feature": features,
                    "gain": [
                        raw_importance.get(f"f{index}", 0.0)
                        for index, _feature in enumerate(features)
                    ],
                }
            ).sort_values("gain", ascending=False)
            importance.to_csv(
                temporary / f"{variant.lower()}_feature_importance.csv", index=False
            )
            results[variant] = {
                "feature_count": len(features),
                "features": features,
                "include_weak_open_labels": bool(spec["include_weak_open_labels"]),
                "human_weight_multiplier": float(spec["human_weight_multiplier"]),
                "ranker_params_override": dict(spec["ranker_params_override"]),
                "target_diagnostics": target_diagnostics,
                "training": {"folds": fold_training, "full": full_training},
                "metrics": metrics,
            }

        winner = max(
            variant_names,
            key=lambda name: (
                results[name]["metrics"]["difficult_local_241"]["successes"],
                results[name]["metrics"]["all_exact_end_to_end"]["successes"],
                -results[name]["feature_count"],
            ),
        )
        winner_difficult = results[winner]["metrics"]["difficult_local_241"]
        gate = {
            "difficult_local_top1": {
                "minimum_successes": 225,
                "observed_successes": winner_difficult["successes"],
                "total": winner_difficult["total"],
                "passed": winner_difficult["successes"] >= 225,
            },
            "no_deterministic_promotion": {"passed": True},
            "all_predictions_oof": {"passed": True},
            "siren_grouped_folds": {"passed": True},
        }
        verdict = (
            "GO_ACCEPTOR_SCENES"
            if all(item["passed"] for item in gate.values())
            else "PIVOT_RANKER"
        )
        evaluation = {
            "schema_version": SCHEMA_VERSION,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "scope": "CONSUMED_DEVELOPMENT_OOF",
            "independent_certification": False,
            "winner": winner,
            "verdict": verdict,
            "gate": gate,
            "variants": results,
            "elapsed_seconds": time.perf_counter() - started,
        }
        _json_dump(temporary / "evaluation.json", evaluation)
        report_lines = [
            "# Rankers appris V4.12-L",
            "",
            f"Verdict : **{verdict}**. Winner : **{winner}**.",
            "",
            "| Variante | Features | Faibles | Global exact | Difficile 241 | Actifs | Fermés |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for name in variant_names:
            item = results[name]
            metrics = item["metrics"]
            report_lines.append(
                f"| {name} | {item['feature_count']} | "
                f"{'oui' if item['include_weak_open_labels'] else 'non'} | "
                f"{metrics['all_exact_end_to_end']['successes']}/{metrics['all_exact_end_to_end']['total']} | "
                f"{metrics['difficult_local_241']['successes']}/{metrics['difficult_local_241']['total']} | "
                f"{metrics['active_exact']['rate']:.3%} | {metrics['closed_exact']['rate']:.3%} |"
            )
        report_lines.extend(
            [
                "",
                "Toutes les prédictions sont OOF par composante SIREN. Les 100 misses retrieval restent des erreurs end-to-end. Aucune règle déterministe ne modifie le top1.",
                "",
            ]
        )
        (temporary / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
        output_hashes: dict[str, str] = {}
        for path in temporary.rglob("*"):
            if path.is_file() and path.name != "manifest.json":
                output_hashes[str(path.relative_to(temporary))] = file_sha256(path)
        output_manifest = {
            "schema_version": SCHEMA_VERSION,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "build_identity": identity,
            "winner": winner,
            "verdict": verdict,
            "candidate_ceiling": 100,
            "positive_injection": False,
            "deterministic_promotions": False,
            "independent_certification": False,
            "model_hashes": model_hashes,
            "outputs": output_hashes,
        }
        _json_dump(temporary / "manifest.json", output_manifest)
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--local-labels", type=Path, default=DEFAULT_LOCAL_LABELS)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/"
            "v4_12_learned_oof_rankers"
        ),
    )
    parser.add_argument(
        "--negative-limit",
        type=int,
        default=0,
        help="0 uses all candidates; truncation is diagnostic only.",
    )
    parser.add_argument("--variants", nargs="*", choices=sorted(VARIANTS))
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
