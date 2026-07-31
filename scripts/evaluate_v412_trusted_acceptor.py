#!/usr/bin/env python3
"""Build trusted ranker OOF scenes and tune the selective acceptor on dev."""

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
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_v411_acceptor_dataset import _dev_partition, build_scene_frame
from scripts.evaluate_v412_acceptor_hard_weight import _fit, _scores
from scripts.evaluate_v412_hard_label_ranker import fit_weighted_ranker
from scripts.evaluate_v412_ranker_acceptor_stack import _rank
from scripts.run_v411_acceptor_development import decision_metrics, select_threshold
from scripts.run_v411_ranker_c_development import eligible_ranker_rows
from src.xgb_matcher.v411_acceptor import MONOTONIC_XGB
from src.xgb_matcher.v49_site_function import SiteFunctionTaxonomy


BASE = Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100")
DEFAULT_DATASET = BASE / "datasets/v4_11_input_blind/ec4326ec57e4411d"
DEFAULT_RANKER = (
    BASE
    / "experiments/v4_12_trusted_label_ranker/2f57628196fefce0/ranker_candidate.json"
)
DEFAULT_OUTPUT_ROOT = BASE / "experiments/v4_12_trusted_acceptor"
DEFAULT_TRUSTED = Path("reports/v412_review_trusted_labels_279.csv")
SCHEMA_VERSION = "sireto-v4.12-trusted-acceptor-development-1"
RANKER_WEIGHT = 0.5
ACCEPTOR_FAMILY = MONOTONIC_XGB
ACCEPTOR_WEIGHT = 10.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> Path:
    dataset = args.dataset.resolve()
    queries = pd.read_parquet(dataset / "queries.parquet")
    audit = pd.read_parquet(dataset / "query_audit.parquet")
    labels = pd.read_parquet(dataset / "labels.parquet")
    assignments = pd.read_parquet(dataset / "split_assignments.parquet")
    candidates = pd.read_parquet(dataset / "candidates_sparse_top100.parquet")
    for frame in (queries, audit, labels, assignments, candidates):
        frame["query_id"] = frame["query_id"].astype(str)

    trusted = pd.read_csv(args.trusted_labels, dtype=str).fillna("")
    if len(trusted) != 279 or trusted["label_kind"].value_counts().to_dict() != {
        "MATCH_EXACT": 254,
        "AMBIGUOUS": 25,
    }:
        raise ValueError("Trusted REVIEW labels changed")
    trusted["ground_truth_siren"] = trusted["ground_truth_siret"].map(
        lambda value: value[:9] if value else None
    )
    trusted_ids = set(trusted["query_id"].astype(str))

    population = (
        queries.merge(audit, on="query_id", validate="one_to_one")
        .merge(labels, on="query_id", validate="one_to_one")
        .merge(assignments, on="query_id", validate="one_to_one")
    )
    indexed = population.set_index("query_id")
    overrides = trusted.set_index("query_id")
    for column in ("label_kind", "ground_truth_siret", "ground_truth_siren"):
        indexed.loc[overrides.index, column] = overrides[column]
    population = indexed.reset_index()
    population["dev_partition"] = ""
    dev = population["split"].eq("dev")
    population.loc[dev, "dev_partition"] = population.loc[
        dev, "siren_component_id"
    ].astype(str).map(_dev_partition)

    truth = population.set_index("query_id")[["label_kind", "ground_truth_siret"]]
    candidates = candidates.drop(columns=["is_ground_truth"]).join(truth, on="query_id")
    candidates["is_ground_truth"] = (
        candidates["label_kind"].eq("MATCH_EXACT")
        & candidates["candidate_siret"].astype(str).eq(
            candidates["ground_truth_siret"].fillna("").astype(str)
        )
    ).astype(np.int8)
    candidates = candidates.drop(columns=["label_kind", "ground_truth_siret"])

    fit_population = population[population["split"].eq("fit")]
    base_rows = eligible_ranker_rows(
        candidates[candidates["query_id"].isin(fit_population["query_id"])],
        fit_population,
    )
    trusted_population = population[population["query_id"].isin(trusted_ids)]
    trusted_candidates = candidates[candidates["query_id"].isin(trusted_ids)].copy()
    trusted_candidates = trusted_candidates.join(
        trusted_population.set_index("query_id")[["oof_fold"]], on="query_id"
    )
    counts = trusted_candidates.groupby("query_id")["is_ground_truth"].sum()
    eligible_trusted_ids = set(counts[counts.eq(1)].index.astype(str))
    if len(eligible_trusted_ids) != 251:
        raise ValueError("Trusted retrieval presence changed")

    ranked_parts: list[pd.DataFrame] = []
    for fold in range(5):
        base_train = base_rows[
            ~base_rows["query_id"].isin(
                set(
                    fit_population.loc[
                        fit_population["oof_fold"].astype(int).eq(fold), "query_id"
                    ].astype(str)
                )
            )
        ]
        trusted_train = trusted_candidates[
            trusted_candidates["query_id"].isin(eligible_trusted_ids)
            & trusted_candidates["oof_fold"].astype(int).ne(fold)
        ]
        ranker = fit_weighted_ranker(
            pd.concat([base_train, trusted_train], ignore_index=True),
            hard_query_ids=set(trusted_train["query_id"].astype(str)),
            hard_weight=RANKER_WEIGHT,
        )
        scored_ids = set(
            fit_population.loc[
                fit_population["oof_fold"].astype(int).eq(fold), "query_id"
            ].astype(str)
        ) | set(
            trusted_population.loc[
                trusted_population["oof_fold"].astype(int).eq(fold), "query_id"
            ].astype(str)
        )
        ranked_parts.append(
            _rank(
                ranker,
                candidates[candidates["query_id"].isin(scored_ids)].copy(),
                "trusted_ranker_oof",
            )
        )

    already_oof = set(fit_population["query_id"].astype(str)) | trusted_ids
    remaining_ids = set(population["query_id"].astype(str)) - already_oof
    full_ranker = xgb.XGBRanker()
    full_ranker.load_model(args.ranker_model)
    ranked_parts.append(
        _rank(
            full_ranker,
            candidates[candidates["query_id"].isin(remaining_ids)].copy(),
            "trusted_ranker_full_fit",
        )
    )
    ranked = pd.concat(ranked_parts, ignore_index=True)
    if (
        len(ranked) != len(candidates)
        or ranked.duplicated(["query_id", "candidate_siret"]).any()
        or ranked["query_id"].nunique() != len(population)
    ):
        raise ValueError("Trusted ranker predictions are incomplete")

    taxonomy = SiteFunctionTaxonomy.load(args.taxonomy)
    scenes = build_scene_frame(population, ranked, taxonomy)
    base_exact = scenes[
        scenes["split"].eq("fit") & scenes["label_kind"].eq("MATCH_EXACT")
    ].copy()
    trusted_scenes = scenes[scenes["query_id"].isin(trusted_ids)].copy()
    if len(base_exact) != 4666 or len(trusted_scenes) != 279:
        raise ValueError("Trusted acceptor scene populations changed")
    if trusted_scenes["acceptor_target"].value_counts().to_dict() != {1: 216, 0: 63}:
        raise ValueError("Trusted OOF ranker targets changed")

    threshold_mask = trusted_scenes["dev_partition"].eq("threshold_dev")
    variants: list[dict[str, Any]] = []
    oof_details: dict[tuple[str, float], pd.DataFrame] = {}
    for family in (ACCEPTOR_FAMILY,):
        for weight in (ACCEPTOR_WEIGHT,):
            held_parts: list[pd.DataFrame] = []
            score_parts: list[np.ndarray] = []
            for fold in range(5):
                fold_base = base_exact[base_exact["oof_fold"].astype(int).ne(fold)]
                fold_trusted = trusted_scenes[
                    trusted_scenes["oof_fold"].astype(int).ne(fold)
                ]
                fit = pd.concat([fold_base, fold_trusted], ignore_index=True)
                model = _fit(
                    fit,
                    set(fold_trusted["query_id"].astype(str)),
                    weight,
                    family,
                )
                held = trusted_scenes[
                    trusted_scenes["oof_fold"].astype(int).eq(fold)
                ].copy()
                held_parts.append(held)
                score_parts.append(_scores(model, held))
            detail = pd.concat(held_parts, ignore_index=True)
            detail["acceptor_score"] = np.concatenate(score_parts)
            threshold = detail[detail["dev_partition"].eq("threshold_dev")]
            comparison = detail[detail["dev_partition"].eq("comparison_dev")]
            selected = select_threshold(
                threshold["acceptor_score"].to_numpy(),
                threshold["acceptor_target"].astype(int).to_numpy(),
                threshold["label_kind"].astype(str).to_numpy(),
            )
            if selected is None:
                variants.append(
                    {"family": family, "hard_weight": weight, "eligible": False}
                )
                continue
            cutoff, threshold_metrics, _ = selected
            comparison_metrics = decision_metrics(
                comparison["acceptor_score"].to_numpy(),
                comparison["acceptor_target"].astype(int).to_numpy(),
                comparison["label_kind"].astype(str).to_numpy(),
                cutoff,
            )
            all_metrics = decision_metrics(
                detail["acceptor_score"].to_numpy(),
                detail["acceptor_target"].astype(int).to_numpy(),
                detail["label_kind"].astype(str).to_numpy(),
                cutoff,
            )
            eligible = (
                comparison_metrics["auto_count"] > 0
                and comparison_metrics["error_auto"] == 0
                and comparison_metrics["ambiguous_auto"] == 0
            )
            variants.append(
                {
                    "family": family,
                    "hard_weight": weight,
                    "threshold": cutoff,
                    "threshold_metrics": threshold_metrics,
                    "comparison_metrics": comparison_metrics,
                    "trusted_oof_metrics": all_metrics,
                    "eligible": eligible,
                }
            )
            oof_details[(family, weight)] = detail

    eligible = [variant for variant in variants if variant.get("eligible")]
    winner = (
        sorted(
            eligible,
            key=lambda variant: (
                -int(variant["comparison_metrics"]["auto_count"]),
                -int(variant["trusted_oof_metrics"]["auto_count"]),
                str(variant["family"]),
                float(variant["hard_weight"]),
            ),
        )[0]
        if eligible
        else None
    )
    final_model = None
    final_detail = pd.DataFrame()
    if winner is not None:
        family = str(winner["family"])
        weight = float(winner["hard_weight"])
        cutoff = float(winner["threshold"])
        final_model = _fit(
            pd.concat([base_exact, trusted_scenes], ignore_index=True),
            trusted_ids,
            weight,
            family,
        )
        final_detail = oof_details[(family, weight)][
            [
                "query_id",
                "label_kind",
                "ground_truth_siret",
                "predicted_siret",
                "acceptor_target",
                "dev_partition",
                "acceptor_score",
            ]
        ].copy()
        final_detail["decision"] = np.where(
            final_detail["acceptor_score"].ge(cutoff), "AUTO_MATCH", "REVIEW"
        )

    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "CONSUMED_DEVELOPMENT_ONLY",
        "final_test_opened": False,
        "production_promotion_authorized": False,
        "inputs": {
            "dataset_manifest_sha256": _sha256(dataset / "manifest.json"),
            "trusted_labels_sha256": _sha256(args.trusted_labels),
            "ranker_sha256": _sha256(args.ranker_model),
        },
        "populations": {
            "base_exact": len(base_exact),
            "trusted": len(trusted_scenes),
            "trusted_target_positive": int(trusted_scenes["acceptor_target"].sum()),
            "trusted_target_negative": int((~trusted_scenes["acceptor_target"].astype(bool)).sum()),
            "threshold": int(threshold_mask.sum()),
            "comparison": int((~threshold_mask).sum()),
        },
        "variants": variants,
        "selected": winner,
        "verdict": "GO_DEV_END_TO_END" if winner is not None else "PIVOT_ACCEPTOR",
    }
    identity = hashlib.sha256(
        json.dumps(
            {
                "schema": SCHEMA_VERSION,
                "dataset": result["inputs"]["dataset_manifest_sha256"],
                "trusted": result["inputs"]["trusted_labels_sha256"],
                "ranker": result["inputs"]["ranker_sha256"],
                "acceptor_family": ACCEPTOR_FAMILY,
                "acceptor_weight": ACCEPTOR_WEIGHT,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    output = args.output_root.resolve() / identity
    output.mkdir(parents=True, exist_ok=False)
    (output / "evaluation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    scenes.to_parquet(output / "acceptor_scenes.parquet", index=False)
    final_detail.to_parquet(output / "trusted_oof_decisions.parquet", index=False)
    if final_model is not None:
        joblib.dump(final_model, output / "acceptor_candidate.joblib")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--trusted-labels", type=Path, default=DEFAULT_TRUSTED)
    parser.add_argument("--ranker-model", type=Path, default=DEFAULT_RANKER)
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=Path("config/v4_9_site_function_taxonomy.json"),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
