#!/usr/bin/env python3
"""Development-only V4.12 ranker + selective acceptor experiment.

The ranker is regenerated jointly out of fold for historical fit queries and
the 83 adjudicated hard queries.  Acceptor thresholds are selected on the
frozen threshold partition, compared on the frozen comparison partition, and
only then applied to the seven-query independent ranker docket.
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
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_v411_acceptor_dataset import (  # noqa: E402
    _dev_partition,
    build_scene_frame,
)
from scripts.evaluate_v412_hard_label_ranker import (  # noqa: E402
    RANKER_C_FEATURE_ORDER,
    eligible_ranker_rows,
    fit_weighted_ranker,
    load_hard_labels,
)
from scripts.run_v411_acceptor_development import (  # noqa: E402
    EXPECTED_MODEL_CONFIGS,
    decision_metrics,
    select_threshold,
)
from src.xgb_matcher.v411_acceptor import (  # noqa: E402
    V411_ACCEPTOR_FAMILIES,
    build_v411_acceptor,
)
from src.xgb_matcher.v411_scene import V411_ACCEPTOR_FEATURE_NAMES  # noqa: E402
from src.xgb_matcher.v49_site_function import SiteFunctionTaxonomy  # noqa: E402


BASE = Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100")
DEFAULT_DATASET = BASE / "datasets/v4_11_input_blind/ec4326ec57e4411d"
DEFAULT_RANKER = (
    BASE
    / "experiments/v4_12_hard_label_ranker/bba02575366ebe80/ranker_candidate.json"
)
DEFAULT_OUTPUT_ROOT = BASE / "experiments/v4_12_ranker_acceptor_stack"
SCHEMA_VERSION = "sireto-v4.12-ranker-acceptor-stack-development-2"
HARD_WEIGHT = 0.5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rank(model: xgb.XGBRanker, rows: pd.DataFrame, origin: str) -> pd.DataFrame:
    output = rows.copy()
    output["ranker_score"] = model.predict(
        output[RANKER_C_FEATURE_ORDER].to_numpy(dtype=np.float32)
    ).astype(np.float32)
    output["prediction_origin"] = origin
    output = output.sort_values(
        ["query_id", "ranker_score", "retrieval_rank", "candidate_siret"],
        ascending=[True, False, True, True],
        kind="mergesort",
    )
    output["ranker_rank"] = output.groupby("query_id", sort=False).cumcount() + 1
    return output


def _load_adjudications(
    r30_path: Path,
    r53_path: Path,
) -> tuple[pd.DataFrame, set[str]]:
    exact, all_ids, _ = load_hard_labels(r30_path, r53_path)
    r30 = pd.read_csv(r30_path, dtype=str).fillna("")
    r53 = pd.read_csv(r53_path, dtype=str).fillna("")
    ambiguous_ids = set(r30.loc[r30["label"].eq("AMBIGUOUS"), "query_id"]) | set(
        r53.loc[r53["label"].eq("AMBIGUOUS"), "query_id"]
    )
    rows = exact.assign(label_kind="MATCH_EXACT")
    ambiguous = pd.DataFrame(
        {
            "query_id": sorted(ambiguous_ids),
            "ground_truth_siret": None,
            "ground_truth_siren": None,
            "label_kind": "AMBIGUOUS",
        }
    )
    output = pd.concat([rows, ambiguous], ignore_index=True)
    if len(output) != 83 or set(output["query_id"]) != all_ids:
        raise ValueError("Expected 77 exact and six ambiguous hard adjudications")
    return output, all_ids


def _load_independent(path: Path) -> pd.DataFrame:
    labels = pd.read_csv(path, dtype=str).fillna("")
    if (
        len(labels) != 7
        or labels["query_id"].duplicated().any()
        or labels["label"].value_counts().to_dict()
        != {"MATCH_EXACT": 6, "AMBIGUOUS": 1}
    ):
        raise ValueError("Independent docket labels changed")
    labels = labels.rename(
        columns={"label": "label_kind", "validated_siret": "ground_truth_siret"}
    )
    labels["ground_truth_siret"] = labels["ground_truth_siret"].replace("", None)
    labels["ground_truth_siren"] = labels["ground_truth_siret"].map(
        lambda value: str(value)[:9] if value else None
    )
    return labels[
        ["query_id", "label_kind", "ground_truth_siret", "ground_truth_siren"]
    ]


def _score_acceptor(model: Any, frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(
        model.predict_proba(
            frame[V411_ACCEPTOR_FEATURE_NAMES].to_numpy(dtype=np.float64)
        )[:, 1],
        dtype=np.float64,
    )


def _fit_acceptor_reproducibly(
    family: str,
    fit: pd.DataFrame,
    scored: list[pd.DataFrame],
) -> tuple[Any, list[np.ndarray]]:
    outputs: list[np.ndarray] | None = None
    first = None
    x = fit[V411_ACCEPTOR_FEATURE_NAMES].to_numpy(dtype=np.float64)
    y = fit["acceptor_target"].astype(int).to_numpy()
    for repetition in range(2):
        model = build_v411_acceptor(family, EXPECTED_MODEL_CONFIGS[family])
        model.fit(x, y)
        current = [_score_acceptor(model, frame) for frame in scored]
        if repetition == 0:
            first = model
            outputs = current
        elif outputs is None or any(
            not np.array_equal(left, right)
            for left, right in zip(outputs, current, strict=True)
        ):
            raise ValueError(f"Acceptor {family} is not reproducible")
    if first is None or outputs is None:
        raise AssertionError("Acceptor training did not execute")
    return first, outputs


def run(args: argparse.Namespace) -> Path:
    dataset = args.dataset.resolve()
    queries = pd.read_parquet(dataset / "queries.parquet")
    audit = pd.read_parquet(dataset / "query_audit.parquet")
    labels = pd.read_parquet(dataset / "labels.parquet")
    assignments = pd.read_parquet(dataset / "split_assignments.parquet")
    candidates = pd.read_parquet(dataset / "candidates_sparse_top100.parquet")
    for frame in (queries, audit, labels, assignments, candidates):
        frame["query_id"] = frame["query_id"].astype(str)

    hard, hard_ids = _load_adjudications(args.r30_labels, args.r53_labels)
    independent = _load_independent(args.independent_labels)
    independent_ids = set(independent["query_id"])
    if hard_ids & independent_ids:
        raise ValueError("Independent docket overlaps hard training adjudications")

    population = (
        queries.merge(audit, on="query_id", validate="one_to_one")
        .merge(labels, on="query_id", validate="one_to_one")
        .merge(assignments, on="query_id", validate="one_to_one")
    )
    overrides = pd.concat([hard, independent], ignore_index=True).set_index("query_id")
    indexed = population.set_index("query_id")
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
    base_candidates = candidates[
        candidates["query_id"].isin(fit_population["query_id"])
    ].copy()
    base_rows = eligible_ranker_rows(base_candidates, fit_population)

    hard_population = population[population["query_id"].isin(hard_ids)]
    hard_candidates = candidates[candidates["query_id"].isin(hard_ids)].copy()
    hard_candidates = hard_candidates.join(
        hard_population.set_index("query_id")[["oof_fold"]], on="query_id"
    )
    hard_positive_counts = hard_candidates.groupby("query_id")["is_ground_truth"].sum()
    eligible_hard_ids = set(
        hard_positive_counts[hard_positive_counts.eq(1)].index.astype(str)
    )

    ranked_parts: list[pd.DataFrame] = []
    for fold in range(5):
        base_train = base_rows[
            ~base_rows["query_id"].isin(
                set(
                    fit_population.loc[
                        fit_population["oof_fold"].astype(int).eq(fold), "query_id"
                    ]
                )
            )
        ]
        hard_train = hard_candidates[
            hard_candidates["query_id"].isin(eligible_hard_ids)
            & hard_candidates["oof_fold"].astype(int).ne(fold)
        ]
        train = pd.concat([base_train, hard_train], ignore_index=True)
        model = fit_weighted_ranker(
            train,
            hard_query_ids=set(hard_train["query_id"]),
            hard_weight=HARD_WEIGHT,
        )
        scored_ids = set(
            fit_population.loc[
                fit_population["oof_fold"].astype(int).eq(fold), "query_id"
            ]
        ) | set(
            hard_population.loc[
                hard_population["oof_fold"].astype(int).eq(fold), "query_id"
            ]
        )
        ranked_parts.append(
            _rank(
                model,
                candidates[candidates["query_id"].isin(scored_ids)].copy(),
                "ranker_v412_joint_oof",
            )
        )

    dev_remaining_ids = set(population.loc[population["split"].eq("dev"), "query_id"]) - hard_ids
    full_ranker = xgb.XGBRanker()
    full_ranker.load_model(args.ranker_model)
    ranked_parts.append(
        _rank(
            full_ranker,
            candidates[candidates["query_id"].isin(dev_remaining_ids)].copy(),
            "ranker_v412_full_fit",
        )
    )
    ranked = pd.concat(ranked_parts, ignore_index=True)
    if (
        len(ranked) != len(candidates)
        or ranked.duplicated(["query_id", "candidate_siret"]).any()
        or ranked["query_id"].nunique() != len(population)
    ):
        raise ValueError("Joint OOF/full-fit ranker predictions are incomplete")

    taxonomy = SiteFunctionTaxonomy.load(args.taxonomy)
    scenes = build_scene_frame(population, ranked, taxonomy)
    fit_scene = scenes[
        (scenes["split"].eq("fit") | scenes["query_id"].isin(hard_ids))
        & scenes["label_kind"].isin(["MATCH_EXACT", "AMBIGUOUS"])
    ].copy()
    development = scenes[
        scenes["split"].eq("dev")
        & ~scenes["query_id"].isin(hard_ids | independent_ids)
        & scenes["label_kind"].isin(["MATCH_EXACT", "AMBIGUOUS"])
    ].copy()
    threshold = development[development["dev_partition"].eq("threshold_dev")].copy()
    comparison = development[development["dev_partition"].eq("comparison_dev")].copy()
    independent_scene = scenes[scenes["query_id"].isin(independent_ids)].copy()
    if len(independent_scene) != 7 or len(fit_scene) != 5630:
        raise ValueError("Acceptor populations changed unexpectedly")

    variants: list[dict[str, Any]] = []
    models: dict[str, Any] = {}
    independent_scores: dict[str, np.ndarray] = {}
    for family in V411_ACCEPTOR_FAMILIES:
        model, scores = _fit_acceptor_reproducibly(
            family, fit_scene, [threshold, comparison, independent_scene]
        )
        threshold_scores, comparison_scores, final_scores = scores
        selected = select_threshold(
            threshold_scores,
            threshold["acceptor_target"].astype(int).to_numpy(),
            threshold["label_kind"].astype(str).to_numpy(),
        )
        if selected is None:
            variants.append({"family": family, "eligible": False, "reason": "NO_THRESHOLD"})
            continue
        cutoff, threshold_metrics, _ = selected
        comparison_metrics = decision_metrics(
            comparison_scores,
            comparison["acceptor_target"].astype(int).to_numpy(),
            comparison["label_kind"].astype(str).to_numpy(),
            cutoff,
        )
        eligible = (
            comparison_metrics["auto_count"] > 0
            and 1000 * comparison_metrics["correct_auto"]
            >= 998 * comparison_metrics["auto_count"]
            and comparison_metrics["ambiguous_auto"] == 0
        )
        variants.append(
            {
                "family": family,
                "eligible": eligible,
                "threshold": cutoff,
                "threshold_metrics": threshold_metrics,
                "comparison_metrics": comparison_metrics,
            }
        )
        models[family] = model
        independent_scores[family] = final_scores

    eligible_variants = [variant for variant in variants if variant.get("eligible")]
    winner = None
    if eligible_variants:
        winner = sorted(
            eligible_variants,
            key=lambda variant: (
                -int(variant["comparison_metrics"]["auto_count"]),
                int(variant["comparison_metrics"]["error_auto"]),
                str(variant["family"]),
            ),
        )[0]

    final_metrics = None
    final_detail = independent_scene[
        [
            "query_id",
            "label_kind",
            "ground_truth_siret",
            "predicted_siret",
            "acceptor_target",
        ]
    ].copy()
    if winner is not None:
        family = str(winner["family"])
        cutoff = float(winner["threshold"])
        final_detail["acceptor_score"] = independent_scores[family]
        final_detail["decision"] = np.where(
            final_detail["acceptor_score"].ge(cutoff), "AUTO_MATCH", "REVIEW"
        )
        final_metrics = decision_metrics(
            independent_scores[family],
            independent_scene["acceptor_target"].astype(int).to_numpy(),
            independent_scene["label_kind"].astype(str).to_numpy(),
            cutoff,
        )

    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "CONSUMED_DEVELOPMENT_PLUS_SEVEN_CASE_INDEPENDENT_DOCKET",
        "final_test_opened": False,
        "production_promotion_authorized": False,
        "ranker": {
            "hard_weight": HARD_WEIGHT,
            "joint_oof_query_count": int(
                fit_population["query_id"].nunique() + len(hard_ids)
            ),
            "hard_retrieval_miss_count": 83 - len(eligible_hard_ids) - 6,
            "full_model_sha256": _sha256(args.ranker_model),
        },
        "populations": {
            "acceptor_fit": len(fit_scene),
            "threshold": len(threshold),
            "comparison": len(comparison),
            "independent": len(independent_scene),
        },
        "variants": variants,
        "selected_family": None if winner is None else winner["family"],
        "independent_metrics": final_metrics,
        "verdict": (
            "GO_EXPAND_INDEPENDENT_ACCEPTOR_VALIDATION"
            if final_metrics is not None
            and final_metrics["auto_count"] > 0
            and final_metrics["error_auto"] == 0
            and final_metrics["ambiguous_auto"] == 0
            else "PIVOT_ACCEPTOR_COVERAGE"
        ),
    }
    identity = hashlib.sha256(
        json.dumps(
            {
                "schema": SCHEMA_VERSION,
                "dataset": _sha256(dataset / "manifest.json"),
                "ranker": _sha256(args.ranker_model),
                "r30": _sha256(args.r30_labels),
                "r53": _sha256(args.r53_labels),
                "independent": _sha256(args.independent_labels),
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
    final_detail.to_parquet(output / "independent_decisions.parquet", index=False)
    if winner is not None:
        joblib.dump(models[str(winner["family"])], output / "acceptor_candidate.joblib")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--ranker-model", type=Path, default=DEFAULT_RANKER)
    parser.add_argument("--r30-labels", type=Path, default=Path("reports/v412_review_adjudication_labels.csv"))
    parser.add_argument("--r53-labels", type=Path, default=Path("reports/v412_review_rerank_counteraudit_53.csv"))
    parser.add_argument("--independent-labels", type=Path, default=Path("reports/v412_ranker_independent_validation_labels.csv"))
    parser.add_argument("--taxonomy", type=Path, default=Path("config/v4_9_site_function_taxonomy.json"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
