#!/usr/bin/env python3
"""Local zero-shot cross-encoder ablation on the stage-1 top 20 candidates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate_v412_hard_label_ranker import (
    _evaluate_predictions,
    fit_weighted_ranker,
)
from scripts.evaluate_v412_ranker_business_features import (
    BASE,
    DEFAULT_DATASET,
    DEFAULT_ETABLISSEMENTS,
    DEFAULT_REFERENCE,
    DEFAULT_TRUSTED,
    DEFAULT_UNITES_LEGALES,
    _read_enriched_sources,
)
from scripts.run_v411_ranker_c_development import (
    RANKER_C_FEATURE_ORDER,
    eligible_ranker_rows,
)


DEFAULT_RANKER = (
    BASE
    / "experiments/v4_12_trusted_label_ranker/2f57628196fefce0/ranker_candidate.json"
)
DEFAULT_MODEL = Path(
    "/Volumes/CATNAT_DATA/SIRETO_RECALL100/models/huggingface/hub/"
    "models--cross-encoder--mmarco-mMiniLMv2-L12-H384-v1/snapshots/"
    "1427fd652930e4ba29e8149678df786c240d8825"
)
DEFAULT_OUTPUT_ROOT = BASE / "experiments/v4_12_cross_encoder_reranker"
DEFAULT_CACHE = BASE / "cache/v4_12_cross_encoder_top20.parquet"
ALPHAS = (0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _score_stage1(model: xgb.XGBRanker, rows: pd.DataFrame) -> pd.DataFrame:
    output = rows.copy()
    output["stage1_score"] = model.predict(
        output[RANKER_C_FEATURE_ORDER].to_numpy(dtype=np.float32)
    ).astype("float32")
    output = output.sort_values(
        ["query_id", "stage1_score", "retrieval_rank", "candidate_siret"],
        ascending=[True, False, True, True],
        kind="mergesort",
    )
    output["stage1_rank"] = output.groupby("query_id", sort=False).cumcount() + 1
    return output


def _cross_fitted_stage1(
    frame: pd.DataFrame,
    population: pd.DataFrame,
    trusted_rows: pd.DataFrame,
    eligible_trusted: set[str],
    hard_weight: float,
) -> pd.DataFrame:
    fit_population = population[population["split"].eq("fit")].copy()
    base_rows = eligible_ranker_rows(
        frame[frame["query_id"].isin(fit_population["query_id"])], fit_population
    )
    parts: list[pd.DataFrame] = []
    for fold in range(5):
        held_base_ids = set(
            fit_population.loc[
                fit_population["oof_fold"].astype(int).eq(fold), "query_id"
            ].astype(str)
        )
        base_train = base_rows[~base_rows["query_id"].isin(held_base_ids)]
        hard_train = trusted_rows[
            trusted_rows["query_id"].isin(eligible_trusted)
            & trusted_rows["oof_fold"].ne(fold)
        ]
        model = fit_weighted_ranker(
            pd.concat([base_train, hard_train], ignore_index=True),
            hard_query_ids=set(hard_train["query_id"].astype(str)),
            hard_weight=hard_weight,
        )
        parts.append(
            _score_stage1(model, trusted_rows[trusted_rows["oof_fold"].eq(fold)])
        )
    return pd.concat(parts, ignore_index=True)


def _serialise_pairs(rows: pd.DataFrame) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for row in rows.itertuples(index=False):
        query = (
            f"nom: {row.crm_name}. adresse: {row.crm_address}, "
            f"{row.crm_postcode} {row.crm_city}."
        )
        names = [
            row.source_ul_name,
            row.source_ul_sigle,
            row.source_ul_usual1,
            row.source_ul_usual2,
            row.source_ul_usual3,
            row.source_enseigne1,
            row.source_enseigne2,
            row.source_enseigne3,
            row.source_etab_usual,
        ]
        name_text = " ; ".join(
            str(value) for value in names if pd.notna(value) and str(value).strip()
        )
        address = " ".join(
            str(value)
            for value in (
                row.raw_street_number,
                row.raw_street_type,
                row.raw_street_name,
                row.raw_postcode,
            )
            if pd.notna(value) and str(value).strip()
        )
        candidate = (
            f"nom légal, sigle et enseignes: {name_text}. adresse: {address}. "
            f"activité NAF: {row.source_etab_activity}. siège: {row.is_siege}."
        )
        pairs.append((query, candidate))
    return pairs


def _cross_encoder_scores(rows: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    keys = ["query_id", "candidate_siret"]
    cache = pd.DataFrame(columns=[*keys, "cross_encoder_score"])
    if args.cache.exists():
        cache = pd.read_parquet(args.cache)
        for column in keys:
            cache[column] = cache[column].astype(str)
    merged = rows[keys].merge(cache, on=keys, how="left", validate="one_to_one")
    missing_mask = merged["cross_encoder_score"].isna()
    if missing_mask.any():
        missing_indices = np.flatnonzero(missing_mask.to_numpy())
        missing_rows = rows.iloc[missing_indices]
        model = CrossEncoder(str(args.model.resolve()), device=args.device)
        scores = model.predict(
            _serialise_pairs(missing_rows),
            batch_size=args.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype("float32")
        additions = missing_rows[keys].copy()
        additions["cross_encoder_score"] = scores
        cache = (
            pd.concat([cache, additions], ignore_index=True)
            .drop_duplicates(keys, keep="last")
            .sort_values(keys, kind="mergesort")
        )
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        cache.to_parquet(args.cache, index=False)
        merged = rows[keys].merge(cache, on=keys, how="left", validate="one_to_one")
    if merged["cross_encoder_score"].isna().any():
        raise ValueError("Cross-encoder cache remains incomplete")
    return merged


def _normalise_within_query(rows: pd.DataFrame, column: str) -> pd.Series:
    grouped = rows.groupby("query_id", sort=False)[column]
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0.0, np.nan)
    return ((rows[column] - mean) / std).fillna(0.0).astype("float32")


def _predictions(rows: pd.DataFrame, alpha: float) -> pd.DataFrame:
    output = rows[["query_id", "candidate_siret", "candidate_siren", "retrieval_rank"]].copy()
    output["ranker_score"] = rows["stage1_z"] + alpha * rows["cross_encoder_z"]
    output = output.sort_values(
        ["query_id", "ranker_score", "retrieval_rank", "candidate_siret"],
        ascending=[True, False, True, True],
        kind="mergesort",
    )
    output["ranker_rank"] = output.groupby("query_id", sort=False).cumcount() + 1
    return output.reset_index(drop=True)


def run(args: argparse.Namespace) -> Path:
    dataset = args.dataset.resolve()
    frame = _read_enriched_sources(dataset, args.etablissements, args.unites_legales)
    queries = pd.read_parquet(dataset / "queries.parquet")
    labels = pd.read_parquet(dataset / "labels.parquet")
    assignments = pd.read_parquet(dataset / "split_assignments.parquet")
    reference = pd.read_parquet(args.reference.resolve() / "ranker_reference.parquet")
    trusted = pd.read_csv(args.trusted_labels, dtype=str).fillna("")
    for value in (frame, queries, labels, assignments, reference, trusted):
        value["query_id"] = value["query_id"].astype(str)
    frame = frame.merge(
        queries[["query_id", "crm_address", "crm_postcode", "crm_city"]],
        on="query_id",
        validate="many_to_one",
    )
    population = labels.merge(assignments, on="query_id", validate="one_to_one")
    trusted_truth = trusted[trusted["label_kind"].eq("MATCH_EXACT")][
        ["query_id", "ground_truth_siret"]
    ].merge(
        assignments[["query_id", "oof_fold", "siren_component_id", "split"]],
        on="query_id",
        validate="one_to_one",
    )
    trusted_truth["ground_truth_siren"] = trusted_truth["ground_truth_siret"].str[:9]
    trusted_truth["oof_fold"] = trusted_truth["oof_fold"].astype(int)
    trusted_rows = frame[frame["query_id"].isin(trusted_truth["query_id"])].drop(
        columns=["is_ground_truth"]
    ).merge(
        trusted_truth[["query_id", "ground_truth_siret", "oof_fold"]],
        on="query_id",
        validate="many_to_one",
    )
    trusted_rows["is_ground_truth"] = trusted_rows["candidate_siret"].eq(
        trusted_rows["ground_truth_siret"]
    ).astype("int8")
    positive_counts = trusted_rows.groupby("query_id")["is_ground_truth"].sum()
    eligible_trusted = set(positive_counts[positive_counts.eq(1)].index.astype(str))
    trusted_scored = _cross_fitted_stage1(
        frame, population, trusted_rows, eligible_trusted, args.hard_weight
    )

    trusted_ids = set(trusted["query_id"])
    trusted_components = set(trusted_truth["siren_component_id"].astype(str))
    control_truth = population[
        population["split"].eq("dev")
        & population["label_kind"].eq("MATCH_EXACT")
        & ~population["query_id"].isin(trusted_ids)
        & ~population["siren_component_id"].astype(str).isin(trusted_components)
    ][["query_id", "ground_truth_siret", "ground_truth_siren"]]
    full_ranker = xgb.XGBRanker()
    full_ranker.load_model(args.ranker.resolve())
    control_scored = _score_stage1(
        full_ranker, frame[frame["query_id"].isin(control_truth["query_id"])]
    )

    top = pd.concat(
        [
            trusted_scored[trusted_scored["stage1_rank"].le(args.top_n)],
            control_scored[control_scored["stage1_rank"].le(args.top_n)],
        ],
        ignore_index=True,
    )
    cache_scores = _cross_encoder_scores(top, args)
    top["cross_encoder_score"] = cache_scores["cross_encoder_score"].to_numpy()
    top["stage1_z"] = _normalise_within_query(top, "stage1_score")
    top["cross_encoder_z"] = _normalise_within_query(top, "cross_encoder_score")

    variants: list[dict[str, object]] = []
    details: dict[float, pd.DataFrame] = {}
    for alpha in ALPHAS:
        predictions = _predictions(top, alpha)
        trusted_metrics, trusted_detail = _evaluate_predictions(
            predictions[predictions["query_id"].isin(trusted_truth["query_id"])],
            trusted_truth,
        )
        control_metrics, control_detail = _evaluate_predictions(
            predictions[predictions["query_id"].isin(control_truth["query_id"])],
            control_truth,
        )
        variants.append(
            {
                "alpha": alpha,
                "trusted": trusted_metrics,
                "control": control_metrics,
                "eligible": control_metrics["top1_correct_count"] == len(control_truth),
            }
        )
        details[alpha] = trusted_detail
    eligible = [item for item in variants if item["eligible"]]
    winner = sorted(
        eligible,
        key=lambda item: (-item["trusted"]["top1_correct_count"], item["alpha"]),
    )[0]
    result = {
        "schema_version": "sireto-v4.12-local-cross-encoder-reranker-development-1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "CONSUMED_DEVELOPMENT_ONLY",
        "final_test_opened": False,
        "model_path": str(args.model.resolve()),
        "model_sha256": _sha256(args.model.resolve() / "model.safetensors"),
        "top_n": args.top_n,
        "hard_weight": args.hard_weight,
        "variants": variants,
        "selected": winner,
    }
    identity = hashlib.sha256(
        json.dumps(
            {
                "schema": result["schema_version"],
                "dataset": _sha256(dataset / "manifest.json"),
                "trusted": _sha256(args.trusted_labels),
                "model": result["model_sha256"],
                "top_n": args.top_n,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    output = args.output_root.resolve() / identity
    output.mkdir(parents=True, exist_ok=False)
    (output / "evaluation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    details[float(winner["alpha"])].to_parquet(
        output / "trusted_selected_comparison.parquet", index=False
    )
    top[["query_id", "candidate_siret", "stage1_rank", "stage1_score", "cross_encoder_score"]].to_parquet(
        output / "top20_scores.parquet", index=False
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--trusted-labels", type=Path, default=DEFAULT_TRUSTED)
    parser.add_argument("--etablissements", type=Path, default=DEFAULT_ETABLISSEMENTS)
    parser.add_argument("--unites-legales", type=Path, default=DEFAULT_UNITES_LEGALES)
    parser.add_argument("--ranker", type=Path, default=DEFAULT_RANKER)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--hard-weight", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="mps")
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
