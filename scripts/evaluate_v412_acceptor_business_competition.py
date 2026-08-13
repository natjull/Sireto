#!/usr/bin/env python3
"""Development-only acceptor ablation with explicit business competition.

The experiment consumes a complete candidate ranking when supplied.  Without
one it reproduces the current trusted-label ranker OOF predictions.  It never
opens the final test and does not alter the canonical V4.11 scene contract.
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

from scripts.build_v411_acceptor_dataset import _dev_partition, build_scene_frame
from scripts.evaluate_v412_hard_label_ranker import fit_weighted_ranker
from scripts.evaluate_v412_ranker_acceptor_stack import _rank
from scripts.evaluate_v412_ranker_business_features import (
    _read_enriched_sources,
    _relational_features,
    _source_features,
)
from scripts.run_v411_acceptor_development import (
    EXPECTED_MODEL_CONFIGS,
    select_threshold,
)
from scripts.run_v411_ranker_c_development import eligible_ranker_rows
from src.xgb_matcher.v411_scene import (
    V411_ACCEPTOR_FEATURE_NAMES,
    V411_MONOTONIC_CONSTRAINTS,
    rank_v411_candidates,
)
from src.xgb_matcher.v49_site_function import SiteFunctionTaxonomy
from src.xgb_matcher.v9_dataset import file_sha256


BASE = Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100")
DEFAULT_DATASET = BASE / "datasets/v4_11_input_blind/ec4326ec57e4411d"
DEFAULT_RANKER = (
    BASE
    / "experiments/v4_12_trusted_label_ranker/2f57628196fefce0/"
    "ranker_candidate.json"
)
DEFAULT_OUTPUT_ROOT = BASE / "experiments/v4_12_acceptor_business_competition"
DEFAULT_TRUSTED = Path("reports/v412_review_trusted_labels_279.csv")
DEFAULT_ETABLISSEMENTS = Path("data/StockEtablissement_utf8.parquet")
DEFAULT_UNITES_LEGALES = Path("data/StockUniteLegale_utf8.parquet")
SCHEMA_VERSION = "sireto-v4.12-acceptor-business-competition-development-1"
RANKER_WEIGHT = 0.5
WEIGHTS = (2.0, 5.0, 10.0)


BUSINESS_ACCEPTOR_FEATURES = [
    # Separate legal-name and establishment-name evidence.
    "top1_ul_name_exact_business",
    "delta_ul_name_exact_business",
    "top1_etab_name_exact_business",
    "delta_etab_name_exact_business",
    # Explicit competition from other legal entities at the same address.
    "top1_same_address_siren_count_business",
    "same_address_strong_name_siren_count_business",
    "same_address_exact_ul_siren_count_business",
    "same_address_exact_etab_siren_count_business",
    "ul_margin_other_siren_same_address_business",
    "etab_margin_other_siren_same_address_business",
    # Legal category and CRM wording.
    "top1_legal_association_business",
    "delta_legal_association_business",
    "top1_legal_public_business",
    "delta_legal_public_business",
    "top1_legal_company_business",
    "delta_legal_company_business",
    "top1_legal_role_match_business",
    "delta_legal_role_match_business",
    "top1_legal_role_conflict_business",
    # NAF/business role and establishment recency.
    "top1_business_role_net_business",
    "delta_business_role_net_business",
    "role_margin_other_siren_same_address_business",
    "start_year_margin_other_siren_same_address_business",
    "top1_newest_same_address_business",
]

ACCEPTOR_FEATURES = V411_ACCEPTOR_FEATURE_NAMES + BUSINESS_ACCEPTOR_FEATURES

# Preserve the canonical constraints and constrain only directions with an
# unambiguous evidential meaning.  Counts and dates remain unconstrained.
_POSITIVE = {
    "top1_ul_name_exact_business",
    "delta_ul_name_exact_business",
    "top1_etab_name_exact_business",
    "delta_etab_name_exact_business",
    "ul_margin_other_siren_same_address_business",
    "etab_margin_other_siren_same_address_business",
    "top1_legal_role_match_business",
    "delta_legal_role_match_business",
    "top1_business_role_net_business",
    "delta_business_role_net_business",
    "role_margin_other_siren_same_address_business",
}
_NEGATIVE = {
    "same_address_strong_name_siren_count_business",
    "same_address_exact_ul_siren_count_business",
    "same_address_exact_etab_siren_count_business",
    "top1_legal_role_conflict_business",
}
MONOTONIC_CONSTRAINTS = list(V411_MONOTONIC_CONSTRAINTS) + [
    1 if name in _POSITIVE else -1 if name in _NEGATIVE else 0
    for name in BUSINESS_ACCEPTOR_FEATURES
]


def _normalise_query_ids(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["query_id"] = frame["query_id"].astype(str)
    return frame


def _load_population(dataset: Path, trusted_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    queries = _normalise_query_ids(pd.read_parquet(dataset / "queries.parquet"))
    audit = _normalise_query_ids(pd.read_parquet(dataset / "query_audit.parquet"))
    labels = _normalise_query_ids(pd.read_parquet(dataset / "labels.parquet"))
    assignments = _normalise_query_ids(
        pd.read_parquet(dataset / "split_assignments.parquet")
    )
    trusted = pd.read_csv(trusted_path, dtype=str).fillna("")
    trusted["query_id"] = trusted["query_id"].astype(str)
    label_counts = trusted["label_kind"].value_counts().to_dict()
    if (
        len(trusted) != 279
        or not set(label_counts).issubset({"MATCH_EXACT", "AMBIGUOUS", "UNRESOLVED"})
        or int(label_counts.get("MATCH_EXACT", 0)) < 200
        or int(label_counts.get("AMBIGUOUS", 0)) < 20
    ):
        raise ValueError("Trusted REVIEW labels changed")
    trusted["ground_truth_siren"] = trusted["ground_truth_siret"].map(
        lambda value: value[:9] if value else None
    )
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
    return population, trusted


def _label_candidates(
    candidates: pd.DataFrame, population: pd.DataFrame
) -> pd.DataFrame:
    truth = population.set_index("query_id")[["label_kind", "ground_truth_siret"]]
    output = candidates.drop(columns=["is_ground_truth"], errors="ignore").join(
        truth, on="query_id"
    )
    output["is_ground_truth"] = (
        output["label_kind"].eq("MATCH_EXACT")
        & output["candidate_siret"].astype(str).eq(
            output["ground_truth_siret"].fillna("").astype(str)
        )
    ).astype(np.int8)
    return output.drop(columns=["label_kind", "ground_truth_siret"])


def _rebuild_canonical_ranked(
    candidates: pd.DataFrame,
    population: pd.DataFrame,
    trusted: pd.DataFrame,
    ranker_model: Path,
) -> pd.DataFrame:
    """Reproduce the trusted-ranker OOF predictions used by the current scenes."""

    trusted_ids = set(trusted["query_id"])
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

    parts: list[pd.DataFrame] = []
    for fold in range(5):
        held_base_ids = set(
            fit_population.loc[
                fit_population["oof_fold"].astype(int).eq(fold), "query_id"
            ].astype(str)
        )
        base_train = base_rows[~base_rows["query_id"].isin(held_base_ids)]
        hard_train = trusted_candidates[
            trusted_candidates["query_id"].isin(eligible_trusted_ids)
            & trusted_candidates["oof_fold"].astype(int).ne(fold)
        ]
        ranker = fit_weighted_ranker(
            pd.concat([base_train, hard_train], ignore_index=True),
            hard_query_ids=set(hard_train["query_id"].astype(str)),
            hard_weight=RANKER_WEIGHT,
        )
        scored_ids = held_base_ids | set(
            trusted_population.loc[
                trusted_population["oof_fold"].astype(int).eq(fold), "query_id"
            ].astype(str)
        )
        parts.append(
            _rank(
                ranker,
                candidates[candidates["query_id"].isin(scored_ids)].copy(),
                "trusted_ranker_oof",
            )
        )
    already_oof = set(fit_population["query_id"].astype(str)) | trusted_ids
    remaining_ids = set(population["query_id"].astype(str)) - already_oof
    full_ranker = xgb.XGBRanker()
    full_ranker.load_model(ranker_model)
    parts.append(
        _rank(
            full_ranker,
            candidates[candidates["query_id"].isin(remaining_ids)].copy(),
            "trusted_ranker_full_fit",
        )
    )
    ranked = pd.concat(parts, ignore_index=True)
    return ranked


def _load_ranked(path: Path, candidates: pd.DataFrame) -> pd.DataFrame:
    ranked = _normalise_query_ids(pd.read_parquet(path))
    keys = ["query_id", "candidate_siret"]
    required = {*keys, "ranker_score"}
    if not required.issubset(ranked.columns):
        raise ValueError(f"Ranked candidates missing {sorted(required - set(ranked))}")
    if ranked.duplicated(keys).any() or candidates.duplicated(keys).any():
        raise ValueError("Duplicate candidate key")
    candidate_keys = set(map(tuple, candidates[keys].astype(str).to_numpy()))
    ranked_keys = set(map(tuple, ranked[keys].astype(str).to_numpy()))
    if not ranked_keys or not ranked_keys.issubset(candidate_keys):
        raise ValueError("Ranked candidates are not a subset of the candidate pools")
    if not np.isfinite(ranked["ranker_score"].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite ranker scores")
    keep = keys + [
        name
        for name in (
            "candidate_siren",
            "retrieval_rank",
            "ranker_score",
            "ranker_rank",
            "prediction_origin",
            "oof_fold",
        )
        if name in ranked.columns
    ]
    return ranked[keep]


def _covers_exact_pools(ranked: pd.DataFrame, candidates: pd.DataFrame) -> bool:
    keys = ["query_id", "candidate_siret"]
    return len(ranked) == len(candidates) and set(
        map(tuple, ranked[keys].astype(str).to_numpy())
    ) == set(map(tuple, candidates[keys].astype(str).to_numpy()))


def _overlay_partial_ranking(
    base: pd.DataFrame,
    overlay: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["query_id", "candidate_siret"]
    if base.duplicated(keys).any() or overlay.duplicated(keys).any():
        raise ValueError("Duplicate ranking key")
    update = overlay.set_index(keys)["ranker_score"]
    output = base.copy().set_index(keys)
    missing = update.index.difference(output.index)
    if len(missing):
        raise ValueError("Partial ranking contains candidates absent from the base")
    output.loc[update.index, "ranker_score"] = update.astype(float)
    output = output.reset_index().sort_values(
        ["query_id", "ranker_score", "retrieval_rank", "candidate_siret"],
        ascending=[True, False, True, True],
        kind="mergesort",
    )
    output["ranker_rank"] = output.groupby("query_id", sort=False).cumcount() + 1
    output["prediction_origin"] = "ranked_overlay_on_canonical_oof"
    return output


def _join_ranking(
    candidates: pd.DataFrame, ranked: pd.DataFrame
) -> pd.DataFrame:
    keys = ["query_id", "candidate_siret"]
    ranking = ranked[
        keys
        + [
            name
            for name in ("ranker_score", "ranker_rank", "prediction_origin", "oof_fold")
            if name in ranked.columns
        ]
    ].copy()
    output = candidates.drop(
        columns=["ranker_score", "ranker_rank", "prediction_origin", "oof_fold"],
        errors="ignore",
    ).merge(ranking, on=keys, validate="one_to_one")
    if "ranker_rank" not in output:
        output = output.sort_values(
            ["query_id", "ranker_score", "retrieval_rank", "candidate_siret"],
            ascending=[True, False, True, True],
            kind="mergesort",
        )
        output["ranker_rank"] = output.groupby("query_id", sort=False).cumcount() + 1
    return output


def _legal_role_signals(row: pd.Series) -> tuple[float, float]:
    crm = str(row.get("_crm", ""))
    association = bool(row.get("legal_is_association", 0))
    public = bool(row.get("legal_is_public", 0))
    company = bool(row.get("legal_is_company", 0))
    asks_association = any(token in crm.split() for token in ("ASSOCIATION", "ASSOC"))
    asks_public = any(
        token in crm.split()
        for token in ("COMMUNE", "MAIRIE", "DEPARTEMENT", "REGION", "METROPOLE")
    )
    asks_company = any(token in crm.split() for token in ("SAS", "SARL", "SA", "SNC"))
    asked = asks_association or asks_public or asks_company
    matches = (
        (asks_association and association)
        or (asks_public and public)
        or (asks_company and company)
    )
    return float(matches), float(asked and not matches)


def _other_siren_max(
    ranked: pd.DataFrame,
    top1: pd.Series,
    column: str,
) -> float | None:
    same_address = ranked[
        ranked["_address_key"].eq(top1["_address_key"])
        & ranked["candidate_siren"].astype(str).ne(str(top1["candidate_siren"]))
    ]
    if same_address.empty:
        return None
    return float(same_address[column].astype(float).max())


def _business_scene_features(pool: pd.DataFrame) -> dict[str, float]:
    ranked = rank_v411_candidates(pool)
    if ranked.empty:
        return {name: 0.0 for name in BUSINESS_ACCEPTOR_FEATURES}
    top1 = ranked.iloc[0]
    top2 = ranked.iloc[1] if len(ranked) > 1 else None

    def value(row: pd.Series | None, name: str) -> float:
        return 0.0 if row is None or pd.isna(row.get(name)) else float(row[name])

    top1_match, top1_conflict = _legal_role_signals(top1)
    top2_match, _ = _legal_role_signals(top2) if top2 is not None else (0.0, 0.0)
    top1_role = value(top1, "business_role_match") - value(
        top1, "business_role_conflict"
    )
    top2_role = value(top2, "business_role_match") - value(
        top2, "business_role_conflict"
    )
    same_address = ranked[ranked["_address_key"].eq(top1["_address_key"])].copy()
    same_address["_strong"] = (
        same_address[["name_sim_max_ul", "name_sim_max_etab"]]
        .astype(float)
        .max(axis=1)
        .ge(0.85)
    )
    same_address["_exact_ul"] = same_address["ul_name_exact"].astype(float).eq(1)
    same_address["_exact_etab"] = same_address["etab_name_exact"].astype(float).eq(1)

    def siren_count(mask: pd.Series) -> float:
        return float(same_address.loc[mask, "candidate_siren"].astype(str).nunique())

    def margin(name: str, top: float, default: float = 1.0) -> float:
        other = _other_siren_max(ranked, top1, name)
        return default if other is None else top - other

    top1_year = value(top1, "establishment_start_year")
    other_year = _other_siren_max(ranked, top1, "establishment_start_year")
    year_margin = 0.0 if other_year is None else (top1_year - other_year) / 100.0
    return {
        "top1_ul_name_exact_business": value(top1, "ul_name_exact"),
        "delta_ul_name_exact_business": value(top1, "ul_name_exact")
        - value(top2, "ul_name_exact"),
        "top1_etab_name_exact_business": value(top1, "etab_name_exact"),
        "delta_etab_name_exact_business": value(top1, "etab_name_exact")
        - value(top2, "etab_name_exact"),
        "top1_same_address_siren_count_business": float(
            value(top1, "same_address_siren_count")
        ),
        "same_address_strong_name_siren_count_business": siren_count(
            same_address["_strong"]
        ),
        "same_address_exact_ul_siren_count_business": siren_count(
            same_address["_exact_ul"]
        ),
        "same_address_exact_etab_siren_count_business": siren_count(
            same_address["_exact_etab"]
        ),
        "ul_margin_other_siren_same_address_business": margin(
            "name_sim_max_ul", value(top1, "name_sim_max_ul")
        ),
        "etab_margin_other_siren_same_address_business": margin(
            "name_sim_max_etab", value(top1, "name_sim_max_etab")
        ),
        "top1_legal_association_business": value(top1, "legal_is_association"),
        "delta_legal_association_business": value(top1, "legal_is_association")
        - value(top2, "legal_is_association"),
        "top1_legal_public_business": value(top1, "legal_is_public"),
        "delta_legal_public_business": value(top1, "legal_is_public")
        - value(top2, "legal_is_public"),
        "top1_legal_company_business": value(top1, "legal_is_company"),
        "delta_legal_company_business": value(top1, "legal_is_company")
        - value(top2, "legal_is_company"),
        "top1_legal_role_match_business": top1_match,
        "delta_legal_role_match_business": top1_match - top2_match,
        "top1_legal_role_conflict_business": top1_conflict,
        "top1_business_role_net_business": top1_role,
        "delta_business_role_net_business": top1_role - top2_role,
        "role_margin_other_siren_same_address_business": margin(
            "_role_signal", top1_role
        ),
        "start_year_margin_other_siren_same_address_business": year_margin,
        "top1_newest_same_address_business": value(
            top1, "best_start_date_same_address"
        ),
    }


def _enrich_scenes(
    scenes: pd.DataFrame, ranked_candidates: pd.DataFrame
) -> pd.DataFrame:
    business = []
    for query_id, pool in ranked_candidates.groupby("query_id", sort=False):
        business.append({"query_id": str(query_id), **_business_scene_features(pool)})
    features = pd.DataFrame(business)
    output = scenes.merge(features, on="query_id", validate="one_to_one")
    matrix = output[ACCEPTOR_FEATURES].to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite enriched acceptor features")
    return output


def _fit_acceptor(
    frame: pd.DataFrame,
    trusted_ids: set[str],
    weight: float,
    features: list[str] = ACCEPTOR_FEATURES,
    constraints: list[int] = MONOTONIC_CONSTRAINTS,
) -> Any:
    config = dict(EXPECTED_MODEL_CONFIGS["MONOTONIC_XGB"])
    config["monotone_constraints"] = tuple(constraints)
    model = xgb.XGBClassifier(**config)
    sample_weight = np.where(frame["query_id"].isin(trusted_ids), weight, 1.0).astype(
        np.float32
    )
    model.fit(
        frame[features].to_numpy(dtype=np.float32),
        frame["acceptor_target"].astype(int).to_numpy(),
        sample_weight=sample_weight,
    )
    return model


def _scores(
    model: Any,
    frame: pd.DataFrame,
    features: list[str] = ACCEPTOR_FEATURES,
) -> np.ndarray:
    return np.asarray(
        model.predict_proba(frame[features].to_numpy(dtype=np.float32))[:, 1],
        dtype=float,
    )


def _decision_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    auto = frame["decision"].eq("AUTO_MATCH")
    correct = auto & frame["acceptor_target"].astype(bool)
    ambiguous = auto & frame["label_kind"].eq("AMBIGUOUS")
    auto_count = int(auto.sum())
    correct_count = int(correct.sum())
    positive_count = int(frame["acceptor_target"].astype(int).sum())
    return {
        "row_count": len(frame),
        "positive_count": positive_count,
        "auto_count": auto_count,
        "correct_auto": correct_count,
        "error_auto": auto_count - correct_count,
        "ambiguous_auto": int(ambiguous.sum()),
        "precision": correct_count / auto_count if auto_count else None,
        "coverage_all": auto_count / len(frame),
        "correct_top1_acceptance": (
            correct_count / positive_count if positive_count else 0.0
        ),
    }


def _nested_oof(
    base: pd.DataFrame,
    trusted: pd.DataFrame,
    weight: float,
    features: list[str] = ACCEPTOR_FEATURES,
    constraints: list[int] = MONOTONIC_CONSTRAINTS,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Calibrate inside each outer component fold, then score that outer fold."""

    outer_parts: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    for outer in range(5):
        calibration_parts: list[pd.DataFrame] = []
        for inner in range(5):
            if inner == outer:
                continue
            fit_base = base[
                ~base["oof_fold"].astype(int).isin([outer, inner])
            ]
            fit_trusted = trusted[
                ~trusted["oof_fold"].astype(int).isin([outer, inner])
            ]
            model = _fit_acceptor(
                pd.concat([fit_base, fit_trusted], ignore_index=True),
                set(fit_trusted["query_id"].astype(str)),
                weight,
                features,
                constraints,
            )
            held = trusted[trusted["oof_fold"].astype(int).eq(inner)].copy()
            held["acceptor_score"] = _scores(model, held, features)
            calibration_parts.append(held)
        calibration = pd.concat(calibration_parts, ignore_index=True)
        selected = select_threshold(
            calibration["acceptor_score"].to_numpy(),
            calibration["acceptor_target"].astype(int).to_numpy(),
            calibration["label_kind"].astype(str).to_numpy(),
        )
        if selected is None:
            raise ValueError(f"No safe inner threshold for outer fold {outer}")
        threshold, calibration_metrics, _ = selected
        outer_base = base[base["oof_fold"].astype(int).ne(outer)]
        outer_trusted = trusted[trusted["oof_fold"].astype(int).ne(outer)]
        model = _fit_acceptor(
            pd.concat([outer_base, outer_trusted], ignore_index=True),
            set(outer_trusted["query_id"].astype(str)),
            weight,
            features,
            constraints,
        )
        held = trusted[trusted["oof_fold"].astype(int).eq(outer)].copy()
        held["acceptor_score"] = _scores(model, held, features)
        held["nested_threshold"] = float(threshold)
        held["decision"] = np.where(
            held["acceptor_score"].ge(threshold), "AUTO_MATCH", "REVIEW"
        )
        outer_parts.append(held)
        folds.append(
            {
                "outer_fold": outer,
                "inner_calibration_count": len(calibration),
                "threshold": float(threshold),
                "inner_calibration_metrics": calibration_metrics,
                "outer_metrics": _decision_metrics(held),
            }
        )
    return pd.concat(outer_parts, ignore_index=True), folds


def run(args: argparse.Namespace) -> Path:
    dataset = args.dataset.resolve()
    population, trusted_labels = _load_population(dataset, args.trusted_labels)
    base_candidates = _normalise_query_ids(
        pd.read_parquet(dataset / "candidates_sparse_top100.parquet")
    )
    base_candidates = _label_candidates(base_candidates, population)

    base_ranked_hash = None
    if args.ranked_candidates is None:
        ranked = _rebuild_canonical_ranked(
            base_candidates, population, trusted_labels, args.ranker_model.resolve()
        )
        ranked_source = "REBUILT_CANONICAL_TRUSTED_RANKER_OOF"
        ranked_hash = None
    else:
        supplied = _load_ranked(args.ranked_candidates.resolve(), base_candidates)
        if _covers_exact_pools(supplied, base_candidates):
            ranked = supplied
            ranked_source = str(args.ranked_candidates.resolve())
        else:
            if args.base_ranked_candidates is None:
                raise ValueError(
                    "Partial --ranked-candidates requires --base-ranked-candidates"
                )
            base_ranked = _load_ranked(
                args.base_ranked_candidates.resolve(), base_candidates
            )
            if not _covers_exact_pools(base_ranked, base_candidates):
                raise ValueError("Base ranking does not cover the exact candidate pools")
            ranked = _overlay_partial_ranking(base_ranked, supplied)
            ranked_source = (
                f"{args.ranked_candidates.resolve()} OVER "
                f"{args.base_ranked_candidates.resolve()}"
            )
            base_ranked_hash = file_sha256(args.base_ranked_candidates.resolve())
        ranked_hash = file_sha256(args.ranked_candidates.resolve())

    enriched = _read_enriched_sources(
        dataset, args.etablissements, args.unites_legales
    )
    enriched = _relational_features(_source_features(enriched))
    enriched = _label_candidates(enriched, population)
    scored = _join_ranking(enriched, ranked)
    taxonomy = SiteFunctionTaxonomy.load(args.taxonomy)
    scenes = build_scene_frame(population, scored, taxonomy)
    scenes = _enrich_scenes(scenes, scored)

    all_trusted_ids = set(trusted_labels["query_id"].astype(str))
    base = scenes[
        scenes["split"].eq("fit") & scenes["label_kind"].eq("MATCH_EXACT")
    ].copy()
    trusted = scenes[
        scenes["query_id"].isin(all_trusted_ids)
        & scenes["label_kind"].isin(["MATCH_EXACT", "AMBIGUOUS"])
    ].copy()
    if len(base) != 4666 or len(trusted) < 250:
        raise ValueError("Acceptor development populations changed")
    trusted_ids = set(trusted["query_id"].astype(str))

    reference_detail, reference_folds = _nested_oof(
        base,
        trusted,
        10.0,
        V411_ACCEPTOR_FEATURE_NAMES,
        list(V411_MONOTONIC_CONSTRAINTS),
    )
    reference_metrics = _decision_metrics(reference_detail)
    variants: dict[str, Any] = {}
    details: dict[float, pd.DataFrame] = {}
    for weight in WEIGHTS:
        detail, fold_metrics = _nested_oof(base, trusted, weight)
        metrics = _decision_metrics(detail)
        eligible = (
            metrics["auto_count"] > 0
            and metrics["error_auto"] == 0
            and metrics["ambiguous_auto"] == 0
            and metrics["correct_top1_acceptance"] >= 0.65
        )
        variants[str(weight)] = {
            "trusted_weight": weight,
            "nested_oof_metrics": metrics,
            "outer_folds": fold_metrics,
            "eligible": eligible,
        }
        details[weight] = detail

    eligible = [
        item for item in variants.values() if bool(item.get("eligible"))
    ]
    ranked_variants = eligible or list(variants.values())
    winner = sorted(
        ranked_variants,
        key=lambda item: (
            int(item["nested_oof_metrics"]["error_auto"]),
            -float(item["nested_oof_metrics"]["correct_top1_acceptance"]),
            float(item["trusted_weight"]),
        ),
    )[0]
    winner_weight = float(winner["trusted_weight"])
    winner_detail = details[winner_weight]

    # Development-only final bundle: model sees all consumed development rows;
    # threshold is obtained from OOF scores, never from in-sample predictions.
    selected = select_threshold(
        winner_detail["acceptor_score"].to_numpy(),
        winner_detail["acceptor_target"].astype(int).to_numpy(),
        winner_detail["label_kind"].astype(str).to_numpy(),
    )
    if selected is None:
        raise ValueError("No final development-only OOF threshold")
    final_threshold, _, _ = selected
    final_model = _fit_acceptor(
        pd.concat([base, trusted], ignore_index=True), trusted_ids, winner_weight
    )

    identity_payload = {
        "schema": SCHEMA_VERSION,
        "dataset_manifest": file_sha256(dataset / "manifest.json"),
        "trusted_labels": file_sha256(args.trusted_labels),
        "ranked_source": ranked_source,
        "ranked_hash": ranked_hash,
        "base_ranked_hash": base_ranked_hash,
        "features": ACCEPTOR_FEATURES,
        "weights": WEIGHTS,
        "builder": file_sha256(Path(__file__)),
    }
    identity = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True).encode()
    ).hexdigest()[:16]
    output = args.output_root.resolve() / identity
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "CONSUMED_DEVELOPMENT_ONLY",
        "final_test_opened": False,
        "production_promotion_authorized": False,
        "ranked_candidates_source": ranked_source,
        "ranked_candidates_sha256": ranked_hash,
        "base_ranked_candidates_sha256": base_ranked_hash,
        "trusted_label_counts": trusted_labels["label_kind"].value_counts().to_dict(),
        "trusted_eligible_scene_count": len(trusted),
        "feature_count": len(ACCEPTOR_FEATURES),
        "business_features": BUSINESS_ACCEPTOR_FEATURES,
        "weights": list(WEIGHTS),
        "canonical_reference_nested_oof": {
            "trusted_weight": 10.0,
            "metrics": reference_metrics,
            "outer_folds": reference_folds,
        },
        "variants": variants,
        "selected": winner,
        "fixed_development_policy": {
            "trusted_weight": winner_weight,
            "threshold_from_consumed_oof": float(final_threshold),
        },
        "gate": {
            "minimum_correct_top1_acceptance": 0.65,
            "minimum_precision": 0.998,
            "observed_small_sample_requirement": "ZERO_ERROR",
        },
        "verdict": "GO_DEV" if bool(winner.get("eligible")) else "PIVOT_FEATURES",
        "limitations": {
            "trusted_labels_used_for_model_selection": True,
            "trusted_oof_used_for_final_development_threshold": True,
            "precision_99_8_certified": False,
        },
    }
    (output / "evaluation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    scenes.to_parquet(output / "acceptor_scenes_business.parquet", index=False)
    winner_detail[
        [
            "query_id",
            "label_kind",
            "ground_truth_siret",
            "predicted_siret",
            "acceptor_target",
            "oof_fold",
            "acceptor_score",
            "nested_threshold",
            "decision",
        ]
    ].to_parquet(output / "trusted_nested_oof_decisions.parquet", index=False)
    joblib.dump(final_model, output / "acceptor_candidate.joblib")
    if args.ranked_candidates is None:
        ranked[
            [
                "query_id",
                "candidate_siret",
                "candidate_siren",
                "retrieval_rank",
                "ranker_score",
                "ranker_rank",
            ]
        ].to_parquet(output / "ranked_candidates.parquet", index=False)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--trusted-labels", type=Path, default=DEFAULT_TRUSTED)
    parser.add_argument("--ranker-model", type=Path, default=DEFAULT_RANKER)
    parser.add_argument("--ranked-candidates", type=Path)
    parser.add_argument("--base-ranked-candidates", type=Path)
    parser.add_argument("--etablissements", type=Path, default=DEFAULT_ETABLISSEMENTS)
    parser.add_argument("--unites-legales", type=Path, default=DEFAULT_UNITES_LEGALES)
    parser.add_argument(
        "--taxonomy", type=Path, default=Path("config/v4_9_site_function_taxonomy.json")
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
