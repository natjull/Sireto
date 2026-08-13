#!/usr/bin/env python3
"""Build one inference-safe query scene from each V4.12-L OOF ranking."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402


SCHEMA_VERSION = "sireto-v4.12-learned-scenes-1"
BASE = Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100")
DEFAULT_DATASET = BASE / "datasets/v4_12_learned_business_features/8800ef53f6927215"
DEFAULT_RANKER = BASE / "experiments/v4_12_learned_oof_rankers/839ef55308d5077e"
CANDIDATE_FEATURES = [
    "retrieval_rank",
    "name_jaro_max",
    "name_jaro_gap",
    "name_token_overlap_max",
    "name_sim_max_etab",
    "name_sim_max_ul",
    "addr_jaro",
    "postcode_match",
    "city_match",
    "addr_token_overlap",
    "street_name_jaro",
    "name_addr_consistency",
    "is_siege",
    "admission_fusion_score",
    "admission_channel_count",
    "source_ul_exact",
    "source_etab_exact",
    "source_name_score",
    "source_name_exact",
    "source_name_address_consistency",
    "legal_is_public",
    "legal_is_company",
    "activity_is_holding",
    "activity_is_property",
    "has_operating_enseigne",
    "is_employer",
    "has_known_effectif",
    "business_role_match",
    "business_role_conflict",
    "role_signal",
    "operating_evidence",
    "same_siren_count",
    "same_address_siren_count",
    "source_name_gap_to_best_query",
    "address_gap_to_best_query",
    "identity_gap_to_best_query",
    "operating_gap_to_best_query",
    "source_name_gap_to_best_same_siren",
    "address_gap_to_best_same_siren",
]
METADATA_COLUMNS = [
    "query_id",
    "oof_fold",
    "label_kind",
    "ground_truth_siret",
    "ground_truth_siren",
    "ground_truth_state",
    "label_is_human_validated",
    "acceptor_weight",
    "predicted_siret",
    "predicted_siren",
    "top1_correct",
]


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _prefixed(rows: pd.DataFrame, prefix: str) -> pd.DataFrame:
    keep = ["query_id", "candidate_siret", "candidate_siren", "ranker_score", *CANDIDATE_FEATURES]
    output = rows[keep].copy()
    return output.rename(
        columns={column: f"{prefix}_{column}" for column in keep if column != "query_id"}
    )


def _build_scene_frame(
    candidates: pd.DataFrame,
    predictions: pd.DataFrame,
    queries: pd.DataFrame,
    labels: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    keys = ["query_id", "candidate_siret"]
    ranked = predictions.merge(
        candidates[keys + [column for column in CANDIDATE_FEATURES if column != "retrieval_rank"]],
        on=keys,
        validate="one_to_one",
    ).sort_values(["query_id", "ranker_rank"], kind="mergesort")
    if ranked.duplicated(keys).any():
        raise ValueError("Scene input has duplicate candidates")

    grouped = ranked.groupby("query_id", sort=False)
    ranked["_score_mean"] = grouped["ranker_score"].transform("mean")
    ranked["_score_std"] = grouped["ranker_score"].transform("std").fillna(0.0)
    ranked["_score_max"] = grouped["ranker_score"].transform("max")
    ranked["_score_from_top"] = ranked["_score_max"] - ranked["ranker_score"]
    safe_std = ranked["_score_std"].replace(0.0, 1.0)
    ranked["_score_z"] = (ranked["ranker_score"] - ranked["_score_mean"]) / safe_std
    exp_score = np.exp((ranked["ranker_score"] - ranked["_score_max"]).clip(-30, 0))
    ranked["_softmax"] = exp_score / exp_score.groupby(ranked["query_id"], sort=False).transform("sum")
    entropy_term = -(ranked["_softmax"] * np.log(ranked["_softmax"].clip(1e-12)))
    entropy = entropy_term.groupby(ranked["query_id"], sort=False).sum()

    top1_rows = ranked[ranked["ranker_rank"].eq(1)].copy()
    top2_rows = ranked[ranked["ranker_rank"].eq(2)].copy()
    top3_rows = ranked[ranked["ranker_rank"].eq(3)].copy()
    if len(top1_rows) != labels["query_id"].nunique():
        raise ValueError("Every query must have exactly one top1")
    top1_siren = top1_rows.set_index("query_id")["candidate_siren"]
    ranked["_top1_siren"] = ranked["query_id"].map(top1_siren)
    alternatives = ranked[ranked["candidate_siren"].ne(ranked["_top1_siren"])]
    best_alt = alternatives.drop_duplicates("query_id", keep="first")

    scene = _prefixed(top1_rows, "top1")
    scene = scene.merge(_prefixed(top2_rows, "top2"), on="query_id", how="left", validate="one_to_one")
    scene = scene.merge(_prefixed(top3_rows, "top3"), on="query_id", how="left", validate="one_to_one")
    scene = scene.merge(_prefixed(best_alt, "alt_siren"), on="query_id", how="left", validate="one_to_one")

    query_stats = grouped.agg(
        candidate_count=("candidate_siret", "size"),
        distinct_siren_count=("candidate_siren", "nunique"),
        score_mean=("ranker_score", "mean"),
        score_std=("ranker_score", "std"),
        score_min=("ranker_score", "min"),
        score_max=("ranker_score", "max"),
    ).reset_index()
    query_stats["score_entropy"] = query_stats["query_id"].map(entropy).astype(float)
    query_stats["top10_distinct_siren_count"] = query_stats["query_id"].map(
        ranked[ranked["ranker_rank"].le(10)].groupby("query_id")["candidate_siren"].nunique()
    )
    for limit in (0.01, 0.05, 0.10, 0.25, 0.50):
        counts = ranked[ranked["_score_from_top"].le(limit)].groupby("query_id").size()
        query_stats[f"candidate_count_within_{str(limit).replace('.', '_')}"] = (
            query_stats["query_id"].map(counts).fillna(0)
        )
    scene = scene.merge(query_stats, on="query_id", validate="one_to_one")

    scene["top1_top2_score_gap"] = scene["top1_ranker_score"] - scene["top2_ranker_score"]
    scene["top1_top3_score_gap"] = scene["top1_ranker_score"] - scene["top3_ranker_score"]
    scene["top1_alt_siren_score_gap"] = scene["top1_ranker_score"] - scene["alt_siren_ranker_score"]
    scene["top1_softmax_probability"] = top1_rows.set_index("query_id")["_softmax"].reindex(scene["query_id"]).to_numpy()
    scene["top1_score_z"] = top1_rows.set_index("query_id")["_score_z"].reindex(scene["query_id"]).to_numpy()
    for feature in CANDIDATE_FEATURES:
        if feature == "retrieval_rank":
            continue
        scene[f"top1_minus_top2_{feature}"] = scene[f"top1_{feature}"] - scene[f"top2_{feature}"]
        scene[f"top1_minus_alt_siren_{feature}"] = scene[f"top1_{feature}"] - scene[f"alt_siren_{feature}"]

    query_fields = queries[["query_id", "crm_name", "crm_address", "crm_postcode", "crm_city", "crm_insee"]].copy()
    for column in ["crm_name", "crm_address", "crm_postcode", "crm_city", "crm_insee"]:
        query_fields[f"missing_{column}"] = query_fields[column].fillna("").astype(str).str.strip().eq("").astype("float32")
    scene = scene.merge(
        query_fields[["query_id", *[f"missing_{column}" for column in ["crm_name", "crm_address", "crm_postcode", "crm_city", "crm_insee"]]]],
        on="query_id",
        validate="one_to_one",
    )
    scene = labels.merge(scene, on="query_id", validate="one_to_one")
    scene = scene.rename(
        columns={
            "top1_candidate_siret": "predicted_siret",
            "top1_candidate_siren": "predicted_siren",
        }
    )
    scene["top1_correct"] = (
        scene["label_kind"].eq("MATCH_EXACT")
        & scene["predicted_siret"].astype(str).eq(scene["ground_truth_siret"].astype(str))
    )
    drop_model = {
        *METADATA_COLUMNS,
        "historical_ground_truth_siret",
        "historical_ground_truth_siren",
        "label_source",
        "validator",
        "reliability",
        "evidence_reference",
        "exact_metric_eligible",
        "identity_training_eligible",
        "operational_training_eligible",
        "ranker_weight",
        "legacy_split",
        "top1_variant",
        "top1_ranker_rank",
    }
    feature_order = [
        column
        for column in scene.columns
        if column not in drop_model and pd.api.types.is_numeric_dtype(scene[column])
    ]
    scene[feature_order] = scene[feature_order].replace([np.inf, -np.inf], np.nan).fillna(-1.0).astype("float32")
    if not np.isfinite(scene[feature_order].to_numpy()).all():
        raise ValueError("Scene features contain non-finite values")
    if any("ground_truth" in column or "correct" in column for column in feature_order):
        raise ValueError("Scene model features leak the target")
    return scene[[*METADATA_COLUMNS, *feature_order]].sort_values("query_id").reset_index(drop=True), feature_order


def build(args: argparse.Namespace) -> Path:
    dataset_manifest = json.loads((args.dataset / "manifest.json").read_text(encoding="utf-8"))
    ranker_manifest = json.loads((args.ranker / "manifest.json").read_text(encoding="utf-8"))
    candidates_path = args.dataset / "candidates_business.parquet"
    queries_path = args.dataset / "queries.parquet"
    labels_path = args.dataset / "labels.parquet"
    predictions_path = args.ranker / "business_learned_oof_candidates.parquet"
    for path in (candidates_path, queries_path, labels_path):
        if dataset_manifest["outputs"].get(path.name) != file_sha256(path):
            raise ValueError(f"Dataset hash mismatch: {path}")
    relative_prediction = predictions_path.name
    if ranker_manifest["outputs"].get(relative_prediction) != file_sha256(predictions_path):
        raise ValueError("Ranker prediction hash mismatch")
    if ranker_manifest.get("deterministic_promotions") is not False:
        raise ValueError("Scenes require an unmodified learned ranking")

    identity = {
        "schema_version": SCHEMA_VERSION,
        "builder_sha256": file_sha256(Path(__file__)),
        "dataset_manifest_sha256": file_sha256(args.dataset / "manifest.json"),
        "ranker_manifest_sha256": file_sha256(args.ranker / "manifest.json"),
        "prediction_sha256": file_sha256(predictions_path),
        "candidate_features": CANDIDATE_FEATURES,
    }
    build_id = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
    destination = args.output_root / build_id
    if destination.exists():
        existing = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        if existing.get("build_identity") != identity:
            raise FileExistsError(f"Conflicting immutable output: {destination}")
        return destination

    candidates = pd.read_parquet(candidates_path, columns=["query_id", "candidate_siret", *CANDIDATE_FEATURES])
    predictions = pd.read_parquet(predictions_path)
    queries = pd.read_parquet(queries_path)
    labels = pd.read_parquet(labels_path)
    for frame in (candidates, predictions, queries, labels):
        frame["query_id"] = frame["query_id"].astype(str)
    candidates["candidate_siret"] = candidates["candidate_siret"].astype(str)
    predictions["candidate_siret"] = predictions["candidate_siret"].astype(str)
    scenes, feature_order = _build_scene_frame(candidates, predictions, queries, labels)

    args.output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{build_id}.", dir=args.output_root))
    try:
        scenes.to_parquet(temporary / "scenes.parquet", index=False)
        report = (
            "# Scènes OOF V4.12-L\n\n"
            f"- scènes : {len(scenes)} ;\n"
            f"- features query-level : {len(feature_order)} ;\n"
            f"- top1 corrects : {int(scenes['top1_correct'].sum())} ;\n"
            "- source : prédictions candidat OOF, sans règle de promotion.\n"
        )
        (temporary / "report.md").write_text(report, encoding="utf-8")
        outputs = {name: file_sha256(temporary / name) for name in ("scenes.parquet", "report.md")}
        _json_dump(
            temporary / "manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "build_id": build_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "build_identity": identity,
                "row_count": len(scenes),
                "feature_order": feature_order,
                "all_ranker_predictions_oof": True,
                "deterministic_promotions": False,
                "outputs": outputs,
            },
        )
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--ranker", type=Path, default=DEFAULT_RANKER)
    parser.add_argument("--output-root", type=Path, default=BASE / "datasets/v4_12_learned_scenes")
    return parser.parse_args()


if __name__ == "__main__":
    print(build(parse_args()))
