#!/usr/bin/env python3
"""Build an immutable, train-only real/synthetic mix for XGBoost and BGE.

The synthetic feature bundle is an upstream artifact produced by running the
frozen top-100 retrieval and the frozen BUSINESS/text projections on promoted
synthetic CRM rows.  This builder never retrieves candidates and never injects
a target.  It validates the upstream bundle, selects a deterministic 2:1
real/synthetic scene mix and assigns the documented 0.5/k synthetic weight.
"""

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

from scripts.build_v412_learned_business_features import BUSINESS_FEATURE_ORDER  # noqa: E402
from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402


BASE = Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100")
DEFAULT_REAL_BUSINESS = BASE / "datasets/v4_12_learned_business_features/8800ef53f6927215"
DEFAULT_REAL_BGE_GROUPS = BASE / "datasets/v4_12_bge_training_groups/114b407f2ccf7b40"
DEFAULT_OUTPUT_ROOT = BASE / "datasets/synthetic_augmented_model_mix_v1"
DEFAULT_PLAN = Path("config/synthetic_augmented_model_eval_v1.json")
SCHEMA_VERSION = "sireto-synthetic-augmented-model-mix-1"
SYNTHETIC_BUNDLE_SCHEMA = "sireto-synthetic-gt-model-features-1"
TRAIN_FOLDS = (2, 3, 4)
STRATA = ("difficulty", "augmentation_stratum")


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _verified_manifest(root: Path, names: tuple[str, ...]) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in names:
        expected = manifest.get("outputs", {}).get(name)
        if not expected or file_sha256(root / name) != expected:
            raise ValueError(f"Manifest mismatch: {root / name}")
    return manifest


def _stable_order(query_id: str, target_siren: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{target_siren}:{query_id}".encode()).hexdigest()


def _allocate_strata(labels: pd.DataFrame, total: int) -> dict[tuple[str, str], int]:
    """Largest-remainder proportional allocation, bounded by cell capacity."""
    counts = labels.groupby(list(STRATA), dropna=False).size().sort_index()
    if total > int(counts.sum()):
        raise ValueError("Synthetic sample target exceeds eligible scenes")
    if total == int(counts.sum()):
        return {tuple(map(str, key)): int(value) for key, value in counts.items()}
    ideals = counts.astype(float) * (float(total) / float(counts.sum()))
    allocation = np.floor(ideals).astype(int)
    remaining = total - int(allocation.sum())
    ranked = sorted(
        counts.index,
        key=lambda key: (-(float(ideals.loc[key]) - int(allocation.loc[key])), tuple(map(str, key))),
    )
    for key in ranked[:remaining]:
        allocation.loc[key] += 1
    return {tuple(map(str, key)): int(value) for key, value in allocation.items()}


def _select_synthetic(labels: pd.DataFrame, total: int, seed: int) -> pd.DataFrame:
    allocation = _allocate_strata(labels, total)
    work = labels.copy()
    work["_stable_order"] = [
        _stable_order(str(row.query_id), str(row.ground_truth_siren), seed)
        for row in work.itertuples(index=False)
    ]
    selected: list[pd.DataFrame] = []
    for key, frame in work.groupby(list(STRATA), dropna=False, sort=True):
        normalised_key = tuple(map(str, key if isinstance(key, tuple) else (key,)))
        count = allocation.get(normalised_key, 0)
        selected.append(frame.sort_values("_stable_order", kind="mergesort").head(count))
    output = pd.concat(selected, ignore_index=True).drop(columns="_stable_order")
    if len(output) != total or output["query_id"].duplicated().any():
        raise AssertionError("Deterministic synthetic selection failed")
    return output.sort_values("query_id", kind="mergesort").reset_index(drop=True)


def _validate_plan(plan: dict[str, Any]) -> int:
    roles = plan.get("fold_roles", {})
    if roles.get("train") != list(TRAIN_FOLDS):
        raise ValueError("Plan must train on folds 2/3/4")
    if roles.get("development") != 0 or roles.get("confirmation_closed") != 1:
        raise ValueError("Plan must keep fold 1 closed and use fold 0 as development")
    ratio = plan.get("sampling", {}).get("real_to_synthetic_scene_ratio")
    if ratio != [2, 1]:
        raise ValueError("Only the frozen 2:1 real/synthetic ratio is allowed")
    formula = plan.get("weights", {}).get("synthetic_scene_formula")
    if formula != "0.5 / variants_for_target_siret_in_complete_eligible_corpus":
        raise ValueError("Synthetic weight formula changed")
    if not all(plan.get("prohibitions", {}).values()):
        raise ValueError("Every synthetic safety prohibition must remain active")
    return int(plan.get("seed", 42))


def _eligible_synthetic(
    labels: pd.DataFrame,
    candidates: pd.DataFrame,
    groups: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    labels = labels.copy()
    candidates = candidates.copy()
    groups = groups.copy()
    for frame in (labels, candidates, groups):
        frame["query_id"] = frame["query_id"].astype(str)
    candidates["candidate_siret"] = candidates["candidate_siret"].astype(str).str.zfill(14)
    labels["ground_truth_siret"] = labels["ground_truth_siret"].astype(str).str.zfill(14)
    joined = candidates.merge(
        labels[["query_id", "ground_truth_siret"]], on="query_id", validate="many_to_one"
    )
    joined["is_target"] = joined["candidate_siret"].eq(joined["ground_truth_siret"])
    positive_counts = joined.groupby("query_id", sort=False)["is_target"].sum()
    group_positive_counts = groups.groupby("query_id", sort=False)["is_positive"].sum()
    eligible_ids = set(positive_counts[positive_counts.eq(1)].index) & set(
        group_positive_counts[group_positive_counts.eq(1)].index
    )
    eligible = labels[labels["query_id"].isin(eligible_ids)].copy()
    diagnostics = {
        "published_synthetic_scenes": int(len(labels)),
        "eligible_non_injected_scenes": int(len(eligible)),
        "target_absent_or_non_unique": int(len(labels) - positive_counts.eq(1).sum()),
        "invalid_or_missing_bge_group": int(len(labels) - group_positive_counts.eq(1).sum()),
    }
    return eligible, diagnostics


def build(args: argparse.Namespace) -> Path:
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    seed = _validate_plan(plan)
    real_manifest = _verified_manifest(
        args.real_business, ("candidates_business.parquet", "labels.parquet")
    )
    real_bge_manifest = _verified_manifest(args.real_bge_groups, ("training_groups.parquet",))
    synthetic_manifest = _verified_manifest(
        args.synthetic_bundle,
        ("candidates_business.parquet", "labels.parquet", "training_groups.parquet"),
    )
    if synthetic_manifest.get("schema_version") != SYNTHETIC_BUNDLE_SCHEMA:
        raise ValueError("Unexpected synthetic model-feature bundle schema")
    if synthetic_manifest.get("positive_injection") is not False:
        raise ValueError("Synthetic candidate bundle is not certified non-injected")
    if int(synthetic_manifest.get("candidate_ceiling", -1)) > 100:
        raise ValueError("Synthetic candidate ceiling exceeds 100")
    forbidden = synthetic_manifest.get("allowed_consumers", {})
    if forbidden.get("risk_model") or forbidden.get("calibration") or forbidden.get("auto_thresholds"):
        raise ValueError("Synthetic bundle authorizes a forbidden consumer")

    real_labels = pd.read_parquet(args.real_business / "labels.parquet")
    real_candidates = pd.read_parquet(args.real_business / "candidates_business.parquet")
    real_groups = pd.read_parquet(args.real_bge_groups / "training_groups.parquet")
    synthetic_labels = pd.read_parquet(args.synthetic_bundle / "labels.parquet")
    synthetic_candidates = pd.read_parquet(args.synthetic_bundle / "candidates_business.parquet")
    synthetic_groups = pd.read_parquet(args.synthetic_bundle / "training_groups.parquet")
    synthetic_labels["ground_truth_siret"] = (
        synthetic_labels["ground_truth_siret"].astype(str).str.zfill(14)
    )
    synthetic_labels["ground_truth_siren"] = (
        synthetic_labels["ground_truth_siren"].astype(str).str.zfill(9)
    )

    required_label_columns = {
        "query_id", "label_kind", "ground_truth_siret", "ground_truth_siren",
        "ground_truth_state", "oof_fold", *STRATA,
    }
    missing = required_label_columns - set(synthetic_labels.columns)
    if missing:
        raise ValueError(f"Synthetic labels are missing columns: {sorted(missing)}")
    if not synthetic_labels["label_kind"].eq("MATCH_EXACT").all():
        raise ValueError("Synthetic training labels must all be MATCH_EXACT")
    if not set(synthetic_labels["oof_fold"].astype(int)).issubset(TRAIN_FOLDS):
        raise ValueError("Synthetic labels must be assigned only to folds 2/3/4")
    if synthetic_labels.groupby("ground_truth_siren")["oof_fold"].nunique().max() != 1:
        raise ValueError("A synthetic truth SIREN crosses training folds")
    for frame, name in ((synthetic_candidates, "candidates"), (synthetic_groups, "BGE groups")):
        if frame.empty:
            raise ValueError(f"Synthetic {name} are empty")
    if synthetic_candidates.groupby(synthetic_candidates["query_id"].astype(str)).size().max() > 100:
        raise ValueError("Synthetic candidate pool exceeds 100")
    missing_features = set(BUSINESS_FEATURE_ORDER) - set(synthetic_candidates.columns)
    if missing_features:
        raise ValueError(f"Synthetic business features are missing: {sorted(missing_features)}")

    real_labels["query_id"] = real_labels["query_id"].astype(str)
    synthetic_labels["query_id"] = synthetic_labels["query_id"].astype(str)
    if set(real_labels["query_id"]) & set(synthetic_labels["query_id"]):
        raise ValueError("Real and synthetic query IDs overlap")
    real_truth_sirens = set(
        real_labels.loc[real_labels["label_kind"].eq("MATCH_EXACT"), "ground_truth_siren"].astype(str)
    )
    synthetic_truth_sirens = set(synthetic_labels["ground_truth_siren"].astype(str))
    leaked = real_truth_sirens & synthetic_truth_sirens
    if leaked:
        raise ValueError(f"Synthetic truth SIRENs overlap real folds: {sorted(leaked)[:5]}")

    eligible_synthetic, diagnostics = _eligible_synthetic(
        synthetic_labels, synthetic_candidates, synthetic_groups
    )
    real_train_ids = set(
        real_labels.loc[
            real_labels["oof_fold"].astype(int).isin(TRAIN_FOLDS)
            & real_labels["label_kind"].eq("MATCH_EXACT"),
            "query_id",
        ]
    )
    real_group_ids = set(real_groups["query_id"].astype(str))
    real_scene_count = len(real_train_ids & real_group_ids)
    synthetic_target = real_scene_count // 2
    if len(eligible_synthetic) < synthetic_target:
        raise ValueError(
            f"Only {len(eligible_synthetic)} eligible synthetic scenes; {synthetic_target} required"
        )
    selected_labels = _select_synthetic(eligible_synthetic, synthetic_target, seed)
    selected_ids = set(selected_labels["query_id"])

    identity_counts = (
        eligible_synthetic.groupby("ground_truth_siret", sort=False).size().astype(int).to_dict()
    )
    selected_labels["source_kind"] = "SYNTHETIC_GT"
    selected_labels["scene_weight"] = selected_labels["ground_truth_siret"].map(
        lambda siret: np.float32(0.5 / identity_counts[str(siret)])
    )
    selected_labels["synthetic_identity_variant_count"] = selected_labels[
        "ground_truth_siret"
    ].map(lambda siret: int(identity_counts[str(siret)]))

    selected_candidates = synthetic_candidates[
        synthetic_candidates["query_id"].astype(str).isin(selected_ids)
    ].copy()
    selected_groups = synthetic_groups[
        synthetic_groups["query_id"].astype(str).isin(selected_ids)
    ].copy()
    weight_by_query = selected_labels.set_index("query_id")["scene_weight"]
    selected_groups["scene_weight"] = selected_groups["query_id"].astype(str).map(weight_by_query)
    selected_groups["source_kind"] = "SYNTHETIC_GT"
    if selected_groups.groupby("query_id")["scene_weight"].nunique().max() != 1:
        raise ValueError("A synthetic BGE group carries multiple scene weights")

    real_groups = real_groups.copy()
    real_groups["scene_weight"] = np.float32(1.0)
    real_groups["source_kind"] = "REAL_GT"
    mixed_groups = pd.concat([real_groups, selected_groups], ignore_index=True, sort=False)
    if mixed_groups.groupby("query_id")["is_positive"].sum().ne(1).any():
        raise ValueError("Every mixed BGE group must contain exactly one positive")

    build_identity = {
        "schema_version": SCHEMA_VERSION,
        "builder_sha256": file_sha256(Path(__file__)),
        "plan_sha256": file_sha256(args.plan),
        "real_business_manifest_sha256": file_sha256(args.real_business / "manifest.json"),
        "real_bge_manifest_sha256": file_sha256(args.real_bge_groups / "manifest.json"),
        "synthetic_bundle_manifest_sha256": file_sha256(args.synthetic_bundle / "manifest.json"),
        "train_folds": list(TRAIN_FOLDS),
        "development_fold": 0,
        "real_to_synthetic_scene_ratio": [2, 1],
        "synthetic_scene_weight": "0.5/k",
        "seed": seed,
        "positive_injection": False,
    }
    build_id = hashlib.sha256(
        json.dumps(build_identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    destination = args.output_root / build_id
    if destination.exists():
        existing = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        if existing.get("build_identity") != build_identity:
            raise FileExistsError(f"Conflicting immutable output: {destination}")
        return destination

    args.output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{build_id}.", dir=args.output_root))
    try:
        selected_labels.to_parquet(temporary / "synthetic_labels_selected.parquet", index=False)
        selected_candidates.to_parquet(temporary / "synthetic_candidates_business_selected.parquet", index=False)
        mixed_groups.to_parquet(temporary / "training_groups.parquet", index=False)
        selection = selected_labels[
            ["query_id", "ground_truth_siret", "ground_truth_siren", "oof_fold", *STRATA,
             "scene_weight", "synthetic_identity_variant_count"]
        ].sort_values("query_id", kind="mergesort")
        selection.to_parquet(temporary / "synthetic_selection.parquet", index=False)
        outputs = (
            "synthetic_labels_selected.parquet",
            "synthetic_candidates_business_selected.parquet",
            "training_groups.parquet",
            "synthetic_selection.parquet",
        )
        stratum_counts = {
            "|".join(map(str, key)): int(value)
            for key, value in selected_labels.groupby(list(STRATA), dropna=False).size().sort_index().items()
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "build_identity": build_identity,
            "row_counts": {
                "real_scenes": real_scene_count,
                "synthetic_scenes_selected": len(selected_labels),
                "mixed_bge_scenes": int(mixed_groups["query_id"].nunique()),
            },
            "synthetic_diagnostics": diagnostics,
            "synthetic_strata_selected": stratum_counts,
            "business_feature_order": BUSINESS_FEATURE_ORDER,
            "positive_injection": False,
            "risk_model_allowed": False,
            "calibration_allowed": False,
            "auto_threshold_selection_allowed": False,
            "confirmation_fold_opened": False,
            "final_test_opened": False,
            "outputs": {name: file_sha256(temporary / name) for name in outputs},
        }
        _json_dump(temporary / "manifest.json", manifest)
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-bundle", type=Path, required=True)
    parser.add_argument("--real-business", type=Path, default=DEFAULT_REAL_BUSINESS)
    parser.add_argument("--real-bge-groups", type=Path, default=DEFAULT_REAL_BGE_GROUPS)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    print(build(parse_args()))
