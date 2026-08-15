#!/usr/bin/env python3
"""Build an identity-free BGE/XGBoost failure archetype catalog from train OOF.

The catalog contains aggregates only.  It never exports query ids, CRM text or
SIRET/SIREN, and refuses protected folds 0/1.  Synthetic selectors can therefore
target *types* of model failure without copying protected examples or labels.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
import pyarrow.parquet as pq


TRAIN_FOLDS = {2, 3, 4}
FAILURE_CELLS = (
    "BOTH_CORRECT", "BGE_ONLY_CORRECT", "XGB_ONLY_CORRECT", "BOTH_WRONG"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def truthy(value: Any) -> bool:
    return bool(value) and not pd.isna(value)


def scene_archetypes(row: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    name = float(row.get("top1_name_jaro_max") or 0.0)
    address = float(row.get("top1_addr_jaro") or 0.0)
    if float(row.get("top1_same_siren_count") or 0.0) > 1:
        tags.append("SAME_SIREN_COMPETITION")
    if float(row.get("top1_same_address_siren_count") or 0.0) > 1:
        tags.append("SAME_ADDRESS_COMPETITION")
    if name < 0.82 and address >= 0.90:
        tags.append("WEAK_NAME_STRONG_ADDRESS")
    if name >= 0.90 and address < 0.82:
        tags.append("STRONG_NAME_WEAK_ADDRESS")
    if name < 0.82 and address < 0.82:
        tags.append("WEAK_NAME_AND_ADDRESS")
    if float(row.get("top1_top2_score_gap") or 0.0) < 0.10:
        tags.append("LOW_XGB_MARGIN")
    if float(row.get("distinct_siren_count") or 0.0) >= 50:
        tags.append("DENSE_CANDIDATE_SCENE")
    if truthy(row.get("top1_business_role_conflict")):
        tags.append("BUSINESS_ROLE_CONFLICT")
    if str(row.get("ground_truth_state") or "") == "F":
        tags.append("CLOSED_TARGET")
    for field in ("name", "address", "postcode", "city", "insee"):
        if truthy(row.get(f"missing_crm_{field}")):
            tags.append(f"MISSING_{field.upper()}")
    return tags or ["NO_DOMINANT_ARCHETYPE"]


def failure_cell(bge_correct: bool, xgb_correct: bool) -> str:
    if bge_correct and xgb_correct:
        return "BOTH_CORRECT"
    if bge_correct:
        return "BGE_ONLY_CORRECT"
    if xgb_correct:
        return "XGB_ONLY_CORRECT"
    return "BOTH_WRONG"


def read_columns(path: Path, columns: Iterable[str]) -> pd.DataFrame:
    available = set(pq.read_schema(path).names)
    selected = [value for value in columns if value in available]
    return pq.read_table(path, columns=selected).to_pandas()


def build(args: argparse.Namespace) -> dict[str, Any]:
    business_columns = (
        "query_id", "oof_fold", "top1_correct", "ground_truth_state",
        "top1_name_jaro_max", "top1_addr_jaro", "top1_top2_score_gap",
        "top1_same_siren_count", "top1_same_address_siren_count",
        "distinct_siren_count", "top1_business_role_conflict",
        "missing_crm_name", "missing_crm_address", "missing_crm_postcode",
        "missing_crm_city", "missing_crm_insee",
    )
    business = read_columns(args.xgb_scenes, business_columns)
    business = business[business["oof_fold"].isin(TRAIN_FOLDS)].copy()
    if set(int(value) for value in business["oof_fold"].unique()) - TRAIN_FOLDS:
        raise ValueError("protected XGBoost fold entered the catalog")
    bge_frames = []
    for path in args.bge_top1:
        frame = read_columns(path, ("query_id", "oof_fold", "correct"))
        folds = set(int(value) for value in frame["oof_fold"].unique())
        if not folds or not folds.issubset(TRAIN_FOLDS):
            raise ValueError(f"BGE source contains a protected/non-train fold: {path}")
        bge_frames.append(frame.rename(columns={"correct": "bge_correct"}))
    bge = pd.concat(bge_frames, ignore_index=True)
    if bge["query_id"].duplicated().any():
        raise ValueError("duplicate BGE query ids across OOF sources")
    joined = business.merge(
        bge[["query_id", "oof_fold", "bge_correct"]],
        on=["query_id", "oof_fold"], how="inner", validate="one_to_one",
    )
    if len(joined) != len(bge):
        raise ValueError(
            f"incomplete OOF join: business={len(business)} bge={len(bge)} joined={len(joined)}"
        )

    cell_counts: Counter[str] = Counter()
    tag_counts: dict[str, Counter[str]] = {
        cell: Counter() for cell in FAILURE_CELLS
    }
    total_tags: Counter[str] = Counter()
    fold_counts: Counter[str] = Counter()
    for row in joined.to_dict("records"):
        cell = failure_cell(bool(row["bge_correct"]), bool(row["top1_correct"]))
        cell_counts[cell] += 1
        fold_counts[str(int(row["oof_fold"]))] += 1
        for tag in scene_archetypes(row):
            tag_counts[cell][tag] += 1
            total_tags[tag] += 1

    row_count = len(joined)
    cell_profiles: dict[str, Any] = {}
    for cell in FAILURE_CELLS:
        count = cell_counts[cell]
        profiles = []
        for tag, tag_count in tag_counts[cell].most_common():
            cell_rate = tag_count / count if count else 0.0
            global_rate = total_tags[tag] / row_count if row_count else 0.0
            profiles.append({
                "archetype": tag,
                "count": tag_count,
                "within_cell_rate": cell_rate,
                "lift_vs_all_train_oof": cell_rate / global_rate if global_rate else 0.0,
            })
        cell_profiles[cell] = {"count": count, "archetypes": profiles}

    sources = [args.xgb_scenes, *args.bge_top1]
    report = {
        "schema_version": "sireto-synthetic-gt-oof-failure-catalog-1",
        "scope": "TRAIN_OOF_AGGREGATES_ONLY",
        "allowed_folds": sorted(TRAIN_FOLDS),
        "protected_folds_excluded": [0, 1, "test"],
        "contains_query_ids_or_entity_ids": False,
        "row_count": row_count,
        "xgb_train_oof_row_count": len(business),
        "bge_eligible_join_rate": row_count / len(business) if len(business) else 0.0,
        "fold_counts": dict(sorted(fold_counts.items())),
        "failure_cell_counts": dict(cell_counts),
        "cell_profiles": cell_profiles,
        "source_hashes": {str(path): sha256(path) for path in sources},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({**report, "cell_profiles": "omitted"}, ensure_ascii=False, sort_keys=True, indent=2))
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--xgb-scenes", type=Path, required=True)
    result.add_argument("--bge-top1", type=Path, action="append", required=True)
    result.add_argument("--output", type=Path, required=True)
    result.set_defaults(func=build)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
