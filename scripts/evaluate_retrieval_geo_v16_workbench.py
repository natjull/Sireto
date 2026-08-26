#!/usr/bin/env python3
"""Truth-side, foldwise-OOF evaluation of the geo v16 side quest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.xgb_matcher.retrieval_geo_v16 import sha256_file


FEATURE_NAMES = (
    "geo_neighbor_rank",
    "candidate_active",
    "candidate_same_postcode",
    "candidate_number_exact",
    "candidate_street_similarity",
    "ban_street_similarity",
    "ban_certification_commune",
    "ban_uncertainty_m",
    "ban_declared_weak",
    "geo_distance_m",
    "geo_distance_log1p",
    "geo_distance_lower_m",
    "geo_distance_upper_m",
    "geo_distance_uncertainty_ratio",
)
FORBIDDEN_MODEL_FEATURES = {"query_id", "siret", "siren", "fold"}


def _verify_generation(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("truth_columns_opened") is not False or manifest.get("positive_injection") is not False:
        raise ValueError("generation is not label-blind")
    if manifest.get("decision_eligible") is not False:
        raise ValueError("geo side quest must remain diagnostic-only")
    for name, record in manifest["files"].items():
        path = root / name
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise ValueError(f"generation artifact drifted: {path}")
    return manifest


def _feature_matrix(rows: list[dict]) -> np.ndarray:
    matrix = []
    for row in rows:
        values = dict(row)
        values["candidate_active"] = 1.0 if str(row.get("candidate_state") or "") == "A" else 0.0
        matrix.append(
            [float(values[name]) if values.get(name) is not None else np.nan for name in FEATURE_NAMES]
        )
    return np.asarray(matrix, dtype=np.float32)


def _metric_block(records: list[dict], universe_query_ids: set[str]) -> dict:
    exact_best: dict[str, int] = {}
    parent_best: dict[str, int] = {}
    model_best: dict[str, int] = {}
    for row in records:
        query_id = str(row["query_id"])
        rank = int(row["geo_neighbor_rank"])
        if row["is_exact_truth"]:
            exact_best[query_id] = min(rank, exact_best.get(query_id, rank))
            model_rank = int(row["oof_model_rank"])
            model_best[query_id] = min(model_rank, model_best.get(query_id, model_rank))
        if row["is_parent_truth"]:
            parent_best[query_id] = min(rank, parent_best.get(query_id, rank))
    total = len(universe_query_ids)
    return {
        "queries": total,
        "exact_oracle_at_250": sum(rank <= 250 for rank in exact_best.values()),
        "exact_oracle_at_512": sum(rank <= 512 for rank in exact_best.values()),
        "exact_nearest_at_1": sum(rank <= 1 for rank in exact_best.values()),
        "exact_nearest_at_10": sum(rank <= 10 for rank in exact_best.values()),
        "exact_nearest_at_100": sum(rank <= 100 for rank in exact_best.values()),
        "exact_oof_model_at_1": sum(rank <= 1 for rank in model_best.values()),
        "exact_oof_model_at_10": sum(rank <= 10 for rank in model_best.values()),
        "exact_oof_model_at_100": sum(rank <= 100 for rank in model_best.values()),
        "parent_oracle_at_128": sum(rank <= 128 for rank in parent_best.values()),
        "parent_oracle_at_250": sum(rank <= 250 for rank in parent_best.values()),
    }


def _parent_metric_block(records: list[dict], universe_query_ids: set[str]) -> dict:
    best: dict[str, int] = {}
    for row in records:
        if row["is_parent_truth"]:
            query_id = str(row["query_id"])
            rank = int(row["geo_parent_rank"])
            best[query_id] = min(rank, best.get(query_id, rank))
    return {
        "queries": len(universe_query_ids),
        "parent_oracle_at_128": sum(rank <= 128 for rank in best.values()),
        "parent_oracle_at_512": sum(rank <= 512 for rank in best.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the sealed geo v16 workbench")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    generation_root = args.generation_dir.resolve()
    generation = _verify_generation(generation_root)
    if generation.get("execution_mode") != "BOUNDED_DIAGNOSTIC":
        raise ValueError("non-decisional preflight cannot be truth-side evaluated")
    labels_path = args.cohort_dir.resolve() / "cohort_labels.parquet"
    labels = pq.read_table(labels_path).to_pylist()
    if {int(row["fold"]) for row in labels} != {2, 3, 4}:
        raise ValueError("fold0/fold1 exposure refused")
    label_by_query = {str(row["query_id"]): row for row in labels}
    neighbors = pq.read_table(generation_root / "geo_neighbor_candidates.parquet").to_pylist()
    if not set(str(row["query_id"]) for row in neighbors) <= set(label_by_query):
        raise ValueError("generation population differs from truth-side cohort")

    xy_rows = []
    for row in neighbors:
        label = label_by_query[str(row["query_id"])]
        truth = str(label["human_ground_truth_siret"])
        xy_rows.append(
            {
                **row,
                "cohort_kind": str(label["cohort_kind"]),
                "is_exact_truth": str(row["siret"]) == truth,
                "is_parent_truth": str(row["siren"]) == truth[:9],
            }
        )
    parent_rows = pq.read_table(generation_root / "geo_neighbor_parents.parquet").to_pylist()
    for row in parent_rows:
        label = label_by_query[str(row["query_id"])]
        row["cohort_kind"] = str(label["cohort_kind"])
        row["is_parent_truth"] = str(row["siren"]) == str(label["human_ground_truth_siret"])[:9]
    if FEATURE_NAMES and FORBIDDEN_MODEL_FEATURES.intersection(FEATURE_NAMES):
        raise AssertionError("identifier leaked into geo model features")
    x = _feature_matrix(xy_rows)
    y = np.asarray([int(row["is_exact_truth"]) for row in xy_rows], dtype=np.int8)
    folds = np.asarray([int(row["fold"]) for row in xy_rows], dtype=np.int8)
    scores = np.full(len(xy_rows), np.nan, dtype=np.float32)
    model_cfg = config["oof_model"]
    for held_out in (2, 3, 4):
        train, test = folds != held_out, folds == held_out
        positives = int(y[train].sum())
        negatives = int(train.sum()) - positives
        if positives == 0:
            raise ValueError(f"no training positives for held-out fold {held_out}")
        model = XGBClassifier(
            n_estimators=int(model_cfg["n_estimators"]),
            max_depth=int(model_cfg["max_depth"]),
            learning_rate=float(model_cfg["learning_rate"]),
            objective="binary:logistic", eval_metric="logloss", tree_method="hist",
            n_jobs=int(config["resources"]["duckdb_threads"]),
            random_state=int(model_cfg["seed"]),
            scale_pos_weight=min(1000.0, negatives / positives),
        )
        model.fit(x[train], y[train])
        scores[test] = model.predict_proba(x[test])[:, 1]
    if np.isnan(scores).any():
        raise AssertionError("OOF score missing")

    by_query: dict[str, list[int]] = {}
    for index, row in enumerate(xy_rows):
        by_query.setdefault(str(row["query_id"]), []).append(index)
    for indices in by_query.values():
        ordered = sorted(indices, key=lambda i: (-float(scores[i]), int(xy_rows[i]["geo_neighbor_rank"]), str(xy_rows[i]["siret"])))
        for rank, index in enumerate(ordered, start=1):
            xy_rows[index]["oof_score"] = float(scores[index])
            xy_rows[index]["oof_model_rank"] = rank

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    xy_path = output / "geo_neighbors_xy_oof.parquet"
    pq.write_table(pa.Table.from_pylist(xy_rows), xy_path, compression="zstd")

    con = duckdb.connect()
    con.register("labels", pa.Table.from_pylist(labels))
    existing = con.execute(
        """SELECT l.query_id, max(c.siret=l.human_ground_truth_siret) exact_hit
           FROM labels l LEFT JOIN read_parquet(?) c USING(query_id, fold)
           GROUP BY l.query_id""",
        [str(generation_root / "candidate_geo_features.parquet")],
    ).fetchall()
    existing_hits = {str(query_id): bool(hit) for query_id, hit in existing}
    exact_geo_hits = {str(row["query_id"]) for row in xy_rows if row["is_exact_truth"]}
    miss_ids = {str(row["query_id"]) for row in labels if row["cohort_kind"] == "BASE_TOP1750_MISS"}

    all_ids = set(label_by_query)
    fold_ids = {
        fold: {str(row["query_id"]) for row in labels if int(row["fold"]) == fold}
        for fold in (2, 3, 4)
    }
    query_geo = pq.read_table(generation_root / "query_geo.parquet").to_pylist()
    weak_ids = {str(row["query_id"]) for row in query_geo if row["ban_declared_weak"]}
    strong_ids = {str(row["query_id"]) for row in query_geo if row["ban_resolved"] and not row["ban_declared_weak"]}
    certified_ids = {str(row["query_id"]) for row in query_geo if row["ban_certification_commune"]}
    uncertified_ids = all_ids - certified_ids
    unresolved_ids = {str(row["query_id"]) for row in query_geo if not row["ban_resolved"]}
    blocks = {"all": _metric_block(xy_rows, all_ids)}
    for fold in (2, 3, 4):
        blocks[f"fold_{fold}"] = _metric_block(
            [row for row in xy_rows if int(row["fold"]) == fold], fold_ids[fold]
        )
    blocks["ban_declared_weak"] = _metric_block([row for row in xy_rows if str(row["query_id"]) in weak_ids], weak_ids)
    blocks["ban_not_declared_weak"] = _metric_block([row for row in xy_rows if str(row["query_id"]) in strong_ids], strong_ids)
    blocks["ban_certified"] = _metric_block([row for row in xy_rows if str(row["query_id"]) in certified_ids], certified_ids)
    blocks["ban_uncertified"] = _metric_block([row for row in xy_rows if str(row["query_id"]) in uncertified_ids], uncertified_ids)
    blocks["ban_unresolved"] = _metric_block([], unresolved_ids)
    parent_blocks = {"all": _parent_metric_block(parent_rows, all_ids)}
    for fold in (2, 3, 4):
        parent_blocks[f"fold_{fold}"] = _parent_metric_block(
            [row for row in parent_rows if int(row["fold"]) == fold], fold_ids[fold]
        )
    parent_blocks["ban_declared_weak"] = _parent_metric_block([row for row in parent_rows if str(row["query_id"]) in weak_ids], weak_ids)
    parent_blocks["ban_not_declared_weak"] = _parent_metric_block([row for row in parent_rows if str(row["query_id"]) in strong_ids], strong_ids)
    parent_blocks["ban_certified"] = _parent_metric_block([row for row in parent_rows if str(row["query_id"]) in certified_ids], certified_ids)
    parent_blocks["ban_uncertified"] = _parent_metric_block([row for row in parent_rows if str(row["query_id"]) in uncertified_ids], uncertified_ids)
    parent_blocks["ban_unresolved"] = _parent_metric_block([], unresolved_ids)
    report = {
        "schema_version": "sireto-retrieval-geo-v16-evaluation-1",
        "status": "DIAGNOSTIC_ONLY_BAN_SCORE_INELIGIBLE",
        "decision_eligible": False,
        "folds_opened": [2, 3, 4],
        "feature_names": list(FEATURE_NAMES),
        "identifier_features_used": False,
        "positive_injection": False,
        "ban_absence_used_as_negative": False,
        "metrics": blocks,
        "parent_metrics": parent_blocks,
        "existing_v15_exact_hits": sum(existing_hits.values()),
        "geo_exact_hits": len(exact_geo_hits),
        "geo_new_exact_rescues_on_v15_misses": len({q for q in exact_geo_hits & miss_ids if not existing_hits.get(q, False)}),
        "limitations": [
            "BAN score_eligible=false: results cannot authorize production scoring",
            "BAN absence never proves that an address or site is absent",
            "selected 522-miss plus 512-control cohort is diagnostic, not population-representative",
            "distance is a soft feature; no distance threshold rejects a candidate",
            "nearest-parent inventory is truncated after the 4096 nearest sites per query",
        ],
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema_version": "sireto-retrieval-geo-v16-evaluation-manifest-1",
        "generation_manifest_sha256": sha256_file(generation_root / "manifest.json"),
        "cohort_labels_sha256": sha256_file(labels_path),
        "config_sha256": sha256_file(args.config),
        "decision_eligible": False,
        "files": {
            "geo_neighbors_xy_oof.parquet": {"sha256": sha256_file(xy_path), "size_bytes": xy_path.stat().st_size, "rows": len(xy_rows)},
            "report.json": {"sha256": sha256_file(report_path), "size_bytes": report_path.stat().st_size},
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "sha256": sha256_file(manifest_path), "report": report}, sort_keys=True))


if __name__ == "__main__":
    main()
