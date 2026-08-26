#!/usr/bin/env python3
"""Train and evaluate the frozen v19 parent@2000 OOF LambdaMART."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import uuid

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.xgb_matcher.retrieval_parent_fusion_v19 import (
    ALLOWED_FOLDS,
    OOF_SCHEMA,
    SCHEMA_VERSION,
    assert_identifier_free_features,
    file_record,
    fit_ranker,
    sha256_file,
    validate_fold_isolation,
)


def _require(record: dict, *, label: str) -> Path:
    path = Path(str(record["path"])).resolve()
    if not path.is_file() or sha256_file(path) != str(record["sha256"]):
        raise ValueError(f"v19 {label} authority drifted")
    return path


def _swapouts() -> int:
    output = subprocess.check_output(["vm_stat"], text=True)
    for line in output.splitlines():
        if line.startswith("Swapouts:"):
            return int(line.split(":", 1)[1].strip().rstrip("."))
    raise RuntimeError("vm_stat did not expose Swapouts")


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _truth(labels: Path, ledger: Path) -> pd.DataFrame:
    connection = duckdb.connect()
    try:
        frame = connection.execute(
            """
            SELECT c.query_id, c.fold::TINYINT fold, c.cohort_kind,
                   l.data_origin, substr(c.human_ground_truth_siret,1,9) gt_siren
            FROM read_parquet(?) c
            JOIN read_parquet(?) l USING(query_id, fold)
            ORDER BY c.fold, c.query_id
            """,
            [str(labels), str(ledger)],
        ).df()
    finally:
        connection.close()
    validate_fold_isolation(frame)
    return frame


def _variant_report(rows: pd.DataFrame, *, gates: dict) -> dict:
    reports: dict[str, dict] = {}
    for variant, group in rows.groupby("variant", sort=True):
        miss = group.loc[group["cohort_kind"].eq("BASE_TOP1750_MISS")]
        controls = group.loc[group["cohort_kind"].eq("BASE_TOP1750_CONTROL")]
        miss_hits = int(miss["parent_hit"].sum())
        reports[str(variant)] = {
            "miss_denominator": int(len(miss)),
            "miss_hits": miss_hits,
            "miss_hits_by_fold": {
                str(fold): int(miss.loc[miss["fold"].eq(fold), "parent_hit"].sum())
                for fold in ALLOWED_FOLDS
            },
            "miss_hits_by_origin": {
                str(origin): {
                    "denominator": int(len(origin_rows)),
                    "hits": int(origin_rows["parent_hit"].sum()),
                }
                for origin, origin_rows in miss.groupby("data_origin", sort=True)
            },
            "control_denominator": int(len(controls)),
            "control_hits": int(controls["parent_hit"].sum()),
            "max_parent_rank": int(group.loc[group["parent_hit"], "parent_rank"].max())
            if group["parent_hit"].any()
            else None,
            "projected_master_exact_hits": 23049 + miss_hits,
            "projected_master_exact_recall": (23049 + miss_hits) / 23571,
        }
    primary = reports["OOF_LAMBDAMART"]
    fold_hits = primary["miss_hits_by_fold"]
    origin_hits = primary["miss_hits_by_origin"]
    confidence = (
        primary["miss_hits"] >= int(gates["confidence_parent_hits"])
        and all(
            int(fold_hits[str(fold)]) >= int(gates["confidence_parent_hits_by_fold"][str(fold)])
            for fold in ALLOWED_FOLDS
        )
    )
    origins_non_decreasing = all(
        int(origin_hits[origin]["hits"]) >= int(baseline)
        for origin, baseline in gates["baseline_v17_hits_by_origin"].items()
    )
    if primary["control_hits"] != int(gates["controls_required"]):
        decision = "STOP_PARENT_2000_CONTROL_REGRESSION"
    elif not origins_non_decreasing:
        decision = "STOP_PARENT_2000_ORIGIN_REGRESSION"
    elif confidence:
        decision = "PASS_PARENT_2000_CONFIDENCE"
    elif primary["miss_hits"] >= int(gates["raw_hits_for_master_99_7"]):
        decision = "PASS_PARENT_2000_MASTER_99_7_DEVELOPMENT"
    elif primary["miss_hits"] >= int(gates["raw_hits_for_master_99_5"]):
        decision = "PASS_PARENT_2000_MASTER_99_5_DEVELOPMENT"
    elif primary["miss_hits"] >= int(gates["raw_hits_for_master_99_2"]):
        decision = "PASS_PARENT_2000_MASTER_99_2_DEVELOPMENT"
    else:
        decision = "PIVOT_PARENT_2000"
    return {
        "variant_reports": reports,
        "checks": {
            "controls_complete": primary["control_hits"] == int(gates["controls_required"]),
            "origins_non_decreasing": origins_non_decreasing,
            "master_99_2_feasible_at_parent_2000": primary["miss_hits"] >= int(gates["raw_hits_for_master_99_2"]),
            "master_99_5_feasible_at_parent_2000": primary["miss_hits"] >= int(gates["raw_hits_for_master_99_5"]),
            "confidence_parent_gate": confidence,
            "external_generalization_claim": False,
        },
        "decision": decision,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != "sireto-parent-fusion-v19-config-1":
        raise ValueError("unexpected v19 config schema")
    config_sha = sha256_file(config_path)
    feature_root = args.feature_dir.resolve()
    feature_manifest_path = feature_root / "manifest.json"
    feature_manifest = json.loads(feature_manifest_path.read_text())
    if feature_manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("v19 feature manifest schema drifted")
    if feature_manifest.get("config_sha256") != config_sha:
        raise ValueError("v19 feature/config lineage drifted")
    if feature_manifest.get("truth_columns_opened") is not False:
        raise ValueError("v19 features are not label blind")
    features_path = feature_root / "parent_features.parquet"
    rrf_path = feature_root / "rrf_parent_top2000.parquet"
    for path in (features_path, rrf_path):
        record = feature_manifest["files"][path.name]
        if sha256_file(path) != record["sha256"] or path.stat().st_size != record["size_bytes"]:
            raise ValueError("v19 feature artifact drifted")
    features = tuple(str(value) for value in feature_manifest["feature_order"])
    assert_identifier_free_features(features)
    labels = _require(config["authorities"]["cohort_labels"], label="cohort labels")
    ledger = _require(config["authorities"]["label_ledger"], label="label ledger")
    truth = _truth(labels, ledger)
    if len(truth) != int(config["population"]["query_count"]):
        raise ValueError("v19 truth population drifted")

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / f".{output.name}.stage-{uuid.uuid4().hex}"
    (stage / "models").mkdir(parents=True)
    started = time.perf_counter()
    swap_before = _swapouts()
    cap = int(config["fusion"]["parent_cap"])
    negative_limit = int(config["fusion"]["training_negative_limit_per_query"])
    outer_protocol = {int(key): tuple(int(value) for value in values) for key, values in config["fusion"]["outer_protocol"].items()}
    truth_table = pa.Table.from_pandas(truth[["query_id", "fold", "gt_siren"]], preserve_index=False)
    fold_paths: list[Path] = []
    model_records: dict[str, dict] = {}
    training_diagnostics: dict[str, dict] = {}
    connection = duckdb.connect()
    connection.execute(f"SET memory_limit='{config['resources']['duckdb_memory_limit']}'")
    connection.execute(f"SET threads={int(config['resources']['duckdb_threads'])}")
    connection.register("truth_rows", truth_table)
    try:
        feature_projection = ", ".join(f"f.{name}" for name in features)
        for outer_fold in ALLOWED_FOLDS:
            train_folds = outer_protocol[outer_fold]
            if set(train_folds) != set(ALLOWED_FOLDS) - {outer_fold}:
                raise ValueError("v19 outer training folds drifted")
            placeholders = ",".join("?" for _ in train_folds)
            training = connection.execute(
                f"""
                WITH joined AS (
                  SELECT f.query_id, f.siren, {feature_projection},
                         CASE WHEN f.siren=t.gt_siren THEN 1 ELSE 0 END::TINYINT is_positive
                  FROM read_parquet(?) f JOIN truth_rows t USING(query_id, fold)
                  WHERE f.fold IN ({placeholders})
                ), eligible AS (
                  SELECT * FROM joined
                  QUALIFY max(is_positive) OVER (PARTITION BY query_id)=1
                      AND count(*) OVER (PARTITION BY query_id)>1
                )
                SELECT *,
                       (row_number() OVER (
                         PARTITION BY query_id
                         ORDER BY is_positive DESC, rrf_score DESC, channel_count DESC, best_rank_reciprocal DESC, siren
                       ) - 1)::INTEGER negative_rank
                FROM eligible
                ORDER BY query_id, is_positive DESC, negative_rank
                """,
                [str(features_path), *train_folds],
            ).df()
            model, diagnostics = fit_ranker(
                training,
                features=features,
                params=config["xgboost"],
                negative_limit=negative_limit,
            )
            diagnostics["outer_fold"] = outer_fold
            diagnostics["train_folds"] = list(train_folds)
            training_diagnostics[str(outer_fold)] = diagnostics
            model_path = stage / "models" / f"outer_fold{outer_fold}.json"
            model.save_model(model_path)
            model_records[str(outer_fold)] = {"path": str(model_path.relative_to(stage)), "sha256": sha256_file(model_path), "size_bytes": model_path.stat().st_size}
            outer = connection.execute(
                f"""
                SELECT f.query_id, f.fold, f.siren, {feature_projection}
                FROM read_parquet(?) f
                WHERE f.fold=?
                ORDER BY f.query_id, f.siren
                """,
                [str(features_path), outer_fold],
            ).df()
            X = outer[list(features)].fillna(0.0).to_numpy(dtype=np.float32)
            scores = model.predict(X).astype(np.float32, copy=False)
            scored = pa.table(
                {
                    "query_id": pa.array(outer["query_id"].astype(str)),
                    "fold": pa.array(outer["fold"].astype(np.int8), type=pa.int8()),
                    "siren": pa.array(outer["siren"].astype(str)),
                    "ltr_score": pa.array(scores, type=pa.float32()),
                    "rrf_score": pa.array(outer["rrf_score"].astype(np.float32), type=pa.float32()),
                }
            )
            connection.register("outer_scored", scored)
            selected = connection.execute(
                f"""
                SELECT query_id, fold, siren,
                       row_number() OVER (
                         PARTITION BY query_id
                         ORDER BY ltr_score DESC, rrf_score DESC, siren
                       )::INTEGER parent_rank,
                       ltr_score, rrf_score
                FROM outer_scored
                QUALIFY parent_rank <= {cap}
                ORDER BY fold, query_id, parent_rank
                """
            ).fetch_arrow_table().cast(OOF_SCHEMA)
            fold_path = stage / f"oof_parent_top2000_fold{outer_fold}.parquet"
            pq.write_table(selected, fold_path, compression="zstd", compression_level=3, row_group_size=65536)
            fold_paths.append(fold_path)
            connection.unregister("outer_scored")
            del training, outer, X, scores, scored, selected, model
        oof_path = stage / "oof_parent_top2000.parquet"
        escaped = ",".join(f"'{str(path).replace(chr(39), chr(39)*2)}'" for path in fold_paths)
        connection.execute(
            f"COPY (SELECT * FROM read_parquet([{escaped}]) ORDER BY fold, query_id, parent_rank) TO '{str(oof_path).replace(chr(39), chr(39)*2)}' (FORMAT PARQUET, COMPRESSION ZSTD, COMPRESSION_LEVEL 3, ROW_GROUP_SIZE 65536)"
        )
        truth_full = pa.Table.from_pandas(truth, preserve_index=False)
        connection.register("truth_full", truth_full)
        outcomes = connection.execute(
            """
            WITH candidates AS (
              SELECT 'FIXED_RRF' variant, query_id, siren, parent_rank FROM read_parquet(?)
              UNION ALL
              SELECT 'OOF_LAMBDAMART' variant, query_id, siren, parent_rank FROM read_parquet(?)
            ), variants AS (
              SELECT unnest(['FIXED_RRF','OOF_LAMBDAMART']) variant
            ), hits AS (
              SELECT c.variant, c.query_id, min(c.parent_rank)::INTEGER parent_rank
              FROM candidates c JOIN truth_full t ON c.query_id=t.query_id AND c.siren=t.gt_siren
              GROUP BY c.variant,c.query_id
            )
            SELECT t.query_id,t.fold,t.cohort_kind,t.data_origin,v.variant,
                   (h.query_id IS NOT NULL) parent_hit,h.parent_rank
            FROM truth_full t CROSS JOIN variants v
            LEFT JOIN hits h ON t.query_id=h.query_id AND v.variant=h.variant
            ORDER BY v.variant,t.fold,t.query_id
            """,
            [str(rrf_path), str(oof_path)],
        ).df()
    finally:
        connection.close()
    outcomes_path = stage / "parent_query_outcomes.parquet"
    pq.write_table(pa.Table.from_pandas(outcomes, preserve_index=False), outcomes_path, compression="zstd", compression_level=3)
    report = _variant_report(outcomes, gates=config["gates"])
    swap_after = _swapouts()
    wall = time.perf_counter() - started
    peak = _peak_rss_bytes()
    resource_policy = config["resources"]
    resource_pass = (
        wall <= float(resource_policy["wall_seconds_max"])
        and peak <= int(resource_policy["peak_rss_bytes_max"])
        and swap_after - swap_before <= int(resource_policy["swapout_pages_delta_max"])
    )
    if not resource_pass:
        report["decision"] = "STOP_PARENT_2000_RESOURCE_POLICY"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "OUTER_OOF_PARENT_TOP2000_EVALUATION",
        "config_sha256": config_sha,
        "feature_manifest_path": str(feature_manifest_path),
        "feature_manifest_sha256": sha256_file(feature_manifest_path),
        "parent_cap": cap,
        "feature_order": list(features),
        "outer_protocol": {str(key): list(value) for key, value in outer_protocol.items()},
        "training_diagnostics": training_diagnostics,
        "models": model_records,
        "files": {
            "oof_parent_top2000.parquet": file_record(stage / "oof_parent_top2000.parquet"),
            "parent_query_outcomes.parquet": file_record(outcomes_path),
            **{path.name: file_record(path) for path in fold_paths},
        },
        **report,
        "checks": {
            **report["checks"],
            "positive_injection": False,
            "fold0_opened": False,
            "fold1_opened": False,
            "features_sealed_before_truth": True,
            "outer_test_labels_used_for_fit": False,
            "parent_cap_respected": True,
            "resource_policy_pass": resource_pass,
        },
        "resources": {
            "wall_seconds": wall,
            "peak_rss_bytes": peak,
            "swapout_pages_before": swap_before,
            "swapout_pages_after": swap_after,
            "swapout_pages_delta": swap_after - swap_before,
            "pass": resource_pass,
        },
        "development_only": True,
        "external_generalization_claim": False,
        "national_run_authorized": False,
    }
    manifest_path = stage / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(stage, output)
    final_manifest = output / "manifest.json"
    print(json.dumps({"manifest": str(final_manifest), "sha256": sha256_file(final_manifest), "decision": manifest["decision"]}, sort_keys=True))


if __name__ == "__main__":
    main()
