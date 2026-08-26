#!/usr/bin/env python3
"""Truth-side evaluation of a sealed v18 raw parent union."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import resource
import shutil
import sys
import time
import uuid

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.xgb_matcher.retrieval_parent_discovery_v18 import (
    EVALUATION_SCHEMA_VERSION,
    file_record,
    sha256_file,
)


OUTCOME_SCHEMA = pa.schema(
    [
        pa.field("query_id", pa.string(), nullable=False),
        pa.field("fold", pa.int8(), nullable=False),
        pa.field("cohort_kind", pa.string(), nullable=False),
        pa.field("data_origin", pa.string(), nullable=False),
        pa.field("variant", pa.string(), nullable=False),
        pa.field("raw_parent_hit", pa.bool_(), nullable=False),
    ]
)


def _require_sha(record: dict, *, label: str) -> Path:
    path = Path(record["path"])
    if not path.is_file() or sha256_file(path) != str(record["sha256"]):
        raise ValueError(f"v18 evaluation authority drifted: {label}")
    return path


def _validate_generation(root: Path, config: dict) -> tuple[Path, dict]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected_keys = {
        "config_sha256", "files", "fold0_opened", "fold1_opened", "model_trained",
        "parent_selection_performed", "positive_injection", "query_count", "queries_sha256",
        "raw_union_only", "resources", "schema_version", "scope", "site_expansion_performed",
        "truth_columns_opened",
    }
    if set(manifest) != expected_keys:
        raise ValueError("v18 generation manifest keyset drifted")
    generation_config = config["evaluation_authorities"]["generation_config"]
    if manifest["config_sha256"] != str(generation_config["sha256"]):
        raise ValueError("v18 generation config lineage drifted")
    if manifest["query_count"] != int(config["cohort"]["query_count"]):
        raise ValueError("v18 generation query count drifted")
    forbidden = (
        manifest["truth_columns_opened"], manifest["positive_injection"],
        manifest["fold0_opened"], manifest["fold1_opened"], manifest["model_trained"],
        manifest["parent_selection_performed"], manifest["site_expansion_performed"],
    )
    if any(forbidden) or manifest["raw_union_only"] is not True:
        raise ValueError("v18 generation crossed its raw-union boundary")
    if manifest["resources"].get("pass") is not True:
        raise ValueError("v18 generation did not pass its resource policy")
    for name, record in manifest["files"].items():
        path = root / name
        if file_record(path) != record:
            raise ValueError(f"v18 generated artifact drifted: {name}")
    return manifest_path, manifest


def _variant_channel_table(config: dict) -> pa.Table:
    rows = []
    for variant, channels in config["variants"].items():
        for channel in channels:
            rows.append({"variant": str(variant), "channel": str(channel)})
    if not rows or len(rows) != len({(row["variant"], row["channel"]) for row in rows}):
        raise ValueError("v18 variant/channel recipe is empty or duplicated")
    return pa.Table.from_pylist(rows)


def _report(
    rows: list[dict],
    *,
    primary_variant: str,
    gates: dict,
    channels: list[str],
) -> tuple[dict, str]:
    variants = sorted({str(row["variant"]) for row in rows})
    reports: dict[str, dict] = {}
    for variant in variants:
        selected = [row for row in rows if row["variant"] == variant]
        misses = [row for row in selected if row["cohort_kind"] == "BASE_TOP1750_MISS"]
        controls = [row for row in selected if row["cohort_kind"] == "BASE_TOP1750_CONTROL"]
        by_fold = {
            str(fold): sum(bool(row["raw_parent_hit"]) for row in misses if int(row["fold"]) == fold)
            for fold in (2, 3, 4)
        }
        by_origin: dict[str, dict] = {}
        for origin in sorted({str(row["data_origin"]) for row in misses}):
            origin_rows = [row for row in misses if row["data_origin"] == origin]
            by_origin[origin] = {
                "denominator": len(origin_rows),
                "hits": sum(bool(row["raw_parent_hit"]) for row in origin_rows),
            }
        miss_hits = sum(bool(row["raw_parent_hit"]) for row in misses)
        projected = int(gates["sealed_baseline_master_exact_hits"]) + miss_hits
        reports[variant] = {
            "miss_denominator": len(misses),
            "miss_hits": miss_hits,
            "miss_hits_by_fold": by_fold,
            "miss_hits_by_origin": by_origin,
            "control_denominator": len(controls),
            "control_hits": sum(bool(row["raw_parent_hit"]) for row in controls),
            "projected_master_exact_hits": projected,
            "projected_master_exact_recall": projected / int(gates["national_denominator"]),
        }
    primary = reports[primary_variant]
    count = int(primary["miss_hits"])
    fold_gate = gates["confidence_parent_hits_by_fold"]
    confidence = bool(
        count >= int(gates["confidence_parent_hits"])
        and all(
            int(primary["miss_hits_by_fold"][fold]) >= int(minimum)
            for fold, minimum in fold_gate.items()
        )
    )
    if confidence:
        decision = "PASS_PARENT_DISCOVERY_CONFIDENCE"
    elif count >= int(gates["raw_hits_for_master_99_7"]):
        decision = "PASS_MASTER_99_7_RAW_FEASIBILITY"
    elif count >= int(gates["raw_hits_for_master_99_5"]):
        decision = "PASS_MASTER_99_5_RAW_FEASIBILITY"
    elif count >= int(gates["raw_hits_for_master_99_2"]):
        decision = "PASS_MASTER_99_2_RAW_FEASIBILITY_ONLY"
    else:
        decision = "PIVOT_PARENT_SOURCE_DISCOVERY"
    baseline = reports["V17_BASE"]
    checks = {
        "raw_master_99_2_feasible": count >= int(gates["raw_hits_for_master_99_2"]),
        "raw_master_99_5_feasible": count >= int(gates["raw_hits_for_master_99_5"]),
        "raw_master_99_7_feasible": count >= int(gates["raw_hits_for_master_99_7"]),
        "confidence_parent_gate": confidence,
        "recent_origin_non_decreasing": (
            primary["miss_hits_by_origin"]["REAL_CRM_20260817"]["hits"]
            >= baseline["miss_hits_by_origin"]["REAL_CRM_20260817"]["hits"]
        ),
        "historical_origin_non_decreasing": (
            primary["miss_hits_by_origin"]["REAL_CRM_HISTORICAL"]["hits"]
            >= baseline["miss_hits_by_origin"]["REAL_CRM_HISTORICAL"]["hits"]
        ),
        "external_generalization_claim": False,
        "fold0_opened": False,
        "fold1_opened": False,
        "positive_injection": False,
        "model_trained": False,
    }
    return {
        "variant_reports": reports,
        "primary_variant": primary_variant,
        "primary_channel_count": len(channels),
        "checks": checks,
    }, decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate sealed v18 parent discovery")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    generation_root = args.generation_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    config = json.loads(config_path.read_text())
    authorities = config["evaluation_authorities"]
    generation_config = _require_sha(authorities["generation_config"], label="generation_config")
    labels = _require_sha(authorities["cohort_labels"], label="cohort_labels")
    ledger = _require_sha(authorities["label_ledger"], label="label_ledger")
    manifest_path, generation_manifest = _validate_generation(generation_root, config)
    variants = _variant_channel_table(config)
    primary_variant = str(config["primary_variant"])
    if primary_variant not in set(variants.column("variant").to_pylist()):
        raise ValueError("unknown v18 primary variant")
    signal_paths = [generation_root / name for name in generation_manifest["files"]]
    signal_sql = ",".join(f"'{str(path).replace(chr(39), chr(39)*2)}'" for path in signal_paths)
    connection = duckdb.connect(":memory:")
    connection.register("v18_variant_channels", variants)
    started = time.perf_counter()
    rows = connection.execute(
        f"""
        WITH truth AS (
          SELECT c.query_id, cast(c.fold AS TINYINT) fold, c.cohort_kind,
                 substr(c.human_ground_truth_siret,1,9) gt_siren,
                 l.data_origin
          FROM read_parquet('{str(labels).replace(chr(39), chr(39)*2)}') c
          JOIN read_parquet('{str(ledger).replace(chr(39), chr(39)*2)}') l USING(query_id, fold)
        ),
        signals AS (SELECT * FROM read_parquet([{signal_sql}])),
        gt_channels AS (
          SELECT DISTINCT t.query_id, s.channel
          FROM truth t JOIN signals s ON t.query_id=s.query_id AND t.gt_siren=s.siren
        ),
        variant_hits AS (
          SELECT DISTINCT g.query_id, v.variant
          FROM gt_channels g JOIN v18_variant_channels v USING(channel)
        ),
        variant_names AS (SELECT DISTINCT variant FROM v18_variant_channels)
        SELECT t.query_id, t.fold, t.cohort_kind, t.data_origin, v.variant,
               (h.query_id IS NOT NULL) raw_parent_hit
        FROM truth t CROSS JOIN variant_names v
        LEFT JOIN variant_hits h ON h.query_id=t.query_id AND h.variant=v.variant
        ORDER BY t.query_id, v.variant
        """
    ).fetch_arrow_table().to_pylist()
    connection.close()
    report, decision = _report(
        rows,
        primary_variant=primary_variant,
        gates=config["gates"],
        channels=[str(value) for value in config["variants"][primary_variant]],
    )
    stage = output.parent / f".{output.name}.stage-{uuid.uuid4().hex}"
    stage.mkdir(parents=True)
    try:
        outcomes = stage / "parent_query_outcomes.parquet"
        pq.write_table(
            pa.Table.from_pylist(rows, schema=OUTCOME_SCHEMA),
            outcomes,
            compression="zstd",
            compression_level=3,
            use_dictionary=True,
        )
        manifest = {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "config_sha256": sha256_file(config_path),
            "generation_config_sha256": sha256_file(generation_config),
            "generation_manifest_path": str(manifest_path),
            "generation_manifest_sha256": sha256_file(manifest_path),
            "decision": decision,
            "development_only": True,
            "external_generalization_claim": False,
            "parent_selection_performed": False,
            "site_expansion_performed": False,
            "model_trained": False,
            "fold0_opened": False,
            "fold1_opened": False,
            "positive_injection": False,
            **report,
            "resources": {
                "wall_seconds": time.perf_counter() - started,
                "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            },
            "files": {"parent_query_outcomes.parquet": file_record(outcomes)},
        }
        path = stage / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        stage.replace(output)
        path = output / "manifest.json"
        print(json.dumps({"manifest": str(path), "sha256": sha256_file(path), "decision": decision}, sort_keys=True))
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
