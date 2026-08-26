#!/usr/bin/env python3
"""Materialize label-blind v19 parent features and the fixed RRF baseline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import uuid

import duckdb
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.xgb_matcher.retrieval_parent_fusion_v19 import (
    SCHEMA_VERSION,
    decode_feature_sql,
    feature_names,
    feature_sql,
    file_record,
    sha256_file,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--query-limit", type=int)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    config_sha = sha256_file(config_path)
    if config.get("schema_version") != "sireto-parent-fusion-v19-config-1":
        raise ValueError("unexpected v19 config schema")
    authorities = config["authorities"]
    generation_manifest = _require(authorities["v18_generation_manifest"], label="generation manifest")
    structured = _require(authorities["structured_signals"], label="structured signals")
    native = _require(authorities["siret_native_signals"], label="SIRET-native signals")
    generation = json.loads(generation_manifest.read_text())
    if generation.get("truth_columns_opened") is not False or generation.get("positive_injection") is not False:
        raise ValueError("v18 generation is not label blind")
    for path, key in ((structured, "structured_parent_signals.parquet"), (native, "siret_native_parent_signals.parquet")):
        if generation["files"][key]["sha256"] != sha256_file(path):
            raise ValueError("v18 signal is not bound to its manifest")

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / f".{output.name}.stage-{uuid.uuid4().hex}"
    stage.mkdir()
    started = time.perf_counter()
    swap_before = _swapouts()
    channels = tuple(str(value) for value in config["channels"])
    features = feature_names(channels)
    rrf_k = int(config["fusion"]["rrf_k"])
    cap = int(config["fusion"]["parent_cap"])
    expression_payload = feature_sql(channels, rrf_k=rrf_k)
    expressions, expression_parameters = decode_feature_sql(expression_payload)
    database = stage / "work.duckdb"
    connection = duckdb.connect(str(database))
    try:
        connection.execute(f"SET memory_limit='{config['resources']['duckdb_memory_limit']}'")
        connection.execute(f"SET threads={int(config['resources']['duckdb_threads'])}")
        connection.execute(f"SET temp_directory='{str(stage / 'duckdb_tmp').replace(chr(39), chr(39)*2)}'")
        channel_placeholders = ",".join("?" for _ in channels)
        query_limit_sql = ""
        parameters: list[object] = [str(structured), str(native), *channels]
        if args.query_limit is not None:
            if not 1 <= args.query_limit <= int(config["population"]["query_count"]):
                raise ValueError("invalid v19 preflight query limit")
            query_limit_sql = "JOIN (SELECT DISTINCT query_id FROM raw_signals ORDER BY query_id LIMIT ?) chosen USING(query_id)"
            parameters.append(int(args.query_limit))
        parameters.extend(expression_parameters)
        connection.execute(
            f"""
            CREATE TABLE parent_features AS
            WITH raw_signals AS (
              SELECT * FROM read_parquet([?, ?])
              WHERE channel IN ({channel_placeholders})
            ), selected AS (
              SELECT r.* FROM raw_signals r {query_limit_sql}
            )
            SELECT query_id, fold, siren,
                   {expressions}
            FROM selected
            GROUP BY query_id, fold, siren
            """,
            parameters,
        )
        observed_channels = {
            str(row[0])
            for row in connection.execute("SELECT DISTINCT channel FROM read_parquet([?, ?])", [str(structured), str(native)]).fetchall()
        }
        if observed_channels != set(channels):
            raise ValueError("v19 channel inventory drifted")
        query_count, feature_rows, min_count, max_count = connection.execute(
            """
            SELECT count(*), sum(n), min(n), max(n)
            FROM (SELECT query_id, count(*) n FROM parent_features GROUP BY query_id)
            """
        ).fetchone()
        expected_queries = args.query_limit or int(config["population"]["query_count"])
        if int(query_count) != expected_queries or int(feature_rows) <= 0:
            raise ValueError("v19 feature population drifted")
        feature_path = stage / "parent_features.parquet"
        connection.execute(
            f"COPY (SELECT * FROM parent_features ORDER BY fold, query_id, siren) TO '{str(feature_path).replace(chr(39), chr(39)*2)}' (FORMAT PARQUET, COMPRESSION ZSTD, COMPRESSION_LEVEL 3, ROW_GROUP_SIZE 65536)"
        )
        rrf_path = stage / "rrf_parent_top2000.parquet"
        connection.execute(
            f"""
            COPY (
              SELECT query_id, fold, siren,
                     row_number() OVER (PARTITION BY query_id ORDER BY rrf_score DESC, channel_count DESC, best_rank_reciprocal DESC, siren)::INTEGER parent_rank,
                     rrf_score
              FROM parent_features
              QUALIFY parent_rank <= {cap}
              ORDER BY fold, query_id, parent_rank
            ) TO '{str(rrf_path).replace(chr(39), chr(39)*2)}'
            (FORMAT PARQUET, COMPRESSION ZSTD, COMPRESSION_LEVEL 3, ROW_GROUP_SIZE 65536)
            """
        )
    finally:
        connection.close()
    if database.exists():
        database.unlink()
    wal = Path(f"{database}.wal")
    if wal.exists():
        wal.unlink()
    temporary = stage / "duckdb_tmp"
    if temporary.exists():
        shutil.rmtree(temporary)

    swap_after = _swapouts()
    wall = time.perf_counter() - started
    peak = _peak_rss_bytes()
    resources = config["resources"]
    passed = (
        wall <= float(resources["wall_seconds_max"])
        and peak <= int(resources["peak_rss_bytes_max"])
        and swap_after - swap_before <= int(resources["swapout_pages_delta_max"])
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "LABEL_BLIND_PARENT_FEATURES_AND_FIXED_RRF",
        "config_sha256": config_sha,
        "v18_generation_manifest_path": str(generation_manifest),
        "v18_generation_manifest_sha256": sha256_file(generation_manifest),
        "query_count": int(query_count),
        "feature_row_count": int(feature_rows),
        "min_parent_count": int(min_count),
        "max_parent_count": int(max_count),
        "feature_order": list(features),
        "rrf_k": rrf_k,
        "parent_cap": cap,
        "truth_columns_opened": False,
        "positive_injection": False,
        "fold0_opened": False,
        "fold1_opened": False,
        "files": {
            "parent_features.parquet": file_record(stage / "parent_features.parquet"),
            "rrf_parent_top2000.parquet": file_record(stage / "rrf_parent_top2000.parquet"),
        },
        "resources": {
            "wall_seconds": wall,
            "peak_rss_bytes": peak,
            "swapout_pages_before": swap_before,
            "swapout_pages_after": swap_after,
            "swapout_pages_delta": swap_after - swap_before,
            "pass": passed,
        },
        "preflight_query_limit": args.query_limit,
    }
    manifest_path = stage / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if not passed:
        raise RuntimeError("v19 feature materialization breached resource policy")
    os.replace(stage, output)
    final_manifest = output / "manifest.json"
    print(json.dumps({"manifest": str(final_manifest), "sha256": sha256_file(final_manifest)}, sort_keys=True))


if __name__ == "__main__":
    main()
