#!/usr/bin/env python3
"""Generate the label-blind v17 address-parent and exhaustive-site surfaces."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.xgb_matcher.retrieval_geo_v16 import normalize_street
from src.xgb_matcher.retrieval_parent_v16 import fuse_parent_signals
from src.xgb_matcher.retrieval_site_inventory_v17 import (
    build_parent_rows,
    sha256_file,
    split_street_number,
    validate_query_selection,
    write_generation_manifest,
)
from src.xgb_matcher.retrieval_v12_scene_ir import SceneIRCache


def _swapouts() -> int:
    try:
        output = subprocess.check_output(["/usr/bin/vm_stat"], text=True)
        for line in output.splitlines():
            if line.startswith("Swapouts:"):
                return int(line.split(":", 1)[1].strip().rstrip("."))
    except Exception:
        return -1
    return -1


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _sql_path(path: Path) -> str:
    return str(Path(path).resolve()).replace("'", "''")


def _resolve(path: str) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def _verify(path: Path, expected: str) -> None:
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"authority drifted: {path}")


def _verify_input_manifest(path: Path, expected: str, *, require_label_blind: bool = True) -> dict[str, Any]:
    _verify(path, expected)
    raw = json.loads(path.read_text())
    if require_label_blind and (
        raw.get("truth_columns_opened") is not False
        or raw.get("positive_injection") is not False
    ):
        raise ValueError(f"input manifest is not label-blind: {path}")
    return raw


def _query_surface(
    config: dict[str, Any], query_rows: tuple[tuple[str, int], ...]
) -> pa.Table:
    scene = SceneIRCache(
        Path(config["authorities"]["scene_ir_dir"]), verify_source_hash=False
    )
    expected = dict(query_rows)
    columns = [
        "query_id",
        "fold",
        "crm_name",
        "crm_address",
        "crm_number",
        "crm_postcode",
        "crm_insee",
        "postcode",
        "insee",
        "union_rank",
    ]
    rows: list[dict[str, Any]] = []
    for scene_query in scene.iter_selected_scenes(expected, columns=columns):
        if scene_query.fold != expected[scene_query.query_id]:
            raise ValueError("query fold differs from sealed SceneIR")
        first = scene_query.rows.slice(0, 1).to_pylist()[0]
        raw_number = str(first.get("crm_number") or "")
        number, suffix = split_street_number(raw_number)
        rows.append(
            {
                "query_id": scene_query.query_id,
                "fold": scene_query.fold,
                "crm_name": str(first.get("crm_name") or ""),
                "crm_number_raw": raw_number,
                "crm_number": number,
                "crm_suffix": suffix,
                "crm_street_norm": normalize_street(
                    str(first.get("crm_address") or ""), raw_number
                ),
                "crm_postcode": str(
                    first.get("crm_postcode") or first.get("postcode") or ""
                ),
                "crm_insee": str(first.get("crm_insee") or first.get("insee") or ""),
            }
        )
    if len(rows) != len(query_rows):
        raise ValueError("SceneIR did not yield the sealed query population")
    return pa.Table.from_pylist(rows)


def _write_address_signals(
    connection: duckdb.DuckDBPyConnection,
    *,
    queries: pa.Table,
    sidecar_path: Path,
    output_path: Path,
    minimum_similarity: float,
    cap: int,
) -> None:
    connection.register("v17_queries", queries)
    source = _sql_path(sidecar_path)
    destination = _sql_path(output_path)
    connection.execute(
        f"""
        COPY (
          WITH pairs AS (
            SELECT q.query_id, q.fold, s.siren, s.siret,
                   s.administrative_state, s.is_headquarters,
                   (q.crm_insee <> '' AND s.insee = q.crm_insee)::TINYINT same_insee,
                   (q.crm_postcode <> '' AND s.postcode = q.crm_postcode)::TINYINT same_postcode,
                   (q.crm_number <> '' AND s.street_number = q.crm_number)::TINYINT number_exact,
                   CASE WHEN q.crm_suffix = '' THEN 1
                        WHEN upper(coalesce(s.street_number_suffix, '')) = q.crm_suffix THEN 1
                        ELSE 0 END::TINYINT suffix_compatible,
                   CASE WHEN q.crm_street_norm <> '' AND s.street_normalized <> ''
                        THEN jaro_winkler_similarity(q.crm_street_norm, s.street_normalized)
                        ELSE 0 END street_similarity
            FROM v17_queries q
            JOIN read_parquet('{source}') s
              ON ((q.crm_insee <> '' AND s.insee = q.crm_insee)
                  OR (q.crm_insee = '' AND q.crm_postcode <> '' AND s.postcode = q.crm_postcode))
             AND ((q.crm_number <> '' AND s.street_number = q.crm_number)
                  OR (q.crm_number = '' AND q.crm_street_norm <> ''
                      AND s.street_normalized = q.crm_street_norm))
          ), eligible AS (
            SELECT *, count(*) OVER (PARTITION BY query_id, siren) matching_site_count,
                   (number_exact = 1 AND suffix_compatible = 1
                    AND street_similarity = 1.0)::TINYINT exact_cell
            FROM pairs
            WHERE street_similarity >= {float(minimum_similarity)}
          ), best_site AS (
            SELECT *, row_number() OVER (
              PARTITION BY query_id, siren
              ORDER BY exact_cell DESC, number_exact DESC, suffix_compatible DESC,
                       street_similarity DESC,
                       (administrative_state = 'A') DESC, is_headquarters DESC, siret
            ) site_choice_rank
            FROM eligible
          ), parents AS (
            SELECT query_id, fold, siren, siret best_address_siret,
                   same_insee, same_postcode, number_exact, suffix_compatible,
                   street_similarity, exact_cell,
                   administrative_state best_site_state,
                   is_headquarters best_site_is_headquarters,
                   matching_site_count
            FROM best_site WHERE site_choice_rank = 1
          ), ranked AS (
            SELECT *, row_number() OVER (
              PARTITION BY query_id
              ORDER BY exact_cell DESC, same_insee DESC, number_exact DESC,
                       suffix_compatible DESC, street_similarity DESC,
                       (best_site_state = 'A') DESC,
                       best_site_is_headquarters DESC, siren
            ) address_parent_rank
            FROM parents
          )
          SELECT query_id, fold, cast(address_parent_rank AS SMALLINT) address_parent_rank,
                 siren, best_address_siret, same_insee, same_postcode,
                 number_exact, suffix_compatible, street_similarity, exact_cell,
                 best_site_state, best_site_is_headquarters,
                 cast(matching_site_count AS INTEGER) matching_site_count
          FROM ranked WHERE address_parent_rank <= {int(cap)}
          ORDER BY query_id, address_parent_rank
        ) TO '{destination}'
          (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )


def _text_orders(
    signals: list[dict[str, Any]], parent_config: dict[str, Any]
) -> dict[str, list[str]]:
    primary_name = str(parent_config["evaluation"]["primary_ablation"])
    ablation = next(
        row for row in parent_config["ablations"] if str(row["name"]) == primary_name
    )
    allowed = set(str(value) for value in ablation["channels"])
    weights = {
        str(row["name"]): float(row.get("weight", 1.0))
        for row in parent_config["channels"]
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in signals:
        grouped[str(row["query_id"])].append(row)
    output: dict[str, list[str]] = {}
    for query_id, rows in grouped.items():
        fused = [
            str(row["siren"])
            for row in fuse_parent_signals(
                rows,
                allowed_channels=allowed,
                channel_weights=weights,
                cap=512,
            )
        ]
        best_raw_rank: dict[str, int] = {}
        for row in rows:
            siren = str(row["siren"])
            rank = int(row["channel_rank"])
            best_raw_rank[siren] = min(best_raw_rank.get(siren, rank), rank)
        remaining = sorted(
            set(best_raw_rank) - set(fused), key=lambda siren: (best_raw_rank[siren], siren)
        )
        output[query_id] = fused + remaining
    return output


def _write_expanded_sites(
    connection: duckdb.DuckDBPyConnection,
    *,
    primary_parents: pa.Table,
    queries: pa.Table,
    sidecar_path: Path,
    output_path: Path,
) -> None:
    connection.register("v17_primary_parents", primary_parents)
    connection.register("v17_queries", queries)
    source = _sql_path(sidecar_path)
    destination = _sql_path(output_path)
    connection.execute(
        f"""
        COPY (
          WITH selected_sirens AS (
            SELECT DISTINCT siren FROM v17_primary_parents
          ), selected_sites AS (
            SELECT s.* FROM read_parquet('{source}') s
            JOIN selected_sirens USING(siren)
          ), source_counts AS (
            SELECT siren, count(*) source_site_count
            FROM selected_sites GROUP BY siren
          ), expanded AS (
            SELECT p.query_id, p.fold, p.parent_rank, p.siren,
                   s.siret, s.insee, s.postcode, s.street_number,
                   s.street_number_suffix, s.street_normalized,
                   s.current_address_normalized, s.administrative_state,
                   s.is_headquarters, c.source_site_count,
                   q.crm_number, q.crm_suffix, q.crm_street_norm,
                   q.crm_insee, q.crm_postcode,
                   (q.crm_insee <> '' AND s.insee = q.crm_insee)::TINYINT same_insee,
                   (q.crm_postcode <> '' AND s.postcode = q.crm_postcode)::TINYINT same_postcode,
                   (q.crm_number <> '' AND s.street_number = q.crm_number)::TINYINT number_exact,
                   CASE WHEN q.crm_suffix = '' THEN 1
                        WHEN upper(coalesce(s.street_number_suffix, '')) = q.crm_suffix THEN 1
                        ELSE 0 END::TINYINT suffix_compatible,
                   CASE WHEN q.crm_street_norm <> '' AND s.street_normalized <> ''
                        THEN jaro_winkler_similarity(q.crm_street_norm, s.street_normalized)
                        ELSE 0 END street_similarity
            FROM v17_primary_parents p
            JOIN selected_sites s USING(siren)
            JOIN source_counts c USING(siren)
            JOIN v17_queries q USING(query_id, fold)
          ), ranked AS (
            SELECT *, count(*) OVER (PARTITION BY query_id, siren) emitted_site_count,
                   row_number() OVER (
                     PARTITION BY query_id, siren
                     ORDER BY (number_exact=1 AND suffix_compatible=1
                               AND street_similarity=1.0) DESC,
                              same_insee DESC, number_exact DESC,
                              suffix_compatible DESC, street_similarity DESC,
                              (administrative_state='A') DESC,
                              is_headquarters DESC, siret
                   ) site_address_rank
            FROM expanded
          )
          SELECT query_id, fold, parent_rank, siren,
                 cast(site_address_rank AS INTEGER) site_address_rank,
                 siret, insee, postcode, street_number, street_number_suffix,
                 street_normalized, current_address_normalized,
                 administrative_state, is_headquarters,
                 cast(source_site_count AS INTEGER) source_site_count,
                 cast(emitted_site_count AS INTEGER) emitted_site_count,
                 same_insee, same_postcode, number_exact, suffix_compatible,
                 street_similarity,
                 (number_exact=1 AND suffix_compatible=1
                  AND street_similarity=1.0
                  AND (same_insee=1 OR same_postcode=1))::TINYINT strict_same_site
          FROM ranked
          ORDER BY query_id, parent_rank, site_address_rank
        ) TO '{destination}'
          (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate v17 exhaustive site inventory")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-query-count", type=int)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    authorities = config["authorities"]
    query_rows = validate_query_selection(args.queries)
    if args.preflight_query_count is not None:
        if not 1 <= args.preflight_query_count <= 16:
            raise ValueError("preflight query count must be within 1..16")
        query_rows = query_rows[: args.preflight_query_count]
        mode = "NON_DECISIONAL_PREFLIGHT"
    else:
        if len(query_rows) != int(config["cohort"]["query_count"]):
            raise ValueError("v17 full pilot query population drifted")
        mode = "BOUNDED_ORACLE_GENERATION"

    _verify(args.queries.resolve(), str(config["cohort"]["queries_sha256"]))
    _verify(
        Path(authorities["scene_ir_dir"]) / "manifest.json",
        str(authorities["scene_ir_manifest_sha256"]),
    )
    _verify(
        _resolve(authorities["v16_parent_config_path"]),
        str(authorities["v16_parent_config_sha256"]),
    )
    _verify(
        _resolve(authorities["operational_policy_path"]),
        str(authorities["operational_policy_sha256"]),
    )
    parent_manifest = _verify_input_manifest(
        Path(authorities["v16_parent_generation_dir"]) / "manifest.json",
        str(authorities["v16_parent_generation_manifest_sha256"]),
    )
    geo_manifest = _verify_input_manifest(
        Path(authorities["v16_geo_generation_dir"]) / "manifest.json",
        str(authorities["v16_geo_generation_manifest_sha256"]),
    )
    _verify(
        Path(authorities["v16_parent_generation_dir"]) / "parent_signals.parquet",
        str(authorities["v16_parent_signals_sha256"]),
    )
    _verify(
        Path(authorities["v16_geo_generation_dir"]) / "geo_neighbor_parents.parquet",
        str(authorities["v16_geo_parent_sha256"]),
    )
    _verify(
        Path(authorities["sidecar_dir"]) / "manifest.json",
        str(authorities["sidecar_manifest_sha256"]),
    )
    sidecar_path = Path(authorities["sidecar_dir"]) / "siren_local_sites.parquet"
    _verify(sidecar_path, str(authorities["siren_local_sites_sha256"]))
    if parent_manifest.get("query_count") != 1034 or geo_manifest.get("query_count") != 1034:
        raise ValueError("sealed v16 parent/geo population drifted")

    output = args.output_dir.resolve()
    stage = output.with_name("." + output.name + ".partial")
    if output.exists() or stage.exists():
        raise FileExistsError(output if output.exists() else stage)
    stage.mkdir(parents=True)
    started, swap_before = time.perf_counter(), _swapouts()
    query_ids = {query_id for query_id, _fold in query_rows}
    queries = _query_surface(config, query_rows)

    resources = config["resources"]
    connection = duckdb.connect()
    connection.execute(f"PRAGMA threads={int(resources['duckdb_threads'])}")
    _write_address_signals(
        connection,
        queries=queries,
        sidecar_path=sidecar_path,
        output_path=stage / "address_parent_signals.parquet",
        minimum_similarity=float(config["recipe"]["address_street_similarity_minimum"]),
        cap=int(config["recipe"]["address_channel_cap"]),
    )

    parent_config = json.loads(_resolve(authorities["v16_parent_config_path"]).read_text())
    parent_signals = [
        row
        for row in pq.read_table(
            Path(authorities["v16_parent_generation_dir"]) / "parent_signals.parquet"
        ).to_pylist()
        if str(row["query_id"]) in query_ids
    ]
    text_orders = _text_orders(parent_signals, parent_config)
    geo_rows = [
        row
        for row in pq.read_table(
            Path(authorities["v16_geo_generation_dir"]) / "geo_neighbor_parents.parquet"
        ).to_pylist()
        if str(row["query_id"]) in query_ids
    ]
    address_rows = pq.read_table(stage / "address_parent_signals.parquet").to_pylist()
    geo_orders: dict[str, list[str]] = defaultdict(list)
    address_orders: dict[str, list[str]] = defaultdict(list)
    for row in sorted(geo_rows, key=lambda item: (str(item["query_id"]), int(item["geo_parent_rank"]))):
        geo_orders[str(row["query_id"])].append(str(row["siren"]))
    for row in sorted(address_rows, key=lambda item: (str(item["query_id"]), int(item["address_parent_rank"]))):
        address_orders[str(row["query_id"])].append(str(row["siren"]))

    candidate_rows: list[dict[str, Any]] = []
    presence_rows: list[dict[str, Any]] = []
    folds = dict(query_rows)
    for query_id, _fold in query_rows:
        built, presence = build_parent_rows(
            query_id=query_id,
            fold=folds[query_id],
            source_orders={
                "TEXT": text_orders.get(query_id, ()),
                "ADDRESS": address_orders.get(query_id, ()),
                "GEO": geo_orders.get(query_id, ()),
            },
            variants=config["recipe"]["variants"],
            primary_variant=str(config["recipe"]["primary_variant"]),
            cap=int(config["recipe"]["parent_cap"]),
        )
        candidate_rows.extend(built)
        presence_rows.extend(presence)
    pq.write_table(
        pa.Table.from_pylist(presence_rows),
        stage / "parent_presence.parquet",
        compression="zstd",
        row_group_size=100_000,
    )
    parent_candidates = pa.Table.from_pylist(candidate_rows)
    pq.write_table(
        parent_candidates,
        stage / "parent_candidates.parquet",
        compression="zstd",
        row_group_size=100_000,
    )
    primary = parent_candidates.filter(pc.equal(parent_candidates["is_primary"], True))
    _write_expanded_sites(
        connection,
        primary_parents=primary.select(["query_id", "fold", "parent_rank", "siren"]),
        queries=queries,
        sidecar_path=sidecar_path,
        output_path=stage / "expanded_all_sites.parquet",
    )
    mismatch = connection.execute(
        """SELECT count(*) FROM read_parquet(?)
           WHERE emitted_site_count <> source_site_count OR source_site_count <= 0""",
        [str(stage / "expanded_all_sites.parquet")],
    ).fetchone()[0]
    if int(mismatch) != 0:
        raise AssertionError("v17 exhaustive site-count proof failed")
    elapsed, swap_after = time.perf_counter() - started, _swapouts()
    rss = _peak_rss_bytes()
    passed = bool(
        elapsed <= float(resources["maximum_wall_seconds"])
        and rss <= float(resources["maximum_peak_rss_gib"]) * 1024**3
        and swap_before >= 0
        and swap_after - swap_before <= int(resources["maximum_swapout_pages"])
    )
    if not passed:
        raise RuntimeError("STOP_RESOURCE_POLICY")
    manifest_path = write_generation_manifest(
        stage,
        {
            "config_sha256": sha256_file(config_path),
            "queries_sha256": sha256_file(args.queries.resolve()),
            "query_count": len(query_rows),
            "execution_mode": mode,
            "primary_variant": config["recipe"]["primary_variant"],
            "all_sites_per_selected_parent": True,
            "address_parent_channel_label_blind": True,
            "resources": {
                "wall_seconds": elapsed,
                "peak_rss_bytes": rss,
                "swapout_pages_before": swap_before,
                "swapout_pages_after": swap_after,
                "swapout_pages_delta": swap_after - swap_before,
                "pass": passed,
            },
        },
    )
    stage.rename(output)
    final_manifest = output / manifest_path.name
    print(json.dumps({"manifest": str(final_manifest), "sha256": sha256_file(final_manifest), "resources": json.loads(final_manifest.read_text())["resources"]}, sort_keys=True))


if __name__ == "__main__":
    main()
