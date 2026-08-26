#!/usr/bin/env python3
"""Generate the sealed, label-blind v18 raw parent union."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import json
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import uuid

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery, TantivyBackend
from src.xgb_matcher.retrieval_parent_discovery_v18 import (
    SIGNAL_SCHEMA,
    SiretChannelSpec,
    collapse_siret_hits_to_parent_rows,
    load_query_selection,
    sha256_file,
    validate_signal_rows,
    write_manifest,
)
from src.xgb_matcher.retrieval_v12_scene_ir import SceneIRCache


def _sql_path(path: Path | str) -> str:
    return str(path).replace("'", "''")


def _swapouts() -> int:
    try:
        output = subprocess.check_output(["/usr/bin/vm_stat"], text=True)
        for line in output.splitlines():
            if line.startswith("Swapouts:"):
                return int(line.split(":", 1)[1].strip().rstrip("."))
    except Exception:
        return -1
    return -1


def _require_sha(record: dict, *, label: str) -> Path:
    path = Path(record["path"])
    if not path.is_file():
        raise ValueError(f"missing v18 authority {label}: {path}")
    if sha256_file(path) != str(record["sha256"]):
        raise ValueError(f"v18 authority drifted: {label}")
    return path


def _validate_authorities(config: dict, queries_path: Path, *, full: bool) -> dict[str, Path]:
    authorities = config["generation_authorities"]
    if sha256_file(queries_path) != str(authorities["queries"]["sha256"]):
        raise ValueError("v18 query selection drifted")
    resolved = {
        key: _require_sha(record, label=key)
        for key, record in authorities.items()
        if key not in {"queries", "scene_ir", "hierarchical_index", "temporal_sources"}
    }
    for key in ("scene_ir", "hierarchical_index"):
        root = Path(authorities[key]["path"])
        manifest = root / "manifest.json"
        if not manifest.is_file() or sha256_file(manifest) != str(authorities[key]["manifest_sha256"]):
            raise ValueError(f"v18 {key} manifest drifted")
        resolved[key] = root
    temporal = authorities["temporal_sources"]
    temporal_index_manifest = Path(temporal["index_path"]) / "manifest.json"
    if sha256_file(temporal_index_manifest) != str(temporal["manifest_sha256"]):
        raise ValueError("v18 temporal index manifest drifted")
    temporal_manifest = json.loads(temporal_index_manifest.read_text())
    source_root = Path(temporal["source_root"])
    files: list[Path] = []
    for record in temporal_manifest["sources"]:
        path = source_root / str(record["projection_build_id"]) / "succession_edges.parquet"
        if not path.is_file():
            raise ValueError(f"missing temporal succession authority: {path}")
        if full and sha256_file(path) != str(record["successions_sha256"]):
            raise ValueError(f"temporal succession source drifted: {path}")
        files.append(path)
    if len(files) != int(temporal["expected_file_count"]):
        raise ValueError("temporal succession file count drifted")
    resolved["temporal_glob"] = source_root / "*" / "succession_edges.parquet"
    return resolved


def _write_structured_signals(
    *,
    config: dict,
    query_rows: tuple[tuple[str, int], ...],
    paths: dict[str, Path],
    output_path: Path,
) -> None:
    recipe = config["recipe"]
    connection = duckdb.connect(":memory:")
    connection.execute(f"SET threads={int(config['resources']['duckdb_threads'])}")
    connection.execute(f"SET memory_limit='{int(config['resources']['duckdb_memory_gib'])}GiB'")
    selection = pa.Table.from_pylist(
        [{"query_id": query_id, "fold": fold} for query_id, fold in query_rows],
        schema=pa.schema([pa.field("query_id", pa.string()), pa.field("fold", pa.int8())]),
    )
    connection.register("v18_queries", selection)
    raw = _sql_path(paths["raw_union"])
    v17_presence = _sql_path(paths["v17_parent_presence"])
    v17_candidates = _sql_path(paths["v17_parent_candidates"])
    roles = _sql_path(paths["rne_roles"])
    bodacc = _sql_path(paths["bodacc_relations"])
    temporal = _sql_path(paths["temporal_glob"])
    destination = _sql_path(output_path)
    rne_seed = int(recipe["rne_seed_parent_count"])
    rne_cap = int(recipe["rne_relation_cap"])
    bodacc_seed = int(recipe["bodacc_seed_parent_count"])
    bodacc_cap = int(recipe["bodacc_relation_cap"])
    succession_seed = int(recipe["succession_seed_siret_count"])
    succession_cap = int(recipe["succession_relation_cap"])
    connection.execute(
        f"""
        COPY (
          WITH
          v17 AS (
            SELECT p.query_id, cast(p.fold AS TINYINT) fold, 'v17_union' channel,
                   row_number() OVER(PARTITION BY p.query_id ORDER BY p.siren)::INTEGER channel_rank,
                   p.siren, '' source_siret, 0.0::DOUBLE score,
                   NULL::INTEGER seed_rank, '' relation_direction, 'V17_UNION' relation_type
            FROM read_parquet('{v17_presence}') p JOIN v18_queries q USING(query_id, fold)
            WHERE length(p.siren)=9
          ),
          raw_selected AS (
            SELECT r.* FROM read_parquet('{raw}') r JOIN v18_queries q USING(query_id, fold)
          ),
          legacy_best AS (
            SELECT query_id, cast(fold AS TINYINT) fold, siren,
                   arg_min(siret, union_rank) source_siret,
                   min(union_rank)::INTEGER best_rank
            FROM raw_selected
            WHERE length(siren)=9 AND length(siret)=14
            GROUP BY query_id, fold, siren
          ),
          legacy AS (
            SELECT query_id, fold, 'legacy_site_pool' channel,
                   row_number() OVER(PARTITION BY query_id ORDER BY best_rank, siren)::INTEGER channel_rank,
                   siren, source_siret, -best_rank::DOUBLE score, best_rank seed_rank,
                   '' relation_direction, 'OFFICIAL_SIRET_TO_SIREN' relation_type
            FROM legacy_best
          ),
          parent_seeds AS (
            SELECT c.query_id, cast(c.fold AS TINYINT) fold, c.siren,
                   c.parent_rank::INTEGER seed_rank
            FROM read_parquet('{v17_candidates}') c JOIN v18_queries q USING(query_id, fold)
            WHERE c.is_primary
          ),
          rne_raw AS (
            SELECT s.query_id, s.fold, r.related_siren siren, min(s.seed_rank)::INTEGER seed_rank,
                   max(r.active::INTEGER)::INTEGER active,
                   max(r.effective_date) effective_date
            FROM parent_seeds s JOIN read_parquet('{roles}') r
              ON r.subject_siren=s.siren
            WHERE s.seed_rank<={rne_seed} AND length(r.related_siren)=9
                  AND r.related_siren<>s.siren
            GROUP BY s.query_id, s.fold, r.related_siren
          ),
          rne_ranked AS (
            SELECT *, row_number() OVER(PARTITION BY query_id
                     ORDER BY seed_rank, active DESC, effective_date DESC NULLS LAST, siren)::INTEGER rank
            FROM rne_raw
          ),
          rne AS (
            SELECT query_id, fold, 'rne_subject_to_related' channel, rank channel_rank,
                   siren, '' source_siret, active::DOUBLE score, seed_rank,
                   'SUBJECT_TO_RELATED' relation_direction, 'RNE_LEGAL_ENTITY_ROLE' relation_type
            FROM rne_ranked WHERE rank<={rne_cap}
          ),
          bodacc_edges AS (
            SELECT s.query_id, s.fold,
                   CASE WHEN b.from_identifier=s.siren THEN b.to_identifier ELSE b.from_identifier END siren,
                   min(s.seed_rank)::INTEGER seed_rank,
                   max(b.effective_date) effective_date,
                   CASE WHEN bool_or(b.from_identifier=s.siren) THEN 'FROM_TO' ELSE 'TO_FROM' END direction
            FROM parent_seeds s JOIN read_parquet('{bodacc}') b
              ON (b.from_identifier=s.siren OR b.to_identifier=s.siren)
            WHERE s.seed_rank<={bodacc_seed} AND b.from_kind='SIREN' AND b.to_kind='SIREN'
                  AND length(CASE WHEN b.from_identifier=s.siren THEN b.to_identifier ELSE b.from_identifier END)=9
            GROUP BY s.query_id, s.fold,
                     CASE WHEN b.from_identifier=s.siren THEN b.to_identifier ELSE b.from_identifier END
          ),
          bodacc_ranked AS (
            SELECT *, row_number() OVER(PARTITION BY query_id
                     ORDER BY seed_rank, effective_date DESC NULLS LAST, siren)::INTEGER rank
            FROM bodacc_edges
          ),
          bodacc_out AS (
            SELECT query_id, fold, 'bodacc_asset_transfer' channel, rank channel_rank,
                   siren, '' source_siret, 0.0::DOUBLE score, seed_rank,
                   direction relation_direction, 'ASSET_TRANSFER' relation_type
            FROM bodacc_ranked WHERE rank<={bodacc_cap}
          ),
          succession_seeds AS (
            SELECT query_id, cast(fold AS TINYINT) fold, siret, union_rank::INTEGER seed_rank
            FROM raw_selected
            WHERE union_rank<={succession_seed} AND length(siret)=14
          ),
          succession_edges AS (
            SELECT seed.query_id, seed.fold, edge.successor_siren siren,
                   min(seed.seed_rank)::INTEGER seed_rank,
                   max(edge.effective_date) effective_date,
                   arg_min(edge.successor_siret, seed.seed_rank) linked_siret,
                   'PREDECESSOR_TO_SUCCESSOR' direction
            FROM succession_seeds seed JOIN read_parquet('{temporal}') edge
              ON seed.siret=edge.predecessor_siret
            WHERE edge.economic_continuity AND length(edge.successor_siren)=9
            GROUP BY seed.query_id, seed.fold, edge.successor_siren
            UNION ALL
            SELECT seed.query_id, seed.fold, edge.predecessor_siren siren,
                   min(seed.seed_rank)::INTEGER seed_rank,
                   max(edge.effective_date) effective_date,
                   arg_min(edge.predecessor_siret, seed.seed_rank) linked_siret,
                   'SUCCESSOR_TO_PREDECESSOR' direction
            FROM succession_seeds seed JOIN read_parquet('{temporal}') edge
              ON seed.siret=edge.successor_siret
            WHERE edge.economic_continuity AND length(edge.predecessor_siren)=9
            GROUP BY seed.query_id, seed.fold, edge.predecessor_siren
          ),
          succession_best AS (
            SELECT query_id, fold, siren, min(seed_rank)::INTEGER seed_rank,
                   max(effective_date) effective_date,
                   arg_min(linked_siret, seed_rank) linked_siret,
                   arg_min(direction, seed_rank) direction
            FROM succession_edges GROUP BY query_id, fold, siren
          ),
          succession_ranked AS (
            SELECT *, row_number() OVER(PARTITION BY query_id
                     ORDER BY seed_rank, effective_date DESC NULLS LAST, siren)::INTEGER rank
            FROM succession_best
          ),
          succession_out AS (
            SELECT query_id, fold, 'sirene_economic_continuity' channel, rank channel_rank,
                   siren, linked_siret source_siret, 1.0::DOUBLE score, seed_rank,
                   direction relation_direction, 'SIRENE_SUCCESSION' relation_type
            FROM succession_ranked WHERE rank<={succession_cap}
          )
          SELECT * FROM v17
          UNION ALL SELECT * FROM legacy
          UNION ALL SELECT * FROM rne
          UNION ALL SELECT * FROM bodacc_out
          UNION ALL SELECT * FROM succession_out
          ORDER BY query_id, channel, channel_rank
        ) TO '{destination}'
          (FORMAT PARQUET, COMPRESSION ZSTD, COMPRESSION_LEVEL 3,
           ROW_GROUP_SIZE 100000)
        """
    )
    connection.close()


def _load_retrieval_queries(
    query_rows: tuple[tuple[str, int], ...], scene_root: Path
) -> list[tuple[str, int, RetrievalQuery]]:
    query_fold = dict(query_rows)
    scene = SceneIRCache(scene_root, verify_source_hash=False)
    columns = [
        "query_id", "fold", "crm_name", "crm_address", "crm_number",
        "crm_postcode", "crm_insee", "postcode", "insee",
    ]
    output: list[tuple[str, int, RetrievalQuery]] = []
    for item in scene.iter_selected_scenes(list(query_fold), columns=columns):
        if query_fold.get(item.query_id) != item.fold:
            raise ValueError("v18 SceneIR fold drifted")
        first = item.rows.slice(0, 1).to_pylist()[0]
        output.append((item.query_id, item.fold, RetrievalQuery.from_mapping(first)))
    if {query_id for query_id, _fold, _query in output} != set(query_fold):
        raise ValueError("v18 SceneIR did not emit the sealed query population")
    ordinal = {query_id: index for index, (query_id, _fold) in enumerate(query_rows)}
    output.sort(key=lambda row: ordinal[row[0]])
    return output


def _write_direct_signals(
    *,
    config: dict,
    retrieval_queries: list[tuple[str, int, RetrievalQuery]],
    index_root: Path,
    output_path: Path,
) -> None:
    specs = tuple(SiretChannelSpec.from_mapping(row) for row in config["siret_channels"])
    if len(specs) != len({spec.name for spec in specs}):
        raise ValueError("duplicate v18 SIRET-native channel name")
    backend = TantivyBackend(index_root)
    writer = pq.ParquetWriter(
        output_path,
        SIGNAL_SCHEMA,
        compression="zstd",
        compression_level=3,
        use_dictionary=True,
        write_statistics=True,
        data_page_version="1.0",
    )
    chunk_size = int(config["resources"]["query_chunk_size"])
    workers = int(config["resources"]["search_workers"])
    channel_order = {spec.name: index for index, spec in enumerate(specs)}
    try:
        for start in range(0, len(retrieval_queries), chunk_size):
            chunk = retrieval_queries[start : start + chunk_size]
            rows: list[dict] = []
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(backend.search, query, spec.backend_channel, spec.limit):
                    (query_id, fold, spec)
                    for query_id, fold, query in chunk
                    for spec in specs
                }
                for future in as_completed(futures):
                    query_id, fold, spec = futures[future]
                    rows.extend(
                        collapse_siret_hits_to_parent_rows(
                            query_id=query_id,
                            fold=fold,
                            channel=spec.name,
                            hits=future.result(),
                        )
                    )
            chunk_order = {query_id: index for index, (query_id, _fold, _query) in enumerate(chunk)}
            rows.sort(
                key=lambda row: (
                    chunk_order[str(row["query_id"])],
                    channel_order[str(row["channel"])],
                    int(row["channel_rank"]),
                )
            )
            validate_signal_rows(rows)
            if rows:
                writer.write_table(pa.Table.from_pylist(rows, schema=SIGNAL_SCHEMA), row_group_size=100000)
    finally:
        writer.close()
        del backend
        gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate v18 SIRET-first parent discovery")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-query-count", type=int, default=0)
    args = parser.parse_args()
    config_path = args.config.resolve()
    queries_path = args.queries.resolve()
    output = args.output_dir.resolve()
    config = json.loads(config_path.read_text())
    query_rows = load_query_selection(queries_path, expected_count=int(config["cohort"]["query_count"]))
    scope = "OFFICIAL_1034_QUERY_DEVELOPMENT_WORKBENCH"
    if args.preflight_query_count:
        if not 1 <= args.preflight_query_count <= 16:
            raise ValueError("v18 preflight is limited to 1..16 queries")
        query_rows = query_rows[: args.preflight_query_count]
        scope = "NON_DECISIONAL_PREFLIGHT"
    if output.exists():
        raise FileExistsError(output)
    paths = _validate_authorities(config, queries_path, full=scope.startswith("OFFICIAL"))
    stage = output.parent / f".{output.name}.stage-{uuid.uuid4().hex}"
    if stage.exists():
        raise FileExistsError(stage)
    stage.mkdir(parents=True)
    started = time.perf_counter()
    swap_before = _swapouts()
    try:
        structured = stage / "structured_parent_signals.parquet"
        _write_structured_signals(
            config=config,
            query_rows=query_rows,
            paths=paths,
            output_path=structured,
        )
        gc.collect()
        retrieval_queries = _load_retrieval_queries(query_rows, paths["scene_ir"])
        direct = stage / "siret_native_parent_signals.parquet"
        _write_direct_signals(
            config=config,
            retrieval_queries=retrieval_queries,
            index_root=paths["hierarchical_index"],
            output_path=direct,
        )
        wall = time.perf_counter() - started
        swap_after = _swapouts()
        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        delta = swap_after - swap_before if min(swap_before, swap_after) >= 0 else None
        limits = config["resources"]
        resource_pass = bool(
            wall <= float(limits["maximum_wall_seconds"])
            and peak <= int(limits["maximum_peak_rss_gib"]) * 1024**3
            and delta == int(limits["maximum_swapout_pages"])
        )
        manifest = write_manifest(
            stage,
            {
                "config_sha256": sha256_file(config_path),
                "queries_sha256": sha256_file(queries_path),
                "query_count": len(query_rows),
                "scope": scope,
                "raw_union_only": True,
                "parent_selection_performed": False,
                "site_expansion_performed": False,
                "resources": {
                    "wall_seconds": wall,
                    "peak_rss_bytes": peak,
                    "swapout_pages_before": swap_before,
                    "swapout_pages_after": swap_after,
                    "swapout_pages_delta": delta,
                    "pass": resource_pass,
                },
            },
            ["structured_parent_signals.parquet", "siret_native_parent_signals.parquet"],
        )
        if scope.startswith("OFFICIAL") and not resource_pass:
            raise RuntimeError("STOP_RESOURCE_POLICY")
        stage.replace(output)
        manifest = output / manifest.name
        print(json.dumps({"manifest": str(manifest), "sha256": sha256_file(manifest)}, sort_keys=True))
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
