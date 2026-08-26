#!/usr/bin/env python3
"""Build the label-blind BAN/SIRENE geographic feature surface for v16."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.xgb_matcher.retrieval_geo_v16 import (
    BanCandidate,
    QUERY_GEO_INTERNAL_SCHEMA,
    QUERY_GEO_SCHEMA,
    choose_ban_match,
    normalize_street,
    sha256_file,
    validate_label_blind_queries,
    write_manifest,
)
from src.xgb_matcher.retrieval_v12_scene_ir import SceneIRCache


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _verify(path: Path, expected: str) -> None:
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"authority drifted: {path}")


def _swapouts() -> int:
    try:
        output = subprocess.check_output(["/usr/bin/vm_stat"], text=True)
        for line in output.splitlines():
            if line.startswith("Swapouts:"):
                return int(line.split(":", 1)[1].strip().rstrip("."))
    except Exception:
        return -1
    return -1


def _query_surface(config: dict, query_rows: tuple[tuple[str, int], ...]) -> pa.Table:
    scene = SceneIRCache(Path(config["authorities"]["scene_ir_dir"]), verify_source_hash=False)
    expected = dict(query_rows)
    rows = []
    columns = [
        "query_id", "fold", "crm_address", "crm_number", "crm_postcode",
        "crm_insee", "postcode", "insee", "union_rank",
    ]
    for item in scene.iter_selected_scenes(expected, columns=columns):
        if item.fold != expected[item.query_id]:
            raise ValueError("query fold differs from sealed SceneIR")
        first = item.rows.slice(0, 1).to_pylist()[0]
        number = str(first.get("crm_number") or "")
        address = str(first.get("crm_address") or "")
        rows.append(
            {
                "query_id": item.query_id,
                "fold": item.fold,
                "crm_number": number,
                "crm_street_norm": normalize_street(address, number),
                "crm_postcode": str(first.get("crm_postcode") or first.get("postcode") or ""),
                "crm_insee": str(first.get("crm_insee") or first.get("insee") or ""),
            }
        )
    if len(rows) != len(query_rows):
        raise ValueError("SceneIR did not yield the sealed query population")
    return pa.Table.from_pylist(rows)


def _resolve_ban(con: duckdb.DuckDBPyConnection, queries: pa.Table, ban_path: Path) -> pa.Table:
    con.register("geo_v16_queries", queries)
    matches = con.execute(
        """
        SELECT q.query_id, b.longitude, b.latitude, b.nom_voie, b.code_postal,
               b.code_commune, b.type_position, b.source_position,
               b.certification_commune,
               CASE WHEN b.code_commune = q.crm_insee THEN 'INSEE' ELSE 'POSTCODE' END geography_match
        FROM geo_v16_queries q
        JOIN read_parquet(?) b
          ON b.numero = q.crm_number
         AND q.crm_number <> ''
         AND (
              (q.crm_insee <> '' AND b.code_commune = q.crm_insee)
              OR (q.crm_postcode <> '' AND b.code_postal = q.crm_postcode)
         )
        WHERE b.longitude IS NOT NULL AND b.latitude IS NOT NULL
        """,
        [str(ban_path)],
    ).fetch_arrow_table()
    by_query: dict[str, list[BanCandidate]] = defaultdict(list)
    for row in matches.to_pylist():
        by_query[str(row["query_id"])].append(
            BanCandidate(
                longitude=float(row["longitude"]), latitude=float(row["latitude"]),
                street=str(row["nom_voie"] or ""), postcode=str(row["code_postal"] or ""),
                insee=str(row["code_commune"] or ""),
                type_position=str(row["type_position"] or ""),
                source_position=str(row["source_position"] or ""),
                certified=bool(row["certification_commune"]),
                geography_match=str(row["geography_match"]),
            )
        )
    output = []
    for query in queries.to_pylist():
        match = choose_ban_match(
            crm_street=str(query["crm_street_norm"]),
            candidates=by_query.get(str(query["query_id"]), ()),
        )
        if match is None:
            match = {
                "ban_resolved": False, "ban_x_l93": None, "ban_y_l93": None,
                "ban_street_similarity": None, "ban_geography_match": "UNRESOLVED",
                "ban_type_position": "", "ban_source_position": "",
                "ban_certification_commune": False, "ban_uncertainty_m": None,
                "ban_declared_weak": True, "ban_match_count": 0,
            }
        output.append({"query_id": query["query_id"], "fold": query["fold"], **match})
    return pa.Table.from_pylist(output, schema=QUERY_GEO_INTERNAL_SCHEMA)


def _write_candidate_features(
    con: duckdb.DuckDBPyConnection,
    *, candidates_path: Path,
    snapshot_path: Path,
    query_surface: pa.Table,
    query_geo: pa.Table,
    output_path: Path,
) -> None:
    candidates = pq.read_table(candidates_path)
    con.register("geo_v16_candidates", candidates)
    con.register("geo_v16_query_surface", query_surface)
    con.register("geo_v16_query_geo", query_geo)
    destination = _sql_path(output_path)
    snapshot = _sql_path(snapshot_path)
    con.execute(
        f"""
        COPY (
          WITH joined AS (
            SELECT c.query_id, c.fold, c.candidate_rank, c.siret, c.source,
                   c.base_rank, c.parent_rank, c.site_rank,
                   qg.ban_resolved, qg.ban_street_similarity,
                   qg.ban_geography_match, qg.ban_type_position,
                   qg.ban_source_position, qg.ban_certification_commune,
                   qg.ban_uncertainty_m, qg.ban_declared_weak,
                   qs.crm_insee, qs.crm_postcode, qs.crm_number, qs.crm_street_norm,
                   s.codeCommuneEtablissement candidate_insee,
                   s.codePostalEtablissement candidate_postcode,
                   s.numeroVoieEtablissement candidate_number,
                   trim(regexp_replace(upper(strip_accents(
                     concat_ws(' ', s.typeVoieEtablissement, s.libelleVoieEtablissement)
                   )), '[^A-Z0-9]+', ' ', 'g')) candidate_street_norm,
                   try_cast(s.coordonneeLambertAbscisseEtablissement AS DOUBLE) candidate_x_l93,
                   try_cast(s.coordonneeLambertOrdonneeEtablissement AS DOUBLE) candidate_y_l93,
                   qg.ban_x_l93 query_x_l93, qg.ban_y_l93 query_y_l93
            FROM geo_v16_candidates c
            JOIN geo_v16_query_surface qs USING(query_id, fold)
            JOIN geo_v16_query_geo qg USING(query_id, fold)
            LEFT JOIN read_parquet('{snapshot}') s USING(siret)
          ), valid AS (
            SELECT *,
              candidate_x_l93 BETWEEN 0 AND 1300000
                AND candidate_y_l93 BETWEEN 6000000 AND 7300000 candidate_coord_present
            FROM joined
          ), distances AS (
            SELECT *, CASE WHEN ban_resolved AND candidate_coord_present
              THEN sqrt(pow(candidate_x_l93-query_x_l93, 2)+pow(candidate_y_l93-query_y_l93, 2))
              ELSE NULL END geo_distance_m
            FROM valid
          )
          SELECT query_id, fold, candidate_rank, siret, source,
                 base_rank, parent_rank, site_rank,
                 (source = 'BASE_V11_TOP1750')::TINYINT source_is_base,
                 ban_resolved, ban_street_similarity,
                 ban_geography_match, ban_type_position, ban_source_position,
                 ban_certification_commune, ban_uncertainty_m, ban_declared_weak,
                 candidate_coord_present,
                 (candidate_insee = crm_insee AND crm_insee <> '')::TINYINT candidate_same_insee,
                 (candidate_postcode = crm_postcode AND crm_postcode <> '')::TINYINT candidate_same_postcode,
                 (candidate_number = crm_number AND crm_number <> '')::TINYINT candidate_number_exact,
                 CASE WHEN crm_street_norm <> '' AND candidate_street_norm <> ''
                   THEN jaro_winkler_similarity(crm_street_norm, candidate_street_norm)
                   ELSE NULL END candidate_street_similarity,
                 geo_distance_m,
                 CASE WHEN geo_distance_m IS NOT NULL THEN ln(1+geo_distance_m) ELSE NULL END geo_distance_log1p,
                 CASE WHEN geo_distance_m IS NOT NULL
                   THEN greatest(0, geo_distance_m-ban_uncertainty_m) ELSE NULL END geo_distance_lower_m,
                 CASE WHEN geo_distance_m IS NOT NULL
                   THEN geo_distance_m+ban_uncertainty_m ELSE NULL END geo_distance_upper_m,
                 CASE WHEN geo_distance_m IS NOT NULL AND ban_uncertainty_m > 0
                   THEN geo_distance_m/ban_uncertainty_m ELSE NULL END geo_distance_uncertainty_ratio,
                 (geo_distance_m <= 25)::TINYINT geo_within_25m,
                 (geo_distance_m <= 100)::TINYINT geo_within_100m,
                 (geo_distance_m <= 500)::TINYINT geo_within_500m,
                 (geo_distance_m <= 2000)::TINYINT geo_within_2km
          FROM distances
          ORDER BY query_id, candidate_rank
        ) TO '{destination}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )


def _write_geo_neighbors(
    con: duckdb.DuckDBPyConnection,
    *, snapshot_path: Path,
    query_surface: pa.Table,
    query_geo: pa.Table,
    output_path: Path,
    parent_output_path: Path,
    neighbor_cap: int,
) -> None:
    """Nearest SIRENE sites in the CRM commune; this never consults truth."""

    con.register("geo_v16_query_surface", query_surface)
    con.register("geo_v16_query_geo", query_geo)
    destination = _sql_path(output_path)
    parent_destination = _sql_path(parent_output_path)
    snapshot = _sql_path(snapshot_path)
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE geo_v16_nearest_sites AS
          WITH sites AS (
            SELECT q.query_id, q.fold, s.siret, substr(s.siret, 1, 9) siren,
                   s.etatAdministratifEtablissement candidate_state,
                   (s.codePostalEtablissement = q.crm_postcode AND q.crm_postcode <> '')::TINYINT candidate_same_postcode,
                   (s.numeroVoieEtablissement = q.crm_number AND q.crm_number <> '')::TINYINT candidate_number_exact,
                   CASE WHEN q.crm_street_norm <> '' THEN jaro_winkler_similarity(
                     q.crm_street_norm,
                     trim(regexp_replace(upper(strip_accents(concat_ws(' ',
                       s.typeVoieEtablissement, s.libelleVoieEtablissement
                     ))), '[^A-Z0-9]+', ' ', 'g'))
                   ) ELSE NULL END candidate_street_similarity,
                   g.ban_street_similarity, g.ban_certification_commune,
                   g.ban_uncertainty_m, g.ban_declared_weak,
                   sqrt(
                     pow(try_cast(s.coordonneeLambertAbscisseEtablissement AS DOUBLE)-g.ban_x_l93, 2)
                     + pow(try_cast(s.coordonneeLambertOrdonneeEtablissement AS DOUBLE)-g.ban_y_l93, 2)
                   ) geo_distance_m
            FROM geo_v16_query_surface q
            JOIN geo_v16_query_geo g USING(query_id, fold)
            JOIN read_parquet('{snapshot}') s
              ON s.codeCommuneEtablissement = q.crm_insee
            WHERE g.ban_resolved
              AND try_cast(s.coordonneeLambertAbscisseEtablissement AS DOUBLE) BETWEEN 0 AND 1300000
              AND try_cast(s.coordonneeLambertOrdonneeEtablissement AS DOUBLE) BETWEEN 6000000 AND 7300000
          ), ranked AS (
            SELECT *, row_number() OVER (
              PARTITION BY query_id ORDER BY geo_distance_m,
              candidate_number_exact DESC, candidate_street_similarity DESC, siret
            ) geo_neighbor_rank
            FROM sites
          )
          SELECT * FROM ranked WHERE geo_neighbor_rank <= 4096
        """
    )
    con.execute(
        f"""
        COPY (
          SELECT query_id, fold, cast(geo_neighbor_rank AS SMALLINT) geo_neighbor_rank,
                 siret, siren, candidate_state, candidate_same_postcode,
                 candidate_number_exact, candidate_street_similarity,
                 ban_street_similarity, ban_certification_commune,
                 ban_uncertainty_m, ban_declared_weak,
                 geo_distance_m, ln(1+geo_distance_m) geo_distance_log1p,
                 greatest(0, geo_distance_m-ban_uncertainty_m) geo_distance_lower_m,
                 geo_distance_m+ban_uncertainty_m geo_distance_upper_m,
                 geo_distance_m/ban_uncertainty_m geo_distance_uncertainty_ratio
          FROM geo_v16_nearest_sites WHERE geo_neighbor_rank <= {int(neighbor_cap)}
          ORDER BY query_id, geo_neighbor_rank
        ) TO '{destination}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )
    con.execute(
        f"""
        COPY (
          WITH per_parent AS (
            SELECT query_id, fold, siren, min(geo_distance_m) geo_parent_distance_m,
                   arg_min(siret, geo_distance_m) nearest_siret,
                   max(candidate_same_postcode) candidate_same_postcode,
                   max(candidate_number_exact) candidate_number_exact,
                   max(candidate_street_similarity) candidate_street_similarity,
                   any_value(ban_street_similarity) ban_street_similarity,
                   any_value(ban_certification_commune) ban_certification_commune,
                   any_value(ban_uncertainty_m) ban_uncertainty_m,
                   any_value(ban_declared_weak) ban_declared_weak
            FROM geo_v16_nearest_sites GROUP BY query_id, fold, siren
          ), ranked AS (
            SELECT *, row_number() OVER (
              PARTITION BY query_id ORDER BY geo_parent_distance_m,
              candidate_number_exact DESC, candidate_street_similarity DESC, siren
            ) geo_parent_rank
            FROM per_parent
          )
          SELECT query_id, fold, cast(geo_parent_rank AS SMALLINT) geo_parent_rank,
                 siren, nearest_siret, candidate_same_postcode,
                 candidate_number_exact, candidate_street_similarity,
                 ban_street_similarity, ban_certification_commune,
                 ban_uncertainty_m, ban_declared_weak,
                 geo_parent_distance_m, ln(1+geo_parent_distance_m) geo_parent_distance_log1p,
                 greatest(0, geo_parent_distance_m-ban_uncertainty_m) geo_parent_distance_lower_m,
                 geo_parent_distance_m+ban_uncertainty_m geo_parent_distance_upper_m,
                 geo_parent_distance_m/ban_uncertainty_m geo_parent_distance_uncertainty_ratio
          FROM ranked WHERE geo_parent_rank <= 512
          ORDER BY query_id, geo_parent_rank
        ) TO '{parent_destination}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build label-blind geo v16 workbench features")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-query-count", type=int)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    authorities = config["authorities"]
    query_rows = validate_label_blind_queries(args.queries)
    if args.preflight_query_count is not None:
        if not 1 <= args.preflight_query_count <= 16:
            raise ValueError("preflight query count must be within 1..16")
        query_rows = query_rows[: args.preflight_query_count]
        execution_mode = "NON_DECISIONAL_PREFLIGHT"
    else:
        execution_mode = "BOUNDED_DIAGNOSTIC"
    if execution_mode != "NON_DECISIONAL_PREFLIGHT" and len(query_rows) != int(config["cohort"]["query_count"]):
        raise ValueError("geo v16 pilot population drifted")
    _verify(args.queries, authorities["queries_sha256"])
    _verify(Path(authorities["v15_candidates_path"]), authorities["v15_candidates_sha256"])
    _verify(Path(authorities["scene_ir_dir"]) / "manifest.json", authorities["scene_ir_manifest_sha256"])
    _verify(Path(authorities["ban_catalog_path"]), authorities["ban_catalog_sha256"])
    _verify(Path(authorities["ban_silver_path"]), authorities["ban_silver_sha256"])
    _verify(Path(authorities["sirene_snapshot_path"]), authorities["sirene_snapshot_sha256"])
    ban_catalog = json.loads(Path(authorities["ban_catalog_path"]).read_text())
    if ban_catalog.get("address_absence_claim_allowed") is not False:
        raise ValueError("BAN absence guardrail drifted")
    if ban_catalog.get("score_eligible") is not False:
        raise ValueError("BAN score eligibility guardrail drifted")
    output = args.output_dir.resolve()
    stage = output.with_name("." + output.name + ".partial")
    if output.exists() or stage.exists():
        raise FileExistsError(output if output.exists() else stage)
    stage.mkdir(parents=True)
    started, swap_before = time.perf_counter(), _swapouts()
    con = duckdb.connect()
    con.execute(f"SET threads={int(config['resources']['duckdb_threads'])}")
    con.execute("SET temp_directory='/tmp/sireto_geo_v16_duckdb'")
    query_surface = _query_surface(config, query_rows)
    query_geo = _resolve_ban(con, query_surface, Path(authorities["ban_silver_path"]))
    pq.write_table(
        query_geo.select(QUERY_GEO_SCHEMA.names),
        stage / "query_geo.parquet",
        compression="zstd",
    )
    _write_candidate_features(
        con,
        candidates_path=Path(authorities["v15_candidates_path"]),
        snapshot_path=Path(authorities["sirene_snapshot_path"]),
        query_surface=query_surface,
        query_geo=query_geo,
        output_path=stage / "candidate_geo_features.parquet",
    )
    _write_geo_neighbors(
        con,
        snapshot_path=Path(authorities["sirene_snapshot_path"]),
        query_surface=query_surface,
        query_geo=query_geo,
        output_path=stage / "geo_neighbor_candidates.parquet",
        parent_output_path=stage / "geo_neighbor_parents.parquet",
        neighbor_cap=int(config["recipe"]["geo_neighbor_cap"]),
    )
    elapsed = time.perf_counter() - started
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    swap_after = _swapouts()
    swap_delta = swap_after - swap_before if min(swap_before, swap_after) >= 0 else None
    limits = config["resources"]
    resource_pass = bool(
        elapsed <= float(limits["maximum_wall_seconds"])
        and peak <= int(limits["maximum_peak_rss_gib"]) * 1024**3
        and swap_delta == int(limits["maximum_swapout_pages"])
    )
    resolved = sum(bool(row["ban_resolved"]) for row in query_geo.to_pylist())
    manifest = write_manifest(
        stage,
        {
            "config_sha256": sha256_file(config_path),
            "queries_sha256": sha256_file(args.queries),
            "query_count": len(query_rows),
            "ban_resolved_queries": resolved,
            "ban_absence_used_as_negative": False,
            "ban_score_eligible": False,
            "folds_opened": sorted({fold for _, fold in query_rows}),
            "execution_mode": execution_mode,
            "decision_eligible": False,
            "resources": {
                "wall_seconds": elapsed, "peak_rss_bytes": peak,
                "swapout_pages_before": swap_before, "swapout_pages_after": swap_after,
                "swapout_pages_delta": swap_delta, "pass": resource_pass,
            },
        },
    )
    if not resource_pass:
        raise RuntimeError("STOP_RESOURCE_POLICY")
    stage.rename(output)
    print(json.dumps({"manifest": str(output / manifest.name), "sha256": sha256_file(output / manifest.name)}, sort_keys=True))


if __name__ == "__main__":
    main()
