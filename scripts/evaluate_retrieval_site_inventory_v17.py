#!/usr/bin/env python3
"""Truth-side oracle evaluation for the sealed v17 site inventory."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import resource
import sys
import time
from typing import Any, Mapping

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.xgb_matcher.retrieval_geo_v16 import normalize_street
from src.xgb_matcher.retrieval_site_inventory_v17 import (
    EVALUATION_SCHEMA_VERSION,
    sha256_file,
    split_street_number,
)
from src.xgb_matcher.retrieval_v12_scene_ir import SceneIRCache


MISS_KIND = "BASE_TOP1750_MISS"


def _resolve(path: str) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def _verify(path: Path, expected: str) -> None:
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"authority drifted: {path}")


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _query_surface(config: Mapping[str, Any], labels: list[dict[str, Any]]) -> pa.Table:
    expected = {str(row["query_id"]): int(row["fold"]) for row in labels}
    scene = SceneIRCache(
        Path(config["authorities"]["scene_ir_dir"]), verify_source_hash=False
    )
    columns = [
        "query_id", "fold", "crm_address", "crm_number", "crm_postcode",
        "crm_insee", "postcode", "insee", "union_rank",
    ]
    rows: list[dict[str, Any]] = []
    for item in scene.iter_selected_scenes(expected, columns=columns):
        first = item.rows.slice(0, 1).to_pylist()[0]
        raw_number = str(first.get("crm_number") or "")
        number, suffix = split_street_number(raw_number)
        rows.append(
            {
                "query_id": item.query_id,
                "fold": item.fold,
                "crm_number": number,
                "crm_suffix": suffix,
                "crm_street_norm": normalize_street(
                    str(first.get("crm_address") or ""), raw_number
                ),
                "crm_postcode": str(first.get("crm_postcode") or first.get("postcode") or ""),
                "crm_insee": str(first.get("crm_insee") or first.get("insee") or ""),
            }
        )
    if len(rows) != len(labels):
        raise ValueError("truth-side SceneIR population drifted")
    return pa.Table.from_pylist(rows)


def _report_parent_hits(
    ranks: Mapping[str, Mapping[str, int]], labels: list[dict[str, Any]]
) -> dict[str, Any]:
    miss_hits_by_fold = {"2": 0, "3": 0, "4": 0}
    control_hits_by_fold = {"2": 0, "3": 0, "4": 0}
    for label in labels:
        query_id = str(label["query_id"])
        fold = str(int(label["fold"]))
        siren = str(label["human_ground_truth_siret"])[:9]
        hit = int(ranks.get(query_id, {}).get(siren, 513) <= 512)
        target = miss_hits_by_fold if str(label["cohort_kind"]) == MISS_KIND else control_hits_by_fold
        target[fold] += hit
    return {
        "miss_hits": sum(miss_hits_by_fold.values()),
        "miss_hits_by_fold": miss_hits_by_fold,
        "control_hits": sum(control_hits_by_fold.values()),
        "control_hits_by_fold": control_hits_by_fold,
    }


def _build_operational_labels(
    connection: duckdb.DuckDBPyConnection,
    *,
    labels: list[dict[str, Any]],
    queries: pa.Table,
    sidecar_path: Path,
) -> list[dict[str, Any]]:
    label_table = pa.Table.from_pylist(labels)
    connection.register("v17_labels", label_table)
    connection.register("v17_truth_queries", queries)
    rows = connection.execute(
        """
        WITH truth AS (
          SELECT l.query_id, l.fold, l.human_ground_truth_siret,
                 substr(l.human_ground_truth_siret, 1, 9) truth_siren,
                 q.crm_number, q.crm_suffix, q.crm_street_norm,
                 q.crm_insee, q.crm_postcode
          FROM v17_labels l JOIN v17_truth_queries q USING(query_id, fold)
        )
        SELECT t.query_id, t.fold, t.human_ground_truth_siret,
               s.siret, s.administrative_state,
               (s.siret = t.human_ground_truth_siret) exact_truth,
               (t.crm_number <> '' AND s.street_number = t.crm_number
                AND (t.crm_suffix = '' OR upper(coalesce(s.street_number_suffix, '')) = t.crm_suffix)
                AND t.crm_street_norm <> '' AND s.street_normalized = t.crm_street_norm
                AND ((t.crm_insee <> '' AND s.insee = t.crm_insee)
                     OR (t.crm_postcode <> '' AND s.postcode = t.crm_postcode))) strict_same_site
        FROM truth t
        JOIN read_parquet(?) s ON s.siren = t.truth_siren
        ORDER BY t.query_id, s.siret
        """,
        [str(sidecar_path)],
    ).fetch_arrow_table().to_pylist()
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_query[str(row["query_id"])].append(row)
    output: list[dict[str, Any]] = []
    for label in labels:
        query_id = str(label["query_id"])
        truth = str(label["human_ground_truth_siret"])
        sites = by_query.get(query_id, [])
        exact_state = next(
            (str(row["administrative_state"] or "") for row in sites if row["exact_truth"]),
            "CATALOG_ABSENT",
        )
        same_site = sorted(
            {str(row["siret"]) for row in sites if bool(row["strict_same_site"])}
        )
        acceptable = sorted(set([truth, *same_site]))
        active_equivalents = sorted(
            {
                str(row["siret"])
                for row in sites
                if bool(row["strict_same_site"])
                and str(row["administrative_state"] or "") == "A"
                and str(row["siret"]) != truth
            }
        )
        output.append(
            {
                "query_id": query_id,
                "fold": int(label["fold"]),
                "ground_truth_siret_exact": truth,
                "ground_truth_state": exact_state,
                "acceptable_sirets_operational": acceptable,
                "operational_equivalent_count": max(0, len(acceptable) - 1),
                "active_successor_sirets": active_equivalents,
                "active_successor_same_site": bool(
                    exact_state != "A" and active_equivalents
                ),
                "multiple_active_equivalents": len(active_equivalents) > 1,
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate sealed v17 exhaustive-site oracle")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    authorities = config["authorities"]
    generation_root = args.generation_dir.resolve()
    generation_path = generation_root / "manifest.json"
    generation = json.loads(generation_path.read_text())
    if generation.get("execution_mode") != "BOUNDED_ORACLE_GENERATION":
        raise ValueError("preflight cannot be truth-side evaluated")
    if generation.get("truth_columns_opened") is not False or generation.get("positive_injection") is not False:
        raise ValueError("v17 generation is not label-blind")
    if generation.get("pre_score_site_cap") is not None:
        raise ValueError("v17 generation applied a forbidden site cap")
    if generation.get("config_sha256") != sha256_file(config_path):
        raise ValueError("v17 generation/config drifted")
    for name, record in generation["files"].items():
        _verify(generation_root / name, str(record["sha256"]))

    cohort_root = Path(authorities["cohort_dir"])
    _verify(cohort_root / "manifest.json", str(authorities["cohort_manifest_sha256"]))
    labels = pq.read_table(cohort_root / "cohort_labels.parquet").to_pylist()
    if len(labels) != int(config["cohort"]["query_count"]):
        raise ValueError("v17 label population drifted")
    if {int(row["fold"]) for row in labels} != {2, 3, 4}:
        raise ValueError("v17 evaluator opened a forbidden fold")

    expanded_path = generation_root / "expanded_all_sites.parquet"
    completeness_connection = duckdb.connect()
    mismatches = completeness_connection.execute(
        """SELECT count(*) FROM read_parquet(?)
           WHERE emitted_site_count <> source_site_count OR source_site_count <= 0""",
        [str(expanded_path)],
    ).fetchone()[0]
    if int(mismatches) != 0:
        raise ValueError("v17 expansion is not exhaustive")
    parent_candidates = pq.read_table(generation_root / "parent_candidates.parquet").to_pylist()
    parent_presence = pq.read_table(generation_root / "parent_presence.parquet").to_pylist()
    forbidden = {"human_ground_truth_siret", "cohort_kind", "label", "target"}
    for name in generation["files"]:
        if forbidden & set(pq.read_schema(generation_root / name).names):
            raise ValueError(f"truth leaked into generated file: {name}")

    ranks_by_variant: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
    for row in parent_candidates:
        ranks_by_variant[str(row["variant"])][str(row["query_id"])][str(row["siren"])] = int(row["parent_rank"])
    raw_ranks: dict[str, dict[str, int]] = defaultdict(dict)
    for row in parent_presence:
        raw_ranks[str(row["query_id"])][str(row["siren"])] = 1
    parent_reports = {"RAW_UNION": _report_parent_hits(raw_ranks, labels)}
    for variant, ranks in ranks_by_variant.items():
        parent_reports[variant] = _report_parent_hits(ranks, labels)

    connection = duckdb.connect()
    connection.execute(f"PRAGMA threads={int(config['resources']['duckdb_threads'])}")
    queries = _query_surface(config, labels)
    sidecar_path = Path(authorities["sidecar_dir"]) / "siren_local_sites.parquet"
    _verify(sidecar_path, str(authorities["siren_local_sites_sha256"]))
    operational = _build_operational_labels(
        connection, labels=labels, queries=queries, sidecar_path=sidecar_path
    )
    operational_table = pa.Table.from_pylist(operational)
    connection.register("v17_operational", operational_table)
    base_path = str((Path(authorities["v15_generation_dir"]) / "candidates.parquet").resolve()).replace("'", "''")
    expanded_sql_path = str(expanded_path.resolve()).replace("'", "''")
    connection.execute(
        f"CREATE OR REPLACE TEMP VIEW v17_base AS SELECT * FROM read_parquet('{base_path}') WHERE source='BASE_V11_TOP1750'"
    )
    connection.execute(
        f"CREATE OR REPLACE TEMP VIEW v17_expanded AS SELECT query_id, fold, siret FROM read_parquet('{expanded_sql_path}')"
    )
    hit_rows = connection.execute(
        """
        WITH candidates AS (
          SELECT query_id, fold, siret FROM v17_base
          UNION
          SELECT query_id, fold, siret FROM v17_expanded
        )
        SELECT o.query_id, o.fold, o.ground_truth_siret_exact,
               coalesce(bool_or(c.siret = o.ground_truth_siret_exact), false) exact_hit,
               coalesce(bool_or(list_contains(o.acceptable_sirets_operational, c.siret)), false) operational_hit
        FROM v17_operational o LEFT JOIN candidates c USING(query_id, fold)
        GROUP BY o.query_id, o.fold, o.ground_truth_siret_exact
        ORDER BY o.query_id
        """
    ).fetch_arrow_table().to_pylist()
    label_by_query = {str(row["query_id"]): row for row in labels}
    exact_rescues_by_fold = {"2": 0, "3": 0, "4": 0}
    operational_rescues_by_fold = {"2": 0, "3": 0, "4": 0}
    controls_exact = controls_operational = 0
    for row in hit_rows:
        label = label_by_query[str(row["query_id"])]
        fold = str(int(row["fold"]))
        if str(label["cohort_kind"]) == MISS_KIND:
            exact_rescues_by_fold[fold] += int(bool(row["exact_hit"]))
            operational_rescues_by_fold[fold] += int(bool(row["operational_hit"]))
        else:
            controls_exact += int(bool(row["exact_hit"]))
            controls_operational += int(bool(row["operational_hit"]))
    exact_rescues = sum(exact_rescues_by_fold.values())
    operational_rescues = sum(operational_rescues_by_fold.values())
    gates = config["gates"]
    denominator = int(gates["national_denominator"])
    baseline = int(gates["sealed_baseline_master_exact_hits"])
    projected_exact_hits = baseline + exact_rescues
    projected_operational_hits = baseline + operational_rescues
    projected_exact_recall = projected_exact_hits / denominator
    projected_operational_recall = projected_operational_hits / denominator
    primary_name = str(config["recipe"]["primary_variant"])
    primary = parent_reports[primary_name]
    raw = parent_reports["RAW_UNION"]
    fold_gate = {str(key): int(value) for key, value in gates["primary_parent_hits_at_512_by_fold"].items()}
    parent_pass = bool(
        int(raw["miss_hits"]) >= int(gates["raw_union_parent_hits_at_512_minimum"])
        and int(primary["miss_hits"]) >= int(gates["primary_parent_hits_at_512_minimum"])
        and all(int(primary["miss_hits_by_fold"][fold]) >= required for fold, required in fold_gate.items())
    )
    exact_gate = projected_exact_recall >= float(gates["minimum_projected_master_exact_recall"])
    confidence = projected_exact_recall >= float(gates["oracle_confidence_projected_master_exact_recall"])
    controls_pass = controls_exact == int(gates["preserved_controls"])
    decision = "PASS_STEPS_1_TO_3" if parent_pass and exact_gate and controls_pass else "PIVOT_PARENT_DISCOVERY"

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    operational_path = output / "operational_labels.parquet"
    hits_path = output / "oracle_query_outcomes.parquet"
    pq.write_table(operational_table, operational_path, compression="zstd")
    pq.write_table(pa.Table.from_pylist(hit_rows), hits_path, compression="zstd")
    elapsed = time.perf_counter() - started
    payload = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "decision": decision,
        "config_sha256": sha256_file(config_path),
        "generation_manifest_sha256": sha256_file(generation_path),
        "parent_reports": parent_reports,
        "site_oracle": {
            "exact_rescues": exact_rescues,
            "exact_rescues_by_fold": exact_rescues_by_fold,
            "operational_rescues": operational_rescues,
            "operational_rescues_by_fold": operational_rescues_by_fold,
            "controls_exact": controls_exact,
            "controls_operational": controls_operational,
            "projected_master_exact_hits": projected_exact_hits,
            "projected_master_exact_recall": projected_exact_recall,
            "projected_master_operational_hits": projected_operational_hits,
            "projected_master_operational_recall": projected_operational_recall,
            "queries_with_operational_equivalents": sum(int(row["operational_equivalent_count"]) > 0 for row in operational),
            "active_successor_same_site": sum(bool(row["active_successor_same_site"]) for row in operational),
            "multiple_active_equivalents": sum(bool(row["multiple_active_equivalents"]) for row in operational),
        },
        "checks": {
            "truth_in_generation": False,
            "positive_injection": False,
            "fold0_opened": False,
            "fold1_opened": False,
            "all_sites_expanded": True,
            "parent_gate": parent_pass,
            "exact_master_gate": exact_gate,
            "confidence_margin_99_7": confidence,
            "controls_preserved": controls_pass,
            "operational_substitutes_exact": False,
            "ban_distance_used_as_same_site_proof": False,
        },
        "resources": {"wall_seconds": elapsed, "peak_rss_bytes": _peak_rss_bytes()},
        "files": {
            operational_path.name: {
                "rows": len(operational), "size_bytes": operational_path.stat().st_size,
                "sha256": sha256_file(operational_path),
            },
            hits_path.name: {
                "rows": len(hit_rows), "size_bytes": hits_path.stat().st_size,
                "sha256": sha256_file(hits_path),
            },
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "sha256": sha256_file(manifest_path), "decision": decision, "site_oracle": payload["site_oracle"]}, sort_keys=True))


if __name__ == "__main__":
    main()
