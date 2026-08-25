"""Truth-side cohort builder and evaluator for Master v15 feasibility."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from .retrieval_master_v15 import sha256_file


COHORT_SCHEMA_VERSION = "sireto-master-v15-feasibility-cohort-1"
EVALUATION_SCHEMA_VERSION = "sireto-master-v15-feasibility-evaluation-1"
BASE_HITS = {2: 7585, 3: 7792, 4: 7672}
DENOMINATORS = {2: 7767, 3: 7967, 4: 7837}
MIN_RESCUES = {2: 155, 3: 148, 4: 138}
CONTROL_COUNTS = {2: 171, 3: 171, 4: 170}


def _sql_path(path: Path) -> str:
    return str(Path(path).resolve()).replace("'", "''")


def _sha_order(query_id: str) -> tuple[str, str]:
    return hashlib.sha256(query_id.encode("utf-8")).hexdigest(), query_id


def build_cohort(
    *,
    raw_union_path: Path,
    label_ledger_path: Path,
    output_dir: Path,
    expected_raw_sha256: str,
    expected_label_sha256: str,
) -> Path:
    if sha256_file(raw_union_path) != expected_raw_sha256:
        raise ValueError("v11 raw union authority drifted")
    if sha256_file(label_ledger_path) != expected_label_sha256:
        raise ValueError("label ledger authority drifted")
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    connection = duckdb.connect()
    try:
        rows = connection.execute(
            f"""
            SELECT r.query_id, CAST(r.fold AS TINYINT) AS fold,
                   l.human_ground_truth_siret,
                   MAX(CASE WHEN r.union_rank <= 1750
                                 AND r.siret=l.human_ground_truth_siret THEN 1 ELSE 0 END) AS base_hit
            FROM read_parquet('{_sql_path(raw_union_path)}') r
            JOIN read_parquet('{_sql_path(label_ledger_path)}') l USING (query_id)
            WHERE r.fold IN (2,3,4)
            GROUP BY r.query_id, r.fold, l.human_ground_truth_siret
            ORDER BY r.fold, MIN(r.union_rank)
            """
        ).fetchall()
    finally:
        connection.close()
    if len(rows) != 23571:
        raise ValueError(f"cohort authority expected 23571 queries, got {len(rows)}")
    misses = [(str(q), int(f), str(t), "BASE_TOP1750_MISS") for q, f, t, hit in rows if not int(hit)]
    controls: list[tuple[str, int, str, str]] = []
    for fold, count in CONTROL_COUNTS.items():
        eligible = sorted(
            ((str(q), int(f), str(t), "BASE_TOP1750_CONTROL") for q, f, t, hit in rows if int(f) == fold and int(hit)),
            key=lambda row: _sha_order(row[0]),
        )
        controls.extend(eligible[:count])
    population = misses + controls
    population.sort(key=lambda row: (row[1], _sha_order(row[0])))
    miss_counts = {fold: sum(1 for row in misses if row[1] == fold) for fold in DENOMINATORS}
    if miss_counts != {2: 182, 3: 175, 4: 165} or len(population) != 1034:
        raise ValueError(f"cohort counts drifted: misses={miss_counts}, total={len(population)}")
    queries_path = output_dir / "queries.parquet"
    labels_path = output_dir / "cohort_labels.parquet"
    pq.write_table(
        pa.table(
            {"query_id": [row[0] for row in population], "fold": pa.array([row[1] for row in population], type=pa.int8())}
        ),
        queries_path,
        compression="zstd",
    )
    pq.write_table(
        pa.table(
            {
                "query_id": [row[0] for row in population],
                "fold": pa.array([row[1] for row in population], type=pa.int8()),
                "human_ground_truth_siret": [row[2] for row in population],
                "cohort_kind": [row[3] for row in population],
            }
        ),
        labels_path,
        compression="zstd",
    )
    manifest = {
        "schema_version": COHORT_SCHEMA_VERSION,
        "query_count": len(population),
        "miss_counts": miss_counts,
        "control_counts": CONTROL_COUNTS,
        "selection": "ALL_TOP1750_MISSES_PLUS_LOWEST_SHA256_QUERY_ID_CONTROLS_PER_FOLD",
        "label_blind_generator_surface": ["query_id", "fold"],
        "authorities": {"raw_union_sha256": expected_raw_sha256, "label_ledger_sha256": expected_label_sha256},
        "files": {
            "queries.parquet": {"sha256": sha256_file(queries_path), "size_bytes": queries_path.stat().st_size, "rows": len(population)},
            "cohort_labels.parquet": {"sha256": sha256_file(labels_path), "size_bytes": labels_path.stat().st_size, "rows": len(population)},
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def _variant_hits(
    candidates: Sequence[Mapping[str, Any]],
    *,
    parent_cap: int,
    rescue_budget: int,
    site_depth: int,
) -> set[tuple[str, str]]:
    by_query: dict[str, list[Mapping[str, Any]]] = {}
    for row in candidates:
        by_query.setdefault(str(row["query_id"]), []).append(row)
    selected: set[tuple[str, str]] = set()
    for query_id, rows in by_query.items():
        base = sorted((row for row in rows if row["source"] == "BASE_V11_TOP1750"), key=lambda row: int(row["candidate_rank"]))
        rescue = sorted(
            (
                row for row in rows
                if row["source"] == "OFFICIAL_PARENT_RRF_SITE_RESCUE"
                and int(row["parent_rank"] or 0) <= parent_cap
                and int(row["site_rank"] or 0) <= site_depth
            ),
            key=lambda row: (int(row["site_rank"]), int(row["parent_rank"]), str(row["siret"])),
        )[:rescue_budget]
        selected.update((query_id, str(row["siret"])) for row in [*base, *rescue])
    return selected


def evaluate_generation(
    *,
    generation_dir: Path,
    cohort_dir: Path,
    output_dir: Path,
    config_sha256: str,
) -> Path:
    generation_dir, cohort_dir, output_dir = map(lambda p: Path(p).resolve(), (generation_dir, cohort_dir, output_dir))
    if output_dir.exists():
        raise FileExistsError(output_dir)
    generation = json.loads((generation_dir / "manifest.json").read_text())
    cohort = json.loads((cohort_dir / "manifest.json").read_text())
    for root, manifest in ((generation_dir, generation), (cohort_dir, cohort)):
        for name, record in manifest["files"].items():
            if sha256_file(root / name) != record["sha256"]:
                raise ValueError(f"artifact hash drifted: {name}")
    candidates = pq.read_table(generation_dir / "candidates.parquet").to_pylist()
    parents = pq.read_table(generation_dir / "parents.parquet").to_pylist()
    labels = pq.read_table(cohort_dir / "cohort_labels.parquet").to_pylist()
    if any("human_ground_truth_siret" in row for row in candidates + parents):
        raise ValueError("generation artifact contains truth-side data")
    label_by_query = {str(row["query_id"]): row for row in labels}
    if {str(row["query_id"]) for row in candidates} != set(label_by_query):
        raise ValueError("generation population differs from cohort")
    counts: dict[str, int] = {}
    for row in candidates:
        counts[str(row["query_id"])] = counts.get(str(row["query_id"]), 0) + 1
    if max(counts.values()) > 2000:
        raise ValueError("candidate cap exceeded")
    variants = {
        "PRIMARY": (512, 250, 16),
        "SENSITIVITY_PARENT384_RESCUE225_DEPTH16": (384, 225, 16),
        "SENSITIVITY_PARENT512_RESCUE200_DEPTH8": (512, 200, 8),
    }
    reports: dict[str, Any] = {}
    for name, (parent_cap, rescue_budget, site_depth) in variants.items():
        hits = _variant_hits(candidates, parent_cap=parent_cap, rescue_budget=rescue_budget, site_depth=site_depth)
        rescues = {fold: 0 for fold in DENOMINATORS}
        controls = {fold: 0 for fold in DENOMINATORS}
        for query_id, label in label_by_query.items():
            hit = (query_id, str(label["human_ground_truth_siret"])) in hits
            fold = int(label["fold"])
            if label["cohort_kind"] == "BASE_TOP1750_MISS":
                rescues[fold] += int(hit)
            else:
                controls[fold] += int(hit)
        final_hits = {fold: BASE_HITS[fold] + rescues[fold] for fold in DENOMINATORS}
        rates = {str(fold): final_hits[fold] / DENOMINATORS[fold] for fold in DENOMINATORS}
        reports[name] = {
            "parent_cap": parent_cap, "rescue_budget": rescue_budget, "site_depth": site_depth,
            "rescued_misses": rescues, "preserved_controls": controls,
            "projected_hits": final_hits, "projected_recall": rates,
            "global_projected_hits": sum(final_hits.values()),
            "global_projected_recall": sum(final_hits.values()) / sum(DENOMINATORS.values()),
        }
    primary = reports["PRIMARY"]
    controls_pass = primary["preserved_controls"] == CONTROL_COUNTS
    fold_pass = all(int(primary["rescued_misses"][fold]) >= MIN_RESCUES[fold] for fold in DENOMINATORS)
    sensitivity_pass = any(
        report["global_projected_recall"] >= 0.995
        and all(float(report["projected_recall"][str(fold)]) >= 0.995 for fold in DENOMINATORS)
        for name, report in reports.items() if name != "PRIMARY"
    )
    parent_truth = {(str(row["query_id"]), str(row["siren"])) for row in parents}
    primary_hits = _variant_hits(candidates, parent_cap=512, rescue_budget=250, site_depth=16)
    unresolved = [row for row in labels if row["cohort_kind"] == "BASE_TOP1750_MISS" and (str(row["query_id"]), str(row["human_ground_truth_siret"])) not in primary_hits]
    parent_present = sum((str(row["query_id"]), str(row["human_ground_truth_siret"])[:9]) in parent_truth for row in unresolved)
    if controls_pass and fold_pass and sensitivity_pass:
        decision = "PASS_MASTER_V15_FEASIBILITY"
    elif not controls_pass:
        decision = "STOP_INTEGRITY"
    elif unresolved and parent_present >= len(unresolved) / 2:
        decision = "PIVOT_SITE_EXPANSION"
    else:
        decision = "PIVOT_PARENT_RETRIEVAL"
    output_dir.mkdir(parents=True)
    payload = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "decision": decision,
        "config_sha256": config_sha256,
        "checks": {"controls_preserved": controls_pass, "fold_rescue_thresholds": fold_pass, "sensitivity": sensitivity_pass},
        "thresholds": {"minimum_rescues": MIN_RESCUES, "control_counts": CONTROL_COUNTS, "minimum_projected_recall": 0.995},
        "variants": reports,
        "unresolved_primary": {"count": len(unresolved), "parent_present": parent_present},
        "generation_manifest_sha256": sha256_file(generation_dir / "manifest.json"),
        "cohort_manifest_sha256": sha256_file(cohort_dir / "manifest.json"),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


__all__ = ["build_cohort", "evaluate_generation"]
