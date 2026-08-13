#!/usr/bin/env python3
"""Prepare the 43 audited fresh queries for frozen selective retrieval replay."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xgb_matcher.v9_dataset import file_sha256, normalize_siret  # noqa: E402


SCHEMA_VERSION = "sireto-v4.12-learned-fresh-retrieval-input-1"
DEFAULT_POPULATION = Path(
    "/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/"
    "v4_12_learned_unified_population/2d29be3ccd8fcc3e"
)
DEFAULT_V411_CANDIDATES = Path(
    "/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/v4_11_input_blind/"
    "ec4326ec57e4411d/candidates_sparse_top100.parquet"
)
DEFAULT_V411_LABELS = DEFAULT_V411_CANDIDATES.with_name("labels.parquet")
DEFAULT_RETRIEVAL_BENCHMARK_MANIFEST = Path(
    "/Volumes/CATNAT_DATA/SIRETO_V9/benchmarks/closed/"
    "c33b80855f560074/manifest.json"
)


def _verify_output(directory: Path, name: str, manifest: dict[str, Any]) -> Path:
    path = directory / name
    expected = manifest.get("outputs", {}).get(name)
    if expected != file_sha256(path):
        raise ValueError(f"Population output hash mismatch: {path}")
    return path


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build(args: argparse.Namespace) -> Path:
    retrieval_benchmark_manifest = json.loads(
        args.retrieval_benchmark_manifest.read_text(encoding="utf-8")
    )
    partitions_sha256 = str(
        retrieval_benchmark_manifest.get("partitions_sha256") or ""
    )
    if not partitions_sha256:
        raise ValueError("Retrieval benchmark manifest lacks partitions_sha256")
    population_manifest_path = args.population / "manifest.json"
    population_manifest = json.loads(
        population_manifest_path.read_text(encoding="utf-8")
    )
    if population_manifest.get("schema_version") != (
        "sireto-v4.12-learned-unified-population-1"
    ):
        raise ValueError("Unexpected learned-population schema")
    queries_path = _verify_output(args.population, "queries.parquet", population_manifest)
    labels_path = _verify_output(args.population, "labels.parquet", population_manifest)
    queries = pd.read_parquet(queries_path)
    labels = pd.read_parquet(labels_path)
    audited_canonical = pd.read_csv(
        args.audited_canonical, dtype=str, keep_default_na=False
    )
    candidates = pd.read_parquet(
        args.v411_candidates,
        columns=["query_id", "candidate_siret", "retrieval_rank"],
    )
    v411_labels = pd.read_parquet(args.v411_labels)
    for frame in (queries, labels, audited_canonical, candidates, v411_labels):
        frame["query_id"] = frame["query_id"].astype(str)

    fresh_queries = queries[queries["query_id"].str.startswith("fresh:")].copy()
    fresh_labels = labels[labels["query_id"].isin(set(fresh_queries["query_id"]))].copy()
    if len(fresh_queries) != args.expected_fresh_count:
        raise ValueError(
            f"Expected {args.expected_fresh_count} fresh queries, found {len(fresh_queries)}"
        )
    if set(fresh_queries["query_id"]) != set(fresh_labels["query_id"]):
        raise ValueError("Fresh query/label IDs differ")

    canonical = audited_canonical.set_index("query_id")
    v411_label_by_id = v411_labels.set_index("query_id")
    missing_canonical = set(fresh_queries["query_id"]) - set(canonical.index)
    if missing_canonical:
        raise ValueError(f"Fresh canonical labels missing: {sorted(missing_canonical)}")
    metric_target = fresh_labels.set_index("query_id")["ground_truth_siret"]
    telemetry_target = metric_target.copy()
    for query_id in telemetry_target.index:
        if not normalize_siret(telemetry_target.loc[query_id]):
            telemetry_target.loc[query_id] = normalize_siret(
                canonical.loc[query_id, "ground_truth_siret"]
            )
        if not normalize_siret(telemetry_target.loc[query_id]):
            telemetry_target.loc[query_id] = normalize_siret(
                v411_label_by_id.loc[query_id, "ground_truth_siret"]
            )
    telemetry_missing = telemetry_target.map(normalize_siret).isna()
    telemetry_target.loc[telemetry_missing] = "00000000000000"

    benchmark = fresh_queries.rename(
        columns={"crm_postcode": "postcode", "crm_insee": "insee"}
    )[
        [
            "query_id",
            "crm_name",
            "crm_address",
            "crm_city",
            "postcode",
            "insee",
        ]
    ].copy()
    label_by_id = fresh_labels.set_index("query_id")
    benchmark["split"] = "fresh_consumed_development"
    benchmark["ground_truth_siret"] = benchmark["query_id"].map(
        telemetry_target.map(normalize_siret)
    )
    benchmark["ground_truth_siren"] = benchmark["ground_truth_siret"].str[:9]
    benchmark["ground_truth_state"] = benchmark["query_id"].map(
        label_by_id["ground_truth_state"]
    )
    benchmark["location_match_type"] = benchmark["insee"].map(
        lambda value: "insee" if str(value or "").strip() else "cp_only"
    )
    benchmark["metric_label_kind"] = benchmark["query_id"].map(
        label_by_id["label_kind"]
    )
    benchmark["metric_ground_truth_siret"] = benchmark["query_id"].map(
        metric_target.map(normalize_siret)
    )
    benchmark["exact_metric_eligible"] = benchmark["metric_label_kind"].eq(
        "MATCH_EXACT"
    )
    benchmark = benchmark.sort_values("query_id").reset_index(drop=True)

    fresh_candidates = candidates[
        candidates["query_id"].isin(set(benchmark["query_id"]))
    ].copy()
    fresh_candidates["candidate_siret"] = fresh_candidates["candidate_siret"].map(
        normalize_siret
    )
    fresh_candidates = fresh_candidates.sort_values(
        ["query_id", "retrieval_rank"], kind="stable"
    )
    duplicate = fresh_candidates.duplicated(["query_id", "candidate_siret"])
    if duplicate.any():
        raise ValueError("V4.11 fresh baseline contains duplicate SIRETs")
    baseline = (
        fresh_candidates.groupby("query_id", sort=False)["candidate_siret"]
        .agg(list)
        .reindex(benchmark["query_id"])
    )
    if baseline.isna().any():
        raise ValueError("At least one fresh query lacks a V4.11 baseline pool")
    if baseline.map(len).max() > 100:
        raise ValueError("V4.11 baseline exceeds 100 candidates")
    baseline_raw = pd.DataFrame(
        {
            "query_id": benchmark["query_id"],
            "candidate_sirets_json": baseline.map(
                lambda values: json.dumps(values, separators=(",", ":"))
            ).values,
        }
    )

    identity = {
        "schema_version": SCHEMA_VERSION,
        "builder_sha256": file_sha256(Path(__file__)),
        "population_manifest_sha256": file_sha256(population_manifest_path),
        "audited_canonical_sha256": file_sha256(args.audited_canonical),
        "v411_candidates_sha256": file_sha256(args.v411_candidates),
        "v411_labels_sha256": file_sha256(args.v411_labels),
        "retrieval_benchmark_manifest_sha256": file_sha256(
            args.retrieval_benchmark_manifest
        ),
        "fresh_count": len(benchmark),
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    destination = args.output_root / build_id
    if destination.exists():
        manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("build_identity") != identity:
            raise FileExistsError(f"Conflicting immutable output: {destination}")
        return destination

    args.output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{build_id}.", dir=args.output_root))
    try:
        benchmark_path = temporary / "benchmark.parquet"
        baseline_path = temporary / "baseline_raw.parquet"
        benchmark.to_parquet(benchmark_path, index=False)
        baseline_raw.to_parquet(baseline_path, index=False)
        report = (
            "# Entrée retrieval des ajouts frais V4.12-L\n\n"
            f"- requêtes : {len(benchmark)} ;\n"
            f"- exactes évaluables : {int(benchmark['exact_metric_eligible'].sum())} ;\n"
            f"- autres labels : {int((~benchmark['exact_metric_eligible']).sum())} ;\n"
            "- plafond baseline : 100 candidats.\n\n"
            "Le SIRET historique des lignes non exactes est transmis au moteur "
            "uniquement pour sa télémétrie de replay. Il n'est ni injecté ni utilisé "
            "pour la qualification ou la métrique finale.\n"
        )
        (temporary / "report.md").write_text(report, encoding="utf-8")
        output_hashes = {
            name: file_sha256(temporary / name)
            for name in ("benchmark.parquet", "baseline_raw.parquet", "report.md")
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "build_identity": identity,
            "query_count": len(benchmark),
            "partitions_sha256": partitions_sha256,
            "exact_metric_eligible": int(benchmark["exact_metric_eligible"].sum()),
            "positive_injection": False,
            "telemetry_target_is_not_a_metric_for_open_labels": True,
            "open_label_telemetry_placeholders": int(telemetry_missing.sum()),
            "output_sha256": output_hashes,
            "outputs": output_hashes,
        }
        _json_dump(temporary / "manifest.json", manifest)
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", type=Path, default=DEFAULT_POPULATION)
    parser.add_argument(
        "--audited-canonical",
        type=Path,
        default=Path("reports/v412_review_trusted_labels_279.csv"),
    )
    parser.add_argument(
        "--v411-candidates", type=Path, default=DEFAULT_V411_CANDIDATES
    )
    parser.add_argument("--v411-labels", type=Path, default=DEFAULT_V411_LABELS)
    parser.add_argument(
        "--retrieval-benchmark-manifest",
        type=Path,
        default=DEFAULT_RETRIEVAL_BENCHMARK_MANIFEST,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/Volumes/CATNAT_DATA/SIRETO_RECALL100/benchmarks/"
            "v4_12_learned_fresh_retrieval"
        ),
    )
    parser.add_argument("--expected-fresh-count", type=int, default=43)
    return parser.parse_args()


if __name__ == "__main__":
    print(build(parse_args()))
