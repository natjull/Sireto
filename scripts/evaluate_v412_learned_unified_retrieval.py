#!/usr/bin/env python3
"""Publish the frozen selective top-100 retrieval on the V4.12-L population."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterable

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_v9_retrieval_experiment import _binary_metric  # noqa: E402
from src.xgb_matcher.v9_dataset import file_sha256, normalize_siret  # noqa: E402


SCHEMA_VERSION = "sireto-v4.12-learned-unified-retrieval-evaluation-1"
POLICY_ID = "selective-sparse-rrf-v3-frozen-top100"
DEFAULT_POPULATION = Path(
    "/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/"
    "v4_12_learned_unified_population/2d29be3ccd8fcc3e"
)
DEFAULT_HISTORICAL_ADMISSIONS = (
    Path(
        "/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/"
        "admission_train_c33b80855f560074_734f792"
    ),
    Path(
        "/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/"
        "admission_diagnostic_dev_c33b80855f560074_5a0e67f"
    ),
    Path(
        "/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/"
        "admission_diagnostic_test_c33b80855f560074_eb0e6a3"
    ),
)
DEFAULT_FRESH_ADMISSION = Path(
    "/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/"
    "v4_12_learned_fresh_admission/1465a04be44b13c3"
)
DEFAULT_QUALIFICATION_DIRS = (
    Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100/benchmarks/qualification_v3/a76eebf6a8b157ea"),
    Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100/benchmarks/qualification_v3/ab8343817551c0a5"),
    Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100/benchmarks/qualification_v3/72cc411a916c4814"),
)


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _verified_output(directory: Path, name: str) -> tuple[Path, dict[str, Any]]:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = directory / name
    expected = manifest.get("outputs", {}).get(name)
    if expected != file_sha256(path):
        raise ValueError(f"Artifact hash mismatch: {path}")
    return path, manifest


def _load_admissions(
    directories: Iterable[Path],
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    frames: list[pd.DataFrame] = []
    records: list[dict[str, str]] = []
    for directory in directories:
        raw_path, _manifest = _verified_output(directory, "raw_results.parquet")
        manifest_path = directory / "manifest.json"
        frames.append(pd.read_parquet(raw_path))
        records.extend(
            [
                {"path": str(raw_path), "sha256": file_sha256(raw_path)},
                {"path": str(manifest_path), "sha256": file_sha256(manifest_path)},
            ]
        )
    output = pd.concat(frames, ignore_index=True)
    output["query_id"] = output["query_id"].astype(str)
    if output["query_id"].duplicated().any():
        raise ValueError("Admission query_id values must be unique")
    return output, records


def _parse_pool(value: str) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for raw in json.loads(value):
        siret = normalize_siret(raw)
        if not siret:
            continue
        if siret in seen:
            raise ValueError(f"Duplicate SIRET in candidate pool: {siret}")
        seen.add(siret)
        output.append(siret)
    if len(output) > 100:
        raise ValueError(f"Candidate pool exceeds ceiling: {len(output)}")
    return output


def _metric(hits: pd.Series) -> dict[str, Any]:
    if hits.empty:
        return {
            "successes": 0,
            "total": 0,
            "rate": None,
            "wilson_95": None,
            "wilson_99": None,
        }
    return _binary_metric(hits.astype(bool))


def _coverage(exact: pd.Series) -> dict[str, Any]:
    return _binary_metric(exact.astype(bool))


def _original_metrics(
    qualified: pd.DataFrame,
    pools: pd.DataFrame,
) -> dict[str, Any]:
    qualified = qualified.copy()
    qualified["query_id"] = qualified["query_id"].astype(str)
    joined = qualified.merge(
        pools[["query_id", "candidate_sirets"]],
        on="query_id",
        validate="one_to_one",
    )
    historical_target = joined["historical_ground_truth_siret"].map(normalize_siret)
    joined["historical_hit"] = [
        target in candidates
        for target, candidates in zip(
            historical_target, joined["candidate_sirets"], strict=True
        )
    ]
    v2_exact = joined["v2_label_kind"].eq("MATCH_EXACT")
    v3_exact = joined["label_kind"].eq("MATCH_EXACT")
    return {
        "historical_all": _metric(joined["historical_hit"]),
        "v2_exact": {
            "coverage": _coverage(v2_exact),
            "recall_at_100": _metric(joined.loc[v2_exact, "historical_hit"]),
        },
        "v3_exact": {
            "coverage": _coverage(v3_exact),
            "recall_at_100": _metric(joined.loc[v3_exact, "historical_hit"]),
        },
    }


def _segment_metrics(outcomes: pd.DataFrame) -> dict[str, Any]:
    exact = outcomes["label_kind"].eq("MATCH_EXACT")
    masks: dict[str, pd.Series] = {
        "active": outcomes["ground_truth_state"].eq("A"),
        "closed": outcomes["ground_truth_state"].eq("F"),
        "mega": outcomes["mega_base_pool"].fillna(False).astype(bool),
        "multi_site": outcomes["multi_site_siren"].fillna(False).astype(bool),
        "historical": ~outcomes["query_id"].str.startswith("fresh:"),
        "fresh_audited": outcomes["query_id"].str.startswith("fresh:"),
    }
    for fold in range(5):
        masks[f"oof_fold_{fold}"] = outcomes["oof_fold"].eq(fold)
    output: dict[str, Any] = {}
    for name, mask in masks.items():
        scoped_exact = mask & exact
        output[name] = {
            "query_count": int(mask.sum()),
            "exact_count": int(scoped_exact.sum()),
            "coverage": _coverage(exact.loc[mask]),
            "recall_at_100": _metric(outcomes.loc[scoped_exact, "hit_at_100"]),
        }
    return output


def build(args: argparse.Namespace) -> Path:
    population_manifest_path = args.population / "manifest.json"
    population_manifest = json.loads(
        population_manifest_path.read_text(encoding="utf-8")
    )
    labels_path = args.population / "labels.parquet"
    queries_path = args.population / "queries.parquet"
    for path in (labels_path, queries_path):
        expected = population_manifest.get("outputs", {}).get(path.name)
        if expected != file_sha256(path):
            raise ValueError(f"Population hash mismatch: {path}")
    labels = pd.read_parquet(labels_path)
    queries = pd.read_parquet(queries_path)
    labels["query_id"] = labels["query_id"].astype(str)
    queries["query_id"] = queries["query_id"].astype(str)

    historical, input_records = _load_admissions(args.historical_admission)
    fresh, fresh_records = _load_admissions([args.fresh_admission])
    input_records.extend(fresh_records)
    if len(historical) != args.expected_historical_count:
        raise ValueError("Historical admission count differs from the frozen population")
    if len(fresh) != args.expected_fresh_count:
        raise ValueError("Fresh admission count differs from the audited additions")
    admissions = pd.concat([historical, fresh], ignore_index=True)
    if set(admissions["query_id"]) != set(labels["query_id"]):
        raise ValueError("Admission and learned-population query IDs differ")

    admissions["candidate_sirets"] = admissions["candidate_sirets_json"].map(
        _parse_pool
    )
    admissions["candidate_count_recomputed"] = admissions[
        "candidate_sirets"
    ].map(len)
    if not admissions["candidate_count_recomputed"].eq(
        admissions["candidate_count"].astype(int)
    ).all():
        raise ValueError("Published and recomputed candidate counts differ")

    pool_columns = [
        "query_id",
        "candidate_sirets_json",
        "candidate_sirets",
        "candidate_count_recomputed",
        "mega_base_pool",
        "multi_site_siren",
        "location_match_type",
    ]
    pools = admissions[pool_columns].rename(
        columns={"candidate_count_recomputed": "candidate_count"}
    )
    merged = labels.merge(
        pools,
        on="query_id",
        validate="one_to_one",
    ).merge(
        queries[["query_id", "crm_insee"]],
        on="query_id",
        validate="one_to_one",
    )
    exact = merged["label_kind"].eq("MATCH_EXACT")
    merged["hit_at_100"] = pd.Series(pd.NA, index=merged.index, dtype="boolean")
    merged.loc[exact, "hit_at_100"] = [
        target in candidates
        for target, candidates in zip(
            merged.loc[exact, "ground_truth_siret"],
            merged.loc[exact, "candidate_sirets"],
            strict=True,
        )
    ]
    merged["miss_reason"] = "NOT_EVALUATED_OPEN_LABEL"
    merged.loc[exact & merged["hit_at_100"].fillna(False), "miss_reason"] = ""
    merged.loc[exact & ~merged["hit_at_100"].fillna(False), "miss_reason"] = (
        "ABSENT_FROM_TOP100"
    )
    merged["retrieval_policy_id"] = POLICY_ID

    qualification_frames: list[pd.DataFrame] = []
    for directory in args.qualification_dir:
        path, _manifest = _verified_output(directory, "benchmark.parquet")
        qualification_frames.append(pd.read_parquet(path))
        input_records.extend(
            [
                {"path": str(path), "sha256": file_sha256(path)},
                {
                    "path": str(directory / "manifest.json"),
                    "sha256": file_sha256(directory / "manifest.json"),
                },
            ]
        )
    qualified = pd.concat(qualification_frames, ignore_index=True)
    historical_pools = pools[~pools["query_id"].str.startswith("fresh:")]
    original = _original_metrics(qualified, historical_pools)

    current_recall = _metric(merged.loc[exact, "hit_at_100"])
    current_coverage = _coverage(exact)
    max_candidates = int(merged["candidate_count"].max())
    gates = {
        "coverage": {
            "minimum": 0.80,
            "observed": current_coverage["rate"],
            "passed": bool(current_coverage["rate"] >= 0.80),
        },
        "recall_at_100": {
            "minimum": 0.99,
            "observed": current_recall["rate"],
            "passed": bool(current_recall["rate"] >= 0.99),
        },
        "candidate_ceiling": {
            "maximum": 100,
            "observed": max_candidates,
            "passed": bool(max_candidates <= 100),
        },
    }
    decision = (
        "GO_RANKER_TRAINING"
        if all(gate["passed"] for gate in gates.values())
        else "PIVOT_RETRIEVAL"
    )
    metrics = {
        "decision": decision,
        "development_status": "CONSUMED_OOF_NOT_INDEPENDENT_CERTIFICATION",
        "gates": gates,
        "v412_learned": {
            "coverage": current_coverage,
            "recall_at_100": current_recall,
        },
        "original_references": original,
        "segments": _segment_metrics(merged),
        "miss_count": int((exact & ~merged["hit_at_100"].fillna(False)).sum()),
    }

    tracked = [population_manifest_path, labels_path, queries_path]
    input_records.extend(
        {"path": str(path), "sha256": file_sha256(path)} for path in tracked
    )
    identity = {
        "schema_version": SCHEMA_VERSION,
        "builder_sha256": file_sha256(Path(__file__)),
        "policy_id": POLICY_ID,
        "inputs": sorted(input_records, key=lambda record: record["path"]),
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    destination = args.output_root / build_id
    if destination.exists():
        existing = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        if existing.get("build_identity") != identity:
            raise FileExistsError(f"Conflicting immutable output: {destination}")
        return destination

    args.output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{build_id}.", dir=args.output_root))
    try:
        candidate_pools = merged[
            ["query_id", "candidate_sirets_json", "candidate_count", "retrieval_policy_id"]
        ].sort_values("query_id")
        outcome_columns = [
            "query_id",
            "label_kind",
            "ground_truth_siret",
            "ground_truth_siren",
            "ground_truth_state",
            "oof_fold",
            "candidate_count",
            "hit_at_100",
            "miss_reason",
            "mega_base_pool",
            "multi_site_siren",
            "location_match_type",
            "retrieval_policy_id",
        ]
        outcomes = merged[outcome_columns].sort_values("query_id")
        candidate_pools.to_parquet(temporary / "candidate_pools.parquet", index=False)
        outcomes.to_parquet(temporary / "query_outcomes.parquet", index=False)
        _json_dump(temporary / "metrics.json", metrics)
        report = (
            "# Retrieval V4.12-L unifié\n\n"
            f"Décision : **{decision}** (développement OOF consommé, pas certification indépendante).\n\n"
            "| Vue | Couverture | Recall@100 |\n"
            "|---|---:|---:|\n"
            f"| Historique 17 054 | 100 % | {original['historical_all']['rate']:.3%} |\n"
            f"| V2 exact | {original['v2_exact']['coverage']['rate']:.3%} | "
            f"{original['v2_exact']['recall_at_100']['rate']:.3%} |\n"
            f"| V3 exact | {original['v3_exact']['coverage']['rate']:.3%} | "
            f"{original['v3_exact']['recall_at_100']['rate']:.3%} |\n"
            f"| V4.12-L corrigé | {current_coverage['rate']:.3%} | "
            f"{current_recall['rate']:.3%} |\n\n"
            f"Le bon SIRET manque dans {metrics['miss_count']} des "
            f"{current_recall['total']} dossiers exacts. Aucun pool ne dépasse "
            f"{max_candidates} candidats.\n"
        )
        (temporary / "report.md").write_text(report, encoding="utf-8")
        output_names = [
            "candidate_pools.parquet",
            "query_outcomes.parquet",
            "metrics.json",
            "report.md",
        ]
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "build_identity": identity,
            "decision": decision,
            "query_count": len(merged),
            "exact_query_count": int(exact.sum()),
            "candidate_ceiling": 100,
            "positive_injection": False,
            "independent_certification": False,
            "outputs": {
                name: file_sha256(temporary / name) for name in output_names
            },
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
        "--historical-admission", type=Path, action="append", default=None
    )
    parser.add_argument("--fresh-admission", type=Path, default=DEFAULT_FRESH_ADMISSION)
    parser.add_argument(
        "--qualification-dir", type=Path, action="append", default=None
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/Volumes/CATNAT_DATA/SIRETO_RECALL100/evaluations/"
            "v4_12_learned_unified_retrieval"
        ),
    )
    parser.add_argument("--expected-historical-count", type=int, default=17_054)
    parser.add_argument("--expected-fresh-count", type=int, default=43)
    args = parser.parse_args()
    if args.historical_admission is None:
        args.historical_admission = list(DEFAULT_HISTORICAL_ADMISSIONS)
    if args.qualification_dir is None:
        args.qualification_dir = list(DEFAULT_QUALIFICATION_DIRS)
    return args


if __name__ == "__main__":
    print(build(parse_args()))
