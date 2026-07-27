#!/usr/bin/env python3
"""Evaluate V4.2 variant B on the frozen representative exact cases."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_v41_training_dataset import _path_signature  # noqa: E402
from src.xgb_matcher.features import preprocess_crm_row  # noqa: E402
from src.xgb_matcher.partitioned_store import PartitionedCandidateStore  # noqa: E402
from src.xgb_matcher.tfidf_cache import TfidfPersistentCache  # noqa: E402
from src.xgb_matcher.v41_retrieval import (  # noqa: E402
    V41CandidateRetriever,
    V41CurrentStateStore,
    V41GlobalCandidateStore,
    V41RetrievalConfig,
    V41RetrievalVariant,
)
from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402


SCHEMA_VERSION = "sireto-v4.2-representative-retrieval-evaluation-1"
EXPECTED_EXACT_COUNT = 242
EXPECTED_RANDOM_EXACT_COUNT = 91
EXPECTED_BASELINE_HIT_COUNT = 237


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _assert_manifest_output(
    *,
    artifact_path: Path,
    manifest: Mapping[str, Any],
) -> None:
    expected = (manifest.get("outputs") or {}).get(artifact_path.name)
    if expected is None:
        raise ValueError(f"Manifest does not declare {artifact_path.name}")
    if file_sha256(artifact_path) != expected:
        raise ValueError(f"Hash mismatch for frozen input {artifact_path}")


def load_frozen_exact_cases(
    *,
    blind_cases_path: Path,
    adjudications_path: Path,
    sample_registry_path: Path,
    baseline_top10_path: Path,
    enforce_contract_counts: bool = True,
) -> pd.DataFrame:
    blind = pd.read_parquet(blind_cases_path)
    labels = pd.read_parquet(adjudications_path)
    registry = pd.read_parquet(
        sample_registry_path,
        columns=["audit_case_id", "sampling_stratum"],
    )
    baseline = pd.read_parquet(
        baseline_top10_path,
        columns=["service_id", "candidate_siret"],
    )
    for name, frame, required in (
        (
            "blind cases",
            blind,
            {
                "audit_case_id",
                "service_id",
                "SITE",
                "CODE_POSTAL",
                "CODE_INSEE",
                "COMMUNE",
                "SIRET",
                "SITE_CLI_ADRESSE",
                "SITE_CLI_COMMUNE",
            },
        ),
        (
            "adjudications",
            labels,
            {
                "audit_case_id",
                "service_id",
                "label_kind",
                "ground_truth_siret",
                "adjudication_status",
            },
        ),
        ("sample registry", registry, {"audit_case_id", "sampling_stratum"}),
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} is missing columns: {missing}")
    if blind["audit_case_id"].duplicated().any():
        raise ValueError("Blind audit_case_id must be unique")
    if labels["audit_case_id"].duplicated().any():
        raise ValueError("Adjudication audit_case_id must be unique")
    if registry["audit_case_id"].duplicated().any():
        raise ValueError("Registry audit_case_id must be unique")
    if set(labels["adjudication_status"].astype(str)) != {"PROVISIONAL"}:
        raise ValueError("V4.2 expects the frozen PROVISIONAL adjudications")

    exact = labels.loc[
        labels["label_kind"].astype(str).eq("MATCH_EXACT")
    ].copy()
    cases = exact.merge(
        blind,
        on=["audit_case_id", "service_id"],
        how="inner",
        validate="one_to_one",
    ).merge(
        registry,
        on="audit_case_id",
        how="inner",
        validate="one_to_one",
    )
    if len(cases) != len(exact):
        raise ValueError("Frozen exact labels do not join one-to-one to audit inputs")
    baseline_pairs = {
        (str(row.service_id), str(row.candidate_siret).zfill(14))
        for row in baseline.itertuples(index=False)
    }
    cases["ground_truth_siret"] = (
        cases["ground_truth_siret"].astype(str).str.zfill(14)
    )
    cases["baseline_a_top10_hit"] = [
        (str(service_id), str(truth)) in baseline_pairs
        for service_id, truth in zip(
            cases["service_id"],
            cases["ground_truth_siret"],
        )
    ]
    cases = cases.sort_values("audit_case_id").reset_index(drop=True)
    if enforce_contract_counts:
        random_count = int(
            cases["sampling_stratum"].eq("RANDOM_POPULATION").sum()
        )
        baseline_count = int(cases["baseline_a_top10_hit"].sum())
        if len(cases) != EXPECTED_EXACT_COUNT:
            raise ValueError(
                f"Frozen exact population changed: {len(cases)} "
                f"!= {EXPECTED_EXACT_COUNT}"
            )
        if random_count != EXPECTED_RANDOM_EXACT_COUNT:
            raise ValueError(
                f"Frozen random exact population changed: {random_count} "
                f"!= {EXPECTED_RANDOM_EXACT_COUNT}"
            )
        if baseline_count != EXPECTED_BASELINE_HIT_COUNT:
            raise ValueError(
                f"Frozen A baseline changed: {baseline_count} "
                f"!= {EXPECTED_BASELINE_HIT_COUNT}"
            )
    return cases


def summarize_results(results: pd.DataFrame) -> dict[str, Any]:
    if results.empty:
        raise ValueError("Cannot summarize empty V4.2 results")
    random = results.loc[
        results["sampling_stratum"].eq("RANDOM_POPULATION")
    ]
    baseline_hits = results.loc[results["baseline_a_top10_hit"].astype(bool)]
    misses = results.loc[~results["hit_at_100"].astype(bool)]
    recall_count = int(results["hit_at_100"].sum())
    random_recall_count = int(random["hit_at_100"].sum())
    regressed_count = int((~baseline_hits["hit_at_100"].astype(bool)).sum())
    max_candidates = int(results["candidate_count"].max())
    closed_count = int(results["closed_candidate_count"].sum())
    truth_state_missing_count = int(results["truth_state"].ne("A").sum())
    positive_injected = bool(results["positive_injected"].any())
    gates = {
        "recall_at_100_gte_0_99": recall_count / len(results) >= 0.99,
        "candidate_ceiling_lte_100": max_candidates <= 100,
        "closed_candidate_count_eq_0": closed_count == 0,
        "positive_injection_false": not positive_injected,
        "truth_state_missing_count_eq_0": truth_state_missing_count == 0,
        "baseline_a_regression_count_eq_0": regressed_count == 0,
    }
    return {
        "query_count": int(len(results)),
        "variant": "B",
        "recall_at_100": {
            "successes": recall_count,
            "total": int(len(results)),
            "rate": recall_count / len(results),
        },
        "random_population_recall_at_100": {
            "successes": random_recall_count,
            "total": int(len(random)),
            "rate": random_recall_count / len(random) if len(random) else 0.0,
        },
        "baseline_a_top10_hit_count": int(
            results["baseline_a_top10_hit"].sum()
        ),
        "baseline_a_regression_count": regressed_count,
        "miss_count": int(len(misses)),
        "miss_service_ids": misses["service_id"].astype(str).tolist(),
        "max_candidate_count": max_candidates,
        "closed_candidate_count": closed_count,
        "truth_state_missing_count": truth_state_missing_count,
        "positive_injection": positive_injected,
        "latency_ms": {
            "p50": float(np.quantile(results["latency_ms"], 0.5)),
            "p95": float(np.quantile(results["latency_ms"], 0.95)),
            "max": float(results["latency_ms"].max()),
        },
        "gates": gates,
        "verdict": "GO_HARD_LABELS" if all(gates.values()) else "PIVOT",
    }


def evaluate(
    *,
    blind_cases_path: Path,
    adjudications_path: Path,
    sample_registry_path: Path,
    sample_manifest_path: Path,
    evidence_manifest_path: Path,
    baseline_top10_path: Path,
    partitions_dir: Path,
    global_store_path: Path,
    state_snapshot_path: Path,
    cache_dir: Path,
    output_dir: Path,
) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"Immutable output directory exists: {output_dir}")
    sample_manifest = json.loads(sample_manifest_path.read_text(encoding="utf-8"))
    evidence_manifest = json.loads(
        evidence_manifest_path.read_text(encoding="utf-8")
    )
    _assert_manifest_output(
        artifact_path=blind_cases_path,
        manifest=sample_manifest,
    )
    _assert_manifest_output(
        artifact_path=sample_registry_path,
        manifest=sample_manifest,
    )
    _assert_manifest_output(
        artifact_path=adjudications_path,
        manifest=evidence_manifest,
    )
    state_snapshot_hash = file_sha256(state_snapshot_path)
    if state_snapshot_hash != evidence_manifest.get(
        "establishment_snapshot_sha256"
    ):
        raise ValueError(
            "Current-state snapshot differs from the frozen evidence snapshot"
        )
    cases = load_frozen_exact_cases(
        blind_cases_path=blind_cases_path,
        adjudications_path=adjudications_path,
        sample_registry_path=sample_registry_path,
        baseline_top10_path=baseline_top10_path,
    )
    config = V41RetrievalConfig(
        variant=V41RetrievalVariant.B_INPUT_EVIDENCE,
        max_candidates=100,
    )
    persistent_cache = TfidfPersistentCache(
        config.sparse_config().tfidf_artifact_hash(),
        cache_dir=cache_dir,
    )
    partitioned_store = PartitionedCandidateStore(partitions_dir)
    records: list[dict[str, Any]] = []
    with (
        V41GlobalCandidateStore(global_store_path) as global_store,
        V41CurrentStateStore(state_snapshot_path) as state_store,
    ):
        truth_states = state_store.get_candidate_states(
            cases["ground_truth_siret"].tolist()
        )
        retriever = V41CandidateRetriever(
            partitioned_store=partitioned_store,
            global_store=global_store,
            current_state_store=state_store,
            config=config,
        )
        for row in cases.itertuples(index=False):
            crm_row = {
                "query_id": str(row.audit_case_id),
                "crm_id": str(row.audit_case_id),
                "crm_name": str(row.SITE or ""),
                "crm_address": str(row.SITE_CLI_ADRESSE or ""),
                "crm_city": str(row.SITE_CLI_COMMUNE or row.COMMUNE or ""),
                "crm_city_addr": str(
                    row.SITE_CLI_COMMUNE or row.COMMUNE or ""
                ),
                "postcode": str(row.CODE_POSTAL or ""),
                "insee": str(row.CODE_INSEE or ""),
            }
            started = time.perf_counter()
            result = retriever.build(
                crm_row=crm_row,
                crm_pre=preprocess_crm_row(crm_row),
                input_siret=row.SIRET,
                gt_siret=None,
                persistent_cache=persistent_cache,
            )
            latency_ms = (time.perf_counter() - started) * 1000.0
            candidate_sirets = [
                str(candidate["siret"]) for candidate in result.candidates
            ]
            truth = str(row.ground_truth_siret)
            records.append(
                {
                    "audit_case_id": str(row.audit_case_id),
                    "service_id": str(row.service_id),
                    "sampling_stratum": str(row.sampling_stratum),
                    "ground_truth_siret": truth,
                    "truth_state": truth_states.get(truth),
                    "baseline_a_top10_hit": bool(row.baseline_a_top10_hit),
                    "hit_at_100": truth in candidate_sirets,
                    "ground_truth_rank": (
                        candidate_sirets.index(truth) + 1
                        if truth in candidate_sirets
                        else None
                    ),
                    "candidate_count": len(candidate_sirets),
                    "closed_candidate_count": sum(
                        str(candidate.get("etat_admin") or "").upper() == "F"
                        for candidate in result.candidates
                    ),
                    "input_siret_state": result.input_siret.state.value,
                    "candidate_sirets_json": json.dumps(candidate_sirets),
                    "channels_json": json.dumps(
                        result.channels,
                        sort_keys=True,
                    ),
                    "latency_ms": latency_ms,
                    "positive_injected": False,
                }
            )
    results = pd.DataFrame(records)
    summary = summarize_results(results)
    summary["tfidf_cache"] = persistent_cache.stats()

    output_dir.mkdir(parents=True, exist_ok=False)
    raw_path = output_dir / "raw_results.parquet"
    misses_path = output_dir / "misses.csv"
    summary_path = output_dir / "summary.json"
    results.to_parquet(raw_path, index=False)
    results.loc[~results["hit_at_100"]].to_csv(misses_path, index=False)
    _json_dump(summary_path, summary)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "positive_injection": False,
        "adjudication_status": "PROVISIONAL",
        "precision_claim_allowed": False,
        "inputs": {
            "blind_cases": {
                "path": str(blind_cases_path),
                "sha256": file_sha256(blind_cases_path),
            },
            "adjudications": {
                "path": str(adjudications_path),
                "sha256": file_sha256(adjudications_path),
            },
            "sample_registry": {
                "path": str(sample_registry_path),
                "sha256": file_sha256(sample_registry_path),
            },
            "baseline_top10": {
                "path": str(baseline_top10_path),
                "sha256": file_sha256(baseline_top10_path),
            },
            "partitions": {
                "path": str(partitions_dir),
                "runtime_signature": _path_signature(partitions_dir),
            },
            "global_store": {
                "path": str(global_store_path),
                "runtime_signature": _path_signature(global_store_path),
            },
            "current_state_snapshot": {
                "path": str(state_snapshot_path),
                "sha256": state_snapshot_hash,
                "row_count": 42_322_035,
            },
        },
        "retrieval": {
            "v41_config": config.to_dict(),
            "v41_signature": config.signature(),
            "state_authority": "COMPLETE_SIRENE_SNAPSHOT",
            "candidate_ceiling": 100,
        },
        "gates": summary["gates"],
        "verdict": summary["verdict"],
        "outputs": {
            "raw_results.parquet": file_sha256(raw_path),
            "misses.csv": file_sha256(misses_path),
            "summary.json": file_sha256(summary_path),
        },
    }
    _json_dump(output_dir / "manifest.json", manifest)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blind-cases", type=Path, required=True)
    parser.add_argument("--adjudications", type=Path, required=True)
    parser.add_argument("--sample-registry", type=Path, required=True)
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--evidence-manifest", type=Path, required=True)
    parser.add_argument("--baseline-top10", type=Path, required=True)
    parser.add_argument("--partitions-dir", type=Path, required=True)
    parser.add_argument("--global-store", type=Path, required=True)
    parser.add_argument("--state-snapshot", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = evaluate(
        blind_cases_path=args.blind_cases,
        adjudications_path=args.adjudications,
        sample_registry_path=args.sample_registry,
        sample_manifest_path=args.sample_manifest,
        evidence_manifest_path=args.evidence_manifest,
        baseline_top10_path=args.baseline_top10,
        partitions_dir=args.partitions_dir,
        global_store_path=args.global_store,
        state_snapshot_path=args.state_snapshot,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
    )
    print(output)


if __name__ == "__main__":
    main()
