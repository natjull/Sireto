#!/usr/bin/env python3
"""Assemble historical and fresh pools, then decide the V4 retrieval gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import sys
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_v9_retrieval_experiment import (  # noqa: E402
    _git_commit,
    wilson_interval,
)
from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402


SCHEMA_VERSION = "sireto-v4-retrieval-gate-1"
EXPECTED = {
    "historical_core": 4932,
    "fit_addition": 819,
    "fit_combined": 5751,
    "dev_new": 305,
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_output(
    path: Path,
    manifest_path: Path,
    *,
    output_key: str | None = None,
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    expected = manifest.get("outputs", {}).get(output_key or path.name)
    if not expected or file_sha256(path) != expected:
        raise ValueError(f"Manifest hash mismatch: {path}")
    return manifest


def _normal_siret(value: Any) -> str:
    digits = "".join(
        character for character in str(value) if character.isdigit()
    )
    return digits.zfill(14)


def _metric(frame: pd.DataFrame) -> dict[str, Any]:
    successes = int(frame["hit_at_100"].sum())
    total = int(len(frame))
    low, high = wilson_interval(successes, total, confidence=0.95)
    return {
        "successes": successes,
        "total": total,
        "rate": successes / total if total else 0.0,
        "wilson_95": [low, high],
    }


def _candidate_result(
    *,
    query_id: str,
    truth: str,
    candidates: list[str],
    split: str,
    subset: str,
    provenance: str,
    oracle_hit: bool | None,
) -> dict[str, Any]:
    normalized = [_normal_siret(value) for value in candidates]
    if len(normalized) > 100:
        raise ValueError(f"{query_id}: more than 100 candidates")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{query_id}: duplicate candidates")
    normalized_truth = _normal_siret(truth)
    rank = (
        normalized.index(normalized_truth) + 1
        if normalized_truth in normalized
        else None
    )
    return {
        "query_id": str(query_id),
        "split": split,
        "subset": subset,
        "ground_truth_siret": normalized_truth,
        "ground_truth_siren": normalized_truth[:9],
        "candidate_count": len(normalized),
        "ground_truth_rank": rank,
        "hit_at_100": rank is not None,
        "source_oracle_hit": oracle_hit,
        "provenance": provenance,
        "candidate_sirets_json": json.dumps(
            normalized,
            separators=(",", ":"),
        ),
    }


def _historical_results(
    *,
    v4_dir: Path,
    downstream_dir: Path,
) -> pd.DataFrame:
    v4_manifest = _read_json(v4_dir / "manifest.json")
    label_paths = []
    for split in ("train", "dev"):
        relative = f"{split}/labels.parquet"
        path = v4_dir / relative
        expected = v4_manifest.get("outputs", {}).get(relative)
        if not expected or file_sha256(path) != expected:
            raise ValueError(f"V4 manifest hash mismatch: {path}")
        label_paths.append(path)
    labels = pd.concat(
        [pd.read_parquet(path) for path in label_paths],
        ignore_index=True,
    )
    labels = labels[labels["label_kind"].eq("MATCH_EXACT")].copy()
    labels["query_id"] = labels["query_id"].astype(str)
    if len(labels) != EXPECTED["historical_core"]:
        raise ValueError("Unexpected historical V4 exact count")

    downstream_manifest_path = downstream_dir / "manifest.json"
    downstream_manifest = _read_json(downstream_manifest_path)
    candidates_path = downstream_dir / "candidates.parquet"
    expected_hash = (
        downstream_manifest.get("input_hashes", {}).get("candidates.parquet")
        or _read_json(downstream_dir / "build_report.json")
        .get("output_hashes", {})
        .get("candidates.parquet")
    )
    if not expected_hash or file_sha256(candidates_path) != expected_hash:
        raise ValueError("Downstream candidate artifact hash mismatch")
    candidates = pd.read_parquet(
        candidates_path,
        columns=["query_id", "candidate_siret", "retrieval_rank"],
    )
    candidates["query_id"] = candidates["query_id"].astype(str)
    wanted = set(labels["query_id"])
    candidates = candidates[candidates["query_id"].isin(wanted)].copy()
    lists = {
        query_id: group.sort_values(
            "retrieval_rank",
            kind="stable",
        )["candidate_siret"].tolist()
        for query_id, group in candidates.groupby("query_id", sort=False)
    }
    if set(lists) != wanted:
        raise ValueError("Historical candidates do not cover all V4 exact rows")
    return pd.DataFrame(
        [
            _candidate_result(
                query_id=str(row.query_id),
                truth=row.ground_truth_siret,
                candidates=lists[str(row.query_id)],
                split="fit",
                subset="historical_core",
                provenance="historical_reuse",
                oracle_hit=None,
            )
            for row in labels.itertuples(index=False)
        ]
    )


def _fresh_results(
    *,
    role: str,
    fresh_dir: Path,
    admission_raw_path: Path,
    admission_manifest_path: Path,
) -> pd.DataFrame:
    fresh_manifest = _read_json(fresh_dir / "manifest.json")
    benchmark_relative = f"{role}/benchmark.parquet"
    benchmark_path = fresh_dir / benchmark_relative
    expected = fresh_manifest.get("outputs", {}).get(benchmark_relative)
    if not expected or file_sha256(benchmark_path) != expected:
        raise ValueError(f"Fresh V4 manifest hash mismatch: {benchmark_path}")
    benchmark = pd.read_parquet(benchmark_path)
    benchmark = benchmark[benchmark["label_kind"].eq("MATCH_EXACT")].copy()
    benchmark["query_id"] = benchmark["query_id"].astype(str)

    admission_manifest = _verify_output(
        admission_raw_path,
        admission_manifest_path,
    )
    if admission_manifest.get("split") != role:
        raise ValueError(f"Admission split mismatch for {role}")
    admission = pd.read_parquet(admission_raw_path)
    admission["query_id"] = admission["query_id"].astype(str)
    admission_by_query = admission.set_index("query_id", drop=False)
    if set(admission_by_query.index) != set(benchmark["query_id"]):
        raise ValueError(f"Admission query IDs do not match {role}")

    split = "fit" if role == "fit_addition" else "dev"
    expected_count = EXPECTED[role]
    if len(benchmark) != expected_count:
        raise ValueError(
            f"{role}: expected {expected_count}, found {len(benchmark)}"
        )
    return pd.DataFrame(
        [
            _candidate_result(
                query_id=str(row.query_id),
                truth=row.ground_truth_siret,
                candidates=json.loads(
                    admission_by_query.loc[
                        str(row.query_id),
                        "candidate_sirets_json",
                    ]
                ),
                split=split,
                subset=role,
                provenance="fresh_frozen_retrieval",
                oracle_hit=bool(
                    admission_by_query.loc[str(row.query_id), "oracle_hit"]
                ),
            )
            for row in benchmark.itertuples(index=False)
        ]
    )


def _verdict(
    *,
    fit_rate: float,
    dev_rate: float,
    fresh_oracle_rate: float,
    controls_pass: bool,
) -> str:
    if not controls_pass or fresh_oracle_rate < 0.99:
        return "STOP_V4_DATA"
    if fit_rate < 0.99 or dev_rate < 0.99:
        return "PIVOT_RETRIEVAL_V4"
    return "GO_RANKER_V4"


def finalize(
    *,
    v4_dir: Path,
    fresh_dir: Path,
    downstream_dir: Path,
    fit_admission_raw: Path,
    fit_admission_manifest: Path,
    dev_admission_raw: Path,
    dev_admission_manifest: Path,
    prepared_manifest_path: Path,
    contract_path: Path,
    output_root: Path,
) -> Path:
    prepared_manifest = _read_json(prepared_manifest_path)
    if prepared_manifest.get("holdout_read") is not False:
        raise ValueError("Prepared input did not preserve holdout sealing")
    if prepared_manifest.get("old_test_read") is not False:
        raise ValueError("Prepared input read the old test")

    historical = _historical_results(
        v4_dir=v4_dir,
        downstream_dir=downstream_dir,
    )
    fresh_fit = _fresh_results(
        role="fit_addition",
        fresh_dir=fresh_dir,
        admission_raw_path=fit_admission_raw,
        admission_manifest_path=fit_admission_manifest,
    )
    fresh_dev = _fresh_results(
        role="dev_new",
        fresh_dir=fresh_dir,
        admission_raw_path=dev_admission_raw,
        admission_manifest_path=dev_admission_manifest,
    )
    fit = pd.concat([historical, fresh_fit], ignore_index=True)
    dev = fresh_dev.reset_index(drop=True)
    all_rows = pd.concat([fit, dev], ignore_index=True)

    fit_sirens = set(fit["ground_truth_siren"])
    dev_sirens = set(dev["ground_truth_siren"])
    controls = {
        "historical_count": len(historical) == EXPECTED["historical_core"],
        "fit_addition_count": len(fresh_fit) == EXPECTED["fit_addition"],
        "fit_combined_count": len(fit) == EXPECTED["fit_combined"],
        "dev_new_count": len(dev) == EXPECTED["dev_new"],
        "unique_query_ids": not all_rows["query_id"].duplicated().any(),
        "zero_fit_dev_siren_overlap": not bool(fit_sirens & dev_sirens),
        "max_candidates_at_most_100": int(
            all_rows["candidate_count"].max()
        ) <= 100,
        "holdout_not_read": True,
        "old_test_not_read": True,
        "no_positive_injected": True,
    }
    metrics = {
        "historical_core": _metric(historical),
        "fit_addition": _metric(fresh_fit),
        "fit_combined": _metric(fit),
        "dev_new": _metric(dev),
        "fresh_source_oracle": _metric(
            pd.concat([fresh_fit, fresh_dev], ignore_index=True).assign(
                hit_at_100=lambda frame: frame["source_oracle_hit"].astype(bool)
            )
        ),
    }
    verdict = _verdict(
        fit_rate=metrics["fit_combined"]["rate"],
        dev_rate=metrics["dev_new"]["rate"],
        fresh_oracle_rate=metrics["fresh_source_oracle"]["rate"],
        controls_pass=all(controls.values()),
    )
    identity = {
        "schema_version": SCHEMA_VERSION,
        "v4_manifest_sha256": file_sha256(v4_dir / "manifest.json"),
        "fresh_manifest_sha256": file_sha256(fresh_dir / "manifest.json"),
        "downstream_manifest_sha256": file_sha256(
            downstream_dir / "manifest.json"
        ),
        "fit_admission_manifest_sha256": file_sha256(
            fit_admission_manifest
        ),
        "dev_admission_manifest_sha256": file_sha256(
            dev_admission_manifest
        ),
        "prepared_manifest_sha256": file_sha256(prepared_manifest_path),
        "contract_sha256": file_sha256(contract_path),
        "git_commit": _git_commit(),
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    output_dir = output_root / build_id
    output_dir.mkdir(parents=True, exist_ok=False)

    fit_path = output_dir / "fit_exact.parquet"
    dev_path = output_dir / "dev_exact.parquet"
    fit_misses_path = output_dir / "misses_fit.parquet"
    dev_misses_path = output_dir / "misses_dev.parquet"
    fit.to_parquet(fit_path, index=False)
    dev.to_parquet(dev_path, index=False)
    fit[~fit["hit_at_100"]].to_parquet(fit_misses_path, index=False)
    dev[~dev["hit_at_100"]].to_parquet(dev_misses_path, index=False)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "build_id": build_id,
        "verdict": verdict,
        "target_recall_at_100": 0.99,
        "metrics": metrics,
        "controls": controls,
        "fit_dev_shared_sirens": sorted(fit_sirens & dev_sirens),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "build_id": build_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "identity": identity,
        "verdict": verdict,
        "holdout_read": False,
        "old_test_read": False,
        "positive_injected": False,
        "outputs": {
            path.name: file_sha256(path)
            for path in (
                fit_path,
                dev_path,
                fit_misses_path,
                dev_misses_path,
                summary_path,
            )
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4-dir", type=Path, required=True)
    parser.add_argument("--fresh-dir", type=Path, required=True)
    parser.add_argument("--downstream-dir", type=Path, required=True)
    parser.add_argument("--fit-admission-raw", type=Path, required=True)
    parser.add_argument("--fit-admission-manifest", type=Path, required=True)
    parser.add_argument("--dev-admission-raw", type=Path, required=True)
    parser.add_argument("--dev-admission-manifest", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(
        finalize(
            v4_dir=args.v4_dir,
            fresh_dir=args.fresh_dir,
            downstream_dir=args.downstream_dir,
            fit_admission_raw=args.fit_admission_raw,
            fit_admission_manifest=args.fit_admission_manifest,
            dev_admission_raw=args.dev_admission_raw,
            dev_admission_manifest=args.dev_admission_manifest,
            prepared_manifest_path=args.prepared_manifest,
            contract_path=args.contract,
            output_root=args.output_root,
        )
    )


if __name__ == "__main__":
    main()
