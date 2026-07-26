#!/usr/bin/env python3
"""Qualify CRM rows outside the frozen benchmark and split them by SIREN."""

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

from scripts.build_benchmark_v4_current_snapshot import (  # noqa: E402
    POLICY_VERSION,
    _label_view,
    apply_current_snapshot_policy,
    audit_split,
    summarize_split,
)
from scripts.freeze_v9_closed_benchmark import directory_tree_sha256  # noqa: E402
from scripts.run_v9_retrieval_experiment import _git_commit  # noqa: E402
from src.xgb_matcher.contracts import GroundTruthKind  # noqa: E402
from src.xgb_matcher.partitioned_store import PartitionedCandidateStore  # noqa: E402
from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402


SCHEMA_VERSION = "sireto-v4-fresh-expansion-1"
SPLIT_SEED = 42
ROLE_ORDER = ("fit_addition", "dev_new", "holdout_sealed")


def _identifier(value: Any, width: int) -> str | None:
    if value is None or pd.isna(value):
        return None
    digits = "".join(character for character in str(value) if character.isdigit())
    return digits.zfill(width) if len(digits) == width else None


def build_fresh_benchmark(
    raw: pd.DataFrame,
    *,
    benchmark_service_ids: set[str],
) -> pd.DataFrame:
    """Return canonical unqualified rows whose service IDs are benchmark-new."""
    required = {
        "SITE",
        "CODE_POSTAL",
        "CODE_INSEE",
        "SERVICE ID",
        "COMMUNE",
        "SIRET",
        "SITE_CLI_ADRESSE",
        "SITE_CLI_COMMUNE",
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"CRM source is missing columns: {sorted(missing)}")
    source = raw.copy()
    source["SERVICE ID"] = source["SERVICE ID"].fillna("").astype(str).str.strip()
    source = source[
        source["SERVICE ID"].ne("")
        & ~source["SERVICE ID"].isin(benchmark_service_ids)
    ].copy()
    if source["SERVICE ID"].duplicated().any():
        raise ValueError("Fresh SERVICE ID values must be unique")
    historical_siret = source["SIRET"].map(lambda value: _identifier(value, 14))
    output = pd.DataFrame(
        {
            "query_id": "fresh:" + source["SERVICE ID"],
            "crm_record_id": source["SERVICE ID"],
            "crm_name": source["SITE"].fillna(""),
            "crm_address": source["SITE_CLI_ADRESSE"].fillna(""),
            "crm_city": source["COMMUNE"].fillna(
                source["SITE_CLI_COMMUNE"]
            ),
            "postcode": source["CODE_POSTAL"].fillna(""),
            "insee": source["CODE_INSEE"].fillna(""),
            "ground_truth_state": None,
            "ground_truth_insee": None,
            "ground_truth_postcode": None,
            "location_match_type": "source_insee_cp",
            "split": "fresh",
            "date_reference": None,
            "label_kind": GroundTruthKind.UNRESOLVED.value,
            "ground_truth_siret": None,
            "ground_truth_siren": None,
            "source": "fresh_crm_outside_benchmark",
            "validator": "mechanical_v4_current_snapshot",
            "reference_date": None,
            "historical_ground_truth_siret": historical_siret,
            "historical_ground_truth_siren": historical_siret.str[:9],
            "qualification_reason": "FRESH_UNQUALIFIED",
            "exact_metric_eligible": False,
        }
    )
    return output.sort_values("query_id").reset_index(drop=True)


def role_for_group(group_key: str, *, force_fit: bool = False) -> str:
    if force_fit:
        return "fit_addition"
    first_byte = hashlib.sha256(
        f"{SPLIT_SEED}:{group_key}".encode("utf-8")
    ).digest()[0]
    if first_byte <= 127:
        return "fit_addition"
    if first_byte <= 191:
        return "dev_new"
    return "holdout_sealed"


def assign_fresh_roles(
    qualified: pd.DataFrame,
    *,
    existing_sirens: set[str],
) -> pd.DataFrame:
    """Assign deterministic, SIREN-grouped roles to fresh qualifications."""
    output = qualified.copy()
    exact = output["label_kind"].eq(GroundTruthKind.MATCH_EXACT.value)
    current = output["ground_truth_siren"].fillna("").astype(str)
    historical = (
        output["historical_ground_truth_siren"].fillna("").astype(str)
    )
    output["fresh_group_key"] = [
        (
            f"SIREN:{current_siren}"
            if is_exact and current_siren
            else (
                f"SIREN:{historical_siren}"
                if historical_siren
                else f"QUERY:{query_id}"
            )
        )
        for is_exact, current_siren, historical_siren, query_id in zip(
            exact,
            current,
            historical,
            output["query_id"].astype(str),
            strict=True,
        )
    ]
    output["fresh_role"] = [
        role_for_group(
            group_key,
            force_fit=(
                group_key.startswith("SIREN:")
                and group_key.split(":", maxsplit=1)[1] in existing_sirens
            ),
        )
        for group_key in output["fresh_group_key"]
    ]
    role_counts = output.groupby("fresh_group_key")["fresh_role"].nunique()
    if role_counts.gt(1).any():
        raise ValueError("A fresh group was assigned to multiple roles")
    return output


def exact_siren_overlap_by_role(frame: pd.DataFrame) -> dict[str, Any]:
    exact = frame[frame["label_kind"].eq(GroundTruthKind.MATCH_EXACT.value)]
    sets = {
        role: set(
            exact.loc[
                exact["fresh_role"].eq(role),
                "ground_truth_siren",
            ]
            .dropna()
            .astype(str)
        )
        for role in ROLE_ORDER
    }
    overlaps = {
        "fit_dev": sorted(sets["fit_addition"] & sets["dev_new"]),
        "fit_holdout": sorted(
            sets["fit_addition"] & sets["holdout_sealed"]
        ),
        "dev_holdout": sorted(sets["dev_new"] & sets["holdout_sealed"]),
    }
    return {
        "exact_sirens_by_role": {
            role: len(values) for role, values in sets.items()
        },
        "overlap_counts": {
            name: len(values) for name, values in overlaps.items()
        },
        "overlaps": overlaps,
    }


def _read_service_ids(path: Path) -> set[str]:
    source = pd.read_csv(
        path,
        sep=";",
        dtype=str,
        encoding="utf-8-sig",
        usecols=["crm_id"],
    )
    return set(source["crm_id"].fillna("").astype(str).str.strip())


def _existing_v4(directory: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = json.loads((directory / "manifest.json").read_text())
    frames = [
        pd.read_parquet(directory / split / "benchmark.parquet")
        for split in ("train", "dev")
    ]
    return pd.concat(frames, ignore_index=True), manifest


def _existing_sirens(frame: pd.DataFrame) -> set[str]:
    historical = set(
        frame["historical_ground_truth_siren"].dropna().astype(str)
    )
    current = set(
        frame.loc[
            frame["label_kind"].eq(GroundTruthKind.MATCH_EXACT.value),
            "ground_truth_siren",
        ]
        .dropna()
        .astype(str)
    )
    return historical | current


def _role_summary(frame: pd.DataFrame) -> dict[str, Any]:
    output = {}
    for role in ROLE_ORDER:
        group = frame[frame["fresh_role"].eq(role)]
        output[role] = {
            "query_count": int(len(group)),
            "label_counts": {
                str(key): int(value)
                for key, value in group["label_kind"]
                .value_counts()
                .sort_index()
                .items()
            },
            "exact_sirens": int(
                group.loc[
                    group["label_kind"].eq(
                        GroundTruthKind.MATCH_EXACT.value
                    ),
                    "ground_truth_siren",
                ].nunique()
            ),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crm-source", type=Path, required=True)
    parser.add_argument("--benchmark-source", type=Path, required=True)
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument("--existing-v4-dir", type=Path, required=True)
    parser.add_argument("--partitions-dir", type=Path, required=True)
    parser.add_argument("--v4-policy-document", type=Path, required=True)
    parser.add_argument("--fresh-contract", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    historical_manifest = json.loads(args.benchmark_manifest.read_text())
    if file_sha256(args.benchmark_source) != historical_manifest.get(
        "source_sha256"
    ):
        raise ValueError("Benchmark source hash mismatch")
    existing_v4, v4_manifest = _existing_v4(args.existing_v4_dir)
    if v4_manifest.get("benchmark_build_id") != historical_manifest.get(
        "build_id"
    ):
        raise ValueError("Existing V4 and benchmark IDs differ")
    partition_fingerprint = directory_tree_sha256(args.partitions_dir)
    if partition_fingerprint["sha256"] != historical_manifest.get(
        "partitions_sha256"
    ):
        raise ValueError("Candidate partition tree hash mismatch")

    benchmark_ids = _read_service_ids(args.benchmark_source)
    raw = pd.read_csv(
        args.crm_source,
        sep=";",
        dtype=str,
        encoding="utf-8-sig",
    )
    fresh = build_fresh_benchmark(
        raw,
        benchmark_service_ids=benchmark_ids,
    )
    if set(fresh["crm_record_id"]) & benchmark_ids:
        raise ValueError("Fresh pool overlaps the frozen benchmark")

    identity = {
        "schema_version": SCHEMA_VERSION,
        "v4_policy_version": POLICY_VERSION,
        "split_seed": SPLIT_SEED,
        "crm_source_sha256": file_sha256(args.crm_source),
        "benchmark_source_sha256": file_sha256(args.benchmark_source),
        "benchmark_build_id": historical_manifest["build_id"],
        "existing_v4_build_id": v4_manifest["build_id"],
        "existing_v4_manifest_sha256": file_sha256(
            args.existing_v4_dir / "manifest.json"
        ),
        "partitions_sha256": partition_fingerprint["sha256"],
        "v4_policy_document_sha256": file_sha256(
            args.v4_policy_document
        ),
        "fresh_contract_sha256": file_sha256(args.fresh_contract),
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    output_dir = args.output_root / build_id
    if output_dir.exists():
        raise FileExistsError(f"Immutable output already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    store = PartitionedCandidateStore(args.partitions_dir)
    query_audit, evidence = audit_split(fresh, store)
    qualified = apply_current_snapshot_policy(
        fresh,
        query_audit,
        snapshot_id=historical_manifest["establishment_snapshot_sha256"],
    )
    qualified = assign_fresh_roles(
        qualified,
        existing_sirens=_existing_sirens(existing_v4),
    )
    overlap_audit = exact_siren_overlap_by_role(qualified)
    role_summary = _role_summary(qualified)
    existing_exact = int(
        existing_v4["label_kind"]
        .eq(GroundTruthKind.MATCH_EXACT.value)
        .sum()
    )
    fit_exact = role_summary["fit_addition"]["label_counts"].get(
        GroundTruthKind.MATCH_EXACT.value,
        0,
    )
    dev_exact = role_summary["dev_new"]["label_counts"].get(
        GroundTruthKind.MATCH_EXACT.value,
        0,
    )
    holdout_exact = role_summary["holdout_sealed"]["label_counts"].get(
        GroundTruthKind.MATCH_EXACT.value,
        0,
    )
    gate_checks = {
        "combined_fit_exact_at_least_5000": existing_exact + fit_exact >= 5_000,
        "dev_new_exact_at_least_300": dev_exact >= 300,
        "holdout_sealed_exact_at_least_300": holdout_exact >= 300,
        "zero_exact_siren_overlap": not any(
            overlap_audit["overlap_counts"].values()
        ),
        "zero_service_id_overlap": not bool(
            set(qualified["crm_record_id"]) & benchmark_ids
        ),
        "zero_closed_exact": bool(
            evidence["candidate_state"].eq("A").all()
        ),
    }
    gate = {"pass": all(gate_checks.values()), "checks": gate_checks}

    outputs: dict[str, str] = {}
    pool_dir = output_dir / "pool"
    pool_dir.mkdir()
    pool_paths = {
        "benchmark.parquet": pool_dir / "benchmark.parquet",
        "labels.parquet": pool_dir / "labels.parquet",
        "query_audit.parquet": pool_dir / "query_audit.parquet",
        "direct_evidence.parquet": pool_dir / "direct_evidence.parquet",
    }
    qualified.to_parquet(pool_paths["benchmark.parquet"], index=False)
    _label_view(qualified).assign(
        fresh_group_key=qualified["fresh_group_key"],
        fresh_role=qualified["fresh_role"],
    ).to_parquet(pool_paths["labels.parquet"], index=False)
    query_audit.to_parquet(pool_paths["query_audit.parquet"], index=False)
    evidence.to_parquet(pool_paths["direct_evidence.parquet"], index=False)
    for name, path in pool_paths.items():
        outputs[f"pool/{name}"] = file_sha256(path)

    for role in ROLE_ORDER:
        role_dir = output_dir / role
        role_dir.mkdir()
        role_frame = qualified[qualified["fresh_role"].eq(role)].copy()
        benchmark_path = role_dir / "benchmark.parquet"
        labels_path = role_dir / "labels.parquet"
        role_frame.to_parquet(benchmark_path, index=False)
        _label_view(role_frame).assign(
            fresh_group_key=role_frame["fresh_group_key"],
            fresh_role=role_frame["fresh_role"],
        ).to_parquet(labels_path, index=False)
        outputs[f"{role}/benchmark.parquet"] = file_sha256(benchmark_path)
        outputs[f"{role}/labels.parquet"] = file_sha256(labels_path)

    summary = {
        "fresh_source_rows": int(len(fresh)),
        "fresh_qualification": summarize_split(qualified),
        "existing_v4_exact": existing_exact,
        "combined_fit_exact": existing_exact + fit_exact,
        "roles": role_summary,
        "exact_siren_overlap": overlap_audit,
        "gate": gate,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    outputs["summary.json"] = file_sha256(summary_path)
    manifest = {
        **identity,
        "build_id": build_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "command": [sys.executable, *sys.argv],
        "status": "V4_FRESH_GATE_PASS" if gate["pass"] else "V4_FRESH_GATE_STOP",
        "source_test_untouched": True,
        "qualification_uses_retrieval_or_model_output": False,
        "holdout_model_evaluated": False,
        "partition_fingerprint": partition_fingerprint,
        "summary": summary,
        "outputs": outputs,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), **summary}, indent=2))


if __name__ == "__main__":
    main()
