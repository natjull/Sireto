#!/usr/bin/env python3
"""Build current-snapshot SIRET labels without using retrieval/model results."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_benchmark_v2_qualification import _validate_bound_file  # noqa: E402
from scripts.build_benchmark_v3_evidence import classify_direct_evidence  # noqa: E402
from scripts.freeze_v9_closed_benchmark import directory_tree_sha256  # noqa: E402
from scripts.run_v9_retrieval_experiment import _git_commit  # noqa: E402
from src.xgb_matcher.blocking import (  # noqa: E402
    address_hash,
    candidate_address_hash,
    normalize_code,
)
from src.xgb_matcher.contracts import GroundTruthKind  # noqa: E402
from src.xgb_matcher.features import (  # noqa: E402
    make_features_from_preprocessed,
    preprocess_crm_row,
)
from src.xgb_matcher.naming import build_candidate_names  # noqa: E402
from src.xgb_matcher.partitioned_store import PartitionedCandidateStore  # noqa: E402
from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402


SCHEMA_VERSION = "sireto-benchmark-v4-current-snapshot-1"
POLICY_VERSION = "active-direct-current-v4.0"
FORBIDDEN_MODEL_COLUMN = re.compile(
    r"(?:^|_)(?:rank|score|hit|confidence|decision)(?:_|$)",
    flags=re.IGNORECASE,
)
DIRECT_EVIDENCE_COLUMNS = [
    "query_id",
    "split",
    "partition_key",
    "candidate_siret",
    "candidate_siren",
    "candidate_state",
    "candidate_names_json",
    "candidate_address_hash",
    "exact_name_anchor",
    "exact_address_anchor",
    "direct_evidence_class",
    "name_jaro_max",
    "name_token_overlap_max",
    "name_norm_exact",
    "addr_jaro",
    "street_name_jaro",
    "street_number_match",
    "postcode_match",
]


def _normalise_identifier(value: Any, width: int) -> str | None:
    if value is None or pd.isna(value):
        return None
    digits = "".join(character for character in str(value) if character.isdigit())
    return digits.zfill(width) if digits else None


def reject_model_outputs(frame: pd.DataFrame, *, source: str) -> None:
    """Refuse any input exposing a retrieval or model outcome."""
    forbidden = [
        column
        for column in frame.columns
        if FORBIDDEN_MODEL_COLUMN.search(str(column))
    ]
    if forbidden:
        raise ValueError(
            f"{source} contains forbidden retrieval/model columns: "
            f"{sorted(forbidden)}"
        )


def canonical_address_anchor(value: str | None) -> str | None:
    """Neutralise punctuation corruption while retaining the street number."""
    if not value:
        return None
    canonical = re.sub(r"[^\w]+", " ", str(value), flags=re.UNICODE)
    canonical = " ".join(canonical.upper().split())
    return canonical or None


@dataclass
class ActivePartitionIndex:
    by_address: dict[str, list[dict[str, Any]]]
    by_name: dict[str, list[dict[str, Any]]]
    active_count: int
    physical_count: int


def build_active_partition_index(
    rows: Iterable[Mapping[str, Any]],
) -> ActivePartitionIndex:
    """Index all active partition rows by exact deterministic anchors."""
    by_address: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    active_by_siret: dict[str, dict[str, Any]] = {}
    physical_count = 0
    for raw in rows:
        physical_count += 1
        if str(raw.get("etat_admin") or "").strip().upper() != "A":
            continue
        siret = _normalise_identifier(raw.get("siret"), 14)
        siren = _normalise_identifier(raw.get("siren"), 9)
        if not siret or not siren:
            continue
        candidate = dict(raw)
        candidate["siret"] = siret
        candidate["siren"] = siren
        active_by_siret.setdefault(siret, candidate)

    for candidate in active_by_siret.values():
        candidate_hash = candidate_address_hash(candidate)
        candidate_anchor = canonical_address_anchor(candidate_hash)
        if candidate_anchor:
            by_address[candidate_anchor].append(candidate)
        names = {name.text for name in build_candidate_names(candidate) if name.text}
        for name in names:
            by_name[name].append(candidate)
    return ActivePartitionIndex(
        by_address=dict(by_address),
        by_name=dict(by_name),
        active_count=len(active_by_siret),
        physical_count=physical_count,
    )


def _preprocess_query(row: Mapping[str, Any]) -> dict[str, Any]:
    return preprocess_crm_row(
        {
            "crm_name": row.get("crm_name") or "",
            "crm_address": row.get("crm_address") or "",
            "crm_city": row.get("crm_city") or "",
            "crm_city_addr": row.get("crm_city") or "",
            "postcode": row.get("postcode") or "",
            "insee": row.get("insee") or "",
        }
    )


def find_direct_active_candidates(
    row: Mapping[str, Any],
    index: ActivePartitionIndex,
    *,
    partition_key: str,
) -> list[dict[str, Any]]:
    """Find unique active candidates anchored exactly on name or address."""
    crm = _preprocess_query(row)
    crm_address_hash = address_hash(
        crm.get("crm_street_num"),
        crm.get("crm_street_name"),
    )
    crm_address_anchor = canonical_address_anchor(crm_address_hash)
    crm_name = str(crm.get("crm_name") or "")
    candidates: dict[str, dict[str, Any]] = {}
    if crm_address_anchor:
        for candidate in index.by_address.get(crm_address_anchor, []):
            candidates[candidate["siret"]] = candidate
    if crm_name:
        for candidate in index.by_name.get(crm_name, []):
            candidates[candidate["siret"]] = candidate

    records: list[dict[str, Any]] = []
    for siret, candidate in sorted(candidates.items()):
        names = sorted(
            {name.text for name in build_candidate_names(candidate) if name.text}
        )
        candidate_hash = candidate_address_hash(candidate)
        candidate_anchor = canonical_address_anchor(candidate_hash)
        exact_name = bool(crm_name and crm_name in names)
        exact_address = bool(
            crm_address_anchor and crm_address_anchor == candidate_anchor
        )
        features = make_features_from_preprocessed(
            crm,
            candidate,
            skip_semantic=True,
        )
        evidence_class, _, _ = classify_direct_evidence(
            features,
            exact_address_hash=exact_address,
            crm_number_present=bool(crm.get("crm_street_num")),
            candidate_number_present=bool(candidate.get("numeroVoie")),
        )
        if evidence_class != "NAME_AND_ADDRESS":
            continue
        if not (exact_name or exact_address):
            continue
        records.append(
            {
                "query_id": str(row["query_id"]),
                "split": str(row["split"]),
                "partition_key": partition_key,
                "candidate_siret": siret,
                "candidate_siren": str(candidate["siren"]),
                "candidate_state": "A",
                "candidate_names_json": json.dumps(
                    names,
                    ensure_ascii=False,
                ),
                "candidate_address_hash": candidate_hash,
                "exact_name_anchor": exact_name,
                "exact_address_anchor": exact_address,
                "direct_evidence_class": evidence_class,
                "name_jaro_max": float(features["name_jaro_max"]),
                "name_token_overlap_max": float(
                    features["name_token_overlap_max"]
                ),
                "name_norm_exact": float(features["name_norm_exact"]),
                "addr_jaro": float(features["addr_jaro"]),
                "street_name_jaro": float(features["street_name_jaro"]),
                "street_number_match": float(
                    features["street_number_match"]
                ),
                "postcode_match": float(features["postcode_match"]),
            }
        )
    return records


def _planned_partition_key(
    row: Mapping[str, Any],
    store: PartitionedCandidateStore,
) -> str:
    insee = normalize_code(row.get("insee"))
    postcode = normalize_code(row.get("postcode"))
    if insee:
        insee_count = store._count_insee_rows(insee)
        if insee_count > 100_000 and postcode:
            return f"insee_cp:{insee}:{postcode}"
        if insee_count > 0:
            return f"insee:{insee}"
    if postcode:
        return f"cp:{postcode}"
    return "none"


def _load_partition(
    key: str,
    store: PartitionedCandidateStore,
) -> list[dict[str, Any]]:
    if key == "none":
        return []
    kind, payload = key.split(":", maxsplit=1)
    if kind == "insee":
        return store.load_by_insee(payload)
    if kind == "cp":
        return store.load_by_postcode(payload)
    if kind == "insee_cp":
        insee, postcode = payload.split(":", maxsplit=1)
        return store.load_by_postcode_filtered_insee(postcode, insee)
    raise ValueError(f"Unsupported partition key: {key}")


def audit_split(
    benchmark: pd.DataFrame,
    store: PartitionedCandidateStore,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit a complete split against full active geographic partitions."""
    reject_model_outputs(benchmark, source="V3 benchmark")
    work = benchmark.copy()
    work["query_id"] = work["query_id"].astype(str)
    if work["query_id"].duplicated().any():
        raise ValueError("V3 query_id values must be unique")
    work["partition_key"] = [
        _planned_partition_key(row, store)
        for row in work.to_dict("records")
    ]
    query_records: list[dict[str, Any]] = []
    evidence_records: list[dict[str, Any]] = []
    for partition_key, group in work.groupby("partition_key", sort=True):
        partition_rows = _load_partition(str(partition_key), store)
        index = build_active_partition_index(partition_rows)
        for row in group.sort_values(
            "query_id",
            key=lambda values: values.astype(str),
        ).to_dict("records"):
            evidence = find_direct_active_candidates(
                row,
                index,
                partition_key=str(partition_key),
            )
            evidence_records.extend(evidence)
            sirets = sorted(record["candidate_siret"] for record in evidence)
            selected = sirets[0] if len(sirets) == 1 else None
            selected_siren = (
                next(
                    record["candidate_siren"]
                    for record in evidence
                    if record["candidate_siret"] == selected
                )
                if selected
                else None
            )
            query_records.append(
                {
                    "query_id": str(row["query_id"]),
                    "split": str(row["split"]),
                    "partition_key": str(partition_key),
                    "partition_physical_count": index.physical_count,
                    "partition_active_count": index.active_count,
                    "direct_active_candidate_count": len(sirets),
                    "direct_active_sirets_json": json.dumps(sirets),
                    "selected_active_siret": selected,
                    "selected_active_siren": selected_siren,
                }
            )
    query_audit = pd.DataFrame(query_records).sort_values(
        "query_id",
        key=lambda values: values.astype(str),
    ).reset_index(drop=True)
    evidence = pd.DataFrame(
        evidence_records,
        columns=DIRECT_EVIDENCE_COLUMNS,
    ).sort_values(
        ["query_id", "candidate_siret"],
        key=lambda values: values.astype(str),
    ).reset_index(drop=True)
    return query_audit, evidence


def apply_current_snapshot_policy(
    v3_benchmark: pd.DataFrame,
    query_audit: pd.DataFrame,
    *,
    snapshot_id: str,
) -> pd.DataFrame:
    """Assign V4 labels from unique direct active candidates only."""
    required = {
        "query_id",
        "split",
        "label_kind",
        "ground_truth_siret",
        "ground_truth_siren",
        "historical_ground_truth_siret",
        "historical_ground_truth_siren",
        "qualification_reason",
    }
    audit_required = {
        "query_id",
        "split",
        "direct_active_candidate_count",
        "direct_active_sirets_json",
        "selected_active_siret",
        "selected_active_siren",
    }
    missing = required - set(v3_benchmark.columns)
    audit_missing = audit_required - set(query_audit.columns)
    if missing:
        raise ValueError(f"V3 benchmark is missing columns: {sorted(missing)}")
    if audit_missing:
        raise ValueError(f"V4 audit is missing columns: {sorted(audit_missing)}")
    reject_model_outputs(v3_benchmark, source="V3 benchmark")
    reject_model_outputs(query_audit, source="V4 audit")

    base = v3_benchmark.copy()
    audit = query_audit.copy()
    base["query_id"] = base["query_id"].astype(str)
    audit["query_id"] = audit["query_id"].astype(str)
    if base["query_id"].duplicated().any() or audit["query_id"].duplicated().any():
        raise ValueError("query_id must be unique")
    if set(base["query_id"]) != set(audit["query_id"]):
        raise ValueError("V3 benchmark and V4 audit query IDs differ")
    if set(base["split"].astype(str)) != set(audit["split"].astype(str)):
        raise ValueError("V3 benchmark and V4 audit splits differ")

    output = base.merge(
        audit.drop(columns=["split"]),
        on="query_id",
        validate="one_to_one",
    )
    output["v3_label_kind"] = output["label_kind"].astype(str)
    output["v3_ground_truth_siret"] = output["ground_truth_siret"]
    output["v3_ground_truth_siren"] = output["ground_truth_siren"]
    output["v3_qualification_reason"] = output["qualification_reason"].astype(
        str
    )
    counts = output["direct_active_candidate_count"].astype(int)
    unique = counts.eq(1)
    multiple = counts.gt(1)
    output["label_kind"] = GroundTruthKind.UNRESOLVED.value
    output.loc[unique, "label_kind"] = GroundTruthKind.MATCH_EXACT.value
    output.loc[multiple, "label_kind"] = GroundTruthKind.AMBIGUOUS.value
    output["ground_truth_siret"] = None
    output["ground_truth_siren"] = None
    output.loc[unique, "ground_truth_siret"] = output.loc[
        unique, "selected_active_siret"
    ]
    output.loc[unique, "ground_truth_siren"] = output.loc[
        unique, "selected_active_siren"
    ]
    output["qualification_reason"] = "NO_ACTIVE_DIRECT_MATCH"
    output.loc[unique, "qualification_reason"] = "UNIQUE_ACTIVE_DIRECT_MATCH"
    output.loc[multiple, "qualification_reason"] = (
        "MULTIPLE_ACTIVE_DIRECT_MATCHES"
    )
    output["exact_metric_eligible"] = unique
    output["current_snapshot_policy_version"] = POLICY_VERSION
    output["sirene_snapshot_id"] = snapshot_id
    output["label_reference"] = "CURRENT_SIRENE_SNAPSHOT"
    output["qualification_is_human_validated"] = False
    historical = output["historical_ground_truth_siret"].map(
        lambda value: _normalise_identifier(value, 14)
    )
    current = output["ground_truth_siret"].map(
        lambda value: _normalise_identifier(value, 14)
    )
    output["siret_changed_from_historical"] = (
        unique & current.notna() & historical.notna() & current.ne(historical)
    )
    output["siren_changed_from_historical"] = (
        output["siret_changed_from_historical"]
        & current.str[:9].ne(historical.str[:9])
    )
    if output.loc[unique, "ground_truth_siret"].isna().any():
        raise ValueError("Every V4 MATCH_EXACT requires a selected SIRET")
    if output.loc[~unique, "ground_truth_siret"].notna().any():
        raise ValueError("Only V4 MATCH_EXACT may carry a SIRET")
    return output.sort_values(
        "query_id",
        key=lambda values: values.astype(str),
    ).reset_index(drop=True)


def summarize_split(qualified: pd.DataFrame) -> dict[str, Any]:
    transitions = (
        qualified.groupby(["v3_label_kind", "label_kind"])
        .size()
        .reset_index(name="count")
    )
    exact = qualified["label_kind"].eq(GroundTruthKind.MATCH_EXACT.value)
    return {
        "query_count": int(len(qualified)),
        "label_counts": {
            str(key): int(value)
            for key, value in qualified["label_kind"]
            .value_counts()
            .sort_index()
            .items()
        },
        "coverage": float(exact.mean()) if len(qualified) else 0.0,
        "transitions": transitions.to_dict("records"),
        "siret_changed_from_historical": int(
            qualified["siret_changed_from_historical"].sum()
        ),
        "siren_changed_from_historical": int(
            qualified["siren_changed_from_historical"].sum()
        ),
        "promoted_from_v3_open": int(
            (
                ~qualified["v3_label_kind"].eq(
                    GroundTruthKind.MATCH_EXACT.value
                )
                & exact
            ).sum()
        ),
        "v3_exact_no_longer_exact": int(
            (
                qualified["v3_label_kind"].eq(
                    GroundTruthKind.MATCH_EXACT.value
                )
                & ~exact
            ).sum()
        ),
        "closed_exact_labels": 0,
    }


def cross_split_audit(
    train: pd.DataFrame,
    dev: pd.DataFrame,
) -> dict[str, Any]:
    train_exact = train[
        train["label_kind"].eq(GroundTruthKind.MATCH_EXACT.value)
    ]
    dev_exact = dev[
        dev["label_kind"].eq(GroundTruthKind.MATCH_EXACT.value)
    ]
    train_sirens = set(train_exact["ground_truth_siren"].dropna().astype(str))
    dev_sirens = set(dev_exact["ground_truth_siren"].dropna().astype(str))
    overlap = sorted(train_sirens & dev_sirens)
    return {
        "train_exact_sirens": len(train_sirens),
        "dev_exact_sirens": len(dev_sirens),
        "shared_exact_siren_count": len(overlap),
        "shared_exact_sirens": overlap,
    }


def _load_v3(directory: Path, split: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("split") != split:
        raise ValueError(f"Expected V3 split {split}, got {manifest.get('split')}")
    benchmark_path = directory / "benchmark.parquet"
    _validate_bound_file(
        benchmark_path,
        manifest,
        manifest_section="outputs",
    )
    benchmark = pd.read_parquet(benchmark_path)
    if set(benchmark["split"].astype(str)) != {split}:
        raise ValueError(f"V3 {split} benchmark contains another split")
    return benchmark, manifest


def _label_view(qualified: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "query_id",
        "split",
        "label_kind",
        "ground_truth_siret",
        "ground_truth_siren",
        "historical_ground_truth_siret",
        "historical_ground_truth_siren",
        "v3_label_kind",
        "v3_ground_truth_siret",
        "v3_ground_truth_siren",
        "direct_active_candidate_count",
        "direct_active_sirets_json",
        "qualification_reason",
        "exact_metric_eligible",
        "siret_changed_from_historical",
        "siren_changed_from_historical",
        "current_snapshot_policy_version",
        "sirene_snapshot_id",
        "label_reference",
        "qualification_is_human_validated",
    ]
    return qualified[columns]


def _report(
    *,
    build_id: str,
    summaries: dict[str, Any],
    split_audit: dict[str, Any],
    gate: dict[str, Any],
) -> str:
    lines = [
        "# Qualification V4 — SIRET actif au snapshot",
        "",
        f"- Build : `{build_id}`",
        "- Test lu : non",
        f"- Gate de viabilité : **{'PASS' if gate['pass'] else 'STOP'}**",
        "",
    ]
    for split in ("train", "dev"):
        summary = summaries[split]
        lines.extend(
            [
                f"## {split}",
                "",
                f"- requêtes : {summary['query_count']} ;",
                f"- `MATCH_EXACT` : "
                f"{summary['label_counts'].get('MATCH_EXACT', 0)} "
                f"({summary['coverage']:.3%}) ;",
                f"- `AMBIGUOUS` : "
                f"{summary['label_counts'].get('AMBIGUOUS', 0)} ;",
                f"- `UNRESOLVED` : "
                f"{summary['label_counts'].get('UNRESOLVED', 0)} ;",
                f"- SIRET changé : "
                f"{summary['siret_changed_from_historical']} ;",
                f"- SIREN changé : "
                f"{summary['siren_changed_from_historical']} ;",
                f"- anciens labels ouverts devenus exacts : "
                f"{summary['promoted_from_v3_open']} ;",
                "",
            ]
        )
    lines.extend(
        [
            "## Séparation",
            "",
            f"- SIREN exacts partagés train/dev : "
            f"{split_audit['shared_exact_siren_count']}.",
            "",
            "Cette qualification est mécanique et actuelle. Elle n'est pas "
            "une certification humaine et ne réouvre pas l'ancien test.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-v3-dir", type=Path, required=True)
    parser.add_argument("--dev-v3-dir", type=Path, required=True)
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument("--partitions-dir", type=Path, required=True)
    parser.add_argument("--policy-document", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    historical_manifest = json.loads(
        args.benchmark_manifest.read_text(encoding="utf-8")
    )
    train_v3, train_manifest = _load_v3(args.train_v3_dir, "train")
    dev_v3, dev_manifest = _load_v3(args.dev_v3_dir, "dev")
    if train_manifest.get("benchmark_build_id") != historical_manifest.get(
        "build_id"
    ):
        raise ValueError("Train V3 and historical benchmark IDs differ")
    if dev_manifest.get("benchmark_build_id") != historical_manifest.get(
        "build_id"
    ):
        raise ValueError("Dev V3 and historical benchmark IDs differ")
    snapshot_id = str(historical_manifest["establishment_snapshot_sha256"])
    for manifest in (train_manifest, dev_manifest):
        if manifest.get("establishment_snapshot_sha256") != snapshot_id:
            raise ValueError("V3 establishment snapshot hash mismatch")

    partition_fingerprint = directory_tree_sha256(args.partitions_dir)
    if partition_fingerprint["sha256"] != historical_manifest.get(
        "partitions_sha256"
    ):
        raise ValueError("Candidate partition tree hash mismatch")
    identity = {
        "schema_version": SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "policy_document_sha256": file_sha256(args.policy_document),
        "benchmark_build_id": historical_manifest["build_id"],
        "establishment_snapshot_sha256": snapshot_id,
        "partitions_sha256": partition_fingerprint["sha256"],
        "train_v3_benchmark_sha256": file_sha256(
            args.train_v3_dir / "benchmark.parquet"
        ),
        "dev_v3_benchmark_sha256": file_sha256(
            args.dev_v3_dir / "benchmark.parquet"
        ),
    }
    build_id = hashlib.sha256(
        json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]
    output_dir = args.output_root / build_id
    if output_dir.exists():
        raise FileExistsError(f"Immutable output already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    store = PartitionedCandidateStore(args.partitions_dir)
    outputs: dict[str, str] = {}
    qualified_by_split: dict[str, pd.DataFrame] = {}
    summaries: dict[str, Any] = {}
    for split, v3 in (("train", train_v3), ("dev", dev_v3)):
        split_dir = output_dir / split
        split_dir.mkdir()
        query_audit, evidence = audit_split(v3, store)
        qualified = apply_current_snapshot_policy(
            v3,
            query_audit,
            snapshot_id=snapshot_id,
        )
        qualified_by_split[split] = qualified
        summaries[split] = summarize_split(qualified)
        paths = {
            "benchmark.parquet": split_dir / "benchmark.parquet",
            "labels.parquet": split_dir / "labels.parquet",
            "query_audit.parquet": split_dir / "query_audit.parquet",
            "direct_evidence.parquet": split_dir / "direct_evidence.parquet",
            "summary.json": split_dir / "summary.json",
        }
        qualified.to_parquet(paths["benchmark.parquet"], index=False)
        _label_view(qualified).to_parquet(
            paths["labels.parquet"],
            index=False,
        )
        query_audit.to_parquet(paths["query_audit.parquet"], index=False)
        evidence.to_parquet(paths["direct_evidence.parquet"], index=False)
        paths["summary.json"].write_text(
            json.dumps(summaries[split], indent=2, sort_keys=True),
            encoding="utf-8",
        )
        for name, path in paths.items():
            outputs[f"{split}/{name}"] = file_sha256(path)

    split_integrity = cross_split_audit(
        qualified_by_split["train"],
        qualified_by_split["dev"],
    )
    train_summary = summaries["train"]
    dev_summary = summaries["dev"]
    gate_checks = {
        "train_coverage_at_least_50pct": train_summary["coverage"] >= 0.50,
        "dev_coverage_at_least_50pct": dev_summary["coverage"] >= 0.50,
        "train_exact_at_least_5000": (
            train_summary["label_counts"].get("MATCH_EXACT", 0) >= 5_000
        ),
        "no_shared_exact_siren": (
            split_integrity["shared_exact_siren_count"] == 0
        ),
        "no_closed_exact": (
            train_summary["closed_exact_labels"] == 0
            and dev_summary["closed_exact_labels"] == 0
        ),
    }
    gate = {"pass": all(gate_checks.values()), "checks": gate_checks}
    audit_path = output_dir / "cross_split_audit.json"
    audit_path.write_text(
        json.dumps(split_integrity, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    gate_path = output_dir / "gate.json"
    gate_path.write_text(
        json.dumps(gate, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report_path = output_dir / "report.md"
    report_path.write_text(
        _report(
            build_id=build_id,
            summaries=summaries,
            split_audit=split_integrity,
            gate=gate,
        ),
        encoding="utf-8",
    )
    outputs["cross_split_audit.json"] = file_sha256(audit_path)
    outputs["gate.json"] = file_sha256(gate_path)
    outputs["report.md"] = file_sha256(report_path)

    manifest = {
        **identity,
        "build_id": build_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "command": [sys.executable, *sys.argv],
        "status": (
            "V4_CURRENT_SNAPSHOT_GATE_PASS"
            if gate["pass"]
            else "V4_CURRENT_SNAPSHOT_GATE_STOP"
        ),
        "source_test_untouched": True,
        "human_validated": False,
        "qualification_uses_retrieval_or_model_output": False,
        "partition_fingerprint": partition_fingerprint,
        "summaries": summaries,
        "cross_split_audit": split_integrity,
        "gate": gate,
        "inputs": {
            "benchmark_manifest_sha256": file_sha256(
                args.benchmark_manifest
            ),
            "train_v3_manifest_sha256": file_sha256(
                args.train_v3_dir / "manifest.json"
            ),
            "dev_v3_manifest_sha256": file_sha256(
                args.dev_v3_dir / "manifest.json"
            ),
        },
        "outputs": outputs,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "summaries": summaries,
                "cross_split_audit": split_integrity,
                "gate": gate,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
