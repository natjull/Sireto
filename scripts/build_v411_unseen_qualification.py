#!/usr/bin/env python3
"""Qualify sanitized V4.11 unseen CRM rows with the frozen mechanical V4 policy.

The only case input is a sanitized artifact containing seven CRM columns.
This builder never opens the source registry and never loads a retrieval,
ranker, acceptor, score, prediction or historical identifier.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import tempfile
from typing import Any, Mapping

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_benchmark_v4_current_snapshot import (  # noqa: E402
    POLICY_VERSION,
    ActivePartitionIndex,
    _load_partition,
    _planned_partition_key,
    build_active_partition_index,
    find_direct_active_candidates,
)
from src.xgb_matcher.partitioned_store import PartitionedCandidateStore  # noqa: E402
from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402


SCHEMA_VERSION = "sireto-v4.11-unseen-mechanical-qualification-1"
SANITIZED_SCHEMA_VERSION = "sireto-v4.11-descriptive-unseen-sanitized-1"
SANITIZED_EXPERIMENT_ID = "V411_DESCRIPTIVE_UNSEEN_SANITIZED"
SANITIZED_QUERIES_FILENAME = "queries_sanitized.parquet"
EXPERIMENT_ID = "V411_DESCRIPTIVE_UNSEEN_225_QUALIFICATION"
EXPECTED_QUERY_COUNT = 225
EXPECTED_SNAPSHOT_SHA256 = (
    "c91180cc5bae86948dd57d752c9bae45e58cc64653e99d5a9357664b67300845"
)
EXPECTED_PARTITIONS_SIGNATURE = (
    "2f6668f60da8bc9fe52b683b32ef35641803679c01f8c8fd124e2e86a41e2b82"
)
EXPECTED_CONTRACT_SHA256 = (
    "28785be1c776f27b9dc9357fe543049bb70d6937b6b03d6f59c33eee67f43026"
)
DEFAULT_CONTRACT = (
    Path(__file__).resolve().parent.parent
    / "docs"
    / "v4_11_descriptive_unseen_225_contract.md"
)
QUERY_COLUMNS = [
    "query_id",
    "crm_record_id",
    "crm_name",
    "crm_address",
    "crm_postcode",
    "crm_city",
    "crm_insee",
]
FORBIDDEN_TOKENS = (
    "siret",
    "siren",
    "truth",
    "label",
    "candidate",
    "rank",
    "score",
    "prediction",
    "service_id",
    "fingerprint",
)
EVIDENCE_COLUMNS = [
    "evidence_ref",
    "query_id",
    "snapshot_sha256",
    "policy_version",
    "partition_kind",
    "partition_key",
    "active_universe_count",
    "candidate_siret",
    "candidate_siren",
    "candidate_state",
    "candidate_names",
    "candidate_address",
    "name_evidence_class",
    "address_evidence_class",
    "direct_match_rule",
]
LABEL_COLUMNS = [
    "query_id",
    "label_kind",
    "ground_truth_siret",
    "ground_truth_siren",
    "direct_active_candidate_count",
    "evidence_refs_json",
    "qualification_reason",
    "snapshot_sha256",
    "policy_version",
    "validator",
    "human_validated",
]
LABEL_KINDS = {"MATCH_EXACT", "AMBIGUOUS", "UNRESOLVED"}
VALIDATOR = "AUTONOMOUS_MECHANICAL_V4"
SIRET_PATTERN = re.compile(r"^\d{14}$")
SIREN_PATTERN = re.compile(r"^\d{9}$")
PINNED_POLICY_SOURCES = {
    "scripts/build_benchmark_v4_current_snapshot.py": (
        "b0451766575f0023d42d598caa23aebb0e81cff5fcb60f5071da64c9b3f0b19b"
    ),
    "scripts/build_benchmark_v3_evidence.py": (
        "9ebf636101de6cd73e4079fbcc14b012e655fdd6ff08910e00127ee915718dcc"
    ),
    "src/xgb_matcher/blocking.py": (
        "e6a0fded2f6496c9f4e901d8ba4fca1b912f5410c3c506a170c434ec02a55736"
    ),
    "src/xgb_matcher/features.py": (
        "839f55b0d8c56e22e75758db88647c910fd8158039d1b0175f9c818e5ac0b191"
    ),
    "src/xgb_matcher/naming.py": (
        "b7ef59a8cb7529179567f6e3ffe3b64757383a9e449a0110886abe640a1b5fc1"
    ),
    "src/xgb_matcher/partitioned_store.py": (
        "181d1c8a56539f6b36e01d9fc040a7fb4135e28a0b10147775abd5b33837a39f"
    ),
}


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _output_hash(manifest: Mapping[str, Any], filename: str) -> str:
    record = (manifest.get("outputs") or {}).get(filename)
    if isinstance(record, Mapping):
        return str(record.get("sha256") or "")
    return str(record or "")


def _external_output(path: Path) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(Path("/Volumes/CATNAT_DATA")):
        raise ValueError("output_root must be under /Volumes/CATNAT_DATA")
    return resolved


def _path_signature(path: Path) -> str:
    path = Path(path)
    if path.is_file():
        return file_sha256(path)
    root_manifest = path / "manifest.json"
    if root_manifest.exists():
        return file_sha256(root_manifest)
    digest = hashlib.sha256()
    for candidate in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(candidate.relative_to(path)).encode())
        digest.update(file_sha256(candidate).encode())
    return digest.hexdigest()


def _validate_policy_sources(repo_root: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in PINNED_POLICY_SOURCES.items():
        path = repo_root / relative
        current = file_sha256(path)
        if current != expected:
            raise ValueError(f"STOP_POLICY_INTEGRITY: source changed: {relative}")
        observed[relative] = current
    return observed


def validate_query_schema(
    queries: pd.DataFrame,
    *,
    expected_count: int = EXPECTED_QUERY_COUNT,
) -> pd.DataFrame:
    if list(queries.columns) != QUERY_COLUMNS:
        raise ValueError("STOP_INPUT_INTEGRITY: sanitized query schema changed")
    forbidden = [
        column
        for column in queries.columns
        if any(token in column.lower() for token in FORBIDDEN_TOKENS)
    ]
    if forbidden:
        raise ValueError("STOP_INPUT_INTEGRITY: forbidden query columns")
    output = queries.copy()
    if len(output) != expected_count:
        raise ValueError(
            f"STOP_INPUT_INTEGRITY: expected {expected_count} sanitized queries"
        )
    for column in QUERY_COLUMNS:
        output[column] = output[column].fillna("").astype(str).str.strip()
        if output[column].eq("").any():
            raise ValueError(
                f"STOP_INPUT_INTEGRITY: empty sanitized field {column}"
            )
    if output["query_id"].duplicated().any():
        raise ValueError("STOP_INPUT_INTEGRITY: duplicate query_id")
    if output["crm_record_id"].duplicated().any():
        raise ValueError("STOP_INPUT_INTEGRITY: duplicate crm_record_id")
    return output


def load_sanitized_artifact(
    artifact_dir: Path,
    *,
    expected_count: int = EXPECTED_QUERY_COUNT,
) -> tuple[dict[str, Any], pd.DataFrame, str]:
    """Open exactly the sanitized manifest and its seven-column query file."""

    root = Path(artifact_dir).resolve()
    manifest_path = root / "manifest.json"
    queries_path = root / SANITIZED_QUERIES_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SANITIZED_SCHEMA_VERSION:
        raise ValueError("STOP_INPUT_INTEGRITY: unsupported sanitized artifact")
    if manifest.get("experiment_id") != SANITIZED_EXPERIMENT_ID:
        raise ValueError("STOP_INPUT_INTEGRITY: wrong sanitized experiment")
    identity = manifest.get("build_identity") or {}
    expected_build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    if (
        manifest.get("build_id") != expected_build_id
        or root.name != expected_build_id
    ):
        raise ValueError("STOP_INPUT_INTEGRITY: sanitized build identity changed")
    invariants = manifest.get("invariants") or {}
    if (
        invariants.get("forbidden_source_columns_loaded") is not False
        or invariants.get("labels_loaded") is not False
        or invariants.get("retrieval_or_model_loaded") is not False
        or invariants.get("input_siret_or_siren_exposed") is not False
    ):
        raise ValueError("STOP_INPUT_INTEGRITY: sanitized invariants changed")
    output_record = (manifest.get("outputs") or {}).get(
        SANITIZED_QUERIES_FILENAME
    )
    if not isinstance(output_record, Mapping):
        raise ValueError("STOP_INPUT_INTEGRITY: missing sanitized query record")
    if output_record.get("columns") != QUERY_COLUMNS:
        raise ValueError("STOP_INPUT_INTEGRITY: declared query schema changed")
    if int(output_record.get("row_count", -1)) != expected_count:
        raise ValueError("STOP_INPUT_INTEGRITY: declared query count changed")
    expected_hash = _output_hash(manifest, SANITIZED_QUERIES_FILENAME)
    observed_hash = file_sha256(queries_path)
    if not expected_hash or observed_hash != expected_hash:
        raise ValueError("STOP_INPUT_INTEGRITY: sanitized query hash mismatch")
    queries = validate_query_schema(
        pd.read_parquet(queries_path, columns=QUERY_COLUMNS),
        expected_count=expected_count,
    )
    return manifest, queries, observed_hash


def validate_runtime_inputs(
    *,
    partitions_dir: Path,
    snapshot_path: Path,
) -> dict[str, str]:
    partition_signature = _path_signature(partitions_dir)
    snapshot_sha256 = file_sha256(snapshot_path)
    if partition_signature != EXPECTED_PARTITIONS_SIGNATURE:
        raise ValueError("STOP_INPUT_INTEGRITY: partition signature changed")
    if snapshot_sha256 != EXPECTED_SNAPSHOT_SHA256:
        raise ValueError("STOP_INPUT_INTEGRITY: snapshot hash changed")
    return {
        "partitions_signature": partition_signature,
        "snapshot_sha256": snapshot_sha256,
    }


def _validate_sanitized_link(
    identity: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> None:
    manifest_record = inputs.get("sanitized_artifact_manifest") or {}
    queries_record = inputs.get("sanitized_queries") or {}
    manifest_path = Path(str(manifest_record.get("path") or ""))
    queries_path = Path(str(queries_record.get("path") or ""))
    manifest_sha = file_sha256(manifest_path)
    queries_sha = file_sha256(queries_path)
    sanitized = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared_queries = (sanitized.get("outputs") or {}).get(
        SANITIZED_QUERIES_FILENAME
    ) or {}
    if (
        manifest_sha != manifest_record.get("sha256")
        or manifest_sha != identity.get("sanitized_manifest_sha256")
        or queries_sha != queries_record.get("sha256")
        or queries_sha != identity.get("sanitized_queries_sha256")
        or queries_sha != declared_queries.get("sha256")
        or identity.get("sanitized_build_id") != sanitized.get("build_id")
    ):
        raise ValueError("Sanitized qualification link drift")


def _partition_kind(partition_key: str) -> str:
    if partition_key == "none":
        return "NONE"
    return partition_key.split(":", 1)[0].upper()


def _direct_rule(row: Mapping[str, Any]) -> str:
    exact_name = bool(row["exact_name_anchor"])
    exact_address = bool(row["exact_address_anchor"])
    if exact_name and exact_address:
        return "EXACT_NAME_AND_ADDRESS"
    if exact_name:
        return "EXACT_NAME_STRONG_ADDRESS"
    if exact_address:
        return "EXACT_ADDRESS_STRONG_NAME"
    raise ValueError("STOP_POLICY_INTEGRITY: direct match lacks an exact anchor")


def _evidence_ref(query_id: str, candidate_siret: str) -> str:
    digest = hashlib.sha256(
        f"v411-unseen-evidence:{query_id}:{candidate_siret}".encode("utf-8")
    ).hexdigest()[:20]
    return f"SIRENE_SNAPSHOT:{digest}"


def qualify_queries(
    queries: pd.DataFrame,
    *,
    partitions_dir: Path,
    snapshot_sha256: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the frozen V4 policy over complete geographic universes."""

    store = PartitionedCandidateStore(partitions_dir)
    work = queries.copy()
    policy_rows = [
        {
            "query_id": row["query_id"],
            "split": "descriptive_unseen",
            "crm_name": row["crm_name"],
            "crm_address": row["crm_address"],
            "crm_city": row["crm_city"],
            "postcode": row["crm_postcode"],
            "insee": row["crm_insee"],
        }
        for row in work.to_dict("records")
    ]
    for row in policy_rows:
        row["partition_key"] = _planned_partition_key(row, store)

    evidence_records: list[dict[str, Any]] = []
    label_records: list[dict[str, Any]] = []
    policy_frame = pd.DataFrame(policy_rows)
    for partition_key, group in policy_frame.groupby("partition_key", sort=True):
        partition_rows = _load_partition(str(partition_key), store)
        index: ActivePartitionIndex = build_active_partition_index(partition_rows)
        for query in group.sort_values("query_id", kind="mergesort").to_dict(
            "records"
        ):
            direct = find_direct_active_candidates(
                query,
                index,
                partition_key=str(partition_key),
            )
            refs: list[str] = []
            for raw in direct:
                evidence_ref = _evidence_ref(
                    str(query["query_id"]), str(raw["candidate_siret"])
                )
                refs.append(evidence_ref)
                evidence_records.append(
                    {
                        "evidence_ref": evidence_ref,
                        "query_id": str(query["query_id"]),
                        "snapshot_sha256": snapshot_sha256,
                        "policy_version": POLICY_VERSION,
                        "partition_kind": _partition_kind(str(partition_key)),
                        "partition_key": str(partition_key),
                        "active_universe_count": int(index.active_count),
                        "candidate_siret": str(raw["candidate_siret"]),
                        "candidate_siren": str(raw["candidate_siren"]),
                        "candidate_state": str(raw["candidate_state"]),
                        "candidate_names": str(raw["candidate_names_json"]),
                        "candidate_address": str(
                            raw["candidate_address_hash"] or ""
                        ),
                        "name_evidence_class": "STRONG",
                        "address_evidence_class": "STRONG",
                        "direct_match_rule": _direct_rule(raw),
                    }
                )
            count = len(direct)
            if count == 1:
                label_kind = "MATCH_EXACT"
                truth_siret = str(direct[0]["candidate_siret"])
                truth_siren = str(direct[0]["candidate_siren"])
                reason = "UNIQUE_ACTIVE_DIRECT_MATCH"
            elif count > 1:
                label_kind = "AMBIGUOUS"
                truth_siret = None
                truth_siren = None
                reason = "MULTIPLE_ACTIVE_DIRECT_MATCHES"
            else:
                label_kind = "UNRESOLVED"
                truth_siret = None
                truth_siren = None
                reason = "NO_ACTIVE_DIRECT_MATCH"
            label_records.append(
                {
                    "query_id": str(query["query_id"]),
                    "label_kind": label_kind,
                    "ground_truth_siret": truth_siret,
                    "ground_truth_siren": truth_siren,
                    "direct_active_candidate_count": count,
                    "evidence_refs_json": json.dumps(
                        sorted(refs), ensure_ascii=False, separators=(",", ":")
                    ),
                    "qualification_reason": reason,
                    "snapshot_sha256": snapshot_sha256,
                    "policy_version": POLICY_VERSION,
                    "validator": VALIDATOR,
                    "human_validated": False,
                }
            )
    evidence = pd.DataFrame(evidence_records, columns=EVIDENCE_COLUMNS).sort_values(
        ["query_id", "candidate_siret"], kind="mergesort"
    ).reset_index(drop=True)
    labels = pd.DataFrame(label_records, columns=LABEL_COLUMNS).sort_values(
        "query_id", kind="mergesort"
    ).reset_index(drop=True)
    validate_qualification(queries, evidence, labels)
    return evidence, labels


def validate_qualification(
    queries: pd.DataFrame,
    evidence: pd.DataFrame,
    labels: pd.DataFrame,
) -> None:
    if list(evidence.columns) != EVIDENCE_COLUMNS:
        raise ValueError("STOP_OUTPUT_INTEGRITY: evidence schema changed")
    if list(labels.columns) != LABEL_COLUMNS:
        raise ValueError("STOP_OUTPUT_INTEGRITY: label schema changed")
    if len(labels) != len(queries) or set(labels["query_id"]) != set(
        queries["query_id"]
    ):
        raise ValueError("STOP_OUTPUT_INTEGRITY: labels do not cover queries")
    if labels["query_id"].duplicated().any():
        raise ValueError("STOP_OUTPUT_INTEGRITY: duplicate labels")
    query_ids = set(queries["query_id"].astype(str))
    if not set(evidence["query_id"].astype(str)).issubset(query_ids):
        raise ValueError("STOP_OUTPUT_INTEGRITY: evidence for unknown query")
    if evidence["evidence_ref"].astype(str).duplicated().any():
        raise ValueError("STOP_OUTPUT_INTEGRITY: duplicate evidence reference")
    if not set(labels["label_kind"]).issubset(LABEL_KINDS):
        raise ValueError("STOP_OUTPUT_INTEGRITY: forbidden label")
    if labels["label_kind"].eq("NO_MATCH").any():
        raise ValueError("STOP_OUTPUT_INTEGRITY: NO_MATCH is forbidden")
    exact = labels["label_kind"].eq("MATCH_EXACT")
    if labels.loc[exact, "ground_truth_siret"].isna().any():
        raise ValueError("STOP_OUTPUT_INTEGRITY: exact label lacks truth")
    if (
        labels.loc[~exact, "ground_truth_siret"].notna().any()
        or labels.loc[~exact, "ground_truth_siren"].notna().any()
    ):
        raise ValueError("STOP_OUTPUT_INTEGRITY: non-exact label carries truth")
    exact_siret = labels.loc[exact, "ground_truth_siret"].astype(str)
    exact_siren = labels.loc[exact, "ground_truth_siren"].astype(str)
    if (
        not exact_siret.map(lambda value: bool(SIRET_PATTERN.fullmatch(value))).all()
        or not exact_siren.map(
            lambda value: bool(SIREN_PATTERN.fullmatch(value))
        ).all()
        or not exact_siren.equals(exact_siret.str[:9])
    ):
        raise ValueError("STOP_OUTPUT_INTEGRITY: exact identifiers inconsistent")
    if labels.loc[exact, "direct_active_candidate_count"].ne(1).any():
        raise ValueError("STOP_OUTPUT_INTEGRITY: exact cardinality changed")
    if labels.loc[
        labels["label_kind"].eq("AMBIGUOUS"), "direct_active_candidate_count"
    ].le(1).any():
        raise ValueError("STOP_OUTPUT_INTEGRITY: ambiguous cardinality changed")
    if labels.loc[
        labels["label_kind"].eq("UNRESOLVED"), "direct_active_candidate_count"
    ].ne(0).any():
        raise ValueError("STOP_OUTPUT_INTEGRITY: unresolved cardinality changed")
    if labels["validator"].ne(VALIDATOR).any() or labels[
        "human_validated"
    ].astype(bool).any():
        raise ValueError("STOP_OUTPUT_INTEGRITY: validator policy changed")
    if (
        labels["snapshot_sha256"].ne(EXPECTED_SNAPSHOT_SHA256).any()
        or labels["policy_version"].ne(POLICY_VERSION).any()
        or evidence["snapshot_sha256"].ne(EXPECTED_SNAPSHOT_SHA256).any()
        or evidence["policy_version"].ne(POLICY_VERSION).any()
    ):
        raise ValueError("STOP_OUTPUT_INTEGRITY: frozen policy metadata changed")
    expected_reasons = labels["label_kind"].map(
        {
            "MATCH_EXACT": "UNIQUE_ACTIVE_DIRECT_MATCH",
            "AMBIGUOUS": "MULTIPLE_ACTIVE_DIRECT_MATCHES",
            "UNRESOLVED": "NO_ACTIVE_DIRECT_MATCH",
        }
    )
    if not labels["qualification_reason"].astype(str).equals(expected_reasons):
        raise ValueError("STOP_OUTPUT_INTEGRITY: qualification reason changed")
    if evidence["candidate_state"].ne("A").any():
        raise ValueError("STOP_OUTPUT_INTEGRITY: non-active evidence")
    if evidence.duplicated(["query_id", "candidate_siret"]).any():
        raise ValueError("STOP_OUTPUT_INTEGRITY: duplicate evidence candidate")
    if not evidence["candidate_siret"].astype(str).map(
        lambda value: bool(SIRET_PATTERN.fullmatch(value))
    ).all() or not evidence["candidate_siren"].astype(str).map(
        lambda value: bool(SIREN_PATTERN.fullmatch(value))
    ).all():
        raise ValueError("STOP_OUTPUT_INTEGRITY: evidence identifier invalid")
    if not evidence["candidate_siren"].astype(str).equals(
        evidence["candidate_siret"].astype(str).str[:9]
    ):
        raise ValueError("STOP_OUTPUT_INTEGRITY: evidence SIREN inconsistent")
    expected_refs = [
        _evidence_ref(str(row.query_id), str(row.candidate_siret))
        for row in evidence.itertuples(index=False)
    ]
    if evidence["evidence_ref"].astype(str).tolist() != expected_refs:
        raise ValueError("STOP_OUTPUT_INTEGRITY: evidence reference changed")
    evidence_counts = evidence.groupby("query_id").size()
    expected_counts = labels.set_index("query_id")[
        "direct_active_candidate_count"
    ].astype(int)
    observed_counts = evidence_counts.reindex(expected_counts.index, fill_value=0)
    if not observed_counts.equals(expected_counts):
        raise ValueError("STOP_OUTPUT_INTEGRITY: evidence cardinality differs")
    exact_source = labels.loc[
        exact, ["query_id", "ground_truth_siret", "ground_truth_siren"]
    ]
    exact_query_ids = set(exact_source["query_id"].astype(str))
    exact_labels = exact_source.merge(
        evidence.loc[
            evidence["query_id"].astype(str).isin(exact_query_ids),
            ["query_id", "candidate_siret", "candidate_siren"],
        ],
        on="query_id",
        how="left",
        validate="one_to_one",
    )
    if (
        exact_labels["candidate_siret"].isna().any()
        or not exact_labels["ground_truth_siret"].astype(str).equals(
            exact_labels["candidate_siret"].astype(str)
        )
        or not exact_labels["ground_truth_siren"].astype(str).equals(
            exact_labels["candidate_siren"].astype(str)
        )
    ):
        raise ValueError("STOP_OUTPUT_INTEGRITY: exact truth differs from evidence")
    refs_by_query = {
        str(query_id): sorted(group["evidence_ref"].astype(str))
        for query_id, group in evidence.groupby("query_id", sort=False)
    }
    for row in labels.itertuples(index=False):
        try:
            declared = json.loads(str(row.evidence_refs_json))
        except json.JSONDecodeError as error:
            raise ValueError(
                "STOP_OUTPUT_INTEGRITY: invalid evidence refs JSON"
            ) from error
        expected = refs_by_query.get(str(row.query_id), [])
        if not isinstance(declared, list) or declared != expected:
            raise ValueError("STOP_OUTPUT_INTEGRITY: evidence refs mismatch")


def build_artifact(
    *,
    sanitized_artifact: Path,
    partitions_dir: Path,
    snapshot_path: Path,
    output_root: Path,
    contract_path: Path = DEFAULT_CONTRACT,
) -> Path:
    output_root = _external_output(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parent.parent
    policy_sources = _validate_policy_sources(repo_root)
    if file_sha256(contract_path) != EXPECTED_CONTRACT_SHA256:
        raise ValueError("STOP_POLICY_INTEGRITY: challenge contract changed")
    sanitized_manifest, queries, queries_sha256 = load_sanitized_artifact(
        sanitized_artifact
    )
    runtime = validate_runtime_inputs(
        partitions_dir=partitions_dir,
        snapshot_path=snapshot_path,
    )
    source_path = Path(__file__).resolve()
    identity = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "sanitized_manifest_sha256": file_sha256(
            Path(sanitized_artifact) / "manifest.json"
        ),
        "sanitized_queries_sha256": queries_sha256,
        "sanitized_build_id": sanitized_manifest.get("build_id"),
        "partitions_signature": runtime["partitions_signature"],
        "snapshot_sha256": runtime["snapshot_sha256"],
        "policy_version": POLICY_VERSION,
        "policy_source_hashes": policy_sources,
        "contract_sha256": file_sha256(contract_path),
        "builder_source_sha256": file_sha256(source_path),
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    target = output_root / build_id
    if target.exists():
        raise FileExistsError(f"Immutable unseen qualification exists: {target}")
    staging = Path(tempfile.mkdtemp(prefix=f".{build_id}.tmp-", dir=output_root))
    try:
        evidence, labels = qualify_queries(
            queries,
            partitions_dir=partitions_dir,
            snapshot_sha256=runtime["snapshot_sha256"],
        )
        evidence_path = staging / "evidence.parquet"
        labels_path = staging / "labels_frozen.parquet"
        evidence.to_parquet(evidence_path, index=False)
        labels.to_parquet(labels_path, index=False)
        outputs = {
            path.name: {
                "sha256": file_sha256(path),
                "size_bytes": int(path.stat().st_size),
                "row_count": int(len(frame)),
                "columns": list(frame.columns),
            }
            for path, frame in (
                (evidence_path, evidence),
                (labels_path, labels),
            )
        }
        manifest = {
            **identity,
            "build_identity": identity,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "inputs": {
                "sanitized_artifact_manifest": {
                    "path": str(
                        (Path(sanitized_artifact) / "manifest.json").resolve()
                    ),
                    "sha256": identity["sanitized_manifest_sha256"],
                },
                "sanitized_queries": {
                    "path": str(
                        (
                            Path(sanitized_artifact)
                            / SANITIZED_QUERIES_FILENAME
                        ).resolve()
                    ),
                    "sha256": queries_sha256,
                    "row_count": len(queries),
                },
                "partitions": {
                    "path": str(Path(partitions_dir).resolve()),
                    "runtime_signature": runtime["partitions_signature"],
                },
                "snapshot": {
                    "path": str(Path(snapshot_path).resolve()),
                    "sha256": runtime["snapshot_sha256"],
                },
                "contract": {
                    "path": str(Path(contract_path).resolve()),
                    "sha256": identity["contract_sha256"],
                },
            },
            "outputs": outputs,
            "label_counts": {
                str(key): int(value)
                for key, value in labels["label_kind"].value_counts().items()
            },
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "logical_cpu_count": os.cpu_count(),
            },
            "invariants": {
                "source_registry_opened": False,
                "models_or_scores_opened": False,
                "retrieval_topk_used": False,
                "full_geographic_universe_used": True,
                "active_only": True,
                "no_match_created": False,
                "labels_and_evidence_closed_atomically": True,
                "human_validated": False,
            },
        }
        _json_dump(staging / "manifest.json", manifest)
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    validate_artifact(target)
    return target


def validate_artifact(artifact_dir: Path) -> None:
    root = Path(artifact_dir).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported unseen qualification artifact")
    identity = manifest.get("build_identity") or {}
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    if manifest.get("build_id") != build_id or root.name != build_id:
        raise ValueError("Unseen qualification build identity mismatch")
    repo_root = Path(__file__).resolve().parent.parent
    observed_sources = _validate_policy_sources(repo_root)
    if identity.get("policy_source_hashes") != observed_sources:
        raise ValueError("Qualification policy source identity changed")
    if identity.get("builder_source_sha256") != file_sha256(
        Path(__file__).resolve()
    ):
        raise ValueError("Qualification builder source drift")
    if (
        identity.get("snapshot_sha256") != EXPECTED_SNAPSHOT_SHA256
        or identity.get("partitions_signature")
        != EXPECTED_PARTITIONS_SIGNATURE
        or identity.get("contract_sha256") != EXPECTED_CONTRACT_SHA256
        or identity.get("policy_version") != POLICY_VERSION
    ):
        raise ValueError("Qualification frozen identity changed")
    inputs = manifest.get("inputs") or {}
    _validate_sanitized_link(identity, inputs)
    contract_record = inputs.get("contract") or {}
    if (
        file_sha256(Path(str(contract_record.get("path") or "")))
        != EXPECTED_CONTRACT_SHA256
        or contract_record.get("sha256") != EXPECTED_CONTRACT_SHA256
    ):
        raise ValueError("Qualification contract drift")
    snapshot_record = inputs.get("snapshot") or {}
    partitions_record = inputs.get("partitions") or {}
    validate_runtime_inputs(
        partitions_dir=Path(str(partitions_record.get("path") or "")),
        snapshot_path=Path(str(snapshot_record.get("path") or "")),
    )
    for filename, record in (manifest.get("outputs") or {}).items():
        if file_sha256(root / filename) != record.get("sha256"):
            raise ValueError(f"Qualification output hash mismatch: {filename}")
    query_record = (manifest.get("inputs") or {}).get("sanitized_queries") or {}
    queries_path = Path(str(query_record.get("path") or ""))
    if file_sha256(queries_path) != query_record.get("sha256"):
        raise ValueError("Sanitized qualification input drift")
    queries = validate_query_schema(
        pd.read_parquet(queries_path, columns=QUERY_COLUMNS)
    )
    evidence = pd.read_parquet(root / "evidence.parquet")
    labels = pd.read_parquet(root / "labels_frozen.parquet")
    validate_qualification(queries, evidence, labels)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sanitized-artifact", type=Path)
    parser.add_argument("--partitions", type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--validate-artifact", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate_artifact is not None:
        validate_artifact(args.validate_artifact)
        print(args.validate_artifact)
        return
    required = ("sanitized_artifact", "partitions", "snapshot", "output_root")
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise SystemExit(f"Missing required arguments: {', '.join(missing)}")
    print(
        build_artifact(
            sanitized_artifact=args.sanitized_artifact,
            partitions_dir=args.partitions,
            snapshot_path=args.snapshot,
            output_root=args.output_root,
            contract_path=args.contract,
        )
    )


if __name__ == "__main__":
    main()
