#!/usr/bin/env python3
"""Build the sanitized V4.11 unseen challenge docket without CRM identifiers."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Mapping

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xgb_matcher.v9_dataset import file_sha256


SCHEMA_VERSION = "sireto-v4.11-descriptive-unseen-sanitized-1"
EXPERIMENT_ID = "V411_DESCRIPTIVE_UNSEEN_SANITIZED"
CHALLENGE_ID = "DESCRIPTIVE_UNSEEN_225"
EXPECTED_ROWS = 225
SOURCE_COLUMNS = [
    "source_row_number",
    "SITE",
    "CODE_POSTAL",
    "CODE_INSEE",
    "COMMUNE",
    "SITE_CLI_ADRESSE",
    "SITE_CLI_COMMUNE",
]
QUERY_COLUMNS = [
    "query_id",
    "crm_record_id",
    "crm_name",
    "crm_address",
    "crm_postcode",
    "crm_city",
    "crm_insee",
]
FORBIDDEN_COLUMN = re.compile(
    r"siret|siren|truth|label|candidate|rank|score|prediction|"
    r"service_?id|fingerprint",
    flags=re.IGNORECASE,
)
EXPECTED_REGISTRY_BUILD_ID = "fd25d1922040d585"
EXPECTED_REGISTRY_MANIFEST_SHA256 = (
    "77711f91fda8dffec3210c49b3df8404e46ff540f30f9597fc7fe7722f2d6962"
)
EXPECTED_UNSEEN_SHA256 = (
    "63ff648f6e326721e0646b0101de079f9a6feadb6e02c0474066c1288d8025a3"
)
EXPECTED_CANDIDATE_BUILD_ID = "9d23bf3deb6b63de"
EXPECTED_CANDIDATE_MANIFEST_SHA256 = (
    "a7fc765fe439392baec61fa8a35a941bb1f778281ccdbb54b55c699e9f0c11d9"
)
EXPECTED_CONTRACT_SHA256 = (
    "28785be1c776f27b9dc9357fe543049bb70d6937b6b03d6f59c33eee67f43026"
)
EXPECTED_CONTAMINATION_REGISTRY_SHA256 = (
    "18424afa7a09a87ce13ce1ee09aaeb3887b6e5c7f5ffc4a91172a5fdbe1ed452"
)


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _query_id(source_row_number: int) -> str:
    return hashlib.sha256(
        f"v4.11-unseen-query:{int(source_row_number)}".encode()
    ).hexdigest()[:24]


def _text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def validate_query_schema(frame: pd.DataFrame, *, canonical: bool = True) -> None:
    if list(frame.columns) != QUERY_COLUMNS:
        raise ValueError("STOP_DESCRIPTIVE_INTEGRITY: sanitized schema changed")
    forbidden = [name for name in frame.columns if FORBIDDEN_COLUMN.search(name)]
    if forbidden:
        raise ValueError(
            f"STOP_DESCRIPTIVE_INTEGRITY: forbidden columns {forbidden}"
        )
    if frame["query_id"].astype(str).duplicated().any():
        raise ValueError("STOP_DESCRIPTIVE_INTEGRITY: duplicate query_id")
    if frame["crm_record_id"].astype(str).duplicated().any():
        raise ValueError("STOP_DESCRIPTIVE_INTEGRITY: duplicate crm_record_id")
    if canonical and len(frame) != EXPECTED_ROWS:
        raise ValueError("STOP_DESCRIPTIVE_INTEGRITY: unseen volume changed")
    required = [
        "crm_name",
        "crm_address",
        "crm_postcode",
        "crm_city",
        "crm_insee",
    ]
    if frame[required].map(_text).eq("").any().any():
        raise ValueError("STOP_DESCRIPTIVE_INTEGRITY: required CRM field empty")


def validate_sealed_mapping(
    mapping: pd.DataFrame,
    queries: pd.DataFrame,
    *,
    canonical: bool = True,
) -> None:
    if list(mapping.columns) != ["query_id", "source_row_number", "cohort"]:
        raise ValueError("STOP_DESCRIPTIVE_INTEGRITY: sealed mapping changed")
    if len(mapping) != len(queries) or set(mapping["query_id"]) != set(
        queries["query_id"]
    ):
        raise ValueError("STOP_DESCRIPTIVE_INTEGRITY: mapping/query mismatch")
    if mapping["source_row_number"].astype(int).duplicated().any():
        raise ValueError("STOP_DESCRIPTIVE_INTEGRITY: duplicate source row")
    expected_ids = mapping["source_row_number"].astype(int).map(_query_id)
    if not mapping["query_id"].astype(str).equals(expected_ids):
        raise ValueError("STOP_DESCRIPTIVE_INTEGRITY: opaque mapping changed")
    exposed = {1102, 1169, 1314}
    expected_cohort = mapping["source_row_number"].astype(int).map(
        lambda value: (
            "EXPOSED_3"
            if value in exposed
            else "DESCRIPTIVE_UNSEEN_BLIND_222"
        )
    )
    if not mapping["cohort"].astype(str).equals(expected_cohort):
        raise ValueError("STOP_DESCRIPTIVE_INTEGRITY: cohort mapping changed")
    counts = mapping["cohort"].value_counts().to_dict()
    if canonical and counts != {
        "DESCRIPTIVE_UNSEEN_BLIND_222": 222,
        "EXPOSED_3": 3,
    }:
        raise ValueError("STOP_DESCRIPTIVE_INTEGRITY: cohort counts changed")


def _validate_frozen_inputs(
    *,
    registry_manifest_path: Path,
    unseen_path: Path,
    candidate_manifest_path: Path,
    contamination_registry_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if file_sha256(registry_manifest_path) != EXPECTED_REGISTRY_MANIFEST_SHA256:
        raise ValueError("STOP_DESCRIPTIVE_INTEGRITY: registry manifest drift")
    registry = json.loads(registry_manifest_path.read_text(encoding="utf-8"))
    if registry.get("build_id") != EXPECTED_REGISTRY_BUILD_ID:
        raise ValueError("STOP_DESCRIPTIVE_INTEGRITY: registry build changed")
    unseen_record = (registry.get("outputs") or {}).get("unseen.parquet") or {}
    if (
        file_sha256(unseen_path) != EXPECTED_UNSEEN_SHA256
        or unseen_record.get("sha256") != EXPECTED_UNSEEN_SHA256
        or int(unseen_record.get("rows", -1)) != EXPECTED_ROWS
    ):
        raise ValueError("STOP_DESCRIPTIVE_INTEGRITY: unseen source drift")
    if file_sha256(candidate_manifest_path) != EXPECTED_CANDIDATE_MANIFEST_SHA256:
        raise ValueError("STOP_DESCRIPTIVE_INTEGRITY: candidate manifest drift")
    candidate = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    if (
        candidate.get("build_id") != EXPECTED_CANDIDATE_BUILD_ID
        or candidate.get("verdict") != "GO_FREEZE_V411_CANDIDATE"
        or candidate.get("winner") != "COMPACT_LOGIT"
    ):
        raise ValueError("STOP_DESCRIPTIVE_INTEGRITY: candidate not frozen GO")
    contamination = json.loads(
        contamination_registry_path.read_text(encoding="utf-8")
    )
    rows = [int(value) for value in contamination.get("source_row_numbers") or []]
    if (
        contamination.get("challenge_id") != CHALLENGE_ID
        or sorted(rows) != [1102, 1169, 1314]
        or contamination.get("reason")
        != "INPUT_SIRET_EXPOSED_TO_ROOT_CONTEXT"
        or not (contamination.get("policy") or {}).get(
            "exclude_from_primary_blind_metrics"
        )
    ):
        raise ValueError(
            "STOP_DESCRIPTIVE_INTEGRITY: contamination registry changed"
        )
    return registry, candidate, contamination


def validate_artifact(artifact_dir: Path, *, canonical: bool = True) -> None:
    artifact_dir = Path(artifact_dir).resolve()
    manifest = json.loads(
        (artifact_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("STOP_DESCRIPTIVE_INTEGRITY: unsupported artifact")
    if manifest.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("STOP_DESCRIPTIVE_INTEGRITY: experiment changed")
    identity = manifest.get("build_identity") or {}
    expected_identity = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "challenge_id": CHALLENGE_ID,
        "registry_manifest_sha256": EXPECTED_REGISTRY_MANIFEST_SHA256,
        "unseen_source_sha256": EXPECTED_UNSEEN_SHA256,
        "candidate_manifest_sha256": EXPECTED_CANDIDATE_MANIFEST_SHA256,
        "contamination_registry_sha256":
            EXPECTED_CONTAMINATION_REGISTRY_SHA256,
        "contract_sha256": EXPECTED_CONTRACT_SHA256,
        "builder_sha256": file_sha256(Path(__file__).resolve()),
        "query_columns": QUERY_COLUMNS,
        "row_count": EXPECTED_ROWS,
    }
    if identity != expected_identity:
        raise ValueError("STOP_DESCRIPTIVE_INTEGRITY: build identity changed")
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    if manifest.get("build_id") != build_id or artifact_dir.name != build_id:
        raise ValueError("STOP_DESCRIPTIVE_INTEGRITY: build identity mismatch")
    for name, record in (manifest.get("outputs") or {}).items():
        path = artifact_dir / name
        if file_sha256(path) != str(record.get("sha256")):
            raise ValueError(
                f"STOP_DESCRIPTIVE_INTEGRITY: output hash mismatch {name}"
            )
    queries = pd.read_parquet(artifact_dir / "queries_sanitized.parquet")
    validate_query_schema(queries, canonical=canonical)
    mapping = pd.read_parquet(artifact_dir / "sealed_mapping.parquet")
    validate_sealed_mapping(mapping, queries, canonical=canonical)
    inputs = manifest.get("inputs") or {}
    expected_inputs = {
        "registry_manifest": EXPECTED_REGISTRY_MANIFEST_SHA256,
        "unseen_source": EXPECTED_UNSEEN_SHA256,
        "candidate_manifest": EXPECTED_CANDIDATE_MANIFEST_SHA256,
        "contamination_registry": EXPECTED_CONTAMINATION_REGISTRY_SHA256,
        "contract": EXPECTED_CONTRACT_SHA256,
    }
    for name, expected_sha in expected_inputs.items():
        record = inputs.get(name) or {}
        path = Path(str(record.get("path") or ""))
        if record.get("sha256") != expected_sha or file_sha256(path) != expected_sha:
            raise ValueError(
                f"STOP_DESCRIPTIVE_INTEGRITY: input drift {name}"
            )


def build_artifact(
    *,
    registry_manifest_path: Path,
    unseen_path: Path,
    candidate_manifest_path: Path,
    contamination_registry_path: Path,
    contract_path: Path,
    output_root: Path,
    canonical: bool = True,
) -> Path:
    _, _, contamination = _validate_frozen_inputs(
        registry_manifest_path=registry_manifest_path,
        unseen_path=unseen_path,
        candidate_manifest_path=candidate_manifest_path,
        contamination_registry_path=contamination_registry_path,
    )
    if file_sha256(contract_path) != EXPECTED_CONTRACT_SHA256:
        raise ValueError("STOP_DESCRIPTIVE_INTEGRITY: contract drift")
    if (
        file_sha256(contamination_registry_path)
        != EXPECTED_CONTAMINATION_REGISTRY_SHA256
    ):
        raise ValueError(
            "STOP_DESCRIPTIVE_INTEGRITY: contamination registry drift"
        )
    # Physical projection is intentional: forbidden source columns are never loaded.
    source = pd.read_parquet(unseen_path, columns=SOURCE_COLUMNS)
    if source["source_row_number"].duplicated().any():
        raise ValueError("STOP_DESCRIPTIVE_INTEGRITY: duplicate source row")
    exposed = {
        int(value) for value in contamination["source_row_numbers"]
    }
    records: list[dict[str, str]] = []
    mappings: list[dict[str, Any]] = []
    for row in source.sort_values("source_row_number").to_dict("records"):
        source_row = int(row["source_row_number"])
        query_id = _query_id(source_row)
        city = _text(row.get("COMMUNE")) or _text(row.get("SITE_CLI_COMMUNE"))
        records.append(
            {
                "query_id": query_id,
                "crm_record_id": query_id,
                "crm_name": _text(row.get("SITE")),
                "crm_address": _text(row.get("SITE_CLI_ADRESSE")),
                "crm_postcode": _text(row.get("CODE_POSTAL")),
                "crm_city": city,
                "crm_insee": _text(row.get("CODE_INSEE")),
            }
        )
        mappings.append(
            {
                "query_id": query_id,
                "source_row_number": source_row,
                "cohort": (
                    "EXPOSED_3"
                    if source_row in exposed
                    else "DESCRIPTIVE_UNSEEN_BLIND_222"
                ),
            }
        )
    queries = pd.DataFrame(records, columns=QUERY_COLUMNS)
    validate_query_schema(queries, canonical=canonical)
    mapping = pd.DataFrame(
        mappings, columns=["query_id", "source_row_number", "cohort"]
    )
    identity = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "challenge_id": CHALLENGE_ID,
        "registry_manifest_sha256": file_sha256(registry_manifest_path),
        "unseen_source_sha256": file_sha256(unseen_path),
        "candidate_manifest_sha256": file_sha256(candidate_manifest_path),
        "contamination_registry_sha256": file_sha256(
            contamination_registry_path
        ),
        "contract_sha256": file_sha256(contract_path),
        "builder_sha256": file_sha256(Path(__file__).resolve()),
        "query_columns": QUERY_COLUMNS,
        "row_count": len(queries),
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / build_id
    if target.exists():
        raise FileExistsError(f"Immutable sanitized artifact exists: {target}")
    staging = Path(tempfile.mkdtemp(prefix=f".{build_id}.tmp-", dir=output_root))
    try:
        queries_path = staging / "queries_sanitized.parquet"
        mapping_path = staging / "sealed_mapping.parquet"
        queries.to_parquet(queries_path, index=False)
        mapping.to_parquet(mapping_path, index=False)
        manifest = {
            **identity,
            "build_identity": identity,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "inputs": {
                "registry_manifest": {
                    "path": str(registry_manifest_path.resolve()),
                    "sha256": file_sha256(registry_manifest_path),
                },
                "unseen_source": {
                    "path": str(unseen_path.resolve()),
                    "sha256": file_sha256(unseen_path),
                },
                "candidate_manifest": {
                    "path": str(candidate_manifest_path.resolve()),
                    "sha256": file_sha256(candidate_manifest_path),
                },
                "contamination_registry": {
                    "path": str(contamination_registry_path.resolve()),
                    "sha256": file_sha256(contamination_registry_path),
                },
                "contract": {
                    "path": str(contract_path.resolve()),
                    "sha256": file_sha256(contract_path),
                },
            },
            "outputs": {
                queries_path.name: {
                    "sha256": file_sha256(queries_path),
                    "row_count": len(queries),
                    "columns": list(queries.columns),
                },
                mapping_path.name: {
                    "sha256": file_sha256(mapping_path),
                    "row_count": len(mapping),
                    "columns": list(mapping.columns),
                },
            },
            "cohort_counts": mapping["cohort"].value_counts().to_dict(),
            "invariants": {
                "physical_source_projection": SOURCE_COLUMNS,
                "forbidden_source_columns_loaded": False,
                "labels_loaded": False,
                "retrieval_or_model_loaded": False,
                "input_siret_or_siren_exposed": False,
            },
        }
        (staging / "manifest.json").write_bytes(_json_bytes(manifest))
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    validate_artifact(target, canonical=canonical)
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-manifest", type=Path)
    parser.add_argument("--unseen", type=Path)
    parser.add_argument("--candidate-manifest", type=Path)
    parser.add_argument("--contamination-registry", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--validate-artifact", type=Path)
    parser.add_argument("--allow-noncanonical", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    canonical = not args.allow_noncanonical
    if args.validate_artifact:
        validate_artifact(args.validate_artifact, canonical=canonical)
        return
    required = [
        args.registry_manifest,
        args.unseen,
        args.candidate_manifest,
        args.contamination_registry,
        args.contract,
        args.output_root,
    ]
    if any(value is None for value in required):
        raise SystemExit("All build paths are required")
    target = build_artifact(
        registry_manifest_path=args.registry_manifest,
        unseen_path=args.unseen,
        candidate_manifest_path=args.candidate_manifest,
        contamination_registry_path=args.contamination_registry,
        contract_path=args.contract,
        output_root=args.output_root,
        canonical=canonical,
    )
    print(target)


if __name__ == "__main__":
    main()
