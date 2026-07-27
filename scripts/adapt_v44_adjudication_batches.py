#!/usr/bin/env python3
"""Adapt reviewed V4.4 JSON batches to immutable canonical builder inputs.

The adapter is deliberately generic: it has no case allow-list and no source
mapping table. Labels and proof taxonomy are copied from the reviewed JSON,
while frozen candidates and model identifiers come only from the V4.1 shadow.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_v44_adjudications import (  # noqa: E402
    LABELS,
    build_adjudications,
    candidate_pool_sha256,
)
from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402


SCHEMA_VERSION = "sireto-v4.4-adjudication-batch-inputs-1"
RULE_VERSION = "two-independent-identity-proofs-v1"
QUEUE_SCHEMA_VERSION = "sireto-v4.3-hard-label-queue-1"
SHADOW_SCHEMA_VERSION = "sireto-shadow-v4.1-1"
FACT_COLUMNS = [
    "audit_case_id",
    "service_id",
    "query_id",
    "frozen_top1_siret",
    "frozen_top1_siren",
    "frozen_model_bundle_id",
    "frozen_ranker_bundle_id",
    "frozen_acceptor_bundle_id",
    "frozen_retrieval_signature",
    "frozen_candidate_sirets_json",
    "frozen_candidate_pool_sha256",
    "frozen_candidate_count",
    "frozen_candidate_source",
    "frozen_candidate_source_path",
    "positive_injection_by_adapter",
    "sampling_stratum",
    "priority_reason",
]
PROOF_COLUMNS = [
    "proof_id",
    "audit_case_id",
    "producer",
    "source_family",
    "source_family_as_reviewed",
    "independence_group",
    "source_locator",
    "source_locators_json",
    "collected_at",
    "collected_at_precision",
    "document_type",
    "document_date",
    "archived_facts_json",
    "local_artifact_path",
    "local_artifact_sha256",
    "local_row_selector_json",
    "proof_kind",
    "supports_label",
    "identity_consistent",
    "contradiction_unresolved",
]
JUDGMENT_COLUMNS = [
    "audit_case_id",
    "adjudication_label",
    "validated_correct_siret",
    "evidence_ref_ids_json",
    "adjudication_reason",
    "adjudication_rule_version",
    "adjudicated_at",
]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _boolean(value: Any, *, field: str) -> bool:
    if value is True or value is False or type(value).__name__ == "bool_":
        return bool(value)
    raise ValueError(f"{field} must be a boolean")


def _siret(value: Any, *, field: str) -> str:
    text = _text(value)
    if len(text) != 14 or not text.isdigit():
        raise ValueError(f"{field} must be a 14-digit SIRET")
    return text


def _canonical_json(values: Iterable[str]) -> str:
    return json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))


def _collected_at(value: Any) -> tuple[str, str]:
    text = _text(value)
    if not text:
        raise ValueError("Every proof requires collected_at")
    if len(text) == 10 and text[4] == "-" and text[7] == "-":
        timestamp = pd.Timestamp(f"{text}T00:00:00", tz="Europe/Paris")
        return timestamp.tz_convert("UTC").isoformat(), "DAY_EUROPE_PARIS"
    timestamp = pd.Timestamp(text)
    if timestamp.tzinfo is None:
        raise ValueError(f"Proof collected_at lacks timezone: {text}")
    return timestamp.tz_convert("UTC").isoformat(), "TIMESTAMP"


def _source_urls(source: Mapping[str, Any], *, case_id: str) -> list[str]:
    raw_urls = source.get("urls")
    if raw_urls is not None and not isinstance(raw_urls, list):
        raise ValueError(f"{case_id}: proof urls must be a list")
    values = [_text(source.get("url"))]
    values.extend(_text(value) for value in (raw_urls or []))
    urls = sorted({value for value in values if value})
    if not urls or any(not value.startswith("https://") for value in urls):
        raise ValueError(f"{case_id}: proof requires public HTTPS URL(s)")
    return urls


def _resolve_recorded_path(value: Any, *, manifest_path: Path) -> Path:
    path = Path(_text(value))
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return manifest_path.parent / path


def _validate_recorded_files(
    records: Mapping[str, Any],
    *,
    manifest_path: Path,
    context: str,
) -> None:
    for name, raw in records.items():
        if not isinstance(raw, Mapping):
            raise ValueError(f"{context} has an invalid file record: {name}")
        path = _resolve_recorded_path(raw.get("path"), manifest_path=manifest_path)
        expected = _text(raw.get("sha256")).lower()
        if not expected or not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"{context} input hash mismatch: {name}")


def _validate_queue_dir(queue_dir: Path) -> tuple[dict[str, Any], Path, pd.DataFrame]:
    queue_dir = Path(queue_dir)
    manifest_path = queue_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != QUEUE_SCHEMA_VERSION:
        raise ValueError("Unsupported V4.3 queue manifest schema")
    queue_path = queue_dir / "hard_label_queue.parquet"
    expected = _text((manifest.get("outputs") or {}).get(queue_path.name))
    if not expected or not queue_path.is_file() or file_sha256(queue_path) != expected:
        raise ValueError("V4.3 hard-label queue hash mismatch")
    _validate_recorded_files(
        manifest.get("inputs") or {},
        manifest_path=manifest_path,
        context="V4.3 queue manifest",
    )
    queue = pd.read_parquet(queue_path)
    required = {
        "audit_case_id",
        "service_id",
        "top1_siret",
        "sampling_stratum",
        "priority_reason",
    }
    missing = required - set(queue.columns)
    if missing:
        raise ValueError(f"V4.3 queue missing columns: {sorted(missing)}")
    if queue["audit_case_id"].astype(str).duplicated().any():
        raise ValueError("V4.3 queue contains duplicate audit_case_id")
    if queue["service_id"].astype(str).duplicated().any():
        raise ValueError("V4.3 queue contains duplicate service_id")
    return manifest, queue_path, queue


def _validate_shadow_dir(
    shadow_dir: Path,
) -> tuple[dict[str, Any], Path, Path, pd.DataFrame, pd.DataFrame]:
    shadow_dir = Path(shadow_dir)
    manifest_path = shadow_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != SHADOW_SCHEMA_VERSION:
        raise ValueError("Unsupported V4.1 shadow manifest schema")
    metadata = manifest.get("run_metadata") or {}
    required_metadata = {
        "release_id",
        "ranker_bundle_id",
        "acceptor_bundle_id",
        "retrieval_signature",
    }
    missing_metadata = sorted(
        key for key in required_metadata if not _text(metadata.get(key))
    )
    if missing_metadata:
        raise ValueError(
            f"Shadow manifest missing frozen identifiers: {missing_metadata}"
        )
    outputs = manifest.get("outputs") or {}
    top10_path = shadow_dir / "candidates_top10.parquet"
    decisions_path = shadow_dir / "decisions.parquet"
    for path in (top10_path, decisions_path):
        expected = _text(outputs.get(path.name))
        if not expected or not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"V4.1 shadow output hash mismatch: {path.name}")
    _validate_recorded_files(
        manifest.get("inputs") or {},
        manifest_path=manifest_path,
        context="V4.1 shadow manifest",
    )
    top10 = pd.read_parquet(top10_path)
    decisions = pd.read_parquet(decisions_path)
    top10_required = {"service_id", "rank", "candidate_siret"}
    decision_required = {"service_id", "predicted_siret"}
    if top10_required - set(top10.columns):
        raise ValueError("V4.1 top10 lacks required candidate columns")
    if decision_required - set(decisions.columns):
        raise ValueError("V4.1 decisions lack required columns")
    if decisions["service_id"].astype(str).duplicated().any():
        raise ValueError("V4.1 decisions contain duplicate service_id")
    return manifest, top10_path, decisions_path, top10, decisions


def _validate_batch_file_references(
    batch: Mapping[str, Any],
    *,
    batch_path: Path,
) -> None:
    checked: dict[Path, str] = {}
    for index, raw in enumerate(batch.get("source_artifacts") or []):
        if not isinstance(raw, Mapping):
            raise ValueError(f"{batch_path}: invalid source_artifacts[{index}]")
        path = _resolve_recorded_path(raw.get("path"), manifest_path=batch_path)
        expected = _text(raw.get("sha256")).lower()
        if not expected:
            raise ValueError(f"{batch_path}: source artifact lacks sha256: {path}")
        if path in checked and checked[path] != expected:
            raise ValueError(
                f"{batch_path}: conflicting hashes for referenced artifact: {path}"
            )
        checked[path] = expected
    for case in batch.get("cases") or []:
        for source in case.get("sources") or []:
            path_text = _text(source.get("artifact_path"))
            expected = _text(source.get("artifact_sha256")).lower()
            if path_text and expected:
                path = _resolve_recorded_path(
                    path_text,
                    manifest_path=batch_path,
                )
                if path in checked and checked[path] != expected:
                    raise ValueError(
                        f"{batch_path}: conflicting hashes for referenced artifact: "
                        f"{path}"
                    )
                checked[path] = expected
    for path, expected in checked.items():
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"{batch_path}: referenced artifact hash mismatch: {path}")


def _load_batches(batch_jsons: Sequence[Path]) -> list[dict[str, Any]]:
    if not batch_jsons:
        raise ValueError("At least one adjudication batch is required")
    paths = [Path(path) for path in batch_jsons]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate adjudication batch path")
    batches: list[dict[str, Any]] = []
    for path in paths:
        batch = _read_json(path)
        if not isinstance(batch.get("cases"), list) or not batch["cases"]:
            raise ValueError(f"{path}: cases must be a non-empty list")
        if not _text(batch.get("created_at")):
            raise ValueError(f"{path}: created_at is required")
        _validate_batch_file_references(batch, batch_path=path)
        batches.append(batch)
    return batches


def _candidate_pools(top10: pd.DataFrame) -> dict[str, list[str]]:
    pools: dict[str, list[str]] = {}
    working = top10.copy()
    working["service_id"] = working["service_id"].map(_text)
    for service_id, group in working.groupby("service_id", sort=False):
        ordered = group.sort_values("rank")
        ranks = [int(value) for value in ordered["rank"]]
        if not 1 <= len(ranks) <= 10 or ranks != list(range(1, len(ranks) + 1)):
            raise ValueError(
                f"{service_id}: shadow top10 ranks must be consecutive from 1"
            )
        sirets = [
            _siret(value, field=f"{service_id}.candidate_siret")
            for value in ordered["candidate_siret"]
        ]
        if len(sirets) != len(set(sirets)):
            raise ValueError(f"{service_id}: duplicate SIRET in shadow top10")
        pools[service_id] = sirets
    return pools


def adapt_batches(
    *,
    batches: Sequence[Mapping[str, Any]],
    queue: pd.DataFrame,
    top10: pd.DataFrame,
    decisions: pd.DataFrame,
    shadow_manifest: Mapping[str, Any],
    top10_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Validate reviewed batches and convert them without source/label mapping."""

    queue_by_case = {
        _text(row["audit_case_id"]): row for row in queue.to_dict("records")
    }
    decision_by_service = {
        _text(row["service_id"]): row for row in decisions.to_dict("records")
    }
    pools = _candidate_pools(top10)
    metadata = shadow_manifest["run_metadata"]
    facts: list[dict[str, Any]] = []
    proofs: list[dict[str, Any]] = []
    judgments: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    seen_service_ids: set[str] = set()
    seen_proof_ids: set[str] = set()

    for batch in batches:
        adjudicated_at = _text(batch.get("created_at"))
        for case in batch["cases"]:
            if not isinstance(case, Mapping):
                raise ValueError("Every batch case must be an object")
            case_id = _text(case.get("audit_case_id"))
            service_id = _text(case.get("service_id"))
            if not case_id or not service_id:
                raise ValueError("Every case requires audit_case_id and service_id")
            if case_id in seen_case_ids:
                raise ValueError(f"Duplicate or conflicting audit_case_id: {case_id}")
            if service_id in seen_service_ids:
                raise ValueError(f"Duplicate or conflicting service_id: {service_id}")
            seen_case_ids.add(case_id)
            seen_service_ids.add(service_id)

            queue_row = queue_by_case.get(case_id)
            if queue_row is None:
                raise ValueError(f"{case_id}: absent from frozen V4.3 queue")
            if _text(queue_row.get("service_id")) != service_id:
                raise ValueError(f"{case_id}: service_id mismatch with V4.3 queue")
            sampling = _text(queue_row.get("sampling_stratum"))
            if not sampling:
                raise ValueError(f"{case_id}: frozen sampling_stratum is empty")
            pool = pools.get(service_id)
            if pool is None:
                raise ValueError(f"{case_id}: service absent from V4.1 top10")
            decision = decision_by_service.get(service_id)
            if decision is None:
                raise ValueError(f"{case_id}: service absent from V4.1 decisions")

            frozen = case.get("frozen_top1")
            if not isinstance(frozen, Mapping):
                raise ValueError(f"{case_id}: frozen_top1 must be an object")
            top1 = _siret(frozen.get("siret"), field=f"{case_id}.frozen_top1.siret")
            supplied_siren = _text(frozen.get("siren"))
            if supplied_siren and supplied_siren != top1[:9]:
                raise ValueError(f"{case_id}: frozen top1 SIREN mismatch")
            if _text(queue_row.get("top1_siret")) != top1:
                raise ValueError(f"{case_id}: top1 mismatch with V4.3 queue")
            if pool[0] != top1:
                raise ValueError(f"{case_id}: top1 mismatch with V4.1 top10")
            if _text(decision.get("predicted_siret")) != top1:
                raise ValueError(f"{case_id}: top1 mismatch with V4.1 decision")

            label = _text(case.get("adjudication_label")).upper()
            if label not in LABELS:
                raise ValueError(f"{case_id}: unsupported adjudication label {label}")
            evidence_validated = _boolean(
                case.get("evidence_validated"),
                field=f"{case_id}.evidence_validated",
            )
            training_eligible = _boolean(
                case.get("training_eligible"),
                field=f"{case_id}.training_eligible",
            )
            if evidence_validated != training_eligible:
                raise ValueError(
                    f"{case_id}: evidence_validated must equal training_eligible"
                )
            reason = _text(case.get("decision_reason"))
            if not reason:
                raise ValueError(f"{case_id}: decision_reason is required")
            validated_siret_raw = case.get("validated_correct_siret")
            validated_siret = (
                _siret(
                    validated_siret_raw,
                    field=f"{case_id}.validated_correct_siret",
                )
                if _text(validated_siret_raw)
                else None
            )
            if label == "TOP1_CORRECT" and validated_siret != top1:
                raise ValueError(
                    f"{case_id}: TOP1_CORRECT must validate the frozen top1"
                )
            if label == "TOP1_WRONG" and validated_siret == top1:
                raise ValueError(
                    f"{case_id}: TOP1_WRONG replacement cannot equal top1"
                )
            if label in {"AMBIGUOUS", "UNRESOLVED"} and validated_siret:
                raise ValueError(f"{case_id}: {label} cannot have an exact SIRET")

            sources = case.get("sources")
            if not isinstance(sources, list):
                raise ValueError(f"{case_id}: sources must be a list")
            cited_ids: list[str] = []
            counted_groups: set[str] = set()
            counted_families: set[str] = set()
            counted_reviewed_families: set[str] = set()
            for source in sources:
                if not isinstance(source, Mapping):
                    raise ValueError(f"{case_id}: every source must be an object")
                proof_id = _text(source.get("evidence_id"))
                if not proof_id or proof_id in seen_proof_ids:
                    raise ValueError(f"{case_id}: missing or duplicate evidence_id")
                seen_proof_ids.add(proof_id)
                producer = _text(source.get("producer"))
                reviewed_family = _text(source.get("source_family"))
                canonical_family = _text(source.get("canonical_source_family")).upper()
                group = _text(source.get("independence_group")).upper()
                proof_kind = _text(source.get("proof_kind")).upper()
                if not producer or not reviewed_family:
                    raise ValueError(f"{case_id}: proof producer/source_family required")
                if not canonical_family or not group:
                    raise ValueError(
                        f"{case_id}: canonical source family and group are required"
                    )
                if not proof_kind.startswith("IDENTITY_"):
                    raise ValueError(
                        f"{case_id}: proof_kind must be explicit IDENTITY_*"
                    )
                counts = _boolean(
                    source.get("counts_for_independence"),
                    field=f"{proof_id}.counts_for_independence",
                )
                locators = _source_urls(source, case_id=case_id)
                archived_facts = source.get("archived_facts")
                if (
                    not isinstance(archived_facts, list)
                    or not archived_facts
                    or any(not _text(value) for value in archived_facts)
                ):
                    raise ValueError(f"{case_id}: proof requires archived_facts")
                collected_at, precision = _collected_at(source.get("collected_at"))
                proofs.append(
                    {
                        "proof_id": proof_id,
                        "audit_case_id": case_id,
                        "producer": producer,
                        "source_family": canonical_family,
                        "source_family_as_reviewed": reviewed_family,
                        "independence_group": group,
                        "source_locator": locators[0],
                        "source_locators_json": _canonical_json(locators),
                        "collected_at": collected_at,
                        "collected_at_precision": precision,
                        "document_type": _text(source.get("document_type")),
                        "document_date": source.get("document_date"),
                        "archived_facts_json": json.dumps(
                            archived_facts,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        "local_artifact_path": source.get("artifact_path"),
                        "local_artifact_sha256": source.get("artifact_sha256"),
                        "local_row_selector_json": (
                            json.dumps(
                                source.get("row_selector"),
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                            if source.get("row_selector") is not None
                            else None
                        ),
                        "proof_kind": proof_kind,
                        "supports_label": label,
                        "identity_consistent": evidence_validated,
                        "contradiction_unresolved": not evidence_validated,
                    }
                )
                if counts:
                    cited_ids.append(proof_id)
                    counted_groups.add(group)
                    counted_families.add(canonical_family)
                    counted_reviewed_families.add(reviewed_family.upper())

            declared_groups = case.get("independent_source_families")
            if not isinstance(declared_groups, list):
                raise ValueError(
                    f"{case_id}: independent_source_families must be a list"
                )
            normalized_declared = {_text(value).upper() for value in declared_groups}
            alias_to_groups: dict[str, set[str]] = {}
            for source in sources:
                if not _boolean(
                    source.get("counts_for_independence"),
                    field=f"{case_id}.counts_for_independence",
                ):
                    continue
                group = _text(source.get("independence_group")).upper()
                for alias in (
                    group,
                    _text(source.get("canonical_source_family")).upper(),
                    _text(source.get("source_family")).upper(),
                ):
                    alias_to_groups.setdefault(alias, set()).add(group)
            resolved_declared: set[str] = set()
            declaration_is_valid = "" not in normalized_declared
            for alias in normalized_declared:
                matching_groups = alias_to_groups.get(alias, set())
                if len(matching_groups) != 1:
                    declaration_is_valid = False
                    break
                resolved_declared.update(matching_groups)
            if not declaration_is_valid or resolved_declared != counted_groups:
                raise ValueError(
                    f"{case_id}: declared independent source groups mismatch sources"
                )
            if evidence_validated and (
                len(cited_ids) < 2
                or len(counted_groups) < 2
                or len(counted_families) < 2
            ):
                raise ValueError(
                    f"{case_id}: validated decision requires two independent proofs"
                )

            facts.append(
                {
                    "audit_case_id": case_id,
                    "service_id": service_id,
                    "query_id": service_id,
                    "frozen_top1_siret": top1,
                    "frozen_top1_siren": top1[:9],
                    "frozen_model_bundle_id": _text(metadata["release_id"]),
                    "frozen_ranker_bundle_id": _text(metadata["ranker_bundle_id"]),
                    "frozen_acceptor_bundle_id": _text(
                        metadata["acceptor_bundle_id"]
                    ),
                    "frozen_retrieval_signature": _text(
                        metadata["retrieval_signature"]
                    ),
                    "frozen_candidate_sirets_json": _canonical_json(pool),
                    "frozen_candidate_pool_sha256": candidate_pool_sha256(pool),
                    "frozen_candidate_count": len(pool),
                    "frozen_candidate_source": (
                        "V4_1_SHADOW_CANDIDATES_TOP10_PARQUET"
                    ),
                    "frozen_candidate_source_path": str(top10_path),
                    "positive_injection_by_adapter": False,
                    "sampling_stratum": sampling,
                    "priority_reason": _text(queue_row.get("priority_reason")),
                }
            )
            judgments.append(
                {
                    "audit_case_id": case_id,
                    "adjudication_label": label,
                    "validated_correct_siret": validated_siret,
                    "evidence_ref_ids_json": _canonical_json(cited_ids),
                    "adjudication_reason": reason,
                    "adjudication_rule_version": RULE_VERSION,
                    "adjudicated_at": adjudicated_at,
                }
            )

    frames = (
        pd.DataFrame(facts, columns=FACT_COLUMNS),
        pd.DataFrame(proofs, columns=PROOF_COLUMNS),
        pd.DataFrame(judgments, columns=JUDGMENT_COLUMNS),
    )
    canonical = build_adjudications(*frames)
    source_eligibility = {
        _text(case["audit_case_id"]): bool(case["training_eligible"])
        for batch in batches
        for case in batch["cases"]
    }
    derived_eligibility = canonical.set_index("audit_case_id")[
        "training_eligible"
    ].astype(bool).to_dict()
    if source_eligibility != derived_eligibility:
        raise ValueError("Source and canonical training eligibility differ")
    return frames


def _validate_queue_shadow_binding(
    queue_manifest: Mapping[str, Any],
    shadow_manifest: Mapping[str, Any],
) -> None:
    queue_top10 = _text(
        ((queue_manifest.get("inputs") or {}).get("top10") or {}).get("sha256")
    )
    shadow_top10 = _text(
        (shadow_manifest.get("outputs") or {}).get("candidates_top10.parquet")
    )
    if not queue_top10 or queue_top10 != shadow_top10:
        raise ValueError("V4.3 queue and V4.1 shadow top10 hashes differ")


def build_input_artifact(
    *,
    batch_jsons: Sequence[Path],
    queue_dir: Path,
    shadow_dir: Path,
    output_root: Path,
) -> Path:
    """Create an atomic, content-addressed facts/proofs/judgments artifact."""

    batch_paths = sorted(
        (Path(path) for path in batch_jsons),
        key=lambda path: (file_sha256(path), str(path)),
    )
    batches = _load_batches(batch_paths)
    queue_manifest, queue_path, queue = _validate_queue_dir(queue_dir)
    (
        shadow_manifest,
        top10_path,
        decisions_path,
        top10,
        decisions,
    ) = _validate_shadow_dir(shadow_dir)
    _validate_queue_shadow_binding(queue_manifest, shadow_manifest)
    frames = adapt_batches(
        batches=batches,
        queue=queue,
        top10=top10,
        decisions=decisions,
        shadow_manifest=shadow_manifest,
        top10_path=top10_path,
    )

    batch_inputs = [
        {"path": str(path), "sha256": file_sha256(path)}
        for path in batch_paths
    ]
    identity = {
        "schema_version": SCHEMA_VERSION,
        "batch_json_sha256": [item["sha256"] for item in batch_inputs],
        "queue_manifest_sha256": file_sha256(Path(queue_dir) / "manifest.json"),
        "queue_sha256": file_sha256(queue_path),
        "shadow_manifest_sha256": file_sha256(
            Path(shadow_dir) / "manifest.json"
        ),
        "shadow_top10_sha256": file_sha256(top10_path),
        "shadow_decisions_sha256": file_sha256(decisions_path),
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    target = Path(output_root) / build_id
    if target.exists():
        raise FileExistsError(f"Immutable adjudication batch input exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{build_id}.tmp-", dir=target.parent))
    try:
        output_hashes: dict[str, str] = {}
        for name, frame in zip(("facts", "proofs", "judgments"), frames):
            output_path = staging / f"{name}.parquet"
            frame.to_parquet(output_path, index=False)
            output_hashes[output_path.name] = file_sha256(output_path)
        manifest = {
            **identity,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "inputs": {
                "batch_jsons": batch_inputs,
                "queue_manifest": {
                    "path": str(Path(queue_dir) / "manifest.json"),
                    "sha256": identity["queue_manifest_sha256"],
                },
                "hard_label_queue": {
                    "path": str(queue_path),
                    "sha256": identity["queue_sha256"],
                },
                "shadow_manifest": {
                    "path": str(Path(shadow_dir) / "manifest.json"),
                    "sha256": identity["shadow_manifest_sha256"],
                },
                "shadow_top10": {
                    "path": str(top10_path),
                    "sha256": identity["shadow_top10_sha256"],
                },
                "shadow_decisions": {
                    "path": str(decisions_path),
                    "sha256": identity["shadow_decisions_sha256"],
                },
            },
            "frozen_identifiers": {
                key: shadow_manifest["run_metadata"][key]
                for key in (
                    "release_id",
                    "ranker_bundle_id",
                    "acceptor_bundle_id",
                    "retrieval_signature",
                )
            },
            "outputs": output_hashes,
            "row_counts": {
                name: int(len(frame))
                for name, frame in zip(("facts", "proofs", "judgments"), frames)
            },
            "invariants": {
                "case_allow_list": False,
                "source_mapping_table": False,
                "labels_copied_without_inference": True,
                "minimum_independent_evidence_groups": 2,
                "candidate_source": "candidates_top10.parquet",
                "candidate_cap": 10,
                "queue_shadow_top10_hash_equal": True,
                "positive_injection_by_adapter": False,
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def validate_input_artifact(path: Path) -> None:
    """Verify hashes and recompute the three canonical input tables."""

    path = Path(path)
    manifest = _read_json(path / "manifest.json")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported adjudication batch input schema")
    inputs = manifest.get("inputs") or {}
    batch_records = inputs.get("batch_jsons")
    if not isinstance(batch_records, list) or not batch_records:
        raise ValueError("Adapter manifest lacks batch JSON inputs")
    batch_paths: list[Path] = []
    for record in batch_records:
        batch_path = Path(_text(record.get("path")))
        if not batch_path.is_file() or file_sha256(batch_path) != record.get("sha256"):
            raise ValueError("Adapter batch JSON input hash mismatch")
        batch_paths.append(batch_path)
    for name in (
        "queue_manifest",
        "hard_label_queue",
        "shadow_manifest",
        "shadow_top10",
        "shadow_decisions",
    ):
        record = inputs.get(name) or {}
        source = Path(_text(record.get("path")))
        if not source.is_file() or file_sha256(source) != record.get("sha256"):
            raise ValueError(f"Adapter input hash mismatch: {name}")
    for filename, expected in (manifest.get("outputs") or {}).items():
        if not (path / filename).is_file() or file_sha256(path / filename) != expected:
            raise ValueError(f"Adapter output hash mismatch: {filename}")

    queue_dir = Path(inputs["queue_manifest"]["path"]).parent
    shadow_dir = Path(inputs["shadow_manifest"]["path"]).parent
    batches = _load_batches(batch_paths)
    queue_manifest, _, queue = _validate_queue_dir(queue_dir)
    shadow_manifest, top10_path, _, top10, decisions = _validate_shadow_dir(
        shadow_dir
    )
    _validate_queue_shadow_binding(queue_manifest, shadow_manifest)
    expected_frames = adapt_batches(
        batches=batches,
        queue=queue,
        top10=top10,
        decisions=decisions,
        shadow_manifest=shadow_manifest,
        top10_path=top10_path,
    )
    for name, expected in zip(("facts", "proofs", "judgments"), expected_frames):
        observed = pd.read_parquet(path / f"{name}.parquet")
        pd.testing.assert_frame_equal(observed, expected, check_dtype=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-json", action="append", type=Path)
    parser.add_argument("--queue-dir", type=Path)
    parser.add_argument("--shadow-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--validate-artifact", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate_artifact:
        validate_input_artifact(args.validate_artifact)
        print(args.validate_artifact)
        return
    missing = [
        name
        for name in ("batch_json", "queue_dir", "shadow_dir", "output_root")
        if not getattr(args, name)
    ]
    if missing:
        raise SystemExit(
            "Building requires: "
            + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        )
    print(
        build_input_artifact(
            batch_jsons=args.batch_json,
            queue_dir=args.queue_dir,
            shadow_dir=args.shadow_dir,
            output_root=args.output_root,
        )
    )


if __name__ == "__main__":
    main()
