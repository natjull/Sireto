#!/usr/bin/env python3
"""Adapt the five traced V4.4 contradictions to canonical builder inputs.

Facts come exclusively from the archived V4.1 shadow evidence JSONL and its
manifest. Proofs and judgments come from the reviewed traceable JSON artifact.
The adapter does not infer labels, replacement SIRETs or candidate rows.
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
from typing import Any, Mapping

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_v44_adjudications import candidate_pool_sha256
from src.xgb_matcher.v9_dataset import file_sha256


SCHEMA_VERSION = "sireto-v4.4-known-contradictions-inputs-1"
EXPECTED_CASE_IDS = {
    "0107123ac3ab0732",
    "00ebcafaaa0a8bf5",
    "007d8c6b8f26962b",
    "003c6fdad046a903",
    "026ff9f27001bebd",
}
ALLOWED_LABELS = {"TOP1_CORRECT", "TOP1_WRONG", "AMBIGUOUS", "UNRESOLVED"}
SOURCE_MAPPING: dict[tuple[str, str], tuple[str, str, str]] = {
    ("REGISTRY_CORE_SIRENE", "INSEE, diffusé par la Direction interministérielle du numérique"): (
        "SIRENE_DERIVED_RECHERCHE_ENTREPRISES_API",
        "SIRENE_REGISTRY",
        "IDENTITY_REGISTRY",
    ),
    ("REGISTRY_CORE_SIRENE", "INSEE"): (
        "SIRENE_SNAPSHOT",
        "SIRENE_REGISTRY",
        "IDENTITY_REGISTRY",
    ),
    ("ENTITY_SELF_DECLARATION", "WELCOOP LOGISTIQUE"): (
        "OFFICIAL_ENTITY_SITE_WELCOOP",
        "WELCOOP_OFFICIAL",
        "IDENTITY_ENTITY_SITE",
    ),
    ("SECTOR_REGISTRY_EDUCATION", "Ministère de l'Éducation nationale"): (
        "OFFICIAL_EDUCATION_MINISTRY_DIRECTORY",
        "EDUCATION_MINISTRY",
        "IDENTITY_SECTOR_REGISTRY",
    ),
    ("ENTITY_SELF_PUBLICATION", "Commune de Merville-Franceville-Plage"): (
        "OFFICIAL_COMMUNE_SITE_MERVILLE_FRANCEVILLE",
        "MERVILLE_FRANCEVILLE_COMMUNE",
        "IDENTITY_ENTITY_SITE",
    ),
    (
        "SECTOR_REGISTRY_EDUCATION",
        "Office national d'information sur les enseignements et les professions",
    ): (
        "OFFICIAL_ONISEP_DIRECTORY",
        "ONISEP",
        "IDENTITY_SECTOR_REGISTRY",
    ),
    ("ENTITY_SELF_DECLARATION", "Centre Scolaire Saint Charles"): (
        "OFFICIAL_ENTITY_SITE_SAINT_CHARLES",
        "SAINT_CHARLES_OFFICIAL",
        "IDENTITY_ENTITY_SITE",
    ),
    (
        "PUBLIC_SERVICE_SELF_PUBLICATION",
        "Communauté d'agglomération Grand Paris Sud",
    ): (
        "OFFICIAL_PUBLIC_SERVICE_SITE_GRAND_PARIS_SUD",
        "GRAND_PARIS_SUD",
        "IDENTITY_PUBLIC_SERVICE",
    ),
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _canonical_json(values: list[str]) -> str:
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def _collected_at(value: Any) -> tuple[str, str]:
    """Map a timestamp honestly, retaining DAY precision when only a date exists."""

    text = str(value or "").strip()
    if not text:
        raise ValueError("Every proof requires collected_at")
    if len(text) == 10 and text[4] == "-" and text[7] == "-":
        timestamp = pd.Timestamp(f"{text}T00:00:00", tz="Europe/Paris")
        return timestamp.tz_convert("UTC").isoformat(), "DAY_EUROPE_PARIS"
    timestamp = pd.Timestamp(text)
    if timestamp.tzinfo is None:
        raise ValueError(f"Proof collected_at lacks timezone: {text}")
    return timestamp.tz_convert("UTC").isoformat(), "TIMESTAMP"


def _validate_shadow_manifest(shadow_dir: Path) -> dict[str, Any]:
    manifest = _read_json(shadow_dir / "manifest.json")
    if manifest.get("schema_version") != "sireto-shadow-v4.1-1":
        raise ValueError("Unsupported shadow manifest schema")
    evidence_path = shadow_dir / "evidence.jsonl"
    expected_hash = (manifest.get("outputs") or {}).get("evidence.jsonl")
    if not expected_hash or file_sha256(evidence_path) != expected_hash:
        raise ValueError("Shadow evidence.jsonl hash mismatch")
    metadata = manifest.get("run_metadata") or {}
    required = {
        "release_id",
        "ranker_bundle_id",
        "acceptor_bundle_id",
        "retrieval_signature",
    }
    missing = sorted(field for field in required if not str(metadata.get(field) or ""))
    if missing:
        raise ValueError(f"Shadow manifest missing frozen identifiers: {missing}")
    return manifest


def _load_shadow_cases(
    evidence_path: Path,
    *,
    service_ids: set[str],
) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    with evidence_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            record = json.loads(line)
            service_id = str(record.get("service_id") or "")
            if service_id not in service_ids:
                continue
            if service_id in found:
                raise ValueError(f"Duplicate shadow service_id: {service_id}")
            record["_line_number"] = line_number
            found[service_id] = record
    missing = sorted(service_ids - set(found))
    if missing:
        raise ValueError(f"Cases absent from shadow evidence JSONL: {missing}")
    return found


def adapt_inputs(
    *,
    evidence_artifact: Mapping[str, Any],
    shadow_manifest: Mapping[str, Any],
    shadow_cases: Mapping[str, Mapping[str, Any]],
    shadow_evidence_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cases = evidence_artifact.get("cases")
    if not isinstance(cases, list):
        raise ValueError("Evidence artifact cases must be a list")
    case_ids = {str(case.get("audit_case_id") or "") for case in cases}
    if case_ids != EXPECTED_CASE_IDS or len(cases) != len(EXPECTED_CASE_IDS):
        raise ValueError("Evidence artifact must contain exactly the five frozen cases")

    metadata = shadow_manifest["run_metadata"]
    facts: list[dict[str, Any]] = []
    proofs: list[dict[str, Any]] = []
    judgments: list[dict[str, Any]] = []
    seen_proof_ids: set[str] = set()

    for case in sorted(cases, key=lambda value: value["audit_case_id"]):
        case_id = str(case["audit_case_id"])
        service_id = str(case.get("service_id") or "")
        shadow = shadow_cases.get(service_id)
        if shadow is None:
            raise ValueError(f"Missing shadow case for service_id={service_id}")
        candidates = shadow.get("top_candidates")
        if not isinstance(candidates, list) or not 1 <= len(candidates) <= 10:
            raise ValueError(f"{case_id}: shadow top_candidates must contain 1 to 10 rows")
        ranks = [candidate.get("rank") for candidate in candidates]
        if ranks != list(range(1, len(candidates) + 1)):
            raise ValueError(f"{case_id}: shadow candidate ranks are not consecutive")
        sirets = [str(candidate.get("candidate_siret") or "") for candidate in candidates]
        if any(len(siret) != 14 or not siret.isdigit() for siret in sirets):
            raise ValueError(f"{case_id}: invalid archived candidate SIRET")
        if len(sirets) != len(set(sirets)):
            raise ValueError(f"{case_id}: duplicate archived candidate SIRET")

        frozen_top1 = str((case.get("frozen_top1") or {}).get("siret") or "")
        shadow_decision = shadow.get("decision") or {}
        if sirets[0] != frozen_top1:
            raise ValueError(f"{case_id}: audited top1 differs from shadow candidate rank 1")
        if str(shadow_decision.get("predicted_siret") or "") != frozen_top1:
            raise ValueError(f"{case_id}: audited top1 differs from shadow decision")

        facts.append(
            {
                "audit_case_id": case_id,
                "service_id": service_id,
                "query_id": service_id,
                "frozen_top1_siret": frozen_top1,
                "frozen_top1_siren": frozen_top1[:9],
                "frozen_model_bundle_id": str(metadata["release_id"]),
                "frozen_ranker_bundle_id": str(metadata["ranker_bundle_id"]),
                "frozen_acceptor_bundle_id": str(metadata["acceptor_bundle_id"]),
                "frozen_retrieval_signature": str(metadata["retrieval_signature"]),
                "frozen_candidate_sirets_json": _canonical_json(sirets),
                "frozen_candidate_pool_sha256": candidate_pool_sha256(sirets),
                "frozen_candidate_count": len(sirets),
                "frozen_candidate_source": "SHADOW_EVIDENCE_JSONL_TOP_CANDIDATES",
                "frozen_candidate_source_path": str(shadow_evidence_path),
                "frozen_candidate_source_line": int(shadow["_line_number"]),
                "positive_injection_by_adapter": False,
                "sampling_stratum": "",
                "priority_reason": "P0_KNOWN_PROVISIONAL_CONTRADICTION",
            }
        )

        label = str(case.get("adjudication_label") or "").upper()
        if label not in ALLOWED_LABELS:
            raise ValueError(f"{case_id}: unsupported decision {label}")
        if bool(case.get("evidence_validated")) != (label != "UNRESOLVED"):
            raise ValueError(f"{case_id}: source decision eligibility is inconsistent")

        proof_ids: list[str] = []
        for source in case.get("sources") or []:
            proof_id = str(source.get("evidence_id") or "")
            if not proof_id or proof_id in seen_proof_ids:
                raise ValueError(f"{case_id}: missing or duplicate evidence_id")
            seen_proof_ids.add(proof_id)
            mapping_key = (
                str(source.get("source_family") or ""),
                str(source.get("producer") or ""),
            )
            if mapping_key not in SOURCE_MAPPING:
                raise ValueError(f"{case_id}: unmapped proof source {mapping_key!r}")
            family, group, proof_kind = SOURCE_MAPPING[mapping_key]
            collected_at, precision = _collected_at(source.get("collected_at"))
            facts_list = source.get("archived_facts")
            if not isinstance(facts_list, list) or not facts_list:
                raise ValueError(f"{case_id}: proof lacks archived facts")
            locator = str(source.get("url") or "")
            if not locator.startswith("https://"):
                raise ValueError(f"{case_id}: proof lacks a public HTTPS locator")
            proofs.append(
                {
                    "proof_id": proof_id,
                    "audit_case_id": case_id,
                    "producer": str(source["producer"]),
                    "source_family": family,
                    "independence_group": group,
                    "source_locator": locator,
                    "collected_at": collected_at,
                    "collected_at_precision": precision,
                    "document_type": str(source.get("document_type") or ""),
                    "document_date": source.get("document_date"),
                    "archived_facts_json": json.dumps(
                        facts_list, ensure_ascii=False, separators=(",", ":")
                    ),
                    "local_artifact_path": source.get("artifact_path"),
                    "local_row_selector_json": json.dumps(
                        source.get("row_selector"),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    if source.get("row_selector") is not None
                    else None,
                    "proof_kind": proof_kind,
                    "supports_label": label,
                    "identity_consistent": True,
                    "contradiction_unresolved": label == "UNRESOLVED",
                }
            )
            proof_ids.append(proof_id)

        if label != "UNRESOLVED" and len(proof_ids) < 2:
            raise ValueError(f"{case_id}: validated judgment lacks two proof records")
        judgments.append(
            {
                "audit_case_id": case_id,
                "adjudication_label": label,
                "validated_correct_siret": case.get("validated_correct_siret"),
                "evidence_ref_ids_json": _canonical_json(proof_ids),
                "adjudication_reason": str(case.get("decision_reason") or ""),
                "adjudication_rule_version": "two-independent-identity-proofs-v1",
                "adjudicated_at": str(evidence_artifact.get("created_at") or ""),
            }
        )

    return pd.DataFrame(facts), pd.DataFrame(proofs), pd.DataFrame(judgments)


def build_input_artifact(
    *,
    evidence_json: Path,
    shadow_dir: Path,
    output_root: Path,
) -> Path:
    evidence_json = Path(evidence_json)
    shadow_dir = Path(shadow_dir)
    shadow_manifest_path = shadow_dir / "manifest.json"
    shadow_evidence_path = shadow_dir / "evidence.jsonl"
    shadow_manifest = _validate_shadow_manifest(shadow_dir)
    evidence_artifact = _read_json(evidence_json)
    service_ids = {
        str(case.get("service_id") or "")
        for case in evidence_artifact.get("cases") or []
    }
    if "" in service_ids:
        raise ValueError("Every evidence case requires service_id")
    shadow_cases = _load_shadow_cases(
        shadow_evidence_path,
        service_ids=service_ids,
    )
    facts, proofs, judgments = adapt_inputs(
        evidence_artifact=evidence_artifact,
        shadow_manifest=shadow_manifest,
        shadow_cases=shadow_cases,
        shadow_evidence_path=shadow_evidence_path,
    )

    identity = {
        "schema_version": SCHEMA_VERSION,
        "evidence_json_sha256": file_sha256(evidence_json),
        "shadow_manifest_sha256": file_sha256(shadow_manifest_path),
        "shadow_evidence_sha256": file_sha256(shadow_evidence_path),
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    target = Path(output_root) / build_id
    if target.exists():
        raise FileExistsError(f"Immutable adapter artifact exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{build_id}.tmp-", dir=target.parent))
    try:
        frames = {"facts": facts, "proofs": proofs, "judgments": judgments}
        output_hashes: dict[str, str] = {}
        for name, frame in frames.items():
            path = staging / f"{name}.parquet"
            frame.to_parquet(path, index=False)
            output_hashes[path.name] = file_sha256(path)
        manifest = {
            **identity,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "inputs": {
                "evidence_json": {
                    "path": str(evidence_json),
                    "sha256": identity["evidence_json_sha256"],
                },
                "shadow_manifest": {
                    "path": str(shadow_manifest_path),
                    "sha256": identity["shadow_manifest_sha256"],
                },
                "shadow_evidence": {
                    "path": str(shadow_evidence_path),
                    "sha256": identity["shadow_evidence_sha256"],
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
                name: int(len(frame)) for name, frame in frames.items()
            },
            "invariants": {
                "case_count": 5,
                "candidate_source": "evidence.jsonl.top_candidates",
                "candidate_cap": 10,
                "positive_injection_by_adapter": False,
                "labels_unchanged": True,
                "replacement_sirets_created": 0,
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
    path = Path(path)
    manifest = _read_json(path / "manifest.json")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported adapter artifact schema")
    for name, record in (manifest.get("inputs") or {}).items():
        source = Path(record.get("path") or "")
        if not source.is_file() or file_sha256(source) != record.get("sha256"):
            raise ValueError(f"Adapter input hash mismatch: {name}")
    for filename, expected_hash in (manifest.get("outputs") or {}).items():
        if file_sha256(path / filename) != expected_hash:
            raise ValueError(f"Adapter output hash mismatch: {filename}")

    shadow_dir = Path(manifest["inputs"]["shadow_manifest"]["path"]).parent
    shadow_manifest = _validate_shadow_manifest(shadow_dir)
    evidence_artifact = _read_json(
        Path(manifest["inputs"]["evidence_json"]["path"])
    )
    service_ids = {
        str(case["service_id"]) for case in evidence_artifact["cases"]
    }
    shadow_evidence_path = Path(manifest["inputs"]["shadow_evidence"]["path"])
    expected = adapt_inputs(
        evidence_artifact=evidence_artifact,
        shadow_manifest=shadow_manifest,
        shadow_cases=_load_shadow_cases(
            shadow_evidence_path,
            service_ids=service_ids,
        ),
        shadow_evidence_path=shadow_evidence_path,
    )
    for name, frame in zip(("facts", "proofs", "judgments"), expected):
        observed = pd.read_parquet(path / f"{name}.parquet")
        pd.testing.assert_frame_equal(observed, frame, check_dtype=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-json", type=Path)
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
        for name in ("evidence_json", "shadow_dir", "output_root")
        if getattr(args, name) is None
    ]
    if missing:
        raise SystemExit(
            "Building requires: "
            + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        )
    print(
        build_input_artifact(
            evidence_json=args.evidence_json,
            shadow_dir=args.shadow_dir,
            output_root=args.output_root,
        )
    )


if __name__ == "__main__":
    main()
