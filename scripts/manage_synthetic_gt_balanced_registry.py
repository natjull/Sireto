#!/usr/bin/env python3
"""Maintain the crash-safe registry of counted balanced synthetic GT.

Only variants present in a sealed full-SIRENE promotion are counted.  A seed
input is retained alongside the promotion so cumulative inspiration, operator
and relation-pair caps can be reconstructed without trusting a mutable summary.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence


SCHEMA_VERSION = "sireto-synthetic-gt-balanced-registry-1"
QUARANTINE_SCHEMA_VERSION = "sireto-synthetic-gt-balanced-realism-quarantine-1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL row: {path}:{line_number}")
            result.append(value)
    return result


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            json.dump(value, target, ensure_ascii=False, sort_keys=True, indent=2)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def empty_registry(target: int) -> dict[str, Any]:
    if target <= 0:
        raise ValueError("target must be positive")
    result = {
        "schema_version": SCHEMA_VERSION,
        "promoted_variant_target": target,
        "batches": [],
        "summary": {},
    }
    return result


def load_registry(path: Path, *, target: int | None = None) -> dict[str, Any]:
    if not path.exists():
        if target is None:
            raise FileNotFoundError(path)
        return empty_registry(target)
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported production registry schema: {path}")
    if not isinstance(value.get("batches"), list):
        raise ValueError(f"invalid production registry batches: {path}")
    if target is not None and int(value["promoted_variant_target"]) != target:
        raise ValueError("registry target differs from requested target")
    return value


def contract_index(seed_input: Path) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in jsonl(seed_input):
        seed_id = str(row.get("seed_id", ""))
        for contract in row.get("seed_card", {}).get("composite_contracts", []):
            key = (seed_id, str(contract.get("variant_id", "")))
            if not all(key) or key in result:
                raise ValueError(f"duplicate/invalid seed contract key in {seed_input}: {key}")
            result[key] = {
                "contract": contract,
                "target_siret": str(row.get("target_siret", "")),
                "target_siren": str(row.get("target_siren", "")),
            }
    return result


def quarantine_records(registry: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    """Load and verify the optional immutable realism-quarantine overlay."""
    descriptor = registry.get("quarantine")
    if descriptor is None:
        return {}
    if not isinstance(descriptor, dict):
        raise ValueError("invalid registry quarantine descriptor")
    path = Path(str(descriptor.get("path", "")))
    if not path.is_file() or sha256(path) != descriptor.get("sha256"):
        raise ValueError("registered quarantine report changed")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema_version") != QUARANTINE_SCHEMA_VERSION:
        raise ValueError("unsupported realism quarantine report")
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for value in report.get("records", []):
        key = (str(value.get("seed_id", "")), str(value.get("variant_id", "")))
        reasons = value.get("reason_codes")
        if (
            not all(key)
            or key in records
            or value.get("quarantined") is not True
            or not isinstance(reasons, list)
            or not reasons
            or not all(isinstance(reason, str) and reason for reason in reasons)
        ):
            raise ValueError(f"invalid or duplicate quarantine record: {key}")
        records[key] = value
    if (
        len(records) != int(report.get("quarantined_rows", -1))
        or len(records) != int(descriptor.get("rows", -1))
    ):
        raise ValueError("quarantine record count differs from descriptor")
    return records


def validate_promoted_batch(
    seed_input: Path,
    promoted: Path,
    promotion_manifest: Path,
) -> dict[str, Any]:
    manifest = json.loads(promotion_manifest.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != "sireto-synthetic-gt-full-exact-promotion-2"
        or manifest.get("exact_witness") != "G_N_A"
        or manifest.get("positive_injection") is not False
        or manifest.get("qualification_uses_retrieval_or_model_scores") is not False
        or manifest.get("promotion_mode") != "per-variant"
    ):
        raise ValueError(f"promotion manifest is not a sealed exact per-variant artifact: {promotion_manifest}")
    hashes = manifest.get("source_hashes", {})
    if hashes.get("seed_input") != sha256(seed_input):
        raise ValueError("promotion manifest seed-input hash mismatch")
    if manifest.get("promoted_sha256") != sha256(promoted):
        raise ValueError("promotion manifest promoted hash mismatch")

    contracts = contract_index(seed_input)
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    surfaces: set[tuple[str, str, str, str, str]] = set()
    for row in jsonl(promoted):
        key = (str(row.get("seed_id", "")), str(row.get("variant_id", "")))
        if key in seen or key not in contracts:
            raise ValueError(f"promoted key is duplicate or absent from frozen seed input: {key}")
        qualification = row.get("full_sirene_qualification", {})
        if (
            row.get("final_decision") != "ACCEPT"
            or qualification.get("decision") != "EXACT_IDENTIFIABLE"
            or qualification.get("exact_witness") != "G_N_A"
            or qualification.get("target_naturally_returned") is not True
            or qualification.get("candidate_sirets", {}).get("G_N_A")
            != [row.get("target_siret")]
        ):
            raise ValueError(f"non-exact row in promoted artifact: {key}")
        frozen = contracts[key]
        if (
            row.get("target_siret") != frozen["target_siret"]
            or row.get("target_siren") != frozen["target_siren"]
        ):
            raise ValueError(f"promoted identity differs from frozen seed input: {key}")
        crm = row.get("crm", {})
        surface = tuple(str(crm.get(field, "")) for field in (
            "name", "address", "postcode", "city", "insee"
        ))
        if surface in surfaces:
            raise ValueError(f"duplicate promoted CRM surface across counted batch: {key}")
        seen.add(key)
        surfaces.add(surface)
        records.append({**frozen, "key": key, "promoted": row})
    if len(records) != int(manifest.get("promoted_variants", -1)):
        raise ValueError("promoted row count differs from promotion manifest")
    result = {
        "records": records,
        "seed_input_sha256": sha256(seed_input),
        "promoted_sha256": sha256(promoted),
        "promotion_manifest_sha256": sha256(promotion_manifest),
        "run_id": manifest.get("run_id"),
    }
    return result


def pair_signature(relations: dict[str, str]) -> str:
    location = [(field, relation) for field, relation in relations.items() if field != "name"]
    if "name" not in relations or len(location) != 1:
        raise ValueError(f"invalid counted field relations: {relations}")
    field, relation = location[0]
    return f"name:{relations['name']}+{field}:{relation}"


def fragment_operator(fragment: dict[str, Any]) -> str:
    payload = {
        "field": fragment["field"],
        "relation": fragment["relation"],
        "parameters": fragment["operation_parameters"],
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def token_subset_signature(fragment: dict[str, Any]) -> str | None:
    if fragment.get("field") != "name" or fragment.get("relation") != "TOKEN_SUBSET":
        return None
    parameters = fragment.get("operation_parameters", {})
    return json.dumps(
        [parameters.get("source_token_count"), parameters.get("retained_positions")],
        separators=(",", ":"),
    )


def snapshot(registry: dict[str, Any]) -> dict[str, Any]:
    quarantined = quarantine_records(registry)
    ref_counts: Counter[str] = Counter()
    operator_counts: Counter[str] = Counter()
    pair_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()
    stratum_counts: Counter[str] = Counter()
    token_subset_signature_counts: Counter[str] = Counter()
    target_sirets: set[str] = set()
    target_sirens: set[str] = set()
    excluded_target_sirets: set[str] = set()
    excluded_target_sirens: set[str] = set()
    keys: set[tuple[str, str]] = set()
    surfaces: set[tuple[str, str, str, str, str]] = set()
    batch_counts: dict[str, int] = {}
    raw_keys: set[tuple[str, str]] = set()
    raw_surfaces: set[tuple[str, str, str, str, str]] = set()
    quarantine_reason_counts: Counter[str] = Counter()
    for batch in registry["batches"]:
        seed_input = Path(batch["seed_input"]["path"])
        promoted = Path(batch["promoted"]["path"])
        manifest = Path(batch["promotion_manifest"]["path"])
        for descriptor, path in (("seed_input", seed_input), ("promoted", promoted),
                                 ("promotion_manifest", manifest)):
            if sha256(path) != batch[descriptor]["sha256"]:
                raise ValueError(f"registered artifact changed: {path}")
        validated = validate_promoted_batch(seed_input, promoted, manifest)
        frozen_contracts = contract_index(seed_input)
        excluded_target_sirets.update(
            value["target_siret"] for value in frozen_contracts.values()
        )
        excluded_target_sirens.update(
            value["target_siren"] for value in frozen_contracts.values()
        )
        if len(validated["records"]) != int(batch["promoted_variants"]):
            raise ValueError(f"registered batch count changed: {batch['batch_id']}")
        safe_batch_count = 0
        for value in validated["records"]:
            key = value["key"]
            crm = value["promoted"]["crm"]
            surface = tuple(str(crm.get(field, "")) for field in (
                "name", "address", "postcode", "city", "insee"
            ))
            if key in raw_keys or surface in raw_surfaces:
                raise ValueError(f"duplicate raw counted key or surface across batches: {key}")
            raw_keys.add(key)
            raw_surfaces.add(surface)
            if key in quarantined:
                quarantine_reason_counts.update(quarantined[key]["reason_codes"])
                continue
            if key in keys or surface in surfaces:
                raise ValueError(f"duplicate counted key or surface across batches: {key}")
            keys.add(key)
            surfaces.add(surface)
            safe_batch_count += 1
            target_sirets.add(value["target_siret"])
            target_sirens.add(value["target_siren"])
            contract = value["contract"]
            difficulty_counts[str(contract["difficulty"])] += 1
            stratum_counts[str(contract["augmentation_stratum"])] += 1
            pair_counts[pair_signature(contract["field_relations"])] += 1
            for fragment in contract.get("field_inspirations", {}).values():
                ref_counts[str(fragment["inspiration_ref"])] += 1
                operator_counts[fragment_operator(fragment)] += 1
                signature = token_subset_signature(fragment)
                if signature is not None:
                    token_subset_signature_counts[signature] += 1
        batch_counts[batch["batch_id"]] = safe_batch_count
    missing_quarantine = set(quarantined) - raw_keys
    if missing_quarantine:
        raise ValueError(
            f"quarantine keys are absent from sealed batches: {sorted(missing_quarantine)[:5]}"
        )
    promoted_variants = len(keys)
    target = int(registry["promoted_variant_target"])
    result = {
        "promoted_variants": promoted_variants,
        "remaining_variants": max(0, target - promoted_variants),
        "completion_ratio": promoted_variants / target,
        "batch_counts": batch_counts,
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "augmentation_stratum_counts": dict(sorted(stratum_counts.items())),
        "distinct_target_sirets": len(target_sirets),
        "distinct_target_sirens": len(target_sirens),
        "promoted_target_sirets": sorted(target_sirets),
        "promoted_target_sirens": sorted(target_sirens),
        "excluded_target_sirets": sorted(excluded_target_sirets),
        "excluded_target_sirens": sorted(excluded_target_sirens),
        "inspiration_ref_counts": dict(sorted(ref_counts.items())),
        "exact_operator_counts": dict(sorted(operator_counts.items())),
        "relation_pair_counts": dict(sorted(pair_counts.items())),
        "name_token_subset_signature_counts": dict(
            sorted(token_subset_signature_counts.items())
        ),
    }
    if quarantined:
        result.update({
            "quarantined_variants": len(quarantined),
            "quarantine_reason_counts": dict(sorted(quarantine_reason_counts.items())),
            "quarantine_report_sha256": str(registry["quarantine"]["sha256"]),
        })
    return result


def register(args: argparse.Namespace) -> dict[str, Any]:
    registry = load_registry(args.registry, target=args.target)
    validated = validate_promoted_batch(args.seed_input, args.promoted, args.promotion_manifest)
    entry = {
        "batch_id": args.batch_id,
        "run_id": validated["run_id"],
        "promoted_variants": len(validated["records"]),
        "seed_input": {"path": str(args.seed_input.resolve()), "sha256": validated["seed_input_sha256"]},
        "promoted": {"path": str(args.promoted.resolve()), "sha256": validated["promoted_sha256"]},
        "promotion_manifest": {
            "path": str(args.promotion_manifest.resolve()),
            "sha256": validated["promotion_manifest_sha256"],
        },
    }
    existing = [value for value in registry["batches"] if value["batch_id"] == args.batch_id]
    if existing:
        if existing != [entry]:
            raise ValueError(f"batch id already registered with different artifacts: {args.batch_id}")
    else:
        registry["batches"].append(entry)
        registry["batches"].sort(key=lambda value: value["batch_id"])
    registry["summary"] = snapshot(registry)
    atomic_json(args.registry, registry)
    print(json.dumps(registry["summary"], ensure_ascii=False, sort_keys=True, indent=2))
    return registry


def status(args: argparse.Namespace) -> dict[str, Any]:
    registry = load_registry(args.registry)
    current = snapshot(registry)
    if registry.get("summary") != current:
        raise ValueError("registry summary differs from reconstructed sealed artifacts")
    print(json.dumps(current, ensure_ascii=False, sort_keys=True, indent=2))
    return current


def quarantine(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_registry.exists():
        raise FileExistsError(args.output_registry)
    registry = load_registry(args.source_registry)
    if registry.get("quarantine"):
        raise ValueError("source registry already has a quarantine overlay")
    if registry.get("summary") != snapshot(registry):
        raise ValueError("source registry summary differs from sealed reconstruction")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report.get("schema_version") != QUARANTINE_SCHEMA_VERSION:
        raise ValueError("unsupported realism quarantine report")
    source = report.get("source_registry", {})
    if source.get("sha256") != sha256(args.source_registry):
        raise ValueError("quarantine report is not bound to the source registry")
    registry["quarantine"] = {
        "path": str(args.report.resolve()),
        "sha256": sha256(args.report),
        "rows": int(report.get("quarantined_rows", -1)),
        "source_registry": {
            "path": str(args.source_registry.resolve()),
            "sha256": sha256(args.source_registry),
        },
        "text_mutation": False,
    }
    registry["summary"] = snapshot(registry)
    atomic_json(args.output_registry, registry)
    print(json.dumps(registry["summary"], ensure_ascii=False, sort_keys=True, indent=2))
    return registry


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    register_parser = subparsers.add_parser("register")
    register_parser.add_argument("--registry", type=Path, required=True)
    register_parser.add_argument("--target", type=int, default=20_000)
    register_parser.add_argument("--batch-id", required=True)
    register_parser.add_argument("--seed-input", type=Path, required=True)
    register_parser.add_argument("--promoted", type=Path, required=True)
    register_parser.add_argument("--promotion-manifest", type=Path, required=True)
    register_parser.set_defaults(func=register)
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--registry", type=Path, required=True)
    status_parser.set_defaults(func=status)
    quarantine_parser = subparsers.add_parser("quarantine")
    quarantine_parser.add_argument("--source-registry", type=Path, required=True)
    quarantine_parser.add_argument("--report", type=Path, required=True)
    quarantine_parser.add_argument("--output-registry", type=Path, required=True)
    quarantine_parser.set_defaults(func=quarantine)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
