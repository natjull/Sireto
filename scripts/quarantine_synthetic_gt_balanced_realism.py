#!/usr/bin/env python3
"""Scan a sealed balanced corpus for bounded deterministic realism failures.

The scanner never edits CRM text and never reruns a model.  It reconstructs
each promoted row from its immutable seed contract, applies only the final
surface-quality gates shared with the Luna preflight, and writes an atomic
quarantine report suitable for a derived production registry.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import manage_synthetic_gt_balanced_registry as registry_lib
from scripts import run_synthetic_gt_agentic_loop as loop


SCHEMA_VERSION = "sireto-synthetic-gt-balanced-realism-quarantine-1"


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            json.dump(value, target, ensure_ascii=False, sort_keys=True, indent=2)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def scan(args: argparse.Namespace) -> dict[str, Any]:
    registry = registry_lib.load_registry(args.registry)
    prior_quarantine = registry_lib.quarantine_records(registry)
    snapshot = registry_lib.snapshot(registry)
    if registry.get("summary") != snapshot:
        raise ValueError("registry summary differs from sealed reconstruction")

    records: list[dict[str, Any]] = [dict(value) for value in prior_quarantine.values()]
    new_records: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    batches: Counter[str] = Counter()
    difficulty: Counter[str] = Counter()
    strata: Counter[str] = Counter()
    states: Counter[str] = Counter()
    scanned = 0
    for record in records:
        batches[str(record.get("batch_id", ""))] += 1
        difficulty[str(record.get("difficulty", ""))] += 1
        strata[str(record.get("augmentation_stratum", ""))] += 1
        states[str(record.get("target_state", ""))] += 1
        reasons.update(record.get("reason_codes", []))
    for batch in sorted(registry["batches"], key=lambda value: value["batch_id"]):
        seed_path = Path(batch["seed_input"]["path"])
        promoted_path = Path(batch["promoted"]["path"])
        manifest_path = Path(batch["promotion_manifest"]["path"])
        validated = registry_lib.validate_promoted_batch(
            seed_path, promoted_path, manifest_path,
        )
        seeds = {
            str(row["seed_id"]): row for row in registry_lib.jsonl(seed_path)
        }
        for value in validated["records"]:
            seed_id, variant_id = value["key"]
            if (seed_id, variant_id) in prior_quarantine:
                continue
            scanned += 1
            seed = seeds.get(seed_id)
            if seed is None:
                raise ValueError(f"missing frozen seed {seed_id}")
            seed_card = seed["seed_card"]
            baseline = loop.official_baseline(seed_card)
            contract = value["contract"]
            promoted = value["promoted"]
            row_errors = loop.final_surface_quality_errors(
                baseline, promoted["crm"], contract, seed_card,
            )
            if not row_errors:
                continue
            state = str(
                seed_card.get("official_context", {}).get("target", {}).get("state", "")
            )
            record = {
                "batch_id": str(batch["batch_id"]),
                "seed_id": seed_id,
                "variant_id": variant_id,
                "target_siret": value["target_siret"],
                "target_siren": value["target_siren"],
                "target_state": state,
                "difficulty": str(contract["difficulty"]),
                "augmentation_stratum": str(contract["augmentation_stratum"]),
                "field_relations": contract["field_relations"],
                "reason_codes": row_errors,
                "crm_sha256": loop.digest_json(promoted["crm"]),
                "variant_contract_sha256": loop.digest_json(contract),
                "quarantined": True,
            }
            records.append(record)
            new_records.append(record)
            batches[record["batch_id"]] += 1
            difficulty[record["difficulty"]] += 1
            strata[record["augmentation_stratum"]] += 1
            states[state] += 1
            reasons.update(row_errors)

    if scanned != int(snapshot["promoted_variants"]):
        raise ValueError("scanner row count differs from registry")
    records.sort(key=lambda value: (value["batch_id"], value["seed_id"], value["variant_id"]))
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "QUARANTINE_REQUIRED" if new_records else "PASS",
        "source_registry": {
            "path": str(args.registry.resolve()),
            "sha256": registry_lib.sha256(args.registry),
        },
        "scanner": {
            "path": str(Path(__file__).resolve()),
            "sha256": registry_lib.sha256(Path(__file__).resolve()),
            "validator_path": str(Path(loop.__file__).resolve()),
            "validator_sha256": registry_lib.sha256(Path(loop.__file__).resolve()),
            "text_mutation_by_scanner": False,
            "model_or_retrieval_used": False,
        },
        "prior_quarantine": {
            "excluded_rows": len(prior_quarantine),
            "report_sha256": registry.get("quarantine", {}).get("sha256"),
        },
        "scanned_rows": scanned,
        "new_quarantined_rows": len(new_records),
        "quarantined_rows": len(records),
        "counts": {
            "reason_codes": dict(sorted(reasons.items())),
            "batches": dict(sorted(batches.items())),
            "difficulty": dict(sorted(difficulty.items())),
            "augmentation_strata": dict(sorted(strata.items())),
            "target_states": dict(sorted(states.items())),
        },
        "review_trigger": {
            "sample_ids": sorted(set(args.reviewed_sample_id)),
            "certain_false_realism": len(set(args.reviewed_sample_id)),
        },
        "records": records,
    }
    atomic_json(args.output, report)
    print(json.dumps({
        "output": str(args.output),
        "scanned_rows": scanned,
        "new_quarantined_rows": len(new_records),
        "quarantined_rows": len(records),
        "counts": report["counts"],
    }, ensure_ascii=False, sort_keys=True, indent=2))
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--registry", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--reviewed-sample-id", action="append", default=[])
    result.set_defaults(func=scan)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
