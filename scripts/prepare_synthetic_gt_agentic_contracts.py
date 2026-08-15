#!/usr/bin/env python3
"""Build official seed cards from Luna-selected, evidence-backed contracts.

Luna selects SIRET and corruption families.  This command only copies official
source fields, attaches the immutable observed-train profile, and validates the
result with the runtime.  It never generates or transforms CRM text.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_synthetic_gt_agentic_loop as loop  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def official_text(value: Any) -> str:
    return str(value or "").strip()


def candidate_card(candidate: dict[str, Any]) -> dict[str, Any]:
    fields = candidate["official_fields"]
    legal_name = official_text(fields.get("name"))
    enseigne = official_text(fields.get("enseigne"))
    baseline_name = legal_name or enseigne
    address = " ".join(
        value for value in (
            official_text(fields.get("street_number")),
            official_text(fields.get("street_type")),
            official_text(fields.get("street")),
        ) if value
    )
    return {
        "siret": official_text(candidate["source_siret"]),
        "siren": official_text(candidate["source_siren"]),
        "state": official_text(fields.get("state")),
        "name_options": [baseline_name] if baseline_name else [],
        "enseigne_options": [enseigne] if enseigne else [],
        "number": official_text(fields.get("street_number")),
        "street_type": official_text(fields.get("street_type")),
        "street": official_text(fields.get("street")),
        "address": address,
        "postcode": official_text(fields.get("postcode")),
        "city": official_text(fields.get("city")),
        "insee": official_text(fields.get("insee")),
        "source_kind": "SIRENE_ONLY_TRAIN",
        "field_missing_forbidden": True,
        "risk_flags": [],
    }


def load_candidates(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for _raw, candidate in loop.iter_jsonl_raw(path):
        siret = official_text(candidate.get("source_siret"))
        siren = official_text(candidate.get("source_siren"))
        if not loop.valid_siret(siret) or not loop.valid_siren(siren) or siret[:9] != siren:
            raise ValueError(f"invalid official candidate identity: {siret}/{siren}")
        if siret in result:
            raise ValueError(f"duplicate official candidate: {siret}")
        result[siret] = candidate
    return result


def prepare(args: argparse.Namespace) -> None:
    candidates = load_candidates(args.candidates)
    assignments = loop.load_json(args.assignments)
    selections = assignments.get("selections") if isinstance(assignments, dict) else None
    if not isinstance(selections, list) or len(selections) != args.count:
        raise ValueError(f"assignments must contain exactly {args.count} selections")
    sirets = [official_text(value.get("siret")) for value in selections]
    if len(set(sirets)) != len(sirets):
        raise ValueError("duplicate SIRET in Luna contract selections")
    unknown = sorted(set(sirets) - set(candidates))
    if unknown:
        raise ValueError(f"Luna selected non-official candidates: {unknown[:3]}")
    profile = loop.load_json(args.profile)
    rows: list[dict[str, Any]] = []
    family_counts: Counter[str] = Counter()
    for selection in selections:
        siret = official_text(selection["siret"])
        candidate = candidates[siret]
        card = candidate_card(candidate)
        card["requested_families"] = selection["requested_families"]
        row = {
            "seed_id": f"{args.seed_prefix}:{siret}",
            "target_siret": siret,
            "target_siren": official_text(candidate["source_siren"]),
            "source_kind": "SIRENE_ONLY_TRAIN",
            "oof_fold": -1,
            "legacy_split": "train_synthetic",
            "seed_card": card,
            "observed_train_profile": profile,
            "risk_flags": [args.risk_flag],
        }
        rows.append(loop.validate_seed(row))
        family_counts.update(selection["requested_families"].values())
    loop.write_jsonl_atomic(args.output, rows)
    manifest = {
        "schema_version": "sireto-synthetic-gt-agentic-contract-build-1",
        "seed_count": len(rows),
        "distinct_siret": len({row["target_siret"] for row in rows}),
        "distinct_siren": len({row["target_siren"] for row in rows}),
        "family_counts": dict(sorted(family_counts.items())),
        "inputs": {
            "official_candidates_sha256": sha256(args.candidates),
            "luna_assignments_sha256": sha256(args.assignments),
            "observed_train_profile_sha256": sha256(args.profile),
        },
        "output_sha256": sha256(args.output),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--candidates", type=Path, required=True)
    result.add_argument("--assignments", type=Path, required=True)
    result.add_argument("--profile", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--count", type=int, default=32)
    result.add_argument("--seed-prefix", default="agentic")
    result.add_argument("--risk-flag", default="AGENTIC_CONTRACT")
    result.set_defaults(func=prepare)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
