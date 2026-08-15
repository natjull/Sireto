#!/usr/bin/env python3
"""Audit accepted synthetic CRM distribution without model outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


FIELDS = ("name", "address", "postcode", "city", "insee")


def fingerprint(crm: dict[str, Any]) -> str:
    return "|".join(" ".join(str(crm.get(field) or "").casefold().split()) for field in FIELDS)


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    field_presence = {
        field: sum(bool(str(row.get("crm", {}).get(field) or "").strip()) for row in rows)
        for field in FIELDS
    }
    lengths = {
        field: {
            "count": len(rows),
            "mean_chars": round(sum(len(str(row.get("crm", {}).get(field) or "")) for row in rows) / len(rows), 4) if rows else 0,
            "max_chars": max((len(str(row.get("crm", {}).get(field) or "")) for row in rows), default=0),
        }
        for field in FIELDS
    }
    families = Counter(family for row in rows for family in row.get("corruption_families_observed", []))
    seeds = Counter(str(row.get("target_siret") or row.get("seed", {}).get("siret", "")) for row in rows)
    return {
        "rows": len(rows),
        "unique_variant_fingerprints": len({fingerprint(row.get("crm", {})) for row in rows}),
        "exact_duplicate_rate": round(1 - len({fingerprint(row.get("crm", {})) for row in rows}) / len(rows), 6) if rows else 0,
        "field_presence": field_presence,
        "field_presence_rate": {key: round(value / len(rows), 6) if rows else 0 for key, value in field_presence.items()},
        "field_lengths": lengths,
        "families": dict(sorted(families.items())),
        "seeds": len(seeds),
        "variants_per_seed": dict(sorted(Counter(seeds.values()).items())),
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accept", type=Path, required=True)
    parser.add_argument("--observed-profile", type=Path)
    parser.add_argument("--observed-profile-request", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.accept.read_text(encoding="utf-8").splitlines() if line.strip()]
    if bool(args.observed_profile) == bool(args.observed_profile_request):
        raise SystemExit("provide exactly one observed profile input")
    profile_path = args.observed_profile or args.observed_profile_request
    if args.observed_profile:
        profile = json.loads(args.observed_profile.read_text(encoding="utf-8"))
    else:
        first = next(line for line in args.observed_profile_request.read_text(encoding="utf-8").splitlines() if line.strip())
        profile = json.loads(first)["observed_train_profile"]
    report = {
        "schema_version": "sireto-synthetic-gt-distribution-report-1",
        "accept_sha256": sha256(args.accept),
        "observed_profile_sha256": sha256(profile_path),
        "source_kind_counts": dict(sorted(Counter(row.get("source_kind", "") for row in rows).items())),
        "accepted": summary(rows),
        "observed_train_profile": profile,
        "retrieval_inputs_used": False,
        "text_generation": "none",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    report["output_sha256"] = sha256(args.output)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
