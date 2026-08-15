#!/usr/bin/env python3
"""Prepare agentic request envelopes from official SIRENE seed cards.

Only source fields are selected or concatenated into the official seed card;
this command never writes a synthetic CRM variant.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def text(value: object) -> str:
    value = str(value or "").strip()
    return "" if value.upper() in {"", "[ND]"} else value


def official_address(fields: dict) -> str:
    parts = [fields.get("street_number"), fields.get("street_type"), fields.get("street"), fields.get("postcode"), fields.get("city")]
    return " ".join(text(part) for part in parts if text(part))


def make_request(row: dict, profile: dict, ordinal: int) -> dict:
    fields = row["official_fields"]
    names = [text(fields.get("name")), text(fields.get("enseigne"))]
    names = list(dict.fromkeys(name for name in names if name))
    card = {
        "siret": row["source_siret"],
        "siren": row["source_siren"],
        "state": text(fields.get("state")),
        "name_options": names,
        "enseigne_options": [text(fields.get("enseigne"))] if text(fields.get("enseigne")) else [],
        "number": text(fields.get("street_number")),
        "street_type": text(fields.get("street_type")),
        "street": text(fields.get("street")),
        "address": official_address(fields),
        "postcode": text(fields.get("postcode")),
        "city": text(fields.get("city")),
        "insee": text(fields.get("insee")),
        "source_kind": "SIRENE_ONLY_TRAIN",
        "risk_flags": [],
    }
    for required, flag in (("name_options", "MISSING_NAME"), ("number", "MISSING_NUMBER"), ("postcode", "MISSING_POSTCODE"), ("city", "MISSING_CITY"), ("insee", "MISSING_INSEE")):
        value = card[required]
        if not value:
            card["risk_flags"].append(flag)
    return {
        "schema_version": "sireto-synthetic-gt-agentic-request-1",
        "batch_id": f"SIRENE-PILOT-{ordinal // 2:04d}",
        "model": "gpt-5.6-luna",
        "thinking": "low",
        "prompt_version": "sireto-gt-generator-v2",
        "seed": {"seed_source": "SIRENE_ONLY_TRAIN", "siret": row["source_siret"], "siren": row["source_siren"], "oof_fold": -1},
        "seed_card": card,
        "observed_train_profile": profile,
        "source_record_sha256": hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--profile-source-request", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()
    if args.offset < 0:
        raise SystemExit("--offset must be non-negative")
    rows = [json.loads(line) for line in args.seeds.read_text(encoding="utf-8").splitlines() if line.strip()][args.offset : args.offset + args.limit]
    if bool(args.profile) == bool(args.profile_source_request):
        raise SystemExit("provide exactly one of --profile or --profile-source-request")
    if args.profile:
        profile = json.loads(args.profile.read_text(encoding="utf-8"))
    else:
        first = next(line for line in args.profile_source_request.read_text(encoding="utf-8").splitlines() if line.strip())
        profile = json.loads(first)["observed_train_profile"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for ordinal, row in enumerate(rows):
            stream.write(json.dumps(make_request(row, profile, ordinal), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
