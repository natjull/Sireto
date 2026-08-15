#!/usr/bin/env python3
"""Schedule evidence-backed agentic contracts without generating CRM text."""

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

from scripts.prepare_synthetic_gt_agentic_contracts import candidate_card  # noqa: E402
from scripts import run_synthetic_gt_agentic_loop as loop  # noqa: E402


NAME_FAMILIES = (
    "LEGAL_FORM", "ACRONYM_TOKENIZATION", "ACCENT_PUNCTUATION",
    "TOKEN_ORDER", "ENSEIGNE_VS_DENOMINATION", "OCR_LIMITED",
)
ADDRESS_FAMILIES = (
    "ADDRESS_ABBREVIATION", "COMMUNE_VARIANT", "ADDRESS_TOKEN_ORDER",
    "ADDRESS_OCR",
)
ORTHOGRAPHIC_FAMILIES = ("ACCENT_PUNCTUATION", "OCR_LIMITED")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()


def identifiers(path: Path) -> set[str]:
    values: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw:
            continue
        row = json.loads(raw)
        value = (
            row.get("target_siret") or row.get("source_siret")
            or row.get("seed", {}).get("siret")
        )
        if value:
            values.add(str(value))
    return values


def feasible_contracts(card: dict[str, Any]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for name in NAME_FAMILIES:
        for address in ADDRESS_FAMILIES:
            for orthographic in ORTHOGRAPHIC_FAMILIES:
                requested = {
                    "name": name, "address": address, "orthographic": orthographic
                }
                candidate = {**card, "requested_families": requested}
                if not all(
                    loop.source_supports_family(candidate, dimension, family)
                    for dimension, family in requested.items()
                ):
                    continue
                contracts = loop.variant_contract(candidate)
                if (
                    contracts[0]["requested_family"] == contracts[2]["requested_family"]
                    and contracts[0]["target_fields"] == contracts[2]["target_fields"]
                ):
                    continue
                result.append(requested)
    return result


def contract_score(
    contract: dict[str, str],
    counts: Counter[str],
    phenomena: dict[str, int],
    seed: int,
    siret: str,
) -> tuple[float, str]:
    pressure = sum(
        (counts[family] + 1) / max(int(phenomena.get(family, 0)), 1)
        for family in contract.values()
    )
    return pressure, stable_key(seed, siret + loop.canonical_json(contract))


def schedule(args: argparse.Namespace) -> dict[str, Any]:
    profile = loop.load_json(args.profile)
    excluded = set().union(*(identifiers(path) for path in args.exclude)) if args.exclude else set()
    rows = [
        json.loads(raw) for raw in args.candidates.read_text(encoding="utf-8").splitlines()
        if raw
    ]
    rows.sort(key=lambda row: stable_key(args.seed, str(row["source_siret"])))
    selections: list[dict[str, Any]] = []
    family_counts: Counter[str] = Counter()
    for row in rows:
        siret = str(row["source_siret"])
        if siret in excluded:
            continue
        card = candidate_card(row)
        card["ocr_substitution_pairs"] = profile.get("ocr_substitution_pairs", [])
        card["address_ocr_substitution_pairs"] = profile.get(
            "address_ocr_substitution_pairs", []
        )
        contracts = feasible_contracts(card)
        if not contracts:
            continue
        chosen = min(
            contracts,
            key=lambda value: contract_score(
                value, family_counts, profile["phenomena"], args.seed, siret
            ),
        )
        selections.append({"siret": siret, "requested_families": chosen})
        family_counts.update(chosen.values())
        if len(selections) == args.count:
            break
    if len(selections) != args.count:
        raise RuntimeError(
            f"only {len(selections)} evidence-backed contracts available; requested {args.count}"
        )
    output = {"selections": selections}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "sireto-synthetic-gt-contract-schedule-1",
        "count": len(selections),
        "distinct_siret": len({value["siret"] for value in selections}),
        "excluded_siret": len(excluded),
        "family_counts": dict(sorted(family_counts.items())),
        "candidates_sha256": sha256(args.candidates),
        "profile_sha256": sha256(args.profile),
        "output_sha256": sha256(args.output),
        "text_generation": "none",
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--candidates", type=Path, required=True)
    result.add_argument("--profile", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--count", type=int, required=True)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--exclude", type=Path, action="append", default=[])
    result.set_defaults(func=schedule)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
