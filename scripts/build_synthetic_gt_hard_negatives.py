#!/usr/bin/env python3
"""Build deterministic hard-negative pairs from immutable SIRENE candidate cards.

This module never edits or synthesizes CRM text. It only selects candidate
SIRETs, assigns an evidence family from source fields, and writes pair
records plus a non-self-referential manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


FAMILIES = (
    "SAME_SIREN_OTHER_SIRET",
    "SHARED_ADDRESS",
    "ACTIVE_CLOSED",
    "LOCAL_HOMONYM",
    "GEOGRAPHIC_NEIGHBOR",
)


def norm(value: Any) -> str:
    return " ".join(str(value or "").upper().split())


def name_keys(candidate: dict[str, Any]) -> set[str]:
    values = list(candidate.get("names") or [])
    values += [candidate.get("legal_denomination"), candidate.get("denomination_usuelle")]
    return {norm(value) for value in values if norm(value) and norm(value) != "[ND]"}


def evidence_family(seed: dict[str, Any], candidate: dict[str, Any]) -> str | None:
    if candidate["siret"] == seed["siret"]:
        return None
    if candidate.get("siren") == seed.get("siren"):
        return "SAME_SIREN_OTHER_SIRET"
    if norm(candidate.get("address_signature")) and norm(candidate.get("address_signature")) == norm(seed.get("address_signature")):
        return "SHARED_ADDRESS"
    if candidate.get("state") != seed.get("state") and {candidate.get("state"), seed.get("state")} == {"A", "F"}:
        return "ACTIVE_CLOSED"
    if candidate.get("insee") == seed.get("insee") and name_keys(seed) & name_keys(candidate):
        return "LOCAL_HOMONYM"
    if candidate.get("insee") == seed.get("insee") or candidate.get("postcode") == seed.get("postcode"):
        return "GEOGRAPHIC_NEIGHBOR"
    return None


def stable_key(seed_siret: str, candidate: dict[str, Any], family: str) -> str:
    raw = f"{seed_siret}|{family}|{candidate['siret']}".encode()
    return hashlib.sha256(raw).hexdigest()


def build(accept_rows: Iterable[dict[str, Any]], card_rows: Iterable[dict[str, Any]], per_positive: int = 10) -> list[dict[str, Any]]:
    cards = {row["siret"]: row for row in card_rows}
    pairs: list[dict[str, Any]] = []
    for accepted in accept_rows:
        seed_siret = str(accepted.get("target_siret") or accepted.get("seed", {}).get("siret", ""))
        if not seed_siret:
            continue
        card = cards.get(seed_siret)
        if not card:
            continue
        seed = next((candidate for candidate in card.get("candidates", []) if candidate.get("siret") == seed_siret), None)
        if not seed:
            continue
        chosen: list[tuple[str, dict[str, Any]]] = []
        seen: set[str] = set()
        candidates = card.get("candidates") or []
        for family in FAMILIES:
            ranked = sorted(
                ((stable_key(seed_siret, c, family), c) for c in candidates),
                key=lambda item: item[0],
            )
            for _, candidate in ranked:
                if candidate.get("siret") in seen:
                    continue
                if evidence_family(seed, candidate) != family:
                    continue
                seen.add(candidate["siret"])
                chosen.append((family, candidate))
                if len(chosen) >= per_positive:
                    break
            if len(chosen) >= per_positive:
                break
        for ordinal, (family, candidate) in enumerate(chosen, 1):
            pairs.append({
                "pair_id": hashlib.sha256(f"{seed_siret}|{accepted['variant_id']}|{candidate['siret']}".encode()).hexdigest(),
                "variant_id": accepted["variant_id"],
                "positive_siret": seed_siret,
                "negative_siret": candidate["siret"],
                "negative_siren": candidate.get("siren"),
                "family": family,
                "ordinal": ordinal,
                "source_kind": "SIRENE_SYNTHETIC_HARD_NEGATIVE",
                "source_card_siret": seed_siret,
                "source_candidate_fields": {
                    "siren": candidate.get("siren"),
                    "siret": candidate.get("siret"),
                    "state": candidate.get("state"),
                    "insee": candidate.get("insee"),
                    "postcode": candidate.get("postcode"),
                    "address_signature": candidate.get("address_signature"),
                },
            })
    return pairs


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accept", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-positive", type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.per_positive <= 10:
        raise SystemExit("--per-positive must be between 1 and 10")
    accepts = [json.loads(line) for line in args.accept.read_text().splitlines() if line.strip()]
    cards = [json.loads(line) for line in args.candidate_pool.read_text().splitlines() if line.strip()]
    pairs = build(accepts, cards, args.per_positive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for pair in pairs:
            stream.write(json.dumps(pair, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    counts = Counter(pair["family"] for pair in pairs)
    manifest = {
        "schema_version": "sireto-synthetic-gt-hard-negative-manifest-1",
        "source_kind": "SIRENE_SYNTHETIC_HARD_NEGATIVE",
        "accept_input_sha256": sha256(args.accept),
        "candidate_pool_sha256": sha256(args.candidate_pool),
        "output_sha256": sha256(args.output),
        "positive_rows": len(accepts),
        "pair_rows": len(pairs),
        "families": dict(sorted(counts.items())),
        "per_positive_cap": args.per_positive,
        "text_generation": "none",
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
