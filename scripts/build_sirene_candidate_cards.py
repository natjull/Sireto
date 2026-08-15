#!/usr/bin/env python3
"""Build immutable SIRENE candidate cards for hard-negative selection.

This command only projects official SIRENE identity fields. It never writes or
alters CRM text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb


def value(raw: object) -> str:
    text = str(raw or "").strip()
    return "" if text.upper() == "[ND]" else text


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accept", type=Path, required=True)
    parser.add_argument("--establishments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    query = """
        WITH seeds AS (
            SELECT DISTINCT target_siret, target_siren
            FROM read_json_auto(?, format='newline_delimited')
        ), seed_facts AS (
            SELECT seeds.target_siret, seeds.target_siren,
                   e.codeCommuneEtablissement AS target_insee,
                   e.codePostalEtablissement AS target_postcode
            FROM seeds JOIN read_parquet(?) e ON e.siret = seeds.target_siret
        )
        SELECT seed_facts.target_siret,
               e.siren, e.siret, e.etatAdministratifEtablissement AS state,
               e.denominationUsuelleEtablissement AS denomination_usuelle,
               e.enseigne1Etablissement AS enseigne,
               e.numeroVoieEtablissement AS number,
               e.typeVoieEtablissement AS street_type,
               e.libelleVoieEtablissement AS street,
               e.codePostalEtablissement AS postcode,
               e.libelleCommuneEtablissement AS city,
               e.codeCommuneEtablissement AS insee
        FROM seed_facts JOIN read_parquet(?) e
          ON e.siren = seed_facts.target_siren
          OR e.codeCommuneEtablissement = seed_facts.target_insee
          OR e.codePostalEtablissement = seed_facts.target_postcode
        WHERE e.siret IS NOT NULL AND length(e.siret)=14
          AND e.etatAdministratifEtablissement IN ('A','F')
        ORDER BY seed_facts.target_siret, e.siret
    """
    con = duckdb.connect()
    rows = con.execute(query, [str(args.accept), str(args.establishments), str(args.establishments)]).fetchall()
    columns = [item[0] for item in con.description]
    cards: dict[str, dict] = {}
    for raw in rows:
        row = dict(zip(columns, raw))
        target = value(row.pop("target_siret"))
        names = [value(row.get("denomination_usuelle")), value(row.get("enseigne"))]
        names = list(dict.fromkeys(name for name in names if name))
        address = " ".join(value(row.get(key)) for key in ("number", "street_type", "street", "postcode", "city") if value(row.get(key)))
        candidate = {
            "siren": value(row.get("siren")), "siret": value(row.get("siret")),
            "state": value(row.get("state")), "insee": value(row.get("insee")),
            "postcode": value(row.get("postcode")), "names": names,
            "legal_denomination": "", "denomination_usuelle": value(row.get("denomination_usuelle")),
            "address_signature": address,
        }
        cards.setdefault(target, {"siret": target, "candidates": []})["candidates"].append(candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for target in sorted(cards):
            stream.write(json.dumps(cards[target], ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    manifest = {
        "schema_version": "sireto-synthetic-gt-sirene-candidate-card-manifest-1",
        "source_kind": "SIRENE_OFFICIAL_CANDIDATE_CARD",
        "accept_sha256": sha256(args.accept),
        "establishments_sha256": sha256(args.establishments),
        "seed_cards": len(cards),
        "candidate_rows": sum(len(card["candidates"]) for card in cards.values()),
        "text_generation": "none",
        "output_sha256": sha256(args.output),
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
