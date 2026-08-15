#!/usr/bin/env python3
"""Select SIRENE-only train seeds without creating CRM text.

The selector is deliberately limited to source selection and provenance. It
never normalizes, corrupts, or writes synthetic CRM fields; Luna remains the
only content writer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import duckdb


def canonical_siren(value: Any) -> str:
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    return digits[:9] if len(digits) >= 9 else ""


def canonical_siret(value: Any) -> str:
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    return digits if len(digits) == 14 else ""


def stable_key(siret: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}|{siret}".encode("ascii")).hexdigest()


def choose_one_per_siren(rows: Iterable[dict[str, Any]], excluded_sirens: set[str], seed: int, limit: int) -> list[dict[str, Any]]:
    best: dict[str, tuple[str, dict[str, Any]]] = {}
    for row in rows:
        siren = canonical_siren(row.get("siren"))
        siret = canonical_siret(row.get("siret"))
        if not siren or not siret or siren in excluded_sirens:
            continue
        state_rank = 0 if row.get("state") == "A" else 1
        identity_rank = 0 if any(str(row.get(key) or "").strip() for key in ("name", "enseigne", "street", "postcode", "city")) else 1
        key = (f"{state_rank}|{identity_rank}|{stable_key(siret, seed)}", row)
        if siren not in best or key[0] < best[siren][0]:
            best[siren] = key
    selected = [row for _, row in best.values()]
    selected.sort(key=lambda row: stable_key(canonical_siret(row["siret"]), seed))
    return selected[:limit]


def read_excluded_sirens(path: Path) -> set[str]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=";")
        return {canonical_siren(row.get("gt_siret")) for row in reader if canonical_siren(row.get("gt_siret"))}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--establishments", type=Path, required=True)
    parser.add_argument("--crm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-identifiable", action="store_true")
    args = parser.parse_args()
    if args.limit < 1:
        raise SystemExit("--limit must be positive")

    con = duckdb.connect()
    identity_filter = """
          AND (upper(trim(coalesce(denominationUsuelleEtablissement, ''))) NOT IN ('', '[ND]')
               OR upper(trim(coalesce(enseigne1Etablissement, ''))) NOT IN ('', '[ND]'))
          AND upper(trim(coalesce(numeroVoieEtablissement, ''))) NOT IN ('', '[ND]')
          AND upper(trim(coalesce(libelleVoieEtablissement, ''))) NOT IN ('', '[ND]')
          AND upper(trim(coalesce(codePostalEtablissement, ''))) NOT IN ('', '[ND]')
          AND upper(trim(coalesce(libelleCommuneEtablissement, ''))) NOT IN ('', '[ND]')
    """ if args.require_identifiable else ""
    query = f"""
        WITH crm_sirens AS (
            SELECT DISTINCT substr(regexp_replace(gt_siret, '[^0-9]', '', 'g'), 1, 9) AS siren
            FROM read_csv(?, delim=';', header=true, all_varchar=true)
            WHERE gt_siret IS NOT NULL
              AND length(regexp_replace(gt_siret, '[^0-9]', '', 'g')) = 14
        ), source AS (
        SELECT siren, siret,
               etatAdministratifEtablissement AS state,
               denominationUsuelleEtablissement AS name,
               enseigne1Etablissement AS enseigne,
               numeroVoieEtablissement AS street_number,
               typeVoieEtablissement AS street_type,
               libelleVoieEtablissement AS street,
               codePostalEtablissement AS postcode,
               libelleCommuneEtablissement AS city,
               codeCommuneEtablissement AS insee,
               etablissementSiege AS is_headquarters,
               dateCreationEtablissement AS creation_date,
               dateDebut AS start_date,
               row_number() OVER (
                   PARTITION BY siren
                   ORDER BY CASE WHEN etatAdministratifEtablissement = 'A' THEN 0 ELSE 1 END,
                            CASE WHEN coalesce(denominationUsuelleEtablissement, '') <> ''
                                      OR coalesce(enseigne1Etablissement, '') <> ''
                                      OR coalesce(libelleVoieEtablissement, '') <> ''
                                      OR coalesce(codePostalEtablissement, '') <> ''
                                      OR coalesce(libelleCommuneEtablissement, '') <> '' THEN 0 ELSE 1 END,
                            sha256(siret || ?)
               ) AS siren_rank
        FROM read_parquet(?)
        WHERE siret IS NOT NULL
          AND length(siret) = 14
          AND etatAdministratifEtablissement IN ('A', 'F')
          {identity_filter}
          AND siren NOT IN (SELECT siren FROM crm_sirens)
        )
        SELECT * EXCLUDE (siren_rank)
        FROM source
        WHERE siren_rank = 1
        ORDER BY sha256(siret || ?)
        LIMIT ?
    """
    result = con.execute(query, [str(args.crm), str(args.seed), str(args.establishments), str(args.seed), args.limit])
    columns = [item[0] for item in result.description]
    selected = [dict(zip(columns, values)) for values in result.fetchall()]
    excluded = read_excluded_sirens(args.crm)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for ordinal, row in enumerate(selected):
            record = {
                "schema_version": "sireto-synthetic-gt-sirene-seed-1",
                "seed_source": "SIRENE_ONLY_TRAIN",
                "selection_seed": args.seed,
                "selection_ordinal": ordinal,
                "source_siret": canonical_siret(row["siret"]),
                "source_siren": canonical_siren(row["siren"]),
                "official_fields": row,
            }
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":")) + "\n")
    manifest = {
        "schema_version": "sireto-synthetic-gt-sirene-seed-manifest-1",
        "source_kind": "SIRENE_ONLY_TRAIN",
        "selection_seed": args.seed,
        "requested_limit": args.limit,
        "selected_rows": len(selected),
        "selected_sirens": len({record["source_siren"] for record in ({"source_siren": canonical_siren(row["siren"])} for row in selected)}),
        "excluded_crm_sirens": len(excluded),
        "establishments_sha256": sha256(args.establishments),
        "crm_sha256": sha256(args.crm),
        "output_sha256": sha256(args.output),
        "text_generation": "none",
        "one_siret_per_siren": True,
        "require_identifiable": args.require_identifiable,
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
