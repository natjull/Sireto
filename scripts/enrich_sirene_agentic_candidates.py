#!/usr/bin/env python3
"""Attach official legal-unit names to SIRENE-only agentic seed cards.

The output contains official source text only.  No CRM field is generated,
corrupted, normalized, or repaired by this command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import duckdb


def clean(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text.upper() in {"", "[ND]", "NAN", "NONE"} else text


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def legal_names(values: dict[str, Any]) -> list[str]:
    person = " ".join(
        item for item in (
            clean(values.get("prenomUsuelUniteLegale")),
            clean(values.get("nomUniteLegale")),
        ) if item
    )
    ordered = [
        clean(values.get("denominationUsuelle1UniteLegale")),
        clean(values.get("denominationUsuelle2UniteLegale")),
        clean(values.get("denominationUsuelle3UniteLegale")),
        clean(values.get("denominationUniteLegale")),
        person,
        clean(values.get("sigleUniteLegale")),
    ]
    return list(dict.fromkeys(value for value in ordered if value))


def enrich_row(row: dict[str, Any], unit: dict[str, Any]) -> dict[str, Any] | None:
    fields = dict(row["official_fields"])
    options = legal_names(unit)
    existing = clean(fields.get("name"))
    enseigne = clean(fields.get("enseigne"))
    if existing:
        options.insert(0, existing)
    options = list(dict.fromkeys(value for value in options if value))
    if not options and enseigne:
        options = [enseigne]
    if not options:
        return None
    fields["name"] = options[0]
    required = (
        "street_number", "street_type", "street", "postcode", "city", "insee"
    )
    if any(not clean(fields.get(key)) for key in required):
        return None
    fields["legal_name_options"] = options
    result = dict(row)
    result["official_fields"] = fields
    result["selection_eligibility"] = {
        "has_name": True,
        "has_distinct_enseigne": bool(
            enseigne and all(enseigne.casefold() != value.casefold() for value in options)
        ),
        "has_full_address": True,
        "legal_unit_enriched": bool(not existing and options),
    }
    result["source_record_sha256"] = hashlib.sha256(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = [json.loads(line) for line in args.seeds.read_text(encoding="utf-8").splitlines() if line]
    sirens = sorted({str(row["source_siren"]) for row in rows})
    connection = duckdb.connect()
    connection.execute("CREATE TEMP TABLE wanted(siren VARCHAR)")
    connection.executemany("INSERT INTO wanted VALUES (?)", [(value,) for value in sirens])
    columns = [
        "denominationUsuelle1UniteLegale", "denominationUsuelle2UniteLegale",
        "denominationUsuelle3UniteLegale", "denominationUniteLegale",
        "prenomUsuelUniteLegale", "nomUniteLegale", "sigleUniteLegale",
    ]
    projection = ",".join(f"CAST(u.{name} AS VARCHAR) AS {name}" for name in columns)
    result = connection.execute(
        f"""SELECT CAST(u.siren AS VARCHAR) AS siren,{projection}
             FROM read_parquet(?) u JOIN wanted w ON CAST(u.siren AS VARCHAR)=w.siren""",
        [str(args.legal_units)],
    )
    names = [item[0] for item in result.description]
    units = {str(values[0]): dict(zip(names, values)) for values in result.fetchall()}
    enriched = [
        value for row in rows
        if (value := enrich_row(row, units.get(str(row["source_siren"]), {}))) is not None
    ]
    if args.limit:
        enriched = enriched[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for value in enriched:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    manifest = {
        "schema_version": "sireto-synthetic-gt-enriched-candidates-1",
        "input_rows": len(rows),
        "output_rows": len(enriched),
        "distinct_siret": len({value["source_siret"] for value in enriched}),
        "distinct_siren": len({value["source_siren"] for value in enriched}),
        "source_sha256": sha256(args.seeds),
        "legal_units_sha256": sha256(args.legal_units),
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
    result.add_argument("--seeds", type=Path, required=True)
    result.add_argument("--legal-units", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--limit", type=int, default=0)
    result.set_defaults(func=run)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
