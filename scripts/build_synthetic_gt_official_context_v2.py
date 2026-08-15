#!/usr/bin/env python3
"""Build complete official sibling/site/name context for SIRENE-only seeds.

The builder reads only pinned local SIRENE and CRM sources.  It does not call
retrieval, Maps, or any model and never writes synthetic CRM text.  Protected
CRM SIRENs participate in fail-closed ambiguity gates but their text is never
included in the bounded view intended for Luna.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import prepare_synthetic_gt_agentic_contracts as contracts  # noqa: E402
from scripts import run_synthetic_gt_agentic_loop as loop  # noqa: E402


def clean(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text.upper() in {"", "[ND]", "NAN", "NONE"} else text


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_crm_sirens(path: Path) -> set[str]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=";")
        result: set[str] = set()
        for row in reader:
            digits = "".join(character for character in str(row.get("gt_siret") or "") if character.isdigit())
            if len(digits) == 14:
                result.add(digits[:9])
        return result


def read_seed_sirets(path: Path) -> list[str]:
    result: list[str] = []
    for _raw, value in loop.iter_jsonl_raw(path):
        siret = clean(value.get("target_siret"))
        if not loop.valid_siret(siret):
            raise ValueError(f"invalid seed SIRET: {siret!r}")
        result.append(siret)
    if len(result) != len(set(result)):
        raise ValueError("duplicate SIRET in seed input")
    return result


def candidate_rows(path: Path, wanted_sirets: set[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for _raw, value in loop.iter_jsonl_raw(path):
        siret = clean(value.get("source_siret"))
        if siret in wanted_sirets:
            result[siret] = value
    missing = sorted(wanted_sirets - set(result))
    if missing:
        raise ValueError(f"seed missing from official candidates: {missing[:3]}")
    return result


def target_frame(candidates: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for siret, candidate in candidates.items():
        fields = candidate["official_fields"]
        card = contracts.candidate_card(candidate)
        name_values = list(dict.fromkeys(
            clean(value)
            for value in [*card["name_options"], *card["enseigne_options"]]
            if clean(value)
        ))
        rows.append({
            "target_siret": siret,
            "target_siren": clean(candidate["source_siren"]),
            "target_state": clean(fields.get("state")),
            "number_key": loop.normalized_alnum(fields.get("street_number")),
            "index_key": loop.normalized_alnum(fields.get("repetition_index")),
            "street_type_key": loop.normalized_alnum(canonical_street_type(fields.get("street_type"))),
            "street_key": loop.normalized_alnum(fields.get("street")),
            "postcode_key": loop.normalized_alnum(fields.get("postcode")),
            "insee_key": loop.normalized_alnum(fields.get("insee")),
            "city_key": loop.normalized_alnum(fields.get("city")),
            "name_values_json": json.dumps(name_values, ensure_ascii=False),
        })
    return pd.DataFrame(rows)


def name_frame(targets: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for value in targets.to_dict("records"):
        for name in json.loads(value["name_values_json"]):
            key = loop.normalized_alnum(name)
            if key:
                rows.append({
                    "target_siret": value["target_siret"],
                    "insee_key": value["insee_key"],
                    "name_key": key,
                })
    return pd.DataFrame(rows).drop_duplicates()


PROJECTION = """
    e.siret AS siret,
    e.siren AS siren,
    e.etatAdministratifEtablissement AS state,
    e.etablissementSiege AS is_headquarters,
    e.denominationUsuelleEtablissement AS usual_name,
    e.enseigne1Etablissement AS enseigne1,
    e.enseigne2Etablissement AS enseigne2,
    e.enseigne3Etablissement AS enseigne3,
    e.numeroVoieEtablissement AS number,
    e.indiceRepetitionEtablissement AS repetition_index,
    e.typeVoieEtablissement AS street_type,
    e.libelleVoieEtablissement AS street,
    e.codePostalEtablissement AS postcode,
    e.libelleCommuneEtablissement AS city,
    e.codeCommuneEtablissement AS insee
"""


def canonical_street_type(value: Any) -> str:
    normalized = loop.normalized_surface(value).upper()
    return loop.STREET_TYPE_ABBREVIATIONS.get(normalized, normalized)


def query_context(
    establishments: Path,
    legal_units: Path,
    targets: pd.DataFrame,
    names: pd.DataFrame,
    temp_directory: Path,
) -> dict[str, list[dict[str, Any]]]:
    temp_directory.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.execute("SET temp_directory = ?", [str(temp_directory)])
    connection.execute("SET max_temp_directory_size = '100GiB'")
    connection.execute("SET memory_limit = '12GB'")
    connection.execute("SET threads = 8")
    connection.execute("SET preserve_insertion_order = false")
    connection.register("targets", targets)
    connection.register("target_names", names)
    connection.execute("""
        CREATE OR REPLACE MACRO norm(value) AS
        regexp_replace(lower(strip_accents(coalesce(value, ''))), '[^a-z0-9]', '', 'g')
    """)
    connection.execute("""
        CREATE OR REPLACE MACRO street_type_norm(value) AS
        CASE norm(value)
          WHEN 'r' THEN 'rue'
          WHEN 'av' THEN 'avenue'
          WHEN 'bd' THEN 'boulevard'
          WHEN 'ch' THEN 'chemin'
          WHEN 'che' THEN 'chemin'
          WHEN 'chem' THEN 'chemin'
          WHEN 'imp' THEN 'impasse'
          WHEN 'pl' THEN 'place'
          WHEN 'rte' THEN 'route'
          WHEN 'all' THEN 'allee'
          WHEN 'qu' THEN 'quai'
          WHEN 'res' THEN 'residence'
          ELSE norm(value)
        END
    """)
    by_target: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

    def ingest(frame: pd.DataFrame, tag: str) -> None:
        for row in frame.to_dict("records"):
            target_siret = clean(row.pop("target_siret"))
            siret = clean(row.get("siret"))
            if not siret or siret == target_siret:
                continue
            cleaned = {
                key: clean(value) if key not in {"is_headquarters"} else bool(value)
                for key, value in row.items()
            }
            names = [
                cleaned.get(key, "")
                for key in ("usual_name", "enseigne1", "enseigne2", "enseigne3", "legal_name")
                if cleaned.get(key, "")
            ]
            existing = by_target[target_siret].setdefault(siret, cleaned)
            # A candidate may be discovered first through its address and only
            # later through a legal-unit name.  Merge every name channel instead
            # of letting discovery order silently discard the legal evidence.
            existing["name_values"] = list(dict.fromkeys([
                *existing.get("name_values", []),
                *[
                    existing.get(key, "")
                    for key in ("usual_name", "enseigne1", "enseigne2", "enseigne3", "legal_name")
                    if existing.get(key, "")
                ],
                *names,
            ]))
            for key, value in cleaned.items():
                if key == "legal_name" and value and not existing.get(key):
                    existing[key] = value
            existing.setdefault("relation_tags", [])
            if tag not in existing["relation_tags"]:
                existing["relation_tags"].append(tag)

    sibling_frame = connection.execute(f"""
        SELECT t.target_siret, {PROJECTION}
        FROM read_parquet(?) e
        JOIN targets t ON e.siren=t.target_siren
        WHERE e.siret<>t.target_siret
        ORDER BY t.target_siret, e.siret
    """, [str(establishments)]).fetchdf()
    ingest(sibling_frame, "SAME_SIREN")

    address_frame = connection.execute(f"""
        SELECT t.target_siret, {PROJECTION}
        FROM read_parquet(?) e
        JOIN targets t
          ON norm(e.numeroVoieEtablissement)=t.number_key
         AND norm(e.indiceRepetitionEtablissement)=t.index_key
         AND street_type_norm(e.typeVoieEtablissement)=t.street_type_key
         AND norm(e.libelleVoieEtablissement)=t.street_key
         AND norm(e.codePostalEtablissement)=t.postcode_key
         AND norm(e.codeCommuneEtablissement)=t.insee_key
        WHERE e.siret<>t.target_siret
        ORDER BY t.target_siret, e.siret
    """, [str(establishments)]).fetchdf()
    ingest(address_frame, "SAME_OFFICIAL_ADDRESS")

    name_geo_frame = connection.execute(f"""
        WITH local AS (
          SELECT t.target_siret, {PROJECTION},
                 norm(e.denominationUsuelleEtablissement) AS k1,
                 norm(e.enseigne1Etablissement) AS k2,
                 norm(e.enseigne2Etablissement) AS k3,
                 norm(e.enseigne3Etablissement) AS k4
          FROM read_parquet(?) e
          JOIN (SELECT DISTINCT target_siret, insee_key FROM targets) t
            ON norm(e.codeCommuneEtablissement)=t.insee_key
          WHERE e.siret<>t.target_siret
        )
        SELECT DISTINCT l.* EXCLUDE(k1,k2,k3,k4)
        FROM local l JOIN target_names n
          ON n.target_siret=l.target_siret
         AND n.name_key IN (l.k1,l.k2,l.k3,l.k4)
        ORDER BY l.target_siret, l.siret
    """, [str(establishments)]).fetchdf()
    ingest(name_geo_frame, "SAME_NAME_GEOGRAPHY")

    legal_name_geo_frame = connection.execute(f"""
        WITH target_geographies AS (
          SELECT DISTINCT insee_key FROM targets
        ), local_sirens AS (
          SELECT DISTINCT g.insee_key, e.siren
          FROM read_parquet(?) e
          JOIN target_geographies g ON norm(e.codeCommuneEtablissement)=g.insee_key
        ), local_units AS (
          SELECT s.insee_key, u.*
          FROM read_parquet(?) u
          JOIN local_sirens s ON u.siren=s.siren
        ), unit_names AS (
          SELECT u.insee_key, u.siren, value AS legal_name, norm(value) AS name_key
          FROM local_units u,
          UNNEST([
            u.denominationUniteLegale,
            u.denominationUsuelle1UniteLegale,
            u.denominationUsuelle2UniteLegale,
            u.denominationUsuelle3UniteLegale,
            u.sigleUniteLegale,
            concat_ws(' ', u.prenomUsuelUniteLegale, u.nomUniteLegale),
            concat_ws(' ', u.nomUniteLegale, u.prenomUsuelUniteLegale),
            concat_ws(' ', u.prenom1UniteLegale, u.nomUniteLegale),
            concat_ws(' ', u.nomUniteLegale, u.prenom1UniteLegale),
            u.nomUsageUniteLegale
          ]) names(value)
          WHERE norm(value)<>''
        ), matching_sirens AS (
          SELECT DISTINCT n.target_siret, u.siren, u.legal_name
          FROM unit_names u
          JOIN target_names n ON n.insee_key=u.insee_key AND n.name_key=u.name_key
        )
        SELECT m.target_siret, {PROJECTION}, m.legal_name
        FROM matching_sirens m
        JOIN targets t ON t.target_siret=m.target_siret
        JOIN read_parquet(?) e ON e.siren=m.siren
        WHERE e.siret<>m.target_siret
          AND norm(e.codeCommuneEtablissement)=t.insee_key
    """, [str(establishments), str(legal_units), str(establishments)]).fetchdf()
    ingest(legal_name_geo_frame, "SAME_NAME_GEOGRAPHY")

    # Address-only and sibling discoveries must receive the same complete legal
    # name evidence as name-geometry discoveries.  Otherwise a post-degradation
    # CRM can collide with a same-site legal name that the deterministic gate and
    # critic never saw.
    contextual_sirens = sorted({
        clean(row.get("siren"))
        for values in by_target.values() for row in values.values()
        if clean(row.get("siren"))
    })
    if contextual_sirens:
        connection.register(
            "contextual_sirens", pd.DataFrame({"siren": contextual_sirens})
        )
        legal_names_frame = connection.execute("""
            WITH units AS (
              SELECT u.* FROM read_parquet(?) u
              JOIN contextual_sirens c ON u.siren=c.siren
            ), names AS (
              SELECT u.siren, value
              FROM units u,
              UNNEST([
                u.denominationUniteLegale,
                u.denominationUsuelle1UniteLegale,
                u.denominationUsuelle2UniteLegale,
                u.denominationUsuelle3UniteLegale,
                u.sigleUniteLegale,
                concat_ws(' ', u.prenomUsuelUniteLegale, u.nomUniteLegale),
                concat_ws(' ', u.nomUniteLegale, u.prenomUsuelUniteLegale),
                concat_ws(' ', u.prenom1UniteLegale, u.nomUniteLegale),
                concat_ws(' ', u.nomUniteLegale, u.prenom1UniteLegale),
                u.nomUsageUniteLegale
              ]) values(value)
              WHERE norm(value)<>''
            )
            SELECT siren, list(DISTINCT value ORDER BY value) AS legal_names
            FROM names GROUP BY siren ORDER BY siren
        """, [str(legal_units)]).fetchdf()
        legal_names_by_siren = {
            clean(row["siren"]): [
                clean(value) for value in row["legal_names"] if clean(value)
            ]
            for row in legal_names_frame.to_dict("records")
        }
        for values in by_target.values():
            for row in values.values():
                legal_names = legal_names_by_siren.get(clean(row.get("siren")), [])
                row["name_values"] = list(dict.fromkeys([
                    *row.get("name_values", []), *legal_names,
                ]))
                if legal_names and not row.get("legal_name"):
                    row["legal_name"] = legal_names[0]
    connection.close()
    return {
        target_siret: sorted(values.values(), key=lambda row: clean(row["siret"]))
        for target_siret, values in by_target.items()
    }


def address_key(row: dict[str, Any]) -> tuple[str, ...]:
    return (
        loop.normalized_alnum(row.get("number")),
        loop.normalized_alnum(row.get("repetition_index")),
        loop.normalized_alnum(canonical_street_type(row.get("street_type"))),
        loop.normalized_alnum(row.get("street")),
        loop.normalized_alnum(row.get("postcode")),
        loop.normalized_alnum(row.get("insee")),
    )


def opaque_ref(target_siret: str, candidate_siret: str) -> str:
    return "CTX-" + hashlib.sha256(f"{target_siret}|{candidate_siret}".encode("ascii")).hexdigest()[:16]


def llm_candidate(target_siret: str, row: dict[str, Any]) -> dict[str, Any]:
    names = list(dict.fromkeys(
        clean(value)
        for value in (
            list(row.get("name_values", []))
            + [row.get(key) for key in ("usual_name", "enseigne1", "enseigne2", "enseigne3", "legal_name")]
        )
        if clean(value)
    ))[:4]
    return {
        "site_ref": opaque_ref(target_siret, clean(row["siret"])),
        "relation_tags": sorted(row["relation_tags"]),
        "state": clean(row.get("state")),
        "is_headquarters": bool(row.get("is_headquarters")),
        "names": names,
        "address": {
            key: clean(row.get(key))
            for key in ("number", "repetition_index", "street_type", "street", "postcode", "city", "insee")
        },
    }


def relation_priority(row: dict[str, Any]) -> int:
    tags = set(row["relation_tags"])
    if "SAME_SIREN" in tags:
        return 0
    if "SAME_OFFICIAL_ADDRESS" in tags:
        return 1
    if "SAME_NAME_GEOGRAPHY" in tags:
        return 2
    return 3


def context_row(
    candidate: dict[str, Any],
    context: list[dict[str, Any]],
    protected_sirens: set[str],
    max_llm_context: int,
) -> dict[str, Any]:
    target_siret = clean(candidate["source_siret"])
    target_siren = clean(candidate["source_siren"])
    card = contracts.candidate_card(candidate)
    target_address_key = tuple(
        loop.normalized_alnum(value)
        for value in (
            card["number"], "", canonical_street_type(card["street_type"]), card["street"], card["postcode"], card["insee"]
        )
    )
    for row in context:
        row["protected_siren"] = clean(row["siren"]) in protected_sirens
    same_site = [row for row in context if address_key(row) == target_address_key]
    operational = [
        row for row in same_site if clean(row["siren"]) == target_siren
    ]
    protected_conflicts = [
        row for row in context
        if row["protected_siren"]
        and ({"SAME_OFFICIAL_ADDRESS", "SAME_NAME_GEOGRAPHY"} & set(row["relation_tags"]))
    ]
    exact_conflicts = [
        row for row in context
        if {"SAME_OFFICIAL_ADDRESS", "SAME_NAME_GEOGRAPHY"} & set(row["relation_tags"])
    ]
    visible = [row for row in context if not row["protected_siren"]]
    visible.sort(
        key=lambda row: (
            relation_priority(row),
            hashlib.sha256(f"{target_siret}|{row['siret']}".encode("ascii")).hexdigest(),
        )
    )
    visible = visible[:max_llm_context]
    internal = [
        {
            **row,
            "relation_tags": sorted(row["relation_tags"]),
            "record_sha256": loop.digest_json({
                key: row.get(key) for key in sorted(row) if key != "record_sha256"
            }),
        }
        for row in context
    ]
    target_view = {
        "site_ref": "TARGET",
        "state": card["state"],
        "names": [
            {"kind": "OFFICIAL_NAME", "value": value} for value in card["name_options"][:4]
        ] + [
            {"kind": "ENSEIGNE", "value": value} for value in card["enseigne_options"][:4]
        ],
        "address": {
            "number": card["number"],
            "repetition_index": "",
            "street_type": card["street_type"],
            "street": card["street"],
            "postcode": card["postcode"],
            "city": card["city"],
            "insee": card["insee"],
        },
    }
    result = {
        "schema_version": "sireto-synthetic-official-context-3",
        "target_siret": target_siret,
        "target_siren": target_siren,
        "target_source_record_sha256": candidate.get("source_record_sha256"),
        "target": target_view,
        "internal_context": internal,
        "llm_view": {
            "target": target_view,
            "official_context": [llm_candidate(target_siret, row) for row in visible],
            "context_summary": {
                "siblings_total": sum("SAME_SIREN" in row["relation_tags"] for row in context),
                "same_address_total": len(same_site),
                "same_name_geo_total": sum("SAME_NAME_GEOGRAPHY" in row["relation_tags"] for row in context),
                "protected_exact_conflicts": len(protected_conflicts),
                "rows_sent": len(visible),
                "rows_total": len(context),
                "exact_conflicts_omitted": max(0, len(exact_conflicts) - sum(
                    row in visible for row in exact_conflicts
                )),
            },
        },
        "qualification": {
            "siblings_complete": True,
            "same_address_complete": True,
            "same_name_geography_complete": True,
            "exact_identifiable_at_official_baseline": not operational,
            "operational_equivalence": bool(operational),
            "operational_equivalent_sirets": sorted(clean(row["siret"]) for row in operational),
            "protected_conflict": bool(protected_conflicts),
            "exact_conflict_count": len(exact_conflicts),
            "pre_generation_exact_eligible": (
                not operational and not protected_conflicts and len(exact_conflicts) <= 24
            ),
        },
    }
    result["context_sha256"] = loop.digest_json({
        key: value for key, value in result.items() if key != "context_sha256"
    })
    return result


def build(args: argparse.Namespace) -> dict[str, Any]:
    seed_sirets = read_seed_sirets(args.seed_input)
    candidates = candidate_rows(args.official_candidates, set(seed_sirets))
    targets = target_frame(candidates)
    names = name_frame(targets)
    contexts = query_context(
        args.establishments, args.legal_units, targets, names, args.temp_directory
    )
    protected_sirens = read_crm_sirens(args.crm)
    rows = [
        context_row(candidates[siret], contexts.get(siret, []), protected_sirens, args.max_llm_context)
        for siret in seed_sirets
    ]
    relation_counts = Counter(
        tag for row in rows for candidate in row["internal_context"]
        for tag in candidate.get("relation_tags", [])
    )
    if len(rows) >= 100 and relation_counts.get("SAME_OFFICIAL_ADDRESS", 0) == 0:
        raise RuntimeError("degenerate context build: zero SAME_OFFICIAL_ADDRESS relations")
    if len(rows) >= 100 and relation_counts.get("SAME_NAME_GEOGRAPHY", 0) == 0:
        raise RuntimeError("degenerate context build: zero SAME_NAME_GEOGRAPHY relations")
    qualification_counts = Counter(
        "EXACT_ELIGIBLE" if row["qualification"]["pre_generation_exact_eligible"]
        else "OPERATIONAL_ONLY" if row["qualification"]["operational_equivalence"]
        else "PROTECTED_CONFLICT" if row["qualification"]["protected_conflict"]
        else "DENSE_CONFLICT"
        for row in rows
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    loop.write_jsonl_atomic(args.output, rows)
    manifest = {
        "schema_version": "sireto-synthetic-official-context-manifest-3",
        "rows": len(rows),
        "distinct_siret": len({row["target_siret"] for row in rows}),
        "distinct_siren": len({row["target_siren"] for row in rows}),
        "qualification_counts": dict(sorted(qualification_counts.items())),
        "relation_counts": dict(sorted(relation_counts.items())),
        "normalization_contract": {
            "unicode": "duckdb_strip_accents_then_lower",
            "pattern": "[^a-z0-9]",
            "street_type_aliases_sha256": loop.digest_json(loop.STREET_TYPE_ABBREVIATIONS),
            "physical_person_names_included": True,
        },
        "max_llm_context": args.max_llm_context,
        "source_hashes": {
            "seed_input": sha256(args.seed_input),
            "official_candidates": sha256(args.official_candidates),
            "establishments": sha256(args.establishments),
            "legal_units": sha256(args.legal_units),
            "crm_protected_sirens": sha256(args.crm),
        },
        "protected_siren_count": len(protected_sirens),
        "maps_requests": 0,
        "model_scores_used": False,
        "text_generation": "none",
        "output_sha256": sha256(args.output),
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--seed-input", type=Path, required=True)
    result.add_argument("--official-candidates", type=Path, required=True)
    result.add_argument("--establishments", type=Path, required=True)
    result.add_argument("--legal-units", type=Path, required=True)
    result.add_argument("--crm", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument(
        "--temp-directory",
        type=Path,
        default=Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100/tmp/duckdb_synthetic_gt_context_v2"),
    )
    result.add_argument("--max-llm-context", type=int, default=32)
    result.set_defaults(func=build)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if not 1 <= args.max_llm_context <= 32:
        raise ValueError("max LLM context must be between 1 and 32")
    args.func(args)


if __name__ == "__main__":
    main()
