#!/usr/bin/env python3
"""Extend the frozen official context from pinned local SIRENE snapshots only.

The extension targets active, complete, one-establishment SIRENs whose exact
official address is unique in the snapshot.  Existing official-context targets,
all CRM SIRENs, the production registry, and explicitly supplied canaries are
excluded before selection.  The existing V2 context builder remains the sole
authority for sibling, same-address, same-name, qualification, and record hashes.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Sequence

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_synthetic_gt_official_context_v2 as context_builder
from scripts import enrich_sirene_agentic_candidates as enrich
from scripts import manage_synthetic_gt_balanced_registry as registry_lib
from scripts import run_synthetic_gt_agentic_loop as loop
from scripts import select_synthetic_gt_balanced_production as production
from scripts import select_synthetic_gt_fragment_pilot as fragments


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [value for _raw, value in loop.iter_jsonl_raw(path)]


def excluded_seed_ids(paths: Sequence[Path]) -> tuple[set[str], set[str]]:
    sirets: set[str] = set()
    sirens: set[str] = set()
    for path in paths:
        for row in jsonl(path):
            siret = str(row.get("target_siret") or row.get("source_siret") or "")
            siren = str(row.get("target_siren") or row.get("source_siren") or "")
            if loop.valid_siret(siret):
                sirets.add(siret)
                sirens.add(siret[:9])
            if loop.valid_siren(siren):
                sirens.add(siren)
    return sirets, sirens


def select_snapshot_rows(
    establishments: Path,
    crm: Path,
    base_candidates: Path,
    extra_excluded_sirens: set[str],
    pool_limit: int,
    candidate_limit: int,
    selection_seed: str,
    temp_directory: Path,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Return active, complete, single-establishment and unique-site records."""
    temp_directory.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.execute("SET memory_limit='8GB'")
    connection.execute("SET threads=4")
    connection.execute("SET temp_directory=?", [str(temp_directory)])
    connection.execute("SET max_temp_directory_size='100GiB'")
    connection.register(
        "extra_excluded",
        pd.DataFrame({"siren": sorted(extra_excluded_sirens)})
        if extra_excluded_sirens else pd.DataFrame({"siren": pd.Series(dtype=str)}),
    )
    connection.execute("""
        CREATE OR REPLACE MACRO norm(value) AS
        regexp_replace(lower(strip_accents(coalesce(value, ''))), '[^a-z0-9]', '', 'g')
    """)
    connection.execute("""
        CREATE OR REPLACE MACRO street_type_norm(value) AS
        CASE norm(value)
          WHEN 'r' THEN 'rue' WHEN 'av' THEN 'avenue'
          WHEN 'bd' THEN 'boulevard' WHEN 'ch' THEN 'chemin'
          WHEN 'che' THEN 'chemin' WHEN 'chem' THEN 'chemin'
          WHEN 'imp' THEN 'impasse' WHEN 'pl' THEN 'place'
          WHEN 'rte' THEN 'route' WHEN 'all' THEN 'allee'
          WHEN 'qu' THEN 'quai' WHEN 'res' THEN 'residence'
          ELSE norm(value)
        END
    """)
    connection.execute("""
        CREATE TEMP TABLE candidate_pool AS
        WITH excluded AS (
          SELECT source_siren AS siren FROM read_json_auto(?)
          UNION
          SELECT DISTINCT substr(regexp_replace(gt_siret, '[^0-9]', '', 'g'), 1, 9)
          FROM read_csv(?, delim=';', header=true, all_varchar=true)
          WHERE length(regexp_replace(gt_siret, '[^0-9]', '', 'g'))=14
          UNION SELECT siren FROM extra_excluded
        ), ranked AS (
          SELECT
            e.siren, e.siret,
            e.etatAdministratifEtablissement AS state,
            e.etablissementSiege AS is_headquarters,
            e.denominationUsuelleEtablissement AS name,
            e.enseigne1Etablissement AS enseigne,
            e.numeroVoieEtablissement AS street_number,
            e.indiceRepetitionEtablissement AS repetition_index,
            e.typeVoieEtablissement AS street_type,
            e.libelleVoieEtablissement AS street,
            e.codePostalEtablissement AS postcode,
            e.libelleCommuneEtablissement AS city,
            e.codeCommuneEtablissement AS insee,
            row_number() OVER (
              PARTITION BY e.siren ORDER BY sha256(e.siret || ?)
            ) AS siren_rank
          FROM read_parquet(?) e ANTI JOIN excluded x USING(siren)
          WHERE e.etatAdministratifEtablissement='A'
            AND trim(coalesce(e.numeroVoieEtablissement, '')) NOT IN ('', '[ND]')
            AND trim(coalesce(e.indiceRepetitionEtablissement, '')) IN ('', '[ND]')
            AND trim(coalesce(e.typeVoieEtablissement, '')) NOT IN ('', '[ND]')
            AND trim(coalesce(e.libelleVoieEtablissement, '')) NOT IN ('', '[ND]')
            AND trim(coalesce(e.codePostalEtablissement, '')) NOT IN ('', '[ND]')
            AND trim(coalesce(e.libelleCommuneEtablissement, '')) NOT IN ('', '[ND]')
            AND trim(coalesce(e.codeCommuneEtablissement, '')) NOT IN ('', '[ND]')
        )
        SELECT * EXCLUDE(siren_rank) FROM ranked WHERE siren_rank=1
        ORDER BY sha256(siret || ?) LIMIT ?
    """, [
        str(base_candidates), str(crm), selection_seed, str(establishments),
        selection_seed, pool_limit,
    ])
    pool_count = connection.execute("SELECT count(*) FROM candidate_pool").fetchone()[0]
    frame = connection.execute("""
        WITH site_counts AS (
          SELECT p.siren, count(*) AS establishment_count
          FROM read_parquet(?) e JOIN candidate_pool p USING(siren)
          GROUP BY p.siren
        ), single_site AS (
          SELECT p.* FROM candidate_pool p JOIN site_counts s USING(siren)
          WHERE s.establishment_count=1
        ), address_counts AS (
          SELECT p.siret, count(*) AS address_count
          FROM single_site p JOIN read_parquet(?) e
            ON norm(e.numeroVoieEtablissement)=norm(p.street_number)
           AND norm(e.indiceRepetitionEtablissement)=''
           AND street_type_norm(e.typeVoieEtablissement)=street_type_norm(p.street_type)
           AND norm(e.libelleVoieEtablissement)=norm(p.street)
           AND norm(e.codePostalEtablissement)=norm(p.postcode)
           AND norm(e.codeCommuneEtablissement)=norm(p.insee)
          GROUP BY p.siret
        )
        SELECT p.* FROM single_site p JOIN address_counts a USING(siret)
        WHERE a.address_count=1
        ORDER BY sha256(p.siret || ?) LIMIT ?
    """, [str(establishments), str(establishments), selection_seed, candidate_limit]).fetchdf()
    connection.close()
    values = frame.to_dict("records")
    records = [
        {
            "schema_version": "sireto-synthetic-gt-sirene-seed-1",
            "seed_source": "SIRENE_ONLY_TRAIN",
            "selection_seed": selection_seed,
            "selection_ordinal": index,
            "source_siret": str(row["siret"]),
            "source_siren": str(row["siren"]),
            "official_fields": row,
        }
        for index, row in enumerate(values)
    ]
    return records, {
        "initial_pool": int(pool_count),
        "single_site_unique_address_before_name_enrichment": len(records),
    }


def enrich_candidates(
    candidates: Sequence[dict[str, Any]], legal_units: Path,
) -> list[dict[str, Any]]:
    sirens = sorted({str(value["source_siren"]) for value in candidates})
    connection = duckdb.connect()
    connection.register("wanted", pd.DataFrame({"siren": sirens}))
    columns = [
        "denominationUsuelle1UniteLegale", "denominationUsuelle2UniteLegale",
        "denominationUsuelle3UniteLegale", "denominationUniteLegale",
        "prenomUsuelUniteLegale", "nomUniteLegale", "sigleUniteLegale",
    ]
    projection = ",".join(f"CAST(u.{name} AS VARCHAR) AS {name}" for name in columns)
    result = connection.execute(
        f"""SELECT CAST(u.siren AS VARCHAR) AS siren,{projection}
             FROM read_parquet(?) u JOIN wanted w ON CAST(u.siren AS VARCHAR)=w.siren""",
        [str(legal_units)],
    )
    names = [item[0] for item in result.description]
    units = {str(row[0]): dict(zip(names, row)) for row in result.fetchall()}
    connection.close()
    return [
        value for row in candidates
        if (value := enrich.enrich_row(row, units.get(str(row["source_siren"]), {})))
        is not None
    ]


def write_jsonl(path: Path, values: Sequence[dict[str, Any]]) -> None:
    loop.write_jsonl_atomic(path, list(values))


def baseline_document_frequencies(
    base_candidates: Path, extension_candidates: Sequence[dict[str, Any]],
) -> Counter[str]:
    result: Counter[str] = Counter()
    for path_values in (jsonl(base_candidates), list(extension_candidates)):
        for candidate in path_values:
            card = context_builder.contracts.candidate_card(candidate)
            result.update(set(loop.normalized_words(card["name_options"][0])))
    return result


def easy_capacity(
    context: dict[str, Any], grouped: dict[tuple[str, str], list[dict[str, Any]]],
    document_frequencies: Counter[str], selection_seed: str,
    support_cache: dict[tuple[Any, ...], list[dict[str, Any]]],
) -> tuple[int, dict[str, Any], dict[tuple[str, str], list[dict[str, Any]]]]:
    names, locations = production.safe_capabilities(
        context, grouped, document_frequencies,
        tuple(production.SAFE_NAME_RELATIONS), support_cache,
    )
    bundles = production.candidate_bundles(
        context, names, locations, selection_seed, limit=64,
    )
    capacity = max((
        sum(production.difficulty(context, pair) == "EASY" for pair in bundle)
        for bundle in bundles
    ), default=0)
    return capacity, names, locations


def merge_jsonl(base: Path, extension: Sequence[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as target, base.open("rb") as source:
        shutil.copyfileobj(source, target, 1024 * 1024)
        if target.tell():
            source.seek(-1, os.SEEK_END)
            if source.read(1) != b"\n":
                target.write(b"\n")
        for value in extension:
            target.write(json.dumps(
                value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8") + b"\n")
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, output)


def build(args: argparse.Namespace) -> dict[str, Any]:
    base_manifest_path = args.base_context.with_suffix(
        args.base_context.suffix + ".manifest.json"
    )
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    if base_manifest.get("output_sha256") != sha256(args.base_context):
        raise ValueError("base official context is not sealed")
    registry = registry_lib.load_registry(args.registry)
    registry_snapshot = registry_lib.snapshot(registry)
    if registry.get("summary") != registry_snapshot:
        raise ValueError("production registry is not sealed")
    _excluded_sirets, excluded_sirens = excluded_seed_ids(args.exclude_seed_input)
    excluded_sirens.update(registry_snapshot["excluded_target_sirens"])
    raw_candidates, selection_counts = select_snapshot_rows(
        args.establishments, args.crm, args.base_candidates, excluded_sirens,
        args.pool_limit, args.candidate_limit, args.selection_seed,
        args.temp_directory / "selection",
    )
    candidates = enrich_candidates(raw_candidates, args.legal_units)
    contexts_by_target = context_builder.query_context(
        args.establishments, args.legal_units,
        context_builder.target_frame({row["source_siret"]: row for row in candidates}),
        context_builder.name_frame(context_builder.target_frame({
            row["source_siret"]: row for row in candidates
        })),
        args.temp_directory / "context",
    )
    protected_sirens = context_builder.read_crm_sirens(args.crm)
    built = [
        context_builder.context_row(
            row, contexts_by_target.get(row["source_siret"], []),
            protected_sirens, args.max_llm_context,
        )
        for row in candidates
    ]
    clean_contexts = [
        value for value in built
        if value["target"]["state"] == "A"
        and value["qualification"]["pre_generation_exact_eligible"]
        and not production.context_flags(value)
    ]
    grouped = fragments.group_fragments(jsonl(args.field_inspiration_bank))
    frequencies = baseline_document_frequencies(args.base_candidates, candidates)
    audited = []
    support_cache: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for value in clean_contexts:
        capacity, _names, _locations = easy_capacity(
            value, grouped, frequencies, args.selection_seed, support_cache,
        )
        if capacity:
            audited.append((capacity, value))
    audited.sort(key=lambda item: (
        -item[0], hashlib.sha256(
            f"{args.selection_seed}|{item[1]['target_siret']}".encode()
        ).hexdigest(),
    ))
    chosen = audited[:args.extension_target_limit]
    extension_contexts = [value for _capacity, value in chosen]
    extension_candidates = {
        value["source_siret"]: value for value in candidates
    }
    selected_candidates = [
        extension_candidates[value["target_siret"]] for value in extension_contexts
    ]
    total_easy_capacity = sum(capacity for capacity, _value in chosen)
    if len(extension_contexts) < args.minimum_extension_targets:
        raise RuntimeError(
            f"only {len(extension_contexts)} safe extension targets; "
            f"minimum is {args.minimum_extension_targets}; "
            f"EASY capacity={total_easy_capacity}; selection={selection_counts}; "
            f"enriched={len(candidates)}; clean={len(clean_contexts)}"
        )
    if total_easy_capacity < args.minimum_easy_capacity:
        raise RuntimeError(
            f"only {total_easy_capacity} safe EASY variants; "
            f"minimum is {args.minimum_easy_capacity}; "
            f"targets={len(extension_contexts)}; "
            f"capacity_counts={dict(sorted(capacity_counts.items()))}; "
            f"selection={selection_counts}; enriched={len(candidates)}; "
            f"clean={len(clean_contexts)}; safe={len(audited)}"
        )
    write_jsonl(args.candidate_output, selected_candidates)
    merge_jsonl(args.base_context, extension_contexts, args.output)
    capacity_counts = Counter(capacity for capacity, _value in chosen)
    manifest = {
        "schema_version": "sireto-synthetic-official-context-extension-1",
        "rows": int(base_manifest["rows"]) + len(extension_contexts),
        "base_rows": int(base_manifest["rows"]),
        "extension_rows": len(extension_contexts),
        "extension_distinct_siret": len({v["target_siret"] for v in extension_contexts}),
        "extension_distinct_siren": len({v["target_siren"] for v in extension_contexts}),
        "selection_counts": {
            **selection_counts,
            "name_enriched": len(candidates),
            "exact_eligible_active_simple": len(clean_contexts),
            "safe_easy_capable": len(audited),
        },
        "selection_parameters": {
            "selection_seed": args.selection_seed,
            "pool_limit": args.pool_limit,
            "candidate_limit": args.candidate_limit,
            "extension_target_limit": args.extension_target_limit,
            "minimum_extension_targets": args.minimum_extension_targets,
            "minimum_easy_capacity": args.minimum_easy_capacity,
            "max_llm_context": args.max_llm_context,
        },
        "extension_relation_counts": dict(sorted(Counter(
            tag
            for value in extension_contexts
            for neighbour in value["internal_context"]
            for tag in neighbour.get("relation_tags", [])
        ).items())),
        "easy_capacity": {
            "total_variants": total_easy_capacity,
            "targets_by_maximum_safe_easy_variants": dict(sorted(capacity_counts.items())),
            "definition": "max EASY variants in one runtime-valid 1..3 exact-operator bundle",
        },
        "selection_contract": {
            "active_only": True,
            "one_establishment_per_siren_snapshot_exact": True,
            "unique_normalized_official_address_snapshot_exact": True,
            "full_address_required": True,
            "blank_repetition_index_required": True,
            "pre_generation_exact_eligible": True,
            "context_flags_empty": True,
            "all_base_context_sirens_excluded": True,
            "all_crm_sirens_excluded": True,
            "registry_and_explicit_seed_sirens_excluded": True,
            "maps_requests": 0,
            "model_or_retrieval_used": False,
            "text_generation": "none",
        },
        "source_hashes": {
            "base_context": sha256(args.base_context),
            "base_context_manifest": sha256(base_manifest_path),
            "base_candidates": sha256(args.base_candidates),
            "establishments": sha256(args.establishments),
            "legal_units": sha256(args.legal_units),
            "crm": sha256(args.crm),
            "field_inspiration_bank": sha256(args.field_inspiration_bank),
            "registry": sha256(args.registry),
            **{
                f"excluded_seed:{path}": sha256(path)
                for path in args.exclude_seed_input
            },
        },
        "candidate_output": str(args.candidate_output),
        "candidate_output_sha256": sha256(args.candidate_output),
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
    result.add_argument("--base-context", type=Path, required=True)
    result.add_argument("--base-candidates", type=Path, required=True)
    result.add_argument("--establishments", type=Path, required=True)
    result.add_argument("--legal-units", type=Path, required=True)
    result.add_argument("--crm", type=Path, required=True)
    result.add_argument("--field-inspiration-bank", type=Path, required=True)
    result.add_argument("--registry", type=Path, required=True)
    result.add_argument("--exclude-seed-input", type=Path, action="append", default=[])
    result.add_argument("--candidate-output", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--selection-seed", default="SIRETO-OFFICIAL-CONTEXT-EXTENSION-1")
    result.add_argument("--pool-limit", type=int, default=30000)
    result.add_argument("--candidate-limit", type=int, default=8000)
    result.add_argument("--extension-target-limit", type=int, default=2500)
    result.add_argument("--minimum-extension-targets", type=int, default=1500)
    result.add_argument("--minimum-easy-capacity", type=int, default=4000)
    result.add_argument("--max-llm-context", type=int, default=32)
    result.add_argument(
        "--temp-directory", type=Path,
        default=Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100/tmp/duckdb_official_context_extension_v1"),
    )
    result.set_defaults(func=build)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if not (0 < args.minimum_extension_targets <= args.extension_target_limit):
        raise ValueError("invalid extension target bounds")
    if args.minimum_easy_capacity <= 0:
        raise ValueError("minimum EASY capacity must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
