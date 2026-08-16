#!/usr/bin/env python3
"""Audit accepted synthetic CRM against the full local SIRENE snapshot.

This is a qualification join, not SIRETO retrieval: candidates come only from
the CRM's exact INSEE code (and exact postcode when present), the positive is
never injected, and no model score/rank is read.  The script never edits CRM or
the agentic ledger.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import stat
import sys
from typing import Any, Iterable, Sequence
import uuid

import duckdb


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_synthetic_gt_agentic_loop as loop  # noqa: E402


DEFAULT_OFFICIAL_CACHE_DIRECTORY = Path(
    "/Volumes/CATNAT_DATA/SIRETO_RECALL100/cache/"
    "synthetic_gt_full_sirene_exact"
)
OFFICIAL_CACHE_SCHEMA_VERSION = "sireto-full-sirene-official-cache-1"
OFFICIAL_CACHE_COLUMNS = (
    ("siret", "VARCHAR", "e.siret"),
    ("siren", "VARCHAR", "e.siren"),
    ("insee", "VARCHAR", "e.codeCommuneEtablissement"),
    ("postcode", "VARCHAR", "e.codePostalEtablissement"),
    ("number", "VARCHAR", "e.numeroVoieEtablissement"),
    ("repetition_index", "VARCHAR", "e.indiceRepetitionEtablissement"),
    ("street_type", "VARCHAR", "e.typeVoieEtablissement"),
    ("street", "VARCHAR", "e.libelleVoieEtablissement"),
    ("establishment_usual", "VARCHAR", "e.denominationUsuelleEtablissement"),
    ("enseigne1", "VARCHAR", "e.enseigne1Etablissement"),
    ("enseigne2", "VARCHAR", "e.enseigne2Etablissement"),
    ("enseigne3", "VARCHAR", "e.enseigne3Etablissement"),
    ("legal_denomination", "VARCHAR", "u.denominationUniteLegale"),
    ("legal_sigle", "VARCHAR", "u.sigleUniteLegale"),
    ("legal_usual1", "VARCHAR", "u.denominationUsuelle1UniteLegale"),
    ("legal_usual2", "VARCHAR", "u.denominationUsuelle2UniteLegale"),
    ("legal_usual3", "VARCHAR", "u.denominationUsuelle3UniteLegale"),
    ("legal_last_name", "VARCHAR", "u.nomUniteLegale"),
    ("legal_usage_name", "VARCHAR", "u.nomUsageUniteLegale"),
    ("legal_usual_first", "VARCHAR", "u.prenomUsuelUniteLegale"),
    ("legal_first", "VARCHAR", "u.prenom1UniteLegale"),
    ("state", "VARCHAR", "e.etatAdministratifEtablissement"),
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def official_cache_schema_sha256() -> str:
    payload = {
        "schema_version": OFFICIAL_CACHE_SCHEMA_VERSION,
        "columns": OFFICIAL_CACHE_COLUMNS,
        "join": "establishments LEFT JOIN legal_units ON establishments.siren=legal_units.siren",
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def official_cache_key(source_hashes: dict[str, str]) -> str:
    payload = {
        "schema_version": OFFICIAL_CACHE_SCHEMA_VERSION,
        "schema_sha256": official_cache_schema_sha256(),
        "source_hashes": source_hashes,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def official_cache_path(cache_directory: Path, source_hashes: dict[str, str]) -> Path:
    return cache_directory / f"official_{official_cache_key(source_hashes)}.duckdb"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text.upper() in {"", "[ND]", "NAN", "NONE"} else text


def official_address(candidate: dict[str, Any]) -> str:
    return " ".join(
        clean(candidate.get(key))
        for key in ("number", "repetition_index", "street_type", "street")
        if clean(candidate.get(key))
    )


def official_names(candidate: dict[str, Any]) -> list[str]:
    direct = [
        clean(candidate.get(key))
        for key in (
            "establishment_usual", "enseigne1", "enseigne2", "enseigne3",
            "legal_denomination", "legal_sigle", "legal_usual1", "legal_usual2",
            "legal_usual3", "legal_last_name", "legal_usage_name",
        )
    ]
    last_name = clean(candidate.get("legal_last_name"))
    usual_first = clean(candidate.get("legal_usual_first"))
    first = clean(candidate.get("legal_first"))
    if last_name and usual_first:
        direct.extend((f"{usual_first} {last_name}", f"{last_name} {usual_first}"))
    if last_name and first:
        direct.extend((f"{first} {last_name}", f"{last_name} {first}"))
    return list(dict.fromkeys(value for value in direct if value))


def _span_cover(observed_words: list[str], official_words: list[str]) -> bool:
    """Match every observed token to a disjoint contiguous official-token span."""
    spans = {
        word: [
            frozenset(range(start, end))
            for start in range(len(official_words))
            for end in range(start + 1, len(official_words) + 1)
            if "".join(official_words[start:end]) == word
        ]
        for word in set(observed_words)
    }
    ordered = sorted(observed_words, key=lambda word: len(spans.get(word, [])))

    def visit(index: int, used: frozenset[int]) -> bool:
        if index == len(ordered):
            return True
        return any(
            not (used & span) and visit(index + 1, used | span)
            for span in spans.get(ordered[index], [])
        )

    return visit(0, frozenset())


def whole_token_language(observed: str, official: str) -> bool:
    """Finite language: whole-token delete/reorder plus boundary join/split."""
    observed_words = loop.normalized_words(observed)
    official_words = loop.normalized_words(official)
    if not observed_words or not official_words:
        return False
    if loop.normalized_alnum(observed) == loop.normalized_alnum(official):
        return True
    if not (Counter(observed_words) - Counter(official_words)):
        return True
    if _span_cover(observed_words, official_words):
        return True
    # Boundary splits after deletion: every official token used by the CRM can
    # consume a contiguous group of observed tokens; unused official tokens are
    # legitimate whole-token deletions.
    return _span_cover(official_words, observed_words) and len(observed_words) <= len(official_words)


def name_compatible(crm_name: str, candidate: dict[str, Any]) -> bool:
    return bool(crm_name.strip()) and any(
        whole_token_language(crm_name, value) for value in official_names(candidate)
    )


def address_compatible(crm_address: str, candidate: dict[str, Any]) -> bool:
    official = official_address(candidate)
    if not crm_address.strip() or not official:
        return False
    crm_digits = [value for value in loop.normalized_words(crm_address) if value.isdigit()]
    official_digits = [value for value in loop.normalized_words(official) if value.isdigit()]
    if crm_digits != official_digits:
        return False
    observed = " ".join(loop.expanded_street_words(crm_address))
    source = " ".join(loop.expanded_street_words(official))
    return whole_token_language(observed, source)


def qualify_variant(
    target_siret: str,
    crm: dict[str, str],
    candidates: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    local = [
        value for value in candidates
        if loop.normalized_alnum(value.get("insee")) == loop.normalized_alnum(crm["insee"])
        and (
            not crm["postcode"]
            or loop.normalized_alnum(value.get("postcode")) == loop.normalized_alnum(crm["postcode"])
        )
    ]
    names = {value["siret"] for value in local if name_compatible(crm["name"], value)}
    addresses = {value["siret"] for value in local if address_compatible(crm["address"], value)}
    both = names & addresses
    witnesses = {"G_N_A": both, "G_A": addresses, "G_N": names}
    # Composite positives require the conjunction of name and address.  A
    # singleton on only one anchor is useful diagnostics, never exact truth.
    exact_witness = "G_N_A" if witnesses["G_N_A"] == {target_siret} else None
    target_natural = any(target_siret in value for value in witnesses.values())
    overflow = any(len(value) > 100 for value in witnesses.values())
    if exact_witness:
        decision = "EXACT_IDENTIFIABLE"
    elif overflow:
        decision = "EXACT_CONTEXT_OVERFLOW"
    elif not target_natural:
        decision = "TARGET_NOT_NATURALLY_MATCHED"
    elif all(not value for value in witnesses.values()):
        decision = "UNRESOLVED_OFFICIAL"
    else:
        decision = "AMBIGUOUS_OFFICIAL"
    by_siret = {value["siret"]: value for value in local}
    target_candidate = by_siret.get(target_siret)
    target_site = (
        tuple(loop.expanded_street_words(official_address(target_candidate)))
        if target_candidate is not None else ()
    )
    operational = sorted(
        siret for siret in (both or addresses or names)
        if siret != target_siret
        and siret[:9] == target_siret[:9]
        and address_compatible(crm["address"], by_siret[siret])
        and tuple(loop.expanded_street_words(official_address(by_siret[siret]))) == target_site
    )
    return {
        "decision": decision,
        "exact_witness": exact_witness,
        "candidate_counts": {key: len(value) for key, value in witnesses.items()},
        "candidate_sirets": {key: sorted(value)[:101] for key, value in witnesses.items()},
        "target_naturally_returned": target_natural,
        "operational_equivalent_sirets": operational,
        "operational_equivalence": bool(operational),
    }


def load_variants(db: Path, run_id: str) -> list[dict[str, Any]]:
    connection = sqlite3.connect(db)
    connection.row_factory = sqlite3.Row
    seeds = connection.execute(
        "SELECT seed_id FROM seeds WHERE run_id=? AND status='COMPLETED'", (run_id,)
    ).fetchall()
    admissible: set[str] = set()
    for seed in seeds:
        decisions = [
            row[0] for row in connection.execute(
                "SELECT final_decision FROM variants WHERE run_id=? AND seed_id=? ORDER BY variant_id",
                (run_id, seed["seed_id"]),
            )
        ]
        if decisions == ["ACCEPT", "ACCEPT", "ACCEPT"]:
            admissible.add(seed["seed_id"])
    rows = connection.execute(
        """SELECT s.seed_id,s.target_siret,s.target_siren,s.seed_card_json,
                  v.variant_id,v.crm_json,v.final_decision
             FROM seeds s JOIN variants v USING(run_id,seed_id)
            WHERE s.run_id=? AND v.final_decision='ACCEPT'
            ORDER BY s.seed_id,v.variant_id""",
        (run_id,),
    ).fetchall()
    result = []
    for row in rows:
        card = json.loads(row["seed_card_json"])
        result.append({
            "seed_id": row["seed_id"],
            "target_siret": row["target_siret"],
            "target_siren": row["target_siren"],
            "variant_id": row["variant_id"],
            "crm": json.loads(row["crm_json"]),
            "seed_ledger_3_of_3_accept": row["seed_id"] in admissible,
            "variant_ledger_accept": row["final_decision"] == "ACCEPT",
            "pre_generation_qualification": card.get("qualification", {}),
        })
    connection.close()
    return result


def _cache_metadata(source_hashes: dict[str, str], row_count: int) -> dict[str, Any]:
    return {
        "schema_version": OFFICIAL_CACHE_SCHEMA_VERSION,
        "schema_sha256": official_cache_schema_sha256(),
        "cache_key": official_cache_key(source_hashes),
        "source_hashes_json": _canonical_json(source_hashes),
        "row_count": row_count,
    }


def validate_official_cache(
    cache: Path,
    source_hashes: dict[str, str],
    expected_row_count: int | None = None,
    *,
    require_read_only: bool = True,
) -> dict[str, Any]:
    if not cache.is_file():
        raise FileNotFoundError(cache)
    if require_read_only and cache.stat().st_mode & 0o222:
        raise ValueError(f"official cache is not filesystem read-only: {cache}")
    connection = duckdb.connect(str(cache), read_only=True)
    try:
        tables = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
        if tables != {"cache_metadata", "official_candidates"}:
            raise ValueError(f"unexpected official cache tables: {sorted(tables)}")
        metadata_row = connection.execute(
            """SELECT schema_version,schema_sha256,cache_key,
                      source_hashes_json,row_count
                 FROM cache_metadata"""
        ).fetchall()
        if len(metadata_row) != 1:
            raise ValueError("official cache must contain exactly one metadata row")
        keys = (
            "schema_version", "schema_sha256", "cache_key",
            "source_hashes_json", "row_count",
        )
        metadata = dict(zip(keys, metadata_row[0], strict=True))
        expected = _cache_metadata(source_hashes, int(metadata["row_count"]))
        for key in keys[:-1]:
            if metadata[key] != expected[key]:
                raise ValueError(f"official cache {key} mismatch")
        if expected_row_count is not None and int(metadata["row_count"]) != expected_row_count:
            raise ValueError(
                "official cache row count mismatch: "
                f"{metadata['row_count']} != {expected_row_count}"
            )
        actual_columns = [
            (row[1], row[2])
            for row in connection.execute("PRAGMA table_info('official_candidates')").fetchall()
        ]
        expected_columns = [(name, data_type) for name, data_type, _ in OFFICIAL_CACHE_COLUMNS]
        if actual_columns != expected_columns:
            raise ValueError("official cache candidate schema mismatch")
        return {
            "schema_version": metadata["schema_version"],
            "schema_sha256": metadata["schema_sha256"],
            "cache_key": metadata["cache_key"],
            "row_count": int(metadata["row_count"]),
        }
    finally:
        connection.close()


def build_official_cache(
    cache: Path,
    establishments: Path,
    legal_units: Path,
    source_hashes: dict[str, str],
    expected_row_count: int | None,
    temp_directory: Path,
) -> None:
    temp_directory.mkdir(parents=True, exist_ok=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.parent / f".{cache.name}.building-{os.getpid()}-{uuid.uuid4().hex}"
    projection = ",\n              ".join(
        f"CAST({expression} AS {data_type}) AS {name}"
        for name, data_type, expression in OFFICIAL_CACHE_COLUMNS
    )
    connection: duckdb.DuckDBPyConnection | None = None
    try:
        connection = duckdb.connect(str(temporary))
        connection.execute("SET temp_directory = ?", [str(temp_directory)])
        connection.execute("SET memory_limit = '12GB'")
        connection.execute("SET threads = 8")
        connection.execute("SET preserve_insertion_order = false")
        connection.execute(
            f"""CREATE TABLE official_candidates AS
                SELECT {projection}
                  FROM read_parquet(?) e
                  LEFT JOIN read_parquet(?) u ON e.siren=u.siren""",
            [str(establishments), str(legal_units)],
        )
        row_count = int(connection.execute("SELECT count(*) FROM official_candidates").fetchone()[0])
        if expected_row_count is not None and row_count != expected_row_count:
            raise ValueError(
                f"built official cache row count mismatch: {row_count} != {expected_row_count}"
            )
        connection.execute(
            "CREATE INDEX official_candidates_insee_idx ON official_candidates(insee)"
        )
        connection.execute(
            """CREATE TABLE cache_metadata(
                   schema_version VARCHAR NOT NULL,
                   schema_sha256 VARCHAR NOT NULL,
                   cache_key VARCHAR NOT NULL,
                   source_hashes_json VARCHAR NOT NULL,
                   row_count UBIGINT NOT NULL
               )"""
        )
        metadata = _cache_metadata(source_hashes, row_count)
        connection.execute(
            "INSERT INTO cache_metadata VALUES (?,?,?,?,?)",
            [
                metadata["schema_version"], metadata["schema_sha256"],
                metadata["cache_key"], metadata["source_hashes_json"],
                metadata["row_count"],
            ],
        )
        connection.execute("CHECKPOINT")
        connection.close()
        connection = None
        validate_official_cache(
            temporary, source_hashes, expected_row_count, require_read_only=False,
        )
        temporary.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        try:
            os.link(temporary, cache)
        except FileExistsError:
            validate_official_cache(cache, source_hashes, expected_row_count)
        else:
            validate_official_cache(cache, source_hashes, expected_row_count)
    finally:
        if connection is not None:
            connection.close()
        temporary.unlink(missing_ok=True)
        Path(f"{temporary}.wal").unlink(missing_ok=True)


def ensure_official_cache(
    cache_directory: Path,
    establishments: Path,
    legal_units: Path,
    source_hashes: dict[str, str],
    expected_row_count: int | None,
    temp_directory: Path,
) -> tuple[Path, dict[str, Any]]:
    cache = official_cache_path(cache_directory, source_hashes)
    if cache.exists():
        return cache, validate_official_cache(cache, source_hashes, expected_row_count)
    actual_hashes = {
        "sirene_establishments": sha256(establishments),
        "sirene_legal_units": sha256(legal_units),
    }
    if actual_hashes != source_hashes:
        raise ValueError(
            f"SIRENE source hash mismatch: expected {source_hashes}, got {actual_hashes}"
        )
    build_official_cache(
        cache, establishments, legal_units, source_hashes,
        expected_row_count, temp_directory,
    )
    return cache, validate_official_cache(cache, source_hashes, expected_row_count)


def query_candidates(
    cache: Path,
    insee_values: list[str],
    temp_directory: Path,
) -> dict[str, list[dict[str, Any]]]:
    temp_directory.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(cache), read_only=True)
    connection.execute("SET temp_directory = ?", [str(temp_directory)])
    connection.execute("SET memory_limit = '12GB'")
    connection.execute("SET threads = 8")
    connection.execute("SET preserve_insertion_order = false")
    frame = connection.execute(
        """SELECT * FROM official_candidates
            WHERE insee IN (SELECT unnest(?::VARCHAR[]))""",
        [insee_values],
    ).fetchdf()
    connection.close()
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for value in frame.to_dict("records"):
        row = {key: clean(item) for key, item in value.items()}
        if loop.valid_siret(row["siret"]):
            result[row["insee"]].append(row)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    sources = plan["sources"]
    establishments = (args.plan.parent.parent / sources["sirene_establishments"]["path"]).resolve()
    legal_units = (args.plan.parent.parent / sources["sirene_legal_units"]["path"]).resolve()
    source_hashes = {
        "sirene_establishments": sources["sirene_establishments"]["sha256"],
        "sirene_legal_units": sources["sirene_legal_units"]["sha256"],
    }
    expected_row_count = int(sources["sirene_establishments"]["row_count"])
    variants = load_variants(args.db, args.run_id)
    if not variants:
        raise ValueError("no ACCEPT variants to audit")
    cache, cache_metadata = ensure_official_cache(
        args.official_cache_directory,
        establishments,
        legal_units,
        source_hashes,
        expected_row_count,
        args.temp_directory,
    )
    insee_values = sorted({value["crm"]["insee"] for value in variants})
    candidates = query_candidates(cache, insee_values, args.temp_directory)
    audited = []
    for value in variants:
        qualification = qualify_variant(
            value["target_siret"], value["crm"], candidates.get(value["crm"]["insee"], [])
        )
        audited.append({**value, "full_sirene_qualification": qualification})
    counts = Counter(value["full_sirene_qualification"]["decision"] for value in audited)
    exact_by_seed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for value in audited:
        if value["seed_ledger_3_of_3_accept"]:
            exact_by_seed[value["seed_id"]].append(value)
    strict_seed_ids = {
        seed_id for seed_id, values in exact_by_seed.items()
        if len(values) == 3 and all(
            value["full_sirene_qualification"]["decision"] == "EXACT_IDENTIFIABLE"
            for value in values
        )
    }
    for value in audited:
        value["seed_promotable_3_of_3_exact"] = value["seed_id"] in strict_seed_ids
        value["variant_promotable_exact"] = (
            value.get("variant_ledger_accept") is True
            and value["full_sirene_qualification"]["decision"] == "EXACT_IDENTIFIABLE"
            and value["full_sirene_qualification"]["exact_witness"] == "G_N_A"
            and value["full_sirene_qualification"]["target_naturally_returned"] is True
        )
    report = {
        "schema_version": "sireto-synthetic-gt-full-sirene-audit-1",
        "run_id": args.run_id,
        "ledger_sha256": sha256(args.db),
        "source_hashes": source_hashes,
        "official_cache": {
            **cache_metadata,
            "path": str(cache),
            "opened_read_only": True,
        },
        "qualification_uses_retrieval_or_model_scores": False,
        "positive_injection": False,
        "geography_query": "exact_insee_and_postcode",
        "audited_accept_variants": len(audited),
        "decision_counts": dict(sorted(counts.items())),
        "admissible_3_of_3_exact_variants": len(strict_seed_ids) * 3,
        "admissible_3_of_3_exact_seeds": len(strict_seed_ids),
        "admissible_per_variant_exact": sum(
            value["variant_promotable_exact"] for value in audited
        ),
        "rows": audited,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    summary = {key: value for key, value in report.items() if key != "rows"}
    summary["output"] = str(args.output)
    summary["output_sha256"] = sha256(args.output)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--db", type=Path, required=True)
    result.add_argument("--run-id", required=True)
    result.add_argument("--plan", type=Path, default=ROOT / "config/synthetic_gt_corpus_plan.json")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--temp-directory", type=Path, default=Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100/tmp/duckdb_synthetic_gt_full_exact"))
    result.add_argument(
        "--official-cache-directory",
        type=Path,
        default=DEFAULT_OFFICIAL_CACHE_DIRECTORY,
        help="Persistent content-addressed read-only cache of official SIRENE columns.",
    )
    result.set_defaults(func=run)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
