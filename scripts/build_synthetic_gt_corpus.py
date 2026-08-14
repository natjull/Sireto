#!/usr/bin/env python3
"""Build the SIRETO synthetic GT corpus without retrieval or model outputs.

The builder intentionally keeps source truth, generated CRM text, and
candidate identities in separate columns.  It only accepts train rows from
OOF folds 2/3/4 and rejects all labelled development/test components before
it reads the SIRENE candidate universe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Iterable

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PLAN_DEFAULT = Path("config/synthetic_gt_corpus_plan.json")
TEST_DEFAULT = Path("scripts/test_build_synthetic_gt_corpus.py")
FAMILY_ORDER = (
    "LEGAL_FORM",
    "ACRONYM_TOKENIZATION",
    "ACCENT_PUNCTUATION",
    "OCR_LIMITED",
    "TOKEN_ORDER",
    "FIELD_MISSING",
    "ADDRESS_ABBREVIATION",
    "ADDRESS_TOKEN_ORDER",
    "ADDRESS_OCR",
    "COMMUNE_VARIANT",
    "ENSEIGNE_VS_DENOMINATION",
)
SIRENE_COLUMNS = (
    "siret",
    "siren",
    "etatAdministratifEtablissement",
    "dateDebut",
    "dateDernierTraitementEtablissement",
    "etablissementSiege",
    "numeroVoieEtablissement",
    "indiceRepetitionEtablissement",
    "typeVoieEtablissement",
    "libelleVoieEtablissement",
    "codePostalEtablissement",
    "libelleCommuneEtablissement",
    "codeCommuneEtablissement",
    "enseigne1Etablissement",
    "enseigne2Etablissement",
    "enseigne3Etablissement",
    "denominationUsuelleEtablissement",
)
UL_COLUMNS = (
    "siren",
    "denominationUniteLegale",
    "denominationUsuelle1UniteLegale",
    "denominationUsuelle2UniteLegale",
    "denominationUsuelle3UniteLegale",
    "sigleUniteLegale",
    "categorieJuridiqueUniteLegale",
    "dateDernierTraitementUniteLegale",
)
QUERY_COLUMNS = (
    "example_id",
    "seed_query_id",
    "seed_source",
    "source_kind",
    "base_view",
    "crm_name",
    "crm_address",
    "crm_postcode",
    "crm_city",
    "crm_insee",
    "target_siret",
    "target_siren",
    "target_state",
    "oof_fold",
    "siren_component_id",
    "corruption_family",
    "variant_index",
    "confidence_weight",
    "guard_status",
    "guard_margin",
    "best_other_siret",
    "provenance_digest",
    "generator_version",
)
PAIR_COLUMNS = (
    "example_id",
    "seed_source",
    "candidate_siret",
    "candidate_siren",
    "is_positive",
    "negative_category",
    "candidate_state",
    "candidate_name",
    "candidate_address",
    "candidate_postcode",
    "candidate_city",
    "candidate_insee",
    "target_siret",
    "target_siren",
    "source_kind",
    "source_snapshot_sha256",
    "oof_fold",
    "siren_component_id",
    "candidate_provenance_digest",
)
REJECTION_COLUMNS = (
    "seed_query_id",
    "variant_index",
    "corruption_family",
    "reason",
    "base_view",
    "target_siret",
    "oof_fold",
    "siren_component_id",
    "detail_json",
)
FIELD_TYPES: dict[str, pa.DataType] = {
    **{column: pa.string() for column in QUERY_COLUMNS if column not in {"variant_index", "oof_fold", "confidence_weight", "guard_margin"}},
    "variant_index": pa.int16(),
    "oof_fold": pa.int8(),
    "confidence_weight": pa.float64(),
    "guard_margin": pa.float64(),
}
PAIR_TYPES: dict[str, pa.DataType] = {
    **{column: pa.string() for column in PAIR_COLUMNS if column not in {"is_positive", "oof_fold"}},
    "is_positive": pa.bool_(),
    "oof_fold": pa.int8(),
}
REJECTION_TYPES: dict[str, pa.DataType] = {column: pa.string() for column in REJECTION_COLUMNS}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def clean(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value)
    if text.lower() in {"nan", "none", "null", "<na>"}:
        return ""
    return text.strip()


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", clean(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.upper()
    text = re.sub(r"[^0-9A-Z]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokens(value: Any) -> set[str]:
    return {token for token in normalize_text(value).split() if token and not token.isdecimal()}


def token_jaccard(left: Any, right: Any) -> float:
    lhs, rhs = tokens(left), tokens(right)
    if not lhs or not rhs:
        return 0.0
    return len(lhs & rhs) / len(lhs | rhs)


def text_ratio(left: Any, right: Any) -> float:
    lhs, rhs = normalize_text(left), normalize_text(right)
    if not lhs or not rhs:
        return 0.0
    return SequenceMatcher(None, lhs, rhs, autojunk=False).ratio()


def max_name_similarity(query_name: Any, candidate_names: Iterable[Any]) -> float:
    values = [clean(value) for value in candidate_names if clean(value)]
    if not values:
        return 0.0
    return max(max(token_jaccard(query_name, value), text_ratio(query_name, value)) for value in values)


def address_signature(record: dict[str, Any]) -> str:
    return normalize_text(
        " ".join(
            [
                clean(record.get("numero")),
                clean(record.get("indice")),
                clean(record.get("type_voie")),
                clean(record.get("voie")),
                clean(record.get("postcode")),
                clean(record.get("insee")),
            ]
        )
    )


def street_signature(record: dict[str, Any]) -> str:
    return normalize_text(
        " ".join(
            [
                clean(record.get("type_voie")),
                clean(record.get("voie")),
                clean(record.get("postcode")),
            ]
        )
    )


def autonomous_decimal_leak(value: Any) -> bool:
    text = unicodedata.normalize("NFKC", clean(value))
    run: list[str] = []
    for char in text + " ":
        if char.isdecimal():
            run.append(char)
            continue
        if len(run) in {9, 14}:
            return True
        run = []
    return False


def valid_siret(value: Any) -> bool:
    return bool(re.fullmatch(r"[0-9]{14}", clean(value)))


def valid_siren(value: Any) -> bool:
    return bool(re.fullmatch(r"[0-9]{9}", clean(value)))


def deterministic_rng(seed: int, seed_id: str, family: str, variant_index: int):
    import random

    payload = f"{seed}|{seed_id}|{family}|{variant_index}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return random.Random(int.from_bytes(digest[:16], "big"))


def load_plan(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        plan = json.load(handle)
    if plan.get("schema_version") != "sireto-synthetic-gt-corpus-plan-1":
        raise ValueError("unexpected synthetic corpus plan schema")
    return plan


def resolve_source_path(plan_path: Path, source_path: str) -> Path:
    path = Path(source_path)
    if path.is_absolute():
        return path
    return (plan_path.parent.parent / path).resolve()


def verify_source_pins(plan: dict[str, Any], plan_path: Path) -> dict[str, dict[str, Any]]:
    verified: dict[str, dict[str, Any]] = {}
    for source_name, source in plan["sources"].items():
        path = resolve_source_path(plan_path, source["path"])
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"source is not a regular file: {path}")
        observed_hash = file_sha256(path)
        if observed_hash != source["sha256"]:
            raise RuntimeError(f"source hash mismatch for {source_name}")
        observed_rows: int | None = None
        if source["format"] == "PARQUET":
            observed_rows = int(pq.ParquetFile(path).metadata.num_rows)
            if observed_rows != int(source["row_count"]):
                raise RuntimeError(f"source row count mismatch for {source_name}")
        elif source["format"].startswith("CSV"):
            observed_rows = int(sum(1 for _ in path.open(encoding="utf-8"))) - 1
            if observed_rows != int(source["row_count"]):
                raise RuntimeError(f"source row count mismatch for {source_name}")
        verified[source_name] = {
            "name": source_name,
            "path": str(path),
            "sha256": observed_hash,
            "row_count": observed_rows,
            "declared": source,
        }
    return verified


def load_train_seeds(plan: dict[str, Any], verified: dict[str, dict[str, Any]]) -> tuple[pd.DataFrame, set[str], set[str], set[str]]:
    crm_source = Path(verified["crm_ok_gt"]["path"])
    assignment_source = Path(verified["fold_assignments"]["path"])
    crm = pd.read_csv(crm_source, sep=";", dtype=str, keep_default_na=False).reset_index(names="query_id")
    assignments = pd.read_parquet(
        assignment_source,
        columns=["query_id", "siren_component_id", "oof_fold", "legacy_split"],
    )
    crm["query_id"] = crm["query_id"].astype(str)
    assignments["query_id"] = assignments["query_id"].astype(str)
    joined = crm.merge(assignments, on="query_id", how="left", validate="one_to_one", indicator=True)
    if not joined["_merge"].eq("both").all():
        raise RuntimeError("crm_ok_gt/fold assignment join is incomplete")
    if len(joined) != int(plan["population"]["expected_joined_rows"]):
        raise RuntimeError("unexpected joined population size")

    forbidden_mask = joined["oof_fold"].astype("Int64").isin(plan["population"]["forbidden_oof_folds"]) | joined["legacy_split"].isin(plan["population"]["forbidden_legacy_splits"])
    forbidden_components = set(joined.loc[forbidden_mask, "siren_component_id"].map(clean))
    forbidden_components.discard("")
    forbidden_sirens = {
        clean(value)[:9]
        for value in joined.loc[forbidden_mask, "gt_siret"].map(clean)
        if valid_siret(value)
    }
    all_crm_sirens = {
        clean(value)[:9]
        for value in joined["gt_siret"].map(clean)
        if valid_siret(value)
    }

    allowed_mask = joined["legacy_split"].eq(plan["population"]["allowed_legacy_split"]) & joined["oof_fold"].astype("Int64").isin(plan["generator"]["allowed_oof_folds"])
    seeds = joined.loc[allowed_mask].copy()
    if set(seeds["siren_component_id"].map(clean)) & forbidden_components:
        raise RuntimeError("allowed SIREN component overlaps a forbidden component")
    seeds["gt_siret"] = seeds["gt_siret"].map(clean)
    if (~seeds["gt_siret"].map(valid_siret)).any():
        raise RuntimeError("allowed seed contains invalid target SIRET")
    if seeds["gt_siret"].duplicated().any():
        raise RuntimeError("allowed seed target SIRET is not unique")
    expected = int(plan["population"]["allowed_rows"])
    if len(seeds) != expected:
        raise RuntimeError(f"unexpected allowed seed count: {len(seeds)} != {expected}")
    if set(seeds["gt_siret"].str[:9]) & forbidden_sirens:
        raise RuntimeError("allowed target SIREN overlaps forbidden SIREN")
    seeds = seeds.drop(columns=["_merge"])
    seeds["seed_source"] = "CRM_OK_GT_TRAIN"
    return seeds, forbidden_components, forbidden_sirens, all_crm_sirens


def _query_sirene(
    parquet_path: Path,
    wanted_sirets: Iterable[str],
    locations: pd.DataFrame | None = None,
) -> pd.DataFrame:
    wanted = pd.DataFrame({"siret": sorted({clean(value) for value in wanted_sirets if valid_siret(value)})})
    if wanted.empty:
        return pd.DataFrame(columns=list(SIRENE_COLUMNS))
    connection = duckdb.connect()
    try:
        connection.register("wanted_sirets", wanted)
        select_columns = ", ".join(f"CAST(s.{column} AS VARCHAR) AS {column}" for column in SIRENE_COLUMNS)
        query = f"SELECT {select_columns} FROM read_parquet(?) AS s INNER JOIN wanted_sirets AS w ON CAST(s.siret AS VARCHAR) = w.siret"
        frame = connection.execute(query, [str(parquet_path)]).fetch_df()
        if locations is not None and not locations.empty:
            loc = locations[["insee", "cp"]].drop_duplicates()
            loc = loc[(loc["insee"] != "") | (loc["cp"] != "")]
            connection.register("wanted_locations", loc)
            query = f"""
                SELECT {select_columns}
                FROM read_parquet(?) AS s
                WHERE COALESCE(CAST(s.codeCommuneEtablissement AS VARCHAR), '') IN
                          (SELECT insee FROM wanted_locations WHERE insee <> '')
                   OR COALESCE(CAST(s.codePostalEtablissement AS VARCHAR), '') IN
                          (SELECT cp FROM wanted_locations WHERE cp <> '')
            """
            local_frame = connection.execute(query, [str(parquet_path)]).fetch_df()
            frame = pd.concat([frame, local_frame], ignore_index=True)
        frame = frame.fillna("").astype(str)
        frame = frame.drop_duplicates(subset=["siret"], keep="first")
        return frame.sort_values("siret").reset_index(drop=True)
    finally:
        connection.close()


def _query_legal_units(parquet_path: Path, sirens: Iterable[str]) -> dict[str, dict[str, str]]:
    wanted = pd.DataFrame({"siren": sorted({clean(value) for value in sirens if valid_siren(value)})})
    if wanted.empty:
        return {}
    connection = duckdb.connect()
    try:
        connection.register("wanted_sirens", wanted)
        select_columns = ", ".join(f"CAST(s.{column} AS VARCHAR) AS {column}" for column in UL_COLUMNS)
        query = f"""
            SELECT {select_columns}
            FROM read_parquet(?) AS s
            INNER JOIN wanted_sirens AS w ON CAST(s.siren AS VARCHAR) = w.siren
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY CAST(s.siren AS VARCHAR)
                ORDER BY CAST(s.dateDernierTraitementUniteLegale AS VARCHAR) DESC,
                         CAST(s.dateDernierTraitementUniteLegale AS VARCHAR) DESC
            ) = 1
        """
        frame = connection.execute(query, [str(parquet_path)]).fetch_df().fillna("").astype(str)
    finally:
        connection.close()
    return {
        clean(row["siren"]): {column: clean(row[column]) for column in UL_COLUMNS}
        for _, row in frame.iterrows()
        if valid_siren(row["siren"])
    }


def _query_sirene_only_seed_frame(
    parquet_path: Path,
    excluded_sirens: set[str],
    requested_rows: int,
) -> pd.DataFrame:
    """Select deterministic SIRENE-only seeds outside every CRM SIREN.

    The query deliberately uses only SIRENE identity and location fields.  It
    does not inspect retrieval artifacts, labels, ranks, scores, or the test
    final.  At most four establishments per new SIREN are retained so a
    future same-SIREN sibling can remain a useful hard negative without
    letting a single legal unit dominate the seed frame.
    """
    excluded = pd.DataFrame({"siren": sorted(value for value in excluded_sirens if valid_siren(value))})
    connection = duckdb.connect()
    try:
        connection.register("excluded_sirens", excluded)
        select_columns = ", ".join(f"CAST(s.{column} AS VARCHAR) AS {column}" for column in SIRENE_COLUMNS)
        query = f"""
            SELECT {select_columns}
            FROM read_parquet(?) AS s
            WHERE regexp_matches(CAST(s.siret AS VARCHAR), '^[0-9]{{14}}$')
              AND regexp_matches(CAST(s.siren AS VARCHAR), '^[0-9]{{9}}$')
              AND CAST(s.siren AS VARCHAR) NOT IN (SELECT siren FROM excluded_sirens)
              AND COALESCE(CAST(s.etatAdministratifEtablissement AS VARCHAR), '') IN ('A', 'F')
              AND COALESCE(CAST(s.codePostalEtablissement AS VARCHAR), '') <> ''
              AND (
                    COALESCE(CAST(s.codeCommuneEtablissement AS VARCHAR), '') <> ''
                    OR COALESCE(CAST(s.libelleCommuneEtablissement AS VARCHAR), '') <> ''
                  )
              AND (
                    COALESCE(CAST(s.denominationUsuelleEtablissement AS VARCHAR), '') <> ''
                    OR COALESCE(CAST(s.enseigne1Etablissement AS VARCHAR), '') <> ''
                    OR COALESCE(CAST(s.enseigne2Etablissement AS VARCHAR), '') <> ''
                    OR COALESCE(CAST(s.enseigne3Etablissement AS VARCHAR), '') <> ''
                  )
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY CAST(s.siren AS VARCHAR)
                ORDER BY md5(CAST(s.siret AS VARCHAR)), CAST(s.siret AS VARCHAR)
            ) <= 4
            ORDER BY md5(CAST(s.siret AS VARCHAR)), CAST(s.siret AS VARCHAR)
            LIMIT ?
        """
        frame = connection.execute(query, [str(parquet_path), int(requested_rows)]).fetch_df()
    finally:
        connection.close()
    return frame.fillna("").astype(str).drop_duplicates(subset=["siret"]).sort_values("siret").reset_index(drop=True)


def make_sirene_only_seed_rows(frame: pd.DataFrame, target_records: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, raw in frame.sort_values("siret").iterrows():
        target = target_records[clean(raw["siret"])]
        siren = target["siren"]
        fold = 2 + int.from_bytes(hashlib.sha256(f"SIRENE_ONLY|{siren}".encode("utf-8")).digest()[:4], "big") % 3
        rows.append(
            {
                "query_id": f"SIRENE_ONLY:{target['siret']}",
                "crm_name": target_official_name(target),
                "crm_adresse": target["address"],
                "crm_cp": target["postcode"],
                "crm_commune": target["city"],
                "crm_insee": target["insee"],
                "gt_siret": target["siret"],
                "oof_fold": fold,
                "siren_component_id": f"SIRENE_ONLY:{siren}",
                "legacy_split": "train_synthetic",
                "seed_source": "SIRENE_ONLY_TRAIN",
            }
        )
    return pd.DataFrame(rows)


def sirene_record(row: pd.Series, ul: dict[str, str] | None = None) -> dict[str, Any]:
    ul = ul or {}
    result: dict[str, Any] = {
        "siret": clean(row.get("siret")),
        "siren": clean(row.get("siren")),
        "state": clean(row.get("etatAdministratifEtablissement")),
        "date_debut": clean(row.get("dateDebut")),
        "snapshot_updated": clean(row.get("dateDernierTraitementEtablissement")),
        "numero": clean(row.get("numeroVoieEtablissement")),
        "indice": clean(row.get("indiceRepetitionEtablissement")),
        "type_voie": clean(row.get("typeVoieEtablissement")),
        "voie": clean(row.get("libelleVoieEtablissement")),
        "postcode": clean(row.get("codePostalEtablissement")),
        "city": clean(row.get("libelleCommuneEtablissement")),
        "insee": clean(row.get("codeCommuneEtablissement")),
        "enseigne1": clean(row.get("enseigne1Etablissement")),
        "enseigne2": clean(row.get("enseigne2Etablissement")),
        "enseigne3": clean(row.get("enseigne3Etablissement")),
        "denomination_usuelle": clean(row.get("denominationUsuelleEtablissement")),
        "legal_denomination": clean(ul.get("denominationUniteLegale")),
        "legal_usual_1": clean(ul.get("denominationUsuelle1UniteLegale")),
        "legal_usual_2": clean(ul.get("denominationUsuelle2UniteLegale")),
        "legal_usual_3": clean(ul.get("denominationUsuelle3UniteLegale")),
        "sigle": clean(ul.get("sigleUniteLegale")),
        "legal_category": clean(ul.get("categorieJuridiqueUniteLegale")),
    }
    result["names"] = [
        value
        for value in [
            result["denomination_usuelle"],
            result["enseigne1"],
            result["enseigne2"],
            result["enseigne3"],
            result["legal_denomination"],
            result["legal_usual_1"],
            result["legal_usual_2"],
            result["legal_usual_3"],
            result["sigle"],
        ]
        if value
    ]
    result["address"] = " ".join(
        value
        for value in [
            result["numero"],
            result["indice"],
            result["type_voie"],
            result["voie"],
            result["postcode"],
            result["city"],
        ]
        if value
    )
    result["address_signature"] = address_signature(result)
    result["street_signature"] = street_signature(result)
    return result


def build_record_maps(frame: pd.DataFrame, legal_units: dict[str, dict[str, str]]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for _, row in frame.iterrows():
        record = sirene_record(row, legal_units.get(clean(row.get("siren"))))
        if valid_siret(record["siret"]):
            if record["siret"] in records and records[record["siret"]] != record:
                raise RuntimeError(f"duplicate non-identical SIRET: {record['siret']}")
            records[record["siret"]] = record
    return records


def target_official_name(target: dict[str, Any]) -> str:
    for value in [target.get("legal_denomination"), target.get("denomination_usuelle"), target.get("enseigne1"), target.get("legal_usual_1"), target.get("sigle")]:
        if clean(value):
            return clean(value)
    return ""


def target_enseigne(target: dict[str, Any]) -> str:
    for value in [target.get("enseigne1"), target.get("enseigne2"), target.get("enseigne3"), target.get("denomination_usuelle"), target.get("legal_usual_1")]:
        if clean(value):
            return clean(value)
    return ""


def choose_base(seed: pd.Series, target: dict[str, Any], base_view: str) -> dict[str, str]:
    crm_name = clean(seed.get("crm_name"))
    crm_address = clean(seed.get("crm_adresse"))
    if base_view == "SIRENE_OFFICIAL_NAME":
        name = target_official_name(target) or crm_name
        address = crm_address or target["address"]
    elif base_view == "SIRENE_OFFICIAL_ENSEIGNE":
        name = target_enseigne(target) or target_official_name(target) or crm_name
        address = crm_address or target["address"]
    elif base_view == "SIRENE_OFFICIAL_ADDRESS":
        name = crm_name or target_official_name(target)
        address = target["address"]
    else:
        name, address = crm_name, crm_address
    return {
        "name": name,
        "address": address,
        "postcode": clean(seed.get("crm_cp")) or target["postcode"],
        "city": clean(seed.get("crm_commune")) or target["city"],
        "insee": clean(seed.get("crm_insee")) or target["insee"],
    }


def _strip_accents(value: str) -> str:
    return "".join(char for char in unicodedata.normalize("NFKD", value) if not unicodedata.combining(char))


def transform_variant(base: dict[str, str], target: dict[str, Any], family: str, rng) -> tuple[dict[str, str], dict[str, Any], bool]:
    result = dict(base)
    params: dict[str, Any] = {}
    applied = False
    name = result["name"]
    address = result["address"]
    if family == "LEGAL_FORM":
        substitutions = (
            (r"\bSOCIETE ANONYME\b", "SA"),
            (r"\bSOCIETE A RESPONSABILITE LIMITEE\b", "SARL"),
            (r"\bSOCIETE PAR ACTIONS SIMPLIFIEE\b", "SAS"),
            (r"\bASSOCIATION\b", "ASSO"),
        )
        for pattern, replacement in substitutions:
            if re.search(pattern, name, flags=re.IGNORECASE):
                result["name"] = re.sub(pattern, replacement, name, flags=re.IGNORECASE)
                params = {"pattern": pattern, "replacement": replacement}
                applied = True
                break
    elif family == "ACRONYM_TOKENIZATION":
        name_tokens = [token for token in re.split(r"[\s,;:/()\-]+", name) if token]
        informative = [token for token in name_tokens if len(normalize_text(token)) >= 2]
        if len(informative) >= 2:
            initials = "".join(token[0] for token in informative[: min(5, len(informative))]).upper()
            result["name"] = initials
            params = {"token_count": len(informative), "kept_initials": len(initials)}
            applied = result["name"] != name
    elif family == "ACCENT_PUNCTUATION":
        transformed_name = re.sub(r"[()\[\]{},;:!?/'\"._-]+", " ", _strip_accents(name)).upper()
        transformed_address = re.sub(r"[()\[\]{},;:!?/'\"._-]+", " ", _strip_accents(address)).upper()
        result["name"] = re.sub(r"\s+", " ", transformed_name).strip()
        result["address"] = re.sub(r"\s+", " ", transformed_address).strip()
        params = {"accents": "removed", "punctuation": "spaced", "case": "upper"}
        applied = result["name"] != name or result["address"] != address
    elif family == "OCR_LIMITED":
        substitutions = {"O": "0", "I": "1", "S": "5", "B": "8"}
        candidates = [index for index, char in enumerate(name.upper()) if char in substitutions]
        if candidates:
            index = candidates[rng.randrange(len(candidates))]
            original = name[index]
            result["name"] = name[:index] + substitutions[original.upper()] + name[index + 1 :]
            params = {"field": "name", "position": index, "from": original, "to": substitutions[original.upper()]}
            applied = result["name"] != name
    elif family == "TOKEN_ORDER":
        parts = [part for part in re.split(r"\s+", name.strip()) if part]
        if len(parts) >= 3:
            shift = 1 + (rng.randrange(len(parts) - 1))
            result["name"] = " ".join(parts[shift:] + parts[:shift])
            params = {"rotation": shift}
            applied = result["name"] != name
    elif family == "FIELD_MISSING":
        if result["city"] and (result["postcode"] or result["insee"]):
            result["city"] = ""
            params = {"removed": "city"}
            applied = True
        elif result["address"] and result["name"]:
            result["address"] = ""
            params = {"removed": "address"}
            applied = True
    elif family == "ADDRESS_ABBREVIATION":
        replacements = {
            r"\bAVENUE\b": "AV",
            r"\bBOULEVARD\b": "BD",
            r"\bRUE\b": "R",
            r"\bCHEMIN\b": "CHE",
            r"\bPLACE\b": "PL",
        }
        for pattern, replacement in replacements.items():
            if re.search(pattern, address, flags=re.IGNORECASE):
                result["address"] = re.sub(pattern, replacement, address, flags=re.IGNORECASE)
                params = {"pattern": pattern, "replacement": replacement}
                applied = True
                break
    elif family == "ADDRESS_TOKEN_ORDER":
        parts = [part for part in re.split(r"\s+", address.strip()) if part]
        if len(parts) >= 4:
            result["address"] = " ".join(parts[1:] + parts[:1])
            params = {"rotation": 1}
            applied = result["address"] != address
    elif family == "ADDRESS_OCR":
        parts = address.split()
        eligible = [index for index, part in enumerate(parts) if not part.isdecimal() and len(part) >= 4]
        if eligible:
            part_index = eligible[rng.randrange(len(eligible))]
            part = parts[part_index]
            replacements = {"O": "0", "I": "1", "S": "5", "B": "8"}
            positions = [index for index, char in enumerate(part.upper()) if char in replacements]
            if positions:
                char_index = positions[rng.randrange(len(positions))]
                original = part[char_index]
                parts[part_index] = part[:char_index] + replacements[original.upper()] + part[char_index + 1 :]
                result["address"] = " ".join(parts)
                params = {"field": "address", "token": part_index, "position": char_index}
                applied = result["address"] != address
    elif family == "COMMUNE_VARIANT":
        transformed = _strip_accents(result["city"]).upper().replace("-", " ")
        transformed = re.sub(r"\s+", " ", transformed).strip()
        if transformed and transformed != result["city"]:
            result["city"] = transformed
            params = {"accents": "removed", "hyphen": "space", "case": "upper"}
            applied = True
    elif family == "ENSEIGNE_VS_DENOMINATION":
        alternate = target_enseigne(target) if normalize_text(name) != normalize_text(target_enseigne(target)) else target_official_name(target)
        if alternate and normalize_text(alternate) != normalize_text(name):
            result["name"] = alternate
            params = {"replacement": "official_enseigne_or_denomination"}
            applied = True
    return result, params, applied


def deterministic_guard(query: dict[str, str], target: dict[str, Any], candidates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if any(autonomous_decimal_leak(query[field]) for field in ("name", "address", "postcode", "city", "insee")):
        return {"status": "REJECT", "reason": "AUTONOMOUS_DECIMAL_LEAK", "margin": -1.0, "best_other_siret": ""}
    if not (query["postcode"] or query["insee"] or query["city"]):
        return {"status": "REJECT", "reason": "NO_GEOGRAPHIC_ANCHOR", "margin": -1.0, "best_other_siret": ""}
    if not (tokens(query["name"]) or tokens(query["address"])):
        return {"status": "REJECT", "reason": "NO_TEXT_SIGNAL", "margin": -1.0, "best_other_siret": ""}

    def score(record: dict[str, Any]) -> float:
        location = 0.0
        if query["insee"] and query["insee"] == record["insee"]:
            location += 2.0
        elif query["postcode"] and query["postcode"] == record["postcode"]:
            location += 1.0
        name_signal = max_name_similarity(query["name"], record["names"])
        address_signal = max(token_jaccard(query["address"], record["address"]), text_ratio(query["address"], record["address"]))
        city_signal = 1.0 if query["city"] and normalize_text(query["city"]) == normalize_text(record["city"]) else 0.0
        return location + 2.0 * name_signal + 2.0 * address_signal + 0.5 * city_signal

    target_score = score(target)
    best_other_score = -1.0
    best_other_siret = ""
    for siret, record in candidates.items():
        if siret == target["siret"]:
            continue
        candidate_score = score(record)
        if candidate_score > best_other_score or (candidate_score == best_other_score and siret < best_other_siret):
            best_other_score, best_other_siret = candidate_score, siret
    margin = target_score - max(best_other_score, 0.0)
    if target_score < 1.35:
        return {"status": "REJECT", "reason": "TARGET_SIGNAL_TOO_WEAK", "margin": margin, "best_other_siret": best_other_siret}
    if best_other_score >= 0.0 and margin < 0.15:
        return {"status": "REJECT", "reason": "LOCAL_COMPETITOR_NOT_SEPARATED", "margin": margin, "best_other_siret": best_other_siret}
    return {"status": "PASS", "reason": "OK", "margin": margin, "best_other_siret": best_other_siret}


def classify_negative(target: dict[str, Any], candidate: dict[str, Any]) -> str:
    if candidate["siren"] == target["siren"]:
        return "SAME_SIREN_OTHER_SIRET"
    if target["insee"] and target["insee"] == candidate["insee"]:
        if max_name_similarity(target_official_name(target), candidate["names"]) >= 0.90:
            return "LOCAL_HOMONYM"
        if target["address_signature"] and target["address_signature"] == candidate["address_signature"]:
            return "SHARED_ADDRESS"
        if target["state"] and candidate["state"] and target["state"] != candidate["state"]:
            return "ACTIVE_CLOSED_COMPETITOR"
    if target["postcode"] and target["postcode"] == candidate["postcode"]:
        if target["street_signature"] and target["street_signature"] == candidate["street_signature"]:
            return "TOPOLOGICAL_NEARBY"
    return "LOCAL_FILL"


NEGATIVE_QUOTAS = {
    "SAME_SIREN_OTHER_SIRET": 2,
    "LOCAL_HOMONYM": 2,
    "SHARED_ADDRESS": 2,
    "ACTIVE_CLOSED_COMPETITOR": 2,
    "TOPOLOGICAL_NEARBY": 1,
    "LOCAL_FILL": 1,
}


def build_candidate_group(
    example_id: str,
    target: dict[str, Any],
    candidates: dict[str, dict[str, Any]],
    forbidden_sirens: set[str],
    source_snapshot_sha256: str,
    seed: pd.Series,
) -> list[dict[str, Any]]:
    allowed_candidates = {
        siret: record
        for siret, record in candidates.items()
        if record["siren"] not in forbidden_sirens or record["siren"] == target["siren"]
    }
    if target["siret"] not in allowed_candidates:
        raise RuntimeError("positive candidate is absent after forbidden-SIREN filtering")
    rows = [
        {
            "example_id": example_id,
            "seed_source": clean(seed.get("seed_source")) or "CRM_OK_GT_TRAIN",
            "candidate_siret": target["siret"],
            "candidate_siren": target["siren"],
            "is_positive": True,
            "negative_category": "POSITIVE_SIRENE_PRESENT",
            "candidate_state": target["state"],
            "candidate_name": target_official_name(target),
            "candidate_address": target["address"],
            "candidate_postcode": target["postcode"],
            "candidate_city": target["city"],
            "candidate_insee": target["insee"],
            "target_siret": target["siret"],
            "target_siren": target["siren"],
            "source_kind": "SIRENE_SYNTHETIC",
            "source_snapshot_sha256": source_snapshot_sha256,
            "oof_fold": int(seed["oof_fold"]),
            "siren_component_id": clean(seed["siren_component_id"]),
            "candidate_provenance_digest": sha256_bytes(canonical_json({"siret": target["siret"], "snapshot": source_snapshot_sha256}).encode()),
        }
    ]
    by_category: dict[str, list[str]] = {category: [] for category in NEGATIVE_QUOTAS}
    for siret, candidate in allowed_candidates.items():
        if siret == target["siret"]:
            continue
        category = classify_negative(target, candidate)
        by_category[category].append(siret)
    for category in by_category:
        by_category[category].sort()
    used: set[str] = {target["siret"]}
    for category, quota in NEGATIVE_QUOTAS.items():
        selected = 0
        for siret in by_category[category]:
            if siret in used:
                continue
            candidate = allowed_candidates[siret]
            rows.append(
                {
            "example_id": example_id,
            "seed_source": clean(seed.get("seed_source")) or "CRM_OK_GT_TRAIN",
                    "candidate_siret": candidate["siret"],
                    "candidate_siren": candidate["siren"],
                    "is_positive": False,
                    "negative_category": category,
                    "candidate_state": candidate["state"],
                    "candidate_name": " | ".join(candidate["names"][:3]),
                    "candidate_address": candidate["address"],
                    "candidate_postcode": candidate["postcode"],
                    "candidate_city": candidate["city"],
                    "candidate_insee": candidate["insee"],
                    "target_siret": target["siret"],
                    "target_siren": target["siren"],
                    "source_kind": "SIRENE_SYNTHETIC",
                    "source_snapshot_sha256": source_snapshot_sha256,
                    "oof_fold": int(seed["oof_fold"]),
                    "siren_component_id": clean(seed["siren_component_id"]),
                    "candidate_provenance_digest": sha256_bytes(canonical_json({"siret": candidate["siret"], "snapshot": source_snapshot_sha256, "category": category}).encode()),
                }
            )
            used.add(siret)
            selected += 1
            if selected >= quota or len(rows) >= 11:
                break
        if len(rows) >= 11:
            break
    return rows


def deterministic_digest(fields: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(fields).encode("utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def write_typed_parquet(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...], types: dict[str, pa.DataType]) -> None:
    arrays = []
    fields = []
    for column in columns:
        values = [row.get(column, "") for row in rows]
        arrays.append(pa.array(values, type=types[column]))
        fields.append(pa.field(column, types[column], nullable=False))
    table = pa.Table.from_arrays(arrays, schema=pa.schema(fields))
    pq.write_table(
        table,
        path,
        version="2.6",
        compression="zstd",
        compression_level=9,
        use_dictionary=False,
        write_statistics=True,
        row_group_size=65536,
        coerce_timestamps=None,
    )


def make_distribution_report(seeds: pd.DataFrame, queries: list[dict[str, Any]]) -> dict[str, Any]:
    def stats(values: Iterable[str]) -> dict[str, float]:
        values = [clean(value) for value in values]
        if not values:
            return {"rows": 0, "mean_chars": 0.0, "p50_chars": 0.0, "missing_rate": 0.0}
        lengths = sorted(len(value) for value in values)
        return {
            "rows": len(values),
            "mean_chars": round(sum(lengths) / len(lengths), 6),
            "p50_chars": float(lengths[(len(lengths) - 1) // 2]),
            "missing_rate": round(sum(not value for value in values) / len(values), 6),
        }

    base = {
        "name": stats(seeds["crm_name"].tolist()),
        "address": stats(seeds["crm_adresse"].tolist()),
        "postcode": stats(seeds["crm_cp"].tolist()),
        "city": stats(seeds["crm_commune"].tolist()),
        "insee": stats(seeds["crm_insee"].tolist()),
    }
    synthetic = {
        field: stats([clean(row[field]) for row in queries])
        for field in ("crm_name", "crm_address", "crm_postcode", "crm_city", "crm_insee")
    }
    return {
        "source_train_distribution": base,
        "synthetic_query_distribution": synthetic,
        "comparison": {
            field: {
                metric: round(synthetic[field][metric] - base[source_field][metric], 6)
                for metric in ("mean_chars", "p50_chars", "missing_rate")
            }
            for field, source_field in {
                "name": "name",
                "address": "address",
                "postcode": "postcode",
                "city": "city",
                "insee": "insee",
            }.items()
        },
    }


def build(args: argparse.Namespace) -> Path:
    plan_path = args.plan.resolve()
    plan = load_plan(plan_path)
    verified = verify_source_pins(plan, plan_path)
    crm_seeds, forbidden_components, forbidden_sirens, all_crm_sirens = load_train_seeds(plan, verified)
    crm_seeds = crm_seeds.sort_values(["oof_fold", "query_id"]).reset_index(drop=True)
    requested_seed_count = args.target_seed_count
    if requested_seed_count is None:
        requested_seed_count = int(plan["quality_gates"]["pilot_seed_count"] if args.pilot else plan["generator"]["minimum_seed_sirets"])
    include_sirene_only = bool(args.include_sirene_only and not args.crm_only)
    if args.pilot and args.crm_only:
        include_sirene_only = False
    if not include_sirene_only:
        if requested_seed_count > len(crm_seeds):
            raise RuntimeError("requested seed count exceeds CRM train seeds while SIRENE-only extension is disabled")
        crm_seeds = crm_seeds.head(requested_seed_count).copy()

    sirene_path = Path(verified["sirene_establishments"]["path"])
    ul_path = Path(verified["sirene_legal_units"]["path"])
    crm_target_sirets = crm_seeds["gt_siret"].tolist()
    source_only_frame = pd.DataFrame(columns=list(SIRENE_COLUMNS))
    if include_sirene_only and requested_seed_count > len(crm_seeds):
        missing_seed_count = max(0, requested_seed_count - len(crm_seeds))
        source_only_frame = _query_sirene_only_seed_frame(
            sirene_path,
            all_crm_sirens,
            max(missing_seed_count * 3, missing_seed_count + 1000),
        )
    source_only_sirets = source_only_frame["siret"].tolist()
    if requested_seed_count <= len(crm_seeds):
        source_only_frame = source_only_frame.iloc[0:0].copy()
        source_only_sirets = []
    target_sirets = crm_target_sirets + source_only_sirets
    if not target_sirets:
        raise RuntimeError("no seeds selected")
    target_frame = _query_sirene(sirene_path, target_sirets)
    missing_targets = sorted(set(target_sirets) - set(target_frame["siret"]))
    if missing_targets:
        raise RuntimeError(f"target SIRETs missing from SIRENE: {missing_targets[:5]}")
    target_sirens = sorted({value[:9] for value in target_sirets})
    legal_units = _query_legal_units(ul_path, target_sirens)
    target_records = build_record_maps(target_frame, legal_units)
    if len(target_records) != len(target_sirets):
        raise RuntimeError("target SIRETs are not unique in the SIRENE snapshot")
    source_only_seeds = make_sirene_only_seed_rows(source_only_frame, target_records)
    if requested_seed_count and len(crm_seeds) + len(source_only_seeds) > requested_seed_count:
        source_only_seeds = source_only_seeds.head(max(0, requested_seed_count - len(crm_seeds))).copy()
    if requested_seed_count <= len(crm_seeds):
        crm_seeds = crm_seeds.head(requested_seed_count).copy()
    seeds = pd.concat([crm_seeds, source_only_seeds], ignore_index=True)
    seeds = seeds.sort_values(["oof_fold", "seed_source", "query_id"]).reset_index(drop=True)
    target_sirets = seeds["gt_siret"].tolist()
    if seeds.empty:
        raise RuntimeError("no seeds selected after SIRENE-only filtering")
    if set(seeds["gt_siret"].str[:9]) & forbidden_sirens:
        raise RuntimeError("selected seed SIREN overlaps forbidden SIREN")

    target_locations = pd.DataFrame(
        [
            {
                "insee": target_records[siret]["insee"] or clean(row["crm_insee"]),
                "cp": target_records[siret]["postcode"] or clean(row["crm_cp"]),
            }
            for siret, (_, row) in zip(target_sirets, seeds.iterrows(), strict=True)
        ]
    )
    local_frame = _query_sirene(sirene_path, target_sirets, target_locations)
    local_records = build_record_maps(local_frame, legal_units)
    for siret, target in target_records.items():
        local_records.setdefault(siret, target)
    by_insee: dict[str, list[str]] = {}
    by_cp: dict[str, list[str]] = {}
    for siret, record in local_records.items():
        if record["insee"]:
            by_insee.setdefault(record["insee"], []).append(siret)
        if record["postcode"]:
            by_cp.setdefault(record["postcode"], []).append(siret)
    for mapping in (by_insee, by_cp):
        for key in mapping:
            mapping[key].sort()

    families = tuple(plan["transformations"]["families"])
    if set(families) != set(FAMILY_ORDER):
        raise RuntimeError("plan family set differs from the versioned generator")
    global_seed = int(plan["generator"]["seed"])
    variants_per_seed = int(args.variants_per_seed)
    if variants_per_seed < int(plan["generator"]["minimum_variants_per_seed"]):
        raise ValueError("variants_per_seed is below the pre-registered minimum")
    query_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []
    seen_queries: set[str] = set()
    seen_input_signatures: set[str] = set()
    family_counts: dict[str, int] = {family: 0 for family in families}
    seed_lookup = {clean(row["gt_siret"]): row for _, row in seeds.iterrows()}
    source_snapshot_sha = verified["sirene_establishments"]["sha256"]
    min_negatives = int(plan["quality_gates"]["min_negatives_per_positive"])
    max_group_candidates = int(plan["quality_gates"]["max_group_candidates"])

    for target_siret, seed_row in seed_lookup.items():
        target = target_records[target_siret]
        if target["siren"] in forbidden_sirens:
            raise RuntimeError("target SIREN is forbidden")
        candidate_ids = set(by_insee.get(target["insee"], [])) | set(by_cp.get(target["postcode"], []))
        candidate_ids.add(target_siret)
        candidates = {siret: local_records[siret] for siret in sorted(candidate_ids) if siret in local_records}
        if target_siret not in candidates:
            raise RuntimeError("target candidate missing from local candidate map")
        for variant_index in range(variants_per_seed):
            family = families[(int.from_bytes(hashlib.sha256(f"{global_seed}|{seed_row['query_id']}|{variant_index}".encode()).digest()[:4], "big")) % len(families)]
            rng = deterministic_rng(global_seed, clean(seed_row["query_id"]), family, variant_index)
            available_views = ["CRM_OBSERVED", "SIRENE_OFFICIAL_NAME"]
            if target_enseigne(target):
                available_views.append("SIRENE_OFFICIAL_ENSEIGNE")
            if target["address"]:
                available_views.append("SIRENE_OFFICIAL_ADDRESS")
            base_view = available_views[rng.randrange(len(available_views))]
            base = choose_base(seed_row, target, base_view)
            transformed, parameters, applied = transform_variant(base, target, family, rng)
            if not applied:
                rejection_rows.append({"seed_query_id": clean(seed_row["query_id"]), "variant_index": str(variant_index), "corruption_family": family, "reason": "NOT_APPLICABLE", "base_view": base_view, "target_siret": target_siret, "oof_fold": str(seed_row["oof_fold"]), "siren_component_id": clean(seed_row["siren_component_id"]), "detail_json": canonical_json({"parameters": parameters})})
                continue
            guard = deterministic_guard(transformed, target, candidates)
            if guard["status"] != "PASS":
                rejection_rows.append({"seed_query_id": clean(seed_row["query_id"]), "variant_index": str(variant_index), "corruption_family": family, "reason": guard["reason"], "base_view": base_view, "target_siret": target_siret, "oof_fold": str(seed_row["oof_fold"]), "siren_component_id": clean(seed_row["siren_component_id"]), "detail_json": canonical_json({"parameters": parameters, "margin": guard["margin"], "best_other_siret": guard["best_other_siret"]})})
                continue
            identity = {"seed_query_id": clean(seed_row["query_id"]), "seed_source": clean(seed_row["seed_source"]), "family": family, "variant_index": variant_index, "name": transformed["name"], "address": transformed["address"], "postcode": transformed["postcode"], "city": transformed["city"], "insee": transformed["insee"], "target_siret": target_siret, "source_kind": "SIRENE_SYNTHETIC"}
            query_digest = deterministic_digest(identity)
            input_signature = deterministic_digest({key: transformed[key] for key in ("name", "address", "postcode", "city", "insee")})
            if query_digest in seen_queries or input_signature in seen_input_signatures:
                rejection_rows.append({"seed_query_id": clean(seed_row["query_id"]), "variant_index": str(variant_index), "corruption_family": family, "reason": "DUPLICATE_GENERATED_QUERY", "base_view": base_view, "target_siret": target_siret, "oof_fold": str(seed_row["oof_fold"]), "siren_component_id": clean(seed_row["siren_component_id"]), "detail_json": canonical_json({"parameters": parameters})})
                continue
            group_example_id = sha256_bytes(f"SIRENE_SYNTHETIC\0{query_digest}".encode("utf-8"))
            group_rows = build_candidate_group(group_example_id, target, candidates, forbidden_sirens, source_snapshot_sha, seed_row)
            if len(group_rows) - 1 < min_negatives or len(group_rows) > max_group_candidates:
                rejection_rows.append({"seed_query_id": clean(seed_row["query_id"]), "variant_index": str(variant_index), "corruption_family": family, "reason": "INSUFFICIENT_HARD_NEGATIVES", "base_view": base_view, "target_siret": target_siret, "oof_fold": str(seed_row["oof_fold"]), "siren_component_id": clean(seed_row["siren_component_id"]), "detail_json": canonical_json({"negative_count": len(group_rows) - 1, "required_min": min_negatives, "max_group": max_group_candidates})})
                continue
            seen_queries.add(query_digest)
            seen_input_signatures.add(input_signature)
            confidence = max(0.5, min(1.0, 0.75 + min(0.25, max(0.0, guard["margin"]) / 4.0)))
            query_rows.append({"example_id": group_example_id, "seed_query_id": clean(seed_row["query_id"]), "seed_source": clean(seed_row["seed_source"]), "source_kind": "SIRENE_SYNTHETIC", "base_view": base_view, "crm_name": transformed["name"], "crm_address": transformed["address"], "crm_postcode": transformed["postcode"], "crm_city": transformed["city"], "crm_insee": transformed["insee"], "target_siret": target_siret, "target_siren": target["siren"], "target_state": target["state"], "oof_fold": int(seed_row["oof_fold"]), "siren_component_id": clean(seed_row["siren_component_id"]), "corruption_family": family, "variant_index": variant_index, "confidence_weight": confidence, "guard_status": "PASS", "guard_margin": float(guard["margin"]), "best_other_siret": guard["best_other_siret"], "provenance_digest": deterministic_digest({"source": verified, "parameters": parameters, "identity": identity}), "generator_version": plan["generator"]["version"]})
            family_counts[family] += 1
            pair_rows.extend(group_rows)

    if not query_rows:
        raise RuntimeError("no publishable synthetic queries")
    query_frame = pd.DataFrame(query_rows)
    pair_frame = pd.DataFrame(pair_rows)
    if query_frame["example_id"].duplicated().any():
        raise RuntimeError("duplicate example IDs")
    grouped = pair_frame.groupby("example_id")
    if grouped["candidate_siret"].nunique().max() > max_group_candidates:
        raise RuntimeError("candidate group exceeds the pre-registered maximum")
    if grouped["is_positive"].sum().ne(1).any() or (grouped.size() - grouped["is_positive"].sum()).lt(min_negatives).any():
        raise RuntimeError("candidate group violates positive/negative cardinality")
    positive_by_example = pair_frame[pair_frame["is_positive"]].set_index("example_id")["candidate_siret"].to_dict()
    if any(positive_by_example[row["example_id"]] != row["target_siret"] for _, row in query_frame.iterrows()):
        raise RuntimeError("positive candidate does not match query truth")
    if set(query_frame["target_siren"]) & forbidden_sirens:
        raise RuntimeError("published target SIREN overlaps forbidden SIREN")
    if any(not valid_siret(value) or not valid_siren(value) for value in query_frame["target_siret"]):
        raise RuntimeError("published query identity is invalid")
    if any(autonomous_decimal_leak(value) for column in ("crm_name", "crm_address", "crm_postcode", "crm_city", "crm_insee") for value in query_frame[column]):
        raise RuntimeError("published CRM text contains an autonomous decimal leak")

    plan_hash = sha256_bytes(plan_path.read_bytes())
    builder_hash = file_sha256(Path(__file__).resolve())
    tests_hash = file_sha256(args.tests.resolve()) if args.tests.exists() else ""
    build_identity = {"schema_version": plan["schema_version"], "plan_sha256": plan_hash, "builder_sha256": builder_hash, "tests_sha256": tests_hash, "seed": global_seed, "seed_count_requested": int(requested_seed_count), "seed_count_used": len(seeds), "variants_per_seed": variants_per_seed, "include_sirene_only": include_sirene_only, "sources": {name: record["sha256"] for name, record in sorted(verified.items())}}
    build_id = sha256_bytes(canonical_json(build_identity).encode())[:16]
    output_root = Path(plan["generator"]["output_root"])
    destination = output_root / ("pilot" if args.pilot else "full") / build_id
    if destination.exists():
        manifest_path = destination / "manifest.json"
        if not manifest_path.is_file():
            raise FileExistsError(f"existing incomplete build: {destination}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("build_identity") != build_identity:
            raise FileExistsError(f"conflicting build identity: {destination}")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{build_id}.", dir=destination.parent))
    try:
        source_manifest = {"schema_version": "sireto-synthetic-gt-corpus-source-manifest-1", "build_identity": build_identity, "verified_sources": verified, "population": {"seed_rows_used": len(seeds), "seed_source_counts": {str(key): int(value) for key, value in seeds["seed_source"].value_counts().sort_index().items()}, "forbidden_component_count": len(forbidden_components), "forbidden_siren_count": len(forbidden_sirens), "all_crm_siren_count": len(all_crm_sirens)}, "maps_assisted": {"enabled": False, "calls_made": 0}}
        write_json(staging / "source_manifest.json", source_manifest)
        write_typed_parquet(staging / "synthetic_queries.parquet", query_rows, QUERY_COLUMNS, FIELD_TYPES)
        write_typed_parquet(staging / "candidate_pairs.parquet", pair_rows, PAIR_COLUMNS, PAIR_TYPES)
        write_typed_parquet(staging / "rejection_ledger.parquet", rejection_rows, REJECTION_COLUMNS, REJECTION_TYPES)
        distribution_report = make_distribution_report(seeds, query_rows)
        write_json(staging / "distribution_report.json", distribution_report)
        family_share = {family: count / len(query_rows) for family, count in family_counts.items()}
        unique_input_rate = len(seen_input_signatures) / max(len(query_rows), 1)
        distinct_seed_count = int(query_frame["target_siret"].nunique())
        variants_per_seed_observed = query_frame.groupby("target_siret").size()
        negative_counts = pair_frame.loc[~pair_frame["is_positive"], "negative_category"].value_counts().sort_index()
        minimums_pass = distinct_seed_count >= int(plan["generator"]["minimum_seed_sirets"]) and len(query_rows) >= int(plan["generator"]["minimum_positive_pairs"]) and int(variants_per_seed_observed.min()) >= int(plan["generator"]["minimum_variants_per_seed"]) and int((grouped.size() - grouped["is_positive"].sum()).min()) >= min_negatives
        quality = {"seed_count": len(seeds), "distinct_seed_siret_count": distinct_seed_count, "seed_source_counts": {str(key): int(value) for key, value in seeds["seed_source"].value_counts().sort_index().items()}, "query_count": len(query_rows), "positive_count": len(query_rows), "pair_count": len(pair_rows), "negative_count": int((~pair_frame["is_positive"]).sum()), "rejection_count": len(rejection_rows), "family_counts": family_counts, "family_share": family_share, "unique_input_rate": round(unique_input_rate, 8), "duplicate_input_rate": round(1.0 - unique_input_rate, 8), "variants_per_seed_min": int(variants_per_seed_observed.min()), "variants_per_seed_p50": float(variants_per_seed_observed.median()), "negative_per_positive_min": int((grouped.size() - grouped["is_positive"].sum()).min()), "negative_per_positive_mean": round(float((grouped.size() - grouped["is_positive"].sum()).mean()), 6), "negative_category_counts": {str(key): int(value) for key, value in negative_counts.items()}, "target_state_counts": {str(key): int(value) for key, value in query_frame["target_state"].value_counts().sort_index().items()}, "guard_status_counts": {str(key): int(value) for key, value in query_frame["guard_status"].value_counts().items()}, "minimum_objectives_pass": bool(minimums_pass), "verdict": "GO_PILOT" if args.pilot else ("GO" if minimums_pass else "PIVOT")}
        write_json(staging / "quality_report.json", quality)
        output_names = ["synthetic_queries.parquet", "candidate_pairs.parquet", "rejection_ledger.parquet", "distribution_report.json", "quality_report.json", "source_manifest.json"]
        output_hashes = {name: file_sha256(staging / name) for name in output_names}
        manifest = {"schema_version": "sireto-synthetic-gt-corpus-manifest-1", "build_id": build_id, "build_identity": build_identity, "generator": {"path": str(Path(__file__).resolve()), "version": plan["generator"]["version"], "seed": global_seed}, "source_hashes": {name: record["sha256"] for name, record in sorted(verified.items())}, "population": {"seed_count": len(seeds), "seed_source_counts": {str(key): int(value) for key, value in seeds["seed_source"].value_counts().sort_index().items()}, "allowed_folds": plan["generator"]["allowed_oof_folds"], "allowed_split": plan["population"]["allowed_legacy_split"], "forbidden_component_count": len(forbidden_components), "forbidden_siren_count": len(forbidden_sirens)}, "quality": quality, "outputs": output_hashes, "maps_assisted": {"enabled": False, "calls_made": 0, "artifact_separate": True}}
        write_json(staging / "manifest.json", manifest)
        staging.rename(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PLAN_DEFAULT)
    parser.add_argument("--tests", type=Path, default=TEST_DEFAULT)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--target-seed-count", "--seed-count", dest="target_seed_count", type=int, default=None)
    parser.add_argument("--variants-per-seed", type=int, default=3)
    parser.add_argument("--include-sirene-only", action="store_true", default=True)
    parser.add_argument("--crm-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(build(parse_args()))
