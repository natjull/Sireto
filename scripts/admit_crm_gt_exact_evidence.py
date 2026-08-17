#!/usr/bin/env python3
"""Admit new CRM labels only with direct identity and exact-site evidence.

The admission is deliberately independent from retrieval. Municipality
agreement remains a prerequisite supplied by the upstream location triage,
but it is never sufficient to establish an exact SIRET label.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
import sys

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_benchmark_v3_evidence import audit_evidence_rows
from scripts.build_crm_gt_v2_population import sha256


SCHEMA_VERSION = "sireto-crm-gt-exact-admission-1"
_NUMBER_RE = re.compile(
    r"^\s*(?P<number>\d+)\s*(?P<repeat>BIS|TER|QUATER|[A-Z])?(?:\s+|$)",
    re.IGNORECASE,
)


def _clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    value = str(value).strip()
    return "" if value.lower() in {"nan", "none", "null"} else value


def _norm(value: object) -> str:
    value = unicodedata.normalize("NFKD", _clean(value))
    value = value.encode("ascii", "ignore").decode().upper()
    return " ".join("".join(c if c.isalnum() else " " for c in value).split())


def _repeat(value: object) -> str:
    value = _norm(value).replace(" ", "")
    return {"B": "B", "BIS": "B", "T": "TER", "TER": "TER"}.get(value, value)


def crm_number_and_repeat(address: object) -> tuple[str, str]:
    match = _NUMBER_RE.match(_clean(address))
    if not match:
        return "", ""
    return match.group("number") or "", _repeat(match.group("repeat") or "")


def strict_admission(evidence: pd.DataFrame) -> pd.DataFrame:
    """Return evidence with a fail-closed exact-SIRET admission decision."""
    out = evidence.copy()
    name_exact = out["name_norm_exact"].ge(1.0)
    name_fuzzy = out["name_jaro_max"].ge(0.90) & out[
        "name_token_overlap_max"
    ].ge(0.60)
    name_contained = (
        out[["name_contains_crm_max", "name_crm_contains_cand_max", "acronym_match_max"]]
        .max(axis=1)
        .ge(1.0)
        & out["name_jaro_max"].ge(0.85)
        & out["name_token_overlap_max"].ge(0.50)
    )
    out["strict_name_evidence"] = name_exact | name_fuzzy | name_contained

    parsed = out["crm_address"].map(crm_number_and_repeat)
    out["crm_number"] = parsed.map(lambda value: value[0])
    out["crm_repeat"] = parsed.map(lambda value: value[1])
    out["official_number"] = out["numeroVoie"].map(_clean)
    out["official_repeat"] = out["indiceRepetition"].map(_repeat)
    number_match = out["crm_number"].eq(out["official_number"])
    repeat_match = out["crm_repeat"].eq(out["official_repeat"])
    both_numberless = out["crm_number"].eq("") & out["official_number"].eq("")
    out["strict_number_evidence"] = (number_match & repeat_match) | both_numberless
    out["strict_location_evidence"] = (
        out["postcode_match"].ge(1.0)
        & out["insee_match"].ge(1.0)
        & out["street_name_jaro"].ge(0.95)
        & out["strict_number_evidence"]
    )
    out["admission_status"] = "QUARANTINE_INSUFFICIENT_DIRECT_EVIDENCE"
    admitted = out["strict_name_evidence"] & out["strict_location_evidence"]
    out.loc[admitted, "admission_status"] = "ADMIT_EXACT_SIRET"
    out["admission_reason"] = out.apply(
        lambda row: (
            "DIRECT_IDENTITY_AND_EXACT_SITE"
            if row["admission_status"] == "ADMIT_EXACT_SIRET"
            else "MISSING_STRICT_IDENTITY_EVIDENCE"
            if not row["strict_name_evidence"]
            else "MISSING_STRICT_SITE_EVIDENCE"
        ),
        axis=1,
    )
    return out


def _official_join(increment: Path, sirene: Path, legal_units: Path) -> pd.DataFrame:
    crm = pd.read_csv(increment, sep=";", dtype=str, keep_default_na=False)
    wanted = crm[["gt_siret"]].drop_duplicates().rename(columns={"gt_siret": "siret"})
    con = duckdb.connect()
    con.register("wanted", wanted)
    official = con.execute(
        """SELECT CAST(e.siret AS VARCHAR) gt_siret,
          CAST(e.siren AS VARCHAR) ground_truth_siren,
          e.denominationUsuelleEtablissement denomination,
          e.enseigne1Etablissement enseigne1,
          e.enseigne2Etablissement enseigne2,
          e.enseigne3Etablissement enseigne3,
          e.numeroVoieEtablissement numeroVoie,
          e.indiceRepetitionEtablissement indiceRepetition,
          e.typeVoieEtablissement typeVoie,
          e.libelleVoieEtablissement libelleVoie,
          e.complementAdresseEtablissement complementAdresse,
          e.codePostalEtablissement candidate_postcode,
          e.libelleCommuneEtablissement candidate_city,
          e.codeCommuneEtablissement candidate_insee,
          e.etatAdministratifEtablissement ground_truth_state,
          u.categorieJuridiqueUniteLegale cj_ul,
          u.sigleUniteLegale sigle_ul,
          u.denominationUniteLegale denomination_ul,
          u.denominationUsuelle1UniteLegale denomination_usuelle_ul,
          u.nomUniteLegale nom_ul,
          u.nomUsageUniteLegale nom_usage_ul,
          u.prenomUsuelUniteLegale prenom_usuel_ul,
          u.pseudonymeUniteLegale pseudonyme_ul
        FROM read_parquet(?) e
        JOIN wanted w ON CAST(e.siret AS VARCHAR) = w.siret
        LEFT JOIN read_parquet(?) u
          ON CAST(e.siren AS VARCHAR) = CAST(u.siren AS VARCHAR)""",
        [str(sirene), str(legal_units)],
    ).fetchdf()
    con.close()
    joined = crm.merge(official, on="gt_siret", validate="many_to_one")
    if len(joined) != len(crm):
        raise ValueError("Official SIRENE join is incomplete")
    joined["query_id"] = joined["crm_gt_fingerprint"].map(lambda value: f"NEWCRM:{value[:32]}")
    joined["split"] = "UNASSIGNED"
    joined["ground_truth_siret"] = joined["gt_siret"]
    joined["crm_address"] = joined["crm_adresse"]
    joined["crm_city"] = joined["crm_commune"]
    joined["postcode"] = joined["crm_cp"]
    joined["insee"] = joined["crm_insee"]
    joined["is_siege"] = False
    return joined


def _site_identity_support(
    joined: pd.DataFrame,
    initial_evidence: pd.DataFrame,
    sirene: Path,
    legal_units: Path,
) -> pd.DataFrame:
    """Enumerate every official SIRET at the target site and test CRM identity."""
    initially_admitted = set(
        initial_evidence.loc[
            initial_evidence["admission_status"].eq("ADMIT_EXACT_SIRET"), "query_id"
        ]
    )
    targets = joined[joined["query_id"].isin(initially_admitted)].copy()
    site_columns = [
        "query_id", "numeroVoie", "indiceRepetition", "typeVoie", "libelleVoie",
        "candidate_postcode", "candidate_insee",
    ]
    sites = targets[site_columns].drop_duplicates()
    con = duckdb.connect()
    con.register("sites", sites)
    candidates = con.execute(
        """SELECT s.query_id root_query_id,
          CAST(e.siret AS VARCHAR) candidate_siret,
          CAST(e.siren AS VARCHAR) ground_truth_siren,
          e.denominationUsuelleEtablissement denomination,
          e.enseigne1Etablissement enseigne1,
          e.enseigne2Etablissement enseigne2,
          e.enseigne3Etablissement enseigne3,
          e.numeroVoieEtablissement numeroVoie,
          e.indiceRepetitionEtablissement indiceRepetition,
          e.typeVoieEtablissement typeVoie,
          e.libelleVoieEtablissement libelleVoie,
          e.complementAdresseEtablissement complementAdresse,
          e.codePostalEtablissement candidate_postcode,
          e.libelleCommuneEtablissement candidate_city,
          e.codeCommuneEtablissement candidate_insee,
          e.etatAdministratifEtablissement ground_truth_state,
          u.categorieJuridiqueUniteLegale cj_ul,
          u.sigleUniteLegale sigle_ul,
          u.denominationUniteLegale denomination_ul,
          u.denominationUsuelle1UniteLegale denomination_usuelle_ul,
          u.nomUniteLegale nom_ul,
          u.nomUsageUniteLegale nom_usage_ul,
          u.prenomUsuelUniteLegale prenom_usuel_ul,
          u.pseudonymeUniteLegale pseudonyme_ul
        FROM read_parquet(?) e
        JOIN sites s
          ON COALESCE(TRIM(UPPER(e.codeCommuneEtablissement)), '') = COALESCE(TRIM(UPPER(s.candidate_insee)), '')
         AND COALESCE(TRIM(UPPER(e.codePostalEtablissement)), '') = COALESCE(TRIM(UPPER(s.candidate_postcode)), '')
         AND COALESCE(TRIM(UPPER(e.numeroVoieEtablissement)), '') = COALESCE(TRIM(UPPER(s.numeroVoie)), '')
         AND COALESCE(TRIM(UPPER(e.indiceRepetitionEtablissement)), '') = COALESCE(TRIM(UPPER(s.indiceRepetition)), '')
         AND COALESCE(TRIM(UPPER(e.typeVoieEtablissement)), '') = COALESCE(TRIM(UPPER(s.typeVoie)), '')
         AND COALESCE(TRIM(UPPER(e.libelleVoieEtablissement)), '') = COALESCE(TRIM(UPPER(s.libelleVoie)), '')
        LEFT JOIN read_parquet(?) u
          ON CAST(e.siren AS VARCHAR) = CAST(u.siren AS VARCHAR)""",
        [str(sirene), str(legal_units)],
    ).fetchdf()
    con.close()
    crm_columns = [
        "query_id", "crm_name", "crm_address", "crm_city", "postcode", "insee"
    ]
    candidates = candidates.merge(
        targets[crm_columns], left_on="root_query_id", right_on="query_id", validate="many_to_one"
    )
    candidates["query_id"] = candidates.apply(
        lambda row: f"{row['root_query_id']}:{row['candidate_siret']}", axis=1
    )
    candidates["ground_truth_siret"] = candidates["candidate_siret"]
    candidates["split"] = "UNASSIGNED"
    candidates["is_siege"] = False
    candidate_evidence = audit_evidence_rows(candidates)
    passthrough = candidates[
        ["query_id", "root_query_id", "candidate_siret", "crm_address", "numeroVoie",
         "indiceRepetition", "candidate_insee", "insee"]
    ].copy()
    passthrough["insee_match"] = passthrough.apply(
        lambda row: float(_clean(row["candidate_insee"]) == _clean(row["insee"])), axis=1
    )
    candidate_evidence = strict_admission(
        candidate_evidence.merge(passthrough, on="query_id", validate="one_to_one")
    )
    supported = candidate_evidence[
        candidate_evidence["admission_status"].eq("ADMIT_EXACT_SIRET")
    ].groupby("root_query_id")["candidate_siret"].apply(
        lambda values: sorted(set(str(value).zfill(14) for value in values))
    )
    result = targets[["query_id", "ground_truth_siret"]].copy()
    result["identity_supported_sirets"] = result["query_id"].map(supported).map(
        lambda value: value if isinstance(value, list) else []
    )
    result["official_site_identity_count"] = result["identity_supported_sirets"].map(len)
    result["site_identity_unique"] = result.apply(
        lambda row: row["identity_supported_sirets"] == [str(row["ground_truth_siret"]).zfill(14)],
        axis=1,
    )
    return result.drop(columns="ground_truth_siret")


def build(args: argparse.Namespace) -> Path:
    joined = _official_join(args.increment, args.sirene, args.legal_units)
    base_evidence = audit_evidence_rows(joined)
    passthrough = joined[
        ["query_id", "crm_address", "numeroVoie", "indiceRepetition", "candidate_insee", "insee"]
    ].copy()
    passthrough["insee_match"] = passthrough.apply(
        lambda row: float(_clean(row["candidate_insee"]) == _clean(row["insee"])), axis=1
    )
    evidence = base_evidence.merge(passthrough, on="query_id", validate="one_to_one")
    evidence = strict_admission(evidence)
    site_support = _site_identity_support(joined, evidence, args.sirene, args.legal_units)
    evidence = evidence.merge(site_support, on="query_id", how="left", validate="one_to_one")
    evidence["identity_supported_sirets"] = evidence["identity_supported_sirets"].map(
        lambda value: value if isinstance(value, list) else []
    )
    evidence["official_site_identity_count"] = evidence["official_site_identity_count"].fillna(0).astype(int)
    evidence["site_identity_unique"] = evidence["site_identity_unique"].fillna(False)
    ambiguous = evidence["admission_status"].eq("ADMIT_EXACT_SIRET") & ~evidence["site_identity_unique"]
    evidence.loc[ambiguous, "admission_status"] = "QUARANTINE_SITE_IDENTITY_AMBIGUOUS"
    evidence.loc[ambiguous, "admission_reason"] = "OFFICIAL_SITE_IDENTITY_NOT_UNIQUE"
    decisions = evidence[["query_id", "admission_status", "admission_reason"]]
    annotated = joined.merge(decisions, on="query_id", validate="one_to_one")
    source_columns = pd.read_csv(
        args.increment, sep=";", dtype=str, keep_default_na=False, nrows=0
    ).columns.tolist()
    admitted = annotated[annotated["admission_status"].eq("ADMIT_EXACT_SIRET")][source_columns]
    quarantined = annotated[~annotated["admission_status"].eq("ADMIT_EXACT_SIRET")]

    identity = {
        "schema_version": SCHEMA_VERSION,
        "inputs": {
            "increment": sha256(args.increment),
            "sirene": sha256(args.sirene),
            "legal_units": sha256(args.legal_units),
        },
        "policy": {
            "retrieval_inputs_used": False,
            "identity": "exact-or-jaro>=0.90+overlap>=0.60; contained requires jaro>=0.85+overlap>=0.50",
            "site": "INSEE+postcode+street_jaro>=0.95+number_and_repetition",
            "uniqueness": "only labelled SIRET has strict identity support at exact official site",
        },
        "builder_sha256": sha256(Path(__file__)),
    }
    build_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    destination = args.output_root / build_id
    if destination.exists():
        return destination
    args.output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{build_id}.", dir=args.output_root))
    admitted.to_csv(temporary / "admitted_exact_siret.csv", sep=";", index=False)
    quarantined.to_parquet(temporary / "quarantine.parquet", index=False)
    evidence.to_parquet(temporary / "evidence.parquet", index=False)
    manifest = {
        **identity,
        "build_id": build_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "input": len(joined),
            "admitted": len(admitted),
            "quarantined": len(quarantined),
            "admission_reasons": {
                str(key): int(value)
                for key, value in evidence["admission_reason"].value_counts().items()
            },
        },
    }
    outputs = ["admitted_exact_siret.csv", "quarantine.parquet", "evidence.parquet"]
    manifest["outputs"] = {name: sha256(temporary / name) for name in outputs}
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    os.replace(temporary, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--increment", type=Path, required=True)
    parser.add_argument("--sirene", type=Path, required=True)
    parser.add_argument("--legal-units", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    print(build(parser.parse_args()))


if __name__ == "__main__":
    main()
