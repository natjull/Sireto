#!/usr/bin/env python3
"""OOF ablation of business/source features on the trusted REVIEW labels.

This experiment deliberately leaves retrieval unchanged.  It enriches the
already closed top-100 pools from the pinned SIRENE snapshot, adds signals that
were lost before ranker training (legal name, legal category, activity and
within-scene comparisons), then evaluates the ranker out of fold.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate_v412_hard_label_ranker import _evaluate_predictions
from scripts.run_v411_ranker_c_development import (
    RANKER_C_FEATURE_ORDER,
    RANKER_PARAMS,
    eligible_ranker_rows,
)


BASE = Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100")
DEFAULT_DATASET = BASE / "datasets/v4_11_input_blind/ec4326ec57e4411d"
DEFAULT_REFERENCE = BASE / "references/v4_12_service_parity/b4b7fef24c5e7036"
DEFAULT_OUTPUT_ROOT = BASE / "experiments/v4_12_ranker_business_features"
DEFAULT_TRUSTED = Path("reports/v412_review_trusted_labels_279.csv")
DEFAULT_ETABLISSEMENTS = Path("data/StockEtablissement_utf8.parquet")
DEFAULT_UNITES_LEGALES = Path("data/StockUniteLegale_utf8.parquet")
DEFAULT_BASELINE_RANKER_DIR = (
    BASE / "models/v4_11_ranker_c/0d2e419158c7a4c0/ranker_c"
)

LEGAL_TERMS = {
    "SA", "SAS", "SASU", "SARL", "EURL", "SCI", "SC", "SNC", "SCM",
    "SELARL", "SELAS", "SELASU", "ASSOC", "ASSOCIATION", "SOCIETE",
}

SOURCE_FEATURES = [
    "ul_name_exact",
    "ul_name_compact_exact",
    "ul_name_starts_with_crm",
    "crm_starts_with_ul_name",
    "etab_name_exact",
    "etab_name_compact_exact",
    "etab_name_starts_with_crm",
    "crm_starts_with_etab_name",
    "parenthetical_alias_exact",
    "ul_name_address_consistency",
    "etab_name_address_consistency",
    "legal_is_association",
    "legal_is_public",
    "legal_is_company",
    "activity_matches_ul_full",
    "activity_matches_ul_division",
    "activity_is_holding",
    "activity_is_property",
    "activity_is_restaurant",
    "has_operating_enseigne",
    "is_employer",
    "has_known_effectif",
    "establishment_start_year",
    "query_says_group_or_holding",
    "holding_role_consistency",
    "business_role_match",
    "business_role_conflict",
    "strict_school_role_match",
    "strict_school_role_conflict",
    "source_ul_exact",
    "source_etab_exact",
    "source_name_score",
    "source_name_exact",
    "source_name_address_consistency",
    "role_signal",
    "candidate_is_operating",
    "operating_evidence",
]

RELATIONAL_FEATURES = [
    "same_siren_count",
    "same_address_count",
    "same_address_siren_count",
    "best_etab_name_same_siren",
    "best_address_same_siren",
    "best_ul_name_same_address",
    "best_etab_name_same_address",
    "best_role_same_address",
    "best_start_date_same_address",
    "ul_gap_to_best_same_address",
    "etab_gap_to_best_same_address",
    "role_gap_to_best_same_address",
    "best_source_name_query",
    "source_name_gap_to_best_query",
    "best_address_query",
    "address_gap_to_best_query",
    "best_identity_consistency_query",
    "identity_gap_to_best_query",
    "best_operating_evidence_query",
    "operating_gap_to_best_query",
    "same_siren_source_exact_count",
    "same_siren_role_match_count",
    "same_siren_employer_count",
    "same_siren_seat_count",
    "only_candidate_same_siren",
    "unique_source_exact_same_siren",
    "unique_role_match_same_siren",
    "unique_seat_same_siren",
    "best_source_name_same_siren",
    "source_name_gap_to_best_same_siren",
    "address_gap_to_best_same_siren",
    "best_operating_evidence_same_siren",
    "operating_gap_to_best_same_siren",
    "best_source_name_same_address",
    "source_name_gap_to_best_same_address",
    "best_identity_same_address",
    "identity_gap_to_best_same_address",
]

VARIANTS = {
    "stacked_targeted": RANKER_C_FEATURE_ORDER
    + ["_baseline_score", "_baseline_rank"]
    + [
        "ul_name_exact",
        "ul_name_compact_exact",
        "ul_name_starts_with_crm",
        "crm_starts_with_ul_name",
        "etab_name_exact",
        "etab_name_compact_exact",
        "etab_name_starts_with_crm",
        "crm_starts_with_etab_name",
        "parenthetical_alias_exact",
        "ul_name_address_consistency",
        "etab_name_address_consistency",
        "legal_is_association",
        "activity_matches_ul_full",
        "activity_is_holding",
        "holding_role_consistency",
        "business_role_match",
        "business_role_conflict",
        "strict_school_role_match",
        "strict_school_role_conflict",
        "best_etab_name_same_siren",
        "best_address_same_siren",
        "best_ul_name_same_address",
        "best_etab_name_same_address",
        "best_role_same_address",
        "ul_gap_to_best_same_address",
        "etab_gap_to_best_same_address",
        "role_gap_to_best_same_address",
    ],
    "targeted": RANKER_C_FEATURE_ORDER
    + [
        "ul_name_exact",
        "ul_name_compact_exact",
        "ul_name_starts_with_crm",
        "crm_starts_with_ul_name",
        "etab_name_exact",
        "etab_name_compact_exact",
        "etab_name_starts_with_crm",
        "crm_starts_with_etab_name",
        "parenthetical_alias_exact",
        "ul_name_address_consistency",
        "etab_name_address_consistency",
        "legal_is_association",
        "activity_matches_ul_full",
        "activity_is_holding",
        "holding_role_consistency",
        "business_role_match",
        "business_role_conflict",
        "strict_school_role_match",
        "strict_school_role_conflict",
        "best_etab_name_same_siren",
        "best_address_same_siren",
        "best_ul_name_same_address",
        "best_etab_name_same_address",
        "best_role_same_address",
        "ul_gap_to_best_same_address",
        "etab_gap_to_best_same_address",
        "role_gap_to_best_same_address",
    ],
    "source": RANKER_C_FEATURE_ORDER + SOURCE_FEATURES,
    "source_relational": RANKER_C_FEATURE_ORDER + SOURCE_FEATURES + RELATIONAL_FEATURES,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.upper()
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("ascii")
        .str.replace(r"[^A-Z0-9]+", " ", regex=True)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


def _strip_legal(series: pd.Series) -> pd.Series:
    pattern = r"\b(?:" + "|".join(sorted(LEGAL_TERMS, key=len, reverse=True)) + r")\b"
    return (
        series.str.replace(pattern, " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def _clean_crm_name(name: str, city: str) -> str:
    name_tokens = str(name).split()
    city_tokens = set(str(city).split())
    cleaned = " ".join(token for token in name_tokens if token not in city_tokens)
    return cleaned or str(name)


def _read_enriched_sources(
    dataset: Path,
    etablissements: Path,
    unites_legales: Path,
    *,
    candidate_filename: str = "candidates_sparse_top100.parquet",
) -> pd.DataFrame:
    with duckdb.connect() as connection:
        frame = connection.execute(
            """
            SELECT
                candidates.*,
                queries.crm_name,
                queries.crm_name_norm,
                queries.crm_city_norm,
                etablissements.identifiantAdresseEtablissement AS address_id,
                etablissements.numeroVoieEtablissement AS raw_street_number,
                etablissements.typeVoieEtablissement AS raw_street_type,
                etablissements.libelleVoieEtablissement AS raw_street_name,
                etablissements.codePostalEtablissement AS raw_postcode,
                etablissements.enseigne1Etablissement AS source_enseigne1,
                etablissements.enseigne2Etablissement AS source_enseigne2,
                etablissements.enseigne3Etablissement AS source_enseigne3,
                etablissements.denominationUsuelleEtablissement AS source_etab_usual,
                etablissements.activitePrincipaleEtablissement AS source_etab_activity,
                etablissements.trancheEffectifsEtablissement AS effectif_band,
                etablissements.caractereEmployeurEtablissement AS employer_flag,
                etablissements.dateDebut AS establishment_start_date,
                unites.denominationUniteLegale AS source_ul_name,
                unites.denominationUsuelle1UniteLegale AS source_ul_usual1,
                unites.denominationUsuelle2UniteLegale AS source_ul_usual2,
                unites.denominationUsuelle3UniteLegale AS source_ul_usual3,
                unites.sigleUniteLegale AS source_ul_sigle,
                unites.categorieJuridiqueUniteLegale AS legal_category_code,
                unites.activitePrincipaleUniteLegale AS source_ul_activity
            FROM read_parquet(?) candidates
            INNER JOIN read_parquet(?) queries USING (query_id)
            LEFT JOIN read_parquet(?) etablissements
                ON candidates.candidate_siret = CAST(etablissements.siret AS VARCHAR)
            LEFT JOIN read_parquet(?) unites
                ON candidates.candidate_siren = CAST(unites.siren AS VARCHAR)
            ORDER BY candidates.query_id, candidates.candidate_siret
            """,
            [
                str(dataset / candidate_filename),
                str(dataset / "queries.parquet"),
                str(etablissements.resolve()),
                str(unites_legales.resolve()),
            ],
        ).fetchdf()
    frame["query_id"] = frame["query_id"].astype(str)
    return frame


def _source_features(frame: pd.DataFrame) -> pd.DataFrame:
    crm = _strip_legal(_normalise(frame["crm_name_norm"]))
    city = _normalise(frame["crm_city_norm"])
    crm_by_query = pd.DataFrame(
        {"query_id": frame["query_id"], "crm": crm, "city": city}
    ).drop_duplicates("query_id")
    crm_by_query["crm"] = [
        _clean_crm_name(name, city_name)
        for name, city_name in zip(crm_by_query["crm"], crm_by_query["city"])
    ]
    crm_map = crm_by_query.set_index("query_id")["crm"]
    frame["_crm"] = frame["query_id"].map(crm_map).fillna("")
    frame["_crm_compact"] = frame["_crm"].str.replace(" ", "", regex=False)

    frame["_ul"] = _strip_legal(_normalise(frame["source_ul_name"]))
    frame["_ul_compact"] = frame["_ul"].str.replace(" ", "", regex=False)
    etab_columns = [
        "source_enseigne1",
        "source_enseigne2",
        "source_enseigne3",
        "source_etab_usual",
    ]
    etab_norm = pd.concat(
        [_strip_legal(_normalise(frame[column])) for column in etab_columns], axis=1
    )
    etab_norm.columns = [f"_etab_{index}" for index in range(len(etab_columns))]
    frame[etab_norm.columns] = etab_norm

    nonempty = frame["_crm"].ne("")
    frame["ul_name_exact"] = (nonempty & frame["_ul"].eq(frame["_crm"])).astype("float32")
    frame["ul_name_compact_exact"] = (
        nonempty & frame["_ul_compact"].eq(frame["_crm_compact"])
    ).astype("float32")
    frame["ul_name_starts_with_crm"] = np.asarray(
        [bool(c) and bool(u) and u.startswith(c) for c, u in zip(frame["_crm"], frame["_ul"])],
        dtype=np.float32,
    )
    frame["crm_starts_with_ul_name"] = np.asarray(
        [bool(u) and bool(c) and c.startswith(u) for c, u in zip(frame["_crm"], frame["_ul"])],
        dtype=np.float32,
    )

    exact_parts = []
    compact_parts = []
    etab_starts_parts = []
    crm_starts_parts = []
    for column in etab_norm.columns:
        exact_parts.append(nonempty & frame[column].eq(frame["_crm"]))
        compact_parts.append(
            nonempty
            & frame[column].str.replace(" ", "", regex=False).eq(frame["_crm_compact"])
        )
        etab_starts_parts.append(
            np.asarray(
                [bool(c) and bool(e) and e.startswith(c) for c, e in zip(frame["_crm"], frame[column])],
                dtype=bool,
            )
        )
        crm_starts_parts.append(
            np.asarray(
                [bool(c) and bool(e) and c.startswith(e) for c, e in zip(frame["_crm"], frame[column])],
                dtype=bool,
            )
        )
    frame["etab_name_exact"] = np.maximum.reduce(exact_parts).astype("float32")
    frame["etab_name_compact_exact"] = np.maximum.reduce(compact_parts).astype("float32")
    frame["etab_name_starts_with_crm"] = np.maximum.reduce(etab_starts_parts).astype("float32")
    frame["crm_starts_with_etab_name"] = np.maximum.reduce(crm_starts_parts).astype("float32")

    aliases = (
        frame[["query_id", "crm_name"]]
        .drop_duplicates("query_id")
        .set_index("query_id")["crm_name"]
        .map(lambda value: re.findall(r"\(([^)]+)\)", str(value)))
        .map(lambda values: {_strip_single(value) for value in values if _strip_single(value)})
    )
    all_names = frame[["_ul", *etab_norm.columns]].fillna("").astype(str)
    frame["parenthetical_alias_exact"] = np.asarray(
        [
            float(bool(aliases.get(query_id, set()) & set(names)))
            for query_id, names in zip(frame["query_id"], all_names.itertuples(index=False, name=None))
        ],
        dtype=np.float32,
    )

    frame["ul_name_address_consistency"] = (
        frame["name_sim_max_ul"].astype(float) * frame["addr_jaro"].astype(float)
    ).astype("float32")
    frame["etab_name_address_consistency"] = (
        frame["name_sim_max_etab"].astype(float) * frame["addr_jaro"].astype(float)
    ).astype("float32")

    legal = frame["legal_category_code"].fillna("").astype(str)
    frame["legal_is_association"] = legal.str.startswith("92").astype("float32")
    frame["legal_is_public"] = legal.str.startswith("7").astype("float32")
    frame["legal_is_company"] = legal.str.startswith(("5", "6")).astype("float32")

    activity_fallback = (
        frame["activity_code"]
        if "activity_code" in frame.columns
        else pd.Series("", index=frame.index)
    )
    activity = frame["source_etab_activity"].fillna(activity_fallback).fillna("").astype(str)
    ul_activity = frame["source_ul_activity"].fillna("").astype(str)
    frame["_activity"] = activity
    frame["activity_matches_ul_full"] = (
        activity.ne("") & activity.eq(ul_activity)
    ).astype("float32")
    frame["activity_matches_ul_division"] = (
        activity.str[:2].ne("") & activity.str[:2].eq(ul_activity.str[:2])
    ).astype("float32")
    frame["activity_is_holding"] = activity.str.startswith("64.20").astype("float32")
    frame["activity_is_property"] = activity.str.startswith("68.20").astype("float32")
    frame["activity_is_restaurant"] = activity.str.startswith("56").astype("float32")
    frame["has_operating_enseigne"] = (
        _normalise(frame["source_enseigne1"]).ne("")
        | _normalise(frame["source_etab_usual"]).ne("")
    ).astype("float32")
    frame["is_employer"] = frame["employer_flag"].fillna("").astype(str).str.upper().eq("O").astype("float32")
    effectif = frame["effectif_band"].fillna("").astype(str)
    frame["has_known_effectif"] = (~effectif.isin(["", "NN", "None", "nan"])).astype("float32")
    years = pd.to_datetime(frame["establishment_start_date"], errors="coerce").dt.year
    frame["establishment_start_year"] = years.fillna(1900).clip(1900, 2030).astype("float32")

    query_group = frame["_crm"].str.contains(r"\b(?:GROUPE|HOLDING)\b", regex=True)
    frame["query_says_group_or_holding"] = query_group.astype("float32")
    frame["holding_role_consistency"] = np.where(
        frame["activity_is_holding"].eq(1), np.where(query_group, 1.0, -1.0), 0.0
    ).astype("float32")

    combined_name = all_names.agg(" ".join, axis=1)
    match = np.zeros(len(frame), dtype=np.float32)
    conflict = np.zeros(len(frame), dtype=np.float32)
    crm_name = frame["_crm"]

    def activity_rule(query_pattern: str, good: tuple[str, ...], bad: tuple[str, ...] = ()) -> None:
        mask = crm_name.str.contains(query_pattern, regex=True)
        good_mask = activity.str.startswith(good)
        bad_mask = activity.str.startswith(bad) if bad else pd.Series(False, index=frame.index)
        match[mask & good_mask] = 1.0
        conflict[mask & bad_mask] = 1.0

    activity_rule(r"\b(?:HOTEL|IBIS)\b", ("55",), ("64.20", "68.20", "78.30"))
    activity_rule(r"\b(?:CLINIQUE|HOPITAL|HOSPITALIER)\b", ("86.10",), ("64.20", "68.20", "86.22"))
    activity_rule(r"\b(?:COMMUNE|MAIRIE)\b", ("84.11",), ("42.99",))
    activity_rule(r"\bGOLF\b", ("93.11",), ("56",))
    activity_rule(r"\bCRECHE\b", ("88.91",), ("64.20", "68.20"))
    activity_rule(r"\b(?:ECOLE|INSTITUT|FORMATION|CFA|LYCEE|COLLEGE)\b", ("85",))
    activity_rule(r"\bAVOCAT", ("69.10",))
    frame["business_role_match"] = match
    frame["business_role_conflict"] = conflict

    school_match = np.zeros(len(frame), dtype=np.float32)
    school_conflict = np.zeros(len(frame), dtype=np.float32)
    for requested, conflicting in (("LYCEE", "COLLEGE"), ("COLLEGE", "LYCEE")):
        mask = crm_name.str.contains(rf"\b{requested}\b", regex=True)
        school_match[mask & combined_name.str.contains(rf"\b{requested}\b", regex=True)] = 1.0
        school_conflict[mask & combined_name.str.contains(rf"\b{conflicting}\b", regex=True)] = 1.0
    frame["strict_school_role_match"] = school_match
    frame["strict_school_role_conflict"] = school_conflict
    frame["source_ul_exact"] = frame[["ul_name_exact", "ul_name_compact_exact"]].max(axis=1).astype("float32")
    frame["source_etab_exact"] = frame[["etab_name_exact", "etab_name_compact_exact"]].max(axis=1).astype("float32")
    frame["source_name_score"] = frame[["name_sim_max_ul", "name_sim_max_etab"]].max(axis=1).astype("float32")
    frame["source_name_exact"] = frame[["source_ul_exact", "source_etab_exact"]].max(axis=1).astype("float32")
    frame["source_name_address_consistency"] = (
        frame["source_name_score"].astype(float) * frame["addr_jaro"].astype(float)
    ).astype("float32")
    frame["role_signal"] = (
        frame["business_role_match"].astype(float)
        - frame["business_role_conflict"].astype(float)
    ).astype("float32")
    frame["candidate_is_operating"] = (
        frame["activity_is_holding"].eq(0)
        & frame["activity_is_property"].eq(0)
    ).astype("float32")
    frame["operating_evidence"] = (
        frame["source_etab_exact"].astype(float)
        + frame["business_role_match"].astype(float)
        - frame["business_role_conflict"].astype(float)
        + 0.50 * frame["has_operating_enseigne"].astype(float)
        + 0.25 * frame["is_employer"].astype(float)
        + 0.10 * frame["has_known_effectif"].astype(float)
        + 0.10 * frame["candidate_is_operating"].astype(float)
    ).astype("float32")
    return frame


def _strip_single(value: str) -> str:
    series = pd.Series([value])
    return str(_strip_legal(_normalise(series)).iloc[0])


def _relational_features(frame: pd.DataFrame) -> pd.DataFrame:
    raw_address = (
        frame["raw_street_number"].fillna("").astype(str)
        + " " + frame["raw_street_type"].fillna("").astype(str)
        + " " + frame["raw_street_name"].fillna("").astype(str)
        + " " + frame["raw_postcode"].fillna("").astype(str)
    )
    fallback = _normalise(raw_address)
    address_id = frame["address_id"].fillna("").astype(str)
    frame["_address_key"] = np.where(address_id.ne(""), address_id, fallback)
    frame["_address_group"] = frame["query_id"] + "\x1f" + frame["_address_key"]
    frame["_siren_group"] = frame["query_id"] + "\x1f" + frame["candidate_siren"].astype(str)

    same_siren = frame.groupby("_siren_group", sort=False)
    same_address = frame.groupby("_address_group", sort=False)
    same_query = frame.groupby("query_id", sort=False)
    frame["same_siren_count"] = same_siren["candidate_siret"].transform("size").astype("float32")
    frame["same_address_count"] = same_address["candidate_siret"].transform("size").astype("float32")
    frame["same_address_siren_count"] = same_address["candidate_siren"].transform("nunique").astype("float32")
    frame["best_etab_name_same_siren"] = (
        frame["name_sim_max_etab"].eq(same_siren["name_sim_max_etab"].transform("max"))
    ).astype("float32")
    frame["best_address_same_siren"] = (
        frame["addr_jaro"].eq(same_siren["addr_jaro"].transform("max"))
    ).astype("float32")
    best_ul_address = same_address["name_sim_max_ul"].transform("max")
    best_etab_address = same_address["name_sim_max_etab"].transform("max")
    role_signal = frame["business_role_match"] - frame["business_role_conflict"]
    frame["_role_signal"] = role_signal
    best_role_address = frame.groupby("_address_group", sort=False)["_role_signal"].transform("max")
    best_date_address = same_address["establishment_start_year"].transform("max")
    frame["best_ul_name_same_address"] = frame["name_sim_max_ul"].eq(best_ul_address).astype("float32")
    frame["best_etab_name_same_address"] = frame["name_sim_max_etab"].eq(best_etab_address).astype("float32")
    frame["best_role_same_address"] = role_signal.eq(best_role_address).astype("float32")
    frame["best_start_date_same_address"] = frame["establishment_start_year"].eq(best_date_address).astype("float32")
    frame["ul_gap_to_best_same_address"] = (frame["name_sim_max_ul"] - best_ul_address).astype("float32")
    frame["etab_gap_to_best_same_address"] = (frame["name_sim_max_etab"] - best_etab_address).astype("float32")
    frame["role_gap_to_best_same_address"] = (role_signal - best_role_address).astype("float32")

    identity = frame["source_name_address_consistency"].astype(float)
    operating = frame["operating_evidence"].astype(float)
    best_source_query = same_query["source_name_score"].transform("max")
    best_address_query = same_query["addr_jaro"].transform("max")
    best_identity_query = identity.groupby(frame["query_id"], sort=False).transform("max")
    best_operating_query = operating.groupby(frame["query_id"], sort=False).transform("max")
    frame["best_source_name_query"] = frame["source_name_score"].eq(best_source_query).astype("float32")
    frame["source_name_gap_to_best_query"] = (frame["source_name_score"] - best_source_query).astype("float32")
    frame["best_address_query"] = frame["addr_jaro"].eq(best_address_query).astype("float32")
    frame["address_gap_to_best_query"] = (frame["addr_jaro"] - best_address_query).astype("float32")
    frame["best_identity_consistency_query"] = identity.eq(best_identity_query).astype("float32")
    frame["identity_gap_to_best_query"] = (identity - best_identity_query).astype("float32")
    frame["best_operating_evidence_query"] = operating.eq(best_operating_query).astype("float32")
    frame["operating_gap_to_best_query"] = (operating - best_operating_query).astype("float32")

    frame["same_siren_source_exact_count"] = same_siren["source_name_exact"].transform("sum").astype("float32")
    frame["same_siren_role_match_count"] = same_siren["business_role_match"].transform("sum").astype("float32")
    frame["same_siren_employer_count"] = same_siren["is_employer"].transform("sum").astype("float32")
    frame["same_siren_seat_count"] = same_siren["is_siege"].transform("sum").astype("float32")
    frame["only_candidate_same_siren"] = frame["same_siren_count"].eq(1).astype("float32")
    frame["unique_source_exact_same_siren"] = (
        frame["source_name_exact"].eq(1) & frame["same_siren_source_exact_count"].eq(1)
    ).astype("float32")
    frame["unique_role_match_same_siren"] = (
        frame["business_role_match"].eq(1) & frame["same_siren_role_match_count"].eq(1)
    ).astype("float32")
    frame["unique_seat_same_siren"] = (
        frame["is_siege"].eq(1) & frame["same_siren_seat_count"].eq(1)
    ).astype("float32")

    best_source_siren = same_siren["source_name_score"].transform("max")
    best_address_siren = same_siren["addr_jaro"].transform("max")
    best_operating_siren = operating.groupby(frame["_siren_group"], sort=False).transform("max")
    frame["best_source_name_same_siren"] = frame["source_name_score"].eq(best_source_siren).astype("float32")
    frame["source_name_gap_to_best_same_siren"] = (frame["source_name_score"] - best_source_siren).astype("float32")
    frame["address_gap_to_best_same_siren"] = (frame["addr_jaro"] - best_address_siren).astype("float32")
    frame["best_operating_evidence_same_siren"] = operating.eq(best_operating_siren).astype("float32")
    frame["operating_gap_to_best_same_siren"] = (operating - best_operating_siren).astype("float32")

    best_source_address = same_address["source_name_score"].transform("max")
    best_identity_address = identity.groupby(frame["_address_group"], sort=False).transform("max")
    frame["best_source_name_same_address"] = frame["source_name_score"].eq(best_source_address).astype("float32")
    frame["source_name_gap_to_best_same_address"] = (frame["source_name_score"] - best_source_address).astype("float32")
    frame["best_identity_same_address"] = identity.eq(best_identity_address).astype("float32")
    frame["identity_gap_to_best_same_address"] = (identity - best_identity_address).astype("float32")
    return frame


def _baseline_scores(
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    model_dir: Path,
) -> pd.Series:
    fold_by_query = assignments.set_index("query_id")["oof_fold"].astype(int)
    scores = pd.Series(np.nan, index=frame.index, dtype="float32")
    for fold in range(5):
        mask = frame["query_id"].map(fold_by_query).eq(fold)
        model = xgb.XGBRanker()
        model.load_model(model_dir / f"oof_fold_{fold}.json")
        scores.loc[mask] = model.predict(
            frame.loc[mask, RANKER_C_FEATURE_ORDER].to_numpy(dtype=np.float32)
        ).astype("float32")
    if scores.isna().any():
        raise ValueError("Baseline hard-negative scores are incomplete")
    return scores


def _fit_ranker(
    rows: pd.DataFrame,
    features: list[str],
    hard_ids: set[str],
    weight: float,
    negative_limit: int,
) -> xgb.XGBRanker:
    if negative_limit:
        ranked = rows.sort_values(
            ["query_id", "_baseline_score", "retrieval_rank", "candidate_siret"],
            ascending=[True, False, True, True],
            kind="mergesort",
        ).copy()
        ranked["_negative_rank"] = ranked.groupby("query_id", sort=False).cumcount() + 1
        rows = ranked[
            ranked["_negative_rank"].le(negative_limit)
            | ranked["is_ground_truth"].eq(1)
        ]
    ordered = rows.sort_values(["query_id", "candidate_siret"], kind="mergesort")
    grouped = ordered.groupby("query_id", sort=False)
    query_order = list(grouped.indices)
    model = xgb.XGBRanker(**RANKER_PARAMS)
    model.fit(
        ordered[features].to_numpy(dtype=np.float32),
        ordered["is_ground_truth"].to_numpy(dtype=np.int8),
        group=grouped.size().to_numpy(),
        sample_weight=np.asarray(
            [weight if query_id in hard_ids else 1.0 for query_id in query_order],
            dtype=np.float32,
        ),
        verbose=False,
    )
    return model


def _rank(model: xgb.XGBRanker, rows: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    output = rows[["query_id", "candidate_siret", "candidate_siren", "retrieval_rank"]].copy()
    output["ranker_score"] = model.predict(rows[features].to_numpy(dtype=np.float32)).astype("float32")
    output = output.sort_values(
        ["query_id", "ranker_score", "retrieval_rank", "candidate_siret"],
        ascending=[True, False, True, True],
        kind="mergesort",
    )
    output["ranker_rank"] = output.groupby("query_id", sort=False).cumcount() + 1
    return output.reset_index(drop=True)


def _compare(reference: pd.DataFrame, candidate: pd.DataFrame, truth: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    reference_metrics, reference_detail = _evaluate_predictions(reference, truth)
    candidate_metrics, candidate_detail = _evaluate_predictions(candidate, truth)
    detail = reference_detail[["query_id", "predicted_siret", "top1_correct"]].merge(
        candidate_detail[["query_id", "predicted_siret", "top1_correct", "truth_in_pool"]],
        on="query_id",
        suffixes=("_reference", "_candidate"),
        validate="one_to_one",
    )
    fixed = int((~detail["top1_correct_reference"] & detail["top1_correct_candidate"]).sum())
    regressed = int((detail["top1_correct_reference"] & ~detail["top1_correct_candidate"]).sum())
    return {
        "reference": reference_metrics,
        "candidate": candidate_metrics,
        "fixed_count": fixed,
        "regressed_count": regressed,
    }, detail


def run(args: argparse.Namespace) -> Path:
    dataset = args.dataset.resolve()
    frame = _read_enriched_sources(dataset, args.etablissements, args.unites_legales)
    frame = _relational_features(_source_features(frame))

    labels = pd.read_parquet(dataset / "labels.parquet")
    assignments = pd.read_parquet(dataset / "split_assignments.parquet")
    reference = pd.read_parquet(args.reference.resolve() / "ranker_reference.parquet")
    trusted = pd.read_csv(args.trusted_labels, dtype=str).fillna("")
    for value in (labels, assignments, reference, trusted):
        value["query_id"] = value["query_id"].astype(str)
    population = labels.merge(assignments, on="query_id", validate="one_to_one")
    frame["_baseline_score"] = _baseline_scores(
        frame, assignments, args.baseline_ranker_dir.resolve()
    )
    frame["_baseline_rank"] = (
        frame.groupby("query_id", sort=False)["_baseline_score"]
        .rank(method="first", ascending=False)
        .astype("float32")
    )
    required = sorted(set(sum(VARIANTS.values(), [])))
    if frame[required].isna().any().any():
        missing = frame[required].columns[frame[required].isna().any()].tolist()
        raise ValueError(f"Non-finite business features: {missing}")
    fit_population = population[population["split"].eq("fit")].copy()
    base_rows = eligible_ranker_rows(
        frame[frame["query_id"].isin(fit_population["query_id"])], fit_population
    )

    trusted_truth = trusted[trusted["label_kind"].eq("MATCH_EXACT")][
        ["query_id", "ground_truth_siret"]
    ].merge(
        assignments[["query_id", "oof_fold", "siren_component_id", "split"]],
        on="query_id",
        validate="one_to_one",
    )
    trusted_truth["ground_truth_siren"] = trusted_truth["ground_truth_siret"].str[:9]
    trusted_truth["oof_fold"] = trusted_truth["oof_fold"].astype(int)
    trusted_rows = frame[frame["query_id"].isin(trusted_truth["query_id"])].drop(columns=["is_ground_truth"]).merge(
        trusted_truth[["query_id", "ground_truth_siret", "oof_fold"]],
        on="query_id",
        validate="many_to_one",
    )
    trusted_rows["is_ground_truth"] = trusted_rows["candidate_siret"].eq(
        trusted_rows["ground_truth_siret"]
    ).astype("int8")
    present = trusted_rows.groupby("query_id")["is_ground_truth"].sum()
    eligible_trusted = set(present[present.eq(1)].index.astype(str))

    trusted_ids = set(trusted["query_id"])
    trusted_components = set(trusted_truth["siren_component_id"].astype(str))
    regression_truth = population[
        population["split"].eq("dev")
        & population["label_kind"].eq("MATCH_EXACT")
        & ~population["query_id"].isin(trusted_ids)
        & ~population["siren_component_id"].astype(str).isin(trusted_components)
    ][["query_id", "ground_truth_siret", "ground_truth_siren"]]

    variant_names = args.variants or list(VARIANTS)
    output_payload: dict[str, Any] = {
        "schema_version": "sireto-v4.12-ranker-business-features-development-1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "CONSUMED_DEVELOPMENT_ONLY",
        "final_test_opened": False,
        "hard_weight": args.hard_weight,
        "negative_limit": args.negative_limit,
        "trusted_exact_count": len(trusted_truth),
        "trusted_truth_present_count": len(eligible_trusted),
        "variants": {},
    }
    details: dict[str, pd.DataFrame] = {}
    ranked_outputs: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    models: dict[str, xgb.XGBRanker] = {}
    for name in variant_names:
        features = VARIANTS[name]
        trusted_parts: list[pd.DataFrame] = []
        base_parts: list[pd.DataFrame] = []
        for fold in range(5):
            held_base_ids = set(
                fit_population.loc[
                    fit_population["oof_fold"].astype(int).eq(fold), "query_id"
                ].astype(str)
            )
            base_train = base_rows[~base_rows["query_id"].isin(held_base_ids)]
            hard_train = trusted_rows[
                trusted_rows["query_id"].isin(eligible_trusted)
                & trusted_rows["oof_fold"].ne(fold)
            ]
            model = _fit_ranker(
                pd.concat([base_train, hard_train], ignore_index=True),
                features,
                set(hard_train["query_id"].astype(str)),
                args.hard_weight,
                args.negative_limit,
            )
            trusted_parts.append(_rank(model, trusted_rows[trusted_rows["oof_fold"].eq(fold)], features))
            base_parts.append(_rank(model, frame[frame["query_id"].isin(held_base_ids)], features))
        trusted_oof = pd.concat(trusted_parts, ignore_index=True)
        base_oof = pd.concat(base_parts, ignore_index=True)
        trusted_metrics, trusted_detail = _compare(
            reference[reference["query_id"].isin(trusted_truth["query_id"])],
            trusted_oof,
            trusted_truth,
        )
        base_truth = fit_population[fit_population["label_kind"].eq("MATCH_EXACT")][
            ["query_id", "ground_truth_siret", "ground_truth_siren"]
        ]
        base_metrics, _ = _evaluate_predictions(base_oof, base_truth)

        full_hard = trusted_rows[trusted_rows["query_id"].isin(eligible_trusted)]
        full_model = _fit_ranker(
            pd.concat([base_rows, full_hard], ignore_index=True),
            features,
            set(full_hard["query_id"].astype(str)),
            args.hard_weight,
            args.negative_limit,
        )
        regression_predictions = _rank(
            full_model,
            frame[frame["query_id"].isin(regression_truth["query_id"])],
            features,
        )
        regression_metrics, regression_detail = _compare(
            reference[reference["query_id"].isin(regression_truth["query_id"])],
            regression_predictions,
            regression_truth,
        )
        output_payload["variants"][name] = {
            "feature_count": len(features),
            "features": features,
            "trusted_oof": trusted_metrics,
            "base_fit_oof": base_metrics,
            "non_trusted_dev": regression_metrics,
        }
        details[name] = trusted_detail
        ranked_outputs[name] = (
            trusted_oof,
            base_oof,
            regression_predictions,
        )
        models[name] = full_model

    identity = hashlib.sha256(
        json.dumps(
            {
                "schema": output_payload["schema_version"],
                "dataset": _sha256(dataset / "manifest.json"),
                "trusted": _sha256(args.trusted_labels),
                "weight": args.hard_weight,
                "negative_limit": args.negative_limit,
                "variants": variant_names,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    output = args.output_root.resolve() / identity
    output.mkdir(parents=True, exist_ok=False)
    (output / "evaluation.json").write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    for name in variant_names:
        details[name].to_parquet(output / f"{name}_trusted_oof_comparison.parquet", index=False)
        trusted_ranked, base_ranked, regression_ranked = ranked_outputs[name]
        trusted_ranked.to_parquet(
            output / f"{name}_trusted_oof_ranked_candidates.parquet", index=False
        )
        base_ranked.to_parquet(
            output / f"{name}_base_fit_oof_ranked_candidates.parquet", index=False
        )
        regression_ranked.to_parquet(
            output / f"{name}_non_trusted_dev_ranked_candidates.parquet", index=False
        )
        models[name].save_model(output / f"{name}_ranker.json")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--trusted-labels", type=Path, default=DEFAULT_TRUSTED)
    parser.add_argument("--etablissements", type=Path, default=DEFAULT_ETABLISSEMENTS)
    parser.add_argument("--unites-legales", type=Path, default=DEFAULT_UNITES_LEGALES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--hard-weight", type=float, default=0.5)
    parser.add_argument("--negative-limit", type=int, default=0)
    parser.add_argument(
        "--baseline-ranker-dir", type=Path, default=DEFAULT_BASELINE_RANKER_DIR
    )
    parser.add_argument("--variants", nargs="*", choices=sorted(VARIANTS))
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
