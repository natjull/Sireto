#!/usr/bin/env python3
"""Materialise the 35 trusted OOF ranker errors whose truth is in top 100."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


BASE = Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100")
DATASET = BASE / "datasets/v4_11_input_blind/ec4326ec57e4411d"
COMPARISON = (
    BASE
    / "experiments/v4_12_trusted_label_ranker/2f57628196fefce0/"
    "trusted_oof_comparison.parquet"
)
OUTPUT = Path("reports/v412_ranker_recoverable_errors_35.csv")
FEATURES = (
    "retrieval_rank",
    "name_jaro_max",
    "name_token_overlap_max",
    "idf_name",
    "name_sim_max_etab",
    "name_sim_max_ul",
    "name_sim_max_sigle",
    "addr_jaro",
    "postcode_match",
    "city_match",
    "street_number_diff",
    "addr_token_overlap",
    "street_name_jaro",
    "name_addr_consistency",
    "is_siege",
    "is_association",
    "geo_exact_match",
    "name_norm_exact",
    "street_number_match",
)


def _lookup_establishments(sirets: set[str]) -> pd.DataFrame:
    if not sirets:
        return pd.DataFrame()
    etab_parquet = Path("data/StockEtablissement_utf8.parquet")
    ul_parquet = Path("data/StockUniteLegale_utf8.parquet")
    if etab_parquet.exists() and ul_parquet.exists():
        import duckdb

        wanted = pd.DataFrame({"siret": sorted(sirets)})
        with duckdb.connect() as connection:
            connection.register("wanted", wanted)
            return connection.execute(
                """
                SELECT
                    e.siret,
                    u.denominationUniteLegale AS denomination_unite_legale,
                    e.enseigne1Etablissement AS sirene_enseigne1,
                    e.denominationUsuelleEtablissement AS sirene_denomination_usuelle,
                    concat_ws(' ', e.numeroVoieEtablissement,
                        e.typeVoieEtablissement, e.libelleVoieEtablissement)
                        AS address_full,
                    e.codePostalEtablissement AS postcode,
                    e.libelleCommuneEtablissement AS city,
                    e.etablissementSiege AS etablissement_siege,
                    e.etatAdministratifEtablissement AS etat_administratif,
                    e.activitePrincipaleEtablissement AS activite_principale,
                    e.dateCreationEtablissement AS date_creation,
                    e.dateDebut AS date_debut
                FROM wanted w
                LEFT JOIN read_parquet(?) e USING (siret)
                LEFT JOIN read_parquet(?) u USING (siren)
                """,
                [str(etab_parquet), str(ul_parquet)],
            ).fetchdf()
    placeholders = ",".join("?" for _ in sirets)
    query = f"""
        SELECT siret, denomination, denomination_unite_legale, enseigne1,
               address_full, postcode, city, etablissement_siege,
               etat_administratif, activite_principale
        FROM establishments WHERE siret IN ({placeholders})
    """
    with sqlite3.connect("data/sirene_cache.sqlite") as connection:
        return pd.read_sql_query(query, connection, params=sorted(sirets), dtype=str)


def build() -> pd.DataFrame:
    comparison = pd.read_parquet(COMPARISON)
    comparison["query_id"] = comparison["query_id"].astype(str)
    errors = comparison[
        ~comparison["top1_correct_candidate"] & comparison["truth_in_pool"]
    ].copy()
    if len(errors) != 35:
        raise ValueError(f"Expected 35 recoverable errors, got {len(errors)}")

    labels = pd.read_csv(
        "reports/v412_review_trusted_labels_279.csv", dtype=str
    ).fillna("")
    queries = pd.read_parquet(DATASET / "queries.parquet")
    candidates = pd.read_parquet(DATASET / "candidates_sparse_top100.parquet")
    queries["query_id"] = queries["query_id"].astype(str)
    candidates["query_id"] = candidates["query_id"].astype(str)
    errors = errors.merge(
        labels[["query_id", "ground_truth_siret", "error_family", "cohort"]],
        on="query_id",
        validate="one_to_one",
    ).merge(
        queries[
            [
                "query_id",
                "crm_name",
                "crm_address",
                "crm_postcode",
                "crm_city",
            ]
        ],
        on="query_id",
        validate="one_to_one",
    )
    errors = errors.rename(columns={"predicted_siret_candidate": "predicted_siret"})
    errors["ground_truth_siren"] = errors["ground_truth_siret"].str[:9]
    errors["predicted_siren"] = errors["predicted_siret"].str[:9]
    errors["error_level"] = errors["ground_truth_siren"].eq(
        errors["predicted_siren"]
    ).map({True: "WRONG_SITE_SAME_SIREN", False: "WRONG_SIREN"})

    pool = candidates[candidates["query_id"].isin(errors["query_id"])].copy()
    truth_keys = set(zip(errors["query_id"], errors["ground_truth_siret"]))
    pred_keys = set(zip(errors["query_id"], errors["predicted_siret"]))
    pool["audit_role"] = [
        "TRUTH"
        if (query_id, siret) in truth_keys
        else "PREDICTED"
        if (query_id, siret) in pred_keys
        else "OTHER"
        for query_id, siret in zip(pool["query_id"], pool["candidate_siret"])
    ]
    pair = pool[pool["audit_role"].isin(["TRUTH", "PREDICTED"])].copy()
    if pair.groupby(["query_id", "audit_role"]).size().ne(1).any():
        raise ValueError("Every error must have exactly one truth and predicted row")

    identity = _lookup_establishments(set(pair["candidate_siret"].astype(str)))
    pair = pair.merge(
        identity,
        left_on="candidate_siret",
        right_on="siret",
        how="left",
        validate="many_to_one",
    )
    fields = [
        "candidate_siret",
        "denomination_unite_legale",
        "sirene_enseigne1",
        "sirene_denomination_usuelle",
        "address_full",
        "postcode",
        "city",
        "etablissement_siege",
        "etat_administratif",
        "activite_principale",
        "date_creation",
        "date_debut",
        *FEATURES,
    ]
    available = [field for field in fields if field in pair.columns]
    wide = pair.pivot(index="query_id", columns="audit_role", values=available)
    wide.columns = [f"{role.lower()}_{field}" for field, role in wide.columns]
    wide = wide.reset_index()
    output = errors.merge(wide, on="query_id", validate="one_to_one")
    output = output.sort_values(
        ["error_level", "error_family", "query_id"], kind="mergesort"
    ).reset_index(drop=True)
    output.insert(0, "selection_ordinal", range(1, len(output) + 1))
    return output


if __name__ == "__main__":
    frame = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT, index=False)
    print(OUTPUT)
