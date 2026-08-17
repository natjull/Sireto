#!/usr/bin/env python3
"""Triage new CRM SIRET labels against the frozen local SIRENE snapshot.

The script never changes ``crm_ok_gt.csv``.  It publishes a strict candidate
set, a review queue, a complete row-level audit and a reproducibility manifest.
INSEE is authoritative; postcode is only a fallback when CRM INSEE is absent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd


SCHEMA_VERSION = "sireto-new-crm-location-triage-1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_text(series: pd.Series) -> pd.Series:
    def normalize(value: object) -> str:
        text = "" if pd.isna(value) else str(value)
        text = unicodedata.normalize("NFKD", text)
        text = text.encode("ascii", "ignore").decode("ascii").upper()
        return " ".join(text.split())

    return series.map(normalize)


def classify_fold_relation(
    existing_crm: Path, fold_assignments: Path
) -> pd.DataFrame:
    crm = pd.read_csv(
        existing_crm, sep=";", dtype=str, keep_default_na=False
    ).reset_index(names="query_id")
    crm["query_id"] = crm["query_id"].astype(str)
    crm["target_siren"] = crm["gt_siret"].str[:9]
    folds = pd.read_parquet(
        fold_assignments,
        columns=["query_id", "oof_fold", "legacy_split", "siren_component_id"],
    )
    joined = crm[["query_id", "gt_siret", "target_siren"]].merge(
        folds, on="query_id", how="left", validate="one_to_one"
    )
    if joined["oof_fold"].isna().any():
        raise ValueError("existing crm_ok_gt rows are missing fold assignments")

    rows: list[dict[str, object]] = []
    for siren, group in joined.groupby("target_siren", sort=True):
        fold_values = sorted({int(value) for value in group["oof_fold"]})
        split_values = sorted(set(group["legacy_split"].astype(str)))
        component_values = sorted(set(group["siren_component_id"].astype(str)))
        safe = bool(
            group["oof_fold"].isin([2, 3, 4]).all()
            and group["legacy_split"].eq("train").all()
            and len(component_values) == 1
        )
        rows.append(
            {
                "target_siren": siren,
                "existing_component_relation": (
                    "EXISTING_TRAIN_COMPONENT"
                    if safe
                    else "QUARANTINE_EXISTING_NONTRAIN_COMPONENT"
                ),
                "existing_oof_folds": ",".join(map(str, fold_values)),
                "existing_legacy_splits": ",".join(split_values),
                "existing_siren_component_ids": ",".join(component_values),
                "existing_fold_mapping_conflict": bool(
                    len(fold_values) != 1
                    or len(split_values) != 1
                    or len(component_values) != 1
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--sirene", type=Path, required=True)
    parser.add_argument("--existing-crm", type=Path, required=True)
    parser.add_argument("--fold-assignments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    for path in (args.input, args.sirene, args.existing_crm, args.fold_assignments):
        if not path.is_file():
            raise FileNotFoundError(path)
    input_hash = sha256(args.input)
    if input_hash != args.expected_input_sha256:
        raise ValueError(
            f"input SHA-256 mismatch: expected {args.expected_input_sha256}, got {input_hash}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.execute("PRAGMA threads=6")
    connection.execute("PRAGMA memory_limit='8GB'")
    connection.execute(
        """
        CREATE TEMP TABLE crm AS
        SELECT row_number() OVER () - 1 AS source_row, *
        FROM read_csv(?, delim=';', header=true, all_varchar=true, encoding='utf-8')
        """,
        [str(args.input)],
    )
    expected_columns = {
        "source_row", "stc_service_id", "stc_reference", "stc_categorie_produit",
        "stc_state", "client_final", "stc_zipcode", "stc_city", "order_siret",
        "stc_street_number", "stc_street_name", "stc_insee_commune",
    }
    actual_columns = {
        row[1] for row in connection.execute("PRAGMA table_info('crm')").fetchall()
    }
    if actual_columns != expected_columns:
        raise ValueError(
            f"unexpected input columns: {sorted(actual_columns)}"
        )

    connection.execute(
        """
        CREATE TEMP TABLE requested_sirets AS
        SELECT DISTINCT trim(order_siret) AS siret
        FROM crm
        WHERE regexp_full_match(coalesce(trim(order_siret), ''), '[0-9]{14}')
        """
    )
    connection.execute(
        """
        CREATE TEMP TABLE official AS
        SELECT r.siret AS target_siret,
               p.siren AS target_siren,
               p.codeCommuneEtablissement AS sirene_insee,
               p.codePostalEtablissement AS sirene_cp,
               p.libelleCommuneEtablissement AS sirene_commune,
               p.etatAdministratifEtablissement AS sirene_etat,
               p.numeroVoieEtablissement AS sirene_street_number,
               p.typeVoieEtablissement AS sirene_street_type,
               p.libelleVoieEtablissement AS sirene_street_name
        FROM read_parquet(?) p
        JOIN requested_sirets r USING (siret)
        """,
        [str(args.sirene)],
    )
    audit = connection.execute(
        """
        WITH compared AS (
          SELECT c.*, o.*,
            coalesce(trim(c.stc_insee_commune), '') <> ''
              AND trim(c.stc_insee_commune) = trim(o.sirene_insee) AS insee_match,
            coalesce(trim(c.stc_zipcode), '') <> ''
              AND lpad(trim(c.stc_zipcode), 5, '0') =
                  lpad(trim(o.sirene_cp), 5, '0') AS cp_match
          FROM crm c
          LEFT JOIN official o ON trim(c.order_siret) = o.target_siret
        )
        SELECT *,
          CASE
            WHEN coalesce(trim(order_siret), '') = '' THEN 'NO_SIRET'
            WHEN NOT regexp_full_match(trim(order_siret), '[0-9]{14}')
              THEN 'INVALID_SIRET_FORMAT'
            WHEN target_siret IS NULL THEN 'SIRET_NOT_IN_SIRENE'
            WHEN insee_match AND cp_match THEN 'INSEE_AND_CP'
            WHEN insee_match THEN 'INSEE_ONLY'
            WHEN cp_match AND coalesce(trim(stc_insee_commune), '') = ''
              THEN 'CP_FALLBACK_INSEE_MISSING'
            WHEN cp_match THEN 'CP_ONLY_EXPLICIT_INSEE_MISMATCH'
            ELSE 'LOCATION_MISMATCH'
          END AS location_status
        FROM compared
        ORDER BY source_row
        """
    ).fetchdf()

    duplicate_columns = [
        "order_siret", "stc_insee_commune", "stc_zipcode", "client_final",
        "stc_street_number", "stc_street_name",
    ]
    conflicts = (
        audit[audit["order_siret"].fillna("").str.strip().ne("")]
        .groupby("stc_service_id")[duplicate_columns]
        .nunique(dropna=False)
        .gt(1)
        .any(axis=1)
    )
    if conflicts.any():
        raise ValueError(
            f"conflicting duplicate service identities: {int(conflicts.sum())}"
        )
    duplicate_counts = audit.groupby("stc_service_id").size().rename(
        "source_duplicate_count"
    )
    audit["_has_siret_for_dedup"] = audit["order_siret"].fillna("").str.strip().ne("")
    services = (
        audit.sort_values(
            ["stc_service_id", "_has_siret_for_dedup", "source_row"],
            ascending=[True, False, True],
        )
        .drop_duplicates("stc_service_id", keep="first")
        .merge(duplicate_counts, on="stc_service_id", how="left")
    )
    audit = audit.drop(columns=["_has_siret_for_dedup"])
    services = services.drop(columns=["_has_siret_for_dedup"])

    fold_relation = classify_fold_relation(args.existing_crm, args.fold_assignments)
    services = services.merge(fold_relation, on="target_siren", how="left")
    unseen = services["target_siren"].notna() & services[
        "existing_component_relation"
    ].isna()
    services.loc[unseen, "existing_component_relation"] = (
        "UNSEEN_SIREN_NEEDS_ASSIGNMENT"
    )
    old_sirets = set(
        pd.read_csv(
            args.existing_crm, sep=";", dtype=str, usecols=["gt_siret"]
        )["gt_siret"]
    )
    services["already_in_crm_ok_gt_exact"] = services["target_siret"].isin(
        old_sirets
    )

    accepted_statuses = {
        "INSEE_AND_CP", "INSEE_ONLY", "CP_FALLBACK_INSEE_MISSING"
    }
    accepted = services[services["location_status"].isin(accepted_statuses)].copy()
    review = services[
        services["location_status"].eq("CP_ONLY_EXPLICIT_INSEE_MISMATCH")
    ].copy()

    candidates = pd.DataFrame(
        {
            "crm_name": accepted["client_final"],
            "crm_cp": accepted["stc_zipcode"],
            "crm_insee": accepted["stc_insee_commune"],
            "crm_id": accepted["stc_service_id"],
            "crm_commune": accepted["stc_city"],
            "gt_siret": accepted["target_siret"],
            "crm_adresse": (
                accepted["stc_street_number"].fillna("").str.strip()
                + " "
                + accepted["stc_street_name"].fillna("").str.strip()
            ).str.strip(),
            "SITE_CLI_COMMUNE": accepted["stc_city"],
            "sirene_insee": accepted["sirene_insee"],
            "sirene_cp": accepted["sirene_cp"],
            "sirene_etat": accepted["sirene_etat"],
            "loc_match_type": accepted["location_status"],
            "source_reference": accepted["stc_reference"],
            "source_product_category": accepted["stc_categorie_produit"],
            "source_contract_state": accepted["stc_state"],
            "source_duplicate_count": accepted["source_duplicate_count"],
            "target_siren": accepted["target_siren"],
            "already_in_crm_ok_gt_exact": accepted["already_in_crm_ok_gt_exact"],
            "existing_component_relation": accepted[
                "existing_component_relation"
            ],
        }
    ).sort_values(["crm_id", "gt_siret"])

    surface_columns = [
        "crm_name", "crm_cp", "crm_insee", "crm_commune", "crm_adresse"
    ]
    normalized_candidate_fields = pd.DataFrame(
        {column: normalized_text(candidates[column]) for column in surface_columns}
    )
    candidates["crm_surface_fingerprint"] = normalized_candidate_fields.agg(
        "|".join, axis=1
    ).map(lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())
    candidates["crm_gt_fingerprint"] = (
        normalized_candidate_fields.assign(gt_siret=candidates["gt_siret"])
        .agg("|".join, axis=1)
        .map(lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())
    )

    old = pd.read_csv(
        args.existing_crm, sep=";", dtype=str, keep_default_na=False
    )
    normalized_old_fields = pd.DataFrame(
        {column: normalized_text(old[column]) for column in surface_columns}
    )
    old_surface_fingerprint = normalized_old_fields.agg("|".join, axis=1).map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    )
    old_gt_fingerprint = (
        normalized_old_fields.assign(gt_siret=old["gt_siret"])
        .agg("|".join, axis=1)
        .map(lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())
    )
    combined_surface_truth = pd.concat(
        [
            pd.DataFrame(
                {
                    "crm_surface_fingerprint": candidates[
                        "crm_surface_fingerprint"
                    ],
                    "gt_siret": candidates["gt_siret"],
                }
            ),
            pd.DataFrame(
                {
                    "crm_surface_fingerprint": old_surface_fingerprint,
                    "gt_siret": old["gt_siret"],
                }
            ),
        ],
        ignore_index=True,
    )
    truth_count_by_surface = combined_surface_truth.groupby(
        "crm_surface_fingerprint"
    )["gt_siret"].nunique()
    ambiguous_fingerprints = set(
        truth_count_by_surface[truth_count_by_surface.gt(1)].index
    )
    candidates["ambiguous_exact_truth_for_crm_surface"] = candidates[
        "crm_surface_fingerprint"
    ].isin(ambiguous_fingerprints)
    candidates["already_in_crm_ok_gt_exact_surface"] = candidates[
        "crm_gt_fingerprint"
    ].isin(set(old_gt_fingerprint))
    ambiguous_surfaces = candidates[
        candidates["ambiguous_exact_truth_for_crm_surface"]
    ].copy()
    unique_increment = (
        candidates[
            ~candidates["ambiguous_exact_truth_for_crm_surface"]
            & ~candidates["already_in_crm_ok_gt_exact_surface"]
        ]
        .sort_values(["crm_gt_fingerprint", "crm_id"])
        .drop_duplicates("crm_gt_fingerprint", keep="first")
        .sort_values(["crm_id", "gt_siret"])
    )
    train_ready_increment = unique_increment[
        unique_increment["existing_component_relation"].eq(
            "EXISTING_TRAIN_COMPONENT"
        )
    ].copy()

    paths = {
        "all_rows_audit.parquet": audit,
        "service_level_triage.parquet": services,
        "crm_ok_gt_candidates_strict.csv": candidates,
        "crm_ok_gt_increment_unique_unambiguous.csv": unique_increment,
        "crm_ok_gt_increment_existing_train_components.csv": train_ready_increment,
        "ambiguous_exact_truth_same_crm_surface_review.csv": ambiguous_surfaces,
        "cp_only_explicit_insee_mismatch_review.csv": review,
    }
    for name, frame in paths.items():
        path = args.output_dir / name
        if path.suffix == ".parquet":
            frame.to_parquet(path, index=False)
        else:
            frame.to_csv(path, sep=";", index=False)

    status_counts = {
        str(key): int(value)
        for key, value in audit["location_status"].value_counts().items()
    }
    service_status_counts = {
        str(key): int(value)
        for key, value in services["location_status"].value_counts().items()
    }
    relation_counts = {
        str(key): int(value)
        for key, value in accepted["existing_component_relation"]
        .fillna("NOT_APPLICABLE")
        .value_counts()
        .items()
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rules": {
            "accepted": sorted(accepted_statuses),
            "insee_authoritative": True,
            "postcode_fallback_only_when_crm_insee_missing": True,
            "explicit_insee_mismatch_never_accepted_from_postcode_alone": True,
            "no_model_score_or_rank_used": True,
            "existing_crm_not_modified": True,
            "unseen_sirens_not_assigned_to_a_fold": True,
        },
        "sources": {
            "input": {"path": str(args.input), "sha256": input_hash},
            "sirene": {"path": str(args.sirene), "sha256": sha256(args.sirene)},
            "existing_crm": {
                "path": str(args.existing_crm), "sha256": sha256(args.existing_crm)
            },
            "fold_assignments": {
                "path": str(args.fold_assignments),
                "sha256": sha256(args.fold_assignments),
            },
            "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
        },
        "counts": {
            "input_rows": int(len(audit)),
            "input_services": int(audit["stc_service_id"].nunique()),
            "input_nonempty_siret_rows": int(
                audit["order_siret"].fillna("").str.strip().ne("").sum()
            ),
            "row_status": status_counts,
            "service_status": service_status_counts,
            "accepted_services": int(len(accepted)),
            "accepted_distinct_sirets": int(accepted["target_siret"].nunique()),
            "accepted_distinct_sirens": int(accepted["target_siren"].nunique()),
            "accepted_active_services": int(accepted["sirene_etat"].eq("A").sum()),
            "accepted_closed_services": int(accepted["sirene_etat"].eq("F").sum()),
            "accepted_existing_exact_siret_services": int(
                accepted["already_in_crm_ok_gt_exact"].sum()
            ),
            "strict_duplicate_gt_surface_excess": int(
                candidates["crm_gt_fingerprint"].duplicated().sum()
            ),
            "strict_ambiguous_exact_truth_services": int(len(ambiguous_surfaces)),
            "unique_unambiguous_increment_rows": int(len(unique_increment)),
            "unique_unambiguous_increment_sirets": int(
                unique_increment["gt_siret"].nunique()
            ),
            "existing_train_component_increment_rows": int(
                len(train_ready_increment)
            ),
            "accepted_component_relation": relation_counts,
        },
        "outputs": {
            name: {"rows": int(len(frame)), "sha256": sha256(args.output_dir / name)}
            for name, frame in paths.items()
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["counts"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
