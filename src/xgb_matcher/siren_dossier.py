"""Parquet/DuckDB dossier store for official SIREN and SIRET evidence.

The dossier is the shared, model-agnostic projection of SIRENE, RNE and
BODACC.  Parquet remains the data plane; the DuckDB file is a lightweight
catalog of external views.  No model score, CRM label or inferred identity is
stored in the dossier.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import duckdb

from .official_source_sync import canonical_json, sha256_file


SIREN_DOSSIER_SCHEMA_VERSION_V1 = "sireto-siren-dossier-v1"
SIREN_DOSSIER_SCHEMA_VERSION_V2 = "sireto-siren-dossier-v2"
SIREN_DOSSIER_SCHEMA_VERSION = "sireto-siren-dossier-v3"
SUPPORTED_SIREN_DOSSIER_SCHEMA_VERSIONS = {
    SIREN_DOSSIER_SCHEMA_VERSION_V1,
    SIREN_DOSSIER_SCHEMA_VERSION_V2,
    SIREN_DOSSIER_SCHEMA_VERSION,
}
SIREN_DOSSIER_FEATURE_SCHEMA_VERSION = "sireto-siren-dossier-features-v1"


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _normalized(expression: str) -> str:
    return (
        "trim(regexp_replace(upper(strip_accents(coalesce("
        f"{expression}, ''))), '[^A-Z0-9]+', ' ', 'g'))"
    )


def _input_identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "name": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


@dataclass(frozen=True)
class SirenDossierInputs:
    sirene_establishments: Path
    sirene_legal_units: Path
    official_evidence: tuple[Path, ...]
    official_relations: tuple[Path, ...]
    rne_account_deposits: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "sirene_establishments", Path(self.sirene_establishments))
        object.__setattr__(self, "sirene_legal_units", Path(self.sirene_legal_units))
        object.__setattr__(self, "official_evidence", tuple(Path(p) for p in self.official_evidence))
        object.__setattr__(self, "official_relations", tuple(Path(p) for p in self.official_relations))
        object.__setattr__(self, "rne_account_deposits", tuple(Path(p) for p in self.rne_account_deposits))
        for path in (
            self.sirene_establishments,
            self.sirene_legal_units,
            *self.official_evidence,
            *self.official_relations,
            *self.rne_account_deposits,
        ):
            if not path.is_file():
                raise FileNotFoundError(path)


@dataclass(frozen=True)
class SirenDossierBuild:
    output_dir: Path
    manifest_path: Path
    catalog_path: Path
    build_id: str
    counts: Mapping[str, int]


_DOSSIER_TABLE_FILES = (
    ("legal_units", "legal_units.parquet"),
    ("establishments", "establishments.parquet"),
    ("name_evidence", "name_evidence.parquet"),
    ("address_evidence", "address_evidence.parquet"),
    ("entity_evidence", "entity_evidence.parquet"),
    ("address_site_resolution", "address_site_resolution.parquet"),
    ("official_relations", "relations.parquet"),
    ("rne_account_deposits", "rne_account_deposits.parquet"),
    ("siren_summary", "siren_summary.parquet"),
)


def open_siren_dossier(
    dossier_dir: Path, *, read_only: bool = True
) -> duckdb.DuckDBPyConnection:
    """Open the catalog and bind temporary views to its sibling Parquets."""
    dossier_dir = Path(dossier_dir).resolve()
    connection = duckdb.connect(str(dossier_dir / "dossier.duckdb"), read_only=read_only)
    rows = connection.execute(
        "SELECT view_name, relative_path FROM dossier_files ORDER BY view_name"
    ).fetchall()
    for view_name, relative_path in rows:
        path = dossier_dir / str(relative_path)
        if not path.is_file():
            connection.close()
            raise FileNotFoundError(path)
        connection.execute(
            f"CREATE OR REPLACE TEMP VIEW {view_name} AS "
            f"SELECT * FROM read_parquet('{_sql_path(path)}')"
        )
    return connection


def _parquet_list(paths: Sequence[Path]) -> str:
    return "[" + ",".join(f"'{_sql_path(path)}'" for path in paths) + "]"


def _copy(connection: duckdb.DuckDBPyConnection, query: str, output: Path) -> int:
    escaped = _sql_path(output)
    connection.execute(
        f"COPY ({query}) TO '{escaped}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 65536)"
    )
    return int(connection.execute(f"SELECT count(*) FROM read_parquet('{escaped}')").fetchone()[0])


def build_siren_dossier(
    inputs: SirenDossierInputs,
    *,
    output_root: Path,
    temp_directory: Path | None = None,
    threads: int = 4,
    memory_limit: str = "12GB",
) -> SirenDossierBuild:
    """Build a content-addressed dossier without loading national data in RAM."""
    identities = {
        "sirene_establishments": _input_identity(inputs.sirene_establishments),
        "sirene_legal_units": _input_identity(inputs.sirene_legal_units),
        "official_evidence": [_input_identity(p) for p in inputs.official_evidence],
        "official_relations": [_input_identity(p) for p in inputs.official_relations],
        "rne_account_deposits": [_input_identity(p) for p in inputs.rne_account_deposits],
    }
    identity = {
        "schema_version": SIREN_DOSSIER_SCHEMA_VERSION,
        "inputs": identities,
        "normalization": "DUCKDB_STRIP_ACCENTS_UPPER_NON_ALNUM_TO_SPACE_V1",
        "policy": {
            "source_evidence_additive": True,
            "bodacc_siren_address_auto_identity": False,
            "site_resolution": "UNIQUE_EXACT_NORMALIZED_ADDRESS_AND_GEO_ONLY",
            "crm_labels_present": False,
            "model_scores_present": False,
            "rne_accounts_level": "SIREN_ONLY",
            "rne_account_model_use_enabled": False,
        },
    }
    build_id = hashlib.sha256(canonical_json(identity)).hexdigest()
    output_root = Path(output_root)
    final = output_root / build_id[:16]
    if final.exists():
        manifest = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("build_id") != build_id:
            raise ValueError("SIREN dossier content-address collision")
        return SirenDossierBuild(
            final,
            final / "manifest.json",
            final / "dossier.duckdb",
            build_id,
            manifest["counts"],
        )
    output_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".siren-dossier-", dir=output_root))
    spill = Path(temp_directory) if temp_directory else stage / "duckdb_tmp"
    spill.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    try:
        connection = duckdb.connect(str(stage / "build.duckdb"))
        connection.execute(f"SET threads={max(1, int(threads))}")
        connection.execute(f"SET memory_limit='{memory_limit}'")
        connection.execute("SET preserve_insertion_order=false")
        connection.execute(f"SET temp_directory='{_sql_path(spill)}'")
        establishments = f"read_parquet('{_sql_path(inputs.sirene_establishments)}')"
        legal_units = f"read_parquet('{_sql_path(inputs.sirene_legal_units)}')"
        evidence = f"read_parquet({_parquet_list(inputs.official_evidence)}, union_by_name=true)"
        relations = f"read_parquet({_parquet_list(inputs.official_relations)}, union_by_name=true)"
        accounts = (
            f"read_parquet({_parquet_list(inputs.rne_account_deposits)}, union_by_name=true)"
            if inputs.rne_account_deposits
            else None
        )

        counts["legal_units"] = _copy(
            connection,
            f"""
            SELECT siren::VARCHAR AS siren,
                   coalesce(etatAdministratifUniteLegale, '')::VARCHAR AS administrative_state,
                   dateCreationUniteLegale::DATE AS creation_date,
                   dateDebut::DATE AS current_period_start,
                   coalesce(categorieJuridiqueUniteLegale::VARCHAR, '') AS legal_category,
                   coalesce(activitePrincipaleUniteLegale, '') AS main_activity,
                   coalesce(categorieEntreprise, '') AS enterprise_category,
                   coalesce(nicSiegeUniteLegale, '') AS headquarters_nic,
                   { _normalized('denominationUniteLegale') } AS legal_name_normalized,
                   { _normalized('coalesce(nomUsageUniteLegale, nomUniteLegale)') } AS person_name_normalized
            FROM {legal_units}
            WHERE regexp_full_match(siren::VARCHAR, '[0-9]{{9}}')
            """,
            stage / "legal_units.parquet",
        )
        current_address = "concat_ws(' ', numeroVoieEtablissement, indiceRepetitionEtablissement, typeVoieEtablissement, libelleVoieEtablissement, complementAdresseEtablissement)"
        counts["establishments"] = _copy(
            connection,
            f"""
            SELECT siren::VARCHAR AS siren, siret::VARCHAR AS siret,
                   coalesce(nic, '')::VARCHAR AS nic,
                   coalesce(etatAdministratifEtablissement, '')::VARCHAR AS administrative_state,
                   coalesce(etablissementSiege, false)::BOOLEAN AS is_headquarters,
                   dateCreationEtablissement::DATE AS creation_date,
                   dateDebut::DATE AS current_period_start,
                   coalesce(codeCommuneEtablissement, '')::VARCHAR AS insee,
                   coalesce(codePostalEtablissement, '')::VARCHAR AS postcode,
                   coalesce(numeroVoieEtablissement, '')::VARCHAR AS street_number,
                   coalesce(indiceRepetitionEtablissement, '')::VARCHAR AS street_number_suffix,
                   coalesce(typeVoieEtablissement, '')::VARCHAR AS street_type,
                   coalesce(libelleVoieEtablissement, '')::VARCHAR AS street_name,
                   coalesce(complementAdresseEtablissement, '')::VARCHAR AS address_complement,
                   { _normalized(current_address) } AS current_address_normalized,
                   coalesce(activitePrincipaleEtablissement, '')::VARCHAR AS main_activity
            FROM {establishments}
            WHERE regexp_full_match(siren::VARCHAR, '[0-9]{{9}}')
              AND regexp_full_match(siret::VARCHAR, '[0-9]{{14}}')
              AND starts_with(siret::VARCHAR, siren::VARCHAR)
            """,
            stage / "establishments.parquet",
        )

        sirene_legal_names = " UNION ALL ".join(
            f"SELECT siren::VARCHAR siren, ''::VARCHAR siret, 'SIREN' subject_kind, "
            f"'SIRENE_CURRENT' AS \"source\", '{kind}' AS name_kind, {column}::VARCHAR raw_value, "
            f"{_normalized(column)} normalized_value, NULL::DATE valid_from, NULL::DATE valid_to, true is_current "
            f",''::VARCHAR evidence_id, ''::VARCHAR source_record_id, NULL::DATE observed_at, 400::SMALLINT source_priority "
            f"FROM {legal_units} WHERE coalesce({column}::VARCHAR, '') <> ''"
            for column, kind in (
                ("denominationUniteLegale", "LEGAL"),
                ("denominationUsuelle1UniteLegale", "USUAL"),
                ("denominationUsuelle2UniteLegale", "USUAL"),
                ("denominationUsuelle3UniteLegale", "USUAL"),
                ("sigleUniteLegale", "TRADE"),
                ("nomUsageUniteLegale", "USUAL"),
                ("nomUniteLegale", "LEGAL"),
            )
        )
        sirene_site_names = " UNION ALL ".join(
            f"SELECT siren::VARCHAR siren, siret::VARCHAR siret, 'SIRET' subject_kind, "
            f"'SIRENE_CURRENT' AS \"source\", '{kind}' AS name_kind, {column}::VARCHAR raw_value, "
            f"{_normalized(column)} normalized_value, NULL::DATE valid_from, NULL::DATE valid_to, true is_current "
            f",''::VARCHAR evidence_id, ''::VARCHAR source_record_id, NULL::DATE observed_at, 400::SMALLINT source_priority "
            f"FROM {establishments} WHERE coalesce({column}::VARCHAR, '') <> ''"
            for column, kind in (
                ("enseigne1Etablissement", "SIGN"),
                ("enseigne2Etablissement", "SIGN"),
                ("enseigne3Etablissement", "SIGN"),
                ("denominationUsuelleEtablissement", "USUAL"),
            )
        )
        counts["name_evidence"] = _copy(
            connection,
            f"""
            SELECT DISTINCT * FROM (
              {sirene_legal_names}
              UNION ALL
              {sirene_site_names}
              UNION ALL
              SELECT e.siren::VARCHAR, coalesce(e.siret, '')::VARCHAR,
                     e.subject_kind::VARCHAR, e.source::VARCHAR,
                     n.kind::VARCHAR, n.raw_value::VARCHAR,
                     n.normalized_value::VARCHAR,
                     try_cast(e.valid_from AS DATE), try_cast(e.valid_to AS DATE),
                     e.is_current::BOOLEAN, e.evidence_id::VARCHAR,
                     e.source_record_id::VARCHAR, try_cast(e.observed_at AS DATE),
                     e.source_priority::SMALLINT
              FROM {evidence} e, unnest(e.names) AS item(n)
              WHERE coalesce(n.normalized_value, '') <> ''
            )
            """,
            stage / "name_evidence.parquet",
        )
        counts["address_evidence"] = _copy(
            connection,
            f"""
            SELECT DISTINCT * FROM (
              SELECT siren::VARCHAR siren, siret::VARCHAR siret, 'SIRET' subject_kind,
                     'SIRENE_CURRENT' AS \"source\",
                     {current_address}::VARCHAR raw_value,
                     {_normalized(current_address)} normalized_value,
                     coalesce(codePostalEtablissement, '')::VARCHAR postcode,
                     coalesce(codeCommuneEtablissement, '')::VARCHAR insee,
                     coalesce(numeroVoieEtablissement, '')::VARCHAR street_number,
                     coalesce(indiceRepetitionEtablissement, '')::VARCHAR street_number_suffix,
                     NULL::DATE valid_from, NULL::DATE valid_to, true is_current,
                     ''::VARCHAR evidence_id, ''::VARCHAR source_record_id,
                     NULL::DATE observed_at, 400::SMALLINT source_priority
              FROM {establishments}
              WHERE {_normalized(current_address)} <> ''
              UNION ALL
              SELECT e.siren::VARCHAR, coalesce(e.siret, '')::VARCHAR,
                     e.subject_kind::VARCHAR, e.source::VARCHAR,
                     a.raw_value::VARCHAR, a.normalized_value::VARCHAR,
                     a.postcode::VARCHAR, a.insee::VARCHAR,
                     a.number::VARCHAR, a.number_suffix::VARCHAR,
                     try_cast(e.valid_from AS DATE), try_cast(e.valid_to AS DATE),
                     e.is_current::BOOLEAN, e.evidence_id::VARCHAR,
                     e.source_record_id::VARCHAR, try_cast(e.observed_at AS DATE),
                     e.source_priority::SMALLINT
              FROM {evidence} e, unnest(e.addresses) AS item(a)
              WHERE coalesce(a.normalized_value, '') <> ''
                 OR coalesce(a.postcode, '') <> '' OR coalesce(a.insee, '') <> ''
            )
            """,
            stage / "address_evidence.parquet",
        )
        counts["entity_evidence"] = _copy(
            connection,
            f"""
            SELECT DISTINCT evidence_id::VARCHAR evidence_id,
                   source_record_id::VARCHAR source_record_id,
                   source::VARCHAR AS "source",siren::VARCHAR siren,
                   coalesce(siret,'')::VARCHAR siret,subject_kind::VARCHAR subject_kind,
                   coalesce(administrative_state,'')::VARCHAR administrative_state,
                   is_headquarters::BOOLEAN is_headquarters,
                   try_cast(valid_from AS DATE) valid_from,try_cast(valid_to AS DATE) valid_to,
                   try_cast(observed_at AS DATE) observed_at,is_current::BOOLEAN is_current,
                   source_priority::SMALLINT source_priority
            FROM {evidence}
            WHERE regexp_full_match(siren::VARCHAR, '[0-9]{{9}}')
            """,
            stage / "entity_evidence.parquet",
        )
        counts["relations"] = _copy(
            connection,
            f"""
            SELECT DISTINCT relation_id::VARCHAR relation_id, source::VARCHAR AS \"source\",
                   source_record_id::VARCHAR source_record_id,
                   relation_type::VARCHAR relation_type,
                   from_kind::VARCHAR from_kind, from_identifier::VARCHAR from_identifier,
                   to_kind::VARCHAR to_kind, to_identifier::VARCHAR to_identifier,
                   try_cast(effective_date AS DATE) effective_date,
                   try_cast(observed_at AS DATE) observed_at,
                   source_priority::SMALLINT source_priority
            FROM {relations}
            """,
            stage / "relations.parquet",
        )
        account_query = (
            f"""
            SELECT DISTINCT source_record_uid::VARCHAR source_record_uid,
                   snapshot_id::VARCHAR snapshot_id,
                   archive_member::VARCHAR archive_member,
                   source_record_ordinal::BIGINT source_record_ordinal,
                   filing_id::VARCHAR filing_id, siren::VARCHAR siren,
                   denomination::VARCHAR denomination,
                   try_cast(filing_date AS DATE) filing_date,
                   try_cast(closing_date AS DATE) closing_date,
                   try_cast(previous_closing_date AS DATE) previous_closing_date,
                   updated_at::VARCHAR updated_at,
                   chronology_number::VARCHAR chronology_number,
                   confidentiality::VARCHAR confidentiality,
                   is_public::BOOLEAN is_public, is_deleted::BOOLEAN is_deleted,
                   account_type::VARCHAR account_type, currency::VARCHAR currency,
                   try_cast(duration_months AS SMALLINT) duration_months,
                   activity_code::VARCHAR activity_code,
                   structured_accounts_present::BOOLEAN structured_accounts_present
            FROM {accounts}
            WHERE regexp_full_match(siren::VARCHAR, '[0-9]{{9}}')
            """
            if accounts
            else """
            SELECT ''::VARCHAR source_record_uid, ''::VARCHAR snapshot_id,
                   ''::VARCHAR archive_member, 0::BIGINT source_record_ordinal,
                   ''::VARCHAR filing_id, ''::VARCHAR siren,
                   ''::VARCHAR denomination, NULL::DATE filing_date,
                   NULL::DATE closing_date, NULL::DATE previous_closing_date,
                   ''::VARCHAR updated_at, ''::VARCHAR chronology_number,
                   ''::VARCHAR confidentiality, false::BOOLEAN is_public,
                   false::BOOLEAN is_deleted, ''::VARCHAR account_type,
                   ''::VARCHAR currency, NULL::SMALLINT duration_months,
                   ''::VARCHAR activity_code,
                   false::BOOLEAN structured_accounts_present WHERE false
            """
        )
        counts["rne_account_deposits"] = _copy(
            connection, account_query, stage / "rne_account_deposits.parquet"
        )
        names_path = _sql_path(stage / "name_evidence.parquet")
        addresses_path = _sql_path(stage / "address_evidence.parquet")
        sites_path = _sql_path(stage / "establishments.parquet")
        relations_path = _sql_path(stage / "relations.parquet")
        accounts_path = _sql_path(stage / "rne_account_deposits.parquet")
        entity_path = _sql_path(stage / "entity_evidence.parquet")
        counts["address_site_resolution"] = _copy(
            connection,
            f"""
            WITH candidates AS (
              SELECT a.siren, a.source, a.normalized_value, a.postcode, a.insee,
                     a.valid_from, a.valid_to, s.siret
              FROM read_parquet('{addresses_path}') a
              JOIN read_parquet('{sites_path}') s ON s.siren=a.siren
               AND a.subject_kind='SIREN'
               AND a.normalized_value<>''
               AND a.normalized_value=s.current_address_normalized
               AND (a.postcode='' OR a.postcode=s.postcode)
               AND (a.insee='' OR a.insee=s.insee)
            )
            SELECT siren, source, normalized_value, postcode, insee, valid_from, valid_to,
                   count(DISTINCT siret)::INTEGER candidate_siret_count,
                   CASE WHEN count(DISTINCT siret)=1 THEN min(siret) ELSE NULL END resolved_siret,
                   CASE WHEN count(DISTINCT siret)=1 THEN 'UNIQUE_EXACT_SITE'
                        ELSE 'AMBIGUOUS_EXACT_SITE' END resolution_status
            FROM candidates
            GROUP BY ALL
            """,
            stage / "address_site_resolution.parquet",
        )
        resolution_path = _sql_path(stage / "address_site_resolution.parquet")
        counts["siren_summary"] = _copy(
            connection,
            f"""
            WITH sites AS (
              SELECT siren, count(*)::INTEGER site_count,
                     count(*) FILTER (WHERE administrative_state='A')::INTEGER active_site_count,
                     count(DISTINCT insee) FILTER (WHERE insee<>'')::INTEGER insee_count,
                     count(DISTINCT postcode) FILTER (WHERE postcode<>'')::INTEGER postcode_count
              FROM read_parquet('{sites_path}') GROUP BY siren
            ), names AS (
              SELECT siren, count(*)::INTEGER name_evidence_count,
                     count(DISTINCT source)::INTEGER name_source_count,
                     count(DISTINCT normalized_value)::INTEGER distinct_name_count
              FROM read_parquet('{names_path}') GROUP BY siren
            ), addresses AS (
              SELECT siren, count(*)::INTEGER address_evidence_count,
                     count(DISTINCT source)::INTEGER address_source_count,
                     count(DISTINCT normalized_value)::INTEGER distinct_address_count
              FROM read_parquet('{addresses_path}') GROUP BY siren
            ), links AS (
              SELECT identifier siren, count(*)::INTEGER relation_count FROM (
                SELECT CASE WHEN from_kind='SIREN' THEN from_identifier ELSE left(from_identifier,9) END identifier
                FROM read_parquet('{relations_path}')
                UNION ALL
                SELECT CASE WHEN to_kind='SIREN' THEN to_identifier ELSE left(to_identifier,9) END identifier
                FROM read_parquet('{relations_path}')
              ) GROUP BY identifier
            ), resolved AS (
              SELECT siren, count(*) FILTER (WHERE resolution_status='UNIQUE_EXACT_SITE')::INTEGER resolved_external_site_count,
                     count(*) FILTER (WHERE resolution_status='AMBIGUOUS_EXACT_SITE')::INTEGER ambiguous_external_site_count
              FROM read_parquet('{resolution_path}') GROUP BY siren
            ), accounts AS (
              SELECT siren,
                     count(*)::INTEGER rne_account_deposit_count,
                     count(*) FILTER (WHERE is_public AND NOT is_deleted)::INTEGER rne_public_account_period_count,
                     count(*) FILTER (WHERE NOT is_public AND NOT is_deleted)::INTEGER rne_confidential_account_period_count,
                     max(filing_date) rne_latest_account_filing_date,
                     max(closing_date) rne_latest_account_closing_date
              FROM read_parquet('{accounts_path}') GROUP BY siren
            ), entity AS (
              SELECT siren,count(*)::INTEGER official_entity_evidence_count,
                     count(DISTINCT source)::INTEGER official_entity_source_count,
                     count(*) FILTER(WHERE is_current)::INTEGER current_entity_evidence_count,
                     count(DISTINCT administrative_state) FILTER(
                       WHERE is_current AND administrative_state<>'')::INTEGER current_state_variant_count,
                     max(observed_at) latest_official_observed_at
              FROM read_parquet('{entity_path}') GROUP BY siren
            )
            SELECT u.siren, u.administrative_state, u.creation_date, u.legal_category,
                   u.main_activity, u.enterprise_category, u.headquarters_nic,
                   coalesce(s.site_count,0) site_count, coalesce(s.active_site_count,0) active_site_count,
                   coalesce(s.insee_count,0) insee_count, coalesce(s.postcode_count,0) postcode_count,
                   coalesce(n.name_evidence_count,0) name_evidence_count,
                   coalesce(n.name_source_count,0) name_source_count,
                   coalesce(n.distinct_name_count,0) distinct_name_count,
                   coalesce(a.address_evidence_count,0) address_evidence_count,
                   coalesce(a.address_source_count,0) address_source_count,
                   coalesce(a.distinct_address_count,0) distinct_address_count,
                   coalesce(l.relation_count,0) relation_count,
                   coalesce(r.resolved_external_site_count,0) resolved_external_site_count,
                   coalesce(r.ambiguous_external_site_count,0) ambiguous_external_site_count
                   ,coalesce(ac.rne_account_deposit_count,0) rne_account_deposit_count
                   ,coalesce(ac.rne_public_account_period_count,0) rne_public_account_period_count
                   ,coalesce(ac.rne_confidential_account_period_count,0) rne_confidential_account_period_count
                   ,ac.rne_latest_account_filing_date, ac.rne_latest_account_closing_date
                   ,coalesce(en.official_entity_evidence_count,0) official_entity_evidence_count
                   ,coalesce(en.official_entity_source_count,0) official_entity_source_count
                   ,coalesce(en.current_entity_evidence_count,0) current_entity_evidence_count
                   ,coalesce(en.current_state_variant_count,0) current_state_variant_count
                   ,en.latest_official_observed_at
            FROM read_parquet('{_sql_path(stage / 'legal_units.parquet')}') u
            LEFT JOIN sites s USING(siren) LEFT JOIN names n USING(siren)
            LEFT JOIN addresses a USING(siren) LEFT JOIN links l USING(siren)
            LEFT JOIN resolved r USING(siren)
            LEFT JOIN accounts ac USING(siren)
            LEFT JOIN entity en USING(siren)
            """,
            stage / "siren_summary.parquet",
        )
        connection.close()
        (stage / "build.duckdb").unlink(missing_ok=True)
        if spill == stage / "duckdb_tmp":
            shutil.rmtree(spill, ignore_errors=True)

        # The persistent catalog stores relative files; ``open_siren_dossier``
        # binds process-local views after publication. This keeps the artifact
        # relocatable and the directory publication atomic.
        catalog = duckdb.connect(str(stage / "dossier.duckdb"))
        catalog.execute(
            "CREATE TABLE dossier_metadata(key VARCHAR PRIMARY KEY, value VARCHAR)"
        )
        catalog.execute(
            "CREATE TABLE dossier_files(view_name VARCHAR PRIMARY KEY, relative_path VARCHAR)"
        )
        catalog.executemany("INSERT INTO dossier_files VALUES (?, ?)", _DOSSIER_TABLE_FILES)
        catalog.executemany(
            "INSERT INTO dossier_metadata VALUES (?, ?)",
            [("schema_version", SIREN_DOSSIER_SCHEMA_VERSION), ("build_id", build_id)],
        )
        catalog.close()
        files = {}
        for path in sorted(stage.iterdir()):
            if path.is_file() and path.name != "manifest.json":
                files[path.name] = {
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
        manifest = {
            **identity,
            "build_id": build_id,
            "counts": counts,
            "files": files,
            "consumer_contract": {
                "retrieval": ["name_evidence", "address_evidence", "address_site_resolution", "official_relations"],
                "ranker": ["establishments", "siren_summary", "name_evidence", "address_evidence", "entity_evidence"],
                "decider": ["siren_summary", "address_site_resolution", "official_relations", "entity_evidence"],
                "risk": ["siren_summary", "address_site_resolution", "official_relations", "entity_evidence"],
                "fusion_text": ["name_evidence", "address_evidence"],
                "held_out_structured": ["rne_account_deposits"],
            },
        }
        (stage / "manifest.json").write_bytes(canonical_json(manifest))
        os.rename(stage, final)
        return SirenDossierBuild(final, final / "manifest.json", final / "dossier.duckdb", build_id, counts)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def project_dossier_candidate_features(
    *, dossier_dir: Path,
    candidates_path: Path,
    output_path: Path,
) -> int:
    """Project shared official features for retrieval/ranker/decider/risk.

    Input must contain ``query_id`` and ``candidate_siret``. Optional normalized
    CRM fields add similarity features without changing the dossier itself.
    """
    dossier_dir = Path(dossier_dir)
    manifest = json.loads((dossier_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") not in SUPPORTED_SIREN_DOSSIER_SCHEMA_VERSIONS:
        raise ValueError("incompatible SIREN dossier")
    connection = duckdb.connect()
    columns = {
        row[0]
        for row in connection.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{_sql_path(Path(candidates_path))}')"
        ).fetchall()
    }
    required = {"query_id", "candidate_siret"}
    if not required.issubset(columns):
        raise ValueError("candidate input requires query_id and candidate_siret")
    name = "coalesce(c.crm_name_normalized, '')" if "crm_name_normalized" in columns else "''"
    address = "coalesce(c.crm_address_normalized, '')" if "crm_address_normalized" in columns else "''"
    insee = "coalesce(c.crm_insee, '')" if "crm_insee" in columns else "''"
    postcode = "coalesce(c.crm_postcode, '')" if "crm_postcode" in columns else "''"
    sites = _sql_path(dossier_dir / "establishments.parquet")
    legal_units = _sql_path(dossier_dir / "legal_units.parquet")
    summary = _sql_path(dossier_dir / "siren_summary.parquet")
    names = _sql_path(dossier_dir / "name_evidence.parquet")
    addresses = _sql_path(dossier_dir / "address_evidence.parquet")
    resolutions = _sql_path(dossier_dir / "address_site_resolution.parquet")
    relations = _sql_path(dossier_dir / "relations.parquet")
    entity_path = dossier_dir / "entity_evidence.parquet"
    entity_relation = (
        f"read_parquet('{_sql_path(entity_path)}')"
        if entity_path.is_file()
        else "(SELECT ''::VARCHAR siren,''::VARCHAR siret,''::VARCHAR source,"
             "''::VARCHAR administrative_state,NULL::BOOLEAN is_headquarters,"
             "false::BOOLEAN is_current,NULL::DATE observed_at WHERE false)"
    )
    candidates = _sql_path(Path(candidates_path))
    query = f"""
      WITH base AS (
        SELECT c.*, left(c.candidate_siret,9) candidate_siren,
               {_normalized(name)} crm_name_norm,
               {_normalized(address)} crm_address_norm,
               {insee} crm_insee_value, {postcode} crm_postcode_value
        FROM read_parquet('{candidates}') c
      ), name_features AS (
        SELECT b.query_id, b.candidate_siret,
          max(CASE WHEN b.crm_name_norm<>'' THEN jaro_winkler_similarity(b.crm_name_norm,n.normalized_value) ELSE 0 END) max_official_name_jw,
          max(CASE WHEN b.crm_name_norm=n.normalized_value AND b.crm_name_norm<>'' THEN 1 ELSE 0 END) exact_official_name,
          max(CASE WHEN n.subject_kind='SIREN' AND n.name_kind='LEGAL' AND n.is_current
                    AND n.source<>'BODACC' AND b.crm_name_norm<>''
                   THEN jaro_winkler_similarity(b.crm_name_norm,n.normalized_value) ELSE 0 END) max_current_legal_name_jw,
          max(CASE WHEN n.subject_kind='SIREN' AND n.name_kind<>'LEGAL' AND n.is_current
                    AND n.source<>'BODACC' AND b.crm_name_norm<>''
                   THEN jaro_winkler_similarity(b.crm_name_norm,n.normalized_value) ELSE 0 END) max_current_trade_name_jw,
          max(CASE WHEN n.subject_kind='SIRET' AND n.siret=b.candidate_siret AND n.is_current
                    AND n.source<>'BODACC' AND b.crm_name_norm<>''
                   THEN jaro_winkler_similarity(b.crm_name_norm,n.normalized_value) ELSE 0 END) max_current_site_name_jw,
          max(CASE WHEN (NOT n.is_current OR n.source='SIRENE_HISTORY' OR n.name_kind='HISTORICAL')
                    AND b.crm_name_norm<>''
                   THEN jaro_winkler_similarity(b.crm_name_norm,n.normalized_value) ELSE 0 END) max_historical_name_jw,
          max(CASE WHEN n.source='RNE' AND b.crm_name_norm<>''
                   THEN jaro_winkler_similarity(b.crm_name_norm,n.normalized_value) ELSE 0 END) max_rne_name_jw,
          max(CASE WHEN n.source='BODACC' AND b.crm_name_norm<>''
                   THEN jaro_winkler_similarity(b.crm_name_norm,n.normalized_value) ELSE 0 END) max_bodacc_name_jw,
          max(CASE WHEN n.subject_kind='SIREN' AND n.name_kind='LEGAL' AND n.is_current
                    AND n.source<>'BODACC' AND b.crm_name_norm=n.normalized_value AND b.crm_name_norm<>''
                   THEN 1 ELSE 0 END) exact_current_legal_name,
          max(CASE WHEN n.subject_kind='SIRET' AND n.siret=b.candidate_siret AND n.is_current
                    AND n.source<>'BODACC' AND b.crm_name_norm=n.normalized_value AND b.crm_name_norm<>''
                   THEN 1 ELSE 0 END) exact_current_site_name,
          count(DISTINCT n.source)::INTEGER official_name_source_count,
          count(*) FILTER (WHERE NOT n.is_current)::INTEGER historical_name_count,
          count(DISTINCT n.normalized_value) FILTER(WHERE n.is_current)::INTEGER current_name_variant_count,
          max(consensus.source_count)::INTEGER matched_name_max_source_consensus
        FROM base b LEFT JOIN read_parquet('{names}') n
          ON n.siren=b.candidate_siren AND (n.siret='' OR n.siret=b.candidate_siret)
        LEFT JOIN (
          SELECT siren,siret,normalized_value,count(DISTINCT source)::INTEGER source_count
          FROM read_parquet('{names}') GROUP BY siren,siret,normalized_value
        ) consensus ON consensus.siren=n.siren AND consensus.siret=n.siret
          AND consensus.normalized_value=n.normalized_value
        GROUP BY b.query_id,b.candidate_siret
      ), address_features AS (
        SELECT b.query_id,b.candidate_siret,
          max(CASE WHEN b.crm_address_norm<>'' THEN jaro_winkler_similarity(b.crm_address_norm,a.normalized_value) ELSE 0 END) max_official_address_jw,
          max(CASE WHEN b.crm_address_norm=a.normalized_value AND b.crm_address_norm<>'' THEN 1 ELSE 0 END) exact_official_address,
          max(CASE WHEN a.subject_kind='SIRET' AND a.siret=b.candidate_siret AND a.source='SIRENE_CURRENT'
                    AND b.crm_address_norm<>''
                   THEN jaro_winkler_similarity(b.crm_address_norm,a.normalized_value) ELSE 0 END) max_current_site_address_jw,
          max(CASE WHEN NOT a.is_current AND b.crm_address_norm<>''
                   THEN jaro_winkler_similarity(b.crm_address_norm,a.normalized_value) ELSE 0 END) max_historical_address_jw,
          max(CASE WHEN a.source='RNE' AND b.crm_address_norm<>''
                   THEN jaro_winkler_similarity(b.crm_address_norm,a.normalized_value) ELSE 0 END) max_rne_address_jw,
          max(CASE WHEN a.source='BODACC' AND b.crm_address_norm<>''
                   THEN jaro_winkler_similarity(b.crm_address_norm,a.normalized_value) ELSE 0 END) max_bodacc_address_jw,
          max(CASE WHEN b.crm_insee_value<>'' AND b.crm_insee_value=a.insee THEN 1 ELSE 0 END) official_insee_agreement,
          max(CASE WHEN b.crm_postcode_value<>'' AND b.crm_postcode_value=a.postcode THEN 1 ELSE 0 END) official_postcode_agreement,
          count(DISTINCT a.source)::INTEGER official_address_source_count,
          count(*) FILTER (WHERE NOT a.is_current)::INTEGER historical_address_count,
          count(DISTINCT a.normalized_value) FILTER(WHERE a.is_current)::INTEGER current_address_variant_count
        FROM base b LEFT JOIN read_parquet('{addresses}') a
          ON a.siren=b.candidate_siren AND (a.siret='' OR a.siret=b.candidate_siret)
        GROUP BY b.query_id,b.candidate_siret
      ), relation_features AS (
        SELECT b.query_id,b.candidate_siret,count(DISTINCT r.relation_id)::INTEGER candidate_relation_count,
          count(DISTINCT r.relation_id) FILTER(WHERE r.relation_type IN
            ('ESTABLISHMENT_SUCCESSION','LEGAL_UNIT_SUCCESSION'))::INTEGER succession_relation_count,
          count(DISTINCT r.relation_id) FILTER(WHERE r.relation_type='ASSET_TRANSFER')::INTEGER asset_transfer_relation_count,
          count(DISTINCT r.source)::INTEGER relation_source_count
        FROM base b LEFT JOIN read_parquet('{relations}') r
          ON (r.from_kind='SIREN' AND r.from_identifier=b.candidate_siren)
          OR (r.to_kind='SIREN' AND r.to_identifier=b.candidate_siren)
          OR (r.from_kind='SIRET' AND r.from_identifier=b.candidate_siret)
          OR (r.to_kind='SIRET' AND r.to_identifier=b.candidate_siret)
        GROUP BY b.query_id,b.candidate_siret
      ), resolution_features AS (
        SELECT b.query_id,b.candidate_siret,
          count(*) FILTER (WHERE x.resolved_siret=b.candidate_siret)::INTEGER exact_external_site_resolution_count,
          count(*) FILTER (WHERE x.resolution_status='AMBIGUOUS_EXACT_SITE')::INTEGER ambiguous_external_site_resolution_count
        FROM base b LEFT JOIN read_parquet('{resolutions}') x ON x.siren=b.candidate_siren
        GROUP BY b.query_id,b.candidate_siret
      ), entity_features AS (
        SELECT b.query_id,b.candidate_siret,
          count(DISTINCT e.source) FILTER(WHERE e.is_current)::INTEGER current_entity_source_count,
          count(DISTINCT e.source) FILTER(
            WHERE e.is_current AND e.administrative_state<>''
              AND e.administrative_state<>s.administrative_state)::INTEGER administrative_state_conflict_source_count,
          count(DISTINCT e.source) FILTER(
            WHERE e.is_current AND e.is_headquarters IS NOT NULL
              AND e.is_headquarters=s.is_headquarters)::INTEGER headquarters_consensus_source_count,
          max(e.observed_at) latest_entity_evidence_observed_at
        FROM base b JOIN read_parquet('{sites}') s ON s.siret=b.candidate_siret
        LEFT JOIN {entity_relation} e ON e.siren=b.candidate_siren
          AND (e.siret='' OR e.siret=b.candidate_siret)
        GROUP BY b.query_id,b.candidate_siret
      )
      SELECT b.query_id,b.candidate_siret,b.candidate_siren,
        s.administrative_state candidate_administrative_state,s.is_headquarters,
        s.creation_date candidate_creation_date,s.current_period_start candidate_period_start,
        s.main_activity candidate_main_activity,u.legal_category,u.main_activity legal_unit_main_activity,
        u.enterprise_category,u.creation_date legal_unit_creation_date,
        d.site_count,d.active_site_count,d.insee_count,d.postcode_count,
        d.name_evidence_count,d.name_source_count,d.distinct_name_count,
        d.address_evidence_count,d.address_source_count,d.distinct_address_count,
        d.relation_count,d.resolved_external_site_count,d.ambiguous_external_site_count,
        n.max_official_name_jw,n.exact_official_name,n.official_name_source_count,n.historical_name_count,
        n.max_current_legal_name_jw,n.max_current_trade_name_jw,n.max_current_site_name_jw,
        n.max_historical_name_jw,n.max_rne_name_jw,n.max_bodacc_name_jw,
        n.exact_current_legal_name,n.exact_current_site_name,
        n.current_name_variant_count,n.matched_name_max_source_consensus,
        a.max_official_address_jw,a.exact_official_address,a.official_insee_agreement,a.official_postcode_agreement,
        a.official_address_source_count,a.historical_address_count,
        a.max_current_site_address_jw,a.max_historical_address_jw,a.max_rne_address_jw,a.max_bodacc_address_jw,
        a.current_address_variant_count,
        r.candidate_relation_count,r.succession_relation_count,r.asset_transfer_relation_count,r.relation_source_count,
        x.exact_external_site_resolution_count,x.ambiguous_external_site_resolution_count,
        z.current_entity_source_count,z.administrative_state_conflict_source_count,
        z.headquarters_consensus_source_count,z.latest_entity_evidence_observed_at
      FROM base b JOIN read_parquet('{sites}') s ON s.siret=b.candidate_siret
      JOIN read_parquet('{legal_units}') u ON u.siren=b.candidate_siren
      JOIN read_parquet('{summary}') d ON d.siren=b.candidate_siren
      JOIN name_features n USING(query_id,candidate_siret)
      JOIN address_features a USING(query_id,candidate_siret)
      JOIN relation_features r USING(query_id,candidate_siret)
      JOIN resolution_features x USING(query_id,candidate_siret)
      JOIN entity_features z USING(query_id,candidate_siret)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = _copy(connection, query, output_path)
    connection.close()
    return count


def materialize_dossier_retrieval_documents(
    *,
    dossier_dir: Path,
    output_dir: Path,
    name_portfolio_policy: Path | None = None,
    document_limit: int | None = None,
) -> Mapping[str, int]:
    """Materialize bounded, typed SIRET and SIREN retrieval documents.

    National RNE/BODACC history makes an unbounded bag of names actively
    harmful: large SIREN receive more terms and present-day identity gets mixed
    with weak historical evidence.  V2 preserves five semantic roles and caps
    them deterministically before anything reaches an index.
    """
    dossier_dir = Path(dossier_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_path = Path(name_portfolio_policy) if name_portfolio_policy else (
        Path(__file__).resolve().parents[2] / "config" / "siren_name_portfolio_v1.json"
    )
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if policy.get("schema_version") != "sireto-siren-name-portfolio-v1":
        raise ValueError("unsupported SIREN name portfolio policy")
    roles = policy["roles"]

    def cap(role: str, grain: str) -> int:
        value = roles[role].get(f"maximum_per_{grain}")
        return int(value or roles[role].get("maximum_per_siren") or 0)

    connection = open_siren_dossier(dossier_dir)
    if document_limit is not None and document_limit < 1:
        raise ValueError("document_limit must be positive")
    scope_limit = f"LIMIT {int(document_limit)}" if document_limit else ""
    connection.execute(
        f"CREATE TEMP VIEW retrieval_establishment_scope AS "
        f"SELECT * FROM establishments ORDER BY siret {scope_limit}"
    )
    connection.execute(
        "CREATE TEMP VIEW retrieval_siren_scope AS "
        "SELECT DISTINCT siren FROM retrieval_establishment_scope"
    )
    connection.execute(
        """CREATE TEMP VIEW retrieval_name_scope AS
        SELECT n.* FROM name_evidence n JOIN retrieval_siren_scope s USING(siren)
        WHERE n.subject_kind='SIREN' OR n.siret IN
          (SELECT siret FROM retrieval_establishment_scope)"""
    )
    connection.execute(
        """CREATE TEMP VIEW retrieval_address_scope AS
        SELECT a.* FROM address_evidence a JOIN retrieval_siren_scope s USING(siren)
        WHERE a.subject_kind='SIREN' OR a.siret IN
          (SELECT siret FROM retrieval_establishment_scope)"""
    )
    connection.execute(
        """CREATE TEMP VIEW retrieval_relation_scope AS
        SELECT r.* FROM official_relations r
        WHERE (r.from_kind='SIREN' AND r.from_identifier IN (SELECT siren FROM retrieval_siren_scope))
           OR (r.to_kind='SIREN' AND r.to_identifier IN (SELECT siren FROM retrieval_siren_scope))
           OR (r.from_kind='SIRET' AND r.from_identifier IN (SELECT siret FROM retrieval_establishment_scope))
           OR (r.to_kind='SIRET' AND r.to_identifier IN (SELECT siret FROM retrieval_establishment_scope))"""
    )
    portfolio_query = f"""
      WITH classified AS (
        SELECT siren,siret,subject_kind,source,name_kind,raw_value,normalized_value,
               valid_from,valid_to,is_current,evidence_id,source_record_id,
               observed_at,source_priority,
          CASE
            WHEN source='BODACC' THEN 'SUPPORTING'
            WHEN NOT is_current OR source='SIRENE_HISTORY' OR name_kind='HISTORICAL'
              THEN 'HISTORICAL'
            WHEN subject_kind='SIRET' THEN 'SITE_CURRENT'
            WHEN name_kind='LEGAL' THEN 'LEGAL_CURRENT'
            ELSE 'TRADE_CURRENT'
          END name_role
        FROM retrieval_name_scope WHERE normalized_value<>''
      ), deduplicated AS (
        SELECT siren,siret,subject_kind,name_role,normalized_value,
               min(raw_value) raw_value,
               string_agg(DISTINCT source, '|' ORDER BY source) sources,
               count(DISTINCT source)::INTEGER source_count,
               max(source_priority)::SMALLINT source_priority,
               max(is_current) is_current,max(valid_from) valid_from,
               max(valid_to) valid_to,max(observed_at) observed_at,
               min(evidence_id) evidence_id,min(source_record_id) source_record_id
        FROM classified
        GROUP BY siren,siret,subject_kind,name_role,normalized_value
      ), ranked AS (
        SELECT *,
          row_number() OVER (
            PARTITION BY subject_kind,CASE WHEN subject_kind='SIRET' THEN siret ELSE siren END,name_role
            ORDER BY is_current DESC,source_priority DESC,source_count DESC,
                     valid_from DESC NULLS LAST,observed_at DESC NULLS LAST,normalized_value
          ) subject_rank,
          row_number() OVER (
            PARTITION BY siren,name_role
            ORDER BY is_current DESC,source_priority DESC,source_count DESC,
                     valid_from DESC NULLS LAST,observed_at DESC NULLS LAST,normalized_value,siret
          ) siren_rank
        FROM deduplicated
      )
      SELECT *,
        CASE name_role
          WHEN 'LEGAL_CURRENT' THEN subject_rank<={cap('LEGAL_CURRENT', 'siren')}
          WHEN 'TRADE_CURRENT' THEN subject_rank<={cap('TRADE_CURRENT', 'siren')}
          WHEN 'SITE_CURRENT' THEN subject_rank<={cap('SITE_CURRENT', 'siret')}
          WHEN 'HISTORICAL' THEN subject_rank<=CASE WHEN subject_kind='SIRET'
            THEN {cap('HISTORICAL', 'siret')} ELSE {cap('HISTORICAL', 'siren')} END
          WHEN 'SUPPORTING' THEN subject_rank<=CASE WHEN subject_kind='SIRET'
            THEN {cap('SUPPORTING', 'siret')} ELSE {cap('SUPPORTING', 'siren')} END
          ELSE false END selected_for_subject,
        CASE name_role
          WHEN 'LEGAL_CURRENT' THEN siren_rank<={cap('LEGAL_CURRENT', 'siren')}
          WHEN 'TRADE_CURRENT' THEN siren_rank<={cap('TRADE_CURRENT', 'siren')}
          WHEN 'SITE_CURRENT' THEN siren_rank<={cap('SITE_CURRENT', 'siren')}
          WHEN 'HISTORICAL' THEN siren_rank<={cap('HISTORICAL', 'siren')}
          WHEN 'SUPPORTING' THEN siren_rank<={cap('SUPPORTING', 'siren')}
          ELSE false END selected_for_siren
      FROM ranked
    """
    portfolio_path = output_dir / "retrieval_name_portfolio.parquet"
    portfolio_count = _copy(connection, portfolio_query, portfolio_path)
    portfolio = _sql_path(portfolio_path)

    site_query = """
      WITH parent_names AS (
        SELECT siren,
          string_agg(normalized_value, ' | ' ORDER BY siren_rank)
            FILTER(WHERE name_role='LEGAL_CURRENT' AND selected_for_siren) legal_current_names,
          string_agg(normalized_value, ' | ' ORDER BY siren_rank)
            FILTER(WHERE name_role='TRADE_CURRENT' AND selected_for_siren) trade_current_names,
          string_agg(normalized_value, ' | ' ORDER BY siren_rank)
            FILTER(WHERE name_role='HISTORICAL' AND selected_for_siren) historical_names,
          string_agg(normalized_value, ' | ' ORDER BY siren_rank)
            FILTER(WHERE name_role='SUPPORTING' AND selected_for_siren) supporting_names
        FROM read_parquet('__PORTFOLIO__') WHERE subject_kind='SIREN' GROUP BY siren
      ), site_names AS (
        SELECT siret,
          string_agg(normalized_value, ' | ' ORDER BY subject_rank)
            FILTER(WHERE name_role='SITE_CURRENT' AND selected_for_subject) site_current_names,
          string_agg(normalized_value, ' | ' ORDER BY subject_rank)
            FILTER(WHERE name_role='HISTORICAL' AND selected_for_subject) historical_names,
          string_agg(normalized_value, ' | ' ORDER BY subject_rank)
            FILTER(WHERE name_role='SUPPORTING' AND selected_for_subject) supporting_names
        FROM read_parquet('__PORTFOLIO__') WHERE subject_kind='SIRET' GROUP BY siret
      ), historical_addresses AS (
        SELECT siret,string_agg(normalized_value, ' | ' ORDER BY evidence_rank) historical_address_text
        FROM (
          SELECT siret,normalized_value,row_number() OVER (
            PARTITION BY siret ORDER BY source_priority DESC,valid_from DESC NULLS LAST,
              observed_at DESC NULLS LAST,normalized_value) evidence_rank
          FROM (SELECT DISTINCT siret,normalized_value,source_priority,valid_from,observed_at
                FROM retrieval_address_scope
                WHERE subject_kind='SIRET' AND NOT is_current AND normalized_value<>'')
        ) WHERE evidence_rank<=8 GROUP BY siret
      ), supporting_addresses AS (
        SELECT siret,string_agg(normalized_value, ' | ' ORDER BY evidence_rank) supporting_address_text
        FROM (
          SELECT siret,normalized_value,row_number() OVER (
            PARTITION BY siret ORDER BY source_priority DESC,observed_at DESC NULLS LAST,normalized_value) evidence_rank
          FROM (SELECT DISTINCT siret,normalized_value,source_priority,observed_at
                FROM retrieval_address_scope
                WHERE subject_kind='SIRET' AND source IN ('RNE','BODACC')
                  AND normalized_value<>'')
        ) WHERE evidence_rank<=6 GROUP BY siret
      ), siret_links AS (
        SELECT identifier siret,string_agg(DISTINCT linked_identifier, ' ' ORDER BY linked_identifier) linked_sirets
        FROM (
          SELECT from_identifier identifier,to_identifier linked_identifier FROM retrieval_relation_scope
          WHERE from_kind='SIRET' AND to_kind='SIRET'
          UNION ALL
          SELECT to_identifier,from_identifier FROM retrieval_relation_scope
          WHERE from_kind='SIRET' AND to_kind='SIRET'
        ) WHERE length(identifier)=14 AND length(linked_identifier)=14 GROUP BY identifier
      ), siren_links AS (
        SELECT identifier siren,string_agg(DISTINCT linked_identifier, ' ' ORDER BY linked_identifier) linked_sirens
        FROM (
          SELECT from_identifier identifier,to_identifier linked_identifier FROM retrieval_relation_scope
          WHERE from_kind='SIREN' AND to_kind='SIREN'
          UNION ALL
          SELECT to_identifier,from_identifier FROM retrieval_relation_scope
          WHERE from_kind='SIREN' AND to_kind='SIREN'
        ) WHERE length(identifier)=9 AND length(linked_identifier)=9 GROUP BY identifier
      )
      SELECT e.siret document_id,e.siren,'SIRET' document_kind,e.insee,e.postcode,
             e.street_number number,e.street_number_suffix number_suffix,
             e.administrative_state,e.is_headquarters,
             coalesce(pn.legal_current_names,'') legal_current_names,
             coalesce(pn.trade_current_names,'') trade_current_names,
             coalesce(sn.site_current_names,'') site_current_names,
             concat_ws(' | ',sn.historical_names,pn.historical_names) historical_names,
             concat_ws(' | ',sn.supporting_names,pn.supporting_names) supporting_names,
             e.current_address_normalized current_address_text,
             coalesce(ha.historical_address_text,'') historical_address_text,
             coalesce(sa.supporting_address_text,'') supporting_address_text,
             coalesce(x.linked_sirets,'') linked_sirets,
             coalesce(y.linked_sirens,'') linked_sirens
      FROM retrieval_establishment_scope e LEFT JOIN parent_names pn USING(siren)
      LEFT JOIN site_names sn USING(siret) LEFT JOIN historical_addresses ha USING(siret)
      LEFT JOIN supporting_addresses sa USING(siret) LEFT JOIN siret_links x USING(siret)
      LEFT JOIN siren_links y USING(siren)
    """.replace("__PORTFOLIO__", portfolio)
    siren_query = """
      WITH names AS (
        SELECT siren,
          string_agg(normalized_value, ' | ' ORDER BY siren_rank)
            FILTER(WHERE name_role='LEGAL_CURRENT' AND selected_for_siren) legal_current_names,
          string_agg(normalized_value, ' | ' ORDER BY siren_rank)
            FILTER(WHERE name_role='TRADE_CURRENT' AND selected_for_siren) trade_current_names,
          string_agg(normalized_value, ' | ' ORDER BY siren_rank)
            FILTER(WHERE name_role='SITE_CURRENT' AND selected_for_siren) site_current_names,
          string_agg(normalized_value, ' | ' ORDER BY siren_rank)
            FILTER(WHERE name_role='HISTORICAL' AND selected_for_siren) historical_names,
          string_agg(normalized_value, ' | ' ORDER BY siren_rank)
            FILTER(WHERE name_role='SUPPORTING' AND selected_for_siren) supporting_names
        FROM read_parquet('__PORTFOLIO__') GROUP BY siren
      ), geos AS (
        SELECT DISTINCT siren,insee,postcode FROM retrieval_establishment_scope
        WHERE insee<>'' OR postcode<>''
      ), siren_links AS (
        SELECT identifier siren,string_agg(DISTINCT linked_identifier, ' ' ORDER BY linked_identifier) linked_sirens
        FROM (
          SELECT from_identifier identifier,to_identifier linked_identifier FROM retrieval_relation_scope
          WHERE from_kind='SIREN' AND to_kind='SIREN'
          UNION ALL SELECT to_identifier,from_identifier FROM retrieval_relation_scope
          WHERE from_kind='SIREN' AND to_kind='SIREN'
        ) WHERE length(identifier)=9 AND length(linked_identifier)=9 GROUP BY identifier
      )
      SELECT u.siren document_id,u.siren,'SIREN' document_kind,
             coalesce(g.insee,'') insee,coalesce(g.postcode,'') postcode,
             '' number,'' number_suffix,u.administrative_state,false is_headquarters,
             coalesce(n.legal_current_names,'') legal_current_names,
             coalesce(n.trade_current_names,'') trade_current_names,
             coalesce(n.site_current_names,'') site_current_names,
             coalesce(n.historical_names,'') historical_names,
             coalesce(n.supporting_names,'') supporting_names,
             '' current_address_text,'' historical_address_text,'' supporting_address_text,
             '' linked_sirets,coalesce(x.linked_sirens,'') linked_sirens
      FROM legal_units u LEFT JOIN names n USING(siren) JOIN geos g USING(siren)
      LEFT JOIN siren_links x USING(siren)
    """.replace("__PORTFOLIO__", portfolio)
    counts = {
        "name_portfolio": portfolio_count,
        "siret_documents": _copy(connection, site_query, output_dir / "retrieval_siret_documents.parquet"),
        "siren_documents": _copy(connection, siren_query, output_dir / "retrieval_siren_documents.parquet"),
    }
    source_counts = {
        str(source): int(count)
        for source, count in connection.execute(
            """
            SELECT source,sum(row_count)::BIGINT FROM (
              SELECT source,count(*) row_count FROM retrieval_name_scope GROUP BY source
              UNION ALL SELECT source,count(*) FROM retrieval_address_scope GROUP BY source
              UNION ALL SELECT source,count(*) FROM retrieval_relation_scope GROUP BY source
            ) GROUP BY source ORDER BY source
            """
        ).fetchall()
    }
    temporal_complete = {"SIRENE_CURRENT", "RNE", "BODACC"}.issubset(source_counts)
    connection.close()
    (output_dir / "manifest.json").write_bytes(
        canonical_json(
            {
                "schema_version": "sireto-siren-dossier-retrieval-documents-v2",
                "dossier_manifest_sha256": sha256_file(dossier_dir / "manifest.json"),
                "name_portfolio_policy_sha256": sha256_file(policy_path),
                "counts": counts,
                "fields_separate": True,
                "blind_name_concatenation": False,
                "current_exact_only": True,
                "historical_and_supporting_rescue_only": True,
                "official_source_counts": source_counts,
                "temporal_complete": temporal_complete,
                "maximum_candidates_contract": 100,
                "document_limit": document_limit,
            }
        )
    )
    return counts


def project_dossier_fusion_text(
    *, dossier_dir: Path, candidates_path: Path, output_path: Path
) -> int:
    """Emit bounded source/role-separated evidence for neural/fusion models."""
    connection = open_siren_dossier(dossier_dir)
    candidates = _sql_path(Path(candidates_path))
    query = f"""
      WITH candidates AS (
        SELECT DISTINCT query_id::VARCHAR query_id,candidate_siret::VARCHAR candidate_siret,
               left(candidate_siret::VARCHAR,9) candidate_siren
        FROM read_parquet('{candidates}')
      ), name_rows AS (
        SELECT c.query_id,c.candidate_siret,'NAME' field,n.source,n.name_kind evidence_kind,
               CASE WHEN n.source='BODACC' THEN 'SUPPORTING'
                    WHEN NOT n.is_current OR n.source='SIRENE_HISTORY' OR n.name_kind='HISTORICAL' THEN 'HISTORICAL'
                    WHEN n.subject_kind='SIRET' THEN 'SITE_CURRENT'
                    WHEN n.name_kind='LEGAL' THEN 'LEGAL_CURRENT'
                    ELSE 'TRADE_CURRENT' END evidence_role,
               n.raw_value text_value,n.normalized_value,n.valid_from,n.valid_to,n.is_current,
               n.observed_at,n.source_priority,n.evidence_id,n.source_record_id,
               row_number() OVER (
                 PARTITION BY c.query_id,c.candidate_siret,
                   CASE WHEN n.source='BODACC' THEN 'SUPPORTING'
                        WHEN NOT n.is_current OR n.source='SIRENE_HISTORY' OR n.name_kind='HISTORICAL' THEN 'HISTORICAL'
                        WHEN n.subject_kind='SIRET' THEN 'SITE_CURRENT'
                        WHEN n.name_kind='LEGAL' THEN 'LEGAL_CURRENT' ELSE 'TRADE_CURRENT' END
                 ORDER BY n.is_current DESC,n.source_priority DESC,n.valid_from DESC NULLS LAST,
                          n.observed_at DESC NULLS LAST,n.normalized_value
               ) evidence_rank
        FROM candidates c JOIN name_evidence n ON n.siren=c.candidate_siren
         AND (n.siret='' OR n.siret=c.candidate_siret)
      ), address_rows AS (
        SELECT c.query_id,c.candidate_siret,'ADDRESS' field,a.source,'ADDRESS' evidence_kind,
               CASE WHEN a.source='SIRENE_CURRENT' AND a.siret=c.candidate_siret THEN 'SITE_CURRENT'
                    WHEN NOT a.is_current THEN 'HISTORICAL' ELSE 'SUPPORTING' END evidence_role,
               a.raw_value text_value,a.normalized_value,a.valid_from,a.valid_to,a.is_current,
               a.observed_at,a.source_priority,a.evidence_id,a.source_record_id,
               row_number() OVER (
                 PARTITION BY c.query_id,c.candidate_siret,
                   CASE WHEN a.source='SIRENE_CURRENT' AND a.siret=c.candidate_siret THEN 'SITE_CURRENT'
                        WHEN NOT a.is_current THEN 'HISTORICAL' ELSE 'SUPPORTING' END
                 ORDER BY a.is_current DESC,a.source_priority DESC,a.valid_from DESC NULLS LAST,
                          a.observed_at DESC NULLS LAST,a.normalized_value
               ) evidence_rank
        FROM candidates c JOIN address_evidence a ON a.siren=c.candidate_siren
         AND (a.siret='' OR a.siret=c.candidate_siret)
      )
      SELECT * FROM name_rows WHERE evidence_rank<=CASE evidence_role
        WHEN 'LEGAL_CURRENT' THEN 4 WHEN 'TRADE_CURRENT' THEN 8
        WHEN 'SITE_CURRENT' THEN 6 WHEN 'HISTORICAL' THEN 12 ELSE 6 END
      UNION ALL
      SELECT * FROM address_rows WHERE evidence_rank<=CASE evidence_role
        WHEN 'SITE_CURRENT' THEN 2 WHEN 'HISTORICAL' THEN 8 ELSE 6 END
    """
    count = _copy(connection, query, Path(output_path))
    connection.close()
    return count


__all__ = [
    "SIREN_DOSSIER_FEATURE_SCHEMA_VERSION",
    "SIREN_DOSSIER_SCHEMA_VERSION",
    "SirenDossierBuild",
    "SirenDossierInputs",
    "build_siren_dossier",
    "materialize_dossier_retrieval_documents",
    "open_siren_dossier",
    "project_dossier_candidate_features",
    "project_dossier_fusion_text",
]
