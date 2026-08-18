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
SIREN_DOSSIER_SCHEMA_VERSION = "sireto-siren-dossier-v2"
SUPPORTED_SIREN_DOSSIER_SCHEMA_VERSIONS = {
    SIREN_DOSSIER_SCHEMA_VERSION_V1,
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
            FROM read_parquet('{_sql_path(stage / 'legal_units.parquet')}') u
            LEFT JOIN sites s USING(siren) LEFT JOIN names n USING(siren)
            LEFT JOIN addresses a USING(siren) LEFT JOIN links l USING(siren)
            LEFT JOIN resolved r USING(siren)
            LEFT JOIN accounts ac USING(siren)
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
                "ranker": ["establishments", "siren_summary", "name_evidence", "address_evidence"],
                "decider": ["siren_summary", "address_site_resolution", "official_relations"],
                "risk": ["siren_summary", "address_site_resolution", "official_relations"],
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
    summary = _sql_path(dossier_dir / "siren_summary.parquet")
    names = _sql_path(dossier_dir / "name_evidence.parquet")
    addresses = _sql_path(dossier_dir / "address_evidence.parquet")
    resolutions = _sql_path(dossier_dir / "address_site_resolution.parquet")
    relations = _sql_path(dossier_dir / "relations.parquet")
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
          count(DISTINCT n.source)::INTEGER official_name_source_count,
          count(*) FILTER (WHERE NOT n.is_current)::INTEGER historical_name_count
        FROM base b LEFT JOIN read_parquet('{names}') n
          ON n.siren=b.candidate_siren AND (n.siret='' OR n.siret=b.candidate_siret)
        GROUP BY b.query_id,b.candidate_siret
      ), address_features AS (
        SELECT b.query_id,b.candidate_siret,
          max(CASE WHEN b.crm_address_norm<>'' THEN jaro_winkler_similarity(b.crm_address_norm,a.normalized_value) ELSE 0 END) max_official_address_jw,
          max(CASE WHEN b.crm_address_norm=a.normalized_value AND b.crm_address_norm<>'' THEN 1 ELSE 0 END) exact_official_address,
          max(CASE WHEN b.crm_insee_value<>'' AND b.crm_insee_value=a.insee THEN 1 ELSE 0 END) official_insee_agreement,
          max(CASE WHEN b.crm_postcode_value<>'' AND b.crm_postcode_value=a.postcode THEN 1 ELSE 0 END) official_postcode_agreement,
          count(DISTINCT a.source)::INTEGER official_address_source_count,
          count(*) FILTER (WHERE NOT a.is_current)::INTEGER historical_address_count
        FROM base b LEFT JOIN read_parquet('{addresses}') a
          ON a.siren=b.candidate_siren AND (a.siret='' OR a.siret=b.candidate_siret)
        GROUP BY b.query_id,b.candidate_siret
      ), relation_features AS (
        SELECT b.query_id,b.candidate_siret,count(DISTINCT r.relation_id)::INTEGER candidate_relation_count
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
      )
      SELECT b.query_id,b.candidate_siret,b.candidate_siren,
        s.administrative_state candidate_administrative_state,s.is_headquarters,
        d.site_count,d.active_site_count,d.insee_count,d.postcode_count,
        d.name_evidence_count,d.name_source_count,d.distinct_name_count,
        d.address_evidence_count,d.address_source_count,d.distinct_address_count,
        d.relation_count,d.resolved_external_site_count,d.ambiguous_external_site_count,
        n.max_official_name_jw,n.exact_official_name,n.official_name_source_count,n.historical_name_count,
        a.max_official_address_jw,a.exact_official_address,a.official_insee_agreement,a.official_postcode_agreement,
        a.official_address_source_count,a.historical_address_count,
        r.candidate_relation_count,x.exact_external_site_resolution_count,x.ambiguous_external_site_resolution_count
      FROM base b JOIN read_parquet('{sites}') s ON s.siret=b.candidate_siret
      JOIN read_parquet('{summary}') d ON d.siren=b.candidate_siren
      JOIN name_features n USING(query_id,candidate_siret)
      JOIN address_features a USING(query_id,candidate_siret)
      JOIN relation_features r USING(query_id,candidate_siret)
      JOIN resolution_features x USING(query_id,candidate_siret)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = _copy(connection, query, output_path)
    connection.close()
    return count


def materialize_dossier_retrieval_documents(
    *, dossier_dir: Path, output_dir: Path
) -> Mapping[str, int]:
    """Materialize direct-site and hierarchical-SIREN retrieval documents."""
    dossier_dir = Path(dossier_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    connection = open_siren_dossier(dossier_dir)
    site_query = """
      WITH parent_names AS (
        SELECT siren, string_agg(DISTINCT normalized_value, ' | ' ORDER BY normalized_value) name_values
        FROM name_evidence WHERE subject_kind='SIREN' GROUP BY siren
      ), site_names AS (
        SELECT siret, string_agg(DISTINCT normalized_value, ' | ' ORDER BY normalized_value) name_values
        FROM name_evidence WHERE subject_kind='SIRET' GROUP BY siret
      ), historical_addresses AS (
        SELECT siret, string_agg(DISTINCT normalized_value, ' | ' ORDER BY normalized_value) address_values
        FROM address_evidence WHERE subject_kind='SIRET' AND NOT is_current GROUP BY siret
      )
      SELECT e.siret document_id,e.siren,'SIRET' document_kind,e.insee,e.postcode,
             e.administrative_state,e.is_headquarters,
             concat_ws(' | ',sn.name_values,pn.name_values) name_text,
             e.current_address_normalized address_text,
             coalesce(ha.address_values,'') historical_address_text
      FROM establishments e LEFT JOIN parent_names pn USING(siren)
      LEFT JOIN site_names sn USING(siret) LEFT JOIN historical_addresses ha USING(siret)
    """
    siren_query = """
      WITH names AS (
        SELECT siren,string_agg(DISTINCT normalized_value, ' | ' ORDER BY normalized_value) name_text
        FROM name_evidence GROUP BY siren
      ), addresses AS (
        SELECT siren,string_agg(DISTINCT normalized_value, ' | ' ORDER BY normalized_value) address_text
        FROM address_evidence GROUP BY siren
      ), geos AS (
        SELECT siren,string_agg(DISTINCT insee, ' ' ORDER BY insee) FILTER(WHERE insee<>'') insee_values,
               string_agg(DISTINCT postcode, ' ' ORDER BY postcode) FILTER(WHERE postcode<>'') postcode_values
        FROM establishments GROUP BY siren
      )
      SELECT u.siren document_id,u.siren,'SIREN' document_kind,
             coalesce(g.insee_values,'') insee,coalesce(g.postcode_values,'') postcode,
             u.administrative_state,false is_headquarters,
             coalesce(n.name_text,'') name_text,coalesce(a.address_text,'') address_text,
             '' historical_address_text
      FROM legal_units u LEFT JOIN names n USING(siren)
      LEFT JOIN addresses a USING(siren) LEFT JOIN geos g USING(siren)
    """
    counts = {
        "siret_documents": _copy(connection, site_query, output_dir / "retrieval_siret_documents.parquet"),
        "siren_documents": _copy(connection, siren_query, output_dir / "retrieval_siren_documents.parquet"),
    }
    connection.close()
    (output_dir / "manifest.json").write_bytes(
        canonical_json(
            {
                "schema_version": "sireto-siren-dossier-retrieval-documents-v1",
                "dossier_manifest_sha256": sha256_file(dossier_dir / "manifest.json"),
                "counts": counts,
                "fields_separate": True,
                "maximum_candidates_contract": 100,
            }
        )
    )
    return counts


def project_dossier_fusion_text(
    *, dossier_dir: Path, candidates_path: Path, output_path: Path
) -> int:
    """Emit source-separated text evidence for BGE/CamemBERT/fusion models."""
    connection = open_siren_dossier(dossier_dir)
    candidates = _sql_path(Path(candidates_path))
    query = f"""
      WITH candidates AS (
        SELECT DISTINCT query_id::VARCHAR query_id,candidate_siret::VARCHAR candidate_siret,
               left(candidate_siret::VARCHAR,9) candidate_siren
        FROM read_parquet('{candidates}')
      )
      SELECT c.query_id,c.candidate_siret,'NAME' field,n.source,n.name_kind evidence_kind,
             n.raw_value text_value,n.normalized_value,n.valid_from,n.valid_to,n.is_current
      FROM candidates c JOIN name_evidence n ON n.siren=c.candidate_siren
       AND (n.siret='' OR n.siret=c.candidate_siret)
      UNION ALL
      SELECT c.query_id,c.candidate_siret,'ADDRESS',a.source,'ADDRESS',
             a.raw_value,a.normalized_value,a.valid_from,a.valid_to,a.is_current
      FROM candidates c JOIN address_evidence a ON a.siren=c.candidate_siren
       AND (a.siret='' OR a.siret=c.candidate_siret)
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
