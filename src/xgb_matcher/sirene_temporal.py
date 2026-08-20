"""Truthful SIRENE history projection for temporal retrieval.

The SIRENE establishment-history stock does not contain addresses.  This
module therefore keeps historical identity observations separate from the
current SIRENE geography used to route a query.  A joined address is always
labelled ``geo_scaffold`` and is never exposed as historical evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from functools import lru_cache
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import duckdb


SIRENE_TEMPORAL_SCHEMA_VERSION = "sireto-sirene-temporal-v1"

_REQUIRED_COLUMNS: Mapping[str, frozenset[str]] = {
    "establishment_history": frozenset(
        {
            "siren", "siret", "dateDebut", "dateFin",
            "etatAdministratifEtablissement", "enseigne1Etablissement",
            "enseigne2Etablissement", "enseigne3Etablissement",
            "denominationUsuelleEtablissement",
        }
    ),
    "legal_unit_history": frozenset(
        {
            "siren", "dateDebut", "dateFin", "etatAdministratifUniteLegale",
            "nomUniteLegale", "nomUsageUniteLegale", "denominationUniteLegale",
            "denominationUsuelle1UniteLegale", "denominationUsuelle2UniteLegale",
            "denominationUsuelle3UniteLegale",
        }
    ),
    "successions": frozenset(
        {
            "siretEtablissementPredecesseur", "siretEtablissementSuccesseur",
            "dateLienSuccession", "transfertSiege", "continuiteEconomique",
            "dateDernierTraitementLienSuccession",
        }
    ),
    "establishments": frozenset(
        {
            "siren", "siret", "etatAdministratifEtablissement",
            "etablissementSiege", "numeroVoieEtablissement",
            "indiceRepetitionEtablissement", "typeVoieEtablissement",
            "libelleVoieEtablissement", "codePostalEtablissement",
            "codeCommuneEtablissement", "libelleCommuneEtablissement",
            "enseigne1Etablissement", "enseigne2Etablissement",
            "enseigne3Etablissement", "denominationUsuelleEtablissement",
        }
    ),
    "legal_units": frozenset(
        {
            "siren", "etatAdministratifUniteLegale", "nomUniteLegale",
            "nomUsageUniteLegale", "denominationUniteLegale",
            "denominationUsuelle1UniteLegale", "denominationUsuelle2UniteLegale",
            "denominationUsuelle3UniteLegale",
        }
    ),
}


@dataclass(frozen=True)
class SireneTemporalInputs:
    establishment_history: Path
    legal_unit_history: Path
    successions: Path
    establishments: Path
    legal_units: Path

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            path = Path(getattr(self, name)).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            object.__setattr__(self, name, path)

    def items(self) -> tuple[tuple[str, Path], ...]:
        return tuple((name, getattr(self, name)) for name in self.__dataclass_fields__)


@dataclass(frozen=True)
class SireneTemporalBuild:
    output_dir: Path
    manifest_path: Path
    counts: Mapping[str, int]


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


@lru_cache(maxsize=32)
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _schema(connection: duckdb.DuckDBPyConnection, path: Path) -> frozenset[str]:
    rows = connection.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{_sql_path(path)}')"
    ).fetchall()
    return frozenset(str(row[0]) for row in rows)


def validate_temporal_source_schemas(
    inputs: SireneTemporalInputs,
    *,
    connection: duckdb.DuckDBPyConnection | None = None,
) -> dict[str, list[str]]:
    """Fail before a national build when a real source lacks required fields."""
    owns_connection = connection is None
    connection = connection or duckdb.connect()
    try:
        observed: dict[str, list[str]] = {}
        for role, path in inputs.items():
            columns = _schema(connection, path)
            missing = sorted(_REQUIRED_COLUMNS[role] - columns)
            if missing:
                raise ValueError(f"{role} missing required columns: {missing}")
            observed[role] = sorted(columns)
        return observed
    finally:
        if owns_connection:
            connection.close()


def _prefix_clause(column: str, prefix: str | None) -> str:
    if prefix is None:
        return "TRUE"
    if not prefix.isdigit() or len(prefix) not in {1, 2}:
        raise ValueError("siren prefix must contain one or two digits")
    lower = prefix.ljust(9, "0")
    if int(prefix) == 10 ** len(prefix) - 1:
        return f"{column} >= '{lower}' AND {column} <= '999999999'"
    next_prefix = str(int(prefix) + 1).zfill(len(prefix))
    upper = next_prefix.ljust(9, "0")
    # Source identifiers are canonical fixed-width VARCHARs.  A range predicate
    # lets Parquet prune row groups; starts_with(cast(...)) forced full scans and
    # created gigabytes of needless DuckDB spill even for one smoke prefix.
    return f"{column} >= '{lower}' AND {column} < '{upper}'"


def _names_sql(columns: Sequence[str], kind: str) -> str:
    values = ",".join(f"cast({column} AS VARCHAR)" for column in columns)
    return (
        f"SELECT raw_name, '{kind}' name_kind FROM "
        f"UNNEST([{values}]) names(raw_name) WHERE nullif(trim(raw_name), '') IS NOT NULL"
    )


def temporal_preflight(
    inputs: SireneTemporalInputs,
    *,
    siren_prefix: str = "00",
    maximum_unmatched_fraction: float = 0.05,
) -> dict[str, Any]:
    """Run the one real-source smoke that gates the expensive projection."""
    if not 0 <= maximum_unmatched_fraction < 1:
        raise ValueError("maximum_unmatched_fraction must be in [0,1)")
    connection = duckdb.connect()
    try:
        schemas = validate_temporal_source_schemas(inputs, connection=connection)
        eh, uh, su, est, ul = (
            _sql_path(inputs.establishment_history),
            _sql_path(inputs.legal_unit_history),
            _sql_path(inputs.successions),
            _sql_path(inputs.establishments),
            _sql_path(inputs.legal_units),
        )
        p_hist_siret = _prefix_clause("h.siren", siren_prefix)
        p_hist_siren = _prefix_clause("h.siren", siren_prefix)
        p_succ = _prefix_clause("substr(cast(siretEtablissementPredecesseur AS VARCHAR),1,9)", siren_prefix)
        result = connection.execute(
            f"""
            WITH establishment_names AS (
              SELECT h.siret,h.siren,n.raw_name
              FROM read_parquet('{eh}') h,
              LATERAL ({_names_sql(['enseigne1Etablissement','enseigne2Etablissement','enseigne3Etablissement','denominationUsuelleEtablissement'], 'SITE')}) n
              WHERE {p_hist_siret}
            ), legal_names AS (
              SELECT h.siren,n.raw_name
              FROM read_parquet('{uh}') h,
              LATERAL ({_names_sql(['nomUniteLegale','nomUsageUniteLegale','denominationUniteLegale','denominationUsuelle1UniteLegale','denominationUsuelle2UniteLegale','denominationUsuelle3UniteLegale'], 'LEGAL')}) n
              WHERE {p_hist_siren}
            ), establishment_join AS (
              SELECT n.siret,(e.siret IS NOT NULL) AS is_matched
              FROM establishment_names n LEFT JOIN read_parquet('{est}') e USING(siret)
            ), legal_join AS (
              SELECT n.siren,(e.siren IS NOT NULL) AS is_matched
              FROM legal_names n LEFT JOIN (SELECT DISTINCT siren FROM read_parquet('{est}')) e USING(siren)
            ), succession AS (
              SELECT * FROM read_parquet('{su}') WHERE {p_succ}
            )
            SELECT
              (SELECT count(*) FROM establishment_names),
              (SELECT count(*) FILTER(WHERE is_matched) FROM establishment_join),
              (SELECT count(*) FROM legal_names),
              (SELECT count(*) FILTER(WHERE is_matched) FROM legal_join),
              (SELECT count(*) FROM succession),
              (SELECT count(*) FROM read_parquet('{ul}') WHERE {_prefix_clause('siren', siren_prefix)})
            """
        ).fetchone()
        counts = {
            "historical_siret_name_rows": int(result[0]),
            "historical_siret_name_rows_joined": int(result[1]),
            "historical_siren_name_rows": int(result[2]),
            "historical_siren_name_rows_joined": int(result[3]),
            "succession_rows": int(result[4]),
            "current_legal_units": int(result[5]),
        }
        required_nonzero = [
            "historical_siret_name_rows", "historical_siren_name_rows",
            "succession_rows", "current_legal_units",
        ]
        zeros = [name for name in required_nonzero if counts[name] == 0]
        siret_unmatched = 1 - counts["historical_siret_name_rows_joined"] / max(1, counts["historical_siret_name_rows"])
        siren_unmatched = 1 - counts["historical_siren_name_rows_joined"] / max(1, counts["historical_siren_name_rows"])
        failures = [f"zero:{name}" for name in zeros]
        if siret_unmatched > maximum_unmatched_fraction:
            failures.append(f"siret_unmatched_fraction:{siret_unmatched:.6f}")
        if siren_unmatched > maximum_unmatched_fraction:
            failures.append(f"siren_unmatched_fraction:{siren_unmatched:.6f}")
        return {
            "schema_version": SIRENE_TEMPORAL_SCHEMA_VERSION,
            "siren_prefix": siren_prefix,
            "counts": counts,
            "join_quality": {
                "siret_unmatched_fraction": siret_unmatched,
                "siren_unmatched_fraction": siren_unmatched,
                "maximum_unmatched_fraction": maximum_unmatched_fraction,
            },
            "source_columns": schemas,
            "failures": failures,
            "gate_pass": not failures,
        }
    finally:
        connection.close()


def _projection_identity(inputs: SireneTemporalInputs, prefix: str | None) -> dict[str, Any]:
    return {
        "schema_version": SIRENE_TEMPORAL_SCHEMA_VERSION,
        "siren_prefix": prefix,
        "inputs": {
            role: {"name": path.name, "size_bytes": path.stat().st_size, "sha256": _sha256_file(path)}
            for role, path in inputs.items()
        },
    }


def build_temporal_projection(
    inputs: SireneTemporalInputs,
    output_root: Path | str,
    *,
    siren_prefix: str | None = None,
    duckdb_memory_limit: str = "8GB",
    duckdb_threads: int = 4,
    preflight_report: Mapping[str, Any] | None = None,
) -> SireneTemporalBuild:
    """Build truthful temporal sidecars; never mutate dossier v3."""
    preflight_prefix = siren_prefix or "00"
    preflight = dict(preflight_report or temporal_preflight(inputs, siren_prefix=preflight_prefix))
    if not preflight["gate_pass"]:
        raise RuntimeError(f"SIRENE temporal preflight failed: {preflight['failures']}")
    identity = _projection_identity(inputs, siren_prefix)
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir = output_root / build_id
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return SireneTemporalBuild(output_dir, manifest_path, manifest["counts"])
    if output_dir.exists():
        raise RuntimeError(f"incomplete temporal projection: {output_dir}")
    stage = Path(tempfile.mkdtemp(prefix=f".{build_id}.building-", dir=output_root))
    (stage / "duckdb-tmp").mkdir()
    connection = duckdb.connect()
    connection.execute("SET preserve_insertion_order=false")
    connection.execute(f"SET memory_limit='{duckdb_memory_limit}'")
    connection.execute(f"SET threads={int(duckdb_threads)}")
    connection.execute(f"SET temp_directory='{_sql_path(stage / 'duckdb-tmp')}'")
    try:
        eh, uh, su, est, ul = (
            _sql_path(inputs.establishment_history), _sql_path(inputs.legal_unit_history),
            _sql_path(inputs.successions), _sql_path(inputs.establishments),
            _sql_path(inputs.legal_units),
        )
        prefix_history = _prefix_clause("h.siren", siren_prefix)
        prefix_current = _prefix_clause("e.siren", siren_prefix)
        prefix_current_ul = _prefix_clause("u.siren", siren_prefix)
        prefix_succ = _prefix_clause("substr(cast(siretEtablissementPredecesseur AS VARCHAR),1,9)", siren_prefix)
        normalize = lambda value: f"trim(regexp_replace(upper(strip_accents(coalesce({value},''))), '[^A-Z0-9]+', ' ', 'g'))"
        current_address = "trim(concat_ws(' ', nullif(trim(cast(e.numeroVoieEtablissement AS VARCHAR)),''), nullif(trim(cast(e.indiceRepetitionEtablissement AS VARCHAR)),''), nullif(trim(cast(e.typeVoieEtablissement AS VARCHAR)),''), nullif(trim(cast(e.libelleVoieEtablissement AS VARCHAR)),'')))"
        current_site_names = "[cast(e.enseigne1Etablissement AS VARCHAR),cast(e.enseigne2Etablissement AS VARCHAR),cast(e.enseigne3Etablissement AS VARCHAR),cast(e.denominationUsuelleEtablissement AS VARCHAR)]"
        current_legal_names = "[cast(u.nomUniteLegale AS VARCHAR),cast(u.nomUsageUniteLegale AS VARCHAR),cast(u.denominationUniteLegale AS VARCHAR),cast(u.denominationUsuelle1UniteLegale AS VARCHAR),cast(u.denominationUsuelle2UniteLegale AS VARCHAR),cast(u.denominationUsuelle3UniteLegale AS VARCHAR)]"
        # These physical barriers are intentional.  Without them DuckDB may
        # decorrelate the lateral name UNNEST before applying the prefix and
        # spill the 95M-row national expansion during a one-prefix smoke.
        connection.execute(
            f"CREATE TEMP TABLE establishment_history_scope AS SELECT * FROM read_parquet('{eh}') h WHERE {prefix_history}"
        )
        connection.execute(
            f"CREATE TEMP TABLE legal_history_scope AS SELECT * FROM read_parquet('{uh}') h WHERE {prefix_history}"
        )
        connection.execute(
            f"CREATE TEMP TABLE current_establishment_scope AS SELECT * FROM read_parquet('{est}') e WHERE {prefix_current}"
        )
        connection.execute(
            f"CREATE TEMP TABLE current_legal_scope AS SELECT * FROM read_parquet('{ul}') u WHERE {prefix_current_ul}"
        )
        connection.execute(
            f"CREATE TEMP TABLE succession_scope AS SELECT * FROM read_parquet('{su}') WHERE {prefix_succ}"
        )
        siret_out = _sql_path(stage / "historical_siret_identity.parquet")
        connection.execute(
            f"""COPY (
              WITH current_establishments AS (
                SELECT * FROM current_establishment_scope
              ), observations AS (
                SELECT h.siret,h.siren,trim(n.raw_name) raw_name,n.name_kind,
                       {normalize('n.raw_name')} normalized_name,
                       try_cast(h.dateDebut AS DATE) valid_from,try_cast(h.dateFin AS DATE) valid_to,
                       h.etatAdministratifEtablissement historical_state
                FROM establishment_history_scope h,
                LATERAL ({_names_sql(['enseigne1Etablissement','enseigne2Etablissement','enseigne3Etablissement','denominationUsuelleEtablissement'], 'SITE')}) n
                WHERE {prefix_history}
              ), joined AS (
                SELECT o.*,e.etatAdministratifEtablissement current_state,
                       cast(e.etablissementSiege AS BOOLEAN) is_headquarters,
                       e.codeCommuneEtablissement insee,e.codePostalEtablissement postcode,
                       cast(e.numeroVoieEtablissement AS VARCHAR) number,
                       cast(e.indiceRepetitionEtablissement AS VARCHAR) number_suffix,
                       {current_address} geo_scaffold_address,
                       list_transform({current_site_names}, x -> {normalize('x')}) current_names
                FROM observations o JOIN current_establishments e USING(siret,siren)
              ), dedup AS (
                SELECT * EXCLUDE(rn,current_names),sha256(concat_ws('|','SIRENE_HISTORY',siret,normalized_name,cast(valid_from AS VARCHAR),cast(valid_to AS VARCHAR))) evidence_id,
                       'SIRENE_CURRENT_ESTABLISHMENT' geo_scaffold_source,true geo_scaffold
                FROM (SELECT *,row_number() OVER(PARTITION BY siret,normalized_name ORDER BY valid_to DESC NULLS LAST,valid_from DESC NULLS LAST,raw_name) rn FROM joined
                      WHERE normalized_name <> '' AND NOT list_contains(current_names,normalized_name))
                WHERE rn=1
              ) SELECT * FROM dedup ORDER BY siret,normalized_name
            ) TO '{siret_out}' (FORMAT PARQUET,COMPRESSION ZSTD)"""
        )
        siren_out = _sql_path(stage / "historical_siren_identity.parquet")
        connection.execute(
            f"""COPY (
              WITH current_legal_units AS (
                SELECT * FROM current_legal_scope
              ), observations AS (
                SELECT h.siren,trim(n.raw_name) raw_name,n.name_kind,{normalize('n.raw_name')} normalized_name,
                       try_cast(h.dateDebut AS DATE) valid_from,try_cast(h.dateFin AS DATE) valid_to,
                       h.etatAdministratifUniteLegale historical_state
                FROM legal_history_scope h,
                LATERAL ({_names_sql(['nomUniteLegale','nomUsageUniteLegale','denominationUniteLegale','denominationUsuelle1UniteLegale','denominationUsuelle2UniteLegale','denominationUsuelle3UniteLegale'], 'LEGAL')}) n
                WHERE {prefix_history}
              ), changed AS (
                SELECT o.*,u.etatAdministratifUniteLegale current_state,
                       list_transform({current_legal_names}, x -> {normalize('x')}) current_names
                FROM observations o JOIN current_legal_units u USING(siren)
              ), dedup AS (
                SELECT * EXCLUDE(rn,current_names)
                FROM (SELECT *,row_number() OVER(PARTITION BY siren,normalized_name ORDER BY valid_to DESC NULLS LAST,valid_from DESC NULLS LAST,raw_name) rn
                      FROM changed WHERE normalized_name <> '' AND NOT list_contains(current_names,normalized_name)) WHERE rn=1
              ), geographies AS (
                SELECT DISTINCT siren,codeCommuneEtablissement insee,codePostalEtablissement postcode
                FROM current_establishment_scope e WHERE nullif(trim(codeCommuneEtablissement),'') IS NOT NULL
              )
              SELECT d.*,g.insee,g.postcode,
                     sha256(concat_ws('|','SIRENE_HISTORY',d.siren,d.normalized_name,coalesce(g.insee,''),cast(d.valid_from AS VARCHAR),cast(d.valid_to AS VARCHAR))) evidence_id
              FROM dedup d JOIN geographies g USING(siren)
              ORDER BY siren,insee,normalized_name
            ) TO '{siren_out}' (FORMAT PARQUET,COMPRESSION ZSTD)"""
        )
        relation_out = _sql_path(stage / "succession_edges.parquet")
        connection.execute(
            f"""COPY (
              SELECT cast(siretEtablissementPredecesseur AS VARCHAR) predecessor_siret,
                     substr(cast(siretEtablissementPredecesseur AS VARCHAR),1,9) predecessor_siren,
                     cast(siretEtablissementSuccesseur AS VARCHAR) successor_siret,
                     substr(cast(siretEtablissementSuccesseur AS VARCHAR),1,9) successor_siren,
                     try_cast(dateLienSuccession AS DATE) effective_date,
                     try_cast(transfertSiege AS BOOLEAN) transfer_headquarters,
                     try_cast(continuiteEconomique AS BOOLEAN) economic_continuity,
                     try_cast(dateDernierTraitementLienSuccession AS DATE) observed_at,
                     sha256(concat_ws('|','SIRENE_SUCCESSION',cast(siretEtablissementPredecesseur AS VARCHAR),cast(siretEtablissementSuccesseur AS VARCHAR),cast(dateLienSuccession AS VARCHAR))) relation_id
              FROM succession_scope
              WHERE regexp_full_match(cast(siretEtablissementPredecesseur AS VARCHAR),'[0-9]{{14}}')
                AND regexp_full_match(cast(siretEtablissementSuccesseur AS VARCHAR),'[0-9]{{14}}')
                AND siretEtablissementPredecesseur <> siretEtablissementSuccesseur
              QUALIFY row_number() OVER(PARTITION BY predecessor_siret,successor_siret,effective_date ORDER BY observed_at DESC NULLS LAST)=1
              ORDER BY predecessor_siret,successor_siret,effective_date
            ) TO '{relation_out}' (FORMAT PARQUET,COMPRESSION ZSTD)"""
        )
        counts = {
            "historical_siret_identity": connection.execute(f"SELECT count(*) FROM read_parquet('{siret_out}')").fetchone()[0],
            "historical_siren_identity": connection.execute(f"SELECT count(*) FROM read_parquet('{siren_out}')").fetchone()[0],
            "succession_edges": connection.execute(f"SELECT count(*) FROM read_parquet('{relation_out}')").fetchone()[0],
        }
        if any(int(value) == 0 for value in counts.values()):
            raise RuntimeError(f"temporal projection emitted an empty required family: {counts}")
        manifest = {
            **identity,
            "build_id": build_id,
            "counts": {key: int(value) for key, value in counts.items()},
            "preflight": preflight,
            "semantics": {
                "historical_addresses_claimed": False,
                "geo_scaffold_source": "SIRENE_CURRENT_ESTABLISHMENT",
                "succession_max_hops": 1,
                "current_authority": "SIRENE_CURRENT",
            },
            "contains_crm_labels": False,
            "contains_synthetic_rows": False,
        }
        payload = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
        (stage / "manifest.json").write_bytes(payload)
        (stage / "manifest.sha256").write_text(hashlib.sha256(payload).hexdigest()+"\n", encoding="ascii")
        connection.close()
        stage.replace(output_dir)
        fd = os.open(output_root, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        return SireneTemporalBuild(output_dir, output_dir / "manifest.json", manifest["counts"])
    except Exception:
        connection.close()
        shutil.rmtree(stage, ignore_errors=True)
        raise


def build_all_temporal_projections(
    inputs: SireneTemporalInputs,
    output_root: Path | str,
    *,
    duckdb_memory_limit: str = "8GB",
    duckdb_threads: int = 4,
) -> tuple[SireneTemporalBuild, ...]:
    """Build 100 restartable prefix artifacts after one real preflight."""
    preflight = temporal_preflight(inputs, siren_prefix="00")
    if not preflight["gate_pass"]:
        raise RuntimeError(f"SIRENE temporal preflight failed: {preflight['failures']}")
    builds = tuple(
        build_temporal_projection(
            inputs, output_root, siren_prefix=f"{value:02d}",
            duckdb_memory_limit=duckdb_memory_limit, duckdb_threads=duckdb_threads,
            preflight_report=preflight,
        )
        for value in range(100)
    )
    root = Path(output_root).resolve()
    manifest = {
        "schema_version": "sireto-sirene-temporal-prefix-set-v1",
        "prefixes": [
            {"prefix": f"{value:02d}", "output_dir": build.output_dir.name, "counts": dict(build.counts)}
            for value, build in enumerate(builds)
        ],
        "counts": {
            key: sum(int(build.counts[key]) for build in builds)
            for key in builds[0].counts
        },
        "preflight": preflight,
    }
    payload = (json.dumps(manifest, sort_keys=True, indent=2)+"\n").encode()
    (root / "prefix_set_manifest.json").write_bytes(payload)
    (root / "prefix_set_manifest.sha256").write_text(hashlib.sha256(payload).hexdigest()+"\n", encoding="ascii")
    return builds


def project_temporal_candidate_features(
    candidate_path: Path | str,
    projection_dirs: Sequence[Path | str],
    output_path: Path | str,
) -> Path:
    """Project temporal evidence for ranker/decider/risk and typed fusion.

    Candidate and business identifiers remain join keys and are explicitly
    excluded from the returned feature list in the sidecar manifest.
    """
    candidate_path = Path(candidate_path).resolve()
    if not candidate_path.is_file():
        raise FileNotFoundError(candidate_path)
    directories = tuple(Path(path).resolve() for path in projection_dirs)
    if not directories:
        raise ValueError("at least one temporal projection is required")
    siret_paths = [directory / "historical_siret_identity.parquet" for directory in directories]
    siren_paths = [directory / "historical_siren_identity.parquet" for directory in directories]
    relation_paths = [directory / "succession_edges.parquet" for directory in directories]
    if not all(path.is_file() for path in (*siret_paths, *siren_paths, *relation_paths)):
        raise FileNotFoundError("temporal projection sidecars are incomplete")
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage = output_path.with_name(f".{output_path.name}.building-{os.getpid()}")
    connection = duckdb.connect()
    try:
        columns = {
            row[0]
            for row in connection.execute(
                f"DESCRIBE SELECT * FROM read_parquet('{_sql_path(candidate_path)}')"
            ).fetchall()
        }
        siret_col = "candidate_siret" if "candidate_siret" in columns else "siret"
        siren_col = "candidate_siren" if "candidate_siren" in columns else "siren"
        insee_col = "candidate_insee" if "candidate_insee" in columns else "insee"
        required = {"query_id", siret_col, siren_col, insee_col, "crm_name"}
        if missing := sorted(required - columns):
            raise ValueError(f"candidate input missing columns: {missing}")
        siret_list = ",".join(f"'{_sql_path(path)}'" for path in siret_paths)
        siren_list = ",".join(f"'{_sql_path(path)}'" for path in siren_paths)
        relation_list = ",".join(f"'{_sql_path(path)}'" for path in relation_paths)
        connection.execute(
            f"""COPY (
              WITH candidates AS (
                SELECT query_id,cast({siret_col} AS VARCHAR) candidate_siret,
                       cast({siren_col} AS VARCHAR) candidate_siren,
                       cast({insee_col} AS VARCHAR) candidate_insee,
                       trim(regexp_replace(upper(strip_accents(coalesce(crm_name,''))), '[^A-Z0-9]+', ' ', 'g')) crm_name_norm
                FROM read_parquet('{_sql_path(candidate_path)}')
              ), siret_history AS (
                SELECT * FROM read_parquet([{siret_list}])
                QUALIFY row_number() OVER(PARTITION BY siret ORDER BY valid_to DESC NULLS LAST,valid_from DESC NULLS LAST,normalized_name)<=6
              ), siren_history AS (
                SELECT * FROM read_parquet([{siren_list}])
                QUALIFY row_number() OVER(PARTITION BY siren,insee ORDER BY valid_to DESC NULLS LAST,valid_from DESC NULLS LAST,normalized_name)<=12
              ), siret_features AS (
                SELECT c.query_id,c.candidate_siret,
                       count(DISTINCT h.normalized_name)::INTEGER temporal_siret_name_count,
                       max(jaro_winkler_similarity(c.crm_name_norm,h.normalized_name)) temporal_siret_name_jw,
                       bool_or(c.crm_name_norm=h.normalized_name) temporal_siret_name_exact,
                       min(h.valid_from) temporal_siret_first_valid_from,
                       max(h.valid_to) temporal_siret_last_valid_to,
                       string_agg(DISTINCT h.normalized_name,' | ' ORDER BY h.normalized_name) FILTER(WHERE h.normalized_name IS NOT NULL) temporal_siret_names
                FROM candidates c LEFT JOIN siret_history h
                  ON h.siret=c.candidate_siret
                GROUP BY c.query_id,c.candidate_siret
              ), siren_features AS (
                SELECT c.query_id,c.candidate_siret,
                       count(DISTINCT h.normalized_name)::INTEGER temporal_siren_name_count,
                       max(jaro_winkler_similarity(c.crm_name_norm,h.normalized_name)) temporal_siren_name_jw,
                       bool_or(c.crm_name_norm=h.normalized_name) temporal_siren_name_exact,
                       min(h.valid_from) temporal_siren_first_valid_from,
                       max(h.valid_to) temporal_siren_last_valid_to,
                       string_agg(DISTINCT h.normalized_name,' | ' ORDER BY h.normalized_name) FILTER(WHERE h.normalized_name IS NOT NULL) temporal_siren_names
                FROM candidates c LEFT JOIN siren_history h
                  ON h.siren=c.candidate_siren AND h.insee=c.candidate_insee
                GROUP BY c.query_id,c.candidate_siret
              ), relation_features AS (
                SELECT c.query_id,c.candidate_siret,
                       count(DISTINCT r.relation_id) FILTER(WHERE r.predecessor_siret=c.candidate_siret)::INTEGER succession_out_count,
                       count(DISTINCT r.relation_id) FILTER(WHERE r.successor_siret=c.candidate_siret)::INTEGER succession_in_count,
                       bool_or(coalesce(r.economic_continuity,false)) succession_economic_continuity,
                       bool_or(coalesce(r.transfer_headquarters,false)) succession_transfer_headquarters,
                       bool_or(r.predecessor_siren<>r.successor_siren) succession_cross_siren,
                       max(r.effective_date) succession_latest_effective_date
                FROM candidates c LEFT JOIN read_parquet([{relation_list}]) r
                  ON r.predecessor_siret=c.candidate_siret OR r.successor_siret=c.candidate_siret
                GROUP BY c.query_id,c.candidate_siret
              )
              SELECT s.query_id,s.candidate_siret,
                     coalesce(s.temporal_siret_name_count,0) temporal_siret_name_count,
                     coalesce(s.temporal_siret_name_jw,0.0) temporal_siret_name_jw,
                     coalesce(s.temporal_siret_name_exact,false) temporal_siret_name_exact,
                     coalesce(n.temporal_siren_name_count,0) temporal_siren_name_count,
                     coalesce(n.temporal_siren_name_jw,0.0) temporal_siren_name_jw,
                     coalesce(n.temporal_siren_name_exact,false) temporal_siren_name_exact,
                     s.temporal_siret_first_valid_from,s.temporal_siret_last_valid_to,
                     n.temporal_siren_first_valid_from,n.temporal_siren_last_valid_to,
                     coalesce(r.succession_out_count,0) succession_out_count,
                     coalesce(r.succession_in_count,0) succession_in_count,
                     coalesce(r.succession_economic_continuity,false) succession_economic_continuity,
                     coalesce(r.succession_transfer_headquarters,false) succession_transfer_headquarters,
                     coalesce(r.succession_cross_siren,false) succession_cross_siren,
                     r.succession_latest_effective_date,
                     concat_ws(' ',
                       CASE WHEN nullif(s.temporal_siret_names,'') IS NOT NULL THEN '[SIRENE_HISTORY_SIRET] '||s.temporal_siret_names ELSE NULL END,
                       CASE WHEN nullif(n.temporal_siren_names,'') IS NOT NULL THEN '[SIRENE_HISTORY_SIREN] '||n.temporal_siren_names ELSE NULL END
                     ) temporal_fusion_text
              FROM siret_features s JOIN siren_features n USING(query_id,candidate_siret)
              JOIN relation_features r USING(query_id,candidate_siret)
              ORDER BY s.query_id,s.candidate_siret
            ) TO '{_sql_path(stage)}' (FORMAT PARQUET,COMPRESSION ZSTD)"""
        )
        if stage.stat().st_size == 0:
            raise RuntimeError("empty temporal feature projection")
        stage.replace(output_path)
        manifest = {
            "schema_version": "sireto-sirene-temporal-candidate-features-v1",
            "candidate_sha256": _sha256_file(candidate_path),
            "projection_build_ids": [
                json.loads((directory / "manifest.json").read_text(encoding="utf-8"))["build_id"]
                for directory in directories
            ],
            "output_sha256": _sha256_file(output_path),
            "identifier_columns_are_features": False,
            "fusion_text_is_source_typed": True,
            "excluded": ["crm_aliases", "bodacc_free_text", "personal_data", "annual_account_values"],
        }
        manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
        manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2)+"\n", encoding="utf-8")
        return output_path
    except Exception:
        stage.unlink(missing_ok=True)
        raise
    finally:
        connection.close()


__all__ = [
    "SIRENE_TEMPORAL_SCHEMA_VERSION", "SireneTemporalBuild", "SireneTemporalInputs",
    "build_temporal_projection", "temporal_preflight", "validate_temporal_source_schemas",
    "build_all_temporal_projections",
    "project_temporal_candidate_features",
]
