"""Content-addressed geographic lookup sidecars for retrieval v11.

The sidecars are a serving projection of a SIREN dossier v3.  They contain
only official establishment fields and never inspect CRM data, labels, model
scores or retrieval outcomes.  Two copies of the same narrow projection are
written with different physical orders:

``geo_site_cells``
    Optimised for commune and exact address-cell lookups.

``siren_local_sites``
    Optimised for expanding one SIREN to its establishments in one geography.

Parquet is the immutable data plane.  :class:`RetrievalSidecarReader` opens an
in-memory DuckDB connection and exposes SELECT-only, bounded APIs; it never
creates or mutates a file in the published sidecar directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping
import unicodedata

import duckdb

from .official_source_sync import canonical_json, sha256_file
from .siren_dossier import SIREN_DOSSIER_SCHEMA_VERSION


RETRIEVAL_SIDECAR_SCHEMA_VERSION = "sireto-retrieval-sidecars-v11"
GEO_SITE_CELLS_SCHEMA_VERSION = "sireto-geo-site-cells-v1"
SIREN_LOCAL_SITES_SCHEMA_VERSION = "sireto-siren-local-sites-v1"
RETRIEVAL_SIDECAR_QUERY_SCHEMA_VERSION = "sireto-retrieval-sidecar-query-v1"
MAX_QUERY_SITES = 2_000
_SIDECAR_FILENAMES = frozenset(
    {"geo_site_cells.parquet", "siren_local_sites.parquet"}
)

_SPACE_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^A-Z0-9]+")


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(character for character in text if not unicodedata.combining(character))
    return _SPACE_RE.sub(" ", _NON_ALNUM_RE.sub(" ", text.upper())).strip()


def _normalize_identifier(value: Any, width: int, *, label: str) -> str:
    normalized = re.sub(r"[^0-9]", "", str(value or ""))
    if len(normalized) != width:
        raise ValueError(f"{label} must contain exactly {width} digits")
    return normalized


def _normalize_insee(value: Any) -> str:
    normalized = re.sub(r"[^0-9A-Z]", "", str(value or "").upper())
    if not normalized or len(normalized) > 5:
        raise ValueError("INSEE must contain one to five alphanumeric characters")
    return normalized


def _normalize_optional_postcode(value: Any) -> str:
    if value in (None, ""):
        return ""
    return _normalize_identifier(value, 5, label="postcode")


def _validate_cap(value: int) -> int:
    cap = int(value)
    if not 1 <= cap <= MAX_QUERY_SITES:
        raise ValueError(f"max_sites must be between 1 and {MAX_QUERY_SITES}")
    return cap


@dataclass(frozen=True)
class RetrievalSidecarBuild:
    output_dir: Path
    manifest_path: Path
    build_id: str
    counts: Mapping[str, int]


@dataclass(frozen=True)
class LocalSite:
    siret: str
    siren: str
    insee: str
    postcode: str
    street_number: str
    street_number_suffix: str
    street_normalized: str
    current_address_normalized: str
    administrative_state: str
    is_headquarters: bool

    @property
    def active(self) -> bool:
        return self.administrative_state == "A"


@dataclass(frozen=True)
class SiteQueryResult:
    sites: tuple[LocalSite, ...]
    strategy: str
    matched_count: int
    max_sites: int
    capped: bool


def _read_dossier_manifest(dossier_dir: Path) -> tuple[dict[str, Any], Path]:
    dossier_dir = Path(dossier_dir).resolve()
    manifest_path = dossier_dir / "manifest.json"
    establishments_path = dossier_dir / "establishments.parquet"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not establishments_path.is_file():
        raise FileNotFoundError(establishments_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SIREN_DOSSIER_SCHEMA_VERSION:
        raise ValueError("retrieval v11 sidecars require a SIREN dossier v3")
    policy = manifest.get("policy") or {}
    if policy.get("crm_labels_present") is not False:
        raise ValueError("source dossier does not prove CRM labels are absent")
    if policy.get("model_scores_present") is not False:
        raise ValueError("source dossier does not prove model scores are absent")
    declared = (manifest.get("files") or {}).get("establishments.parquet") or {}
    if declared.get("size_bytes") != establishments_path.stat().st_size:
        raise ValueError("dossier establishments size does not match its manifest")
    observed_sha = sha256_file(establishments_path)
    if declared.get("sha256") != observed_sha:
        raise ValueError("dossier establishments hash does not match its manifest")
    return manifest, establishments_path


def _site_projection(source: str) -> str:
    street_normalized = (
        "trim(regexp_replace(upper(strip_accents(concat_ws(' ', street_type, "
        "street_name))), '[^A-Z0-9]+', ' ', 'g'))"
    )
    return f"""
        SELECT
            siret::VARCHAR AS siret,
            siren::VARCHAR AS siren,
            coalesce(insee, '')::VARCHAR AS insee,
            coalesce(postcode, '')::VARCHAR AS postcode,
            coalesce(street_number, '')::VARCHAR AS street_number,
            coalesce(street_number_suffix, '')::VARCHAR AS street_number_suffix,
            {street_normalized}::VARCHAR AS street_normalized,
            coalesce(current_address_normalized, '')::VARCHAR
                AS current_address_normalized,
            coalesce(administrative_state, '')::VARCHAR AS administrative_state,
            coalesce(is_headquarters, false)::BOOLEAN AS is_headquarters
        FROM {source}
        WHERE regexp_full_match(siret::VARCHAR, '[0-9]{{14}}')
          AND regexp_full_match(siren::VARCHAR, '[0-9]{{9}}')
          AND starts_with(siret::VARCHAR, siren::VARCHAR)
          AND coalesce(insee, '') <> ''
    """


def _copy_parquet(
    connection: duckdb.DuckDBPyConnection,
    query: str,
    output_path: Path,
) -> int:
    connection.execute(
        f"COPY ({query}) TO '{_sql_path(output_path)}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 65536)"
    )
    return int(
        connection.execute(
            f"SELECT count(*) FROM read_parquet('{_sql_path(output_path)}')"
        ).fetchone()[0]
    )


def _validate_existing_publication(target: Path, build_id: str) -> RetrievalSidecarBuild:
    manifest_path = target / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("existing retrieval sidecar publication is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != RETRIEVAL_SIDECAR_SCHEMA_VERSION
        or manifest.get("build_id") != build_id
    ):
        raise ValueError("retrieval sidecar content-address collision")
    file_records = manifest.get("files") or {}
    if set(file_records) != _SIDECAR_FILENAMES:
        raise ValueError("published retrieval sidecar file set is invalid")
    for name, record in file_records.items():
        path = target / name
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(record["size_bytes"]):
            raise ValueError(f"published sidecar size mismatch: {name}")
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"published sidecar hash mismatch: {name}")
    counts = {str(key): int(value) for key, value in manifest["counts"].items()}
    return RetrievalSidecarBuild(target, manifest_path, build_id, counts)


def build_retrieval_sidecars(
    *,
    dossier_dir: Path,
    output_root: Path,
    temp_directory: Path | None = None,
    threads: int = 4,
    memory_limit: str = "4GB",
) -> RetrievalSidecarBuild:
    """Build immutable v11 site lookups from one label-free dossier v3."""
    dossier_dir = Path(dossier_dir).resolve()
    dossier_manifest, establishments_path = _read_dossier_manifest(dossier_dir)
    dossier_manifest_sha = sha256_file(dossier_dir / "manifest.json")
    establishments_record = dossier_manifest["files"]["establishments.parquet"]
    identity = {
        "schema_version": RETRIEVAL_SIDECAR_SCHEMA_VERSION,
        "source": {
            "dossier_schema_version": dossier_manifest["schema_version"],
            "dossier_build_id": dossier_manifest["build_id"],
            "dossier_manifest_sha256": dossier_manifest_sha,
            "establishments_sha256": establishments_record["sha256"],
        },
        "projection": {
            "geo_site_cells_schema_version": GEO_SITE_CELLS_SCHEMA_VERSION,
            "siren_local_sites_schema_version": SIREN_LOCAL_SITES_SCHEMA_VERSION,
            "query_schema_version": RETRIEVAL_SIDECAR_QUERY_SCHEMA_VERSION,
            "normalization": "DUCKDB_STRIP_ACCENTS_UPPER_NON_ALNUM_TO_SPACE_V1",
            "closed_sites_retained": True,
            "crm_labels_present": False,
            "model_scores_present": False,
        },
    }
    build_id = hashlib.sha256(canonical_json(identity)).hexdigest()
    output_root = Path(output_root)
    target = output_root / build_id[:16]
    if target.exists():
        return _validate_existing_publication(target, build_id)

    output_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".retrieval-sidecars-v11-", dir=output_root))
    spill = Path(temp_directory).resolve() if temp_directory else stage / "duckdb_tmp"
    spill.mkdir(parents=True, exist_ok=True)
    connection: duckdb.DuckDBPyConnection | None = None
    try:
        connection = duckdb.connect()
        connection.execute(f"SET threads={max(1, int(threads))}")
        connection.execute(f"SET memory_limit='{str(memory_limit).replace(chr(39), chr(39) * 2)}'")
        connection.execute("SET preserve_insertion_order=false")
        connection.execute(f"SET temp_directory='{_sql_path(spill)}'")
        source = f"read_parquet('{_sql_path(establishments_path)}')"
        projection = _site_projection(source)

        geo_path = stage / "geo_site_cells.parquet"
        local_path = stage / "siren_local_sites.parquet"
        geo_count = _copy_parquet(
            connection,
            f"""
                WITH sites AS ({projection})
                SELECT *, count(*) OVER (PARTITION BY insee)::INTEGER
                    AS insee_site_count
                FROM sites
                ORDER BY insee, postcode, street_number, street_normalized, siret
            """,
            geo_path,
        )
        local_count = _copy_parquet(
            connection,
            f"""
                WITH sites AS ({projection})
                SELECT * FROM sites
                ORDER BY siren, insee, postcode, street_number,
                         street_normalized, siret
            """,
            local_path,
        )
        if geo_count != local_count:
            raise AssertionError("v11 site sidecars disagree on establishment count")
        connection.close()
        connection = None
        if spill == stage / "duckdb_tmp":
            shutil.rmtree(spill, ignore_errors=True)

        files: dict[str, dict[str, Any]] = {}
        for path in (geo_path, local_path):
            files[path.name] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        counts = {
            "geo_site_cells": geo_count,
            "siren_local_sites": local_count,
        }
        manifest = {
            **identity,
            "build_id": build_id,
            "counts": counts,
            "files": files,
            "query_contract": {
                "schema_version": RETRIEVAL_SIDECAR_QUERY_SCHEMA_VERSION,
                "maximum_sites_per_call": MAX_QUERY_SITES,
                "geo_cascade": [
                    "ALL_SITES_BY_INSEE_IF_COUNT_FITS",
                    "EXACT_INSEE_POSTCODE_NUMBER_STREET",
                    "EXACT_INSEE_POSTCODE_NUMBER",
                ],
                "siren_geo_state_policy": "ACTIVE_AND_CLOSED_EXPLICIT_FILTERS",
                "siren_geo_order": [
                    "ADDRESS_SIMILARITY",
                    "EXACT_NUMBER",
                    "EXACT_STREET",
                    "ACTIVE",
                    "HEADQUARTERS",
                    "SIRET",
                ],
                "address_precedes_headquarters": True,
                "read_only": True,
            },
        }
        (stage / "manifest.json").write_bytes(canonical_json(manifest))
        os.rename(stage, target)
        return RetrievalSidecarBuild(
            target, target / "manifest.json", build_id, counts
        )
    except Exception:
        if connection is not None:
            connection.close()
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _site_from_row(row: tuple[Any, ...]) -> LocalSite:
    return LocalSite(
        siret=str(row[0]),
        siren=str(row[1]),
        insee=str(row[2]),
        postcode=str(row[3]),
        street_number=str(row[4]),
        street_number_suffix=str(row[5]),
        street_normalized=str(row[6]),
        current_address_normalized=str(row[7]),
        administrative_state=str(row[8]),
        is_headquarters=bool(row[9]),
    )


_SITE_COLUMNS = (
    "siret,siren,insee,postcode,street_number,street_number_suffix,"
    "street_normalized,current_address_normalized,administrative_state,"
    "is_headquarters"
)


class RetrievalSidecarReader:
    """Bounded, cached SELECT-only access to a published v11 sidecar."""

    def __init__(
        self,
        sidecar_dir: Path,
        *,
        cache_size: int = 4_096,
        verify_hashes: bool = True,
    ) -> None:
        if cache_size < 1:
            raise ValueError("cache_size must be positive")
        self.sidecar_dir = Path(sidecar_dir).resolve()
        manifest_path = self.sidecar_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != RETRIEVAL_SIDECAR_SCHEMA_VERSION:
            raise ValueError("unsupported retrieval sidecar schema")
        file_records = self.manifest.get("files") or {}
        if set(file_records) != _SIDECAR_FILENAMES:
            raise ValueError("retrieval sidecar file set is invalid")
        for filename in sorted(_SIDECAR_FILENAMES):
            path = self.sidecar_dir / filename
            record = file_records.get(filename)
            if not path.is_file() or not isinstance(record, dict):
                raise FileNotFoundError(path)
            if path.stat().st_size != int(record.get("size_bytes", -1)):
                raise ValueError(f"retrieval sidecar size mismatch: {filename}")
            if verify_hashes and sha256_file(path) != record.get("sha256"):
                raise ValueError(f"retrieval sidecar hash mismatch: {filename}")

        # The connection has no persistent database path and the class exposes
        # no SQL mutation method.  All published files remain read-only inputs.
        self._connection = duckdb.connect()
        self._geo_path = self.sidecar_dir / "geo_site_cells.parquet"
        self._local_path = self.sidecar_dir / "siren_local_sites.parquet"
        self._all_by_insee_cached = lru_cache(maxsize=cache_size)(
            self._all_by_insee_uncached
        )
        self._exact_cell_cached = lru_cache(maxsize=cache_size)(
            self._exact_cell_uncached
        )
        self._number_cell_cached = lru_cache(maxsize=cache_size)(
            self._number_cell_uncached
        )
        self._siren_geo_cached = lru_cache(maxsize=cache_size)(
            self._siren_geo_uncached
        )
        self._sirens_geo_cached = lru_cache(maxsize=max(64, cache_size // 8))(
            self._sirens_geo_uncached
        )
        self._closed = False

    def __enter__(self) -> "RetrievalSidecarReader":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("retrieval sidecar reader is closed")

    def _rows(self, sql: str, parameters: list[Any]) -> tuple[LocalSite, ...]:
        self._ensure_open()
        return tuple(
            _site_from_row(row)
            for row in self._connection.execute(sql, parameters).fetchall()
        )

    def _all_by_insee_uncached(
        self, insee: str, max_sites: int
    ) -> SiteQueryResult:
        source = _sql_path(self._geo_path)
        count_row = self._connection.execute(
            f"SELECT max(insee_site_count) FROM read_parquet('{source}') WHERE insee=?",
            [insee],
        ).fetchone()
        count = int(count_row[0] or 0)
        if count > max_sites:
            return SiteQueryResult((), "INSEE_TOO_LARGE", count, max_sites, True)
        sites = self._rows(
            f"SELECT {_SITE_COLUMNS} FROM read_parquet('{source}') "
            "WHERE insee=? ORDER BY postcode,street_number,street_normalized,siret",
            [insee],
        )
        return SiteQueryResult(sites, "INSEE_ALL", count, max_sites, False)

    def all_sites_by_insee_if_fits(
        self, insee: str, *, max_sites: int = MAX_QUERY_SITES
    ) -> SiteQueryResult:
        """Return the whole commune only when its cardinality fits the cap."""
        self._ensure_open()
        return self._all_by_insee_cached(
            _normalize_insee(insee), _validate_cap(max_sites)
        )

    def _exact_cell_uncached(
        self,
        insee: str,
        postcode: str,
        street_number: str,
        street: str,
        max_sites: int,
    ) -> SiteQueryResult:
        source = _sql_path(self._geo_path)
        filters = (
            "insee=? AND postcode=? AND street_number=? AND street_normalized=?"
        )
        parameters = [insee, postcode, street_number, street]
        count = int(
            self._connection.execute(
                f"SELECT count(*) FROM read_parquet('{source}') WHERE {filters}",
                parameters,
            ).fetchone()[0]
        )
        if count > max_sites:
            return SiteQueryResult(
                (), "EXACT_GEO_CELL_TOO_LARGE", count, max_sites, True
            )
        rows = self._rows(
            f"SELECT {_SITE_COLUMNS} FROM read_parquet('{source}') "
            f"WHERE {filters} ORDER BY siret",
            parameters,
        )
        return SiteQueryResult(
            rows,
            "EXACT_GEO_CELL",
            count,
            max_sites,
            False,
        )

    def sites_by_exact_geo_cell(
        self,
        *,
        insee: str,
        postcode: str,
        street_number: str,
        street: str,
        max_sites: int = MAX_QUERY_SITES,
    ) -> SiteQueryResult:
        """Lookup an exact INSEE+postcode+number+street cell."""
        self._ensure_open()
        normalized_number = _normalize_text(street_number)
        normalized_street = _normalize_text(street)
        if not normalized_number or not normalized_street:
            raise ValueError("street_number and street are required")
        return self._exact_cell_cached(
            _normalize_insee(insee),
            _normalize_optional_postcode(postcode),
            normalized_number,
            normalized_street,
            _validate_cap(max_sites),
        )

    def _number_cell_uncached(
        self,
        insee: str,
        postcode: str,
        street_number: str,
        max_sites: int,
    ) -> SiteQueryResult:
        source = _sql_path(self._geo_path)
        filters = "insee=? AND postcode=? AND street_number=?"
        parameters = [insee, postcode, street_number]
        count = int(
            self._connection.execute(
                f"SELECT count(*) FROM read_parquet('{source}') WHERE {filters}",
                parameters,
            ).fetchone()[0]
        )
        if count > max_sites:
            return SiteQueryResult((), "NUMBER_CELL_TOO_LARGE", count, max_sites, True)
        rows = self._rows(
            f"SELECT {_SITE_COLUMNS} FROM read_parquet('{source}') "
            f"WHERE {filters} ORDER BY street_normalized,siret",
            parameters,
        )
        return SiteQueryResult(
            rows,
            "NUMBER_CELL",
            count,
            max_sites,
            False,
        )

    def geo_candidates(
        self,
        *,
        insee: str,
        postcode: str = "",
        street_number: str = "",
        street: str = "",
        max_sites: int = MAX_QUERY_SITES,
    ) -> SiteQueryResult:
        """Apply the bounded commune -> exact cell -> number-only cascade."""
        self._ensure_open()
        insee_value = _normalize_insee(insee)
        cap = _validate_cap(max_sites)
        commune = self._all_by_insee_cached(insee_value, cap)
        if not commune.capped:
            return commune
        postcode_value = _normalize_optional_postcode(postcode)
        number_value = _normalize_text(street_number)
        street_value = _normalize_text(street)
        if not postcode_value or not number_value:
            return SiteQueryResult(
                (),
                "INSUFFICIENT_LARGE_COMMUNE_GEO",
                commune.matched_count,
                cap,
                True,
            )
        if street_value:
            exact = self._exact_cell_cached(
                insee_value, postcode_value, number_value, street_value, cap
            )
            if exact.sites or exact.capped:
                return exact
        return self._number_cell_cached(
            insee_value, postcode_value, number_value, cap
        )

    def _siren_geo_uncached(
        self,
        siren: str,
        insee: str,
        postcode: str,
        address: str,
        street_number: str,
        street: str,
        include_active: bool,
        include_closed: bool,
        max_sites: int,
    ) -> SiteQueryResult:
        source = _sql_path(self._local_path)
        filters = ["siren=?"]
        parameters: list[Any] = [siren]
        if insee:
            filters.append("insee=?")
            parameters.append(insee)
        elif postcode:
            filters.append("postcode=?")
            parameters.append(postcode)
        states: list[str] = []
        if include_active:
            states.append("A")
        if include_closed:
            states.append("F")
        if len(states) == 1:
            filters.append("administrative_state=?")
            parameters.append(states[0])
        elif len(states) == 2:
            filters.append("administrative_state IN ('A','F')")
        else:
            return SiteQueryResult((), "SIREN_GEO", 0, max_sites, False)
        # Address evidence deliberately precedes active/headquarters state.
        order_parameters = [address, address, street_number, street_number, street, street]
        where_sql = " AND ".join(filters)
        count = int(
            self._connection.execute(
                f"SELECT count(*) FROM read_parquet('{source}') WHERE {where_sql}",
                parameters,
            ).fetchone()[0]
        )
        sql = (
            f"SELECT {_SITE_COLUMNS} FROM read_parquet('{source}') WHERE "
            + where_sql
            + " ORDER BY "
            "CASE WHEN ?<>'' THEN jaro_winkler_similarity("
            "current_address_normalized,?) ELSE 0 END DESC,"
            "CASE WHEN ?<>'' AND street_number=? THEN 1 ELSE 0 END DESC,"
            "CASE WHEN ?<>'' AND street_normalized=? THEN 1 ELSE 0 END DESC,"
            "CASE WHEN administrative_state='A' THEN 1 ELSE 0 END DESC,"
            "is_headquarters DESC,siret LIMIT ?"
        )
        rows = self._rows(
            sql,
            parameters + order_parameters + [max_sites],
        )
        return SiteQueryResult(
            rows, "SIREN_GEO", count, max_sites, count > max_sites
        )

    def sites_by_siren_geo(
        self,
        *,
        siren: str,
        insee: str = "",
        postcode: str = "",
        address: str = "",
        street_number: str = "",
        street: str = "",
        include_active: bool = True,
        include_closed: bool = True,
        max_sites: int = 32,
    ) -> SiteQueryResult:
        """Return active/closed local sites, ordered by address before HQ."""
        self._ensure_open()
        if not insee and not postcode:
            raise ValueError("sites_by_siren_geo requires INSEE or postcode")
        return self._siren_geo_cached(
            _normalize_identifier(siren, 9, label="SIREN"),
            _normalize_insee(insee) if insee else "",
            _normalize_optional_postcode(postcode),
            _normalize_text(address),
            _normalize_text(street_number),
            _normalize_text(street),
            bool(include_active),
            bool(include_closed),
            _validate_cap(max_sites),
        )

    def _sirens_geo_uncached(
        self,
        sirens: tuple[str, ...],
        insee: str,
        postcode: str,
        address: str,
        street_number: str,
        street: str,
        include_active: bool,
        include_closed: bool,
        max_sites_per_siren: int,
    ) -> tuple[tuple[str, SiteQueryResult], ...]:
        source = _sql_path(self._local_path)
        filters = ["siren IN (SELECT unnest(?))"]
        where_parameters: list[Any] = [list(sirens)]
        if insee:
            filters.append("insee=?")
            where_parameters.append(insee)
        elif postcode:
            filters.append("postcode=?")
            where_parameters.append(postcode)
        states: list[str] = []
        if include_active:
            states.append("A")
        if include_closed:
            states.append("F")
        if len(states) == 1:
            filters.append("administrative_state=?")
            where_parameters.append(states[0])
        elif len(states) == 2:
            filters.append("administrative_state IN ('A','F')")
        else:
            return tuple(
                (
                    siren,
                    SiteQueryResult(
                        (), "SIREN_GEO_BATCH", 0, max_sites_per_siren, False
                    ),
                )
                for siren in sirens
            )
        ranking_parameters = [
            address,
            address,
            street_number,
            street_number,
            street,
            street,
        ]
        sql = f"""
            WITH ranked AS (
                SELECT {_SITE_COLUMNS},
                       count(*) OVER (PARTITION BY siren) AS total_count,
                       row_number() OVER (
                           PARTITION BY siren ORDER BY
                           CASE WHEN ?<>'' THEN jaro_winkler_similarity(
                               current_address_normalized, ?) ELSE 0 END DESC,
                           CASE WHEN ?<>'' AND street_number=? THEN 1 ELSE 0 END DESC,
                           CASE WHEN ?<>'' AND street_normalized=? THEN 1 ELSE 0 END DESC,
                           CASE WHEN administrative_state='A' THEN 1 ELSE 0 END DESC,
                           is_headquarters DESC, siret
                       ) AS site_rank
                FROM read_parquet('{source}')
                WHERE {' AND '.join(filters)}
            )
            SELECT {_SITE_COLUMNS}, total_count
            FROM ranked WHERE site_rank <= ?
            ORDER BY siren, site_rank
        """
        rows = self._connection.execute(
            sql,
            ranking_parameters + where_parameters + [max_sites_per_siren],
        ).fetchall()
        grouped: dict[str, list[LocalSite]] = {siren: [] for siren in sirens}
        totals = {siren: 0 for siren in sirens}
        for row in rows:
            site = _site_from_row(row)
            grouped[site.siren].append(site)
            totals[site.siren] = int(row[10])
        return tuple(
            (
                siren,
                SiteQueryResult(
                    tuple(grouped[siren]),
                    "SIREN_GEO_BATCH",
                    totals[siren],
                    max_sites_per_siren,
                    totals[siren] > max_sites_per_siren,
                ),
            )
            for siren in sirens
        )

    def sites_by_sirens_geo(
        self,
        *,
        sirens: tuple[str, ...] | list[str],
        insee: str = "",
        postcode: str = "",
        address: str = "",
        street_number: str = "",
        street: str = "",
        include_active: bool = True,
        include_closed: bool = True,
        max_sites_per_siren: int = MAX_QUERY_SITES,
    ) -> dict[str, SiteQueryResult]:
        """Expand many parents with one Parquet scan and stable site ordering."""
        self._ensure_open()
        if not insee and not postcode:
            raise ValueError("sites_by_sirens_geo requires INSEE or postcode")
        normalized_sirens = tuple(
            dict.fromkeys(
                _normalize_identifier(value, 9, label="SIREN") for value in sirens
            )
        )
        if not normalized_sirens:
            return {}
        if len(normalized_sirens) > 512:
            raise ValueError("sites_by_sirens_geo accepts at most 512 parents")
        return dict(
            self._sirens_geo_cached(
                normalized_sirens,
                _normalize_insee(insee) if insee else "",
                _normalize_optional_postcode(postcode),
                _normalize_text(address),
                _normalize_text(street_number),
                _normalize_text(street),
                bool(include_active),
                bool(include_closed),
                _validate_cap(max_sites_per_siren),
            )
        )

    def cache_info(self) -> dict[str, Any]:
        """Expose aggregate cache counters without exposing the SQL handle."""
        return {
            "all_sites_by_insee": self._all_by_insee_cached.cache_info(),
            "exact_geo_cell": self._exact_cell_cached.cache_info(),
            "number_cell": self._number_cell_cached.cache_info(),
            "siren_geo": self._siren_geo_cached.cache_info(),
            "sirens_geo": self._sirens_geo_cached.cache_info(),
        }

    def close(self) -> None:
        if self._closed:
            return
        self._all_by_insee_cached.cache_clear()
        self._exact_cell_cached.cache_clear()
        self._number_cell_cached.cache_clear()
        self._siren_geo_cached.cache_clear()
        self._sirens_geo_cached.cache_clear()
        self._connection.close()
        self._closed = True


__all__ = [
    "GEO_SITE_CELLS_SCHEMA_VERSION",
    "LocalSite",
    "MAX_QUERY_SITES",
    "RETRIEVAL_SIDECAR_SCHEMA_VERSION",
    "RETRIEVAL_SIDECAR_QUERY_SCHEMA_VERSION",
    "RetrievalSidecarBuild",
    "RetrievalSidecarReader",
    "SIREN_LOCAL_SITES_SCHEMA_VERSION",
    "SiteQueryResult",
    "build_retrieval_sidecars",
]
