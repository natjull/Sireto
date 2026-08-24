"""Memory-mapped exact SIREN site lookup for retrieval v12.

The published store is derived byte-for-byte at the field level from the
sorted v11 ``siren_local_sites.parquet`` sidecar.  An Arrow IPC data file is
kept in source order and three NumPy mmap arrays map a numeric SIREN to its
global row interval.  A lookup therefore touches only the requested parents,
instead of asking DuckDB to inspect the 43.5M-row Parquet file for every
query.

DuckDB is still used on the tiny gathered Arrow table for the final ordering.
This deliberately reuses the exact ``jaro_winkler_similarity`` expression and
tie-breaks of :class:`retrieval_sidecars_v11.RetrievalSidecarReader`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from .retrieval_sidecars_v11 import (
    LocalSite,
    MAX_QUERY_SITES,
    RETRIEVAL_SIDECAR_SCHEMA_VERSION,
    SiteQueryResult,
    _normalize_identifier,
    _normalize_insee,
    _normalize_optional_postcode,
    _normalize_text,
    _site_from_row,
    _validate_cap,
)


MMAP_SIDECAR_SCHEMA_VERSION = "sireto-retrieval-v12-mmap-local-sites-1"
MMAP_SIDECAR_MANIFEST_VERSION = (
    "sireto-retrieval-v12-mmap-local-sites-manifest-1"
)

SITE_COLUMNS = (
    "siret",
    "siren",
    "insee",
    "postcode",
    "street_number",
    "street_number_suffix",
    "street_normalized",
    "current_address_normalized",
    "administrative_state",
    "is_headquarters",
)
_SITE_SQL = ",".join(SITE_COLUMNS)
_OUTPUT_FILES = frozenset(
    {
        "sites.arrow",
        "siren_keys.npy",
        "siren_starts.npy",
        "siren_counts.npy",
        "batch_starts.npy",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _schema_descriptor(schema: pa.Schema) -> list[dict[str, Any]]:
    return [
        {
            "name": field.name,
            "type": str(field.type),
            "nullable": bool(field.nullable),
        }
        for field in schema
    ]


def _outside_workspace(output_root: Path) -> None:
    workspace = Path(__file__).resolve().parents[2]
    resolved = output_root.resolve()
    if resolved == workspace or workspace in resolved.parents:
        raise ValueError(
            "v12 mmap sidecar output must be outside the workspace "
            "(use the external volume)"
        )


def _read_source(sidecar_dir: Path) -> tuple[dict[str, Any], Path, str, str]:
    manifest_path = sidecar_dir / "manifest.json"
    local_path = sidecar_dir / "siren_local_sites.parquet"
    if not manifest_path.is_file() or not local_path.is_file():
        raise FileNotFoundError("v11 retrieval sidecar publication is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != RETRIEVAL_SIDECAR_SCHEMA_VERSION:
        raise ValueError("v12 mmap store requires the v11 retrieval sidecar")
    record = (manifest.get("files") or {}).get(local_path.name) or {}
    if record.get("size_bytes") != local_path.stat().st_size:
        raise ValueError("v11 local-sites size mismatch")
    local_sha = sha256_file(local_path)
    if record.get("sha256") != local_sha:
        raise ValueError("v11 local-sites hash mismatch")
    source_schema = pq.read_schema(local_path)
    if tuple(source_schema.names) != SITE_COLUMNS:
        raise ValueError("v11 local-sites schema/order drift")
    return manifest, local_path, local_sha, sha256_file(manifest_path)


def _count_sirens(parquet: pq.ParquetFile, batch_size: int) -> int:
    previous = -1
    count = 0
    rows = 0
    for batch in parquet.iter_batches(batch_size=batch_size, columns=["siren"]):
        for raw in batch.column(0).to_pylist():
            text = str(raw or "")
            if len(text) != 9 or not text.isdigit():
                raise ValueError("mmap local-sites source contains an invalid SIREN")
            value = int(text)
            if value < previous:
                raise ValueError("mmap local-sites source is not sorted by SIREN")
            if value != previous:
                count += 1
                previous = value
            rows += 1
    if rows != parquet.metadata.num_rows or count < 1:
        raise ValueError("mmap local-sites source is empty or truncated")
    return count


@dataclass(frozen=True)
class MMapLocalSitesBuild:
    output_dir: Path
    manifest_path: Path
    build_id: str
    site_rows: int
    distinct_sirens: int


def _array_descriptor(path: Path) -> dict[str, Any]:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
    }


def _validate_existing(target: Path, build_id: str) -> MMapLocalSitesBuild:
    manifest_path = target / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("existing mmap sidecar publication is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != MMAP_SIDECAR_MANIFEST_VERSION
        or manifest.get("build_id") != build_id
        or manifest.get("build_identity") is None
        or hashlib.sha256(
            _canonical_json(manifest["build_identity"])
        ).hexdigest()
        != build_id
        or target.name != build_id[:16]
    ):
        raise ValueError("mmap sidecar content-address collision")
    files = manifest.get("files") or {}
    if set(files) != _OUTPUT_FILES:
        raise ValueError("mmap sidecar published file set drift")
    for name, record in files.items():
        path = target / name
        if (
            not path.is_file()
            or record.get("size_bytes") != path.stat().st_size
            or record.get("sha256") != sha256_file(path)
        ):
            raise ValueError(f"mmap sidecar integrity mismatch: {name}")
        if name.endswith(".npy") and record.get("array") != _array_descriptor(path):
            raise ValueError(f"mmap sidecar array metadata mismatch: {name}")
    return MMapLocalSitesBuild(
        output_dir=target,
        manifest_path=manifest_path,
        build_id=build_id,
        site_rows=int(manifest["site_rows"]),
        distinct_sirens=int(manifest["distinct_sirens"]),
    )


def build_mmap_local_sites(
    *,
    sidecar_dir: Path,
    output_root: Path,
    batch_size: int = 262_144,
) -> MMapLocalSitesBuild:
    """Publish a content-addressed mmap store from the official v11 sidecar."""

    sidecar_dir = Path(sidecar_dir).resolve()
    output_root = Path(output_root).resolve()
    _outside_workspace(output_root)
    if batch_size < 1:
        raise ValueError("mmap local-sites batch_size must be positive")
    _manifest, local_path, local_sha, source_manifest_sha = _read_source(
        sidecar_dir
    )
    parquet = pq.ParquetFile(local_path)
    distinct_sirens = _count_sirens(parquet, batch_size)
    source_schema = parquet.schema_arrow
    identity = {
        "schema_version": MMAP_SIDECAR_SCHEMA_VERSION,
        "source_manifest_sha256": source_manifest_sha,
        "source_local_sites_sha256": local_sha,
        "source_rows": parquet.metadata.num_rows,
        "source_schema": _schema_descriptor(source_schema),
        "batch_size": int(batch_size),
        "site_order": "SOURCE_ORDER_PRESERVED",
        "query_order": (
            "JARO_ADDRESS_DESC,NUMBER_EXACT_DESC,STREET_EXACT_DESC,"
            "ACTIVE_DESC,HEADQUARTERS_DESC,SIRET_ASC"
        ),
        "crm_labels_present": False,
        "model_scores_present": False,
    }
    build_id = hashlib.sha256(_canonical_json(identity)).hexdigest()
    target = output_root / build_id[:16]
    if target.exists():
        return _validate_existing(target, build_id)

    output_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".mmap-local-sites-v12-", dir=output_root))
    try:
        keys = np.lib.format.open_memmap(
            stage / "siren_keys.npy", mode="w+", dtype="<u4", shape=(distinct_sirens,)
        )
        starts = np.lib.format.open_memmap(
            stage / "siren_starts.npy", mode="w+", dtype="<u8", shape=(distinct_sirens,)
        )
        counts = np.lib.format.open_memmap(
            stage / "siren_counts.npy", mode="w+", dtype="<u4", shape=(distinct_sirens,)
        )
        arrow_path = stage / "sites.arrow"
        sink = pa.OSFile(str(arrow_path), "wb")
        writer = ipc.new_file(sink, source_schema)
        batch_starts_values: list[int] = []
        absolute_row = 0
        index_position = 0
        current_key: int | None = None
        current_start = 0
        current_count = 0
        try:
            for batch in parquet.iter_batches(
                batch_size=batch_size, columns=list(SITE_COLUMNS)
            ):
                batch_starts_values.append(absolute_row)
                writer.write_batch(batch)
                sirens = batch.column(batch.schema.get_field_index("siren")).to_pylist()
                for raw in sirens:
                    value = int(str(raw))
                    if current_key is None:
                        current_key = value
                        current_start = absolute_row
                    elif value != current_key:
                        keys[index_position] = current_key
                        starts[index_position] = current_start
                        counts[index_position] = current_count
                        index_position += 1
                        current_key = value
                        current_start = absolute_row
                        current_count = 0
                    current_count += 1
                    absolute_row += 1
            if current_key is not None:
                keys[index_position] = current_key
                starts[index_position] = current_start
                counts[index_position] = current_count
                index_position += 1
        finally:
            writer.close()
            sink.close()
        if (
            absolute_row != parquet.metadata.num_rows
            or index_position != distinct_sirens
        ):
            raise AssertionError("mmap local-sites index cardinality drift")
        keys.flush()
        starts.flush()
        counts.flush()
        del keys, starts, counts
        batch_starts = np.asarray(
            [*batch_starts_values, absolute_row], dtype="<u8"
        )
        np.save(stage / "batch_starts.npy", batch_starts, allow_pickle=False)

        files: dict[str, dict[str, Any]] = {}
        for name in sorted(_OUTPUT_FILES):
            path = stage / name
            record: dict[str, Any] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            if name.endswith(".npy"):
                record["array"] = _array_descriptor(path)
            files[name] = record
        publication = {
            **identity,
            "schema_version": MMAP_SIDECAR_MANIFEST_VERSION,
            "store_schema_version": MMAP_SIDECAR_SCHEMA_VERSION,
            "build_id": build_id,
            "build_identity": identity,
            "source_sidecar_dir": str(sidecar_dir),
            "site_rows": absolute_row,
            "distinct_sirens": distinct_sirens,
            "record_batches": len(batch_starts_values),
            "positive_injection": False,
            "human_labels_opened": False,
            "files": files,
        }
        (stage / "manifest.json").write_bytes(_canonical_json(publication))
        os.rename(stage, target)
        return MMapLocalSitesBuild(
            output_dir=target,
            manifest_path=target / "manifest.json",
            build_id=build_id,
            site_rows=absolute_row,
            distinct_sirens=distinct_sirens,
        )
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


class MMapLocalSitesReader:
    """Selective local-site reader with the v11 ordering contract."""

    def __init__(
        self,
        store_dir: Path,
        *,
        verify_hashes: bool = True,
        cache_size: int = 4_096,
    ) -> None:
        if cache_size < 1:
            raise ValueError("mmap reader cache_size must be positive")
        self.store_dir = Path(store_dir).resolve()
        manifest_path = self.store_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != MMAP_SIDECAR_MANIFEST_VERSION:
            raise ValueError("unsupported mmap local-sites manifest")
        identity = self.manifest.get("build_identity")
        build_id = self.manifest.get("build_id")
        if (
            not isinstance(identity, Mapping)
            or not isinstance(build_id, str)
            or hashlib.sha256(_canonical_json(identity)).hexdigest() != build_id
            or self.store_dir.name != build_id[:16]
        ):
            raise ValueError("mmap local-sites content address mismatch")
        files = self.manifest.get("files") or {}
        if set(files) != _OUTPUT_FILES:
            raise ValueError("mmap local-sites file set drift")
        for name, record in files.items():
            path = self.store_dir / name
            if not path.is_file() or record.get("size_bytes") != path.stat().st_size:
                raise ValueError(f"mmap local-sites size mismatch: {name}")
            if verify_hashes and record.get("sha256") != sha256_file(path):
                raise ValueError(f"mmap local-sites hash mismatch: {name}")
        self._keys = np.load(
            self.store_dir / "siren_keys.npy", mmap_mode="r", allow_pickle=False
        )
        self._starts = np.load(
            self.store_dir / "siren_starts.npy", mmap_mode="r", allow_pickle=False
        )
        self._counts = np.load(
            self.store_dir / "siren_counts.npy", mmap_mode="r", allow_pickle=False
        )
        self._batch_starts = np.load(
            self.store_dir / "batch_starts.npy", mmap_mode="r", allow_pickle=False
        )
        expected = int(self.manifest.get("distinct_sirens", -1))
        if not (
            len(self._keys) == len(self._starts) == len(self._counts) == expected
            and len(self._batch_starts) == int(self.manifest.get("record_batches", -1)) + 1
            and int(self._batch_starts[-1]) == int(self.manifest.get("site_rows", -1))
            and (len(self._keys) < 2 or bool(np.all(self._keys[1:] > self._keys[:-1])))
            and self._keys.dtype == np.dtype("<u4")
            and self._starts.dtype == np.dtype("<u8")
            and self._counts.dtype == np.dtype("<u4")
            and self._batch_starts.dtype == np.dtype("<u8")
        ):
            raise ValueError("mmap local-sites index arrays are inconsistent")
        self._arrow_source = pa.memory_map(str(self.store_dir / "sites.arrow"), "r")
        self._arrow = ipc.open_file(self._arrow_source)
        if tuple(self._arrow.schema.names) != SITE_COLUMNS:
            raise ValueError("mmap local-sites Arrow schema drift")
        self._connection = duckdb.connect()
        self._lookup_cached = lru_cache(maxsize=cache_size)(self._lookup_uncached)
        self._closed = False

    def __enter__(self) -> "MMapLocalSitesReader":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("mmap local-sites reader is closed")

    def _interval(self, siren: str) -> tuple[int, int] | None:
        key = np.uint32(int(siren))
        position = int(np.searchsorted(self._keys, key, side="left"))
        if position >= len(self._keys) or self._keys[position] != key:
            return None
        return int(self._starts[position]), int(self._counts[position])

    def _slice_global(self, start: int, count: int) -> pa.Table:
        stop = start + count
        batch_index = int(np.searchsorted(self._batch_starts, start, side="right") - 1)
        pieces: list[pa.Table] = []
        position = start
        while position < stop:
            batch = self._arrow.get_batch(batch_index)
            batch_start = int(self._batch_starts[batch_index])
            local = position - batch_start
            take = min(stop - position, batch.num_rows - local)
            pieces.append(pa.Table.from_batches([batch.slice(local, take)]))
            position += take
            batch_index += 1
        return pieces[0] if len(pieces) == 1 else pa.concat_tables(pieces)

    def _gather(self, sirens: Sequence[str]) -> pa.Table:
        pieces: list[pa.Table] = []
        for siren in sirens:
            interval = self._interval(siren)
            if interval is not None:
                pieces.append(self._slice_global(*interval))
        if not pieces:
            return pa.Table.from_arrays(
                [pa.array([], type=field.type) for field in self._arrow.schema],
                schema=self._arrow.schema,
            )
        return pieces[0] if len(pieces) == 1 else pa.concat_tables(pieces)

    def _lookup_uncached(
        self,
        sirens: tuple[str, ...],
        insee: str,
        postcode: str,
        address: str,
        street_number: str,
        street: str,
        include_active: bool,
        include_closed: bool,
        max_sites: int,
    ) -> tuple[tuple[str, SiteQueryResult], ...]:
        if not include_active and not include_closed:
            return tuple(
                (siren, SiteQueryResult((), "SIREN_GEO_BATCH", 0, max_sites, False))
                for siren in sirens
            )
        table = self._gather(sirens)
        if table.num_rows == 0:
            return tuple(
                (siren, SiteQueryResult((), "SIREN_GEO_BATCH", 0, max_sites, False))
                for siren in sirens
            )
        self._connection.register("_v12_mmap_sites", table)
        try:
            geo_filter = "insee=?" if insee else "postcode=?"
            geo_value = insee or postcode
            if include_active and include_closed:
                state_filter = "administrative_state IN ('A','F')"
                state_parameters: list[Any] = []
            else:
                state_filter = "administrative_state=?"
                state_parameters = ["A" if include_active else "F"]
            sql = f"""
                WITH filtered AS (
                    SELECT * FROM _v12_mmap_sites
                    WHERE {geo_filter} AND {state_filter}
                ), ranked AS (
                    SELECT {_SITE_SQL},
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
                    FROM filtered
                )
                SELECT {_SITE_SQL}, total_count
                FROM ranked WHERE site_rank <= ?
                ORDER BY siren, site_rank
            """
            parameters = [
                geo_value,
                *state_parameters,
                address,
                address,
                street_number,
                street_number,
                street,
                street,
                max_sites,
            ]
            rows = self._connection.execute(sql, parameters).fetchall()
        finally:
            self._connection.unregister("_v12_mmap_sites")
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
                    max_sites,
                    totals[siren] > max_sites,
                ),
            )
            for siren in sirens
        )

    def sites_by_sirens_geo(
        self,
        *,
        sirens: Sequence[str],
        insee: str = "",
        postcode: str = "",
        address: str = "",
        street_number: str = "",
        street: str = "",
        include_active: bool = True,
        include_closed: bool = True,
        max_sites_per_siren: int = MAX_QUERY_SITES,
    ) -> dict[str, SiteQueryResult]:
        self._ensure_open()
        if not insee and not postcode:
            raise ValueError("sites_by_sirens_geo requires INSEE or postcode")
        normalized = tuple(
            dict.fromkeys(
                _normalize_identifier(value, 9, label="SIREN") for value in sirens
            )
        )
        if not normalized:
            return {}
        if len(normalized) > 512:
            raise ValueError("sites_by_sirens_geo accepts at most 512 parents")
        values = self._lookup_cached(
            normalized,
            _normalize_insee(insee) if insee else "",
            _normalize_optional_postcode(postcode),
            _normalize_text(address),
            _normalize_text(street_number),
            _normalize_text(street),
            bool(include_active),
            bool(include_closed),
            _validate_cap(max_sites_per_siren),
        )
        return dict(values)

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
        normalized = _normalize_identifier(siren, 9, label="SIREN")
        result = self.sites_by_sirens_geo(
            sirens=[normalized],
            insee=insee,
            postcode=postcode,
            address=address,
            street_number=street_number,
            street=street,
            include_active=include_active,
            include_closed=include_closed,
            max_sites_per_siren=max_sites,
        )[normalized]
        return SiteQueryResult(
            result.sites,
            "SIREN_GEO",
            result.matched_count,
            result.max_sites,
            result.capped,
        )

    def cache_info(self) -> Any:
        return self._lookup_cached.cache_info()

    def close(self) -> None:
        if self._closed:
            return
        self._lookup_cached.cache_clear()
        self._connection.close()
        self._arrow_source.close()
        self._closed = True


__all__ = [
    "MMAP_SIDECAR_MANIFEST_VERSION",
    "MMAP_SIDECAR_SCHEMA_VERSION",
    "MMapLocalSitesBuild",
    "MMapLocalSitesReader",
    "SITE_COLUMNS",
    "build_mmap_local_sites",
]
