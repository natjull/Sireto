"""Cache-neutral I/O bridge from immutable v12 W1 to a fresh v13 W3.

This module has two deliberately separate phases.

``build_immutable_w1_io_projection_v13`` reads the pinned v12 benchmark and
produces a content-addressed projection of its W1 source extents.  It never
runs W1.  ``compare_w3_to_immutable_w1_io_v13`` then compares one independently
sealed v13 W3 run against that projection.  Process and machine disk counters
are never consulted, so the decision is independent of page-cache state.

The v12 STOP remains immutable: neither API writes to, authorises, or changes
the interpretation of any v12 artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
import bisect
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from .retrieval_fold_fused_oof_v12 import sha256_file
from .retrieval_top100_cascade_v12 import production_projection


W1_PROJECTION_SCHEMA_VERSION = (
    "sireto-retrieval-v13-immutable-w1-logical-io-projection-1"
)
IO_GATE_SCHEMA_VERSION = "sireto-retrieval-v13-cache-neutral-io-gate-1"
W3_RUN_SCHEMA_VERSION = "sireto-retrieval-v13-sharded-run-1"
V12_BENCHMARK_SCHEMA_VERSION = "sireto-retrieval-v12-physical-worker-benchmark-1"
V12_SHARDED_RUN_SCHEMA_VERSION = "sireto-retrieval-v12-sharded-oof-manifest-1"
V12_SHARD_SPEC_SCHEMA_VERSION = "sireto-retrieval-v12-shard-spec-1"
PHYSICAL_CONFIG_SCHEMA_VERSION = "sireto-retrieval-v13-physical-execution-1"
ALLOWED_FOLDS = (2, 3, 4)
CANONICAL_DATASET_KEYS = (
    "train_outer2",
    "train_outer3",
    "train_outer4",
    "heldout_spool",
    "prerank500_compact",
    "top100_full",
    "outcomes",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_EXTENT_SCHEMA = pa.schema(
    [
        pa.field("surface", pa.string(), nullable=False),
        pa.field("actor_id", pa.string(), nullable=False),
        pa.field("actor_ordinal", pa.int16(), nullable=False),
        pa.field("source_sha256", pa.string(), nullable=False),
        pa.field("projection_sha256", pa.string(), nullable=False),
        pa.field("source_row_group", pa.int32(), nullable=False),
        pa.field("row_start", pa.int64(), nullable=False),
        pa.field("row_stop", pa.int64(), nullable=False),
        pa.field("row_count", pa.int64(), nullable=False),
        pa.field("projected_chunk_count", pa.int32(), nullable=False),
        pa.field("projected_extent_start_offset", pa.int64(), nullable=False),
        pa.field("projected_extent_stop_offset", pa.int64(), nullable=False),
        pa.field("projected_compressed_bytes", pa.int64(), nullable=False),
    ]
)


class LogicalIOInvariantError(ValueError):
    """A sealed input or content-addressed output is incomplete or corrupt."""


@dataclass(frozen=True)
class ImmutableW1IOProjectionV13:
    manifest_path: Path
    manifest_sha256: str
    build_id: str
    projection_sha256: str
    unique_extent_count: int
    logical_compressed_bytes: int
    materialized_output_bytes: int


@dataclass(frozen=True)
class LogicalIOGateArtifactV13:
    manifest_path: Path
    manifest_sha256: str
    build_id: str
    status: str
    gate_pass: bool
    logical_io_ratio: float
    materialized_io_ratio: float


@dataclass(frozen=True)
class _Extent:
    ordinal: int
    row_start: int
    row_stop: int
    projected_chunk_count: int
    projected_extent_start_offset: int
    projected_extent_stop_offset: int
    projected_compressed_bytes: int

    @property
    def row_count(self) -> int:
        return self.row_stop - self.row_start

    def ledger_row(
        self,
        *,
        surface: str,
        actor_id: str,
        actor_ordinal: int,
        source_sha256: str,
        projection_sha256: str,
    ) -> dict[str, Any]:
        return {
            "surface": surface,
            "actor_id": actor_id,
            "actor_ordinal": actor_ordinal,
            "source_sha256": source_sha256,
            "projection_sha256": projection_sha256,
            "source_row_group": self.ordinal,
            "row_start": self.row_start,
            "row_stop": self.row_stop,
            "row_count": self.row_count,
            "projected_chunk_count": self.projected_chunk_count,
            "projected_extent_start_offset": self.projected_extent_start_offset,
            "projected_extent_stop_offset": self.projected_extent_stop_offset,
            "projected_compressed_bytes": self.projected_compressed_bytes,
        }


@dataclass(frozen=True)
class _Baseline:
    config_path: Path
    config_sha256: str
    config: Mapping[str, Any]
    benchmark_path: Path
    benchmark_sha256: str
    w1_manifest_path: Path
    w1_manifest_sha256: str
    plan_path: Path
    plan_sha256: str
    source_path: Path
    source_sha256: str
    source_size_bytes: int
    source_rows: int
    source_row_groups: int
    projection_columns: tuple[str, ...]
    projection_sha256: str
    source_metadata_sha256: str
    specs: tuple[Mapping[str, Any], ...]
    query_sequence_sha256: str
    range_union_sha256: str
    rows: tuple[dict[str, Any], ...]
    materialized_output_bytes: int
    expected_extent_count: int
    expected_logical_bytes: int
    maximum_logical_ratio: float
    materialized_ratio_numerator: int
    materialized_ratio_denominator: int


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _read_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LogicalIOInvariantError(f"{label} is not readable JSON") from error
    if not isinstance(value, Mapping):
        raise LogicalIOInvariantError(f"{label} must be a JSON object")
    return value


def _require_file(path: Path, expected_sha256: str, *, label: str) -> Path:
    resolved = Path(path).resolve()
    if not _valid_sha256(expected_sha256):
        raise LogicalIOInvariantError(f"{label} expected SHA-256 is invalid")
    if not resolved.is_file() or sha256_file(resolved) != expected_sha256:
        raise LogicalIOInvariantError(f"{label} differs from its pinned SHA-256")
    return resolved


def _resolve_from_workspace(config_path: Path, value: Any) -> Path:
    path = Path(str(value or ""))
    if path.is_absolute():
        return path.resolve()
    # The v13 config lives in <workspace>/config and its authority paths are
    # workspace-relative.
    return (config_path.parent.parent / path).resolve()


def _resolve_child(root: Path, value: Any, *, label: str) -> Path:
    raw = Path(str(value or ""))
    path = (root / raw).resolve() if not raw.is_absolute() else raw.resolve()
    if path == root or root not in path.parents or not path.is_file():
        raise LogicalIOInvariantError(f"{label} escaped or is absent from its root")
    return path


def _normalise_ranges(
    raw: Any, *, row_limit: int, label: str
) -> tuple[tuple[int, int], ...]:
    if not isinstance(raw, list) or not raw:
        raise LogicalIOInvariantError(f"{label} has no source ranges")
    output: list[tuple[int, int]] = []
    for position, value in enumerate(raw):
        if (
            not isinstance(value, list)
            or len(value) != 2
            or isinstance(value[0], bool)
            or isinstance(value[1], bool)
        ):
            raise LogicalIOInvariantError(f"{label} range {position} is malformed")
        start, stop = int(value[0]), int(value[1])
        if start < 0 or stop <= start or stop > row_limit:
            raise LogicalIOInvariantError(f"{label} range {position} is out of bounds")
        if output and start < output[-1][1]:
            raise LogicalIOInvariantError(f"{label} ranges overlap or are unordered")
        if output and start == output[-1][1]:
            output[-1] = (output[-1][0], stop)
        else:
            output.append((start, stop))
    return tuple(output)


def _union_ranges(
    values: Sequence[Sequence[tuple[int, int]]],
) -> tuple[tuple[tuple[int, int], ...], int]:
    tagged = sorted(
        (start, stop, owner)
        for owner, ranges in enumerate(values)
        for start, stop in ranges
    )
    if not tagged:
        return (), 0
    union: list[tuple[int, int]] = []
    overlap = 0
    start, stop, owner = tagged[0]
    for next_start, next_stop, next_owner in tagged[1:]:
        if next_start < stop:
            if next_owner != owner:
                overlap += min(stop, next_stop) - next_start
            stop = max(stop, next_stop)
            if next_owner != owner:
                owner = -1
        elif next_start == stop:
            stop = next_stop
            if next_owner != owner:
                owner = -1
        else:
            union.append((start, stop))
            start, stop, owner = next_start, next_stop, next_owner
    union.append((start, stop))
    return tuple(union), overlap


_SPEC_KEYS = {
    "schema_version",
    "shard_id",
    "shard_ordinal",
    "sequence_start",
    "sequence_stop",
    "query_ids",
    "query_folds",
    "source_ordinals",
    "source_ranges",
    "candidate_row_count",
    "query_sequence_sha256",
    "fold0_opened",
    "fold1_opened",
}


def _validate_spec(
    value: Any,
    *,
    expected_ordinal: int,
    row_limit: int,
    label: str,
) -> tuple[Mapping[str, Any], tuple[tuple[int, int], ...]]:
    if not isinstance(value, Mapping) or set(value) != _SPEC_KEYS:
        raise LogicalIOInvariantError(f"{label} is not an exact ShardSpecV12 record")
    query_ids = value.get("query_ids")
    query_folds = value.get("query_folds")
    source_ordinals = value.get("source_ordinals")
    start = int(value.get("sequence_start", -1))
    stop = int(value.get("sequence_stop", -1))
    if (
        value.get("schema_version") != V12_SHARD_SPEC_SCHEMA_VERSION
        or value.get("fold0_opened") is not False
        or value.get("fold1_opened") is not False
        or int(value.get("shard_ordinal", -1)) != expected_ordinal
        or not isinstance(value.get("shard_id"), str)
        or not value.get("shard_id")
        or not isinstance(query_ids, list)
        or not query_ids
        or any(not isinstance(item, str) or not item for item in query_ids)
        or len(set(query_ids)) != len(query_ids)
        or not isinstance(query_folds, list)
        or len(query_folds) != len(query_ids)
        or any(type(fold) is not int or fold not in ALLOWED_FOLDS for fold in query_folds)
        or not isinstance(source_ordinals, list)
        or len(source_ordinals) != len(query_ids)
        or any(type(item) is not int or item < 0 for item in source_ordinals)
        or any(source_ordinals[index] >= source_ordinals[index + 1] for index in range(len(source_ordinals) - 1))
        or start < 0
        or stop - start != len(query_ids)
        or value.get("query_sequence_sha256")
        != _sha256_json([[query_id, fold] for query_id, fold in zip(query_ids, query_folds, strict=True)])
    ):
        raise LogicalIOInvariantError(f"{label} safety, order, or sequence digest drifted")
    ranges = _normalise_ranges(value.get("source_ranges"), row_limit=row_limit, label=label)
    if int(value.get("candidate_row_count", -1)) != sum(
        range_stop - range_start for range_start, range_stop in ranges
    ):
        raise LogicalIOInvariantError(f"{label} candidate row count drifted")
    return value, ranges


def _column_chunk_start(column: Any) -> int:
    offsets = [
        int(value)
        for value in (
            getattr(column, "dictionary_page_offset", None),
            getattr(column, "data_page_offset", None),
        )
        if value is not None and int(value) >= 0
    ]
    if not offsets and int(getattr(column, "file_offset", 0)) > 0:
        offsets.append(int(column.file_offset))
    if not offsets:
        raise LogicalIOInvariantError("projected Parquet chunk has no usable offset")
    return min(offsets)


def _projected_extents(
    source_path: Path,
) -> tuple[tuple[_Extent, ...], tuple[str, ...], str, int, str]:
    try:
        parquet = pq.ParquetFile(source_path)
    except Exception as error:
        raise LogicalIOInvariantError("source candidate is not readable Parquet") from error
    available = tuple(parquet.schema_arrow.names)
    columns = tuple(production_projection(available))
    if not columns or len(columns) != len(set(columns)):
        raise LogicalIOInvariantError("production source projection is empty or duplicated")
    index_by_name = {name: index for index, name in enumerate(available)}
    if any(name not in index_by_name for name in columns):
        raise LogicalIOInvariantError("production projection contains an absent source column")
    projection_sha = _sha256_json(list(columns))
    extents: list[_Extent] = []
    row_start = 0
    metadata_identity: list[dict[str, Any]] = []
    for ordinal in range(parquet.num_row_groups):
        group = parquet.metadata.row_group(ordinal)
        row_stop = row_start + int(group.num_rows)
        starts: list[int] = []
        stops: list[int] = []
        compressed = 0
        for name in columns:
            column = group.column(index_by_name[name])
            size = int(column.total_compressed_size)
            if size <= 0:
                raise LogicalIOInvariantError("projected Parquet chunk has invalid size")
            chunk_start = _column_chunk_start(column)
            starts.append(chunk_start)
            stops.append(chunk_start + size)
            compressed += size
        if row_stop <= row_start:
            raise LogicalIOInvariantError("source candidate contains an empty row group")
        extent = _Extent(
            ordinal=ordinal,
            row_start=row_start,
            row_stop=row_stop,
            projected_chunk_count=len(columns),
            projected_extent_start_offset=min(starts),
            projected_extent_stop_offset=max(stops),
            projected_compressed_bytes=compressed,
        )
        extents.append(extent)
        metadata_identity.append(
            {
                "row_group": ordinal,
                "row_start": row_start,
                "row_stop": row_stop,
                "projection_sha256": projection_sha,
                "projected_compressed_bytes": compressed,
                "projected_extent_start_offset": extent.projected_extent_start_offset,
                "projected_extent_stop_offset": extent.projected_extent_stop_offset,
            }
        )
        row_start = row_stop
    if not extents:
        raise LogicalIOInvariantError("source candidate contains no row groups")
    return tuple(extents), columns, projection_sha, row_start, _sha256_json(metadata_identity)


def _touch(
    ranges: Sequence[tuple[int, int]], extents: Sequence[_Extent]
) -> tuple[_Extent, ...]:
    stops = [extent.row_stop for extent in extents]
    selected: set[int] = set()
    for start, stop in ranges:
        position = bisect.bisect_right(stops, start)
        while position < len(extents) and extents[position].row_start < stop:
            selected.add(position)
            position += 1
    return tuple(extents[position] for position in sorted(selected))


def _strict_config(path: Path, expected_sha256: str) -> tuple[Path, Mapping[str, Any]]:
    config_path = _require_file(path, expected_sha256, label="v13 physical config")
    config = _read_json(config_path, label="v13 physical config")
    if (
        config.get("schema_version") != PHYSICAL_CONFIG_SCHEMA_VERSION
        or config.get("status") != "PREREGISTERED_V13_PHYSICAL_BENCHMARK_NOT_RUN"
        or config.get("revision_scope") != "PHYSICAL_ONLY"
        or config.get("logical_recipe_changes") is not False
        or config.get("national_oof_allowed") is not False
    ):
        raise LogicalIOInvariantError("v13 physical config is not the preregistered authority")
    return config_path, config


def _baseline_from_config(
    physical_config_path: Path,
    physical_config_sha256: str,
    *,
    verify_source_hash: bool,
) -> _Baseline:
    config_path, config = _strict_config(physical_config_path, physical_config_sha256)
    frozen = config.get("immutable_v12_stop")
    authority = (config.get("authorities") or {}).get("v12_sharded_plan") or {}
    gate = config.get("cache_neutral_io_gate") or {}
    source_gate = gate.get("source_extent") or {}
    materialized_gate = gate.get("materialized") or {}
    if not isinstance(frozen, Mapping) or not isinstance(authority, Mapping):
        raise LogicalIOInvariantError("v13 config lacks the immutable v12 baseline")
    benchmark_sha = str(frozen.get("benchmark_manifest_sha256") or "")
    benchmark_path = _require_file(
        Path(str(frozen.get("benchmark_manifest_path") or "")),
        benchmark_sha,
        label="immutable v12 benchmark",
    )
    benchmark = _read_json(benchmark_path, label="immutable v12 benchmark")
    w1_pin = frozen.get("w1_trial") or {}
    trials = benchmark.get("trials") or {}
    w1_record = trials.get("1") or {}
    w1_manifest_record = w1_record.get("artifact_manifest") or {}
    w1_sha = str(w1_pin.get("manifest_sha256") or "")
    w1_path = _resolve_child(
        benchmark_path.parent,
        w1_pin.get("relative_manifest_path"),
        label="immutable W1 trial manifest",
    )
    if (
        benchmark.get("schema_version") != V12_BENCHMARK_SCHEMA_VERSION
        or frozen.get("verdict") != "STOP_NO_PARALLEL_WORKER_AUTHORIZED"
        or frozen.get("reinterpretation_forbidden") is not True
        or frozen.get("replacement_forbidden") is not True
        or frozen.get("execution_authorization_published") is not False
        or not isinstance(w1_record, Mapping)
        or w1_manifest_record.get("sha256") != w1_sha
        or _resolve_child(
            benchmark_path.parent,
            w1_manifest_record.get("path"),
            label="benchmark W1 trial manifest",
        )
        != w1_path
        or sha256_file(w1_path) != w1_sha
        or int(w1_record.get("materialized_io_bytes", -1))
        != int(w1_pin.get("materialized_bytes", -2))
        or int(w1_pin.get("query_count", -1)) != 512
        or w1_pin.get("folds") != list(ALLOWED_FOLDS)
        or w1_pin.get("fold0_opened") is not False
        or w1_pin.get("fold1_opened") is not False
        or w1_pin.get("positive_injection") is not False
        or dict(w1_record.get("digests") or {}) != dict(w1_pin.get("digests") or {})
    ):
        raise LogicalIOInvariantError("immutable W1 pin differs from the v12 benchmark")
    w1 = _read_json(w1_path, label="immutable W1 trial manifest")
    specs_raw = w1.get("specs")
    if (
        w1.get("schema_version") != V12_SHARDED_RUN_SCHEMA_VERSION
        or w1.get("run_kind") != "PHYSICAL_BENCHMARK_MINIATURE"
        or w1.get("status") != "EXECUTED_NO_READY_CLAIM"
        or w1.get("worker_count") != 1
        or w1.get("shard_count") != 1
        or w1.get("fold0_opened") is not False
        or w1.get("fold1_opened") is not False
        or w1.get("positive_injection") is not False
        or not isinstance(specs_raw, list)
        or len(specs_raw) != 1
    ):
        raise LogicalIOInvariantError("immutable W1 run contract drifted")
    plan_record = benchmark.get("sharded_execution_plan") or {}
    plan_sha = str(authority.get("sha256") or "")
    plan_path = _require_file(
        _resolve_from_workspace(config_path, authority.get("path")),
        plan_sha,
        label="v12 sharded execution plan",
    )
    if (
        plan_record.get("sha256") != plan_sha
        or Path(str(plan_record.get("path") or "")).resolve() != plan_path
    ):
        raise LogicalIOInvariantError("v12 benchmark/plan binding drifted")
    plan = _read_json(plan_path, label="v12 sharded execution plan")
    scene = ((plan.get("input_lineage") or {}).get("scene_ir") or {})
    source_path = Path(str(scene.get("source_candidate_path") or "")).resolve()
    source_sha = str(scene.get("source_candidate_sha256") or "")
    source_size = int(scene.get("source_candidate_size_bytes", -1))
    expected_rows = int(scene.get("candidate_rows", -1))
    expected_groups = int(scene.get("source_row_groups", -1))
    if (
        not source_path.is_file()
        or not _valid_sha256(source_sha)
        or source_sha != (config.get("input_lineage") or {}).get("source_candidate_sha256")
        or source_path.stat().st_size != source_size
        or source_size <= 0
    ):
        raise LogicalIOInvariantError("source candidate pin drifted")
    if verify_source_hash and sha256_file(source_path) != source_sha:
        raise LogicalIOInvariantError("source candidate bytes drifted")
    extents, projection, projection_sha, source_rows, metadata_sha = _projected_extents(
        source_path
    )
    projection_pin = source_gate.get("production_projection") or {}
    if (
        not isinstance(projection_pin, Mapping)
        or projection_pin.get("derivation")
        != "retrieval_v12_scene_ir.production_projection(source_schema_names)"
        or projection_pin.get("canonical_encoding")
        != "JSON_UTF8_ENSURE_ASCII_FALSE_COMPACT_ARRAY_IN_SOURCE_ORDER"
        or int(projection_pin.get("source_schema_column_count", -1))
        != len(pq.ParquetFile(source_path).schema_arrow.names)
        or int(projection_pin.get("projected_column_count", -1)) != len(projection)
        or projection_pin.get("columns") != list(projection)
        or projection_pin.get("sha256") != projection_sha
        or projection_pin.get("drift_policy") != "STOP_V13_IO_BEFORE_W3"
    ):
        raise LogicalIOInvariantError("preregistered production projection drifted")
    if source_rows != expected_rows or len(extents) != expected_groups:
        raise LogicalIOInvariantError("source candidate Parquet metadata drifted")
    spec, ranges = _validate_spec(
        specs_raw[0], expected_ordinal=0, row_limit=source_rows, label="immutable W1"
    )
    touched = _touch(ranges, extents)
    rows = tuple(
        extent.ledger_row(
            surface="IMMUTABLE_V12_W1",
            actor_id="w1",
            actor_ordinal=0,
            source_sha256=source_sha,
            projection_sha256=projection_sha,
        )
        for extent in touched
    )
    expected_count = int(source_gate.get("sealed_sample_expected_unique_row_groups", -1))
    expected_bytes = int(source_gate.get("sealed_sample_expected_logical_compressed_bytes", -1))
    logical_bytes = sum(row["projected_compressed_bytes"] for row in rows)
    if len(rows) != expected_count or logical_bytes != expected_bytes:
        raise LogicalIOInvariantError(
            "immutable W1 logical extent identity differs from the preregistration"
        )
    logical_max = float(source_gate.get("logical_amplification_maximum", math.nan))
    numerator = int(materialized_gate.get("maximum_ratio_numerator", -1))
    denominator = int(materialized_gate.get("maximum_ratio_denominator", -1))
    baseline_bytes = int(materialized_gate.get("baseline_w1_bytes", -1))
    if (
        logical_max != 1.0
        or numerator != 115
        or denominator != 100
        or baseline_bytes != int(w1_pin.get("materialized_bytes", -2))
        or materialized_gate.get("comparison")
        != "W3_BYTES_TIMES_100_LE_W1_BYTES_TIMES_115"
    ):
        raise LogicalIOInvariantError("v13 cache-neutral I/O thresholds drifted")
    return _Baseline(
        config_path=config_path,
        config_sha256=physical_config_sha256,
        config=config,
        benchmark_path=benchmark_path,
        benchmark_sha256=benchmark_sha,
        w1_manifest_path=w1_path,
        w1_manifest_sha256=w1_sha,
        plan_path=plan_path,
        plan_sha256=plan_sha,
        source_path=source_path,
        source_sha256=source_sha,
        source_size_bytes=source_size,
        source_rows=source_rows,
        source_row_groups=len(extents),
        projection_columns=projection,
        projection_sha256=projection_sha,
        source_metadata_sha256=metadata_sha,
        specs=(spec,),
        query_sequence_sha256=str(spec["query_sequence_sha256"]),
        range_union_sha256=_sha256_json(list(ranges)),
        rows=rows,
        materialized_output_bytes=baseline_bytes,
        expected_extent_count=expected_count,
        expected_logical_bytes=expected_bytes,
        maximum_logical_ratio=logical_max,
        materialized_ratio_numerator=numerator,
        materialized_ratio_denominator=denominator,
    )


def _schema_descriptor(schema: pa.Schema) -> list[dict[str, Any]]:
    return [
        {"name": field.name, "type": str(field.type), "nullable": bool(field.nullable)}
        for field in schema
    ]


def _atomic_content_addressed(
    *,
    output_root: Path,
    rows: Sequence[Mapping[str, Any]],
    manifest_without_artifact: Mapping[str, Any],
) -> tuple[Path, str, str]:
    root = Path(output_root).resolve()
    if root.exists() and not root.is_dir():
        raise LogicalIOInvariantError("content-address output root is not a directory")
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".logical-io-v13-", dir=root))
    promoted = False
    try:
        artifact_path = temporary / "logical_extents.parquet"
        table = pa.Table.from_pylist(list(rows), schema=_EXTENT_SCHEMA)
        pq.write_table(
            table,
            artifact_path,
            compression="zstd",
            use_dictionary=False,
            write_statistics=True,
            row_group_size=max(1, min(65_536, table.num_rows)),
        )
        artifact_record = {
            "path": "logical_extents.parquet",
            "sha256": sha256_file(artifact_path),
            "size_bytes": artifact_path.stat().st_size,
            "rows": table.num_rows,
            "schema": _schema_descriptor(_EXTENT_SCHEMA),
            "logical_rows_sha256": _sha256_json(table.to_pylist()),
            "sort_order": [
                "surface ASC",
                "actor_ordinal ASC",
                "source_row_group ASC",
            ],
        }
        manifest = {**manifest_without_artifact, "artifact": artifact_record}
        identity = manifest.get("build_identity")
        build_id = _sha256_json(identity)
        manifest["build_id"] = build_id
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        target = root / build_id[:16]
        if target.exists():
            if (
                not (target / "manifest.json").is_file()
                or not (target / "logical_extents.parquet").is_file()
                or (target / "manifest.json").read_bytes() != manifest_path.read_bytes()
                or sha256_file(target / "logical_extents.parquet")
                != artifact_record["sha256"]
            ):
                raise LogicalIOInvariantError("content-address collision")
        else:
            os.replace(temporary, target)
            promoted = True
        return target / "manifest.json", build_id, sha256_file(target / "manifest.json")
    finally:
        if not promoted and temporary.exists():
            shutil.rmtree(temporary)


def build_immutable_w1_io_projection_v13(
    *,
    physical_config_path: Path,
    physical_config_sha256: str,
    output_root: Path,
    verify_source_hash: bool = False,
) -> ImmutableW1IOProjectionV13:
    """Project pinned W1 metadata without executing W1 or warming source pages.

    The pre-W3 phase trusts the already content-addressed v12 source binding
    and reads only Parquet metadata.  The post-W3 comparison defaults to a
    complete source hash verification, after physical timing has stopped.
    """

    if verify_source_hash:
        raise LogicalIOInvariantError(
            "pre-W3 source rehash is forbidden because it would warm candidate pages"
        )
    baseline = _baseline_from_config(
        physical_config_path,
        physical_config_sha256,
        verify_source_hash=verify_source_hash,
    )
    logical_rows_sha = _sha256_json(list(baseline.rows))
    identity = {
        "schema_version": W1_PROJECTION_SCHEMA_VERSION,
        "physical_config_sha256": baseline.config_sha256,
        "v12_benchmark_sha256": baseline.benchmark_sha256,
        "v12_w1_manifest_sha256": baseline.w1_manifest_sha256,
        "v12_sharded_plan_sha256": baseline.plan_sha256,
        "source_sha256": baseline.source_sha256,
        "source_metadata_sha256": baseline.source_metadata_sha256,
        "projection_sha256": baseline.projection_sha256,
        "logical_extent_rows_sha256": logical_rows_sha,
    }
    manifest = {
        "schema_version": W1_PROJECTION_SCHEMA_VERSION,
        "status": "SEALED_IMMUTABLE_W1_LOGICAL_IO_PROJECTION_V13",
        "build_identity": identity,
        "folds": list(ALLOWED_FOLDS),
        "fold0_opened": False,
        "fold1_opened": False,
        "positive_injection": False,
        "w1_executed_by_v13": False,
        "v12_stop_reinterpreted": False,
        "source_bytes_rehashed_before_w3": False,
        "full_source_hash_required_post_w3": True,
        "inputs": {
            "physical_config": {"path": str(baseline.config_path), "sha256": baseline.config_sha256},
            "v12_benchmark": {"path": str(baseline.benchmark_path), "sha256": baseline.benchmark_sha256},
            "v12_w1_manifest": {"path": str(baseline.w1_manifest_path), "sha256": baseline.w1_manifest_sha256},
            "v12_sharded_plan": {"path": str(baseline.plan_path), "sha256": baseline.plan_sha256},
        },
        "source_candidate": {
            "path": str(baseline.source_path),
            "sha256": baseline.source_sha256,
            "size_bytes": baseline.source_size_bytes,
            "rows": baseline.source_rows,
            "row_groups": baseline.source_row_groups,
            "metadata_sha256": baseline.source_metadata_sha256,
        },
        "projection": {
            "columns": list(baseline.projection_columns),
            "column_count": len(baseline.projection_columns),
            "sha256": baseline.projection_sha256,
            "derivation": "production_projection(source_schema_order)",
        },
        "query_count": 512,
        "query_sequence_sha256": baseline.query_sequence_sha256,
        "source_range_union_sha256": baseline.range_union_sha256,
        "unique_extent_count": baseline.expected_extent_count,
        "logical_compressed_bytes": baseline.expected_logical_bytes,
        "materialized_output_bytes": baseline.materialized_output_bytes,
        "thresholds": {
            "maximum_logical_io_ratio": baseline.maximum_logical_ratio,
            "materialized_ratio_numerator": baseline.materialized_ratio_numerator,
            "materialized_ratio_denominator": baseline.materialized_ratio_denominator,
        },
    }
    path, build_id, manifest_sha = _atomic_content_addressed(
        output_root=output_root,
        rows=baseline.rows,
        manifest_without_artifact=manifest,
    )
    return ImmutableW1IOProjectionV13(
        manifest_path=path,
        manifest_sha256=manifest_sha,
        build_id=build_id,
        projection_sha256=baseline.projection_sha256,
        unique_extent_count=baseline.expected_extent_count,
        logical_compressed_bytes=baseline.expected_logical_bytes,
        materialized_output_bytes=baseline.materialized_output_bytes,
    )


def load_immutable_w1_io_projection_v13(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
    verify_source_hash: bool = False,
) -> tuple[ImmutableW1IOProjectionV13, Mapping[str, Any], tuple[dict[str, Any], ...], _Baseline]:
    """Reload the W1 projection without warming source data by default.

    This loader is part of the pre-W3 path.  The post-timing comparison calls
    it with ``verify_source_hash=True`` and therefore still performs the full
    source-byte verification before publishing a gate.
    """

    path = _require_file(
        manifest_path, expected_manifest_sha256, label="immutable W1 I/O projection"
    )
    raw = _read_json(path, label="immutable W1 I/O projection")
    identity = raw.get("build_identity")
    build_id = str(raw.get("build_id") or "")
    inputs = raw.get("inputs") or {}
    artifact_record = raw.get("artifact") or {}
    if (
        raw.get("schema_version") != W1_PROJECTION_SCHEMA_VERSION
        or raw.get("status") != "SEALED_IMMUTABLE_W1_LOGICAL_IO_PROJECTION_V13"
        or not isinstance(identity, Mapping)
        or not _valid_sha256(build_id)
        or build_id != _sha256_json(identity)
        or path.parent.name != build_id[:16]
        or raw.get("fold0_opened") is not False
        or raw.get("fold1_opened") is not False
        or raw.get("positive_injection") is not False
        or raw.get("w1_executed_by_v13") is not False
        or raw.get("v12_stop_reinterpreted") is not False
        or raw.get("source_bytes_rehashed_before_w3") is not False
        or raw.get("full_source_hash_required_post_w3") is not True
        or not isinstance(inputs, Mapping)
        or not isinstance(artifact_record, Mapping)
    ):
        raise LogicalIOInvariantError("immutable W1 projection content address drifted")
    config_record = inputs.get("physical_config") or {}
    baseline = _baseline_from_config(
        Path(str(config_record.get("path") or "")),
        str(config_record.get("sha256") or ""),
        verify_source_hash=verify_source_hash,
    )
    ledger_path = _resolve_child(path.parent, artifact_record.get("path"), label="W1 extent ledger")
    if (
        sha256_file(ledger_path) != artifact_record.get("sha256")
        or ledger_path.stat().st_size != int(artifact_record.get("size_bytes", -1))
    ):
        raise LogicalIOInvariantError("immutable W1 extent ledger bytes drifted")
    table = pq.read_table(ledger_path)
    rows = tuple(table.to_pylist())
    if (
        not table.schema.remove_metadata().equals(_EXTENT_SCHEMA)
        or table.num_rows != int(artifact_record.get("rows", -1))
        or _sha256_json(list(rows)) != artifact_record.get("logical_rows_sha256")
        or rows != baseline.rows
    ):
        raise LogicalIOInvariantError("immutable W1 extent ledger is not reproducible")
    projection = raw.get("projection") or {}
    expected_identity = {
        "schema_version": W1_PROJECTION_SCHEMA_VERSION,
        "physical_config_sha256": baseline.config_sha256,
        "v12_benchmark_sha256": baseline.benchmark_sha256,
        "v12_w1_manifest_sha256": baseline.w1_manifest_sha256,
        "v12_sharded_plan_sha256": baseline.plan_sha256,
        "source_sha256": baseline.source_sha256,
        "source_metadata_sha256": baseline.source_metadata_sha256,
        "projection_sha256": baseline.projection_sha256,
        "logical_extent_rows_sha256": _sha256_json(list(baseline.rows)),
    }
    if (
        dict(identity) != expected_identity
        or projection.get("columns") != list(baseline.projection_columns)
        or projection.get("sha256") != baseline.projection_sha256
        or raw.get("unique_extent_count") != baseline.expected_extent_count
        or raw.get("logical_compressed_bytes") != baseline.expected_logical_bytes
        or raw.get("materialized_output_bytes") != baseline.materialized_output_bytes
    ):
        raise LogicalIOInvariantError("immutable W1 projection manifest is not reproducible")
    return (
        ImmutableW1IOProjectionV13(
            manifest_path=path,
            manifest_sha256=expected_manifest_sha256,
            build_id=build_id,
            projection_sha256=baseline.projection_sha256,
            unique_extent_count=baseline.expected_extent_count,
            logical_compressed_bytes=baseline.expected_logical_bytes,
            materialized_output_bytes=baseline.materialized_output_bytes,
        ),
        raw,
        rows,
        baseline,
    )


_W3_KEYS = {
    "schema_version",
    "run_kind",
    "query_count",
    "worker_count",
    "fold0_opened",
    "fold1_opened",
    "positive_injection",
    "source_candidate",
    "source_specs",
    "canonical_datasets",
    "artifact_inventory",
    "materialized_output_bytes",
    "logical_digests",
    "no_merged_files",
}
_SOURCE_KEYS = {"path", "sha256", "projection_sha256"}
_DATASET_RECORD_KEYS = {
    "path",
    "sha256",
    "size_bytes",
    "merkle_root",
    "rows",
    "fragment_count",
}
_INVENTORY_KEYS = {"role", "family", "path", "sha256", "size_bytes"}


def _relative_inventory_path(root: Path, value: Any) -> tuple[str, Path]:
    raw = str(value or "")
    posix = Path(raw)
    if not raw or posix.is_absolute() or ".." in posix.parts or raw.startswith("/"):
        raise LogicalIOInvariantError("W3 inventory path is not a safe relative path")
    resolved = (root / posix).resolve()
    if root not in resolved.parents or not resolved.is_file() or resolved.is_symlink():
        raise LogicalIOInvariantError("W3 inventory path escaped, is missing, or is a symlink")
    return posix.as_posix(), resolved


def _validate_w3_tree(
    path: Path, raw: Mapping[str, Any]
) -> tuple[int, Mapping[str, Mapping[str, Any]]]:
    root = path.parent
    datasets = raw.get("canonical_datasets")
    inventory = raw.get("artifact_inventory")
    if not isinstance(datasets, Mapping) or set(datasets) != set(CANONICAL_DATASET_KEYS):
        raise LogicalIOInvariantError("W3 canonical dataset family set drifted")
    dataset_records: dict[str, Mapping[str, Any]] = {}
    for family in CANONICAL_DATASET_KEYS:
        record = datasets[family]
        if (
            not isinstance(record, Mapping)
            or set(record) != _DATASET_RECORD_KEYS
            or not _valid_sha256(record.get("sha256"))
            or not _valid_sha256(record.get("merkle_root"))
            or type(record.get("size_bytes")) is not int
            or int(record["size_bytes"]) <= 0
            or type(record.get("rows")) is not int
            or int(record["rows"]) <= 0
            or type(record.get("fragment_count")) is not int
            or int(record["fragment_count"]) <= 0
        ):
            raise LogicalIOInvariantError(f"W3 canonical dataset {family} record drifted")
        _, dataset_path = _relative_inventory_path(root, record.get("path"))
        if (
            dataset_path.stat().st_size != int(record["size_bytes"])
            or sha256_file(dataset_path) != record["sha256"]
        ):
            raise LogicalIOInvariantError(f"W3 canonical dataset {family} bytes drifted")
        dataset_records[family] = record
    if not isinstance(inventory, list) or not inventory:
        raise LogicalIOInvariantError("W3 artifact inventory is empty")
    inventory_by_path: dict[str, Mapping[str, Any]] = {}
    inode_sizes: dict[tuple[int, int], int] = {}
    root_stat = path.stat()
    root_inode = (int(root_stat.st_dev), int(root_stat.st_ino))
    for position, record in enumerate(inventory):
        if (
            not isinstance(record, Mapping)
            or set(record) != _INVENTORY_KEYS
            or not isinstance(record.get("role"), str)
            or not record.get("role")
            or not isinstance(record.get("family"), str)
            or not record.get("family")
            or not _valid_sha256(record.get("sha256"))
            or type(record.get("size_bytes")) is not int
            or int(record["size_bytes"]) <= 0
        ):
            raise LogicalIOInvariantError(f"W3 inventory record {position} drifted")
        relative, child = _relative_inventory_path(root, record.get("path"))
        if relative in inventory_by_path or child == path:
            raise LogicalIOInvariantError("W3 inventory duplicates a path or includes its root manifest")
        if any(part.startswith(("merged_", "canonical_merged_")) for part in Path(relative).parts):
            raise LogicalIOInvariantError("W3 inventory contains a forbidden merged artifact")
        if child.stat().st_size != int(record["size_bytes"]) or sha256_file(child) != record["sha256"]:
            raise LogicalIOInvariantError(f"W3 inventory bytes drifted for {relative}")
        stat = child.stat()
        inode = (int(stat.st_dev), int(stat.st_ino))
        if int(stat.st_nlink) != 1 or inode == root_inode or inode in inode_sizes:
            raise LogicalIOInvariantError("W3 tree contains a forbidden hardlink")
        inode_sizes[inode] = int(stat.st_size)
        inventory_by_path[relative] = record
    actual_children: set[str] = set()
    for child in root.rglob("*"):
        if child.is_symlink():
            raise LogicalIOInvariantError("W3 tree contains a symlink")
        if child.is_file() and child.resolve() != path:
            relative = child.resolve().relative_to(root).as_posix()
            actual_children.add(relative)
            if any(part.startswith(("merged_", "canonical_merged_")) for part in Path(relative).parts):
                raise LogicalIOInvariantError("W3 tree contains a forbidden merged artifact")
    if actual_children != set(inventory_by_path):
        raise LogicalIOInvariantError("W3 inventory is not exhaustive over the run tree")
    for family, record in dataset_records.items():
        relative = Path(str(record["path"])).as_posix()
        inventory_record = inventory_by_path.get(relative)
        if (
            inventory_record is None
            or inventory_record.get("family") != family
            or inventory_record.get("sha256") != record.get("sha256")
            or inventory_record.get("size_bytes") != record.get("size_bytes")
        ):
            raise LogicalIOInvariantError(f"W3 dataset {family} is not exactly inventory-bound")
    actual_total = path.stat().st_size + sum(inode_sizes.values())
    if int(raw.get("materialized_output_bytes", -1)) != actual_total:
        raise LogicalIOInvariantError("W3 materialized bytes differ from the complete tree")
    return actual_total, dataset_records


def _load_w3(
    path: Path,
    expected_sha256: str,
    *,
    baseline: _Baseline,
) -> tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...], tuple[tuple[tuple[int, int], ...], ...], int]:
    manifest_path = _require_file(path, expected_sha256, label="fresh v13 W3 manifest")
    raw = _read_json(manifest_path, label="fresh v13 W3 manifest")
    source = raw.get("source_candidate")
    specs_raw = raw.get("source_specs")
    digests = raw.get("logical_digests")
    if (
        set(raw) != _W3_KEYS
        or raw.get("schema_version") != W3_RUN_SCHEMA_VERSION
        or raw.get("run_kind") != "PHYSICAL_BENCHMARK_W3_V13"
        or raw.get("query_count") != 512
        or raw.get("worker_count") != 3
        or raw.get("fold0_opened") is not False
        or raw.get("fold1_opened") is not False
        or raw.get("positive_injection") is not False
        or raw.get("no_merged_files") is not True
        or not isinstance(source, Mapping)
        or set(source) != _SOURCE_KEYS
        or Path(str(source.get("path") or "")).resolve() != baseline.source_path
        or source.get("sha256") != baseline.source_sha256
        or source.get("projection_sha256") != baseline.projection_sha256
        or not isinstance(specs_raw, list)
        or len(specs_raw) != 3
        or not isinstance(digests, Mapping)
        or not digests
        or any(not isinstance(key, str) or not key or not _valid_sha256(value) for key, value in digests.items())
    ):
        raise LogicalIOInvariantError("fresh v13 W3 wrapper contract drifted")
    specs: list[Mapping[str, Any]] = []
    ranges: list[tuple[tuple[int, int], ...]] = []
    previous_stop = 0
    for ordinal, value in enumerate(specs_raw):
        spec, spec_ranges = _validate_spec(
            value,
            expected_ordinal=ordinal,
            row_limit=baseline.source_rows,
            label=f"fresh W3 actor {ordinal}",
        )
        if int(spec["sequence_start"]) != previous_stop:
            raise LogicalIOInvariantError("fresh W3 query shards are not contiguous")
        expected_fold = ALLOWED_FOLDS[ordinal]
        if (
            spec["shard_id"] != f"fold{expected_fold}"
            or any(fold != expected_fold for fold in spec["query_folds"])
        ):
            raise LogicalIOInvariantError(
                "fresh W3 actors are not the exact fold-owned shard sequence"
            )
        previous_stop = int(spec["sequence_stop"])
        specs.append(spec)
        ranges.append(spec_ranges)
    sequence = [
        [query_id, fold]
        for spec in specs
        for query_id, fold in zip(spec["query_ids"], spec["query_folds"], strict=True)
    ]
    if previous_stop != 512 or _sha256_json(sequence) != baseline.query_sequence_sha256:
        raise LogicalIOInvariantError("fresh W3 query population/order differs from immutable W1")
    baseline_spec = baseline.specs[0]
    source_ordinals = [item for spec in specs for item in spec["source_ordinals"]]
    if source_ordinals != list(baseline_spec["source_ordinals"]):
        raise LogicalIOInvariantError(
            "fresh W3 source ordinal sequence differs from immutable W1"
        )
    actual_total, _ = _validate_w3_tree(manifest_path, raw)
    return raw, tuple(specs), tuple(ranges), actual_total


def compare_w3_to_immutable_w1_io_v13(
    *,
    w1_projection_manifest_path: Path,
    w1_projection_manifest_sha256: str,
    w3_run_manifest_path: Path,
    w3_run_manifest_sha256: str,
    output_root: Path,
    verify_source_hash: bool = True,
) -> LogicalIOGateArtifactV13:
    """Seal the cache-neutral I/O verdict for one fresh W3-only run."""

    w1_artifact, w1_manifest, w1_rows, baseline = load_immutable_w1_io_projection_v13(
        w1_projection_manifest_path,
        expected_manifest_sha256=w1_projection_manifest_sha256,
        verify_source_hash=verify_source_hash,
    )
    w3_path = _require_file(w3_run_manifest_path, w3_run_manifest_sha256, label="fresh v13 W3 manifest")
    w3, specs, ranges_by_actor, materialized_w3 = _load_w3(
        w3_path, w3_run_manifest_sha256, baseline=baseline
    )
    extents, projection, projection_sha, rows, metadata_sha = _projected_extents(
        baseline.source_path
    )
    if (
        projection != baseline.projection_columns
        or projection_sha != baseline.projection_sha256
        or rows != baseline.source_rows
        or metadata_sha != baseline.source_metadata_sha256
    ):
        raise LogicalIOInvariantError("source projection changed after immutable W1 sealing")
    actor_groups: list[frozenset[int]] = []
    w3_rows: list[dict[str, Any]] = []
    per_actor: list[dict[str, Any]] = []
    for ordinal, (spec, ranges) in enumerate(zip(specs, ranges_by_actor, strict=True)):
        touched = _touch(ranges, extents)
        groups = frozenset(extent.ordinal for extent in touched)
        actor_groups.append(groups)
        logical_bytes = sum(extent.projected_compressed_bytes for extent in touched)
        per_actor.append(
            {
                "actor_id": str(spec["shard_id"]),
                "actor_ordinal": ordinal,
                "unique_extent_count": len(groups),
                "logical_compressed_bytes": logical_bytes,
                "extent_set_sha256": _sha256_json(sorted(groups)),
            }
        )
        w3_rows.extend(
            extent.ledger_row(
                surface="FRESH_V13_W3",
                actor_id=str(spec["shard_id"]),
                actor_ordinal=ordinal,
                source_sha256=baseline.source_sha256,
                projection_sha256=baseline.projection_sha256,
            )
            for extent in touched
        )
    w3_rows.sort(key=lambda row: (row["actor_ordinal"], row["source_row_group"]))
    w1_groups = frozenset(int(row["source_row_group"]) for row in w1_rows)
    w3_union = frozenset().union(*actor_groups)
    pairwise_overlap = sum(
        len(actor_groups[left] & actor_groups[right])
        for left in range(len(actor_groups))
        for right in range(left)
    )
    range_union, range_overlap_rows = _union_ranges(ranges_by_actor)
    logical_w1 = baseline.expected_logical_bytes
    logical_w3 = sum(int(row["projected_compressed_bytes"]) for row in w3_rows)
    reference_count = baseline.expected_extent_count
    w3_reference_count = sum(len(groups) for groups in actor_groups)
    logical_ratio = logical_w3 / logical_w1
    materialized_ratio = materialized_w3 / baseline.materialized_output_bytes
    checks = {
        "source_sha256_identical": True,
        "projection_columns_identical": True,
        "projection_sha256_identical": True,
        "query_population_and_order_identical": True,
        "source_range_union_identical": _sha256_json(list(range_union)) == baseline.range_union_sha256,
        "source_range_overlap_rows_zero": range_overlap_rows == 0,
        "source_extent_union_identical": w3_union == w1_groups,
        "source_extents_pairwise_disjoint": pairwise_overlap == 0,
        "unique_extent_count_identical": w3_reference_count == reference_count,
        "logical_compressed_bytes_identical": logical_w3 == logical_w1,
        "logical_amplification_within_maximum": logical_ratio <= baseline.maximum_logical_ratio,
        "materialized_ratio_within_maximum": (
            materialized_w3 * baseline.materialized_ratio_denominator
            <= baseline.materialized_output_bytes * baseline.materialized_ratio_numerator
        ),
    }
    failures = tuple(name for name, passed in checks.items() if not passed)
    status = "PASS_V13_IO" if not failures else "STOP_V13_IO"
    combined_rows = tuple(sorted(
        (*w1_rows, *w3_rows),
        key=lambda row: (row["surface"], row["actor_ordinal"], row["source_row_group"]),
    ))
    identity = {
        "schema_version": IO_GATE_SCHEMA_VERSION,
        "w1_projection_manifest_sha256": w1_projection_manifest_sha256,
        "w3_run_manifest_sha256": w3_run_manifest_sha256,
        "source_sha256": baseline.source_sha256,
        "projection_sha256": baseline.projection_sha256,
        "logical_extent_rows_sha256": _sha256_json(list(combined_rows)),
        "thresholds": {
            "maximum_logical_io_ratio": baseline.maximum_logical_ratio,
            "materialized_ratio_numerator": baseline.materialized_ratio_numerator,
            "materialized_ratio_denominator": baseline.materialized_ratio_denominator,
        },
    }
    manifest = {
        "schema_version": IO_GATE_SCHEMA_VERSION,
        "status": status,
        "gate_pass": not failures,
        "build_identity": identity,
        "folds": list(ALLOWED_FOLDS),
        "fold0_opened": False,
        "fold1_opened": False,
        "positive_injection": False,
        "v12_stop_reinterpreted": False,
        "process_or_machine_io_used_for_decision": False,
        "inputs": {
            "immutable_w1_projection": {
                "path": str(w1_artifact.manifest_path),
                "sha256": w1_projection_manifest_sha256,
            },
            "fresh_w3_run": {"path": str(w3_path), "sha256": w3_run_manifest_sha256},
            "physical_config_sha256": baseline.config_sha256,
        },
        "source": {
            "sha256": baseline.source_sha256,
            "projection_columns": list(baseline.projection_columns),
            "projection_sha256": baseline.projection_sha256,
        },
        "w1": {
            "unique_extent_count": reference_count,
            "logical_compressed_bytes": logical_w1,
            "materialized_output_bytes": baseline.materialized_output_bytes,
            "extent_set_sha256": _sha256_json(sorted(w1_groups)),
        },
        "w3": {
            "per_actor": per_actor,
            "per_actor_unique_extent_count_sum": w3_reference_count,
            "union_unique_extent_count": len(w3_union),
            "inter_actor_duplicate_extent_references": w3_reference_count - len(w3_union),
            "logical_compressed_bytes": logical_w3,
            "materialized_output_bytes": materialized_w3,
            "extent_set_sha256": _sha256_json(sorted(w3_union)),
            "logical_digests": dict(w3["logical_digests"]),
        },
        "checks": checks,
        "failure_reasons": list(failures),
        "ratios": {
            "logical_compressed_bytes_w3_over_w1": logical_ratio,
            "logical_extent_references_w3_over_w1": w3_reference_count / reference_count,
            "materialized_output_bytes_w3_over_w1": materialized_ratio,
        },
    }
    manifest_path, build_id, manifest_sha = _atomic_content_addressed(
        output_root=output_root,
        rows=combined_rows,
        manifest_without_artifact=manifest,
    )
    return LogicalIOGateArtifactV13(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha,
        build_id=build_id,
        status=status,
        gate_pass=not failures,
        logical_io_ratio=logical_ratio,
        materialized_io_ratio=materialized_ratio,
    )


def load_logical_io_gate_v13(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
    verify_source_hash: bool = True,
) -> LogicalIOGateArtifactV13:
    """Strictly reload and independently reproduce a final W1/W3 I/O gate."""

    path = _require_file(
        manifest_path, expected_manifest_sha256, label="v13 logical-I/O gate"
    )
    raw = _read_json(path, label="v13 logical-I/O gate")
    identity = raw.get("build_identity")
    build_id = str(raw.get("build_id") or "")
    inputs = raw.get("inputs")
    artifact_record = raw.get("artifact")
    if (
        raw.get("schema_version") != IO_GATE_SCHEMA_VERSION
        or raw.get("status") not in {"PASS_V13_IO", "STOP_V13_IO"}
        or not isinstance(identity, Mapping)
        or not _valid_sha256(build_id)
        or build_id != _sha256_json(identity)
        or path.parent.name != build_id[:16]
        or not isinstance(inputs, Mapping)
        or not isinstance(artifact_record, Mapping)
    ):
        raise LogicalIOInvariantError("v13 logical-I/O gate content address drifted")
    ledger_path = _resolve_child(path.parent, artifact_record.get("path"), label="final I/O extent ledger")
    if (
        sha256_file(ledger_path) != artifact_record.get("sha256")
        or ledger_path.stat().st_size != int(artifact_record.get("size_bytes", -1))
    ):
        raise LogicalIOInvariantError("final I/O extent ledger bytes drifted")
    table = pq.read_table(ledger_path)
    if (
        not table.schema.remove_metadata().equals(_EXTENT_SCHEMA)
        or table.num_rows != int(artifact_record.get("rows", -1))
        or _sha256_json(table.to_pylist()) != artifact_record.get("logical_rows_sha256")
    ):
        raise LogicalIOInvariantError("final I/O extent ledger content drifted")
    w1_record = inputs.get("immutable_w1_projection") or {}
    w3_record = inputs.get("fresh_w3_run") or {}
    if (
        not isinstance(w1_record, Mapping)
        or not isinstance(w3_record, Mapping)
        or not _valid_sha256(w1_record.get("sha256"))
        or not _valid_sha256(w3_record.get("sha256"))
    ):
        raise LogicalIOInvariantError("final I/O gate input records drifted")
    with tempfile.TemporaryDirectory(prefix="sireto-logical-io-v13-reload-") as temporary:
        reproduced = compare_w3_to_immutable_w1_io_v13(
            w1_projection_manifest_path=Path(str(w1_record.get("path") or "")),
            w1_projection_manifest_sha256=str(w1_record["sha256"]),
            w3_run_manifest_path=Path(str(w3_record.get("path") or "")),
            w3_run_manifest_sha256=str(w3_record["sha256"]),
            output_root=Path(temporary),
            verify_source_hash=verify_source_hash,
        )
        if (
            reproduced.manifest_path.read_bytes() != path.read_bytes()
            or sha256_file(reproduced.manifest_path.parent / "logical_extents.parquet")
            != artifact_record.get("sha256")
        ):
            raise LogicalIOInvariantError("v13 logical-I/O gate is not reproducible")
    return LogicalIOGateArtifactV13(
        manifest_path=path,
        manifest_sha256=expected_manifest_sha256,
        build_id=build_id,
        status=str(raw["status"]),
        gate_pass=bool(raw.get("gate_pass")),
        logical_io_ratio=float(
            (raw.get("ratios") or {}).get("logical_compressed_bytes_w3_over_w1")
        ),
        materialized_io_ratio=float(
            (raw.get("ratios") or {}).get("materialized_output_bytes_w3_over_w1")
        ),
    )


__all__ = [
    "CANONICAL_DATASET_KEYS",
    "IO_GATE_SCHEMA_VERSION",
    "ImmutableW1IOProjectionV13",
    "LogicalIOGateArtifactV13",
    "LogicalIOInvariantError",
    "W1_PROJECTION_SCHEMA_VERSION",
    "W3_RUN_SCHEMA_VERSION",
    "build_immutable_w1_io_projection_v13",
    "compare_w3_to_immutable_w1_io_v13",
    "load_immutable_w1_io_projection_v13",
    "load_logical_io_gate_v13",
]
