"""Sealed evidence of Parquet row groups actually decoded by v13 build actors.

The v13 logical-I/O gate previously reconstructed touched row groups from shard
ranges.  This module records the cache-miss callbacks emitted by
``SceneIRCache.iter_selected_scenes`` and independently reconciles those reads
with the immutable Parquet footer and shard specifications.  It never opens a
human label and never changes candidate generation or ranking.
"""

from __future__ import annotations

from dataclasses import dataclass
import bisect
import hashlib
import json
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
from .retrieval_v12_scene_ir import SceneIRRowGroupReadEvent


ROW_GROUP_LEDGER_SCHEMA_VERSION = (
    "sireto-retrieval-v13-actual-row-group-ledger-1"
)
ROW_GROUP_LEDGER_STATUS = "SEALED_ACTUAL_ROW_GROUP_READS"
ROW_GROUP_LEDGER_DIRNAME = "source_row_group_ledger"
ROW_GROUP_LEDGER_MANIFEST = "manifest.json"
ROW_GROUP_LEDGER_PARQUET = "actual_row_group_reads.parquet"
ROW_GROUP_LEDGER_RECORD_KEYS = frozenset(
    {
        "actor_id",
        "actor_ordinal",
        "manifest_path",
        "manifest_sha256",
        "manifest_size_bytes",
        "parquet_path",
        "parquet_sha256",
        "parquet_size_bytes",
        "shard_spec_sha256",
        "source_candidate_sha256",
        "source_manifest_sha256",
        "scene_ir_build_id",
        "projection_sha256",
        "sharded_execution_plan_sha256",
        "execution_lineage_sha256",
        "actual_cache_miss_count",
        "logical_compressed_bytes",
        "extent_set_sha256",
    }
)
ALLOWED_FOLDS = frozenset({2, 3, 4})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_LEDGER_SCHEMA = pa.schema(
    [
        pa.field("actor_id", pa.string(), nullable=False),
        pa.field("actor_ordinal", pa.int16(), nullable=False),
        pa.field("read_order", pa.int32(), nullable=False),
        pa.field("source_row_group", pa.int32(), nullable=False),
        pa.field("row_start", pa.int64(), nullable=False),
        pa.field("row_stop", pa.int64(), nullable=False),
        pa.field("row_count", pa.int64(), nullable=False),
        pa.field("source_row_group_count", pa.int32(), nullable=False),
        pa.field("projected_column_count", pa.int16(), nullable=False),
        pa.field("projection_sha256", pa.string(), nullable=False),
        pa.field("projected_compressed_bytes", pa.int64(), nullable=False),
        pa.field("source_candidate_sha256", pa.string(), nullable=False),
        pa.field("source_manifest_sha256", pa.string(), nullable=False),
        pa.field("scene_ir_build_id", pa.string(), nullable=False),
        pa.field("shard_spec_sha256", pa.string(), nullable=False),
        pa.field("query_sequence_sha256", pa.string(), nullable=False),
        pa.field("sharded_execution_plan_sha256", pa.string(), nullable=False),
        pa.field("execution_lineage_sha256", pa.string(), nullable=False),
    ]
)


class RowGroupLedgerInvariantErrorV13(ValueError):
    """Actual-read evidence is incomplete, forged, or inconsistent."""


@dataclass(frozen=True)
class ActualRowGroupLedgerArtifactV13:
    manifest_path: Path
    manifest_sha256: str
    parquet_path: Path
    parquet_sha256: str
    build_id: str
    actor_id: str
    actor_ordinal: int
    shard_spec_sha256: str
    source_candidate_sha256: str
    source_manifest_sha256: str
    scene_ir_build_id: str
    projection_sha256: str
    projected_columns: tuple[str, ...]
    sharded_execution_plan_sha256: str
    execution_lineage: Mapping[str, str]
    execution_lineage_sha256: str
    row_group_ordinals: tuple[int, ...]
    logical_compressed_bytes: int
    rows: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class ActualRowGroupLedgerSetV13:
    artifacts: tuple[ActualRowGroupLedgerArtifactV13, ...]
    union_row_group_ordinals: tuple[int, ...]
    unique_row_group_count: int
    logical_compressed_bytes: int
    cross_actor_overlap_count: int
    union_sha256: str
    per_actor: tuple[Mapping[str, Any], ...]


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _schema_descriptor(schema: pa.Schema) -> list[dict[str, Any]]:
    return [
        {
            "name": field.name,
            "type": str(field.type),
            "nullable": bool(field.nullable),
        }
        for field in schema
    ]


def _read_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RowGroupLedgerInvariantErrorV13(
            f"{label} is not readable JSON"
        ) from error
    if not isinstance(value, Mapping):
        raise RowGroupLedgerInvariantErrorV13(f"{label} must be an object")
    return value


def _normalise_spec(value: Any) -> Mapping[str, Any]:
    raw = value.as_manifest() if callable(getattr(value, "as_manifest", None)) else value
    if not isinstance(raw, Mapping):
        raise RowGroupLedgerInvariantErrorV13("row-group ledger needs a shard spec")
    required = {
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
    if set(raw) != required:
        raise RowGroupLedgerInvariantErrorV13("shard spec field set drifted")
    query_ids = tuple(str(item) for item in raw.get("query_ids") or ())
    query_folds = tuple(int(item) for item in raw.get("query_folds") or ())
    source_ordinals = tuple(int(item) for item in raw.get("source_ordinals") or ())
    ranges = tuple(
        (int(item[0]), int(item[1])) for item in raw.get("source_ranges") or ()
    )
    actor_id = str(raw.get("shard_id") or "")
    actor_ordinal = int(raw.get("shard_ordinal", -1))
    sequence_start = int(raw.get("sequence_start", -1))
    sequence_stop = int(raw.get("sequence_stop", -1))
    query_sequence_sha = str(raw.get("query_sequence_sha256") or "")
    candidate_rows = int(raw.get("candidate_row_count", -1))
    if (
        raw.get("fold0_opened") is not False
        or raw.get("fold1_opened") is not False
        or not actor_id
        or actor_ordinal < 0
        or sequence_start < 0
        or sequence_stop <= sequence_start
        or len(query_ids) != sequence_stop - sequence_start
        or len(query_ids) != len(query_folds)
        or len(query_ids) != len(source_ordinals)
        or len(set(query_ids)) != len(query_ids)
        or set(query_folds).issubset(ALLOWED_FOLDS) is False
        or len(set(query_folds)) != 1
        or any(
            source_ordinals[index] >= source_ordinals[index + 1]
            for index in range(len(source_ordinals) - 1)
        )
        or not ranges
        or any(start < 0 or stop <= start for start, stop in ranges)
        or any(ranges[index][1] > ranges[index + 1][0] for index in range(len(ranges) - 1))
        or candidate_rows != sum(stop - start for start, stop in ranges)
        or not _valid_sha256(query_sequence_sha)
        or query_sequence_sha
        != _sha256_json(
            [
                [query_id, fold]
                for query_id, fold in zip(query_ids, query_folds, strict=True)
            ]
        )
    ):
        raise RowGroupLedgerInvariantErrorV13(
            "shard spec order, fold, range, or digest drifted"
        )
    return dict(raw)


def _spec_summary(spec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": str(spec["schema_version"]),
        "sha256": _sha256_json(spec),
        "shard_id": str(spec["shard_id"]),
        "shard_ordinal": int(spec["shard_ordinal"]),
        "sequence_start": int(spec["sequence_start"]),
        "sequence_stop": int(spec["sequence_stop"]),
        "query_count": len(spec["query_ids"]),
        "query_sequence_sha256": str(spec["query_sequence_sha256"]),
        "source_ordinals_sha256": _sha256_json(spec["source_ordinals"]),
        "source_ranges_sha256": _sha256_json(spec["source_ranges"]),
        "candidate_row_count": int(spec["candidate_row_count"]),
    }


def _strict_lineage(value: Mapping[str, str]) -> dict[str, str]:
    lineage = {str(key): str(item) for key, item in value.items()}
    if (
        not lineage
        or any(not key or not _valid_sha256(item) for key, item in lineage.items())
        or not _valid_sha256(lineage.get("source_candidate_sha256"))
        or not _valid_sha256(lineage.get("source_manifest_sha256"))
        or not _valid_sha256(lineage.get("scene_ir_sha256"))
    ):
        raise RowGroupLedgerInvariantErrorV13(
            "actual-read ledger lineage is not fully SHA-256 sealed"
        )
    return dict(sorted(lineage.items()))


class ActualRowGroupLedgerSinkV13:
    """Callable observer that seals exactly the row groups a build actor read."""

    def __init__(
        self,
        *,
        shard_spec: Any,
        execution_lineage: Mapping[str, str],
        sharded_execution_plan_sha256: str,
        expected_projection_sha256: str,
        expected_projection_column_count: int,
    ) -> None:
        self.spec = _normalise_spec(shard_spec)
        self.spec_summary = _spec_summary(self.spec)
        self.lineage = _strict_lineage(execution_lineage)
        self.lineage_sha256 = _sha256_json(self.lineage)
        if not _valid_sha256(sharded_execution_plan_sha256):
            raise RowGroupLedgerInvariantErrorV13(
                "sharded plan lineage is not SHA-256 sealed"
            )
        if (
            not _valid_sha256(expected_projection_sha256)
            or expected_projection_column_count <= 0
        ):
            raise RowGroupLedgerInvariantErrorV13(
                "expected production projection is invalid"
            )
        self.sharded_execution_plan_sha256 = sharded_execution_plan_sha256
        self.expected_projection_sha256 = expected_projection_sha256
        self.expected_projection_column_count = expected_projection_column_count
        self._rows: list[dict[str, Any]] = []
        self._columns: tuple[str, ...] | None = None
        self._source_path: str | None = None
        self._sealed = False

    @property
    def actor_id(self) -> str:
        return str(self.spec_summary["shard_id"])

    @property
    def actor_ordinal(self) -> int:
        return int(self.spec_summary["shard_ordinal"])

    def __call__(self, event: SceneIRRowGroupReadEvent) -> None:
        if self._sealed:
            raise RowGroupLedgerInvariantErrorV13(
                "actual-read ledger received an event after sealing"
            )
        if not isinstance(event, SceneIRRowGroupReadEvent):
            raise RowGroupLedgerInvariantErrorV13(
                "actual-read observer received an unsupported event"
            )
        columns = tuple(str(value) for value in event.projected_columns)
        ordinal = int(event.source_row_group)
        previous = int(self._rows[-1]["source_row_group"]) if self._rows else -1
        if (
            not event.source_path
            or not _valid_sha256(event.source_candidate_sha256)
            or event.source_candidate_sha256
            != self.lineage["source_candidate_sha256"]
            or event.source_manifest_sha256
            != self.lineage["source_manifest_sha256"]
            or event.scene_ir_build_id != self.lineage["scene_ir_sha256"]
            or event.source_row_group_count <= 0
            or ordinal < 0
            or ordinal >= event.source_row_group_count
            or ordinal <= previous
            or event.row_start < 0
            or event.row_stop <= event.row_start
            or event.row_count != event.row_stop - event.row_start
            or len(columns) != len(set(columns))
            or len(columns) != self.expected_projection_column_count
            or _sha256_json(list(columns)) != event.projection_sha256
            or event.projection_sha256 != self.expected_projection_sha256
            or event.projected_compressed_bytes <= 0
        ):
            raise RowGroupLedgerInvariantErrorV13(
                "actual row-group read order, projection, or lineage drifted"
            )
        if self._columns is not None and columns != self._columns:
            raise RowGroupLedgerInvariantErrorV13(
                "production projection changed inside one build actor"
            )
        if self._source_path is not None and event.source_path != self._source_path:
            raise RowGroupLedgerInvariantErrorV13(
                "source path changed inside one build actor"
            )
        self._columns = columns
        self._source_path = event.source_path
        self._rows.append(
            {
                "actor_id": self.actor_id,
                "actor_ordinal": self.actor_ordinal,
                "read_order": len(self._rows),
                "source_row_group": ordinal,
                "row_start": int(event.row_start),
                "row_stop": int(event.row_stop),
                "row_count": int(event.row_count),
                "source_row_group_count": int(event.source_row_group_count),
                "projected_column_count": len(columns),
                "projection_sha256": event.projection_sha256,
                "projected_compressed_bytes": int(
                    event.projected_compressed_bytes
                ),
                "source_candidate_sha256": event.source_candidate_sha256,
                "source_manifest_sha256": event.source_manifest_sha256,
                "scene_ir_build_id": event.scene_ir_build_id,
                "shard_spec_sha256": str(self.spec_summary["sha256"]),
                "query_sequence_sha256": str(
                    self.spec_summary["query_sequence_sha256"]
                ),
                "sharded_execution_plan_sha256": (
                    self.sharded_execution_plan_sha256
                ),
                "execution_lineage_sha256": self.lineage_sha256,
            }
        )

    def seal(self, output_dir: Path) -> ActualRowGroupLedgerArtifactV13:
        if self._sealed or not self._rows or self._columns is None:
            raise RowGroupLedgerInvariantErrorV13(
                "actual-read ledger is empty or already sealed"
            )
        target = Path(output_dir).resolve()
        if target.exists():
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=".actual-row-groups-v13-", dir=target.parent)
        )
        promoted = False
        try:
            parquet_path = temporary / ROW_GROUP_LEDGER_PARQUET
            table = pa.Table.from_pylist(self._rows, schema=_LEDGER_SCHEMA)
            pq.write_table(
                table,
                parquet_path,
                compression="zstd",
                compression_level=3,
                use_dictionary=False,
                write_statistics=True,
                data_page_version="1.0",
                row_group_size=table.num_rows,
            )
            artifact = {
                "path": ROW_GROUP_LEDGER_PARQUET,
                "sha256": sha256_file(parquet_path),
                "size_bytes": parquet_path.stat().st_size,
                "rows": table.num_rows,
                "row_groups": pq.ParquetFile(parquet_path).num_row_groups,
                "schema": _schema_descriptor(_LEDGER_SCHEMA),
                "logical_rows_sha256": _sha256_json(table.to_pylist()),
            }
            ordinals = [int(row["source_row_group"]) for row in self._rows]
            summary = {
                "actual_cache_miss_count": len(self._rows),
                "unique_row_group_count": len(set(ordinals)),
                "logical_compressed_bytes": sum(
                    int(row["projected_compressed_bytes"]) for row in self._rows
                ),
                "read_order_sha256": _sha256_json(ordinals),
                "extent_set_sha256": _sha256_json(sorted(set(ordinals))),
                "first_source_row_group": ordinals[0],
                "last_source_row_group": ordinals[-1],
            }
            source = {
                "candidate_sha256": self.lineage["source_candidate_sha256"],
                "manifest_sha256": self.lineage["source_manifest_sha256"],
                "scene_ir_build_id": self.lineage["scene_ir_sha256"],
                "row_group_count": int(self._rows[0]["source_row_group_count"]),
            }
            projection = {
                "columns": list(self._columns),
                "column_count": len(self._columns),
                "sha256": self.expected_projection_sha256,
            }
            identity = {
                "schema_version": ROW_GROUP_LEDGER_SCHEMA_VERSION,
                "actor_id": self.actor_id,
                "actor_ordinal": self.actor_ordinal,
                "shard_spec_sha256": self.spec_summary["sha256"],
                "source_candidate_sha256": source["candidate_sha256"],
                "source_manifest_sha256": source["manifest_sha256"],
                "scene_ir_build_id": source["scene_ir_build_id"],
                "projection_sha256": projection["sha256"],
                "sharded_execution_plan_sha256": (
                    self.sharded_execution_plan_sha256
                ),
                "execution_lineage_sha256": self.lineage_sha256,
                "logical_rows_sha256": artifact["logical_rows_sha256"],
            }
            manifest = {
                "schema_version": ROW_GROUP_LEDGER_SCHEMA_VERSION,
                "status": ROW_GROUP_LEDGER_STATUS,
                "build_id": _sha256_json(identity),
                "build_identity": identity,
                "folds": sorted(set(int(value) for value in self.spec["query_folds"])),
                "fold0_opened": False,
                "fold1_opened": False,
                "positive_injection": False,
                "actor": {
                    "id": self.actor_id,
                    "ordinal": self.actor_ordinal,
                },
                "source": source,
                "projection": projection,
                "shard_spec": self.spec_summary,
                "execution_lineage": self.lineage,
                "sharded_execution_plan_sha256": (
                    self.sharded_execution_plan_sha256
                ),
                "read_summary": summary,
                "artifact": artifact,
            }
            manifest_path = temporary / ROW_GROUP_LEDGER_MANIFEST
            manifest_path.write_text(
                json.dumps(
                    manifest,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            os.rename(temporary, target)
            promoted = True
            self._sealed = True
            return load_actor_row_group_ledger_v13(
                target / ROW_GROUP_LEDGER_MANIFEST,
                expected_manifest_sha256=sha256_file(
                    target / ROW_GROUP_LEDGER_MANIFEST
                ),
            )
        finally:
            if not promoted and temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)


_MANIFEST_KEYS = {
    "schema_version",
    "status",
    "build_id",
    "build_identity",
    "folds",
    "fold0_opened",
    "fold1_opened",
    "positive_injection",
    "actor",
    "source",
    "projection",
    "shard_spec",
    "execution_lineage",
    "sharded_execution_plan_sha256",
    "read_summary",
    "artifact",
}


def load_actor_row_group_ledger_v13(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
) -> ActualRowGroupLedgerArtifactV13:
    path = Path(manifest_path).resolve()
    if (
        not path.is_file()
        or path.name != ROW_GROUP_LEDGER_MANIFEST
        or not _valid_sha256(expected_manifest_sha256)
        or sha256_file(path) != expected_manifest_sha256
        or path.is_symlink()
        or path.stat().st_nlink != 1
    ):
        raise RowGroupLedgerInvariantErrorV13(
            "actual-read ledger manifest bytes drifted"
        )
    raw = _read_json(path, label="actual row-group ledger")
    identity = raw.get("build_identity")
    actor = raw.get("actor")
    source = raw.get("source")
    projection = raw.get("projection")
    spec = raw.get("shard_spec")
    summary = raw.get("read_summary")
    artifact = raw.get("artifact")
    lineage_raw = raw.get("execution_lineage")
    if (
        set(raw) != _MANIFEST_KEYS
        or raw.get("schema_version") != ROW_GROUP_LEDGER_SCHEMA_VERSION
        or raw.get("status") != ROW_GROUP_LEDGER_STATUS
        or raw.get("fold0_opened") is not False
        or raw.get("fold1_opened") is not False
        or raw.get("positive_injection") is not False
        or not isinstance(raw.get("folds"), list)
        or not set(int(value) for value in raw["folds"]).issubset(ALLOWED_FOLDS)
        or not isinstance(identity, Mapping)
        or not isinstance(actor, Mapping)
        or set(actor) != {"id", "ordinal"}
        or not isinstance(source, Mapping)
        or set(source)
        != {
            "candidate_sha256",
            "manifest_sha256",
            "scene_ir_build_id",
            "row_group_count",
        }
        or not isinstance(projection, Mapping)
        or set(projection) != {"columns", "column_count", "sha256"}
        or not isinstance(spec, Mapping)
        or not isinstance(summary, Mapping)
        or not isinstance(artifact, Mapping)
        or not isinstance(lineage_raw, Mapping)
        or raw.get("build_id") != _sha256_json(identity)
    ):
        raise RowGroupLedgerInvariantErrorV13(
            "actual-read ledger manifest contract drifted"
        )
    lineage = _strict_lineage(
        {str(key): str(value) for key, value in lineage_raw.items()}
    )
    columns = tuple(str(value) for value in projection.get("columns") or ())
    actor_id = str(actor.get("id") or "")
    actor_ordinal = int(actor.get("ordinal", -1))
    plan_sha = str(raw.get("sharded_execution_plan_sha256") or "")
    expected_identity = {
        "schema_version": ROW_GROUP_LEDGER_SCHEMA_VERSION,
        "actor_id": actor_id,
        "actor_ordinal": actor_ordinal,
        "shard_spec_sha256": str(spec.get("sha256") or ""),
        "source_candidate_sha256": str(source.get("candidate_sha256") or ""),
        "source_manifest_sha256": str(source.get("manifest_sha256") or ""),
        "scene_ir_build_id": str(source.get("scene_ir_build_id") or ""),
        "projection_sha256": str(projection.get("sha256") or ""),
        "sharded_execution_plan_sha256": plan_sha,
        "execution_lineage_sha256": _sha256_json(lineage),
        "logical_rows_sha256": str(artifact.get("logical_rows_sha256") or ""),
    }
    if (
        dict(identity) != expected_identity
        or actor_id != str(spec.get("shard_id") or "")
        or actor_ordinal != int(spec.get("shard_ordinal", -1))
        or actor_ordinal < 0
        or not _valid_sha256(spec.get("sha256"))
        or not _valid_sha256(spec.get("query_sequence_sha256"))
        or not _valid_sha256(source.get("candidate_sha256"))
        or source.get("candidate_sha256") != lineage["source_candidate_sha256"]
        or source.get("manifest_sha256") != lineage["source_manifest_sha256"]
        or source.get("scene_ir_build_id") != lineage["scene_ir_sha256"]
        or int(source.get("row_group_count", -1)) <= 0
        or not columns
        or len(columns) != len(set(columns))
        or int(projection.get("column_count", -1)) != len(columns)
        or projection.get("sha256") != _sha256_json(list(columns))
        or not _valid_sha256(plan_sha)
        or set(artifact)
        != {
            "path",
            "sha256",
            "size_bytes",
            "rows",
            "row_groups",
            "schema",
            "logical_rows_sha256",
        }
        or artifact.get("path") != ROW_GROUP_LEDGER_PARQUET
        or not _valid_sha256(artifact.get("sha256"))
        or artifact.get("schema") != _schema_descriptor(_LEDGER_SCHEMA)
    ):
        raise RowGroupLedgerInvariantErrorV13(
            "actual-read ledger identity or lineage drifted"
        )
    parquet_path = (path.parent / ROW_GROUP_LEDGER_PARQUET).resolve()
    if (
        parquet_path.parent != path.parent
        or not parquet_path.is_file()
        or parquet_path.is_symlink()
        or parquet_path.stat().st_nlink != 1
        or parquet_path.stat().st_size != int(artifact.get("size_bytes", -1))
        or sha256_file(parquet_path) != artifact.get("sha256")
        or {child.name for child in path.parent.iterdir()}
        != {ROW_GROUP_LEDGER_MANIFEST, ROW_GROUP_LEDGER_PARQUET}
    ):
        raise RowGroupLedgerInvariantErrorV13(
            "actual-read ledger artifact tree drifted"
        )
    parquet = pq.ParquetFile(parquet_path)
    table = parquet.read()
    rows = tuple(table.to_pylist())
    ordinals = tuple(int(row["source_row_group"]) for row in rows)
    common = {
        "actor_id": actor_id,
        "actor_ordinal": actor_ordinal,
        "source_row_group_count": int(source["row_group_count"]),
        "projected_column_count": len(columns),
        "projection_sha256": str(projection["sha256"]),
        "source_candidate_sha256": str(source["candidate_sha256"]),
        "source_manifest_sha256": str(source["manifest_sha256"]),
        "scene_ir_build_id": str(source["scene_ir_build_id"]),
        "shard_spec_sha256": str(spec["sha256"]),
        "query_sequence_sha256": str(spec["query_sequence_sha256"]),
        "sharded_execution_plan_sha256": plan_sha,
        "execution_lineage_sha256": _sha256_json(lineage),
    }
    if (
        table.schema != _LEDGER_SCHEMA
        or not rows
        or parquet.num_row_groups != int(artifact.get("row_groups", -1))
        or table.num_rows != int(artifact.get("rows", -1))
        or artifact.get("logical_rows_sha256") != _sha256_json(list(rows))
        or tuple(int(row["read_order"]) for row in rows)
        != tuple(range(len(rows)))
        or len(set(ordinals)) != len(ordinals)
        or tuple(sorted(ordinals)) != ordinals
        or any(
            any(row.get(key) != value for key, value in common.items())
            or int(row["source_row_group"]) < 0
            or int(row["source_row_group"]) >= int(source["row_group_count"])
            or int(row["row_start"]) < 0
            or int(row["row_stop"]) <= int(row["row_start"])
            or int(row["row_count"])
            != int(row["row_stop"]) - int(row["row_start"])
            or int(row["projected_compressed_bytes"]) <= 0
            for row in rows
        )
        or set(summary)
        != {
            "actual_cache_miss_count",
            "unique_row_group_count",
            "logical_compressed_bytes",
            "read_order_sha256",
            "extent_set_sha256",
            "first_source_row_group",
            "last_source_row_group",
        }
        or int(summary.get("actual_cache_miss_count", -1)) != len(rows)
        or int(summary.get("unique_row_group_count", -1)) != len(rows)
        or int(summary.get("logical_compressed_bytes", -1))
        != sum(int(row["projected_compressed_bytes"]) for row in rows)
        or summary.get("read_order_sha256") != _sha256_json(list(ordinals))
        or summary.get("extent_set_sha256")
        != _sha256_json(sorted(set(ordinals)))
        or int(summary.get("first_source_row_group", -1)) != ordinals[0]
        or int(summary.get("last_source_row_group", -1)) != ordinals[-1]
    ):
        raise RowGroupLedgerInvariantErrorV13(
            "actual-read ledger rows, order, or summary drifted"
        )
    return ActualRowGroupLedgerArtifactV13(
        manifest_path=path,
        manifest_sha256=expected_manifest_sha256,
        parquet_path=parquet_path,
        parquet_sha256=str(artifact["sha256"]),
        build_id=str(raw["build_id"]),
        actor_id=actor_id,
        actor_ordinal=actor_ordinal,
        shard_spec_sha256=str(spec["sha256"]),
        source_candidate_sha256=str(source["candidate_sha256"]),
        source_manifest_sha256=str(source["manifest_sha256"]),
        scene_ir_build_id=str(source["scene_ir_build_id"]),
        projection_sha256=str(projection["sha256"]),
        projected_columns=columns,
        sharded_execution_plan_sha256=plan_sha,
        execution_lineage=lineage,
        execution_lineage_sha256=_sha256_json(lineage),
        row_group_ordinals=ordinals,
        logical_compressed_bytes=int(summary["logical_compressed_bytes"]),
        rows=rows,
    )


def _touched_row_groups(
    source_ranges: Sequence[Sequence[int]],
    *,
    row_starts: Sequence[int],
    row_stops: Sequence[int],
) -> tuple[int, ...]:
    selected: set[int] = set()
    for raw in source_ranges:
        if len(raw) != 2:
            raise RowGroupLedgerInvariantErrorV13("source range is malformed")
        start, stop = int(raw[0]), int(raw[1])
        if start < 0 or stop <= start or stop > row_stops[-1]:
            raise RowGroupLedgerInvariantErrorV13("source range escaped Parquet")
        position = bisect.bisect_right(row_stops, start)
        while position < len(row_starts) and row_starts[position] < stop:
            selected.add(position)
            position += 1
    return tuple(sorted(selected))


def validate_actual_row_group_ledger_set_v13(
    ledger_manifests: Sequence[tuple[Path, str]],
    *,
    expected_specs: Sequence[Any],
    source_path: Path,
    expected_source_candidate_sha256: str,
    expected_projection_columns: Sequence[str],
    expected_projection_sha256: str,
    expected_sharded_execution_plan_sha256: str,
    expected_execution_lineage: Mapping[str, str],
    expected_unique_row_groups: int,
    expected_logical_compressed_bytes: int,
    verify_source_hash: bool = False,
    enforce_gate_totals: bool = True,
) -> ActualRowGroupLedgerSetV13:
    """Reconcile three observed actor ledgers with the immutable source footer."""

    if len(ledger_manifests) != 3 or len(expected_specs) != 3:
        raise RowGroupLedgerInvariantErrorV13(
            "v13 requires exactly three fold-owned actual-read ledgers"
        )
    path = Path(source_path).resolve()
    if (
        not path.is_file()
        or not _valid_sha256(expected_source_candidate_sha256)
        or not _valid_sha256(expected_projection_sha256)
        or not _valid_sha256(expected_sharded_execution_plan_sha256)
        or expected_unique_row_groups <= 0
        or expected_logical_compressed_bytes <= 0
    ):
        raise RowGroupLedgerInvariantErrorV13(
            "actual-read ledger set authority is invalid"
        )
    if verify_source_hash and sha256_file(path) != expected_source_candidate_sha256:
        raise RowGroupLedgerInvariantErrorV13("source candidate bytes drifted")
    lineage = _strict_lineage(expected_execution_lineage)
    if lineage["source_candidate_sha256"] != expected_source_candidate_sha256:
        raise RowGroupLedgerInvariantErrorV13("source candidate lineage drifted")
    columns = tuple(str(value) for value in expected_projection_columns)
    if (
        not columns
        or len(columns) != len(set(columns))
        or _sha256_json(list(columns)) != expected_projection_sha256
    ):
        raise RowGroupLedgerInvariantErrorV13("production projection pin drifted")
    parquet = pq.ParquetFile(path)
    if tuple(production_projection(parquet.schema_arrow.names)) != columns:
        raise RowGroupLedgerInvariantErrorV13(
            "source production projection differs from the frozen projection"
        )
    source_indexes = {
        name: ordinal for ordinal, name in enumerate(parquet.schema_arrow.names)
    }
    starts: list[int] = []
    stops: list[int] = []
    compressed: list[int] = []
    row_start = 0
    for ordinal in range(parquet.num_row_groups):
        metadata = parquet.metadata.row_group(ordinal)
        row_count = int(metadata.num_rows)
        if row_count <= 0:
            raise RowGroupLedgerInvariantErrorV13(
                "source contains an empty Parquet row group"
            )
        starts.append(row_start)
        row_start += row_count
        stops.append(row_start)
        compressed.append(
            sum(
                int(
                    metadata.column(source_indexes[name]).total_compressed_size
                )
                for name in columns
            )
        )
    specs = tuple(_normalise_spec(value) for value in expected_specs)
    artifacts = tuple(
        load_actor_row_group_ledger_v13(
            manifest_path,
            expected_manifest_sha256=manifest_sha256,
        )
        for manifest_path, manifest_sha256 in ledger_manifests
    )
    if tuple(artifact.actor_ordinal for artifact in artifacts) != (0, 1, 2):
        raise RowGroupLedgerInvariantErrorV13(
            "actual-read ledger actor order drifted"
        )
    actor_sets: list[frozenset[int]] = []
    per_actor: list[Mapping[str, Any]] = []
    for ordinal, (artifact, spec) in enumerate(
        zip(artifacts, specs, strict=True)
    ):
        expected_groups = _touched_row_groups(
            spec["source_ranges"], row_starts=starts, row_stops=stops
        )
        if (
            artifact.actor_id != str(spec["shard_id"])
            or artifact.actor_ordinal != ordinal
            or artifact.shard_spec_sha256 != _sha256_json(spec)
            or artifact.source_candidate_sha256
            != expected_source_candidate_sha256
            or artifact.projection_sha256 != expected_projection_sha256
            or artifact.projected_columns != columns
            or (
                enforce_gate_totals
                and artifact.row_group_ordinals != expected_groups
            )
        ):
            raise RowGroupLedgerInvariantErrorV13(
                f"actor {ordinal} actual row groups differ from its sealed shard"
            )
        observed_groups = artifact.row_group_ordinals
        expected_rows = [
            {
                "source_row_group": group,
                "row_start": starts[group],
                "row_stop": stops[group],
                "row_count": stops[group] - starts[group],
                "projected_compressed_bytes": compressed[group],
            }
            for group in observed_groups
        ]
        for observed, expected in zip(artifact.rows, expected_rows, strict=True):
            if any(observed.get(key) != value for key, value in expected.items()):
                raise RowGroupLedgerInvariantErrorV13(
                    f"actor {ordinal} row-group footer evidence drifted"
                )
            if (
                observed.get("source_row_group_count") != parquet.num_row_groups
                or observed.get("sharded_execution_plan_sha256")
                != expected_sharded_execution_plan_sha256
                or observed.get("execution_lineage_sha256")
                != _sha256_json(lineage)
            ):
                raise RowGroupLedgerInvariantErrorV13(
                    f"actor {ordinal} source or execution lineage drifted"
                )
        groups = frozenset(observed_groups)
        actor_sets.append(groups)
        per_actor.append(
            {
                "actor_id": artifact.actor_id,
                "actor_ordinal": ordinal,
                "manifest_path": str(artifact.manifest_path),
                "manifest_sha256": artifact.manifest_sha256,
                "parquet_sha256": artifact.parquet_sha256,
                "shard_spec_sha256": artifact.shard_spec_sha256,
                "actual_cache_miss_count": len(observed_groups),
                "logical_compressed_bytes": artifact.logical_compressed_bytes,
                "extent_set_sha256": _sha256_json(list(observed_groups)),
            }
        )
    overlap_count = 0
    for left in range(len(actor_sets)):
        for right in range(left + 1, len(actor_sets)):
            overlap = len(actor_sets[left] & actor_sets[right])
            overlap_count += overlap
            if enforce_gate_totals and overlap:
                raise RowGroupLedgerInvariantErrorV13(
                    "actual build actors decoded overlapping row groups"
                )
    union = tuple(sorted(set().union(*actor_sets)))
    logical_bytes = sum(compressed[ordinal] for ordinal in union)
    if enforce_gate_totals and (
        len(union) != expected_unique_row_groups
        or logical_bytes != expected_logical_compressed_bytes
        or sum(artifact.logical_compressed_bytes for artifact in artifacts)
        != expected_logical_compressed_bytes
    ):
        raise RowGroupLedgerInvariantErrorV13(
            "actual row-group union count or compressed-byte total drifted"
        )
    return ActualRowGroupLedgerSetV13(
        artifacts=artifacts,
        union_row_group_ordinals=union,
        unique_row_group_count=len(union),
        logical_compressed_bytes=logical_bytes,
        cross_actor_overlap_count=overlap_count,
        union_sha256=_sha256_json(list(union)),
        per_actor=tuple(per_actor),
    )


def actor_row_group_ledger_record_v13(
    artifact: ActualRowGroupLedgerArtifactV13,
    *,
    root: Path,
) -> dict[str, Any]:
    """Return the strict wrapper/national record for one inventoried ledger."""

    base = Path(root).resolve()
    manifest = artifact.manifest_path.resolve()
    parquet = artifact.parquet_path.resolve()
    if base not in manifest.parents or base not in parquet.parents:
        raise RowGroupLedgerInvariantErrorV13(
            "row-group ledger escaped its run root"
        )
    return {
        "actor_id": artifact.actor_id,
        "actor_ordinal": artifact.actor_ordinal,
        "manifest_path": manifest.relative_to(base).as_posix(),
        "manifest_sha256": artifact.manifest_sha256,
        "manifest_size_bytes": manifest.stat().st_size,
        "parquet_path": parquet.relative_to(base).as_posix(),
        "parquet_sha256": artifact.parquet_sha256,
        "parquet_size_bytes": parquet.stat().st_size,
        "shard_spec_sha256": artifact.shard_spec_sha256,
        "source_candidate_sha256": artifact.source_candidate_sha256,
        "source_manifest_sha256": artifact.source_manifest_sha256,
        "scene_ir_build_id": artifact.scene_ir_build_id,
        "projection_sha256": artifact.projection_sha256,
        "sharded_execution_plan_sha256": (
            artifact.sharded_execution_plan_sha256
        ),
        "execution_lineage_sha256": artifact.execution_lineage_sha256,
        "actual_cache_miss_count": len(artifact.row_group_ordinals),
        "logical_compressed_bytes": artifact.logical_compressed_bytes,
        "extent_set_sha256": _sha256_json(list(artifact.row_group_ordinals)),
    }


def load_actor_row_group_ledger_records_v13(
    records: Any,
    *,
    root: Path,
) -> tuple[ActualRowGroupLedgerArtifactV13, ...]:
    """Resolve the exact three wrapper records without trusting their paths."""

    base = Path(root).resolve()
    if not isinstance(records, list) or len(records) != 3:
        raise RowGroupLedgerInvariantErrorV13(
            "v13 wrapper requires three actual-read ledger records"
        )
    artifacts: list[ActualRowGroupLedgerArtifactV13] = []
    for ordinal, record in enumerate(records):
        if (
            not isinstance(record, Mapping)
            or set(record) != set(ROW_GROUP_LEDGER_RECORD_KEYS)
            or record.get("actor_ordinal") != ordinal
        ):
            raise RowGroupLedgerInvariantErrorV13(
                f"actual-read ledger record {ordinal} drifted"
            )
        manifest_relative = Path(str(record.get("manifest_path") or ""))
        parquet_relative = Path(str(record.get("parquet_path") or ""))
        if (
            manifest_relative.is_absolute()
            or parquet_relative.is_absolute()
            or ".." in manifest_relative.parts
            or ".." in parquet_relative.parts
        ):
            raise RowGroupLedgerInvariantErrorV13(
                "actual-read ledger record escaped its run root"
            )
        manifest = (base / manifest_relative).resolve()
        parquet = (base / parquet_relative).resolve()
        if (
            base not in manifest.parents
            or base not in parquet.parents
            or not manifest.is_file()
            or not parquet.is_file()
            or manifest.is_symlink()
            or parquet.is_symlink()
            or manifest.stat().st_size
            != int(record.get("manifest_size_bytes", -1))
            or parquet.stat().st_size
            != int(record.get("parquet_size_bytes", -1))
            or sha256_file(manifest) != record.get("manifest_sha256")
            or sha256_file(parquet) != record.get("parquet_sha256")
        ):
            raise RowGroupLedgerInvariantErrorV13(
                "actual-read ledger wrapper bytes drifted"
            )
        artifact = load_actor_row_group_ledger_v13(
            manifest,
            expected_manifest_sha256=str(record["manifest_sha256"]),
        )
        expected = actor_row_group_ledger_record_v13(artifact, root=base)
        if dict(record) != expected:
            raise RowGroupLedgerInvariantErrorV13(
                "actual-read ledger wrapper record is not reproducible"
            )
        artifacts.append(artifact)
    return tuple(artifacts)


__all__ = [
    "ActualRowGroupLedgerArtifactV13",
    "ActualRowGroupLedgerSetV13",
    "ActualRowGroupLedgerSinkV13",
    "ROW_GROUP_LEDGER_DIRNAME",
    "ROW_GROUP_LEDGER_MANIFEST",
    "ROW_GROUP_LEDGER_PARQUET",
    "ROW_GROUP_LEDGER_RECORD_KEYS",
    "ROW_GROUP_LEDGER_SCHEMA_VERSION",
    "ROW_GROUP_LEDGER_STATUS",
    "RowGroupLedgerInvariantErrorV13",
    "actor_row_group_ledger_record_v13",
    "load_actor_row_group_ledger_v13",
    "load_actor_row_group_ledger_records_v13",
    "validate_actual_row_group_ledger_set_v13",
]
