"""Null-canonical logical Arrow table hashing for retrieval v14.

Arrow validity bitmaps make bytes stored underneath a null value semantically
irrelevant.  Parquet readers are therefore free to expose different masked
bytes for the same logical table.  Arrow IPC nevertheless serialises those
bytes, so hashing a merely sliced or concatenated table can report a false
content drift.

This module defines a new, explicitly versioned algorithm.  It rebuilds
canonical zero-offset buffers, zeroes fixed-width values hidden by nulls,
normalises validity/data padding bits, and rebuilds variable-width offsets
from logical values before writing the canonical table to Arrow IPC.  The v13
hash remains untouched; callers must opt in to this v14 API.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

import numpy as np
import pyarrow as pa


LOGICAL_TABLE_HASH_ALGORITHM_ID_V14 = (
    "SIRETO_ARROW_IDENTITY_TAKE_IPC_SHA256_NULL_CANONICAL_V14_1"
)
LOGICAL_TABLE_HASH_RECORD_SCHEMA_ID_V14 = (
    "sireto-retrieval-logical-table-hash-record-v14-1"
)

_RECORD_KEYS = frozenset(
    {
        "algorithm_id",
        "record_schema_id",
        "arrow_schema_sha256",
        "row_count",
        "column_count",
        "sha256",
    }
)
_SHA256_CHARS = frozenset("0123456789abcdef")


class LogicalHashInvariantErrorV14(ValueError):
    """Raised when a v14 hash request or sealed record is not canonical."""


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(_SHA256_CHARS)
    )


def _canonical_schema_v14(schema: pa.Schema) -> pa.Schema:
    """Keep logical field identity while discarding non-semantic metadata."""

    if not isinstance(schema, pa.Schema):
        raise TypeError("v14 logical hash requires a pyarrow.Schema")
    return pa.schema(
        [
            pa.field(field.name, field.type, nullable=field.nullable)
            for field in schema
        ]
    )


def logical_schema_sha256_v14(schema: pa.Schema) -> str:
    """Return the content ID of the canonical Arrow schema."""

    canonical = _canonical_schema_v14(schema)
    return hashlib.sha256(canonical.serialize().to_pybytes()).hexdigest()


def _canonical_table_v14(table: pa.Table) -> pa.Table:
    if not isinstance(table, pa.Table):
        raise TypeError("v14 logical hash requires a pyarrow.Table")
    schema = _canonical_schema_v14(table.schema)
    for index, field in enumerate(schema):
        if table.column(index).type != field.type:
            raise LogicalHashInvariantErrorV14(
                f"column type differs from schema for {field.name}"
            )
        if table.column(index).null_count and not field.nullable:
            raise LogicalHashInvariantErrorV14(
                f"non-nullable field contains nulls: {field.name}"
            )
        if pa.types.is_dictionary(field.type):
            raise LogicalHashInvariantErrorV14(
                f"dictionary encoding is not a logical v14 type: {field.name}"
            )

    # ``combine_chunks`` removes chunk-layout identity.  The C++ identity take
    # then allocates fresh arrays and copies *logical* values: offsets become
    # zero, null data slots become zero/empty, and trailing bitmap bits are
    # cleared.  This is materially faster than a Python value round-trip while
    # retaining valid float and integer bytes exactly.
    combined = table.combine_chunks()
    index_type = np.int32 if table.num_rows <= np.iinfo(np.int32).max else np.int64
    indices = pa.array(np.arange(table.num_rows, dtype=index_type))
    canonical = combined.take(indices)
    return pa.Table.from_arrays(list(canonical.columns), schema=schema)


def _canonical_header_v14(
    *, arrow_schema_sha256: str, row_count: int, column_count: int
) -> bytes:
    value = {
        "algorithm_id": LOGICAL_TABLE_HASH_ALGORITHM_ID_V14,
        "record_schema_id": LOGICAL_TABLE_HASH_RECORD_SCHEMA_ID_V14,
        "arrow_schema_sha256": arrow_schema_sha256,
        "row_count": row_count,
        "column_count": column_count,
    }
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


@dataclass(frozen=True)
class LogicalTableHashRecordV14:
    algorithm_id: str
    record_schema_id: str
    arrow_schema_sha256: str
    row_count: int
    column_count: int
    sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "algorithm_id": self.algorithm_id,
            "record_schema_id": self.record_schema_id,
            "arrow_schema_sha256": self.arrow_schema_sha256,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "sha256": self.sha256,
        }


def logical_table_hash_v14(table: pa.Table) -> LogicalTableHashRecordV14:
    """Hash a table after canonicalising all supported logical buffers."""

    canonical = _canonical_table_v14(table)
    schema_sha = logical_schema_sha256_v14(canonical.schema)
    header = _canonical_header_v14(
        arrow_schema_sha256=schema_sha,
        row_count=canonical.num_rows,
        column_count=canonical.num_columns,
    )
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, canonical.schema) as writer:
        writer.write_table(canonical, max_chunksize=max(1, canonical.num_rows))
    digest = hashlib.sha256(header + sink.getvalue().to_pybytes()).hexdigest()
    return LogicalTableHashRecordV14(
        algorithm_id=LOGICAL_TABLE_HASH_ALGORITHM_ID_V14,
        record_schema_id=LOGICAL_TABLE_HASH_RECORD_SCHEMA_ID_V14,
        arrow_schema_sha256=schema_sha,
        row_count=canonical.num_rows,
        column_count=canonical.num_columns,
        sha256=digest,
    )


def logical_table_sha256_v14(table: pa.Table) -> str:
    """Convenience surface returning only the v14 content digest."""

    return logical_table_hash_v14(table).sha256


def verify_logical_table_hash_v14(
    table: pa.Table,
    record: LogicalTableHashRecordV14 | Mapping[str, Any],
) -> LogicalTableHashRecordV14:
    """Strictly verify algorithm, record schema, Arrow schema and digest."""

    raw = record.as_dict() if isinstance(record, LogicalTableHashRecordV14) else dict(record)
    if set(raw) != _RECORD_KEYS:
        raise LogicalHashInvariantErrorV14("v14 logical hash record keyset drifted")
    if raw.get("algorithm_id") != LOGICAL_TABLE_HASH_ALGORITHM_ID_V14:
        raise LogicalHashInvariantErrorV14("v14 logical hash algorithm drifted")
    if raw.get("record_schema_id") != LOGICAL_TABLE_HASH_RECORD_SCHEMA_ID_V14:
        raise LogicalHashInvariantErrorV14("v14 logical hash record schema drifted")
    if (
        not _valid_sha256(raw.get("arrow_schema_sha256"))
        or not _valid_sha256(raw.get("sha256"))
        or type(raw.get("row_count")) is not int
        or raw["row_count"] < 0
        or type(raw.get("column_count")) is not int
        or raw["column_count"] < 0
    ):
        raise LogicalHashInvariantErrorV14("v14 logical hash record is malformed")
    observed = logical_table_hash_v14(table)
    if raw != observed.as_dict():
        if raw.get("arrow_schema_sha256") != observed.arrow_schema_sha256:
            raise LogicalHashInvariantErrorV14("v14 logical Arrow schema drifted")
        raise LogicalHashInvariantErrorV14("v14 logical table content drifted")
    return observed


__all__ = [
    "LOGICAL_TABLE_HASH_ALGORITHM_ID_V14",
    "LOGICAL_TABLE_HASH_RECORD_SCHEMA_ID_V14",
    "LogicalHashInvariantErrorV14",
    "LogicalTableHashRecordV14",
    "logical_schema_sha256_v14",
    "logical_table_hash_v14",
    "logical_table_sha256_v14",
    "verify_logical_table_hash_v14",
]
