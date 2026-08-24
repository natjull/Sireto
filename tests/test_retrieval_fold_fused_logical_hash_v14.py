from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.xgb_matcher.retrieval_fold_fused_logical_hash_v14 import (
    LOGICAL_TABLE_HASH_ALGORITHM_ID_V14,
    LOGICAL_TABLE_HASH_RECORD_SCHEMA_ID_V14,
    LogicalHashInvariantErrorV14,
    logical_schema_sha256_v14,
    logical_table_hash_v14,
    logical_table_sha256_v14,
    verify_logical_table_hash_v14,
)


ROOT = Path(__file__).resolve().parents[1]


def _table(array: pa.Array | pa.ChunkedArray, name: str = "value") -> pa.Table:
    return pa.Table.from_arrays([array], schema=pa.schema([(name, array.type)]))


@pytest.mark.parametrize("bitmap_offset", range(8))
def test_v14_null_hash_is_independent_of_bitmap_offset(
    bitmap_offset: int,
) -> None:
    length = 16
    validity = bytearray(b"\xff\xff")
    validity[bitmap_offset // 8] &= ~(1 << (bitmap_offset % 8))
    values = np.arange(length, dtype="<i2")
    values[bitmap_offset] = 1
    source = pa.Array.from_buffers(
        pa.int16(),
        length,
        [pa.py_buffer(bytes(validity)), pa.py_buffer(values.tobytes())],
        null_count=1,
    )
    sliced = _table(source.slice(bitmap_offset, 1))
    standalone = _table(pa.array([None], type=pa.int16(), from_pandas=False))

    assert sliced.to_pylist() == standalone.to_pylist()
    assert logical_table_hash_v14(sliced) == logical_table_hash_v14(standalone)


def test_v14_hash_is_independent_of_slice_and_chunk_layout() -> None:
    source = pa.table(
        {
            "integer": pa.array([0, 1, None, 3, 4, None, 6, 7], type=pa.int32()),
            "string": pa.array(["a", "b", None, "d", "e", None, "g", "h"]),
            "floating": pa.array(
                np.array([0.0, -0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float32)
            ),
            "boolean": pa.array([True, False, None, True, False, None, True, False]),
        }
    )
    sliced = source.slice(2, 5)
    rebuilt = pa.Table.from_pylist(sliced.to_pylist(), schema=sliced.schema)
    chunked = pa.Table.from_arrays(
        [
            pa.chunked_array(
                [rebuilt.column(index).slice(0, 2), rebuilt.column(index).slice(2)]
            )
            for index in range(rebuilt.num_columns)
        ],
        schema=rebuilt.schema,
    )

    assert sliced.to_pylist() == rebuilt.to_pylist() == chunked.to_pylist()
    assert logical_table_sha256_v14(sliced) == logical_table_sha256_v14(rebuilt)
    assert logical_table_sha256_v14(rebuilt) == logical_table_sha256_v14(chunked)


def test_v14_ignores_masked_primitive_string_binary_and_bitmap_bytes() -> None:
    validity = pa.py_buffer(b"\x02")
    clean = pa.table(
        {
            "rank": pa.array([None, 7], type=pa.int16()),
            "score": pa.array([None, np.float32(2.5)], type=pa.float32()),
            "flag": pa.array([True, None], type=pa.bool_()),
            "name": pa.array([None, "x"], type=pa.string()),
            "blob": pa.array([None, b"x"], type=pa.binary()),
            "fixed": pa.array([None, b"WXYZ"], type=pa.binary(4)),
        }
    )
    dirty_rank = pa.Array.from_buffers(
        pa.int16(),
        2,
        [validity, pa.py_buffer(b"\x01\x00\x07\x00")],
        null_count=1,
    )
    dirty_score = pa.Array.from_buffers(
        pa.float32(),
        2,
        [
            validity,
            pa.py_buffer(np.array([np.nan, np.float32(2.5)], dtype="<f4").tobytes()),
        ],
        null_count=1,
    )
    dirty_flag = pa.Array.from_buffers(
        pa.bool_(),
        2,
        [pa.py_buffer(b"\x01"), pa.py_buffer(b"\xff")],
        null_count=1,
    )
    offsets = pa.py_buffer(np.array([0, 3, 4], dtype="<i4").tobytes())
    dirty_name = pa.Array.from_buffers(
        pa.string(),
        2,
        [validity, offsets, pa.py_buffer(b"abcx")],
        null_count=1,
    )
    dirty_blob = pa.Array.from_buffers(
        pa.binary(),
        2,
        [validity, offsets, pa.py_buffer(b"abcx")],
        null_count=1,
    )
    dirty_fixed = pa.Array.from_buffers(
        pa.binary(4),
        2,
        [validity, pa.py_buffer(b"ABCDWXYZ")],
        null_count=1,
    )
    dirty = pa.Table.from_arrays(
        [dirty_rank, dirty_score, dirty_flag, dirty_name, dirty_blob, dirty_fixed],
        schema=clean.schema,
    )

    assert clean.to_pylist() == dirty.to_pylist()
    assert logical_table_hash_v14(clean) == logical_table_hash_v14(dirty)


def test_v14_identity_take_canonicalizes_nested_null_ranges() -> None:
    dirty = pa.Array.from_buffers(
        pa.list_(pa.int16()),
        2,
        [
            pa.py_buffer(b"\x02"),
            pa.py_buffer(np.array([0, 2, 3], dtype="<i4").tobytes()),
        ],
        children=[pa.array([7, 8, 9], type=pa.int16())],
        null_count=1,
    )
    clean = pa.array([None, [9]], type=pa.list_(pa.int16()))

    assert dirty.to_pylist() == clean.to_pylist()
    assert logical_table_sha256_v14(_table(dirty)) == logical_table_sha256_v14(
        _table(clean)
    )


def test_v14_preserves_valid_float32_bits() -> None:
    positive_zero = pa.table(
        {"value": pa.array(np.array([0.0], dtype=np.float32))}
    )
    negative_zero = pa.table(
        {"value": pa.array(np.array([-0.0], dtype=np.float32))}
    )

    assert positive_zero.to_pylist() == negative_zero.to_pylist()
    assert logical_table_sha256_v14(positive_zero) != logical_table_sha256_v14(
        negative_zero
    )


def test_v14_parquet_roundtrip_preserves_logical_hash(tmp_path: Path) -> None:
    validity = pa.py_buffer(b"\x06")
    ranks = pa.Array.from_buffers(
        pa.int16(),
        3,
        [validity, pa.py_buffer(b"\x01\x00\x02\x00\x03\x00")],
        null_count=1,
    )
    names = pa.Array.from_buffers(
        pa.string(),
        3,
        [
            validity,
            pa.py_buffer(np.array([0, 5, 6, 7], dtype="<i4").tobytes()),
            pa.py_buffer(b"dirtyxy"),
        ],
        null_count=1,
    )
    source = pa.table({"rank": ranks, "name": names})
    parquet = tmp_path / "roundtrip.parquet"
    pq.write_table(source, parquet, compression="zstd", row_group_size=3)
    reopened = pq.read_table(parquet)

    assert reopened.to_pylist() == source.to_pylist()
    assert logical_table_hash_v14(reopened) == logical_table_hash_v14(source)


def test_v14_hash_is_repeatable_across_fresh_subprocesses() -> None:
    program = """
import json
import numpy as np
import pyarrow as pa
from xgb_matcher.retrieval_fold_fused_logical_hash_v14 import logical_table_hash_v14
validity = pa.py_buffer(b"\\x02")
rank = pa.Array.from_buffers(pa.int16(), 2, [validity, pa.py_buffer(b"\\x01\\x00\\x07\\x00")], null_count=1)
offsets = pa.py_buffer(np.array([0, 3, 4], dtype="<i4").tobytes())
name = pa.Array.from_buffers(pa.string(), 2, [validity, offsets, pa.py_buffer(b"abcx")], null_count=1)
print(json.dumps(logical_table_hash_v14(pa.table({"rank": rank, "name": name})).as_dict(), sort_keys=True))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    outputs = [
        subprocess.check_output(
            [sys.executable, "-c", program],
            cwd=ROOT,
            env=environment,
            text=True,
        ).strip()
        for _ in range(3)
    ]

    assert len(set(outputs)) == 1
    record = json.loads(outputs[0])
    assert record["algorithm_id"] == LOGICAL_TABLE_HASH_ALGORITHM_ID_V14
    assert record["record_schema_id"] == LOGICAL_TABLE_HASH_RECORD_SCHEMA_ID_V14


def test_v14_verifier_rejects_algorithm_record_schema_and_arrow_schema_drift() -> None:
    table = pa.table({"value": pa.array([1, None, 3], type=pa.int32())})
    record = logical_table_hash_v14(table)
    assert logical_schema_sha256_v14(table.schema) == record.arrow_schema_sha256
    assert verify_logical_table_hash_v14(table, record) == record

    algorithm_drift = record.as_dict()
    algorithm_drift["algorithm_id"] = "SIRETO_OTHER_ALGORITHM"
    with pytest.raises(LogicalHashInvariantErrorV14, match="algorithm drifted"):
        verify_logical_table_hash_v14(table, algorithm_drift)

    record_schema_drift = record.as_dict()
    record_schema_drift["record_schema_id"] = "sireto-other-record"
    with pytest.raises(LogicalHashInvariantErrorV14, match="record schema drifted"):
        verify_logical_table_hash_v14(table, record_schema_drift)

    different_schema = pa.table(
        {"value": pa.array([1, None, 3], type=pa.int64())}
    )
    assert logical_schema_sha256_v14(different_schema.schema) != record.arrow_schema_sha256
    assert logical_table_sha256_v14(different_schema) != record.sha256
    with pytest.raises(LogicalHashInvariantErrorV14, match="Arrow schema drifted"):
        verify_logical_table_hash_v14(different_schema, record)

    different_content = pa.table(
        {"value": pa.array([1, None, 4], type=pa.int32())}
    )
    with pytest.raises(LogicalHashInvariantErrorV14, match="content drifted"):
        verify_logical_table_hash_v14(different_content, record)


def test_v14_rejects_dictionary_encoding_as_physical_not_logical_schema() -> None:
    dictionary = pa.DictionaryArray.from_arrays(
        pa.array([0, 1, 0], type=pa.int8()), pa.array(["a", "b"])
    )
    with pytest.raises(LogicalHashInvariantErrorV14, match="dictionary encoding"):
        logical_table_hash_v14(_table(dictionary))
