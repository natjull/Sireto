from collections import Counter
from datetime import datetime

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.project_retrieval_v82_compressed import CompressedConnection,project_compressed
from scripts.project_retrieval_v82_scoped_succession_names import project
from tests.test_retrieval_scoped_succession_names_v82 import fixture


def test_compression_preserves_schema_full_rows_provenance_and_stats(tmp_path):
    bindings,names=fixture(tmp_path)
    rows=pq.read_table(names).to_pylist()
    for i,row in enumerate(rows):
        row['observed_at']=datetime(2023,1,1) if i%2 else None
        row['valid_from']=datetime(2020,1,1) if i%3 else None
        row['valid_to']=datetime(2022,1,1) if i%4 else None
        row['source']='BODACC' if i%2 else 'RNE'
    pq.write_table(pa.Table.from_pylist(rows),names)
    baseline=tmp_path/'baseline'; compressed=tmp_path/'compressed'
    baseline.mkdir(); compressed.mkdir()
    c=duckdb.connect(); left=project(c,bindings,names,baseline); c.close()
    c=duckdb.connect()
    right=project_compressed(c,bindings,names,compressed); c.close()
    for field in left:
        if field!='phases': assert left[field]==right[field]
    for name in ('endpoint_scope.parquet','scoped_name_observations.parquet'):
        a=pq.read_table(baseline/name); b=pq.read_table(compressed/name)
        assert a.schema==b.schema
        def multiset(table):
            return Counter(tuple(row[column] for column in table.column_names) for row in table.to_pylist())
        assert multiset(a)==multiset(b)
    assert right['compression_level']==6


def test_adapter_rejects_unexpected_copy_options():
    c=duckdb.connect(); adapter=CompressedConnection(c)
    with pytest.raises(ValueError,match='skeleton drift'):
        adapter.execute('COPY (SELECT 1) TO ? (FORMAT PARQUET)', ['unused'])
    c.close()


def test_compressed_projection_keeps_conflicting_parent_guard(tmp_path):
    bindings,names=fixture(tmp_path,True); c=duckdb.connect()
    with pytest.raises(ValueError,match='Conflicting official parent'):
        project_compressed(c,bindings,names,tmp_path)
    c.close()
