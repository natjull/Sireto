from types import SimpleNamespace
from pathlib import Path
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.retrieval_v34_runtime import BoundedCache, CachedSearcher, FastRuntime


def test_cache_bounds_and_lru():
    c=BoundedCache(capacity=2,max_bytes=1000)
    c.put('a','one');c.put('b','two');assert c.get('a')=='one'
    c.put('c','three');assert c.get('b') is None
    c.put('large','x'*2000);assert c.get('large') is None
    assert c.bytes<=1000 and len(c.values)<=2


def test_document_cache_scoped_to_searcher_and_segment():
    class Native:
        def __init__(self,prefix):self.calls=0;self.prefix=prefix
        def doc(self,a):
            self.calls+=1
            return SimpleNamespace(to_dict=lambda:dict(value=[self.prefix,a.segment_ord,a.doc]))
    a=SimpleNamespace(segment_ord=0,doc=1);b=SimpleNamespace(segment_ord=1,doc=1)
    native=Native('a');cache=CachedSearcher(native)
    first=cache.doc(a).to_dict();assert cache.doc(a).to_dict()==first
    assert native.calls==1 and cache.doc(b).to_dict()!=first
    assert CachedSearcher(Native('b')).doc(a).to_dict()!=first
    first['new']='caller';assert 'new' not in cache.doc(a).to_dict()


def test_projection_retains_complete_geographic_rows_and_counts(tmp_path):
    source=tmp_path/'source.parquet'
    rows=[dict(insee='A',postcode='1',siret='x',insee_site_count=7),
          dict(insee='A',postcode='2',siret='y',insee_site_count=7),
          dict(insee='B',postcode='3',siret='z',insee_site_count=5),
          dict(insee='C',postcode='4',siret='w',insee_site_count=9)]
    pq.write_table(pa.Table.from_pylist(rows),source)
    runtime=FastRuntime();connection=duckdb.connect()
    runtime.side=SimpleNamespace(close=connection.close,_connection=connection)
    runtime.original_local=source;runtime.original_geo=source
    output=tmp_path/'projection';output.mkdir()
    try:
        runtime.project_sidecars([(None,None,SimpleNamespace(insee='A',postcode='9')),
                                 (None,None,SimpleNamespace(insee='',postcode='3'))],output)
        assert pq.read_table(runtime.side._local_path).to_pylist()==rows[:3]
        assert pq.read_table(runtime.side._geo_path).to_pylist()==rows[:3]
    finally:runtime.close()
