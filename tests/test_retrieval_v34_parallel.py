import json
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scripts import retrieval_v34_parallel_legacy as parallel


def test_parallel_merge_restores_order_without_dropping_rows(tmp_path,monkeypatch):
    monkeypatch.setattr(parallel.Budget,'swap',staticmethod(lambda:0))
    class Job:
        returncode=0
        def __init__(self,args,**kwargs):
            path=Path(args[-1]);queries=json.loads((path/'queries.json').read_text())
            rows=[dict(query_id=q['query_id'],fold=q['fold'],siren='p',siret='s'+str(rank),union_rank=rank)
              for q in reversed(queries) for rank in (2,1)]
            pq.write_table(pa.Table.from_pylist(rows,schema=parallel.v33.RAW_SCHEMA),path/'raw.parquet')
            (path/'legacy_timing.json').write_text(json.dumps([dict(query_id=q['query_id']) for q in queries]))
            (path/'manifest.json').write_text(json.dumps(dict(resources=dict(pass_resource_policy=True,peak_rss_bytes=1))))
        def poll(self):return 0
        def wait(self,**kwargs):return 0
    monkeypatch.setattr(parallel.subprocess,'Popen',Job)
    queries=[(q,2,parallel.RetrievalQuery(name='ATELIER')) for q in ('z','b','a')]
    parallel.run_parallel(queries,tmp_path)
    result=pq.read_table(tmp_path/'raw.parquet').to_pylist()
    assert [(r['query_id'],r['union_rank']) for r in result]==[('z',1),('z',2),('b',1),('b',2),('a',1),('a',2)]
    report=json.loads((tmp_path/'parallel_legacy.json').read_text())
    assert report['workers']==2 and report['swapout_pages_delta']==0


def test_child_failure_cannot_publish_raw(tmp_path,monkeypatch):
    monkeypatch.setattr(parallel.Budget,'swap',staticmethod(lambda:0))
    class Job:
        returncode=1
        def __init__(self,*args,**kwargs):pass
        def poll(self):return 1
        def wait(self,**kwargs):return 1
    monkeypatch.setattr(parallel.subprocess,'Popen',Job)
    with pytest.raises(RuntimeError,match='Legacy child failed'):
        parallel.run_parallel([('q',2,parallel.RetrievalQuery(name='ATELIER'))],tmp_path)
    assert not (tmp_path/'raw.parquet').exists()
