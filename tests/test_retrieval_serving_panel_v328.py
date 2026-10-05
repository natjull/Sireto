"""Frozen selection/serving-BAN parity and original-source controls, fixtures only."""
from copy import deepcopy
import inspect
import json
from types import SimpleNamespace

import pytest

from scripts import retrieval_v322_large_panel as original
from scripts import retrieval_v328_serving_panel as work
from test_retrieval_large_panel_v322 import all_queries


def test_no_new_seed_population_encoder_model_or_selection():
    rows=all_queries();pilot=rows[:128];train=[q['query_id'] for q in rows[128:640]]
    assert work.SEED==original.SEED and work.SIZE==original.SIZE==2000
    assert work.choose(rows,pilot,train)==original.choose(rows,pilot,train)
    assert work.population_rows.__code__.co_code==original.population_rows.__code__.co_code
    assert work.encode_worker.__code__.co_code==original.encode_worker.__code__.co_code
    assert work.LIMITS==original.LIMITS
    assert 'human_ground_truth' not in inspect.getsource(work)
    assert 'ban.parquet' not in inspect.getsource(work.read_sources)


@pytest.mark.parametrize('change',[None,'partial','parity','order','fold','pilot_overlap'])
def test_all128_exact_parity_and_all2000_geometry_before_seal(monkeypatch,change):
    rows=all_queries();pilot=rows[:128];selected=rows[640:2640]
    result=[dict(query_id=q['query_id'],fold=q['fold'],ban_resolved=False) for q in [*pilot,*selected]]
    expected=deepcopy(result[:128]);settings=[];calls=[]
    class Connection:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def execute(self,s):settings.append(s)
    monkeypatch.setattr(work.duckdb,'connect',Connection)
    def resolve(c,qs,files):calls.append(qs);return result
    monkeypatch.setattr(work.ban,'resolve',resolve)
    if change=='partial':result.pop()
    elif change=='parity':result[27]['ban_resolved']=True
    elif change=='order':result[-1],result[-2]=result[-2],result[-1]
    elif change=='fold':result[-1]['fold']=1
    elif change=='pilot_overlap':selected[0]=pilot[0]
    if change:
        with pytest.raises(ValueError):work.resolved_geometry(object(),pilot,selected,expected)
    else:
        assert len(work.resolved_geometry(object(),pilot,selected,expected))==2000
        assert calls==[[*pilot,*selected]]
        assert settings==['SET threads=1',"SET memory_limit='128MiB'","SET max_temp_directory_size='0B'",'SET enable_progress_bar=false']


def test_read_full_original_crm_and_never_adopt_legacy_ban(monkeypatch):
    rows=all_queries()
    for i,q in enumerate(rows):q['fold']=2 if i<7767 else 3 if i<15734 else 4
    manifests={'generation':dict(status='SEALED_FULL_PAIRED_GENERATION',queries=23571,batches=[])}
    parts={};touched=[]
    for start in range(0,23571,32):
        n=min(32,23571-start);b='batch'+str(start);s='shard'+str(start);p='queries'+str(start)
        manifests['generation']['batches'].append({'path':b})
        manifests[b]=dict(start=start,count=n,shards=[{'path':s}])
        manifests[s]=dict(files={'queries.parquet':{'path':p},'ban.parquet':{'path':'FORBIDDEN_BAN'}})
        parts[p]=rows[start:start+n]
    class File:
        def __init__(self,key):self.key=key
        def read_text(self):return json.dumps(manifests[self.key])
    class Files:
        def file(self,ref):
            key='generation' if ref['path']==str(work.v38.GEN) else ref['path']
            assert key!='FORBIDDEN_BAN';touched.append(key);return File(key)
    monkeypatch.setattr(work.pq,'read_schema',lambda p:SimpleNamespace(names=list(work.FIELDS)))
    monkeypatch.setattr(work.pq,'read_metadata',lambda p:SimpleNamespace(num_rows=len(parts[p.key])))
    def read(p,**kw):
        assert kw==dict(filters=[('fold','in',[2,3,4])],use_threads=False)
        return SimpleNamespace(to_pylist=lambda:parts[p.key])
    monkeypatch.setattr(work.pq,'read_table',read)
    assert work.read_sources(Files())==rows
    assert len(touched)==1+3*737
