import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import duckdb
import numpy as np
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.xgb_matcher.retrieval_address_access_v152 import SCHEMA, stage_addresses as old
from src.xgb_matcher.retrieval_address_channels_v164 import stage_addresses
from src.xgb_matcher import retrieval_supervisor_v164 as supervisor
from src.xgb_matcher.retrieval_memory_policy_v153 import LIMITS
from test_retrieval_supervisor_v157 import Child


@pytest.mark.parametrize('group', [1, 3, 8])
def test_every_whole_alias_channel_feature_and_export_equal_to_native131(tmp_path, group):
    texts = ['RUE TEST', 'AVENUE DE LA REPUBLIQUE', '123456789', 'A', 'ZZ',
        'SAS STE ET FILS', 'NUMERO 12345678901234', 'RUE TEST'] * 30
    table = pa.Table.from_pydict(dict(address_id=np.arange(9,9+len(texts),dtype=np.uint32),address_text=texts),schema=SCHEMA)
    before, after = tmp_path/'old', tmp_path/'new'
    before.mkdir();after.mkdir()
    with duckdb.connect() as c:
        a = old(c,table,before,700,9,64)
    with duckdb.connect() as c:
        b = stage_addresses(c,table,after,700,9,64,group)
        def rows(files):
            return c.execute('SELECT * FROM read_parquet(?,hive_partitioning=false) ORDER BY alias_id,channel,token',[files]).fetchall()
        assert rows(a['files']) == rows(b['files'])
    assert pq.read_table(before/'features.parquet').equals(pq.read_table(after/'features.parquet'))
    for key,value in a.items():
        if key!='files':assert b[key] == value
    assert b['maximum_physical_writers_per_copy']<=group
    assert b['copy_operations']==(64+group-1)//group
    assert b['physical_writer_group_is_not_a_candidate_cap']
    assert b['alias_or_proof_cap'] is None


def test_guarded_empty_features_and_immutable_stage(tmp_path):
    table = pa.Table.from_pydict(dict(address_id=[0],address_text=['123456789']),schema=SCHEMA)
    with duckdb.connect() as c:
        s = stage_addresses(c,table,tmp_path,0,0,64)
        assert s['addresses']==1 and s['postings']==0 and s['copy_operations']==0
        with pytest.raises(ValueError,match='Immutable'):
            stage_addresses(c,table,tmp_path,0,0,64)


@pytest.mark.parametrize('group', [0, 9, True])
def test_bound_is_explicit_not_a_hidden_alias_limit(tmp_path, group):
    table = pa.Table.from_pydict(dict(address_id=[0],address_text=['RUE']),schema=SCHEMA)
    with duckdb.connect() as c,pytest.raises(ValueError,match='bounded physical'):
        stage_addresses(c,table,tmp_path,0,0,64,group)


def fake_child(monkeypatch, code=0, rss=None):
    child=Child(code)
    monkeypatch.setattr(supervisor.subprocess,'Popen',lambda *a,**k:child)
    values=iter(range(42,1000))
    monkeypatch.setattr(supervisor,'snapshot',lambda:dict(swapouts=next(values),free_percent=73,pressure_level=1))
    if rss is None:
        def missing(pid):raise psutil.NoSuchProcess(pid)
        monkeypatch.setattr(supervisor.psutil,'Process',missing)
    else:
        monkeypatch.setattr(supervisor.psutil,'Process',lambda pid:SimpleNamespace(memory_info=lambda:SimpleNamespace(rss=rss)))
    return child


def test_race_success_only_when_own_child_reaped_actual_zero(monkeypatch,tmp_path):
    child=fake_child(monkeypatch)
    with (tmp_path/'log').open('x') as log:
        m=supervisor.supervise(['own-child'],log,{},tmp_path/'stop.json')
    assert m['exit_code']==0 and m['own_child_exit_confirmed'] and child.waited
    assert not (tmp_path/'stop.json').exists()


@pytest.mark.parametrize('code', [1,-9,137])
def test_nonzero_exit_stop_keeps_diagnostic_not_success(monkeypatch,tmp_path,code):
    child=fake_child(monkeypatch,code)
    with (tmp_path/'log').open('x') as log,pytest.raises(subprocess.CalledProcessError):
        supervisor.supervise(['own-child'],log,{},tmp_path/'stop.json')
    m=json.loads((tmp_path/'stop.json').read_text())
    assert child.waited and m['exit_code']==code and m['status']=='STOP'
    assert m['limits']==LIMITS and m['global_swap_is_worker_attributed'] is False


def test_actual_rss_stop_exports_peak_and_kills_only_owned_child(monkeypatch,tmp_path):
    child=fake_child(monkeypatch,rss=LIMITS['max_rss_bytes']+1)
    with (tmp_path/'log').open('x') as log,pytest.raises(RuntimeError,match='RESOURCE_POLICY'):
        supervisor.supervise(['own-child'],log,{},tmp_path/'stop.json')
    m=json.loads((tmp_path/'stop.json').read_text())
    assert child.terminated and m['peak_rss_bytes']==LIMITS['max_rss_bytes']+1
    assert m['failure_wall_seconds']>=0 and m['exit_code']==-15


def test_pressure_stop_exports_sample_even_before_pid_disappears(monkeypatch,tmp_path):
    child=fake_child(monkeypatch)
    monkeypatch.setattr(supervisor,'snapshot',lambda:dict(swapouts=42,free_percent=49,pressure_level=2))
    with (tmp_path/'log').open('x') as log,pytest.raises(RuntimeError,match='MEMORY_PRESSURE'):
        supervisor.supervise(['own-child'],log,{},tmp_path/'stop.json')
    m=json.loads((tmp_path/'stop.json').read_text())
    assert child.terminated and m['minimum_system_free_percent']==49
    assert m['last_sample']['pressure_level']==2 and m['sample_count']==1


def test_probe_uses_whole_successful_source_audits_not_stopped_lexical_outputs(tmp_path,monkeypatch):
    import scripts.probe_retrieval_v164_address_channels as probe
    import scripts.compile_retrieval_v163_address_access as access
    from src.xgb_matcher.retrieval_v20_workbench import record,write_json
    from test_compile_retrieval_v163_address_access import setup
    root,out,cfg=setup(tmp_path,monkeypatch)
    access.init(out,record(root/'manifest.json'))
    dest=access.directory(out,'part',0);dest.mkdir(parents=True)
    write_json(dest/'status.json',dict(status='STOP'))
    _,p,spec=probe.source(record(out/'launch.json'),0)
    assert p['sites']==11 and spec==p['parts'][0]
    with pytest.raises(ValueError,match='source audit launch'):
        probe.source(record(dest/'status.json'),0)
    # The probe borrows source certificates, never the failed postings.
    assert spec['source']['path'].endswith('addresses.parquet')


def test_complete_one_part_probe_worker_certificate_not_an_index(tmp_path,monkeypatch):
    import scripts.probe_retrieval_v164_address_channels as probe
    import scripts.compile_retrieval_v163_address_access as access
    from src.xgb_matcher.retrieval_v20_workbench import record
    from src.xgb_matcher.retrieval_memory_policy_v153 import POLICY
    from test_compile_retrieval_v163_address_access import setup
    from test_resume_retrieval_v153_addresses import gate
    root,out,cfg=setup(tmp_path,monkeypatch)
    access.init(out,record(root/'manifest.json'))
    monkeypatch.setattr(probe,'OUTPUT_ROOT',tmp_path)
    monkeypatch.setattr(probe,'quiet',gate)
    def supervised(args,log,env,stop_path):
        probe.worker(Path(args[args.index('--output')+1]))
        return dict(policy_id=POLICY,limits=LIMITS,child_pid=1,exit_code=0,
            own_child_exit_confirmed=True,peak_rss_bytes=1,minimum_system_free_percent=73,
            sample_count=1,wall_seconds=0.,swapout_pages_delta=0,global_swap_is_worker_attributed=False)
    monkeypatch.setattr(probe,'supervise',supervised)
    dest=tmp_path/'cost_probe'
    probe.run(dest,record(out/'launch.json'),0)
    m=json.loads((dest/'manifest.json').read_text())
    assert not m['complete_retrieval_index'] and not m['ready_for_fold1']
    assert m['stats']['addresses']==1 and m['stats']['alias_or_proof_cap'] is None
    assert m['stats']['all_scalar_channel_sets_exact'] and m['stats']['all_exported_postings_exact']
    assert m['resources']['limits']==LIMITS and all(m[k] is False for k in probe.PROTECTED)
