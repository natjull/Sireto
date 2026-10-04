"""Tiny complete-source fixtures, fake pressure/supervision, no national claim."""
import json
from pathlib import Path

import numpy as np
import pytest

import scripts.probe_retrieval_v169_paged_names as workbench
import src.xgb_matcher.retrieval_paged_names_v169 as paged
from src.xgb_matcher import retrieval_memory_policy_v153 as policy
from src.xgb_matcher.retrieval_v20_workbench import record,verify,write_json
from test_probe_retrieval_v150_rebound_names import setup as previous_setup
from test_resume_retrieval_v153_addresses import gate


def setup(tmp_path,monkeypatch):
    old,out=previous_setup(tmp_path,monkeypatch)
    cfg=json.loads(workbench.CONFIG.read_text())
    cfg.update({k:old[k] for k in ('names','joins','query_only_launch','output_root')});cfg.update(queries=1,chunk=2)
    config=tmp_path/'v169_config.json';write_json(config,cfg);monkeypatch.setattr(workbench,'CONFIG',config)
    from scripts.probe_retrieval_v150_rebound_names import NationalNames as FixtureIndex
    monkeypatch.setattr(workbench,'ScopedNamePages',FixtureIndex)
    monkeypatch.setattr(workbench,'quiet',gate)
    monkeypatch.setattr(policy,'snapshot',lambda:dict(swapouts=42,free_percent=73,pressure_level=1))
    def fixture_coverage(index,text,directory,guard,chunk):
        c,e,stats=index.coverage(text)
        vectors=[]
        for name,value in (('alias_coverage',c),('alias_exact',e)):
            a=paged.DiskVector(directory/(name+'.npy'),len(value),value.dtype);a.assign(0,value);vectors.append(a)
        return *vectors,stats
    monkeypatch.setattr(paged,'coverage',fixture_coverage)
    def supervise(args,log,env,stop_path):
        workbench.worker(Path(args[args.index('--output')+1]))
        return dict(policy_id=policy.POLICY,limits=policy.LIMITS,child_pid=1,exit_code=0,
            own_child_exit_confirmed=True,peak_rss_bytes=1,minimum_system_free_percent=73,
            sample_count=1,wall_seconds=0.,swapout_pages_delta=0,global_swap_is_worker_attributed=False)
    monkeypatch.setattr(workbench,'supervise',supervise)
    return cfg,out


def test_entire_site_gather_closed_labels_successful_exit_and_owned_cleanup(tmp_path,monkeypatch):
    cfg,out=setup(tmp_path,monkeypatch);workbench.run(out)
    m=json.loads((out/'manifest.json').read_text());rows=json.loads(verify(m['outputs']).read_text())['rows']
    assert m['status']==workbench.STATUS and m['queries']==1 and rows[0]['query_id']==0
    assert rows[0]['gather']['sites']==5 and rows[0]['gather']['every_site_gathered_once']
    assert len(rows[0]['gather']['sha256'])==64 and m['query_features_float64_unchanged']
    assert not list((out/'queries').rglob('*.npy'))  # Only owned successful-query scratch deleted.
    assert not m['truth_opened'] and not m['recall_measured'] and not m['fold0_opened'] and not m['fold1_opened']
    assert not m['initial100_emitted'] and not m['ready_for_fold1'] and not m['complete_retrieval_index']
    assert json.loads((out/'supervision.json').read_text())['own_child_exit_confirmed']
    with pytest.raises(FileExistsError):workbench.run(out)
    with pytest.raises(ValueError,match='no retry'):workbench.worker(out)


@pytest.mark.parametrize('corruption',['fold0','fold1','truth','old_hits','extra_field'])
def test_query_source_closed_before_root_or_worker(tmp_path,monkeypatch,corruption):
    cfg,out=setup(tmp_path,monkeypatch);path=verify(cfg['query_only_launch']);q=json.loads(path.read_text())
    if corruption=='fold0':q['queries'][0]['fold']=0
    if corruption=='fold1':q['queries'][0]['fold']=1
    if corruption=='truth':q['truth_opened']=True
    if corruption=='old_hits':q['old_candidates_opened']=True
    if corruption=='extra_field':q['queries'][0]['target_siret']='never a feature'
    write_json(path,q);cfg['query_only_launch']=record(path);write_json(workbench.CONFIG,cfg)
    with pytest.raises(ValueError,match='Frozen query-only'):workbench.run(out)
    assert not out.exists()


def test_worker_budget_failure_never_publishes_success(tmp_path,monkeypatch):
    cfg,out=setup(tmp_path,monkeypatch);before=policy.Budget.__exit__
    def fail(self,*args):before(self,*args);raise RuntimeError('STOP_RESOURCE_POLICY')
    monkeypatch.setattr(policy.Budget,'__exit__',fail)
    with pytest.raises(RuntimeError,match='RESOURCE_POLICY'):workbench.run(out)
    assert (out/'worker_stop.json').exists() and (out/'status.json').exists() and not (out/'manifest.json').exists()


def test_no_completed_certificate_without_confirmed_actual_child_exit(tmp_path,monkeypatch):
    cfg,out=setup(tmp_path,monkeypatch);before=workbench.supervise
    def disappear(*args):
        s=before(*args);s['own_child_exit_confirmed']=False;return s
    monkeypatch.setattr(workbench,'supervise',disappear)
    with pytest.raises(ValueError,match='UNCONFIRMED'):workbench.run(out)
    assert (out/'status.json').exists()  # Worker manifest alone is not a PASS authority.


def test_query_failure_keeps_owned_files_and_has_no_success(tmp_path,monkeypatch):
    cfg,out=setup(tmp_path,monkeypatch)
    def stop(*args):
        folder=Path(args[3]);folder.mkdir(parents=True)
        paged.DiskVector(folder/'failed.npy',4,np.float64)
        raise TimeoutError('STOP_PAGED_NAME_QUERY_COST')
    monkeypatch.setattr(workbench.PagedReboundNames,'query',stop)
    with pytest.raises(TimeoutError):workbench.run(out)
    assert (out/'queries'/'query0000'/'failed.npy').exists() and not (out/'manifest.json').exists()


@pytest.mark.parametrize('key,value',[('max_rss_bytes',3221225472),('min_system_free_percent',10),('required_pressure_level',2)])
def test_resource_policy_cannot_implicitly_be_relaxed(tmp_path,monkeypatch,key,value):
    cfg,out=setup(tmp_path,monkeypatch);cfg['resources'][key]=value;write_json(workbench.CONFIG,cfg)
    with pytest.raises(ValueError,match='authorized memory policy'):workbench.run(out)
    assert not out.exists()
