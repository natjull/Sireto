import json

import pytest

import scripts.probe_retrieval_v151_name_pages as workbench
from src.xgb_matcher.retrieval_v20_workbench import Budget,record,verify,write_json
from test_probe_retrieval_v150_rebound_names import setup as original_setup
from test_compile_retrieval_v149_name_rebind import gate


def setup(tmp_path,monkeypatch):
    cfg,out=original_setup(tmp_path,monkeypatch)
    config=tmp_path/'v151_config.json';write_json(config,cfg);monkeypatch.setattr(workbench,'CONFIG',config)
    # Scope the runtime substitution to this test, including STATUS.
    monkeypatch.setattr(workbench.original,'STATUS',workbench.original.STATUS)
    monkeypatch.setattr(workbench,'ScopedNamePages',workbench.original.NationalNames)
    return cfg,out


def start(out,cfg,policy='V151_FULL_READONLY_MAP_ROTATION_AFTER_EVERY_COMPLETE_QUERY'):
    out.mkdir();write_json(out/'preflight.json',gate())
    write_json(out/'launch.json',dict(config=cfg,preflight=record(out/'preflight.json'),implementation=workbench.implementation(),
        source_page_policy=policy,old_candidates_opened=False,**workbench.original.FLAGS))


def test_specialized_worker_retains_complete_original_query_protections(tmp_path,monkeypatch):
    cfg,out=setup(tmp_path,monkeypatch);start(out,cfg);workbench.worker(out)
    m=json.loads((out/'manifest.json').read_text());rows=json.loads(verify(m['outputs']).read_text())['rows']
    assert m['status']==workbench.STATUS and m['queries']==2 and [r['query_id'] for r in rows]==[0,1]
    assert m['resources']['swapout_pages_delta']==0 and m['every_used_input_full_hash_verified_before_and_after']
    assert not m['initial100_emitted'] and not m['recall_measured'] and not m['fold0_opened'] and not m['fold1_opened']
    assert not m['ready_for_fold1'] and not m['complete_retrieval_index']
    with pytest.raises(ValueError,match='no retry'):workbench.worker(out)
    with pytest.raises(FileExistsError):workbench.run(out)


def test_unsealed_page_policy_and_weakened_budget_not_implicitly_accepted(tmp_path,monkeypatch):
    cfg,out=setup(tmp_path,monkeypatch);start(out,cfg,policy='approximate')
    with pytest.raises(ValueError,match='Exact sealed'):workbench.worker(out)
    assert not (out/'manifest.json').exists()
    cfg['resources']['max_swapout_pages_delta']=1;write_json(workbench.CONFIG,cfg)
    fresh=tmp_path/'fresh'
    with pytest.raises(ValueError,match='unchanged zero swap'):workbench.run(fresh)
    assert not fresh.exists()


def test_resource_failure_after_query_does_not_publish_success(tmp_path,monkeypatch):
    cfg,out=setup(tmp_path,monkeypatch);start(out,cfg);before=Budget.__exit__
    def fail(self,*args):before(self,*args);raise RuntimeError('STOP_RESOURCE_POLICY')
    monkeypatch.setattr(Budget,'__exit__',fail)
    with pytest.raises(RuntimeError,match='STOP_RESOURCE_POLICY'):workbench.worker(out)
    assert (out/'worker_stop.json').exists() and not (out/'manifest.json').exists()


def test_frozen_fold1_input_refused_before_any_new_root(tmp_path,monkeypatch):
    cfg,out=setup(tmp_path,monkeypatch);p=verify(cfg['query_only_launch']);q=json.loads(p.read_text());q['queries'][0]['fold']=1
    write_json(p,q);cfg['query_only_launch']=record(p);write_json(workbench.CONFIG,cfg)
    with pytest.raises(ValueError,match='Frozen query-only'):workbench.run(out)
    assert not out.exists()
