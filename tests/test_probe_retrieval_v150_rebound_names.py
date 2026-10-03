import json

import pytest

import scripts.probe_retrieval_v150_rebound_names as workbench
from src.xgb_matcher.retrieval_v20_workbench import Budget,record,write_json,verify
from test_retrieval_rebound_names_v150 import setup as source_setup,Index
from test_compile_retrieval_v149_name_rebind import gate


def setup(tmp_path,monkeypatch):
    ref,_=source_setup(tmp_path,monkeypatch);cfg=json.loads(workbench.CONFIG.read_text())
    query=tmp_path/'query_only.json'
    rows=[dict(query_id=i,fold=2+i%3,crm_name='fixture',crm_number='',crm_suffix='',crm_street='',crm_postcode='',crm_insee='') for i in range(128)]
    write_json(query,dict(queries=rows,truth_opened=False,old_candidates_opened=False))
    lexical=tmp_path/'lexical_fixture.json';write_json(lexical,dict(fixture_only=True))
    cfg.update(joins=ref,names=record(lexical),query_only_launch=record(query),queries=2,output_root=str(tmp_path))
    config=tmp_path/'probe_config.json';write_json(config,cfg);monkeypatch.setattr(workbench,'CONFIG',config)
    m=json.loads(verify(ref).read_text());owners=json.loads(verify(m['sources']['owners']).read_text())
    class LocalIndex(Index):
        def __init__(self,ref):super().__init__(owners['source']);self.metadata=dict(access=[])
        def close(self):pass
    monkeypatch.setattr(workbench,'NationalNames',LocalIndex);monkeypatch.setattr(workbench,'quiet',gate)
    return cfg,tmp_path/'probe'


def start(out,cfg):
    out.mkdir();write_json(out/'preflight.json',gate())
    write_json(out/'launch.json',dict(config=cfg,preflight=record(out/'preflight.json'),implementation=workbench.implementation(),old_candidates_opened=False,**workbench.FLAGS))


def test_probe_only_outputs_frozen_query_prefix_without_recall_or_candidates(tmp_path,monkeypatch):
    cfg,out=setup(tmp_path,monkeypatch);start(out,cfg);workbench.worker(out)
    m=json.loads((out/'manifest.json').read_text());o=json.loads(verify(m['outputs']).read_text())
    assert m['status']==workbench.STATUS and m['queries']==2
    assert [r['query_id'] for r in o['rows']]==[0,1] and [r['fold'] for r in o['rows']]==[2,3]
    assert m['every_used_input_full_hash_verified_before_and_after'] and m['resources']['pass_resource_policy']
    assert m['crm_opened'] and not m['truth_opened'] and not m['fold0_opened'] and not m['fold1_opened']
    assert not m['initial100_emitted'] and not m['recall_measured'] and not m['complete_retrieval_index'] and not m['ready_for_fold1']
    with pytest.raises(ValueError,match='no retry'):workbench.worker(out)
    with pytest.raises(FileExistsError):workbench.run(out)


@pytest.mark.parametrize('corruption',['fold0','fold1','truth','old_hits','extra_field'])
def test_query_source_no_closed_fold_truth_hit_or_extra_feature_before_worker(tmp_path,monkeypatch,corruption):
    cfg,out=setup(tmp_path,monkeypatch);p=verify(cfg['query_only_launch']);q=json.loads(p.read_text())
    if corruption=='fold0':q['queries'][0]['fold']=0
    if corruption=='fold1':q['queries'][0]['fold']=1
    if corruption=='truth':q['truth_opened']=True
    if corruption=='old_hits':q['old_candidates_opened']=True
    if corruption=='extra_field':q['queries'][0]['target_siret']='never read'
    write_json(p,q);cfg['query_only_launch']=record(p);write_json(workbench.CONFIG,cfg)
    with pytest.raises(ValueError,match='Frozen query-only'):workbench.run(out)
    assert not out.exists()


def test_quiet_stop_does_not_create_root_or_launch_worker(tmp_path,monkeypatch):
    cfg,out=setup(tmp_path,monkeypatch)
    def stop(*args):raise RuntimeError('STOP_SYSTEM_NOT_QUIET_NO_WORKER_LAUNCHED')
    monkeypatch.setattr(workbench,'quiet',stop)
    with pytest.raises(RuntimeError,match='NO_WORKER_LAUNCHED'):workbench.run(out)
    assert not out.exists()


def test_budget_failure_after_queries_has_no_success_manifest(tmp_path,monkeypatch):
    cfg,out=setup(tmp_path,monkeypatch);start(out,cfg);exit_before=Budget.__exit__
    def fail(self,*args):exit_before(self,*args);raise RuntimeError('STOP_RESOURCE_POLICY')
    monkeypatch.setattr(Budget,'__exit__',fail)
    with pytest.raises(RuntimeError,match='STOP_RESOURCE_POLICY'):workbench.worker(out)
    assert (out/'worker_stop.json').exists() and not (out/'manifest.json').exists()
    with pytest.raises(ValueError,match='no retry'):workbench.worker(out)


def test_relaxed_global_swap_and_wrong_root_never_start(tmp_path,monkeypatch):
    cfg,out=setup(tmp_path,monkeypatch);cfg['resources']['max_swapout_pages_delta']=1;write_json(workbench.CONFIG,cfg)
    with pytest.raises(ValueError,match='unchanged zero swap'):workbench.run(out)
    assert not out.exists()
    cfg['resources']['max_swapout_pages_delta']=0;cfg['output_root']=str(tmp_path/'restricted');write_json(workbench.CONFIG,cfg)
    with pytest.raises(ValueError,match='CATNAT-only'):workbench.run(out)
    assert not out.exists()
