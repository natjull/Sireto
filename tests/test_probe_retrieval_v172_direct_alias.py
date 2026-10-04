"""Complete tiny part, fake pressure and supervision; not a national claim."""
import json
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pytest

import scripts.probe_retrieval_v172_direct_alias as workbench
from src.xgb_matcher.retrieval_address_channels_v166 import stage_addresses
from src.xgb_matcher.retrieval_address_access_v152 import SCHEMA
from src.xgb_matcher.retrieval_direct_alias_v172 import evaluate
from src.xgb_matcher import retrieval_memory_policy_v153 as policy
from src.xgb_matcher.retrieval_v20_workbench import record,write_json,verify
from test_resume_retrieval_v153_addresses import gate

ORIGINAL_INPUTS=workbench.inputs


def part(tmp_path):
    folder=tmp_path/'source_part';folder.mkdir()
    c=duckdb.connect()
    table=pa.table(dict(address_id=pa.array(np.arange(6,dtype=np.uint32)),
        address_text=pa.array(['RUE JEAN JAURES','JEAN JEAN JEAN','JEANNETTE','CH 390865069','390865069','AAAAAAA'])))
    table=pa.Table.from_arrays(table.columns,schema=SCHEMA)
    try:stats=stage_addresses(c,table,folder,5,0,8)
    finally:c.close()
    files=[Path(x) for x in stats.pop('files')];files.append(folder/'features.parquet')
    manifest=dict(stats=stats,artifacts={str(f.relative_to(folder)):record(f) for f in files})
    write_json(folder/'manifest.json',manifest)
    return folder,manifest


@pytest.mark.parametrize('text',['RUE JEAN JAURES','JEAN','CH 390865069','AA AA AAA',''])
def test_every_feature_equals_every_original_posting_across_physical_reader_groups(tmp_path,text):
    import pyarrow.parquet as pq
    folder,m=part(tmp_path);c=duckdb.connect()
    try:
        table=pq.ParquetFile(folder/'features.parquet').read()
        ids,cov,exact,_=evaluate(c,table,text)
        files=[verify(ref) for name,ref in m['artifacts'].items() if name!='features.parquet']
        oc,oe=workbench.oracle(c,files,ids,text)
    finally:c.close()
    np.testing.assert_array_equal(cov,oc);np.testing.assert_array_equal(exact,oe)
    assert len(ids)==6 and ids.tolist()==list(range(5,11))


def setup(tmp_path,monkeypatch):
    folder,m=part(tmp_path);out=tmp_path/'output';cfg=json.loads(workbench.CONFIG.read_text())
    cfg.update(output_root=str(tmp_path),part_manifest=record(folder/'manifest.json'))
    launch=tmp_path/'source_launch.json';write_json(launch,{'fake':True});cfg['source_launch']=record(launch)
    query=tmp_path/'query.json';write_json(query,{'fake':True});cfg['query_only_launch']=record(query)
    config=tmp_path/'config.json';write_json(config,cfg);monkeypatch.setattr(workbench,'CONFIG',config)
    specs=[{} for _ in range(50)];specs[49]=dict(alias_start=5,row_start=0,row_stop=6)
    monkeypatch.setattr(workbench,'inputs',lambda cfg:(folder,{},dict(parts=specs),m,dict(query_id='fixture',fold=3,crm_street='RUE JEAN JAURES')))
    monkeypatch.setattr(workbench.backend,'revalidate',lambda *args:None)
    monkeypatch.setattr(workbench,'quiet',gate)
    monkeypatch.setattr(policy,'snapshot',lambda:dict(swapouts=42,free_percent=73,pressure_level=1))
    def supervise(args,log,env,stop_path):
        workbench.worker(Path(args[args.index('--output')+1]))
        return dict(policy_id=policy.POLICY,limits=policy.LIMITS,child_pid=1,exit_code=0,
            own_child_exit_confirmed=True,peak_rss_bytes=1,minimum_system_free_percent=73,
            sample_count=1,wall_seconds=0.,swapout_pages_delta=0,global_swap_is_worker_attributed=False)
    monkeypatch.setattr(workbench,'supervise',supervise)
    return out


def test_success_is_only_one_certified_part_not_national_or_recall(tmp_path,monkeypatch):
    out=setup(tmp_path,monkeypatch);workbench.run(out)
    m=json.loads((out/'manifest.json').read_text())
    assert m['status']==workbench.STATUS and m['every_score_bit_exact'] and m['aliases']==6
    assert not m['national_corpus_measured'] and not m['initial100_emitted'] and not m['recall_measured']
    assert not m['truth_opened'] and not m['fold0_opened'] and not m['fold1_opened'] and not m['ready_for_fold1']
    with pytest.raises(FileExistsError):workbench.run(out)
    with pytest.raises(ValueError,match='no retry'):workbench.worker(out)


def test_mismatched_oracle_cannot_publish_success(tmp_path,monkeypatch):
    out=setup(tmp_path,monkeypatch);original=workbench.oracle
    def broken(*args):
        cov,exact=original(*args);cov[0]=0.;return cov,exact
    monkeypatch.setattr(workbench,'oracle',broken)
    with pytest.raises(ValueError,match='NOT_BIT_EXACT'):workbench.run(out)
    assert (out/'worker_stop.json').exists() and not (out/'manifest.json').exists()


def test_worker_budget_failure_cannot_publish_success(tmp_path,monkeypatch):
    out=setup(tmp_path,monkeypatch);original=policy.Budget.__exit__
    def fail(self,*args):original(self,*args);raise RuntimeError('STOP_RSS_OR_WALL_RESOURCE_POLICY')
    monkeypatch.setattr(policy.Budget,'__exit__',fail)
    with pytest.raises(RuntimeError,match='RESOURCE_POLICY'):workbench.run(out)
    assert (out/'worker_stop.json').exists() and not (out/'manifest.json').exists()


def test_worker_manifest_not_success_if_parent_cannot_confirm_actual_exit(tmp_path,monkeypatch):
    out=setup(tmp_path,monkeypatch);original=workbench.supervise
    def bad(*args):
        s=original(*args);s['own_child_exit_confirmed']=False;return s
    monkeypatch.setattr(workbench,'supervise',bad)
    with pytest.raises(ValueError,match='UNCONFIRMED'):workbench.run(out)
    assert (out/'status.json').exists()


@pytest.mark.parametrize('corruption',['none','fold0','fold1','truth','old_hits','extra','uncertified_features','changed_scope','alias_cap','rss_relaxed'])
def test_input_gate_checks_closed_queries_and_exact_source_certificate(tmp_path,monkeypatch,corruption):
    setup(tmp_path,monkeypatch);cfg=json.loads(workbench.CONFIG.read_text())
    folder=verify(cfg['part_manifest']).parent;m=json.loads(verify(cfg['part_manifest']).read_text())
    spec=dict(alias_start=5,row_start=0,row_stop=6);m['inputs']=dict(part=spec)
    if corruption=='uncertified_features':m['stats']['all_scalar_feature_texts_exact']=False
    if corruption=='changed_scope':m['stats']['address_fact_attributes_or_scope_changed']=True
    if corruption=='alias_cap':m['stats']['alias_or_proof_cap']=100
    write_json(folder/'manifest.json',m);cfg['part_manifest']=record(folder/'manifest.json')
    monkeypatch.setattr(workbench.backend,'load',lambda root:({},dict(parts=[{}]*49+[spec])))
    monkeypatch.setattr(workbench.backend,'directory',lambda *args:folder)
    monkeypatch.setattr(workbench.backend,'phase_record',lambda *args,**kwargs:m)
    rows=[dict(query_id=i,fold=3,crm_name='JEAN',crm_number='',crm_suffix='',crm_street='JEAN',crm_postcode='',crm_insee='') for i in range(128)]
    launch=dict(queries=rows,truth_opened=False,old_candidates_opened=False)
    if corruption=='fold0':rows[-1]['fold']=0
    if corruption=='fold1':rows[-1]['fold']=1
    if corruption=='truth':launch['truth_opened']=True
    if corruption=='old_hits':launch['old_candidates_opened']=True
    if corruption=='extra':rows[-1]['target_siret']='never a feature'
    if corruption=='rss_relaxed':cfg['resources']['max_rss_bytes']*=2
    path=Path(cfg['query_only_launch']['path']);write_json(path,launch);cfg['query_only_launch']=record(path)
    if corruption=='none':assert ORIGINAL_INPUTS(cfg)[-1]['query_id']==0
    else:
        with pytest.raises(ValueError):ORIGINAL_INPUTS(cfg)
