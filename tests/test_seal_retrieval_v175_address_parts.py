"""Entire tiny official-style source, fake runtime pressure/exits, no national claim."""
import json
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import scripts.seal_retrieval_v175_address_parts as workbench
from src.xgb_matcher.retrieval_address_access_v152 import SCHEMA
from src.xgb_matcher.retrieval_address_channels_v166 import stage_addresses
from src.xgb_matcher import retrieval_memory_policy_v153 as policy
from src.xgb_matcher.retrieval_v20_workbench import record,verify,write_json
from test_resume_retrieval_v153_addresses import gate


def supervision():
    return dict(policy_id=policy.POLICY,limits=policy.LIMITS,child_pid=1,exit_code=0,
        own_child_exit_confirmed=True,peak_rss_bytes=1,minimum_system_free_percent=73,
        sample_count=2,wall_seconds=0.,swapout_pages_delta=0,global_swap_is_worker_attributed=False)


def resource_report():
    return dict(policy_id=policy.POLICY,limits=policy.LIMITS,pass_resource_policy=True,
        global_swap_is_worker_attributed=False,wall_seconds=0.,peak_rss_bytes=1,
        minimum_system_free_percent=73,pressure_levels=[1],swapout_pages_delta=0,sample_count=2)


def setup(tmp_path,monkeypatch):
    root=tmp_path/'source';root.mkdir();out=tmp_path/'output'
    cfg=json.loads(workbench.CONFIG.read_text());cfg.update(parts=2,aliases=12,facts=20,sites=13,
        original_observations=25,lexical_observations=21,output_root=str(tmp_path))
    text=['RUE JEAN JAURES','JEAN JEAN JEAN','JEANNETTE','CH 390865069','390865069','AAAAAAA']
    specs=[]
    for i in range(2):
        dest=workbench.backend.directory(root,'part',i);dest.mkdir(parents=True)
        table=pa.Table.from_arrays([pa.array(np.arange(6,dtype=np.uint32)),pa.array(text)],schema=SCHEMA)
        c=duckdb.connect()
        try:stats=stage_addresses(c,table,dest,i*6,0,64)
        finally:c.close()
        files=[Path(p) for p in stats.pop('files')];files.append(dest/'features.parquet')
        counts={str(b):{str(ch):0 for ch in range(4)} for b in range(64)}
        for f in files:
            if f.name=='features.parquet':continue
            b=f.parent.name.removeprefix('term_bucket=')
            values=pq.ParquetFile(f).read(columns=['channel'])['channel'].to_numpy()
            for ch in range(4):counts[b][str(ch)]=int(np.count_nonzero(values==ch))
        stats['bucket_channel_counts']=counts
        spec=dict(part_id=i,alias_start=i*6,row_start=0,row_stop=6);specs.append(spec)
        write_json(dest/'preflight.json',gate());write_json(dest/'supervision.json',supervision())
        m=dict(stats=stats,artifacts={str(f.relative_to(dest)):record(f) for f in files},
            inputs=dict(part=spec),preflight=record(dest/'preflight.json'),output_bytes=sum(f.stat().st_size for f in files))
        write_json(dest/'manifest.json',m)
    for name in ('source_store','dictionary','catalog'):write_json(root/(name+'.json'),{'fixture':True})
    p=dict(parts=specs,resumed_successful_parts=[],source_epochs=[],
        **{k:cfg[k] for k in workbench.COUNT_FIELDS},
        **{k:record(root/(name+'.json')) for k,name in [('source_store','source_store'),('dictionary','dictionary'),('catalog','catalog')]})
    write_json(root/'plan.json',p);write_json(root/'launch.json',{'fixture':True})
    cfg.update(source_launch=record(root/'launch.json'),source_plan=record(root/'plan.json'))
    config=tmp_path/'config.json';write_json(config,cfg);monkeypatch.setattr(workbench,'CONFIG',config)
    l=dict(plan=record(root/'plan.json'),config=dict(term_buckets=64,max_total_bytes=100000000))
    monkeypatch.setattr(workbench.backend,'load',lambda root:(l,p))
    monkeypatch.setattr(workbench.backend,'revalidate',lambda *args:None)
    def original_phase(root,phase,index,context=None):
        dest=workbench.backend.directory(root,phase,index);m=json.loads((dest/'manifest.json').read_text())
        for ref in m['artifacts'].values():verify(ref)
        return m
    monkeypatch.setattr(workbench.backend,'phase_record',original_phase)
    monkeypatch.setattr(workbench,'quiet',gate)
    monkeypatch.setattr(policy,'snapshot',lambda:dict(swapouts=42,free_percent=73,pressure_level=1))
    def supervised(args,log,env,stop_path):
        folder=Path(args[args.index('--output')+1])
        if args[4]=='worker':workbench.worker(folder,int(args[args.index('--index')+1]))
        else:workbench.seal_worker(folder)
        return supervision()
    monkeypatch.setattr(workbench,'supervise',supervised)
    def audit(root,tag,kind):
        dest=root/'source_audits'/tag/kind
        return record(dest/'manifest.json'),json.loads((dest/'manifest.json').read_text())
    def run_audits(root,tag):
        for kind in ('prefix','reverse'):
            dest=root/'source_audits'/tag/kind;dest.mkdir(parents=True)
            write_json(dest/'supervision.json',supervision());write_json(dest/'manifest.json',p)
    monkeypatch.setattr(workbench.backend,'audit_record',audit)
    monkeypatch.setattr(workbench.backend,'run_audits',run_audits)
    return root,out,cfg,p


def test_all_source_parts_feature_ids_posting_counts_and_final_audits_required(tmp_path,monkeypatch):
    root,out,cfg,p=setup(tmp_path,monkeypatch);workbench.init(out)
    assert not (out/'staging_manifest.json').exists()
    workbench.remaining(out);workbench.seal(out)
    m=json.loads((out/'staging_manifest.json').read_text())
    assert m['status']==workbench.STATUS and m['aliases']==12 and len(m['features'])==2
    assert m['every_original_whole_alias_and_posting_retained']
    assert len(m['phase_authorities'])==2 and len(m['audit_authorities'])==1
    assert m['postings']==sum(sum(sum(x['channel_counts'].values()) for x in items) for items in m['buckets'].values())
    assert set(m['sealed_source_audits'])=={'prefix','reverse'}
    assert not m['complete_retrieval_index'] and not m['ready_for_fold1']
    assert all(m[k] is False for k in workbench.PROTECTED)
    assert json.loads((out/'seal'/'supervision.json').read_text())['own_child_exit_confirmed']
    with pytest.raises(FileExistsError):workbench.init(out)
    with pytest.raises(FileExistsError):workbench.seal(out)
    with pytest.raises(ValueError,match='no retry'):workbench.worker(out,0)


@pytest.mark.parametrize('corruption',['hole','duplicate','short_denominator','rss','missing_part','source_stop'])
def test_bad_full_source_never_launches_audit_batch(tmp_path,monkeypatch,corruption):
    root,out,cfg,p=setup(tmp_path,monkeypatch)
    if corruption=='hole':p['parts'][1]['alias_start']+=1
    if corruption=='duplicate':p['parts'][1]['part_id']=0
    if corruption=='short_denominator':p['aliases']-=1
    if corruption=='rss':cfg['resources']['max_rss_bytes']*=2;write_json(workbench.CONFIG,cfg)
    if corruption=='missing_part':(root/'parts'/'part00001'/'manifest.json').rename(root/'missing_manifest.json')
    if corruption=='source_stop':write_json(root/'parts'/'part00001'/'worker_stop.json',{'status':'STOP'})
    with pytest.raises((ValueError,FileNotFoundError)):workbench.init(out)
    assert not (out/'batches').exists() and not (out/'staging_manifest.json').exists()


def test_changed_source_artifact_after_init_not_certified(tmp_path,monkeypatch):
    root,out,cfg,p=setup(tmp_path,monkeypatch);workbench.init(out)
    feature=root/'parts'/'part00001'/'features.parquet';feature.write_bytes(feature.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='EPOCH_DRIFT'):workbench.remaining(out)
    assert not (out/'staging_manifest.json').exists() and (out/'batches'/'batch0000'/'worker_stop.json').exists()


@pytest.mark.parametrize('corruption',['counts','scope','cap','scalar','feature_ids'])
def test_whole_channel_reread_and_exact_original_semantics_checked(tmp_path,monkeypatch,corruption):
    root,out,cfg,p=setup(tmp_path,monkeypatch);dest=root/'parts'/'part00001';path=dest/'manifest.json';m=json.loads(path.read_text())
    if corruption=='counts':m['stats']['bucket_channel_counts']['0']['0']+=1
    if corruption=='scope':m['stats']['address_fact_attributes_or_scope_changed']=True
    if corruption=='cap':m['stats']['alias_or_proof_cap']=100
    if corruption=='scalar':m['stats']['all_scalar_channel_sets_exact']=False
    if corruption=='feature_ids':
        f=dest/'features.parquet';table=pq.ParquetFile(f).read()
        ids=table['alias_id'].to_numpy().copy();ids[-1]=ids[0]
        pq.write_table(table.set_column(0,'alias_id',pa.array(ids)),f);m['artifacts']['features.parquet']=record(f)
    write_json(path,m);workbench.init(out)
    with pytest.raises(ValueError):workbench.remaining(out)
    assert not (out/'staging_manifest.json').exists()


def test_batch_budget_failure_no_success_manifest(tmp_path,monkeypatch):
    root,out,cfg,p=setup(tmp_path,monkeypatch);workbench.init(out);original=policy.Budget.__exit__
    def fail(self,*args):original(self,*args);raise RuntimeError('STOP_RESOURCE_POLICY')
    monkeypatch.setattr(policy.Budget,'__exit__',fail)
    with pytest.raises(RuntimeError,match='RESOURCE_POLICY'):workbench.remaining(out)
    assert not (out/'batches'/'batch0000'/'manifest.json').exists()


def test_staging_manifest_without_confirmed_seal_child_exit_rejected(tmp_path,monkeypatch):
    root,out,cfg,p=setup(tmp_path,monkeypatch);workbench.init(out);workbench.remaining(out)
    original=workbench.supervise
    def disappear(*args):
        s=original(*args);s['own_child_exit_confirmed']=False;return s
    monkeypatch.setattr(workbench,'supervise',disappear)
    with pytest.raises(ValueError,match='UNCONFIRMED'):workbench.seal(out)
    assert (out/'seal_stop.json').exists()
    with pytest.raises(ValueError,match='STOP_SEAL'):workbench.context(out)


def test_no_global_rehash_of_artifacts_in_init_epoch_capture(tmp_path,monkeypatch):
    path=tmp_path/'feature.parquet';path.write_bytes(b'not yet certified')
    ref=dict(path=str(path),sha256='a'*64,size_bytes=path.stat().st_size)
    # INIT records claimed source hashes and actual epochs only. This wrong
    # hash would be rejected in full by the batch, not globally hashed here.
    items=workbench.epochs([ref]);assert items[0]['reference']==ref
    with pytest.raises(ValueError):verify(ref)


@pytest.mark.parametrize('corruption',['feature','counts','empty','bytes','missing_row','open_truth'])
def test_audit_result_cannot_substitute_fields_or_filter_denominator(tmp_path,monkeypatch,corruption):
    root,out,cfg,p=setup(tmp_path,monkeypatch);workbench.init(out);workbench.remaining(out)
    path=out/'batches'/'batch0000'/'manifest.json';m=json.loads(path.read_text())
    if corruption=='feature':m['rows'][0]['feature']['artifact']=m['rows'][1]['feature']['artifact']
    if corruption=='counts':m['rows'][0]['posts'][0]['channel_counts']['0']+=1
    if corruption=='empty':m['rows'][0]['feature_empty']+=1
    if corruption=='bytes':m['rows'][0]['output_bytes']+=1
    if corruption=='missing_row':m['rows'].pop()
    if corruption=='open_truth':m['truth_opened']=True
    write_json(path,m)
    with pytest.raises(ValueError):workbench.batch_record(out,0)
    assert not (out/'staging_manifest.json').exists()


def test_incomplete_source_audit_never_publishes_full_stage(tmp_path,monkeypatch):
    root,out,cfg,p=setup(tmp_path,monkeypatch);workbench.init(out);workbench.remaining(out)
    def incomplete(*args):raise ValueError('UNFINISHED_ALL_RAW_SOURCE_AUDITS')
    monkeypatch.setattr(workbench.backend,'run_audits',incomplete)
    with pytest.raises(ValueError,match='UNFINISHED'):workbench.seal(out)
    assert (out/'seal_stop.json').exists() and not (out/'staging_manifest.json').exists()
