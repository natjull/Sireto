"""Small complete source, native CSR, fake pressure/supervision; not national."""
import json
from pathlib import Path
from types import SimpleNamespace

import duckdb
import numpy as np
import pytest

import scripts.compile_retrieval_v176_packed_addresses as workbench
import scripts.seal_retrieval_v175_address_parts as prior
from src.xgb_matcher import retrieval_memory_policy_v153 as policy
from src.xgb_matcher.retrieval_v20_workbench import record,verify,write_json
from src.xgb_matcher.retrieval_national_names_v138 import NationalNames,PackedTerms
from src.xgb_matcher.retrieval_coupled_blocks_v121 import channels
from src.xgb_matcher.retrieval_variants_v118 import normalized
from test_seal_retrieval_v175_address_parts import setup as source_setup,supervision,gate


def setup(tmp_path,monkeypatch):
    root,staged,old,p=source_setup(tmp_path,monkeypatch)
    l,_=prior.backend.load(root);l['implementation']={}
    prior.init(staged);prior.remaining(staged);prior.seal(staged)
    cfg=json.loads(workbench.CONFIG.read_text());m=json.loads((staged/'staging_manifest.json').read_text())
    cfg.update(staging=record(staged/'staging_manifest.json'),aliases=m['aliases'],postings=m['postings'],output_root=str(tmp_path))
    path=tmp_path/'v176_config.json';write_json(path,cfg);monkeypatch.setattr(workbench,'CONFIG',path)
    monkeypatch.setattr(workbench,'quiet',gate)
    def supervised(args,log,env,stop_path):
        output=Path(args[args.index('--output')+1])
        if args[4]=='worker':workbench.worker(output,int(args[args.index('--channel')+1]),int(args[args.index('--bucket')+1]))
        else:workbench.seal_worker(output)
        return supervision()
    monkeypatch.setattr(workbench,'supervise',supervised)
    return cfg,tmp_path/'packed',staged,root


def test_all_packs_exact_source_tokens_and_snapshot_match_scalar(tmp_path,monkeypatch):
    cfg,out,staged,root=setup(tmp_path,monkeypatch);workbench.init(out)
    assert not (out/'manifest.json').exists()
    workbench.remaining(out);workbench.seal(out)
    m=json.loads((out/'manifest.json').read_text());assert m['status']==workbench.STATUS
    assert m['aliases']==12 and len(m['access'])==256 and len(m['phase_authorities'])==256
    assert m['postings']==cfg['postings'] and all(m[k] is False for k in workbench.PROTECTED)
    assert not m['complete_retrieval_index'] and not m['ready_for_fold1']
    c=duckdb.connect();index=SimpleNamespace(c=c,buckets=64,names=12,access={})
    index.posting=lambda channel,token:NationalNames.posting(index,channel,token)
    texts=['RUE JEAN JAURES','JEAN JEAN JEAN','JEANNETTE','CH 390865069','390865069','AAAAAAA']*2
    try:
        for item in m['access']:
            index.access[item['channel'],item['bucket']]=PackedTerms({k.removesuffix('.npy'):np.load(verify(ref),mmap_mode='r') for k,ref in item['artifacts'].items()})
        for text in ('RUE JEAN JAURES','JEAN','CH 390865069','AAAAAAA',''):
            cov,exact,_=NationalNames.coverage(index,text);q=channels(text)
            expected=np.array([max(len(a&b)/len(a) if a else 0. for a,b in zip(q,channels(s))) for s in texts])
            np.testing.assert_array_equal(cov,expected)
            np.testing.assert_array_equal(exact,[bool(normalized(text)) and normalized(s)==normalized(text) for s in texts])
    finally:
        for access in index.access.values():
            for a in access.a.values():a._mmap.close()
        c.close()
    with pytest.raises(FileExistsError):workbench.init(out)
    with pytest.raises(FileExistsError):workbench.seal(out)
    with pytest.raises(ValueError,match='no retry'):workbench.worker(out,0,0)


@pytest.mark.parametrize('corruption',['aliases','posts','missing_feature','missing_audit','open_truth','partial','seal_stop','seal_exit'])
def test_no_compact_plan_from_incomplete_or_unprotected_source(tmp_path,monkeypatch,corruption):
    cfg,out,staged,root=setup(tmp_path,monkeypatch);path=staged/'staging_manifest.json';m=json.loads(path.read_text())
    if corruption=='aliases':m['aliases']-=1
    if corruption=='posts':m['postings']-=1
    if corruption=='missing_feature':m['features'].pop()
    if corruption=='missing_audit':m['audit_authorities'].pop()
    if corruption=='open_truth':m['truth_opened']=True
    if corruption=='partial':m['every_original_whole_alias_and_posting_retained']=False
    if corruption=='seal_stop':write_json(staged/'seal_stop.json',{'status':'STOP'})
    if corruption=='seal_exit':
        path2=staged/'seal/supervision.json';s=json.loads(path2.read_text());s['own_child_exit_confirmed']=False;write_json(path2,s)
    write_json(path,m);cfg['staging']=record(path);write_json(workbench.CONFIG,cfg)
    with pytest.raises(ValueError):workbench.init(out)
    assert not (out/'plan.json').exists() and not (out/'access').exists()


def test_global_epoch_drift_in_unconsumed_bucket_still_stops(tmp_path,monkeypatch):
    cfg,out,staged,root=setup(tmp_path,monkeypatch);workbench.init(out)
    _,p,e=workbench.load(out)
    refs=[x['artifact'] for b,items in p['buckets'].items() if b!='0' for x in items]
    assert refs;path=Path(refs[0]['path']);path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='EPOCH_DRIFT'):workbench.execute(out,0,0)
    assert not (out/'access').exists()


def test_only_relevant_bucket_is_rehashed_and_every_posting_unchanged(tmp_path,monkeypatch):
    cfg,out,staged,root=setup(tmp_path,monkeypatch);workbench.init(out);original=workbench.verify;seen=[]
    _,p,_=workbench.load(out);bucket=next(int(b) for b,items in p['buckets'].items() if items)
    def tracking(ref):
        if '/staging/term_bucket=' in ref['path']:seen.append(ref['path'])
        return original(ref)
    monkeypatch.setattr(workbench,'verify',tracking);workbench.execute(out,0,bucket)
    assert seen and all('/term_bucket='+str(bucket)+'/' in s for s in seen)
    m=workbench.phase_record(out,0,bucket)
    assert m['stats']['every_original_posting_and_token_reconstructed_exactly']
    assert m['stats']['no_posting_or_owner_cap'] and not (out/'manifest.json').exists()


def test_pack_failure_kept_and_never_restarted_or_published(tmp_path,monkeypatch):
    cfg,out,staged,root=setup(tmp_path,monkeypatch);workbench.init(out)
    def stop(*args):raise RuntimeError('STOP_RESOURCE_POLICY')
    monkeypatch.setattr(workbench,'pack',stop)
    with pytest.raises(RuntimeError):workbench.execute(out,0,0)
    assert (out/'access/channel0/bucket000/worker_stop.json').exists()
    with pytest.raises(ValueError,match='STOP_PACK'):workbench.remaining(out)
    assert not (out/'manifest.json').exists()


def test_no_worker_manifest_acceptance_without_actual_confirmed_exit(tmp_path,monkeypatch):
    cfg,out,staged,root=setup(tmp_path,monkeypatch);workbench.init(out);original=workbench.supervise
    def disappear(*args):
        s=original(*args);s['own_child_exit_confirmed']=False;return s
    monkeypatch.setattr(workbench,'supervise',disappear)
    with pytest.raises(ValueError,match='UNCONFIRMED'):workbench.execute(out,0,0)
    assert (out/'access/channel0/bucket000/status.json').exists()


@pytest.mark.parametrize('field,value',[('max_rss_bytes',2147483648),('min_system_free_percent',1),('required_pressure_level',2)])
def test_no_implicit_budget_relaxation(tmp_path,monkeypatch,field,value):
    cfg,out,staged,root=setup(tmp_path,monkeypatch);cfg['resources'][field]=value;write_json(workbench.CONFIG,cfg)
    with pytest.raises(ValueError,match='resource policy'):workbench.init(out)
    assert not out.exists()
