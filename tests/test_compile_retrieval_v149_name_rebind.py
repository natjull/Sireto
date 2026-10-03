import json

import numpy as np
import pytest

import scripts.compile_retrieval_v149_name_rebind as workbench
from src.xgb_matcher.retrieval_v20_workbench import Budget,record,write_json
from test_retrieval_name_rebind_v149 import keys,facts,old_bridge


def gate(*unused):
    return dict(pass_system_quiet_policy=True,worker_swap_budget_changed=False,final_snapshot=dict(swapouts=42,free_percent=80))


def setup(tmp_path,monkeypatch):
    protected={k:False for k in workbench.PROTECTED};budget=dict(pass_resource_policy=True,swapout_pages_delta=0)
    launch=tmp_path/'source_launch.json';write_json(launch,dict(implementation={}))
    def seal(name,m):
        p=tmp_path/(name+'.json');write_json(p,m);return record(p)
    def arrays(prefix,values):
        result={}
        for k,v in values.items():
            p=tmp_path/(prefix+k+'.npy');np.save(p,v,allow_pickle=False);result[k+'.npy']=record(p)
        return result
    b=dict(bucket=0,names=6,facts=7,name_start=0,fact_start=0,artifacts=arrays('source_',facts()))
    phase_stats=dict(names=6,facts=7,every_original_fact_reconstructed_exactly=True,every_raw_name_has_original_owner=True,
        owners_outside_candidate_catalog_preserved=True,identifiers_as_features=False,site_cartesian_product_materialized=False,owner_or_name_or_period_cap=None)
    phase=seal('owner_phase',dict(status='SEALED_V132_EXACT_ALL_OWNER_BUCKET',launch=record(launch),bucket=b,
        stats=phase_stats,artifacts=b['artifacts'],resources=budget,complete_retrieval_index=False,**protected))
    owners=seal('owners',dict(status='SEALED_V132_COMPLETE_ALL_OFFICIAL_NAME_OWNER_CSR',launch=record(launch),
        complete_retrieval_index=False,ready_for_fold1=False,names=6,facts=7,buckets=[b],phase_authorities=[phase],
        source=seal('original_names',dict(fixture=True)),every_owner_fact_reconstructed_exactly=True,**protected))
    oldcat=seal('old_catalog',dict(fixture=True))
    item=dict(bucket=b,artifacts=arrays('old_',old_bridge()),stats=dict(facts=7,site_facts=2,unbound_parent_facts=4,
        every_original_fact_reconstructed_exactly=True,every_site_fact_admitted=True,unbound_parent_observations_preserved=True,
        name_or_owner_or_member_cap=None,identifiers_as_features=False,site_cartesian_product_materialized=False))
    bridge=seal('bridge',dict(status='SEALED_V137_COMPLETE_OFFICIAL_PROOF_POSITION_BRIDGE',launch=record(launch),resources=budget,
        complete_retrieval_index=False,ready_for_fold1=False,sources=dict(owners=owners,catalog=oldcat),buckets=[item],
        stats=dict(sites=2,parents=2,names=6,facts=7,site_facts=2,unbound_parent_facts=4),**protected))
    cat=seal('catalog',dict(status='SEALED_V141_APPEND_ALL_V140_OFFICIAL_SITE_IDENTITIES',launch=record(launch),resources=budget,
        complete_retrieval_index=False,ready_for_fold1=False,previous_catalog=oldcat,
        artifacts=arrays('new_',dict(new_site_parent_positions=np.array([2,0,3],dtype=np.uint32))),
        stats=dict(sites=5,parents=4,previous_sites=2,previous_parents=2,additional_sites=3,
            every_previous_site_and_parent_position_unchanged=True,every_new_site_admitted_once=True),**protected))
    cfg=json.loads(workbench.CONFIG.read_text());cfg.update(bridge=bridge,catalog=cat,expected_names=6,expected_facts=7,
        expected_site_facts=2,expected_previous_unbound_facts=4,expected_sites=5,expected_parents=4,expected_buckets=1,output_root=str(tmp_path))
    c=tmp_path/'config.json';write_json(c,cfg);monkeypatch.setattr(workbench,'CONFIG',c)
    monkeypatch.setattr(Budget,'swap',staticmethod(lambda:42));monkeypatch.setattr(workbench,'quiet',gate)
    k=keys();k.references=();monkeypatch.setattr(workbench,'AppendKeys',lambda ref:k)
    return cfg,tmp_path/'out'


def start(out,cfg):
    out.mkdir();write_json(out/'preflight.json',gate())
    write_json(out/'launch.json',dict(config=cfg,preflight=record(out/'preflight.json'),implementation=workbench.implementation(),**{k:False for k in workbench.PROTECTED}))


def test_whole_source_worker_emits_exact_sparse_patches_and_seals_after_budget(tmp_path,monkeypatch):
    cfg,out=setup(tmp_path,monkeypatch);start(out,cfg);workbench.worker(out)
    m=json.loads((out/'manifest.json').read_text())
    assert m['status']==workbench.STATUS and m['resources']['pass_resource_policy']
    assert m['stats']['facts']==7 and m['stats']['recovered_parent_facts']==3
    assert m['stats']['remaining_unbound_parent_facts']==1 and m['stats']['recovered_parents']==2
    assert m['stats']['appended_sites_with_recovered_parent_proofs']==2
    assert np.load(out/'bucket000/fact_rows.npy').tolist()==[1,2,3]
    assert np.load(out/'bucket000/parent_positions.npy').tolist()==[2,2,3]
    assert all(m[k] is False for k in workbench.PROTECTED) and not m['complete_retrieval_index']
    with pytest.raises(ValueError,match='no retry'):workbench.worker(out)
    with pytest.raises(FileExistsError):workbench.run(out)


@pytest.mark.parametrize('flag',['fold0_opened','fold1_opened','truth_opened','crm_opened','model_fit'])
def test_closed_sources_are_checked_before_quiet_or_root(tmp_path,monkeypatch,flag):
    cfg,out=setup(tmp_path,monkeypatch);p=workbench.verify(cfg['bridge']);m=json.loads(p.read_text());m[flag]=True;write_json(p,m)
    cfg['bridge']=record(p);write_json(workbench.CONFIG,cfg)
    with pytest.raises(ValueError,match='Protected'):workbench.run(out)
    assert not out.exists()


def test_missing_bucket_and_resource_policy_change_are_never_reinterpreted(tmp_path,monkeypatch):
    cfg,out=setup(tmp_path,monkeypatch);cfg['expected_buckets']=2;write_json(workbench.CONFIG,cfg)
    with pytest.raises(ValueError,match='Every original'):workbench.run(out)
    assert not out.exists()
    cfg['expected_buckets']=1;cfg['resources']['max_swapout_pages_delta']=1;write_json(workbench.CONFIG,cfg)
    with pytest.raises(ValueError,match='unchanged zero swap'):workbench.run(out)
    assert not out.exists()


def test_original_owner_phase_resource_certificate_required(tmp_path,monkeypatch):
    cfg,out=setup(tmp_path,monkeypatch);b,_,s=workbench.authority(cfg)
    p=workbench.verify(s['phase_authorities'][0]);phase=json.loads(p.read_text());phase['resources']['swapout_pages_delta']=7;write_json(p,phase)
    # Rehash enclosing references to ensure the semantic check, not merely SHA drift, catches this.
    sp=workbench.verify(b['sources']['owners']);s['phase_authorities'][0]=record(p);write_json(sp,s)
    bp=workbench.verify(cfg['bridge']);b['sources']['owners']=record(sp);write_json(bp,b);cfg['bridge']=record(bp)
    with pytest.raises(ValueError,match='bucket resource'):workbench.authority(cfg)


def test_hash_drift_and_size_stop_never_publish_success(tmp_path,monkeypatch):
    cfg,out=setup(tmp_path,monkeypatch);cfg['max_output_bytes']=1;start(out,cfg)
    with pytest.raises(RuntimeError,match='OUTPUT_SIZE'):workbench.worker(out)
    assert (out/'worker_stop.json').exists() and not (out/'manifest.json').exists()
    with pytest.raises(ValueError,match='no retry'):workbench.worker(out)


def test_resource_failure_after_complete_audit_still_prevents_success(tmp_path,monkeypatch):
    cfg,out=setup(tmp_path,monkeypatch);start(out,cfg);previous_exit=Budget.__exit__
    def fail(self,*args):
        previous_exit(self,*args);raise RuntimeError('STOP_RESOURCE_POLICY')
    monkeypatch.setattr(Budget,'__exit__',fail)
    with pytest.raises(RuntimeError,match='STOP_RESOURCE_POLICY'):workbench.worker(out)
    assert (out/'worker_stop.json').exists() and not (out/'manifest.json').exists()


def test_used_original_array_drift_is_stop(tmp_path,monkeypatch):
    cfg,out=setup(tmp_path,monkeypatch);b,_,_=workbench.authority(cfg)
    p=workbench.verify(b['buckets'][0]['bucket']['artifacts']['scope_code.npy'])
    with p.open('ab') as f:f.write(b'changed')
    start(out,cfg)
    with pytest.raises(ValueError,match='Authority drift'):workbench.worker(out)
    assert not (out/'manifest.json').exists()


def test_output_outside_configured_root_cannot_start(tmp_path,monkeypatch):
    cfg,out=setup(tmp_path,monkeypatch);cfg['output_root']=str(tmp_path/'restricted');write_json(workbench.CONFIG,cfg)
    with pytest.raises(ValueError,match='CATNAT-only'):workbench.run(out)
    assert not out.exists()
