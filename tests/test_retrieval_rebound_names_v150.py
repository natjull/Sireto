import json
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.compile_retrieval_v149_name_rebind as original
import src.xgb_matcher.retrieval_rebound_names_v150 as module
from src.xgb_matcher.retrieval_proof_bridge_v137 import PROTECTED,scatter_name_facts,one_hop_names
from src.xgb_matcher.retrieval_v20_workbench import record,verify,write_json
from test_compile_retrieval_v149_name_rebind import setup as original_setup,start
from test_retrieval_name_rebind_v149 import keys,facts,old_bridge


def setup(tmp_path,monkeypatch):
    cfg,out=original_setup(tmp_path,monkeypatch);bridge=json.loads(verify(cfg['bridge']).read_text());cat=json.loads(verify(cfg['catalog']).read_text())
    def arrays(prefix,values):
        result={}
        for k,v in values.items():
            p=tmp_path/(prefix+k+'.npy');np.save(p,v,allow_pickle=False);result[k+'.npy']=record(p)
        return result
    pp=np.array([0,2,3],dtype=np.uint32)
    cat['artifacts']=arrays('v150_new_',dict(new_site_parent_positions=pp))
    cp=tmp_path/'v150_catalog.json';write_json(cp,cat);cfg['catalog']=record(cp)
    bridge['graph']=arrays('v150_graph_',dict(site_parent_positions=np.array([0,1],dtype=np.uint32),
        graph_site_positions=np.array([0,1],dtype=np.uint32),graph_parent_positions=np.array([0,1],dtype=np.uint32),
        predecessor_graph_positions=np.array([0],dtype=np.uint32),successor_graph_positions=np.array([1],dtype=np.uint32)))
    graph=dict(status='SEALED_V133_COMPLETE_ALL_RAW_OFFICIAL_SUCCESSION',launch=cat['launch'],
        resources=cat['resources'],stats=dict(endpoints=2,observations=1),complete_retrieval_index=False,ready_for_fold1=False,
        artifacts=arrays('v150_rawgraph_',dict(continuity_codes=np.array([2],dtype=np.uint8))),**{k:False for k in PROTECTED})
    gp=tmp_path/'v150_graph.json';write_json(gp,graph);bridge['sources']['graph']=record(gp)
    bridge['stats'].update(graph_endpoints=2,graph_observations=1)
    bp=tmp_path/'v150_bridge.json';write_json(bp,bridge);cfg['bridge']=record(bp);write_json(original.CONFIG,cfg)
    start(out,cfg);original.worker(out)
    k=keys();k.new_site=np.array([100002,200001,900001],dtype=np.uint64);k.new_parent=np.array([2,9],dtype=np.uint32);k.references=()
    monkeypatch.setattr(module,'AppendKeys',lambda ref:k)
    return record(out/'manifest.json'),out


class Index:
    def __init__(self,source):self.names=6;self.source=source
    def coverage(self,text,**unused):
        return np.array([.2,1.,.8,.5,.3,.9]),np.array([False,True,False,False,False,False]),dict(fixture_only=True)


def test_actual_loader_query_scatter_relation_and_snapshot_match_every_fact_oracle(tmp_path,monkeypatch):
    ref,_=setup(tmp_path,monkeypatch);reader=module.ReboundNames(ref)
    try:
        snapshot,trace=reader.query(Index(reader.source),'fixture',chunk=2)
        src=facts();old=old_bridge();pp=np.array([0,2,2,3,module.MISSING,1,0],dtype=np.uint32)
        pc=np.zeros(4);pe=np.zeros(4,dtype=bool);sc=np.zeros(2);se=np.zeros(2,dtype=bool)
        c,e,_=Index(reader.source).coverage('fixture')
        scatter_name_facts(c,e,src['name_fact_offsets'],src['subject_code'],src['current_exact'],pp,old['fact_site_positions'],pc,pe,sc,se,3)
        relation=one_hop_names(pc,sc,reader.graph['graph_site_positions'],reader.graph['graph_parent_positions'],
            reader.graph['predecessor_graph_positions'],reader.graph['successor_graph_positions'],reader.graph['continuity_codes'])
        for a,b in ((snapshot.snapshot.parent_cov,pc),(snapshot.snapshot.parent_exact,pe),(snapshot.snapshot.previous.site_cov,sc),
            (snapshot.snapshot.previous.site_exact,se),(snapshot.snapshot.previous.relation_cov,relation)):
            np.testing.assert_array_equal(a,b);assert not a.flags.writeable
        n,ne=snapshot.gather(np.arange(5,dtype=np.uint32))
        assert n.tolist()==[.9,.45,.2,1.,.8] and ne.tolist()==[0.,0.,0.,1.,0.]
        assert trace['references_processed']==7 and trace['appended_parent_proofs']==2
        assert trace['appended_parent_exact']==1 and trace['appended_sites_with_name_proofs']==3
        assert trace['all_sealed_parent_patches_applied'] and trace['aliases_released_before_relations']
        assert not trace['recall_measured'] and not trace['initial100_emitted']
        reader.verify_after()
    finally:reader.close()
    with pytest.raises(ValueError,match='Closed'):snapshot.gather(np.arange(5,dtype=np.uint32))
    reader.close() # Idempotent, no double mmap access.


@pytest.mark.parametrize('field',['fold0_opened','fold1_opened','truth_opened','crm_opened','model_fit'])
def test_complete_join_source_closed_flags_required(tmp_path,monkeypatch,field):
    ref,out=setup(tmp_path,monkeypatch);m=json.loads(verify(ref).read_text());m[field]=True;write_json(out/'manifest.json',m)
    with pytest.raises(ValueError,match='Protected'):module.ReboundNames(record(out/'manifest.json'))


def test_missing_bucket_and_semantically_wrong_patch_are_not_success(tmp_path,monkeypatch):
    ref,out=setup(tmp_path,monkeypatch);m=json.loads(verify(ref).read_text());saved=json.loads(json.dumps(m));m['buckets']=[];write_json(out/'manifest.json',m)
    with pytest.raises(ValueError,match='Whole unchanged'):module.ReboundNames(record(out/'manifest.json'))
    m=saved;path=verify(m['buckets'][0]['artifacts']['parent_positions.npy']);np.save(path,np.array([2,2,2],dtype=np.uint32),allow_pickle=False)
    m['buckets'][0]['artifacts']['parent_positions.npy']=record(path);write_json(out/'manifest.json',m)
    with pytest.raises(ValueError,match='exact official UL'):module.ReboundNames(record(out/'manifest.json'))


def test_wrong_alias_source_and_query_cost_stop_without_partial_snapshot(tmp_path,monkeypatch):
    ref,_=setup(tmp_path,monkeypatch);reader=module.ReboundNames(ref)
    try:
        index=Index(dict(path='different',sha256='0'*64))
        with pytest.raises(ValueError,match='identical alias'):reader.query(index,'fixture')
        index.source=reader.source
        with pytest.raises(TimeoutError,match='QUERY_COST'):reader.query(index,'fixture',max_seconds=1e-12)
        with pytest.raises(ValueError):reader.query(index,'fixture',chunk=0)
    finally:reader.close()


def test_source_or_patch_drift_after_query_is_detected_before_sealing(tmp_path,monkeypatch):
    ref,_=setup(tmp_path,monkeypatch);reader=module.ReboundNames(ref)
    try:
        reader.query(Index(reader.source),'fixture')
        path=verify(reader.items[0]['artifacts']['fact_rows.npy'])
        with path.open('ab') as f:f.write(b'changed')
        with pytest.raises(ValueError,match='Authority drift'):reader.verify_after()
    finally:reader.close()


def test_original_direct_site_alias_never_becomes_sibling_proof(tmp_path,monkeypatch):
    ref,_=setup(tmp_path,monkeypatch);reader=module.ReboundNames(ref)
    class OnlySite(Index):
        def coverage(self,*unused,**kwargs):
            return np.array([0.,0.,0.,0.,0.,1.]),np.array([False,False,False,False,False,True]),dict(fixture_only=True)
    try:
        snapshot,trace=reader.query(OnlySite(reader.source),'fixture')
        n,ne=snapshot.gather(np.arange(5,dtype=np.uint32))
        assert n.tolist()==[1.,.5,0.,0.,0.] and ne.tolist()==[1.,0.,0.,0.,0.]
        assert trace['appended_parent_proofs']==0 and trace['appended_sites_with_name_proofs']==0
    finally:reader.close()


def test_graph_source_closed_and_zero_swap_certificates_not_ignored(tmp_path,monkeypatch):
    ref,out=setup(tmp_path,monkeypatch);m=json.loads(verify(ref).read_text());b=json.loads(verify(m['sources']['bridge']).read_text())
    p=verify(b['sources']['graph']);g=json.loads(p.read_text());g['fold1_opened']=True;write_json(p,g)
    # The source hash drift alone is sufficient to refuse; never open fold1.
    with pytest.raises(ValueError,match='Authority drift'):module.ReboundNames(record(out/'manifest.json'))
