from types import SimpleNamespace

import numpy as np
import pytest

from src.xgb_matcher.retrieval_append_catalog_v141 import append_positions
from src.xgb_matcher.retrieval_proof_bridge_v137 import CatalogKeys,MISSING,scatter_name_facts
from src.xgb_matcher.retrieval_national_names_v138 import scatter_positive
from src.xgb_matcher.retrieval_name_rebind_v149 import rebind,patched_positions,ReboundNameSnapshot


def keys():
    old=object.__new__(CatalogKeys)
    old.site=np.array([100001,300001],dtype=np.uint64);old.extra_site=np.empty(0,dtype=np.uint64)
    old.parent=np.array([1,3],dtype=np.uint32);old.extra_parent=np.empty(0,dtype=np.uint32)
    old.sites=2;old.parents=2
    new=np.array([2,9],dtype=np.uint32)
    def reconstruct(pos,grain):
        assert grain=='parent'
        return np.array([1,3,2,9],dtype=np.uint32)[pos]
    return SimpleNamespace(previous=old,parents=4,sites=5,
        parent_positions=lambda values:append_positions(2,new,values,old.parent_positions(values)),reconstruct=reconstruct)


def facts():
    return dict(parent_key=np.array([1,2,2,9,4,3,1],dtype=np.uint32),
        owner_key=np.array([1,2,2,9,4,300001,100001],dtype=np.uint64),
        subject_code=np.array([1,1,1,1,1,2,2],dtype=np.uint8),scope_code=np.array([1,1,2,2,1,3,3],dtype=np.uint8),
        current_exact=np.array([True,False,True,False,True,True,True]),
        name_fact_offsets=np.array([0,1,3,4,5,6,7],dtype=np.uint64))


def old_bridge():
    return dict(fact_parent_positions=np.array([0,MISSING,MISSING,MISSING,MISSING,1,0],dtype=np.uint32),
        fact_site_positions=np.array([MISSING,MISSING,MISSING,MISSING,MISSING,1,0],dtype=np.uint32))


def test_all_original_facts_audited_only_missing_parent_identity_recovered():
    src=facts();old=old_bridge();before=old['fact_parent_positions'].copy()
    r,p,s=rebind(keys(),src,old,2)
    assert r.tolist()==[1,2,3] and p.tolist()==[2,2,3]
    assert s['facts']==7 and s['site_facts']==2 and s['previous_unbound_parent_facts']==4
    assert s['remaining_unbound_parent_facts']==1 and s['recovered_parent_facts']==3
    assert s['recovered_parent_positions']==[2,3] and s['every_original_fact_audited']
    patched=patched_positions(before,r,p,2,4)
    assert patched.tolist()==[0,2,2,3,MISSING,1,0]
    np.testing.assert_array_equal(old['fact_parent_positions'],before)
    assert not s['site_alias_lifted_to_parent'] and s['name_or_owner_or_member_cap'] is None


@pytest.mark.parametrize('corrupt',['scope','owner','bound_identity','site_identity','hidden_old','dtype','missing_site'])
def test_all_fact_grains_and_old_positions_are_checked_not_only_recovered(corrupt):
    src=facts();old=old_bridge()
    if corrupt=='scope':src['scope_code'][6]=1
    if corrupt=='owner':src['owner_key'][2]=1
    if corrupt=='bound_identity':old['fact_parent_positions'][5]=0
    if corrupt=='site_identity':old['fact_site_positions'][6]=1
    if corrupt=='hidden_old':old['fact_parent_positions'][0]=MISSING
    if corrupt=='dtype':old['fact_parent_positions']=old['fact_parent_positions'].astype(np.int64)
    if corrupt=='missing_site':old['fact_site_positions'][6]=MISSING
    with pytest.raises(ValueError):rebind(keys(),src,old,2)


def test_no_intersection_and_empty_fact_bucket_are_valid_evidence():
    src=facts();src['parent_key'][1:4]=8;src['owner_key'][1:4]=8
    r,p,s=rebind(keys(),src,old_bridge(),3)
    assert len(r)==len(p)==0 and s['remaining_unbound_parent_facts']==4
    src={k:v[:0] for k,v in facts().items() if k not in ('current_exact','name_fact_offsets')}
    old={k:v[:0] for k,v in old_bridge().items()}
    r,p,s=rebind(keys(),src,old)
    assert s['facts']==0 and s['recovered_parent_facts']==0 and len(r)==len(p)==0


def test_no_cap_on_shared_name_or_many_original_parent_facts():
    n=10001;src={k:np.repeat(v[1:2],n) for k,v in facts().items() if k not in ('current_exact','name_fact_offsets')}
    old={k:np.full(n,MISSING,dtype=np.uint32) for k in old_bridge()}
    r,p,s=rebind(keys(),src,old,17)
    assert r.tolist()==list(range(n)) and np.all(p==2) and s['recovered_parent_facts']==n


@pytest.mark.parametrize('rows,pos',[([2,1],[2,2]),([1,1],[2,2]),([0],[2]),([1],[1]),([1],[4]),([7],[2])])
def test_sparse_overlay_cannot_modify_old_fact_duplicate_or_hide_new_position(rows,pos):
    with pytest.raises(ValueError):patched_positions(old_bridge()['fact_parent_positions'],np.array(rows,dtype=np.uint32),np.array(pos,dtype=np.uint32),2,4)


def test_actual_v138_scatter_matches_full_fact_oracle_with_original_current_flags():
    src=facts();old=old_bridge();r,p,s=rebind(keys(),src,old)
    patched=dict(old,fact_parent_positions=patched_positions(old['fact_parent_positions'],r,p,2,4))
    cov=np.array([.2,1.,.8,.5,.3,.9]);exact=np.array([False,True,False,False,False,False])
    pc=np.zeros(4);pe=np.zeros(4,dtype=bool);sc=np.zeros(2);se=np.zeros(2,dtype=bool)
    trace=scatter_positive(cov,exact,src,patched,pc,pe,sc,se,2)
    epc=np.zeros(4);epe=np.zeros(4,dtype=bool);esc=np.zeros(2);ese=np.zeros(2,dtype=bool)
    scatter_name_facts(cov,exact,src['name_fact_offsets'],src['subject_code'],src['current_exact'],
        patched['fact_parent_positions'],patched['fact_site_positions'],epc,epe,esc,ese,3)
    for a,b in ((pc,epc),(pe,epe),(sc,esc),(se,ese)):np.testing.assert_array_equal(a,b)
    assert pc.tolist()==[.2,0.,1.,.8] and pe.tolist()==[False,False,True,False]
    assert sc.tolist()==[.9,.3] and trace['references_processed']==7
    cat=SimpleNamespace(sites=5,parents=4,previous=SimpleNamespace(sites=2,parents=2),
        new={'new_site_parent_positions':np.array([2,0,3],dtype=np.uint32)})
    snapshot=ReboundNameSnapshot(cat,np.array([0,1],dtype=np.uint32),pc,pe,sc,se,np.zeros(2))
    n,ne=snapshot.gather(np.arange(5,dtype=np.uint32))
    assert n.tolist()==[.9,.3,1.,.2,.8] and ne.tolist()==[0.,0.,1.,0.,0.]
    # Site0's name does NOT become a parent proof for appended sibling3.
    assert n[3]==.2 and ne[3]==0.


def test_full_parent_snapshot_requires_complete_namespace_and_consistent_exact():
    cat=SimpleNamespace(sites=5,parents=4,previous=SimpleNamespace(sites=2,parents=2),
        new={'new_site_parent_positions':np.array([2,0,3],dtype=np.uint32)})
    def build(pc,pe):return ReboundNameSnapshot(cat,np.array([0,1],dtype=np.uint32),pc,pe,np.zeros(2),np.zeros(2,dtype=bool),np.zeros(2))
    with pytest.raises(ValueError):build(np.zeros(2),np.zeros(2,dtype=bool))
    with pytest.raises(ValueError):build(np.array([0.,0.,.4,0.]),np.array([False,False,True,False]))
    with pytest.raises(ValueError):build(np.array([0.,0.,np.nan,0.]),np.zeros(4,dtype=bool))
