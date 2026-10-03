import numpy as np
import pytest

from src.xgb_matcher.retrieval_proof_bridge_v137 import CatalogKeys,facts_bridge,graph_bridge,scatter_name_facts,one_hop_names,MISSING


def keys():
    c=object.__new__(CatalogKeys)
    c.site=np.array([100001,100002,300001],dtype=np.uint64);c.extra_site=np.array([200001,900001],dtype=np.uint64)
    c.parent=np.array([1,3],dtype=np.uint32);c.extra_parent=np.array([2,9],dtype=np.uint32)
    c.sites=5;c.parents=4;return c


def facts():
    return dict(parent_key=np.array([1,1,1,2,4],dtype=np.uint32),
        owner_key=np.array([1,100001,100002,200001,4],dtype=np.uint64),
        subject_code=np.array([1,2,2,2,1],dtype=np.uint8),scope_code=np.array([1,3,3,3,2],dtype=np.uint8))


def test_every_fact_and_scope_including_extra_and_unbound_parent(tmp_path):
    c=keys();s=facts_bridge(c,facts(),tmp_path,batch_rows=2)
    assert s['facts']==5 and s['site_facts']==3 and s['unbound_parent_facts']==1
    assert np.load(tmp_path/'fact_parent_positions.npy').tolist()==[0,0,0,2,MISSING]
    assert np.load(tmp_path/'fact_site_positions.npy').tolist()==[MISSING,0,1,3,MISSING]
    assert not s['site_cartesian_product_materialized'] and s['name_or_owner_or_member_cap'] is None


@pytest.mark.parametrize('change', ['parent','scope','missing_site','float'])
def test_bad_fact_grain_missing_site_and_float_stop(tmp_path,change):
    src=facts()
    if change=='parent':src['parent_key'][1]=2
    if change=='scope':src['scope_code'][1]=1
    if change=='missing_site':src['owner_key'][1]=100003
    if change=='float':src['owner_key']=src['owner_key'].astype(float)
    with pytest.raises(ValueError):facts_bridge(keys(),src,tmp_path,2)


def test_whole_graph_arrays_reconstruct_all_observations_and_extra_site(tmp_path):
    c=keys();src=dict(endpoint_keys=np.array([100001,100002,200001,900001],dtype=np.uint64),
        predecessor_keys=np.array([100001,100001,200001,900001],dtype=np.uint64),
        successor_keys=np.array([100002,100002,900001,900001],dtype=np.uint64))
    s=graph_bridge(c,src,tmp_path,2)
    assert s['observations']==4 and s['endpoints']==4 and s['sites']==5
    assert np.load(tmp_path/'site_parent_positions.npy').tolist()==[0,0,1,2,3]
    assert np.load(tmp_path/'graph_site_positions.npy').tolist()==[0,1,3,4]
    assert np.load(tmp_path/'graph_parent_positions.npy').tolist()==[0,0,2,3]
    assert np.load(tmp_path/'predecessor_graph_positions.npy').tolist()==[0,0,2,3]
    assert np.load(tmp_path/'successor_graph_positions.npy').tolist()==[1,1,3,3]


def test_large_owner_all_facts_no_seed_cap(tmp_path):
    n=2501;src={k:np.repeat(v[:1],n) for k,v in facts().items()}
    s=facts_bridge(keys(),src,tmp_path,17)
    assert s['facts']==2501 and np.all(np.load(tmp_path/'fact_parent_positions.npy')==0)


def test_max_whole_alias_and_site_only_scope_no_frankenproof_or_false_exact():
    cov=np.array([.6,.7,1.,.9]);ex=np.array([False,False,True,True]);offsets=np.array([0,1,3,4,5],dtype=np.uint64)
    sub=np.array([1,1,2,2,1],dtype=np.uint8);cur=np.array([True,True,False,True,False])
    pp=np.array([0,0,0,1,MISSING],dtype=np.uint32);sp=np.array([MISSING,MISSING,0,1,MISSING],dtype=np.uint32)
    pc=np.zeros(2);pe=np.zeros(2,dtype=bool);sc=np.zeros(2);se=np.zeros(2,dtype=bool)
    scatter_name_facts(cov,ex,offsets,sub,cur,pp,sp,pc,pe,sc,se,2)
    assert pc.tolist()==[.7,0.] and sc.tolist()==[.7,1.] # no sum of unrelated aliases
    assert pe.tolist()==[False,False] and se.tolist()==[False,True]
    # The exact site proof NEVER becomes the target parent alias.
    assert pe[1]==False and pc[1]==0.


def test_one_hop_snapshot_duplicate_max_self_context_and_unknown_false_grades():
    pc=np.zeros(4);sc=np.array([1.,0.,0.,0.]);gs=np.arange(4,dtype=np.uint32);gp=gs.copy()
    pred=np.array([0,0,1,0,0,0],dtype=np.uint32);succ=np.array([1,1,2,0,3,3],dtype=np.uint32)
    code=np.array([2,2,2,2,0,1],dtype=np.uint8)
    result=one_hop_names(pc,sc,gs,gp,pred,succ,code,2)
    assert result.tolist()==[0.,.5,0.,.25] # neither repeated boosts nor A→B→C
    assert sc.tolist()==[1.,0.,0.,0.] # snapshot input not mutated


def test_relation_alias_from_parent_targets_only_endpoint_not_other_sites():
    pc=np.array([.8,0.]);sc=np.zeros(3);gs=np.array([0,1],dtype=np.uint32);gp=np.array([0,1],dtype=np.uint32)
    r=one_hop_names(pc,sc,gs,gp,np.array([0],dtype=np.uint32),np.array([1],dtype=np.uint32),np.array([2],dtype=np.uint8))
    assert r.tolist()==[0.,.4,0.]


def test_random_whole_graph_scalar_oracle_exact_both_directions():
    rng=np.random.default_rng(73);n=177;pc=rng.random(19);sc=rng.random(n);gs=np.arange(n,dtype=np.uint32)
    gp=rng.integers(0,19,n,dtype=np.uint32);p=rng.integers(0,n,901,dtype=np.uint32);s=rng.integers(0,n,901,dtype=np.uint32)
    code=rng.integers(0,3,len(p),dtype=np.uint8);expected=np.zeros(n);snapshot=np.maximum(sc,pc[gp])
    for a,b,c in zip(p,s,code):
        if a==b:continue
        grade=.5 if c==2 else .25
        expected[a]=max(expected[a],snapshot[b]*grade);expected[b]=max(expected[b],snapshot[a]*grade)
    assert np.array_equal(one_hop_names(pc,sc,gs,gp,p,s,code,13),expected)


def test_graph_unknown_out_of_range_and_signed_types_stop():
    pc=np.zeros(1);sc=np.zeros(1);ids=np.array([0],dtype=np.uint32);code=np.array([2],dtype=np.uint8)
    for bad in (np.array([MISSING],dtype=np.uint32),np.array([0],dtype=np.int64)):
        with pytest.raises(ValueError):one_hop_names(pc,sc,ids,ids,bad,ids,code)
    with pytest.raises(ValueError):one_hop_names(pc,sc,ids,ids,ids,ids,np.array([3],dtype=np.uint8))
    with pytest.raises(ValueError):one_hop_names(pc,sc,ids,ids,ids,ids,code,0)
