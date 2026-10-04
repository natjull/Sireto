"""Synthetic source/native algebra fixtures only, not business recall."""
from types import SimpleNamespace

import numpy as np
import pytest

from test_retrieval_corroborated_initial_v217 import tiny_native
from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery
from src.xgb_matcher.retrieval_corroborated_initial_v217 import compose as baseline
from src.xgb_matcher.retrieval_official_fragments_v236 import Evidence, compose, fragment_grades, grouped, search
from src.xgb_matcher.retrieval_training_score_v231 import points


def test_all_contiguous_exact_official_fragments_generic_not_alias_dictionary():
    grades = fragment_grades('COMMUNE DE ALPHA ECOLE BETA')
    assert grades['COMMUNE DE ALPHA'] == 1. and grades['ECOLE BETA'] == 1.
    assert grades['ALPHA'] == .25 and 'DE' not in grades
    assert 'COMMUNE DE ALPHA ECOLE BETA' not in grades
    assert fragment_grades('ALPHA') == {}
    assert '123456789' not in fragment_grades('ALPHA 12345678900001 BETA')


def test_grouped_no_scope_borrow_or_sum():
    groups = grouped({('parent', '100000000'): 1., ('site', '20000000000003'): .5})
    assert groups == [({'100000000'}, set(), 1.), (set(), {'20000000000003'}, .5)]
    with pytest.raises(ValueError):
        grouped({('sibling', '10000000000001'): 1.})


def test_no_new_fragment_native_score_parity_baseline_owners_relations():
    b = tiny_native(); q = RetrievalQuery(name='ALPHA', address='ROAD', postcode='11111', insee='11111')
    ids = [f'100000000{i:05d}' for i in (1,2,3)]
    weights = [2.2,.7,.9,.5,1.4,1.1,1.2,0.,.01,0.,.13]
    p = {'100000000'}; s = set(); related = {ids[1]}; rings = [({ids[0]}, .8)]
    expected = points(b, baseline(b,q,weights,[],0.,p,s,related,rings), ids)
    actual = points(b, compose(b,q,weights,[],0.,[(p,s,1.)],[(related,.5)],rings), ids)
    assert np.allclose(actual, expected, atol=2e-5, rtol=2e-6)


def test_owner_repeated_in_many_fragment_groups_not_double_bonus():
    b=tiny_native(); q=RetrievalQuery(name='UNRELATED',address='ROAD',postcode='11111')
    ids=[f'100000000{i:05d}' for i in (1,2,3)]; groups=[({'100000000'},set(),1.)]
    a=points(b,compose(b,q,[0.]*11,[],0.,groups,[],[]),ids)
    c=points(b,compose(b,q,[0.]*11,[],0.,groups*3+[(set(),{ids[0]},.25)],[],[]),ids)
    assert np.array_equal(a,c) and np.all(a==1.)


def test_site_alias_never_promoted_to_siblings_and_off_zone_reachable():
    b=tiny_native();q=RetrievalQuery(name='UNRELATED',address='ROAD',postcode='11111')
    ids=[f'100000000{i:05d}' for i in (1,2,3)]
    actual=points(b,compose(b,q,[0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[],0.,
                          [(set(),{ids[0]},.5)],[],[]),ids)
    assert actual[0]==1.5 and actual[1]==actual[2]==1.


def fixture():
    # Native-source CSR fixture only. Two aliases: same UL twice + one site.
    arrays=dict(name_fact_offsets=np.array([0,2,4],dtype=np.uint64),
                owner_key=np.array([100000000,100000000,100000000,20000000000003],dtype=np.uint64),
                parent_key=np.array([100000000,100000000,100000000,200000000],dtype=np.uint32),
                subject_code=np.array([1,1,1,2],dtype=np.uint8), scope_code=np.array([1,2,1,3],dtype=np.uint8))
    class Array:
        def __init__(self,a): self.a=a; self._mmap=SimpleNamespace(close=lambda:None)
        def __getitem__(self,k):return self.a[k]
    files=SimpleNamespace(guard=lambda:None,check_epochs=lambda:None,
        mapped=lambda ref,dtype,shape:Array(arrays[ref]))
    e=object.__new__(Evidence);e.files=files
    e.items=[dict(name_start=0,names=2,facts=4,artifacts={k+'.npy':k for k in arrays})]
    def lookup(channel,text):
        assert channel==3 # No all-word intersection on a shortened fragment.
        return np.array({'ALPHA':[0],'ALPHA BETA':[0,1],'BETA':[1]}.get(text,[]),dtype=np.uint32)
    e.lookup=lookup
    return e,arrays


def test_entire_official_owner_intervals_read_and_strongest_alias_preserved():
    e,_=fixture();groups,trace=e.fragment_owners('ALPHA BETA GAMMA')
    assert groups==[({'100000000'},{'20000000000003'},1.)]
    assert trace['owner_facts']==4 and trace['aliases']==2 and trace['owner_cap'] is None
    assert trace['crm_alias_dictionary'] is False


def test_inconsistent_scope_stops_never_inherits_parent():
    e,arrays=fixture();arrays['scope_code'][-1]=1
    with pytest.raises(ValueError,match='scope conflict'):e.fragment_owners('ALPHA BETA GAMMA')


def test_exactly_one_global_native_collector_no_pool_or_point_lookup():
    b=tiny_native(); original=b.searcher; calls=[]
    def searched(query,**kwargs):
        calls.append(kwargs)
        return original.search(query,**kwargs)
    b.searcher=SimpleNamespace(search=searched,doc=original.doc)
    original_guard=lambda:None
    e=SimpleNamespace(files=SimpleNamespace(guard=original_guard),
        name_owners=lambda text:({'100000000'},set(),{}),
        fragment_owners=lambda text:([(set(),{'10000000000001'},.25)],{}),
        related=lambda p,s:(set(),{}),radial_proofs=lambda geo:([],{}))
    q=RetrievalQuery(name='ALPHA',address='ROAD',postcode='11111')
    found,trace=search(b,e,q,{},[1.]*11,[],0.)
    assert len(found)==3 and calls==[dict(limit=100,count=False)]
    assert trace['label_lookup'] is False and trace['fixed_reservoir'] is None
    assert trace['source_access_hypothesis']=='V236_WHOLE_OFFICIAL_FRAGMENT_ROLE_OWNERS'
    assert e.files.guard is original_guard
