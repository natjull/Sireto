import numpy as np
import pytest

from src.xgb_matcher.retrieval_nested_v53 import positions,inner_entity,select_and_refit


def fixture():
    rng=np.random.default_rng(53); q=rng.normal(size=(180,8)); p=q+rng.normal(size=q.shape)*.3
    members=[]; negatives=[]; registry={str(f):[] for f in (2,3,4)}
    for i in range(180):
        fold=2+i//60; parent=f'{100000000+i}'; registry[str(fold)].append(parent)
        members.append(dict(query_id=str(i),fold=fold,positive_siren=parent,negative_sirens=['100000000','999999999']))
        negatives.append(rng.normal(size=(2,8)))
    cfg=dict(capacities=['FIELD_SCALE','DIAGONAL'],ridges=[.1,1.],temperature=10.,hard_count=8,maxiter=50)
    return q,p,negatives,members,registry,cfg


def test_inner_split_is_by_entity_not_query_and_heldout_is_excluded():
    _,_,_,m,r,_=fixture(); train,val,inner=positions(m,2,r)
    assert not set(train)&set(val) and not (set(train)|set(val))&set(range(60))
    assert all(m[i]['positive_siren'] not in inner for i in train)
    assert all(inner_entity(m[i]['positive_siren']) for i in val)
    assert set(train)|set(val)==set(range(60,180))


def test_outer_control_vectors_do_not_affect_selection_or_final_weights():
    q,p,n,m,r,cfg=fixture(); w,first=select_and_refit(q,p,n,m,2,r,cfg)
    altered=q.copy(); altered[:60]=np.random.default_rng(530).normal(size=(60,8))
    other,second=select_and_refit(altered,p,n,m,2,r,cfg)
    assert first['selected']==second['selected'] and first['candidates']==second['candidates']
    assert np.array_equal(w,other)
    assert first['outer']!=second['outer'] and not first['outer_controls_used_in_selection']


@pytest.mark.parametrize('change',['duplicate','forbidden_fold','missing_registry'])
def test_invalid_source_scope_fails_closed(change):
    _,_,_,m,r,_=fixture()
    if change=='duplicate': m[1]['query_id']=m[0]['query_id']
    if change=='forbidden_fold': m[1]['fold']=1
    if change=='missing_registry': r['2']=[]
    with pytest.raises(ValueError): positions(m,2,r)
