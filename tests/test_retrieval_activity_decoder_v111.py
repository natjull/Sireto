"""Small correctness fixtures only; never injected into CRM training/evaluation."""
import copy

import numpy as np
import pytest

from src.xgb_matcher.retrieval_activity_decoder_v111 import (
    controls,fit,query_design,scores,select,summary,validate)


def scenes():
    rng=np.random.default_rng(11);rows=[];codes={};vectors=[];registry={'2':[],'3':[],'4':[]}
    for i in range(180):
        parent=f'{100000000+i:09d}';site=parent+'00001';fold=2+i%3;category=2+i%4
        negative=[]
        for j in range(4):
            np_=f'{600000000+i*10+j:09d}';ns=np_+'00001'
            negative.append(dict(siret=ns,siren=np_));codes[ns]=2+j
        rows.append(dict(query_id=f'q{i}',fold=fold,positive_siren=parent,positive_siret=site,negatives=negative))
        codes[site]=category;registry[str(fold)].append(parent)
        q=np.zeros(6);q[category-2]=1.;q+=rng.normal(scale=.01,size=6);vectors.append(q)
    return np.array(vectors),rows,codes,registry


def test_nested_decoder_signal_not_siret_recall():
    x,rows,codes,registry=scenes()
    projection,report=select(x,rows,registry,codes,6,2)
    assert report['selected']['family']=='NAME_ACTIVITY_RIDGE'
    assert report['outer_selected']['query_equal_pair_error']<report['outer_frequency']['query_equal_pair_error']
    assert report['outer_selected']['same_activity_pairs']==60
    assert report['outer_selected']['tied_pairs']>=60
    assert report['outer_selected']['queries']==60
    assert report['outer_selected']['raw_activity_top1']==1.
    assert report['training_folds']==[3,4]
    assert len(report['query_ids'])==120
    assert not report['recall_measured']
    assert not report['outer_controls_used_in_selection']
    assert report['activity_top5_is_not_siret_recall']
    assert not report['identifiers_as_features']
    assert projection.shape==(7,6)
    assert np.all(projection[:,:2]==0)


def test_outer_labels_cannot_change_model_or_selection():
    x,rows,codes,registry=scenes()
    p,r=select(x,rows,registry,codes,6,2)
    changed=codes.copy()
    for row in rows:
        if row['fold']==2:changed[row['positive_siret']]=2+(changed[row['positive_siret']]-1)%4
    other,s=select(x,rows,registry,changed,6,2)
    np.testing.assert_array_equal(p,other)
    assert r['selected']==s['selected']
    assert r['candidates']==s['candidates']
    assert r['outer_selected']!=s['outer_selected']


def test_shared_family_excluded_even_if_other_fold():
    x,rows,codes,registry=scenes();protected=rows[1]['positive_siren']
    registry['2'].append(protected)
    p,r=select(x,rows,registry,codes,6,2)
    assert rows[1]['query_id'] not in r['query_ids']
    changed=codes.copy();changed[rows[1]['positive_siret']]=5
    other,s=select(x,rows,registry,changed,6,2)
    np.testing.assert_array_equal(p,other)
    assert r['selected']==s['selected']


def test_protected_control_negatives_excluded_not_queries():
    x,rows,codes,registry=scenes();x=query_design(x);projection=np.zeros((7,6),np.float32)
    blocked={rows[0]['negatives'][0]['siren']}
    measured=controls(x,projection,rows,codes,[0],blocked)
    assert len(measured)==1 and measured[0]['pairs']==3
    with pytest.raises(ValueError,match='denominator'):
        controls(x,projection,rows,codes,[0],{n['siren'] for n in rows[0]['negatives']})


def test_unknown_targets_neutral_not_filtered():
    x,rows,codes,registry=scenes();x=query_design(x)
    for row in rows:codes[row['positive_siret']]=0
    projection=fit(x,rows,codes,list(range(20)),6,set(),10.)
    assert not np.any(projection)
    measured=controls(x,projection,rows,codes,[0,1,2])
    s=summary(measured)
    assert s['queries']==3 and s['unknown_positives']==3
    assert s['raw_activity_top1']==s['raw_activity_top5']==0.
    assert s['query_equal_pair_error']==1.


@pytest.mark.parametrize('mutation',['closed','duplicate','missing','badcode','identifier'])
def test_invalid_training_fails_without_row_filter(mutation):
    _,rows,codes,registry=scenes();rows=copy.deepcopy(rows)
    if mutation=='closed':rows[0]['fold']=1
    elif mutation=='duplicate':rows[1]['query_id']=rows[0]['query_id']
    elif mutation=='missing':del codes[rows[0]['positive_siret']]
    elif mutation=='badcode':codes[rows[0]['positive_siret']]=6
    elif mutation=='identifier':rows[0]['positive_siret']='１００００００００００００１'
    with pytest.raises(ValueError):validate(rows,registry,codes,6)


def test_protected_positive_in_fit_stops():
    x,rows,codes,_=scenes();x=query_design(x)
    with pytest.raises(ValueError,match='Protected'):
        fit(x,rows,codes,list(range(20)),6,{rows[3]['positive_siren']},1.)


def test_bounded_all_classes_and_missing_zero():
    x=np.array([[1.,1.]])
    p=np.array([[0.,0.,-2.,3.,.25],[0.,0.,0.,0.,0.]],dtype=np.float32)
    assert scores(x,p).tolist()==[[0.,0.,0.,1.,.25]]
    p[0,0]=1.
    with pytest.raises(ValueError,match='missing'):
        scores(x,p)


@pytest.mark.parametrize('vectors',[np.zeros((2,6)),np.array([[np.nan,1.]]),np.ones(6)])
def test_invalid_fixed_vectors(vectors):
    with pytest.raises(ValueError):query_design(vectors)
