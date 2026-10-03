"""Small independent learning/correctness fixtures, not business training data."""
import numpy as np
import pytest

from tests.test_retrieval_activity_decoder_v111 import scenes
from src.xgb_matcher.retrieval_activity_semantic_v113 import (
    class_lookup,controls,fit,normalized,select)
from src.xgb_matcher.retrieval_activity_decoder_v111 import query_design


def fixture(rotated=False):
    q,rows,old,registry=scenes();codes={k:v-2 for k,v in old.items()};docs=np.eye(6)[:4]
    if rotated:q=q[:,[1,2,3,0,4,5]]
    return q,docs,rows,codes,registry


def test_zero_shot_inner_selection_and_natural_ties_preserved():
    q,docs,rows,codes,registry=fixture()
    p,r=select(q,docs,rows,registry,codes,2)
    assert r['selected']['family']=='ZERO_SHOT'
    assert r['outer']['selected']['query_equal_pair_error']==.25
    assert r['outer']['selected']['same_activity_pairs']==60
    assert r['outer']['selected']['queries']==60
    assert set(r['outer'])=={'frequency','zero_shot','selected'}
    assert not r['recall_measured'] and not r['outer_controls_used_in_selection']
    assert not r['identifiers_as_features'] and r['unknown_descriptions_preserved']
    assert p.shape==(7,6)
    assert r['training_folds']==[3,4]


def test_semantic_target_learns_relation_not_categorical_integer():
    q,docs,rows,codes,registry=fixture(rotated=True)
    _,r=select(q,docs,rows,registry,codes,2)
    assert r['selected']['family']=='SEMANTIC_RIDGE'
    assert r['outer']['selected']['query_equal_pair_error']<r['outer']['zero_shot']['query_equal_pair_error']
    assert r['outer']['selected']['query_equal_pair_error']==.25


def test_outer_target_changes_do_not_change_selection_or_projection():
    q,docs,rows,codes,registry=fixture(rotated=True)
    p,r=select(q,docs,rows,registry,codes,2)
    changed=codes.copy()
    for row in rows:
        if row['fold']==2:changed[row['positive_siret']]=(codes[row['positive_siret']]+1)%4
    other,s=select(q,docs,rows,registry,changed,2)
    np.testing.assert_array_equal(p,other)
    assert r['selected']==s['selected'] and r['candidates']==s['candidates']
    assert r['outer']!=s['outer']


def test_shared_outer_family_target_never_trains_projection():
    q,docs,rows,codes,registry=fixture(rotated=True);registry['2'].append(rows[1]['positive_siren'])
    p,r=select(q,docs,rows,registry,codes,2)
    assert rows[1]['query_id'] not in r['query_ids']
    changed=codes.copy();changed[rows[1]['positive_siret']]=(codes[rows[1]['positive_siret']]+1)%4
    other,s=select(q,docs,rows,registry,changed,2)
    np.testing.assert_array_equal(p,other)
    assert r['selected']==s['selected']


def test_version_and_raw_code_exact_join_no_guess():
    labels=[dict(nomenclature='NAF1993',code='51.3T',label='First'),
            dict(nomenclature='NAFRev1',code='51.3T',label='Second')]
    mapping=class_lookup(['NAF1993','NAFRev1','NAP',None],['51.3T','513T',''],labels)
    assert mapping.tolist()==[[0,-1,-1],[1,-1,-1],[-1,-1,-1],[-1,-1,-1]]
    with pytest.raises(ValueError,match='Duplicate'):class_lookup(['NAF1993'],['51.3T'],labels+[labels[0]])


def test_unknown_definition_targets_and_controls_stay_neutral():
    q,docs,rows,codes,registry=fixture();x=query_design(q)
    for row in rows:codes[row['positive_siret']]=-1
    p=fit(x,docs,rows,codes,list(range(20)),set(),'SEMANTIC_RIDGE',10.)
    assert not np.any(p)
    measured=controls(x,docs,p,'SEMANTIC_RIDGE',rows,codes,[0,1,2])
    assert len(measured)==3 and all(r['unknown_positive'] for r in measured)
    assert all(r['outrank']==4 and not r['top5'] and r['same_activity']==0 for r in measured)


def test_protected_fit_or_empty_control_fails_not_filters():
    q,docs,rows,codes,_=fixture();x=query_design(q)
    with pytest.raises(ValueError,match='family'):
        fit(x,docs,rows,codes,list(range(20)),{rows[0]['positive_siren']},'SEMANTIC_RIDGE',10.)
    p=fit(x,docs,rows,codes,list(range(20)),set(),'ZERO_SHOT')
    with pytest.raises(ValueError,match='filtering'):
        controls(x,docs,p,'ZERO_SHOT',rows,codes,[0],{n['siren'] for n in rows[0]['negatives']})


def test_zero_projection_no_nan_or_invented_evidence():
    assert np.array_equal(normalized(np.zeros((4,6))),np.zeros((4,6)))
    with pytest.raises(ValueError):normalized(np.array([[np.nan]]))


def test_duplicate_definition_vectors_not_fake_disambiguation():
    q,_,rows,codes,registry=fixture();docs=np.tile(np.array([1.,0.,0.,0.,0.,0.]),(4,1))
    _,r=select(q,docs,rows,registry,codes,2)
    # Any semantic family is tied across all definitions; frequency baseline
    # may prefer common classes, but the fixed zero-shot cannot identify sites.
    assert r['outer']['zero_shot']['query_equal_pair_error']==1.
    assert r['outer']['zero_shot']['tied_pairs']==240
