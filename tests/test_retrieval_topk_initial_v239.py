"""Convex algebra/held invariance fixtures ONLY, never national recall."""
from copy import deepcopy
import numpy as np
import pytest

from test_retrieval_listwise_initial_v233 import fixture
from src.xgb_matcher.retrieval_topk_initial_v239 import objective, solve, fit, controls


def priors(original):
    result=deepcopy(original)
    for m in result.values():
        m['weights'][4]=1.5
        m['place_training']={'SYNTHETIC_PRIOR_METADATA_NOT_AUTHORITY':True}
    return result


def test_lp_matches_direct_hinge_top100_not_max_negative_or_top1():
    x=np.zeros((200,11));x[:,0]=1.;prior=np.zeros(11)
    # One very hard competitor,199 easy: top1 high while top100 mean zero.
    o=np.r_[-10.,np.full(199,10.)]
    assert objective(prior,[x],[o],prior)==0.
    w,report=solve([x],[o],prior)
    assert np.array_equal(w,prior) and report['objective_after']==0.


def test_top100_rank_failure_has_hinge_at_least_one_and_lp_improves():
    x=np.zeros((200,11));x[:,4]=1.;o=np.zeros(200);prior=np.zeros(11)
    assert objective(prior,[x],[o],prior)==1.
    w,report=solve([x],[o],prior)
    assert w[4]==pytest.approx(1.) and report['objective_after']<report['objective_before']


def test_direct_loss_is_convex_for_fixed_offsets_and_balanced_queries():
    rng=np.random.default_rng(5);xs=[rng.normal(size=(200,11)),rng.normal(size=(37,11))]
    os=[rng.normal(size=len(x)) for x in xs];prior=np.ones(11)
    a=rng.uniform(0,2,size=11);b=rng.uniform(0,2,size=11)
    assert objective((a+b)/2,xs,os,prior)<= (objective(a,xs,os,prior)+objective(b,xs,os,prior))/2+1e-12


def test_held_queries_and_entities_cannot_enter_model_or_selection():
    rows,registry,old,allowed=fixture();p=priors(old)
    models={str(f):fit(rows,registry,old,allowed,f,p) for f in (2,3,4)}
    for f,m in models.items():
        assert not set(m['native_training']['query_ids'])&{r['query_id'] for r in rows if r['fold']==int(f)}
        assert m['native_training']['parameters_selected_on_outer_controls'] is False
        assert m['native_training']['k']==100 and m['native_training']['queries_with_fewer_than100_negative_classes']>0
    report=controls(rows,registry,old,models,allowed,p)
    assert report['recall_measured'] is False and report['controls_used_in_model_selection'] is False


def test_changing_held_features_cannot_change_its_parameters():
    rows,registry,old,allowed=fixture();p=priors(old);before=fit(rows,registry,old,allowed,2,p)
    for r in rows:
        if r['fold']==2:r['features']=[[900.]*11,[700.]*11]
    assert fit(rows,registry,old,allowed,2,p)==before


def test_acceptable_held_negatives_partial_or_wrong_prior_stop():
    rows,registry,old,allowed=fixture();p=priors(old)
    p['2']['weights'][1]=3.
    with pytest.raises(ValueError,match='prior'):fit(rows,registry,old,allowed,2,p)
    p=priors(old);allowed[rows[1]['query_id']].update(rows[1]['references'])
    with pytest.raises(ValueError,match='negative'):fit(rows,registry,old,allowed,2,p)
    with pytest.raises(ValueError):solve([np.ones((3,11))],[],np.ones(11))
    with pytest.raises(ValueError):fit(rows[:-1],registry,old,allowed,2,p)
