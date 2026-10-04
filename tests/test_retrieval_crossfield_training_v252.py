from copy import deepcopy
import numpy as np
import pytest
from test_retrieval_listwise_initial_v233 import fixture as old_fixture
from test_retrieval_topk_initial_v239 import priors
from src.xgb_matcher.retrieval_crossfield_training_v252 import fit,controls,loss,design


def fixture():
    rows,registry,old,acceptable=old_fixture()
    for row in rows:row['extra_features']=[[1.,0.,0.],[0.,0.,0.]]
    return rows,registry,old,acceptable,priors(old)


def test_gradient_equals_finite_difference_fixed_old_head():
    rng=np.random.default_rng(7);x=rng.normal(size=(6,3));o=rng.normal(size=6);w=rng.uniform(0,1,size=3);sw=np.ones(6)/6
    _,g=loss(w,x,o,sw);observed=[]
    for i in range(3):
        d=np.zeros(3);d[i]=1e-6
        observed.append((loss(w+d,x,o,sw)[0]-loss(w-d,x,o,sw)[0])/2e-6)
    assert np.allclose(g,observed,atol=1e-8)


def test_only_extra_three_learn_and_prior11_metadata_unchanged():
    rows,registry,old,acceptable,p=fixture()
    models={str(f):fit(rows,registry,old,acceptable,f,p) for f in (2,3,4)}
    for f,m in models.items():
        assert {k:v for k,v in m.items() if k not in ('extra_weights','extra_proof_training')}==p[f]
        assert m['extra_weights'][0]>0 and m['extra_weights'][1:]==[0.,0.]
        assert not set(m['extra_proof_training']['query_ids'])&{r['query_id'] for r in rows if r['fold']==int(f)}
    c=controls(rows,registry,old,models,acceptable,p)
    assert c['recall_measured'] is False and c['controls_used_in_model_selection'] is False


def test_held_features_and_extra_cannot_change_held_model():
    rows,registry,old,acceptable,p=fixture();before=fit(rows,registry,old,acceptable,2,p)
    for row in rows:
        if row['fold']==2:
            row['features']=[[999.]*11,[777.]*11];row['extra_features']=[[500.]*3,[700.]*3]
    assert fit(rows,registry,old,acceptable,2,p)==before


def test_acceptables_and_protected_entities_not_negative_and_no_silent_drop():
    rows,registry,old,acceptable,p=fixture();r=rows[1]
    acceptable[r['query_id']].update(r['references'])
    with pytest.raises(ValueError,match='negative'):fit(rows,registry,old,acceptable,2,p)


def test_changed_prior_conditional_or_partial_fails():
    rows,registry,old,acceptable,p=fixture()
    p['2']['weights'][1]=1.
    with pytest.raises(ValueError,match='prior'):fit(rows,registry,old,acceptable,2,p)
    p=priors(old);rows[1]['extra_features'][0][1]=1.
    with pytest.raises(ValueError,match='conditional'):fit(rows,registry,old,acceptable,2,p)
    with pytest.raises(ValueError):fit(rows[:-1],registry,old,acceptable,2,p)
