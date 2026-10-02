import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

from src.xgb_matcher.retrieval_joint_contrastive_v63 import (
    FEATURES, CONTEXT, unit_parts, context, design, head_loss, query_kernel,
    validate, bases, select, summarize)


def fixture():
    rng = np.random.default_rng(71); rows = []; scenes = []; registry = {str(f): [] for f in (2,3,4)}
    for i in range(96):
        f = 2 + i % 3; p = f'{100000000+i:09d}'; registry[str(f)].append(p)
        rows.append(dict(query_id=str(i), fold=f, positive_siren=p, positive_siret=p+'00001',
            crm_name='EXEMPLE', crm_number='2', crm_street='RUE TEST', negatives=[dict(siren=p, siret=p+'00002')]))
        scenes.append(dict(query_id=str(i), fold=f, positive_siren=p,
            positive_features=rng.random(len(FEATURES)).tolist(),
            negatives=[dict(siren=p, features=rng.random(len(FEATURES)).tolist())]))
    ids = [s for r in rows for s in [r['positive_siret'], r['negatives'][0]['siret']]]
    q = unit_parts(rng.normal(size=(96,8))); docs = unit_parts(rng.normal(size=(192,8)))
    c = np.tile([1,1,0,1,1,0,0,0], (96,1)).astype(float)
    return rows, scenes, q, docs, ids, c, registry


def test_head_gradient_and_query_weighting():
    rng = np.random.default_rng(9); x = rng.normal(size=(7,5)); w = rng.normal(size=5)
    sizes = np.array([1,2,4]); prior = np.zeros(5); penalty = np.full(5,.03)
    loss, grad = head_loss(w,x,sizes,prior,penalty)
    numeric = approx_derivative(lambda v: head_loss(v,x,sizes,prior,penalty)[0], w).reshape(-1)
    np.testing.assert_allclose(grad,numeric,rtol=1e-5,atol=1e-7)
    repeated = np.repeat(x[:1],3,axis=0)
    one = head_loss(w,x[:1],np.array([1]),prior,penalty)[0]
    three = head_loss(w,repeated,np.array([3]),prior,penalty)[0]
    assert three > one  # More competitors are not hidden by per-pair averaging.


def test_query_kernel_matches_training_score_with_separate_views():
    rows, scenes, q, docs, ids, c, reg = fixture(); byid = validate(rows, scenes, q, docs, ids, reg)
    latent = bases(rows,q,docs,byid,list(range(6)),())
    x,sizes,tags = design(rows,scenes,q,docs,byid,c,[9],(),'JOINT_ADAPTED',latent)
    rng = np.random.default_rng(8); w = rng.random(x.shape[1]); w[len(CONTEXT)*len(FEATURES)+2:] -= .5
    model = dict(family='JOINT_ADAPTED',weights=w.tolist(),
                 latent=[dict(query=u.tolist(), document=d.tolist()) for u,d in latent])
    native, transformed = query_kernel(model,q[9].reshape(-1),c[9])
    delta = docs[byid[rows[9]['positive_siret']]] - docs[byid[rows[9]['negatives'][0]['siret']]]
    proof = np.array(scenes[9]['positive_features']) - np.array(scenes[9]['negatives'][0]['features'])
    np.testing.assert_allclose(x@w, [native@proof + np.sum(transformed*delta)], atol=1e-12)
    assert np.all(native>=0) and tags[0]['same_parent']


def test_outer_entities_are_absent_from_basis_and_training():
    rows, scenes, q, docs, ids, c, reg = fixture(); byid = validate(rows,scenes,q,docs,ids,reg)
    eligible = [i for i,r in enumerate(rows) if r['fold']!=2]
    a = bases(rows,q,docs,byid,eligible,reg['2'])
    q2=q.copy(); d2=docs.copy()
    for i,r in enumerate(rows):
        if r['fold']==2:
            q2[i] *= -1; d2[byid[r['positive_siret']]] *= -1; d2[byid[r['negatives'][0]['siret']]] *= -1
    b = bases(rows,q2,d2,byid,eligible,reg['2'])
    for aa,bb in zip(a,b):
        for aaa,bbb in zip(aa,bb): np.testing.assert_array_equal(aaa,bbb)


def test_protected_negative_exclusion_and_missing_proof_stop():
    rows, scenes, q, docs, ids, c, reg = fixture(); byid = validate(rows,scenes,q,docs,ids,reg)
    with pytest.raises(ValueError,match='No safe natural negative'):
        design(rows,scenes,q,docs,byid,c,[0],reg['2'],'HEAD19')
    scenes[0]['negatives'][0]['features'] = [0]
    with pytest.raises(ValueError,match='proof drift|inhomogeneous'):
        validate(rows,scenes,q,docs,ids,reg)


def test_empty_views_stay_zero_and_identifiers_are_not_context():
    np.testing.assert_array_equal(unit_parts(np.zeros((1,8))),np.zeros((1,2,4)))
    geo = dict(ban_resolved=False,spatial_crs_compatible=False)
    a = dict(crm_name='TEST 123456789',crm_number='2',crm_street='RUE A')
    b = dict(a,crm_name='TEST 987654321')
    np.testing.assert_array_equal(context(a,geo),context(b,geo))


def test_outer_mutation_does_not_change_inner_choice_or_model():
    rows, scenes, q, docs, ids, c, reg = fixture()
    a, ar = select(rows,scenes,q,docs,ids,c,reg,2,[.01],180)
    q2 = q.copy(); d2 = docs.copy(); c2 = c.copy()
    import copy
    s2 = copy.deepcopy(scenes)
    byid = dict(zip(ids,range(len(ids))))
    for i,r in enumerate(rows):
        if r['fold']==2:
            q2[i] *= -1; c2[i] = 1
            d2[byid[r['positive_siret']]] *= -1; d2[byid[r['negatives'][0]['siret']]] *= -1
            s2[i]['positive_features'] = [0.]*len(FEATURES)
    b, br = select(rows,s2,q2,d2,ids,c2,reg,2,[.01],180)
    assert a == b and ar['selected'] == br['selected']
    assert ar['outer_after'] != br['outer_after']
