"""Independent small fixtures only: never business data or retrieval positives."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from tests.test_retrieval_activity_semantic_v113 import fixture as semantic_fixture
from src.xgb_matcher.retrieval_activity_decoder_v111 import query_design
from src.xgb_matcher.retrieval_activity_semantic_v113 import fit as decoder_fit
from src.xgb_matcher.retrieval_nested_v53 import positions
from src.xgb_matcher.retrieval_joint_contrastive_v63 import head_loss
from src.xgb_matcher import retrieval_activity_joint_v114 as joint


def fixture():
    q,docs,rows,codes,registry=semantic_fixture(rotated=True)
    rng=np.random.default_rng(114);scenes=[]
    for row in rows:
        def native():
            f=rng.random(19);f[8]*=2;f[15:]*=f[11:15]
            return f.tolist()
        scenes.append(dict(query_id=row['query_id'],fold=row['fold'],positive_siren=row['positive_siren'],
            positive_features=native(),negatives=[dict(siren=n['siren'],features=native()) for n in row['negatives']]))
    return q,docs,rows,codes,registry,scenes


def test_independent_interaction_formula_and_native_reduction():
    f=np.linspace(.01,.19,19);f[15:]=f[11:15]*.5;a=.37;w=np.linspace(0,3,30)
    expected=np.r_[f,[a*f[i] for i in (4,5,7,11,12,13,14,15,16,17,18)]]
    np.testing.assert_array_equal(joint.vector(f,a),expected)
    assert expected@w==pytest.approx(f@joint.linear_weights(w,a),abs=1e-12)
    np.testing.assert_array_equal(joint.vector(f,0)[:19],f)
    assert not np.any(joint.vector(f,0)[19:])


def test_monotone_bounds_dominate_all_native_and_activity_points():
    rng=np.random.default_rng(77)
    for _ in range(200):
        f=rng.random(19);f[8]*=2;f[15:]*=f[11:15]
        upper=np.ones(19);upper[8]=2.;a=rng.random();amax=a+(1-a)*rng.random();w=rng.random(30)*64
        exact=joint.vector(f,a)@w;bound=upper@joint.linear_weights(w,amax)
        assert exact<=bound+1e-10
        np.testing.assert_allclose(exact,f@joint.linear_weights(w,a),rtol=1e-14)


def test_cached_native_float32_excess_is_not_clipped():
    f=np.zeros(19);f[0]=1.0000013113;f[8]=2.0000019073
    assert joint.vector(f,.3)[0]==f[0] and joint.vector(f,.3)[8]==f[8]


@pytest.mark.parametrize('mutation',['negative','nan','shape','unreliable','geo','individual','activity'])
def test_corrupt_proofs_fail_not_filter(mutation):
    f=np.zeros(19);a=.5
    if mutation=='negative':f[0]=-1
    if mutation=='nan':f[3]=np.nan
    if mutation=='shape':f=f[:18]
    if mutation=='unreliable':f[15]=.1
    if mutation=='geo':f[11]=1.1
    if mutation=='individual':f[0]=1.5
    if mutation=='activity':a=1.01
    with pytest.raises(ValueError):joint.vector(f,a)


@pytest.mark.parametrize('weights,activity',[(np.ones(29),.5),(-np.ones(30),.5),(np.full(30,np.nan),.5),(np.full(30,65),.5),(np.ones(30),1.1)])
def test_unsafe_monotone_coefficients_rejected(weights,activity):
    with pytest.raises(ValueError):joint.linear_weights(weights,activity)


def test_crossfit_own_fold_labels_cannot_change_own_decoder(monkeypatch):
    q,docs,rows,codes,registry,_=fixture();x=query_design(q)
    indices=[i for i,r in enumerate(rows) if r['fold']!=2];blocked=set(registry['2'])
    recipe=dict(family='SEMANTIC_RIDGE',ridge=1.)
    fitted=[]
    def capture(*args,**kwargs):
        parameter=decoder_fit(*args,**kwargs);fitted.append(parameter.copy());return parameter
    monkeypatch.setattr(joint,'decoder_fit',capture)
    strength,audit=joint.crossfit(x,docs,rows,registry,codes,indices,blocked,recipe)
    changed=codes.copy()
    for r in rows:
        if r['fold']==3:changed[r['positive_siret']]=(changed[r['positive_siret']]+1)%4
    other,other_audit=joint.crossfit(x,docs,rows,registry,changed,indices,blocked,recipe)
    assert audit==other_audit
    # Controlled fold 3's decoder is trained on fold 4 only. Changing its
    # source activity joins naturally changes the grade of the changed site,
    # but must not alter the decoder or grades of unchanged negative sites.
    np.testing.assert_array_equal(fitted[0],fitted[2])
    assert not np.array_equal(fitted[1],fitted[3])
    for i in indices:
        if rows[i]['fold']==3:np.testing.assert_array_equal(strength[i][1:],other[i][1:])
    assert any(not np.array_equal(strength[i],other[i]) for i in indices if rows[i]['fold']==4)
    for item in audit:
        assert item['controlled_fold'] not in item['training_folds']
        assert not set(item['training_query_ids'])&set(item['controlled_query_ids'])


def test_crossfit_shared_source_family_is_excluded_from_other_fold_fit():
    q,docs,rows,codes,registry,_=fixture();registry['3'].append(rows[2]['positive_siren'])
    indices=[i for i,r in enumerate(rows) if r['fold']!=2]
    _,audits=joint.crossfit(query_design(q),docs,rows,registry,codes,indices,set(registry['2']),dict(family='SEMANTIC_RIDGE',ridge=1.))
    audit=next(a for a in audits if a['controlled_fold']==3)
    assert rows[2]['query_id'] not in audit['training_query_ids']


def test_outer_labels_and_proofs_do_not_change_inner_selection_or_model():
    q,docs,rows,codes,registry,scenes=fixture();recipe=dict(family='SEMANTIC_RIDGE',ridge=1.)
    p,model,r=joint.select(q,docs,rows,scenes,registry,codes,2,recipe)
    changed=codes.copy();other_scenes=copy.deepcopy(scenes)
    for i,row in enumerate(rows):
        if row['fold']==2:
            changed[row['positive_siret']]=(changed[row['positive_siret']]+1)%4
            other_scenes[i]['positive_features']=[0.]*19
    other,second,s=joint.select(q,docs,rows,other_scenes,registry,changed,2,recipe)
    np.testing.assert_array_equal(p,other)
    assert model==second and r['selected']==s['selected'] and r['candidates']==s['candidates']
    assert r['inner_crossfit']==s['inner_crossfit'] and r['final_crossfit']==s['final_crossfit']
    assert r['outer']!=s['outer']
    train,inner,_=positions(rows,2,registry)
    expected=decoder_fit(query_design(q),docs,rows,codes,sorted(train+inner),set(registry['2']),recipe['family'],recipe['ridge'])
    np.testing.assert_array_equal(p,expected)
    assert len(model['query_ids'])==120 and model['training_folds']==[3,4]
    assert not model['serving_integrated'] and not model['identifiers_as_features']
    assert not r['recall_measured'] and not r['outer_controls_used_in_selection']
    assert all(v['queries']==60 and v['pairs']==240 for v in r['outer'].values())


def test_unknown_is_not_conflated_with_known_zero_and_denominator_stays():
    _,_,rows,codes,_,scenes=fixture();scenes[0]['positive_features']=[0.]*19
    strength={0:np.zeros(5),1:np.zeros(5)};codes[rows[0]['positive_siret']]=-1
    matrix,sizes,tags=joint.design(rows,scenes,[0,1],strength,(),codes)
    r=joint.report(matrix,sizes,np.ones(30),tags)
    assert r['queries']==2 and r['pairs']==8
    assert r['unknown_activity']['pairs']==4 and r['zero_activity_evidence']['pairs']==8
    assert r['no_geometry']['pairs']==4
    assert not np.any(matrix[:4,19:])


def test_natural_negative_order_and_missing_scene_fail():
    _,docs,rows,codes,registry,scenes=fixture()
    joint.aligned(rows,scenes,registry,codes,len(docs))
    with pytest.raises(ValueError,match='All TRAIN'):joint.aligned(rows,scenes[:-1],registry,codes,len(docs))
    scenes[0]['negatives'].reverse()
    with pytest.raises(ValueError,match='order'):joint.aligned(rows,scenes,registry,codes,len(docs))


def test_no_safe_negative_or_protected_positive_stops_never_drops():
    q,docs,rows,codes,registry,scenes=fixture()
    with pytest.raises(ValueError,match='No safe'):
        joint.design(rows,scenes,[0],{0:np.ones(5)},{n['siren'] for n in rows[0]['negatives']},codes)
    with pytest.raises(ValueError,match='family'):
        joint.crossfit(query_design(q),docs,rows,registry,codes,[0],{rows[0]['positive_siren']},dict(family='SEMANTIC_RIDGE',ridge=1.))
    with pytest.raises(ValueError,match='duplicates'):
        joint.crossfit(query_design(q),docs,rows,registry,codes,[0,0],set(),dict(family='SEMANTIC_RIDGE',ridge=1.))


def test_head_optimization_and_equal_query_gradient():
    rng=np.random.default_rng(44);matrix=rng.normal(size=(9,30));sizes=np.array([2,7]);w=rng.random(30)
    prior=np.zeros(30);penalty=np.full(30,.05);value,gradient=head_loss(w,matrix,sizes,prior,penalty)
    for i in [0,19,29]:
        delta=np.zeros(30);delta[i]=1e-6
        expected=(head_loss(w+delta,matrix,sizes,prior,penalty)[0]-head_loss(w-delta,matrix,sizes,prior,penalty)[0])/2e-6
        assert gradient[i]==pytest.approx(expected,abs=1e-8)
    for family in ['NATIVE11','GEOMETRY19','ACTIVITY30']:
        fitted,report=joint.fit_head(matrix,sizes,family)
        assert report['converged'] and report['objective_after']<=report['objective_before']
        assert fitted.shape==(30,) and np.all((fitted>=0)&(fitted<=64))
        assert not np.any(fitted[report['parameters']:])


def test_nonfinite_or_nonmonotone_optimizer_fails(monkeypatch):
    matrix=np.ones((2,30));sizes=np.array([2])
    monkeypatch.setattr(joint,'minimize',lambda *a,**k:SimpleNamespace(x=np.full(30,-1.),fun=0.,success=True,message='fake',nit=1))
    with pytest.raises(ValueError,match='Nonmonotone'):joint.fit_head(matrix,sizes,'ACTIVITY30')
    matrix[0,0]=np.nan
    with pytest.raises(ValueError,match='design'):joint.fit_head(matrix,sizes,'ACTIVITY30')
