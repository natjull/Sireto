"""Synthetic algebra and OOF safety fixtures, never retrieval proof."""
from copy import deepcopy
import numpy as np
import pytest
from src.xgb_matcher.retrieval_listwise_initial_v233 import loss,fit,controls,design


def fixture():
    registry={str(f):[f'{f:09d}'] for f in (2,3,4)};rows=[];acceptable={}
    for i in range(64):
        fold=2+i%3;pos=f'{fold:09d}00001';neg='90000000100001'
        # refs sorted: positive first. A nonlocal homonym has stronger name.
        p=[0.,0.,0.,0.,1.]+[0.]*6;n=[1.,0.,0.,0.,0.]+[0.]*6
        rows.append(dict(query_id=f'T{i}',fold=fold,positive_siret=pos,positive_siren=pos[:9],
            references={pos:pos[:9],neg:neg[:9]},features=[p,n],fixed_scores={str(f):[0.,0.] for f in (2,3,4)}))
        acceptable[f'T{i}']={pos}
    original={str(f):dict(weights=[1.]+[0.]*10,query_ids=[r['query_id'] for r in rows if r['fold']!=f]) for f in (2,3,4)}
    return rows,registry,original,acceptable


def test_listwise_gradient_with_fixed_scores_matches_finite_differences():
    rng=np.random.default_rng(3);x=rng.normal(size=(5,11));o=rng.normal(size=5);w=rng.normal(size=11);prior=np.ones(11);sizes=np.array([2,3])
    value,g=loss(w,x,o,sizes,prior)
    finite=[]
    for i in range(11):
        step=np.zeros(11);step[i]=1e-6
        finite.append((loss(w+step,x,o,sizes,prior)[0]-loss(w-step,x,o,sizes,prior)[0])/2e-6)
    assert np.allclose(g,finite,atol=1e-8)


def test_protected_held_queries_never_enter_fit_or_tuning():
    rows,registry,old,acceptable=fixture()
    models={str(f):fit(rows,registry,old,acceptable,f) for f in (2,3,4)}
    for f,m in models.items():
        assert m['weights'][4]>0 and m['native_training']['objective_after']<m['native_training']['objective_before']
        assert not set(m['native_training']['query_ids'])&{r['query_id'] for r in rows if r['fold']==int(f)}
    report=controls(rows,registry,old,models,acceptable)
    assert report['mean_negative_outrank_count_after']<report['mean_negative_outrank_count_before']
    assert not report['recall_measured'] and not report['controls_used_in_model_selection']


def test_changing_held_point_text_features_does_not_change_its_model():
    rows,registry,old,acceptable=fixture();before=fit(rows,registry,old,acceptable,2)
    for r in rows:
        if r['fold']==2:r['features']=[[999.]*11,[777.]*11]
    assert fit(rows,registry,old,acceptable,2)==before


def test_acceptable_and_held_negatives_cannot_fit():
    rows,registry,old,acceptable=fixture();r=rows[1]
    acceptable[r['query_id']].update(r['references'])
    with pytest.raises(ValueError,match='negative'):fit(rows,registry,old,acceptable,2)


def test_bad_or_partial_namespace_stops():
    rows,registry,old,acceptable=fixture();rows.pop()
    with pytest.raises(ValueError):fit(rows,registry,old,acceptable,2)
