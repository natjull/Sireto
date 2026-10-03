import copy

import numpy as np
import pytest

from src.xgb_matcher.retrieval_sparse_geo_v79 import loss,fit_geometry,select,crossfit_views,design
from src.xgb_matcher.retrieval_sparse_query_v59 import field_bags
from tests.test_retrieval_safe_sparse_v77 import fixture


def test_geographic_loss_gradient_with_offsets_and_equal_query_weight():
    rng=np.random.default_rng(79); x=rng.normal(size=(19,8)); o=rng.normal(size=19)
    sizes=np.array([2,10,7]); w=rng.uniform(0,1,8)
    _,g=loss(w,o,x,sizes,.005); step=1e-6
    for j in range(8):
        v=np.zeros(8); v[j]=step
        numeric=(loss(w+v,o,x,sizes,.005)[0]-loss(w-v,o,x,sizes,.005)[0])/(2*step)
        assert np.isclose(g[j],numeric,rtol=1e-5,atol=1e-7)


def test_nonnegative_geometry_fit_converges_and_improves_loss():
    o=np.full(32,-2.); x=np.ones((32,8)); sizes=np.ones(32,dtype=int)
    w,r=fit_geometry(o,x,sizes,dict(geographic_ridge=.005,maxiter=180))
    assert r['optimizer_converged'] and np.all(w>=0)
    assert r['objective_after']<r['objective_before'] and np.all(o+x@w>0)


def test_crossfit_never_fits_same_source_fold():
    q,rows,scenes,docs,registry,cfg=fixture()
    choice=dict(alpha=0.,ridge=None); indices=list(range(50,150))
    views,scope=crossfit_views(q,rows,scenes,docs,registry,indices,set(registry['2']),cfg,choice)
    assert set(views)=={'3','4'}
    for f,r in scope.items():
        assert int(f) not in r['fitted_folds'] and len(r['fitted_folds'])==1
        assert set(views[f]['query_ids']).isdisjoint({rows[i]['query_id'] for i in indices if rows[i]['fold']==int(f)})


def test_outer_text_queries_and_geometry_cannot_change_selected_metric():
    q,rows,scenes,docs,registry,cfg=fixture(); cfg.update(geographic_ridge=.005,maxiter=180)
    final=dict(weights=[1.]*11,alpha=0.,projection=np.zeros((6,1)),vocabulary=['commune'],
               query_ids=[r['query_id'] for r in rows[50:]])
    choice=dict(alpha=0.,ridge=None)
    w,r=select(q,rows,scenes,docs,registry,2,cfg,choice,final)
    qq=q.copy(); qq[:50]*=100; ss=copy.deepcopy(scenes); dd=copy.deepcopy(docs)
    for row,s in zip(rows[:50],ss[:50]):
        s['positive_features']=[.123]*19; dd[row['positive_siret']]={'legal_names':['OUTER SECRET']}
    ww,rr=select(qq,rows,ss,dd,registry,2,cfg,choice,final)
    assert np.array_equal(w,ww) and r['selected']==rr['selected'] and r['inner_fit']==rr['inner_fit']
    assert final['weights']==[1.]*11


def test_no_safe_negative_is_an_error_not_a_dropped_query():
    q,rows,scenes,docs,registry,cfg=fixture()
    model=dict(weights=[1.]*11,alpha=0.,projection=np.zeros((6,1)),vocabulary=['commune'])
    blocked={n['siren'] for n in rows[0]['negatives']}
    with pytest.raises(ValueError,match='No safe negatives'):
        design(q,rows,scenes,{s:field_bags(f) for s,f in docs.items()},[0],{'2':model},blocked)


def test_invalid_geometry_and_denominator_fail_closed():
    cfg=dict(geographic_ridge=.005,maxiter=180)
    for sizes,x in ((np.ones(16,dtype=int),np.zeros((15,8))),
                    (np.ones(15,dtype=int),np.zeros((15,8)))):
        with pytest.raises(ValueError,match='denominator'):
            fit_geometry(np.zeros(15),x,sizes,cfg)
    x=np.zeros((16,8)); x[0,0]=np.nan
    with pytest.raises(ValueError,match='Nonfinite'):
        fit_geometry(np.zeros(16),x,np.ones(16,dtype=int),cfg)


def test_outer_counterfactual_with_nonzero_sparse_semantics():
    q,rows,scenes,docs,registry,cfg=fixture(); cfg.update(geographic_ridge=.005,maxiter=180)
    final=dict(weights=[1.]*11,alpha=1.,projection=np.ones((6,1)),vocabulary=['commune'],
               query_ids=[r['query_id'] for r in rows[50:]])
    choice=dict(alpha=1.,ridge=10)
    w,r=select(q,rows,scenes,docs,registry,2,cfg,choice,final)
    qq=q.copy(); qq[:50]=-qq[:50]; dd=copy.deepcopy(docs)
    for row in rows[:50]:
        dd[row['positive_siret']]={'legal_names':['UNIQUE OUTER CHANGED']}
    ww,rr=select(qq,rows,scenes,dd,registry,2,cfg,choice,final)
    assert np.array_equal(w,ww) and r['selected']==rr['selected']
    assert r['inner_fit']==rr['inner_fit']
