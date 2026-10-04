"""Tiny synthetic native index proves score algebra, NEVER business recall."""
import numpy as np
import pytest
from test_retrieval_corroborated_initial_v217 import tiny_native
from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery
from src.xgb_matcher.retrieval_training_score_v231 import native_features,fixed_scores,parity,points
from src.xgb_matcher.retrieval_corroborated_initial_v217 import compose


def fixture():
    b=tiny_native();q=RetrievalQuery(name='COMMUNE DE ALPHA ECOLE BETA',address='ROAD',
                                     postcode='11111',insee='11111',number='1')
    ids=['10000000000003','10000000000001','10000000000002']
    return b,q,ids


def test_native217_features_include_containment_and_parent_site_not_old56():
    b,q,ids=fixture();ordered,x=native_features(b,q,ids)
    assert ordered==sorted(ids) and x.shape==(3,11)
    assert x[0,2]==1. # contained official legal/site alias, not full CRM alias
    assert x[0,6]>x[1,6] # coherent THIS legal+site content, no sibling borrowing
    assert x[2,4]==0. # geography remains a soft feature, not removed document


def test_score_decomposition_parity_every_reference_fixed_relations_xy_and_semantic():
    b,q,ids=fixture();ids,x=native_features(b,q,ids)
    parents={'100000000'};sites=set();related={ids[1]};rings=[({ids[0]},.8)]
    terms=[('beta',1.)];alpha=2.
    fixed=fixed_scores(b,q,ids,terms,alpha,parents,sites,related,rings)
    for weights in ([1.]*11,[2.2,.7,.9,.5,1.4,1.1,1.2,0.,.01,0.,.13]):
        assert parity(b,q,ids,weights,terms,alpha,parents,sites,related,rings,x,fixed)<2e-5


def test_bad_algebra_stops_never_adjusts_or_injects():
    b,q,ids=fixture();ids,x=native_features(b,q,ids);fixed=np.zeros(3)
    x[0,0]+=1.
    with pytest.raises(ValueError,match='mismatch'):parity(b,q,ids,[1.]*11,[],0.,set(),set(),set(),[],x,fixed)


def test_guard_and_explicit_point_identity():
    b,q,ids=fixture();events=[]
    native_features(b,q,ids,lambda:events.append(True))
    assert len(events)==12
    with pytest.raises(ValueError):points(b,compose(b,q,[1.]*11,[],0.,set(),set(),set(),[]),['NOT_SIRET'])
