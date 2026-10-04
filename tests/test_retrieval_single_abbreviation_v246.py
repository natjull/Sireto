import numpy as np
import pytest

from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery
from src.xgb_matcher.retrieval_single_abbreviation_v246 import tokens,patterns,membership,component
from src.xgb_matcher.retrieval_proof_v56 import components
from src.xgb_matcher.retrieval_corroborated_initial_v247 import compose
from src.xgb_matcher.retrieval_corroborated_initial_v217 import compose as original
from src.xgb_matcher.retrieval_training_score_v231 import points
from test_retrieval_compound_initialisms_v222 import engine
from test_retrieval_corroborated_initial_v217 import tiny_native
from types import SimpleNamespace


def test_semantic_whole_query_not_embedded_fragments():
    assert tokens('SAS ABCD SARL')==['ABCD']
    for q in ('AB','ASS HOPITAL ST MICHEL ET ST VINCENT','CRIJ REGION','123456789','[ND]','ABCDEFG'):
        assert tokens(q)==[] and patterns(q)==[]
    assert len(patterns('ABCDEF'))==31


@pytest.mark.parametrize('q,a',[('AZB','AZIMUTH BRAVO'),('AZB','ALPHA ZULU BRAVO'),
    ('ABCDEF','ABC DEFENSE'),('ABCD','SAS ALPHA DE BRAVO CHARLIE DELTA SARL')])
def test_every_partition_and_legal_boundary(q,a):
    assert membership(q,[a])


def test_no_cross_alias_siblings_or_extra_words():
    for aliases in (['ALPHA','ZULU BRAVO'],['ALPHA ZULU BRAVO EXTRA'],['BRAVO ZULU ALPHA']):
        assert not membership('AZB',aliases)


def test_native_whole_alias_and_soft_place():
    b=engine();q=RetrievalQuery(name='AZB',insee='11111')
    hits=b.searcher.search(component(b,q,components(b,q)),limit=100,count=False).hits
    assert {b._first(b.searcher.doc(loc).to_dict(),'siret') for _,loc in hits}=={
        '10000000000004','10000000000005'}
    assert component(b,RetrievalQuery(name='AZB'),components(b,RetrievalQuery(name='AZB'))) is None


def test_unchanged_algebra_without_abbreviation():
    b=tiny_native();q=RetrievalQuery(name='COMMUNE DE ALPHA ECOLE BETA',address='ROAD',postcode='11111')
    ids=[f'100000000{i:05d}' for i in (1,2,3)];w=[2.2,.7,.9,.5,1.4,1.1,1.2,.2,.01,.1,.13]
    args=(b,q,w,[('beta',1.)],2.,{'100000000'},set(),set(),[({ids[0]},.8)])
    assert np.allclose(points(b,compose(*args),ids),points(b,original(*args),ids),atol=2e-5,rtol=2e-6)


def test_abbreviation_is_name_not_exact_string():
    b=engine();q=RetrievalQuery(name='AZB',insee='11111');ids=['10000000000004','10000000000006']
    for i,expected in ((0,[1.,0.]),(6,[1.,0.]),(2,[0.,0.])):
        w=np.zeros(11);w[i]=1.
        assert np.array_equal(points(b,compose(b,q,w,[],0.,set(),set(),set(),[]),ids),expected)


def test_one_global_collector_and_no_post_emission_pool():
    from src.xgb_matcher.retrieval_corroborated_initial_v247 import search
    b=engine();original_searcher=b.searcher;calls=[]
    def queried(parsed,**kw):calls.append(kw);return original_searcher.search(parsed,**kw)
    b.searcher=SimpleNamespace(search=queried,doc=original_searcher.doc)
    guard=lambda:None
    e=SimpleNamespace(files=SimpleNamespace(guard=guard),name_owners=lambda text:(set(),set(),{}),
        related=lambda parents,sites:(set(),{}),radial_proofs=lambda geo:([],{}))
    found,trace=search(b,e,RetrievalQuery(name='AZB',insee='11111'),{},[1.]*11,[],0.)
    assert calls==[dict(limit=100,count=False)] and len(found)<=100
    assert len({r['siret'] for r in found})==len(found)
    assert trace['name_feature_semantics_changed'] and not trace['actual217_fit_parity_claimed']
    assert trace['label_lookup'] is False and e.files.guard is guard
