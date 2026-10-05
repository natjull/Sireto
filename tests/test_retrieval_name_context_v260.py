from types import SimpleNamespace
import numpy as np
import pytest

from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery
from src.xgb_matcher.retrieval_name_context_v260 import residual_name, augmented_parts
from src.xgb_matcher.retrieval_proof_v56 import components
from src.xgb_matcher.retrieval_corroborated_initial_v260 import search
from test_retrieval_corroborated_initial_v217 import tiny_native


@pytest.mark.parametrize('name,address,expected', [
    ('ALPHA BETA RUE DU LAC', '12 RUE DU LAC', 'ALPHA BETA'),
    ('RUE DU LAC ALPHA BETA', 'RUE DU LAC', 'ALPHA BETA'),
    ('ALPHA RUE DU LAC BETA', 'RUE DU LAC', 'ALPHA BETA'),
    ('ALPHA BETA LAC', 'RUE DU LAC', None),
    ('ALPHA RUE DU LAC', 'RUE DU LAC', None),
    ('ALPHA ALPHA RUE DU LAC', 'RUE DU LAC', None),
    ('ALPHA BETA DE LA', 'DE LA', None),
    ('ALPHA BETA RUE DU LAC', 'LAC DU RUE', None),
    ('123456789 ALPHA BETA RUE DU LAC', 'RUE DU LAC', 'ALPHA BETA'),
    ('ALPHA BETA RUE DU LAC RUE DU LAC', 'RUE DU LAC', 'ALPHA BETA'),
])
def test_query_only_all_shared_spans_and_residual_guard(name,address,expected):
    q=RetrievalQuery(name=name,address=address)
    assert residual_name(q)==expected and q.name==name and q.address==address


def scores(b, p):
    if p is None:return {}
    return {b._first(b.searcher.doc(loc).to_dict(),'siret'):float(s)
        for s,loc in b.searcher.search(p,limit=100,count=False).hits}


def test_real_native_coverage_improves_at_place_only_never_exact_alias():
    b=tiny_native()
    q=RetrievalQuery(name='COMMUNE DE ALPHA RUE BETA',address='RUE BETA',postcode='11111')
    original=components(b,q);new=augmented_parts(b,q)
    a=scores(b,original[0]);n=scores(b,new[0])
    assert n['10000000000001']>a['10000000000001']
    assert n['10000000000003']==pytest.approx(a['10000000000003'])
    for i in (1,2,3,4,5,7,9,10):assert scores(b,new[i])==scores(b,original[i])
    assert scores(b,new[2])=={} # Discontinuous residual is NOT a new exact alias.


def test_no_place_cannot_create_a_residual_proof():
    b=tiny_native();q=RetrievalQuery(name='COMMUNE DE ALPHA RUE BETA',address='RUE BETA')
    for a,n in zip(components(b,q),augmented_parts(b,q)):
        assert scores(b,a)==scores(b,n)


def test_no_removal_preserves_every_original_component_object(monkeypatch):
    from src.xgb_matcher import retrieval_name_context_v260 as m
    originals=[object() for i in range(11)]
    monkeypatch.setattr(m,'components',lambda b,q:originals)
    assert m.augmented_parts(None,RetrievalQuery(name='ALPHA BETA')) is originals


def test_one_global_collector100_original_owner_query_and_guard_restored():
    b=tiny_native();native=b.searcher;calls=[];texts=[]
    def queried(p,**kw):calls.append(kw);return native.search(p,**kw)
    b.searcher=SimpleNamespace(search=queried,doc=native.doc)
    guard=lambda:None
    def owners(text):texts.append(text);return set(),set(),{}
    e=SimpleNamespace(files=SimpleNamespace(guard=guard),name_owners=owners,
        related=lambda p,s:(set(),{}),radial_proofs=lambda geo:([],{}))
    q=RetrievalQuery(name='COMMUNE DE ALPHA RUE BETA',address='RUE BETA',postcode='11111')
    rows,t=search(b,e,q,{},np.ones(11),[],0.)
    assert calls==[dict(limit=100,count=False)] and texts==[q.name]
    assert len(rows)<=100 and len({r['siret'] for r in rows})==len(rows)
    assert t['original_query_preserved'] and t['residual_nominal_view_added']
    assert not t['invented_exact_alias'] and not t['actual217_fit_parity_claimed']
    assert not t['score_after_fixed_pool'] and e.files.guard is guard
