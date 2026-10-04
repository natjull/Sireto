"""Native/source synthetic algebra only. No CRM quality guarantee."""
from types import SimpleNamespace

import numpy as np
import pytest

from test_retrieval_corroborated_initial_v217 import tiny_native
from test_retrieval_official_fragments_v236 import fixture as source_fixture
from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery
from src.xgb_matcher.retrieval_corroborated_initial_v217 import compose as original
from src.xgb_matcher.retrieval_official_alias_proofs_v243 import compose,fragment_grades,search,Evidence
from src.xgb_matcher.retrieval_training_score_v231 import points


def test_whole_query_and_every_eligible_span_exact_official_source():
    g=fragment_grades('OLD BRAND SITE')
    assert g['OLD BRAND SITE']==1. and g['OLD BRAND']==1. and g['BRAND']==pytest.approx(1/3)
    assert fragment_grades('BRAND')=={'BRAND':1.}
    assert '12345678900001' not in fragment_grades('BRAND 12345678900001')


def test_no_alias_all_components_parity_with_actual217_not_extra_bonuses():
    b=tiny_native();q=RetrievalQuery(name='COMMUNE DE ALPHA ECOLE BETA',address='ROAD',postcode='11111',number='1')
    ids=[f'100000000{i:05d}' for i in (1,2,3)];w=[2.2,.7,.9,.5,1.4,1.1,1.2,.2,.01,.1,.13]
    p={'100000000'};s={ids[1]};related={ids[2]};rings=[({ids[0]},.8)]
    before=points(b,original(b,q,w,[('beta',1.)],2.,p,s,related,rings),ids)
    after=points(b,compose(b,q,w,[('beta',1.)],2.,[(p,s,1.)],[(related,.5)],rings,[]),ids)
    assert np.allclose(before,after,atol=2e-5,rtol=2e-6)


def test_historical_site_alias_is_primary_name_proof_not_current_name_assertion():
    b=tiny_native();q=RetrievalQuery(name='OLD BRAND',address='ROAD',postcode='11111')
    ids=[f'100000000{i:05d}' for i in (1,2,3)];groups=[(set(),{ids[0]},1.)]
    for index,expected in ((0,1.),(2,1.),(6,1.),(8,2.)):
        unit=np.zeros(11);unit[index]=1.
        a=points(b,compose(b,q,unit,[],0.,[],[],[],groups),ids)
        assert a[0]==pytest.approx(expected) and a[1]==a[2]==0.
    # Site proof does NOT enter sibling IDs; all temporal names remain hypotheses.


def test_global_alias_keeps_off_zone_sites_reachable_conditional_proofs_do_not():
    b=tiny_native();q=RetrievalQuery(name='OLD BRAND',address='ROAD',postcode='11111')
    ids=[f'100000000{i:05d}' for i in (1,2,3)];groups=[({'100000000'},set(),1.)]
    w=np.zeros(11);w[0]=1.
    assert np.all(points(b,compose(b,q,w,[],0.,[],[],[],groups),ids)==1.)
    w[0]=0.;w[6]=1.
    assert np.array_equal(points(b,compose(b,q,w,[],0.,[],[],[],groups),ids),[1.,1.,0.])


def test_partial_alias_confidence_preserved_no_double_evidence():
    b=tiny_native();q=RetrievalQuery(name='OLD BRAND SITE',address='ROAD',postcode='11111')
    ids=[f'100000000{i:05d}' for i in (1,2,3)];groups=[(set(),{ids[0]},1/3)]
    w=np.zeros(11);w[2]=1.
    a=points(b,compose(b,q,w,[],0.,[],[],[],groups*3),ids)
    assert a[0]==pytest.approx(1/3) and a[1]==a[2]==0.
    groups.append((set(),{ids[0]},1.))
    assert points(b,compose(b,q,w,[],0.,[],[],[],groups),ids)[0]==1.


def test_all_owner_rows_preserved_fulltext_lookup_not_shortened_allwords():
    old,arrays=source_fixture();e=object.__new__(Evidence);e.files=old.files;e.items=old.items;e.lookup=old.lookup
    groups,trace=e.fragment_owners('ALPHA BETA')
    assert groups==[({'100000000'},{'20000000000003'},1.)]
    assert trace['owner_facts']==4 and trace['whole_official_alias_exact_only'] is True
    arrays['scope_code'][-1]=1
    with pytest.raises(ValueError,match='scope conflict'):e.fragment_owners('ALPHA BETA')


def test_one_native_collector_and_explicit_changed_feature_semantics():
    b=tiny_native();old=b.searcher;calls=[]
    def searched(parsed,**kw):calls.append(kw);return old.search(parsed,**kw)
    b.searcher=SimpleNamespace(search=searched,doc=old.doc)
    guard=lambda:None
    e=SimpleNamespace(files=SimpleNamespace(guard=guard),name_owners=lambda name:(set(),set(),{}),
        fragment_owners=lambda name:([(set(),{'10000000000001'},1.)],{}),
        related=lambda p,s:(set(),{}),radial_proofs=lambda geo:([],{}))
    found,trace=search(b,e,RetrievalQuery(name='OLD BRAND',address='ROAD',postcode='11111'),{},[1.]*11,[],0.)
    assert len(found)==3 and calls==[dict(limit=100,count=False)]
    assert trace['new_native_feature_semantics'] and not trace['actual217_fit_parity_claimed']
    assert trace['official_any_period_name_proof_not_current_name_assertion']
    assert trace['label_lookup'] is False and e.files.guard is guard
