from types import SimpleNamespace
import numpy as np
import pytest

from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery
from src.xgb_matcher.retrieval_commune_context_v268 import Context,residual_name,augmented_parts
from src.xgb_matcher.retrieval_corroborated_initial_v269 import compose,search
from src.xgb_matcher.retrieval_corroborated_initial_v257 import compose as previous
from src.xgb_matcher.retrieval_training_score_v231 import points
from test_retrieval_compound_initialisms_v222 import engine

@pytest.mark.parametrize('name,names,expected',[
 ('ALPHA BETA PARIS',{'PARIS'},'ALPHA BETA'),
 ('PARIS ALPHA BETA PARIS',{'PARIS'},'ALPHA BETA'),
 ('ALPHA LYON BETA PARIS',{'LYON','PARIS'},'ALPHA BETA'),
 ('ALPHA BETA VILLE NOUVELLE',{'VILLE NOUVELLE'},'ALPHA BETA'),
 ('AZB PARIS',{'PARIS'},'AZB'),
 ('SAS AZB PARIS',{'PARIS'},'SAS AZB'),
 ('AB PARIS',{'PARIS'},None),
 ('AB123 PARIS',{'PARIS'},None),
 ('MAIRIE PARIS',{'PARIS'},None),
 ('PARIS',{'PARIS'},None),
 ('ALPHA BETA PARISIEN',{'PARIS'},None),
 ('ALPHA BETA VILLE GRANDE NOUVELLE',{'VILLE NOUVELLE'},None),
 ('123456789 ALPHA BETA PARIS',{'PARIS'},'ALPHA BETA'),
])
def test_all_official_context_spans_and_original_kept(name,names,expected):
 q=RetrievalQuery(name=name)
 assert residual_name(q,names)==expected and q.name==name

def test_all_postal_memberships_and_insee_no_prioritization_or_idfeature():
 c=Context.__new__(Context)
 c.postcode={'11111':{'PARIS','LYON'}}
 c.insee={'22222':{'VILLE NOUVELLE'}}
 assert c.names(RetrievalQuery(name='',postcode='11111',insee='22222'))=={'PARIS','LYON','VILLE NOUVELLE'}
 assert c.names(RetrievalQuery(name='',postcode='11112',insee='22223'))==set()

def test_official_context_sigle_is_place_only_not_exact_identity():
 b=engine();ids=['10000000000004','10000000000006']
 q=RetrievalQuery(name='AZB PARIS',insee='11111')
 for i,expected in ((0,[1.,0.]),(6,[1.,0.]),(2,[0.,0.]),(10,[0.,0.])):
  w=np.zeros(11);w[i]=1.
  assert np.array_equal(points(b,compose(b,q,w,[],0.,set(),set(),set(),[],{'PARIS'}),ids),expected)
 q=RetrievalQuery(name='AZB PARIS')
 for a,n in zip(augmented_parts(b,q,set()),augmented_parts(b,q,{'PARIS'})):
  assert (a is None)==(n is None)
  if a is not None:assert np.allclose(points(b,a,ids),points(b,n,ids))

def test_no_official_context_native_algebra_previous257_preserved():
 b=engine();q=RetrievalQuery(name='AZB',insee='11111')
 ids=[f'100000000{i:05d}' for i in range(1,7)]
 args=(b,q,[2.2,.7,.9,.5,1.4,1.1,1.2,.2,.01,.1,.13],[],0.,set(),set(),set(),[])
 assert np.allclose(points(b,compose(*args,set()),ids),points(b,previous(*args),ids),atol=2e-5,rtol=2e-6)

def test_original_owner_query_and_one_collector100():
 b=engine();native=b.searcher;calls=[];texts=[]
 def queried(p,**kw):calls.append(kw);return native.search(p,**kw)
 b.searcher=SimpleNamespace(search=queried,doc=native.doc);guard=lambda:None
 def owners(text):texts.append(text);return set(),set(),{}
 e=SimpleNamespace(files=SimpleNamespace(guard=guard),name_owners=owners,
   related=lambda p,s:(set(),{}),radial_proofs=lambda geo:([],{}))
 ctx=SimpleNamespace(names=lambda q:{'PARIS'},rows=35014)
 q=RetrievalQuery(name='AZB PARIS',insee='11111')
 rows,t=search(b,e,q,{},np.ones(11),[],0.,ctx)
 assert calls==[dict(limit=100,count=False)] and texts==[q.name] and len(rows)<=100
 assert t['context_nominal_view_added'] and t['original_query_preserved']
 assert not t['invented_exact_alias'] and not t['score_after_fixed_pool']
 assert e.files.guard is guard
