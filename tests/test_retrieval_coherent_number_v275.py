import numpy as np
import pytest
import tantivy
from types import SimpleNamespace

from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery,TantivyBackend
from src.xgb_matcher.retrieval_coherent_number_v275 import core_terms,component
from src.xgb_matcher.retrieval_proof_v56 import components
from src.xgb_matcher.retrieval_training_score_v231 import points
from src.xgb_matcher.retrieval_corroborated_initial_v276 import compose,search
from src.xgb_matcher.retrieval_corroborated_initial_v257 import compose as previous
from test_retrieval_corroborated_initial_v217 import tiny_native

@pytest.mark.parametrize('address,number,expected',[
 ('12 RUE DE LA PAIX','12',['paix']),
 ('12 BD VICTOR HUGO','12',['hugo','victor']),
 ('4 RUE DU 4 SEPTEMBRE','4',['4','septembre']),
 ('12 RUE 1918','12',['1918']),
 ('12 ROUTE D10','12',['d10']),
 ('12 RUE','12',[]),
 ('12 RUE ALPHA 123456789 12345678900001','12',['alpha']),
 ('12 RUE ALPHA ALPHA','12',['alpha']),
])
def test_generic_address_syntax_all_unique_core_no_global_digit_drop(address,number,expected):
 q=RetrievalQuery(name='NAME',address=address,number=number)
 assert core_terms(q)==expected and q.address==address

def road_engine():
 b=tiny_native();w=b.index.writer()
 for i,a,num,place in [(11,'12 RUE VICTOR HUGO','12','11111'),
   (12,'12 RUE ALPHA BETA','12','11111'),(13,'14 RUE VICTOR HUGO','14','11111'),
   (14,'12 RUE VICTOR HUGO','12','22222'),(15,'12 RUE VICTOR BETA','12','11111')]:
  w.add_document(tantivy.Document.from_dict(dict(document_type='siret',
   siret=f'100000000{i:05d}',siren='100000000',legal_names='ALPHA',
   legal_names_exact='ALPHA',addresses=a,number=num,insee=place,postcode=place),b.index.schema))
 w.commit();b.index.reload()
 return SimpleNamespace(_tantivy=tantivy,index=b.index,searcher=b.index.searcher(),_first=TantivyBackend._first)

def test_native_number_requires_this_leaf_road_and_place_bounded_not_additive():
 b=road_engine();q=RetrievalQuery(name='ALPHA',address='12 RUE VICTOR HUGO',number='12',postcode='11111')
 ids=[f'100000000{i:05d}' for i in range(11,16)]
 p=component(b,q,components(b,q))
 assert np.allclose(points(b,p,ids),[1.,0.,0.,0.,.5])
 w=np.zeros(11);w[5]=1.
 assert np.allclose(points(b,compose(b,q,w,[],0.,set(),set(),set(),[]),ids),[1.,0.,0.,0.,.5])

@pytest.mark.parametrize('q',[
 RetrievalQuery(name='ALPHA',address='12 RUE',number='12',postcode='11111'),
 RetrievalQuery(name='ALPHA',address='12 RUE VICTOR',postcode='11111'),
 RetrievalQuery(name='ALPHA',address='12 RUE VICTOR',number='12'),
])
def test_missing_road_number_place_has_no_invented_number_proof(q):
 b=road_engine();assert component(b,q,components(b,q)) is None

def test_all_nonnumber_native_scores_preserved():
 b=road_engine();q=RetrievalQuery(name='ALPHA',address='12 RUE VICTOR HUGO',number='12',postcode='11111')
 ids=[f'100000000{i:05d}' for i in range(11,16)]
 w=np.ones(11);w[5]=0.
 args=(b,q,w,[],0.,set(),set(),set(),[])
 assert np.allclose(points(b,compose(*args),ids),points(b,previous(*args),ids),atol=2e-5,rtol=2e-6)

def test_no_road_global_filter_original_owner_and_one_collect():
 b=road_engine();native=b.searcher;calls=[];texts=[]
 def queried(p,**kw):calls.append(kw);return native.search(p,**kw)
 b.searcher=SimpleNamespace(search=queried,doc=native.doc);guard=lambda:None
 def owners(t):texts.append(t);return set(),set(),{}
 e=SimpleNamespace(files=SimpleNamespace(guard=guard),name_owners=owners,
  related=lambda p,s:(set(),{}),radial_proofs=lambda g:([],{}))
 q=RetrievalQuery(name='ALPHA',address='12 RUE VICTOR HUGO',number='12',postcode='11111')
 rows,t=search(b,e,q,{},np.ones(11),[],0.)
 assert calls==[dict(limit=100,count=False)] and texts==[q.name]
 assert {f'100000000{i:05d}' for i in range(11,16)} <= {r['siret'] for r in rows}
 assert not t['road_is_global_hard_filter'] and not t['number_without_road_bonus']
 assert not t['actual217_fit_parity_claimed'] and e.files.guard is guard
