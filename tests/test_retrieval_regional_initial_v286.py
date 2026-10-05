from types import SimpleNamespace
import numpy as np
from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery
from src.xgb_matcher.retrieval_regional_initial_v286 import compose,search
from src.xgb_matcher.retrieval_corroborated_initial_v257 import compose as previous
from src.xgb_matcher.retrieval_training_score_v231 import points
from test_retrieval_regional_abbreviation_v285 import engine,REGIONS

IDS=[f'100000000{i:05d}' for i in (11,12,13)]

def test_new_nominal_proof_is_not_identity_and_outside_place_global_open():
 b=engine();q=RetrievalQuery(name='AZB ANCIENNE REGION',postcode='11111')
 for i in (0,6):
  w=np.zeros(11);w[i]=1.;v=points(b,compose(b,q,w,[],0.,set(),set(),set(),[],REGIONS),IDS)
  old=points(b,previous(b,q,w,[],0.,set(),set(),set(),[]),IDS)
  assert np.isclose(v[0],1.) and v[0]>old[0] and np.allclose(v[1:],old[1:])
 for i in (2,10):
  w=np.zeros(11);w[i]=1.
  assert np.allclose(points(b,compose(b,q,w,[],0.,set(),set(),set(),[],REGIONS),IDS),[0.,0.,0.])

def test_original_metric_without_known_regional_qualifier():
 b=engine();q=RetrievalQuery(name='ALPHA',address='ROAD',postcode='11111');args=(b,q,np.ones(11),[],0.,set(),set(),set(),[])
 assert np.allclose(points(b,compose(*args,REGIONS),IDS),points(b,previous(*args),IDS),atol=2e-5)

def test_one_global_collector_and_original_owners_query():
 b=engine();original=b.searcher;calls=[];texts=[]
 def called(p,**kw):calls.append(kw);return original.search(p,**kw)
 b.searcher=SimpleNamespace(search=called,doc=original.doc);guard=lambda:None
 def owners(text):texts.append(text);return set(),set(),{}
 e=SimpleNamespace(files=SimpleNamespace(guard=guard),name_owners=owners,related=lambda p,s:(set(),{}),radial_proofs=lambda g:([],{}))
 q=RetrievalQuery(name='AZB ANCIENNE REGION',address='ROAD',postcode='11111')
 rows,t=search(b,e,q,{},np.ones(11),[],0.,REGIONS)
 assert calls==[dict(limit=100,count=False)] and texts==[q.name]
 assert set(IDS)<={r['siret'] for r in rows} and not t['regional_is_global_hard_filter']
 assert not t['actual217_fit_parity_claimed'] and not t['label_lookup'] and e.files.guard is guard
