import inspect
from types import SimpleNamespace
import numpy as np
import pytest
from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery
from src.xgb_matcher.retrieval_public_initial_v282 import public_component,compose,search,RoleModels
from src.xgb_matcher.retrieval_corroborated_initial_v257 import compose as previous
from src.xgb_matcher.retrieval_proof_v56 import components
from src.xgb_matcher.retrieval_training_score_v231 import points
from test_retrieval_corroborated_initial_v217 import tiny_native

PARENTS=dict(commune={'100000000'},epci={'100000000'})
IDS=[f'100000000{i:05d}' for i in (1,2,3)]

def test_role_requires_leaf_parent_and_place_and_max_not_sum():
 b=tiny_native();q=RetrievalQuery(name='UNKNOWN',postcode='11111')
 p=public_component(b,components(b,q),PARENTS,[.3,.7])
 assert np.allclose(points(b,p,IDS),[.7,.7,0.])
 for i in (0,6):
  w=np.zeros(11);w[i]=1.
  v=points(b,compose(b,q,w,[],0.,set(),set(),set(),[],PARENTS,[.3,.7]),IDS)
  assert np.allclose(v,[.7,.7,0.])
 q=RetrievalQuery(name='UNKNOWN',address='ROAD',postcode='11111');w=np.zeros(11);w[0]=1.;w[6]=1.
 assert np.allclose(points(b,compose(b,q,w,[],0.,set(),set(),set(),[],PARENTS,[.3,.7]),IDS),[1.4,1.4,0.])

def test_role_not_exact_identity():
 b=tiny_native();q=RetrievalQuery(name='UNKNOWN',address='ROAD',postcode='11111')
 for i in (2,10):
  w=np.zeros(11);w[i]=1.
  assert np.allclose(points(b,compose(b,q,w,[],0.,set(),set(),set(),[],PARENTS,[1.,1.]),IDS),[0.,0.,0.])

@pytest.mark.parametrize('grades',[[float('nan'),0.],[-.1,0.],[1.1,0.],[.1]])
def test_invalid_role_grades(grades):
 b=tiny_native();q=RetrievalQuery(name='UNKNOWN',postcode='11111')
 with pytest.raises(ValueError):public_component(b,components(b,q),PARENTS,grades)

def test_no_invented_proof_missing_place_or_parent():
 b=tiny_native();q=RetrievalQuery(name='UNKNOWN')
 assert public_component(b,components(b,q),PARENTS,[1.,1.]) is None
 q=RetrievalQuery(name='UNKNOWN',postcode='11111')
 assert public_component(b,components(b,q),dict(commune=set(),epci=set()),[1.,1.]) is None

def test_original_native_metric_when_zero_roles():
 b=tiny_native();q=RetrievalQuery(name='COMMUNE DE ALPHA ECOLE BETA',address='ROAD',postcode='11111')
 w=np.ones(11);args=(b,q,w,[],0.,set(),set(),set(),[])
 assert np.allclose(points(b,compose(*args,PARENTS,[0.,0.]),IDS),points(b,previous(*args),IDS),atol=2e-5)

def test_one_global_collector_original_text_and_outside_public_reachable():
 b=tiny_native();native=b.searcher;calls=[];texts=[];join=[]
 def queried(p,**kw):calls.append(kw);return native.search(p,**kw)
 b.searcher=SimpleNamespace(search=queried,doc=native.doc);guard=lambda:None
 def owners(t):texts.append(t);return set(),set(),{}
 e=SimpleNamespace(files=SimpleNamespace(guard=guard),name_owners=owners,related=lambda p,s:(set(),{}),radial_proofs=lambda g:([],{}))
 def parents(q):join.append(q);return PARENTS
 q=RetrievalQuery(name='ALPHA',address='ROAD',postcode='11111')
 rows,t=search(b,e,q,{},np.ones(11),[],0.,SimpleNamespace(parents=parents),[.3,.7])
 assert calls==[dict(limit=100,count=False)] and texts==[q.name] and join==[q]
 assert set(IDS)<={r['siret'] for r in rows} and not t['public_is_global_hard_filter']
 assert not t['training_scenes_reopened'] and not t['actual217_fit_parity_claimed']
 assert e.files.guard is guard

def test_no_training_scene_loader_in_serving_role_class():
 text=inspect.getsource(RoleModels)
 assert 'trainer.training(' not in text and "['training_inputs']" not in text
 assert 'check_epochs' in text
