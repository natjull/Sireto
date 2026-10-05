from types import SimpleNamespace
import numpy as np
import pytest
from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery
from src.xgb_matcher.retrieval_administrative_owner_v289 import administrative_support,compose,search
from src.xgb_matcher.retrieval_corroborated_initial_v257 import compose as previous
from src.xgb_matcher.retrieval_training_score_v231 import points
from test_retrieval_corroborated_initial_v217 import tiny_native

PARENTS=dict(commune={'100000000'},epci={'100000000'})
IDS=[f'100000000{i:05d}' for i in (1,2,3)]

def test_independent_grade_max_roles_not_nominal_weight_amplification():
 b=tiny_native();q=RetrievalQuery(name='UNKNOWN',postcode='11111')
 p=administrative_support(b,q,PARENTS,[.3,.7])
 assert np.allclose(points(b,p,IDS),[.7,.7,0.])
 w=np.zeros(11)
 assert np.allclose(points(b,compose(b,q,w,[],0.,set(),set(),set(),[],PARENTS,[.3,.7]),IDS),[.7,.7,0.])
 w[0]=10.;w[6]=10.
 assert np.allclose(points(b,compose(b,q,w,[],0.,set(),set(),set(),[],PARENTS,[.3,.7]),IDS),[.7,.7,0.])

def test_address_can_corroborate_without_place_but_no_support_no_bonus():
 b=tiny_native();q=RetrievalQuery(name='UNKNOWN',address='ROAD')
 assert np.allclose(points(b,administrative_support(b,q,PARENTS,[.3,.7]),IDS),[.7,.7,.7])
 assert administrative_support(b,RetrievalQuery(name='UNKNOWN'),PARENTS,[.3,.7]) is None

def test_explicit_owner_relation_not_double_counted():
 b=tiny_native();q=RetrievalQuery(name='ALPHA',address='ROAD',postcode='11111')
 args=(b,q,np.ones(11),[],0.,{'100000000'},set(),set(),[])
 assert np.allclose(points(b,compose(*args,PARENTS,[1.,1.]),IDS),points(b,previous(*args),IDS),atol=2e-5)

@pytest.mark.parametrize('grades',[[float('nan'),0.],[.1],[-.1,0.],[0.,1.1]])
def test_invalid_grades_rejected(grades):
 b=tiny_native()
 with pytest.raises(ValueError):administrative_support(b,RetrievalQuery(name='UNKNOWN',postcode='11111'),PARENTS,grades)

def test_same_global_base_with_empty_roles():
 b=tiny_native();q=RetrievalQuery(name='ALPHA',address='ROAD',postcode='11111');args=(b,q,np.ones(11),[],0.,set(),set(),set(),[])
 assert np.allclose(points(b,compose(*args,dict(commune=set(),epci=set()),[1.,1.]),IDS),points(b,previous(*args),IDS),atol=2e-5)

def test_one_global_collect_original_text_and_outside_place_reachable():
 b=tiny_native();native=b.searcher;calls=[];texts=[]
 def queried(p,**kw):calls.append(kw);return native.search(p,**kw)
 b.searcher=SimpleNamespace(search=queried,doc=native.doc);guard=lambda:None
 def owners(t):texts.append(t);return set(),set(),{}
 e=SimpleNamespace(files=SimpleNamespace(guard=guard),name_owners=owners,related=lambda p,s:(set(),{}),radial_proofs=lambda g:([],{}))
 q=RetrievalQuery(name='ALPHA',address='ROAD',postcode='11111')
 rows,t=search(b,e,q,{},np.ones(11),[],0.,SimpleNamespace(parents=lambda q:PARENTS),[.3,.7])
 assert calls==[dict(limit=100,count=False)] and texts==[q.name] and set(IDS)<={r['siret'] for r in rows}
 assert t['nominal_components_unchanged_from_v257'] and not t['metric_equivalent_to_v282']
 assert not t['training_scenes_reopened'] and not t['public_is_global_hard_filter'] and e.files.guard is guard
