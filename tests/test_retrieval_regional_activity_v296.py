from types import SimpleNamespace
import numpy as np
from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery
from src.xgb_matcher import retrieval_regional_activity_v296 as m
from src.xgb_matcher.retrieval_regional_initial_v286 import compose
from test_retrieval_regional_abbreviation_v285 import engine,REGIONS

def evidence():
 guard=lambda:None
 return SimpleNamespace(files=SimpleNamespace(guard=guard),name_owners=lambda t:(set(),set(),{}),
  related=lambda p,s:(set(),{}),radial_proofs=lambda g:([],{}))

def test_no_activity_preserves_complete_successful_regional_metric(monkeypatch):
 b=engine();q=RetrievalQuery(name='AZB ANCIENNE REGION',postcode='11111');w=[1.]*11
 previous=b.searcher.search(compose(b,q,w,[],0.,set(),set(),set(),[],REGIONS),limit=100,count=False).hits
 expected=sorted((float(s),b._first(b.searcher.doc(loc).to_dict(),'siret')) for s,loc in previous)
 monkeypatch.setattr(m,'collect_groups',lambda *a:({},{}));e=evidence();guard=e.files.guard
 rows,t=m.search(b,e,q,{},w,[],0.,np.zeros(2),np.array([[0]],np.int32),REGIONS)
 assert sorted((r['source_score'],r['siret']) for r in rows)==expected and e.files.guard is guard
 assert t['continuous_grades_quantized'] and not t['metric_equivalent_to_v272'] and not t['previous_stop_outputs_reused']

def test_activity_and_regional_whole_alias_fused_before_one_collect(monkeypatch):
 b=engine();native=b.searcher;calls=[]
 def queried(p,**kw):calls.append(kw);return native.search(p,**kw)
 b.searcher=SimpleNamespace(search=queried,doc=native.doc)
 monkeypatch.setattr(m,'collect_groups',lambda *a:({115/128:{'10000000000011'}},{}))
 e=evidence();q=RetrievalQuery(name='AZB ANCIENNE REGION',postcode='11111')
 rows,t=m.search(b,e,q,{},[1.]*11,[],0.,np.zeros(2),np.array([[0]],np.int32),REGIONS)
 assert calls==[dict(limit=100,count=False)] and rows[0]['siret']=='10000000000011'
 assert t['activity_coefficient']==1. and not t['score_after_fixed_pool'] and t['activity_quantization_max_absolute_error']==1/128
