"""Real native toy index fixtures only; no national recall assertion."""
from types import SimpleNamespace
import numpy as np
from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery
from src.xgb_matcher import retrieval_activity_abbreviation_v272 as module
from src.xgb_matcher.retrieval_activity_abbreviation_v272 import search
from src.xgb_matcher.retrieval_corroborated_initial_v257 import compose
from test_retrieval_compound_initialisms_v222 import engine

def evidence():
 guard=lambda:None
 return SimpleNamespace(files=SimpleNamespace(guard=guard),name_owners=lambda t:(set(),set(),{}),
   related=lambda p,s:(set(),{}),radial_proofs=lambda geo:([],{}))

def test_no_activity_preserves_successful_abbreviation_metric(monkeypatch):
 b=engine();q=RetrievalQuery(name='AZB',insee='11111');w=[1.]*11
 previous=b.searcher.search(compose(b,q,w,[],0.,set(),set(),set(),[]),limit=100,count=False).hits
 expected=sorted((float(s),b._first(b.searcher.doc(loc).to_dict(),'siret')) for s,loc in previous)
 monkeypatch.setattr(module,'collect_groups',lambda *a:({},{}))
 e=evidence();guard=e.files.guard
 rows,t=search(b,e,q,{},w,[],0.,np.zeros(2),np.array([[0]],np.int32))
 assert sorted((r['source_score'],r['siret']) for r in rows)==expected
 assert e.files.guard is guard and not t['metric_equivalent_to_v265'] and not t['previous_stop_outputs_reused']

def test_both_proofs_fused_before_one_collect100(monkeypatch):
 b=engine();native=b.searcher;calls=[]
 def queried(p,**kw):calls.append(kw);return native.search(p,**kw)
 b.searcher=SimpleNamespace(search=queried,doc=native.doc)
 monkeypatch.setattr(module,'collect_groups',lambda *a:({.9:{'10000000000004'}},{}))
 e=evidence();q=RetrievalQuery(name='AZB',insee='11111')
 rows,t=search(b,e,q,{},[1.]*11,[],0.,np.zeros(2),np.array([[0]],np.int32))
 assert calls==[dict(limit=100,count=False)] and rows[0]['siret']=='10000000000004'
 assert len(rows)<=100 and len({r['siret'] for r in rows})==len(rows)
 assert t['activity_coefficient']==1. and not t['score_after_fixed_pool']
