import numpy as np
import pytest
from src.xgb_matcher.retrieval_quantized_activity_v295 import quantized,collect_groups,LEVELS
from src.xgb_matcher.retrieval_activity_postings_v263 import block_groups
from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery
from test_retrieval_activity_postings_v263 import arrays

def test_all_classes_retained_no_top_class_cut_and_explicit_score_error():
 x=np.linspace(0,1,2140);y=quantized(x)
 assert y.shape==x.shape and LEVELS==128 and np.all(y<=x) and np.max(x-y)<1/128+1e-15
 assert y[-1]==1 and np.all(np.diff(y)>=0)

@pytest.mark.parametrize('bad',[[float('nan')],[-.1],[1.1],[],[[.1,.2]]])
def test_invalid_scores_not_silently_repaired(bad):
 with pytest.raises(ValueError):quantized(bad)

def test_same_quantized_class_grade_groups_do_not_sum_or_filter_known_place():
 g,c=block_groups(*arrays(),RetrievalQuery(name='N',postcode='11111',insee='11111'),quantized([.401,.402]),np.array([[0,-1],[1,-1]],np.int32))
 assert g=={51/128:{'10000000000001','10000000000002'}} and c['at_place']==3

def test_collect_calls_complete_scanner_with_all_quantized_classes(monkeypatch):
 from src.xgb_matcher import retrieval_quantized_activity_v295 as m
 x=np.linspace(0,1,2140);e=object();q=object();mapping=object();calls=[]
 def whole(ev,qu,gr,ma,ch):
  assert ev is e and qu is q and ma is mapping and ch==7
  assert np.array_equal(gr,quantized(x));calls.append(len(gr));return {},dict(scanned=43896818)
 monkeypatch.setattr(m,'continuous_collect',whole)
 groups,t=m.collect_groups(e,q,x,mapping,7)
 assert not groups and calls==[2140] and t['scanned']==43896818 and not t['top_class_pruning']
 assert t['quantization_max_absolute_error_less_than']==1/128
