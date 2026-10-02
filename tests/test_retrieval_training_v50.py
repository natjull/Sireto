import numpy as np
import pytest

from scripts import retrieval_v42_indexed_metric as v42
from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery
from src.xgb_matcher.retrieval_training_v50 import align_scenes,serving_features
from tests.test_retrieval_v38_initial_top100 import backend


def test_native_serving_predicate_and_cached_feature_mask_agree(tmp_path):
    b=backend(tmp_path,[dict(legal_names='QQQQQQ',addresses='ZZZZZZ',insee='75056',postcode='75001'),
        dict(legal_names='ALPHA',addresses='RUE BLEUE',insee='75056',postcode='75001')])
    q=RetrievalQuery(name='ALPHA',address='RUE BLEUE',insee='75056',postcode='75001')
    ids=['10000000000001','10000000100001']; w=np.array([.2,.04,1.5,.4,0,0,1.4,0])
    _,raw=v42.training_features(b,q,ids)
    native={b._first(b.searcher.doc(loc).to_dict(),'siret'):float(s)
        for s,loc in b.searcher.search(v42.index_query(b,q,w),limit=2,count=False).hits}
    assert raw[0,6]>0 and ids[0] not in native
    for i,s in enumerate(ids):
        aligned=serving_features([*raw[i],.2,.1])
        assert np.isclose(np.array(aligned[:8])@w,native.get(s,0.),rtol=1e-5)
        assert aligned[8:]==[.2,.1]


def test_no_query_or_negative_removed_or_input_mutated():
    x=[0.,0.,0.,0.,0.,0.,1.,1.,.5,.5]
    scenes=[dict(positive_features=x,negatives=[dict(features=x,siren='123456789')])]
    aligned,counts=align_scenes(scenes)
    assert scenes[0]['positive_features']==x and aligned[0]['positive_features']==[0.]*8+[.5,.5]
    assert len(aligned)==len(scenes) and len(aligned[0]['negatives'])==1
    assert counts['positive_changed']==counts['negative_changed']==1


@pytest.mark.parametrize('x',[[0.]*9,[0.]*9+[float('nan')],[-1.]+[0.]*9])
def test_invalid_cached_features_fail_closed(x):
    with pytest.raises(ValueError): serving_features(x)


def test_address_only_is_eligible():
    x=[0.,1.,0.,1.,0.,0.,1.,0.,.2,0.]
    assert serving_features(x)==x
