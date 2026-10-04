import numpy as np
import pytest
from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery
from src.xgb_matcher.retrieval_crossfield_proofs_v250 import parts,compose,train_points,conditional_parity
from src.xgb_matcher.retrieval_corroborated_initial_v217 import compose as original
from src.xgb_matcher.retrieval_training_score_v231 import native_features,points
from test_retrieval_corroborated_initial_v217 import tiny_native


def test_crossfield_is_same_leaf_address_not_legal_name_or_sibling():
    import tantivy
    b=tiny_native();q=RetrievalQuery(name='ROAD',address='UNRELATED',postcode='11111')
    writer=b.index.writer();writer.add_document(tantivy.Document.from_dict(dict(document_type='siret',
        siret='10000000000004',siren='100000000',legal_names='ROAD',addresses='OTHER',
        postcode='11111',insee='11111'),b.index.schema));writer.commit();b.index.reload();b.searcher=b.index.searcher()
    ids=[f'100000000{i:05d}' for i in (1,2,3,4)]
    y=train_points(b,q,ids)
    assert np.array_equal(y[:,0],[1.,1.,0.,0.])
    assert np.all(y[:,1:]==0.)


def test_conditionals_preserve_fractional_confidence_and_native_parity():
    b=tiny_native();q=RetrievalQuery(name='COMMUNE DE ALPHA ECOLE BETA',address='ROAD',postcode='11111',number='1')
    ids=[f'100000000{i:05d}' for i in (1,2,3)]
    ordered,x=native_features(b,q,ids,lambda:None);y=train_points(b,q,ordered)
    assert conditional_parity(x,y)<2e-5
    broken=y.copy();broken[0,1]+=1.
    with pytest.raises(ValueError):conditional_parity(x,broken)


def test_zero_coefficients_return_original_global_query_algebra():
    b=tiny_native();q=RetrievalQuery(name='ROAD',address='ROAD',postcode='11111',number='1')
    ids=[f'100000000{i:05d}' for i in (1,2,3)]
    args=(b,q,[1.]*11,[],0.,set(),set(),set(),[])
    assert np.allclose(points(b,compose(*args,[0.]*3),ids),points(b,original(*args),ids))
    base=points(b,original(*args),ids);extra=train_points(b,q,ids);w=np.array([.4,.7,.2])
    assert np.allclose(points(b,compose(*args,w),ids),base+extra@w,atol=2e-5,rtol=2e-6)


def test_missing_or_outside_place_does_not_invent_condition_but_original_global_stays():
    b=tiny_native();q=RetrievalQuery(name='ROAD',address='ROAD');ids=[f'100000000{i:05d}' for i in (1,2,3)]
    assert np.all(train_points(b,q,ids)==0.)
    assert points(b,compose(b,q,[1.]*11,[],0.,set(),set(),set(),[],[1.]*3),ids)[0]>0


@pytest.mark.parametrize('w',[[1.],[-1.,0.,0.],[65.,0.,0.],[float('nan'),0.,0.]])
def test_extra_coefficient_contract(w):
    with pytest.raises(ValueError):compose(None,None,[1.]*11,[],0.,set(),set(),set(),[],w)
