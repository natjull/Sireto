import numpy as np
import pytest

from scripts import retrieval_v42_indexed_metric as v42
from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery
from src.xgb_matcher.retrieval_index_pruning_v44 import search
from tests.test_retrieval_v38_initial_top100 import backend


def test_positive_local_cutoff_exact_native_output_even_boundary_ties(tmp_path):
    b=backend(tmp_path,[dict(legal_names='ALPHA',addresses='RUE BLEUE',
        insee='75056' if i%3 else '99999',postcode='75001' if i%3 else '99999') for i in range(750)])
    q=RetrievalQuery(name='ALPHA',address='RUE BLEUE',insee='75056',postcode='75001')
    w=np.array([0,0,1.,.1,0,0,2.,0])
    found,trace=search(b,q,w)
    assert found==v42.search(b,q,w)
    assert trace['pruned'] and trace['excluded_score_upper_bound']==0
    assert trace['fixed_reservoir'] is None and trace['native_cutoff']>0


@pytest.mark.parametrize('n',[10,150])
def test_positive_nonlocal_score_disables_pruning(tmp_path,n):
    b=backend(tmp_path,[dict(legal_names='ALPHA') for _ in range(n)])
    q=RetrievalQuery(name='ALPHA',insee='75056')
    w=np.array([.000001,0,1.,0,0,0,2.,0])
    found,trace=search(b,q,w)
    assert not trace['pruned'] and found==v42.search(b,q,w)


def test_short_local_output_falls_back_without_losing_nonlocal_sites(tmp_path):
    b=backend(tmp_path,[dict(legal_names='ALPHA',insee='75056' if i<10 else '99999',
        postcode='75001' if i<10 else '99999') for i in range(150)])
    q=RetrievalQuery(name='ALPHA',insee='75056',postcode='75001')
    w=np.array([0,0,1.,0,0,0,2.,0])
    found,trace=search(b,q,w)
    assert len(found)==100 and not trace['pruned'] and found==v42.search(b,q,w)


def test_missing_locality_and_zero_cutoff_have_no_unsafe_pruning(tmp_path):
    b=backend(tmp_path,[dict(legal_names='ALPHA') for _ in range(150)])
    w=np.array([0,0,1.,0,0,0,2.,0])
    for q in (RetrievalQuery(name='ALPHA'),RetrievalQuery(name='BETA',address='RUE BLEUE',insee='75056')):
        found,trace=search(b,q,w)
        assert not trace['pruned'] and found==v42.search(b,q,w)
