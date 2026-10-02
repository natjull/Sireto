import numpy as np
import pytest
from scipy.spatial import cKDTree
from src.xgb_matcher.retrieval_geo_postings_v64 import radial_memberships


def test_full_inclusive_nested_radii_and_no_nearest_cap():
    xy=np.array([[0,0],[25,0],[100,0],[500,0],[2000,0],[2001,0]],dtype=float)
    keys=np.arange(10000000000001,10000000000007,dtype=np.uint64)
    geo=dict(spatial_crs_compatible=True,ban_x_l93=0,ban_y_l93=0,ban_uncertainty_m=50,ban_declared_weak=True)
    bands,trace=radial_memberships(cKDTree(xy),keys,geo)
    assert [len(b) for b in bands]==[2,3,4,5] and trace['reliable'] is False
    assert bands[-1]==[str(s) for s in keys[:5]] and trace['truncated'] is False


def test_guard_stops_not_truncates_dense_sites():
    keys=np.arange(10000000000001,10000000000006,dtype=np.uint64)
    geo=dict(spatial_crs_compatible=True,ban_x_l93=0,ban_y_l93=0,ban_uncertainty_m=50,ban_declared_weak=False)
    with pytest.raises(RuntimeError,match='STOP_COMPLETE_GEO_JOIN_TOO_LARGE'):
        radial_memberships(cKDTree(np.zeros((5,2))),keys,geo,max_joined_sites=4)


def test_missing_or_incompatible_geometry_has_no_predicate():
    bands,trace=radial_memberships(cKDTree(np.zeros((1,2))),np.array([10000000000001]),dict(spatial_crs_compatible=False))
    assert bands==[[],[],[],[]] and trace['joined_sites']==0 and not trace['compatible']


def test_native_geometry_score_matches_complete_source_predicate_without_filtering_far_sites(tmp_path):
    from tests.test_retrieval_proof_v56 import backend
    from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery
    from src.xgb_matcher.retrieval_proof_v56 import point_features
    from src.xgb_matcher.retrieval_geo_postings_v64 import query
    rng=np.random.default_rng(64)
    b=backend(tmp_path,[dict(legal_names='ALPHA',addresses='12 RUE BLEUE') for _ in range(350)])
    q=RetrievalQuery(name='ALPHA',address='12 RUE BLEUE',number='12',insee='75056')
    ids=[f'{100000000+i}00001' for i in range(350)]
    xy=rng.uniform(0,3000,(350,2)); keys=np.array([int(s) for s in ids],dtype=np.uint64)
    g=dict(spatial_crs_compatible=True,ban_x_l93=0,ban_y_l93=0,ban_uncertainty_m=50,ban_declared_weak=False)
    bands,trace=radial_memberships(cKDTree(xy),keys,g)
    w=rng.uniform(.1,2,19); ordered,proof=point_features(b,q,ids)
    geo=np.array([[s in band for band in bands] for s in ordered],dtype=float)
    expected=proof@w[:11]+geo@(w[11:15]+w[15:19])
    native={b._first(b.searcher.doc(loc).to_dict(),'siret'):float(score)
        for score,loc in b.searcher.search(query(b,q,w,bands,trace['reliable']),limit=350,count=False).hits}
    assert len(native)==350  # Geo does not exclude the complete lexical corpus.
    np.testing.assert_allclose([native[s] for s in ordered],expected,atol=1e-5,rtol=0)
    hits=b.searcher.search(query(b,q,w,bands,trace['reliable']),limit=100,count=False).hits
    assert len(hits)==100 and min(score for score,_ in hits)>=sorted(expected,reverse=True)[99]-1e-5
