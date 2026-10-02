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
