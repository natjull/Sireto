import numpy as np
import pytest
from src.xgb_matcher.retrieval_spatial_v43 import build_tree,nearest_sites,validate_arrays


def arrays(xy):
    xy=np.asarray(xy,dtype=np.float64)
    ids=10000000000000+np.arange(len(xy),dtype=np.uint64)
    parents=(ids//100000).astype(np.uint32)
    return xy,ids,parents


def test_full_corpus_exact_nearest100_matches_bruteforce():
    rng=np.random.default_rng(43)
    xy,ids,parents=arrays(rng.uniform(0,1000,size=(2000,2)))
    validate_arrays(xy,ids,parents); tree=build_tree(xy)
    point=np.array([456.1,731.4])
    found,trace=nearest_sites(tree,xy,ids,parents,point)
    dist=np.sqrt(np.sum((xy-point)**2,axis=1))
    expected=np.lexsort((ids,dist))[:100]
    assert [r['siret'] for r in found]==[f'{int(ids[i]):014d}' for i in expected]
    assert [r['distance_m'] for r in found]==dist[expected].tolist()
    assert trace['fixed_reservoir'] is None and not trace['approximate_search']


def test_co_located_sites_are_not_collapsed_and_ties_are_canonical():
    xy,ids,parents=arrays(np.zeros((500,2)))
    result,trace=nearest_sites(build_tree(xy),xy,ids,parents,[0,0])
    assert len(result)==100 and len({r['siret'] for r in result})==100
    assert [r['siret'] for r in result]==[f'{int(x):014d}' for x in ids[:100]]
    assert trace['boundary_tie_overflow']==400


def test_no_source_geography_or_active_filter_and_short_corpus():
    xy,ids,parents=arrays([[0,0],[500,500],[1000,1000]])
    found,_=nearest_sites(build_tree(xy),xy,ids,parents,[300,300])
    assert len(found)==3 and found[0]['siret']==f'{int(ids[1]):014d}'
    with pytest.raises(ValueError): nearest_sites(build_tree(xy),xy,ids,parents,[np.nan,0])
    with pytest.raises(ValueError): nearest_sites(build_tree(xy),xy,ids,parents,[0,0],101)


def test_membership_and_duplicate_keys_fail_closed():
    xy,ids,parents=arrays([[0,0],[0,0]])
    validate_arrays(xy,ids,parents)
    with pytest.raises(ValueError): validate_arrays(xy,ids,parents+1)
    with pytest.raises(ValueError): validate_arrays(xy,np.array([ids[0],ids[0]]),parents)
    with pytest.raises(ValueError): validate_arrays(xy.astype(np.float32),ids,parents)
