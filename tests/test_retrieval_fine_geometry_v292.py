import numpy as np
import pytest
from src.xgb_matcher.retrieval_fine_geometry_v292 import grades,LEVELS,SCALE_M

def test_monotonic_fine_distance_not_same_old_band():
 d=np.array([0.,25.,101.,115.,135.,299.,500.,2000.]);g=grades(d,True)
 assert g[0]==128 and np.all(np.diff(g)<=0) and g[2]>g[5]
 assert LEVELS==128 and SCALE_M==500.

@pytest.mark.parametrize('reliable',[True,False])
def test_explicit_approximation_bound_all_distances(reliable):
 d=np.linspace(0,100000,10001);actual=(1. if reliable else .5)/(1+d/500)
 q=grades(d,reliable)/128
 assert np.all(q<=actual+1e-14) and np.max(actual-q)<1/128+1e-14

def test_weak_ban_remains_weaker_not_identity():
 d=np.array([0.,25.,115.,300.,2000.])
 assert np.all(grades(d,False)<=grades(d,True)) and grades([0],False)[0]==64

@pytest.mark.parametrize('bad',[[float('nan')],[float('inf')],[-1.]])
def test_bad_physical_distance_is_not_shared_missing_evidence(bad):
 with pytest.raises(ValueError):grades(bad,True)

def test_whole_physical_coordinate_scan_disjoint_postings_no_nearest_k(tmp_path):
 from src.xgb_matcher.retrieval_fine_geometry_v292 import Evidence
 from src.xgb_matcher.retrieval_structural_channels_v210 import SourceFiles
 from src.xgb_matcher.retrieval_v20_workbench import record
 point=np.array([700000.,6600000.]);dist=[101.,299.,2001.,float('nan'),25.]
 xy=np.array([point+[d,0.] for d in dist],dtype=np.float64)
 keys=np.array([10000000000001+i for i in range(5)],dtype=np.uint64)
 np.save(tmp_path/'xy.npy',xy,allow_pickle=False);np.save(tmp_path/'keys.npy',keys,allow_pickle=False)
 e=Evidence.__new__(Evidence);e.files=SourceFiles(lambda:None);e.sites=5
 e.xy_ref=record(tmp_path/'xy.npy');e.keys_ref=record(tmp_path/'keys.npy')
 geo=dict(spatial_crs_compatible=True,ban_x_l93=point[0],ban_y_l93=point[1],ban_uncertainty_m=150.,ban_declared_weak=True)
 rings,t=e.radial_proofs(geo,chunk=2);e.files.verify_after()
 emitted={s:g for ids,g in rings for s in ids}
 assert len(emitted)==3 and emitted['10000000000001']>emitted['10000000000002']
 assert emitted['10000000000005']>emitted['10000000000001']
 assert t['sites_scanned']==5 and t['candidate_or_nearest_k_cut'] is None and not t['hard_geographic_filter']
 assert t['outside_radius_reachable_in_global_native_query']

def test_missing_ban_never_opens_coordinate_sources():
 from src.xgb_matcher.retrieval_fine_geometry_v292 import Evidence
 e=Evidence.__new__(Evidence)
 rings,t=e.radial_proofs({})
 assert rings==[] and t['sites_scanned']==0 and not t['compatible']
