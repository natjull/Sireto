"""Distance-informed native proof with explicit <=1/128 quantization error.

Same500m scale, BAN reliability factors and radius coverage as257. ALL native
coordinates considered, no nearest-K candidate reservoir or output rerank.
"""
import numpy as np
from .retrieval_corroborated_initial_v257 import Evidence as PreviousEvidence,search as previous_search
from .retrieval_joint_v46 import geo_parameters

LEVELS=128
SCALE_M=500.


def grades(distance_m,reliable):
 d=np.asarray(distance_m,dtype=float)
 if not np.isfinite(d).all() or (d<0).any():raise ValueError('Finite nonnegative physical distance required')
 factor=1. if reliable else .5
 return np.floor(LEVELS*factor/(1.+d/SCALE_M)).astype(np.int16)


class Evidence(PreviousEvidence):
 def radial_proofs(self,geo,chunk=131072):
  point,sigma,reliable=geo_parameters(geo)
  if point is None:return [],dict(compatible=False,sites_scanned=0,hard_geographic_filter=False)
  radius=max(2000.,3.*sigma);path=self.files.file(self.xy_ref)
  keys=self.files.mapped(self.keys_ref,np.uint64,(self.sites,));groups={};zero=0
  try:
   for lo in range(0,self.sites,chunk):
    self.files.guard();a=np.load(path,mmap_mode='r',allow_pickle=False)
    try:
     if a.dtype!=np.float64 or a.shape!=(self.sites,2) or a.flags.writeable:
      raise ValueError('Complete readonly physical Lambert93 coordinates required')
     xy=a[lo:min(lo+chunk,self.sites)].copy()
    finally:a._mmap.close()
    d2=np.sum((xy-point)**2,axis=1);valid=np.isfinite(xy).all(axis=1)&(d2<=radius**2+1e-6)
    local=np.flatnonzero(valid);g=grades(np.sqrt(d2[local]),reliable);values=keys[local+lo]
    zero+=int(np.count_nonzero(g==0))
    for value in np.unique(g[g>0]):
     groups.setdefault(int(value),set()).update(f'{int(s):014d}' for s in values[g==value])
   proofs=[(groups[g],g/LEVELS) for g in sorted(groups,reverse=True)]
   return proofs,dict(compatible=True,reliable=bool(reliable),radius_m=radius,sites_scanned=self.sites,
    grade_counts={str(g):len(ids) for g,ids in sorted(groups.items())},zero_quantized_memberships=zero,
    scale_m=SCALE_M,levels=LEVELS,absolute_quantization_error_less_than=1/LEVELS,
    metric='FLOOR128_OF_BAN_FACTOR_OVER_ONE_PLUS_DISTANCE_OVER_500M',
    candidate_or_nearest_k_cut=None,hard_geographic_filter=False,
    outside_radius_reachable_in_global_native_query=True,appended_coordinates_unknown=True)
  finally:keys._mmap.close();self.files.check_epochs()


def search(*args,**kwargs):
 found,trace=previous_search(*args,**kwargs)
 trace.update(source_hypothesis='V292_DISTANCE_FINE_GEOGRAPHY_SAME_500M_SCALE_BAN_FACTORS_RADIUS',
  geometry_metric_equivalent_to_v257=False,geometry_quantization_levels=LEVELS,
  geometry_max_absolute_quantization_error=1/LEVELS)
 return found,trace
