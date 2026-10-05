"""Explicit score quantization, NOT top-class pruning or an unchanged metric."""
import numpy as np
from .retrieval_activity_postings_v263 import collect_groups as continuous_collect,native_proof

LEVELS=128


def quantized(grades):
 x=np.asarray(grades,dtype=float)
 if x.ndim!=1 or not len(x) or not np.isfinite(x).all() or np.any((x<0)|(x>1)):
  raise ValueError('All finite bounded class relevances required')
 return np.floor(LEVELS*x)/LEVELS


def collect_groups(evidence,q,grades,mapping,chunk=131072):
 values=quantized(grades)
 groups,stats=continuous_collect(evidence,q,values,mapping,chunk)
 stats.update(continuous_classes=len(values),positive_continuous_classes=int(np.count_nonzero(np.asarray(grades)>0)),
  positive_quantized_classes=int(np.count_nonzero(values>0)),score_quantization_levels=LEVELS,
  quantization_max_absolute_error_less_than=1/LEVELS,top_class_pruning=False,
  quantized_zero_is_not_identity_or_global_exclusion=True)
 return groups,stats
