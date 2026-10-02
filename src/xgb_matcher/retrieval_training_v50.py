"""Align cached training features with the mandatory serving text predicate.

This is a training correctness repair, not a new retrieval architecture.
Missing text does not remove a query or document: spatial features remain.
"""
from copy import deepcopy

import numpy as np

from src.xgb_matcher.retrieval_joint_v46 import fit_metric


def serving_features(values):
    x=np.asarray(values,dtype=np.float64)
    if x.shape!=(10,) or not np.isfinite(x).all() or np.any(x<0):
        raise ValueError('Invalid cached joint features')
    x=x.copy()
    if x[0]==0 and x[1]==0:
        x[:8]=0.
    return x.tolist()


def align_scenes(scenes):
    aligned=deepcopy(scenes)
    counts=dict(queries=len(scenes),negative_pairs=0,positive_changed=0,negative_changed=0,
                positives_retained_without_text=0)
    for r in aligned:
        if r['positive_features'] is not None:
            original=r['positive_features']; transformed=serving_features(original)
            counts['positive_changed']+=original!=transformed
            counts['positives_retained_without_text']+=not bool(original[0] or original[1])
            r['positive_features']=transformed
        for n in r['negatives']:
            original=n['features']; transformed=serving_features(original)
            counts['negative_changed']+=original!=transformed
            counts['negative_pairs']+=1
            n['features']=transformed
    return aligned,counts


def fit_aligned(scenes,held,registry,ridge=.0005):
    aligned,counts=align_scenes(scenes)
    fitted=fit_metric(aligned,held,registry,ridge)
    fitted['training_feature_policy']='ZERO_ALL_LEXICAL_IF_NO_NAME_OR_ADDRESS_TEXT_KEEP_GEO'
    return fitted,counts
