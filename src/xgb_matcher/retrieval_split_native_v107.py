"""Frozen V67 weights as two independently indexed score components."""
import numpy as np
from src.xgb_matcher.retrieval_factored_native_v105 import query,BoundedNativeStream


def queries(backend,q,weights,terms):
    w=np.asarray(weights,dtype=np.float64)
    if w.shape!=(16,) or not np.isfinite(w).all() or np.any(w<0):
        raise ValueError('Frozen split native model drift')
    lexical=w.copy(); lexical[11:15]=0.
    initialism=np.zeros(16); initialism[11:15]=w[11:15]
    return {'lexical':query(backend,q,lexical,terms),'initialism':query(backend,q,initialism,[])}


def streams(backend,q,weights,terms):
    return {k:BoundedNativeStream(backend,p) for k,p in queries(backend,q,weights,terms).items()}
