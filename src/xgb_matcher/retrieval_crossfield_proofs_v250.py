"""Typed cross-field/conditional proofs compiled BEFORE initial collection.

No CRM alias dictionary or candidate reranker. Original217/228 proof head
is fixed; the three additional coefficients will be TRAIN-only learned.
"""
import math
from dataclasses import replace

import numpy as np

from .retrieval_corroborated_initial_v217 import compose as original
from .retrieval_proof_v56 import components

FEATURES=('crm_name_in_official_address_at_place',
          'current_exact_name_at_place','address_coverage_at_number_and_place')


def parts(backend,q):
    tv=backend._tantivy; base=components(backend,q)
    def const(p):return tv.Query.const_score_query(p,0.) if p is not None else None
    def both(*items):
        return tv.Query.boolean_query([(tv.Occur.Must,p) for p in items]) if all(p is not None for p in items) else None
    gate=const(base[4]);zero=np.zeros(11)
    unit=zero.copy();unit[2]=1.
    exact=original(backend,q,unit,[],0.,set(),set(),set(),[])
    # Crossed text remains address evidence; it never asserts legal identity
    # or an exact historical alias. Same leaf physical address, not siblings.
    crossed=components(backend,replace(q,address=q.name))[1]
    return [both(crossed,gate),both(exact,gate),both(base[1],const(base[5]))]


def compose(backend,q,weights,terms,alpha,parents,sites,related,rings,extra_weights):
    extra=np.asarray(extra_weights,dtype=float)
    if extra.shape!=(3,) or not np.isfinite(extra).all() or np.any((extra<0)|(extra>64)):
        raise ValueError('Three finite bounded nonnegative TRAIN coefficients')
    tv=backend._tantivy
    original_query=original(backend,q,weights,terms,alpha,parents,sites,related,rings)
    additions=[tv.Query.boost_query(p,float(w)) for p,w in zip(parts(backend,q),extra) if p is not None and w>0]
    if not additions:return original_query
    leaf=tv.Query.const_score_query(tv.Query.term_query(backend.index.schema,'document_type','siret','basic'),0.)
    return tv.Query.boolean_query([(tv.Occur.Must,leaf),(tv.Occur.Must,
        tv.Query.boolean_query([(tv.Occur.Should,p) for p in [original_query,*additions]]))])


def train_points(backend,q,ids):
    """TRAIN-only caller with explicit supervised references, never serving."""
    from .retrieval_training_score_v231 import points
    return np.column_stack([points(backend,p,ids) if p is not None else np.zeros(len(ids)) for p in parts(backend,q)])


def conditional_parity(original_features,extra_features):
    x=np.asarray(original_features,dtype=float);y=np.asarray(extra_features,dtype=float)
    if x.ndim!=2 or x.shape[1]!=11 or y.shape!=(len(x),3) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Complete aligned actual native proof matrices')
    expected=np.column_stack([x[:,2]*(x[:,4]>0),x[:,1]*(x[:,5]>0)])
    if not np.allclose(y[:,1:],expected,atol=2e-5,rtol=2e-6):
        raise ValueError('Native conditional proofs differ from actual217 semantics')
    if np.any(y<0) or np.any(y>1.00002):raise ValueError('Bounded extra proofs')
    return float(np.max(np.abs(y[:,1:]-expected),initial=0.))
