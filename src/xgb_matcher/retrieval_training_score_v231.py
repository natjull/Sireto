"""Supervised TRAIN/postseal point scores of the ACTUAL immutable V217 kernel.

Never use point target lookups during initial evaluation generation. The
caller must supply an independently authorized supervised TRAIN namespace
or a complete postseal diagnostic. This helper emits no product candidates.
"""
import numpy as np
from .retrieval_corroborated_initial_v217 import compose


def points(native, parsed, ids):
    ids=sorted(set(ids))
    if any(not isinstance(s,str) or len(s)!=14 or not s.isascii() or not s.isdigit() for s in ids):
        raise ValueError('Explicit supervised point references required')
    tv=native._tantivy
    gate=tv.Query.const_score_query(tv.Query.term_set_query(native.index.schema,'siret',ids),0.)
    query=tv.Query.boolean_query([(tv.Occur.Must,gate),(tv.Occur.Must,parsed)])
    values={s:0. for s in ids}
    for score,loc in native.searcher.search(query,limit=len(ids),count=False).hits:
        s=native._first(native.searcher.doc(loc).to_dict(),'siret')
        if s not in values or not np.isfinite(score) or score<0:raise ValueError('Actual finite native point score required')
        values[s]=float(score)
    return np.asarray([values[s] for s in ids])


def native_features(native,q,ids,guard=lambda:None):
    """Compile unit coefficients, including containment and coherent-site proof.

    Eleven independent components of217, NOT the older11 components56.
    Field role, max/constant/conditional semantics come from the serving
    kernel itself. Identifier set only selects supervised observation rows.
    """
    ids=sorted(set(ids));vectors=np.empty((len(ids),11))
    for i in range(11):
        guard();unit=np.zeros(11);unit[i]=1.
        parsed=compose(native,q,unit,[],0.,set(),set(),set(),[])
        vectors[:,i]=points(native,parsed,ids)
    guard()
    return ids,vectors


def fixed_scores(native,q,ids,terms,alpha,parents,sites,related,rings,guard=lambda:None):
    """Sparse semantic + official owners/successions + XY, independent of11w."""
    guard();parsed=compose(native,q,np.zeros(11),terms,alpha,parents,sites,related,rings)
    values=points(native,parsed,ids);guard();return values


def parity(native,q,ids,weights,terms,alpha,parents,sites,related,rings,features,fixed,guard=lambda:None):
    guard();actual=points(native,compose(native,q,weights,terms,alpha,parents,sites,related,rings),ids)
    predicted=np.asarray(features)@np.asarray(weights)+np.asarray(fixed)
    # Native f32 tree addition vs float64 feature dot; no score-changing
    # correction, tolerance cannot become an admission/promotion heuristic.
    if not np.allclose(actual,predicted,atol=2e-5,rtol=2e-6):
        raise ValueError('Actual217/native decomposed TRAIN score mismatch')
    guard();return float(np.max(np.abs(actual-predicted),initial=0.))
