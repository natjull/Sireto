"""Exact physical pruning of an initial index metric, not a new ranking recipe.

When all non-local score components are identically zero, documents outside
the query's administrative locality have a proven score upper bound of zero.
If the native local top100 has a strictly positive cutoff, none can enter it.
Otherwise the original complete query is executed. No learned weight changes,
truth lookups, fixed candidate reservoir or post-retrieval classifier occur.
"""
import math

import numpy as np

from scripts import retrieval_v42_indexed_metric as v42


def search(backend, query, weights):
    parsed=v42.index_query(backend,query,weights)
    w=np.asarray(weights,dtype=np.float64)
    trace=dict(pruned=False,excluded_score_upper_bound=None,fixed_reservoir=None)
    # Global name, address and their conjunction are the only components
    # capable of a positive score outside place; require their exact zero.
    if np.any(w[[0,1,4]]!=0):
        return v42.search(backend,query,w),trace
    place=v42.components(backend,query)[6]
    if place is None: return v42.search(backend,query,w),trace
    tv=backend._tantivy
    local=tv.Query.boolean_query([(tv.Occur.Must,parsed),
        (tv.Occur.Must,tv.Query.const_score_query(place,0.))])
    hits=backend.searcher.search(local,limit=100,count=False).hits
    if len(hits)<100 or min(score for score,_ in hits)<=0:
        return v42.search(backend,query,w),trace
    rows=[]
    for score,loc in hits:
        doc=backend.searcher.doc(loc).to_dict()
        siren,siret=(backend._first(doc,k) for k in ('siren','siret'))
        if not (len(siren)==9 and siren.isdigit() and len(siret)==14 and siret.isdigit()
                and siret.startswith(siren) and math.isfinite(score)):
            raise ValueError('Official identity/score drift')
        rows.append(dict(siren=siren,siret=siret,score=float(score)))
    if len({r['siret'] for r in rows})!=100: raise ValueError('Duplicate initial output')
    trace.update(pruned=True,excluded_score_upper_bound=0.,native_cutoff=min(r['score'] for r in rows))
    return sorted(rows,key=lambda r:(-r['score'],r['siret'])),trace
