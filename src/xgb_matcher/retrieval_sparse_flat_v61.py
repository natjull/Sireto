"""Flatten an equivalent native sparse query; no model/metric change.

One shared SIRET guard, lexical components in the same declared order,
semantic component last. IEEE native scorer rounding remains observable.
"""
import math
import numpy as np
from src.xgb_matcher.retrieval_proof_v56 import components,LEXICAL
from src.xgb_matcher.retrieval_sparse_query_v59 import compiled_component

def query(backend,q,weights,terms,alpha):
    tv=backend._tantivy; weights=np.asarray(weights)
    if weights.shape!=(len(LEXICAL),) or not np.isfinite(weights).all() or np.any(weights<0) or not math.isfinite(alpha) or alpha<0:
        raise ValueError('Flattened sparse query drift')
    parts=[tv.Query.boost_query(part,float(w)) for part,w in zip(components(backend,q),weights) if part is not None and w>0]
    sem=compiled_component(backend,q,terms)
    if sem is not None and alpha>0: parts.append(tv.Query.boost_query(sem,float(alpha)))
    if not parts: return tv.Query.empty_query()
    site=tv.Query.const_score_query(tv.Query.term_query(backend.index.schema,'document_type','siret','basic'),0.)
    return tv.Query.boolean_query([(tv.Occur.Must,site),(tv.Occur.Must,tv.Query.boolean_query([(tv.Occur.Should,p) for p in parts]))])

def search(backend,q,weights,terms,alpha):
    rows=[]
    for score,loc in backend.searcher.search(query(backend,q,weights,terms,alpha),limit=100,count=False).hits:
        doc=backend.searcher.doc(loc).to_dict(); s=backend._first(doc,'siret'); p=backend._first(doc,'siren')
        if not (len(s)==14 and s.isdigit() and len(p)==9 and p.isdigit() and s.startswith(p) and math.isfinite(score)):
            raise ValueError('Official native sparse identity/score drift')
        rows.append(dict(siret=s,siren=p,score=float(score)))
    if len(rows)>100 or len({r['siret'] for r in rows})!=len(rows): raise ValueError('Native sparse cap drift')
    rows.sort(key=lambda r:(-r['score'],r['siret']))
    return rows,dict(certified=True,native_exact_top100=True,fixed_reservoir=None,unique_documents_visited=len(rows),
        internal_postings_visits_not_instrumented=True,learned_expansion_terms=terms,semantic_alpha=alpha,
        score_cutoff=rows[-1]['score'] if rows else None,ties='CANONICAL_AMONG_RETURNED_NATIVE_SCORE_EQUALS')
