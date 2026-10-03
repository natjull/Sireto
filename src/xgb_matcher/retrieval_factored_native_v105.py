"""Factor the four official-current-alias predicates into one native subtree."""
import numpy as np
from src.xgb_matcher.retrieval_initialisms_v67 import FEATURES,patterns,EXACT_FIELDS
from src.xgb_matcher.retrieval_proof_v56 import components
from src.xgb_matcher.retrieval_sparse_query_v59 import compiled_component
from src.xgb_matcher.retrieval_scoped_paths_v90 import TimedNativeStream


def query(backend,q,weights,terms):
    w=np.asarray(weights,dtype=np.float64); tv=backend._tantivy
    if w.shape!=(16,) or not np.isfinite(w).all() or np.any(w<0):
        raise ValueError('Frozen initialism model drift')
    base=components(backend,q)
    scored=[tv.Query.boost_query(p,float(value)) for p,value in zip(base,w[:11]) if p is not None and value>0]
    regexes=[tv.Query.const_score_query(tv.Query.regex_query(backend.index.schema,f,p),1.)
        for p in patterns(q.name) for f in EXACT_FIELDS]
    bonuses=[]
    if regexes and np.any(w[11:15]>0):
        if w[11]>0: bonuses.append(tv.Query.const_score_query(tv.Query.all_query(),float(w[11])))
        for part,value in ((base[4],w[12]),(base[5],w[13]),(base[7],w[14])):
            if part is not None and value>0: bonuses.append(tv.Query.boost_query(part,float(value)))
        if bonuses:
            alias=tv.Query.disjunction_max_query(regexes,tie_breaker=0.)
            gate=tv.Query.const_score_query(alias,0.)
            bonus=tv.Query.boolean_query([(tv.Occur.Should,p) for p in bonuses])
            scored.append(tv.Query.boolean_query([(tv.Occur.Must,gate),(tv.Occur.Must,bonus)]))
    semantic=compiled_component(backend,q,terms)
    if semantic is not None and w[15]>0: scored.append(tv.Query.boost_query(semantic,float(w[15])))
    if not scored: return tv.Query.empty_query()
    site=tv.Query.const_score_query(tv.Query.term_query(backend.index.schema,'document_type','siret','basic'),0.)
    return tv.Query.boolean_query([(tv.Occur.Must,site),(tv.Occur.Must,tv.Query.boolean_query([(tv.Occur.Should,p) for p in scored]))])


class BoundedNativeStream(TimedNativeStream):
    def __init__(self,*args):
        super().__init__(*args); self.unread_bound=float('inf')

    def fetch(self,n):
        rows=super().fetch(n)
        self.unread_bound=0. if self.exhausted else rows[-1]['score']
        return rows

    def bound(self,siret):
        if siret in self.cache:
            return dict(native_score=self.cache[siret],native_upper_bound=self.cache[siret])
        return dict(native_score=None,native_upper_bound=self.unread_bound)
