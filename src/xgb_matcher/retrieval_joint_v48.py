"""Same joint formula, simpler zero-score eligibility and bounded point access.

The text eligibility predicate is an OR of token membership, not its BM25
score. Native term sets therefore replace its discarded scoring machinery.
Point lookups may use the locality-pruned lexical query only when non-local
lexical score is mathematically zero. Spatial admission is never filtered.
"""
import numpy as np

from scripts import retrieval_v38_initial_top100 as v38
from scripts import retrieval_v42_indexed_metric as v42
from src.xgb_matcher.retrieval_joint_v46 import LexicalStream as OldLexicalStream,SpatialStream,geo_features,geo_parameters,validate_weights
from src.xgb_matcher.retrieval_joint_v47 import collect,UncertifiedSearchBudget


def index_query(backend,query,weights):
    w=np.asarray(weights,dtype=np.float64)
    if w.shape!=(8,) or not np.isfinite(w).all() or np.any(w<0) or not np.any(w>0):
        raise ValueError('Invalid native lexical metric')
    tv=backend._tantivy; parts=v42.components(backend,query); terms=[]
    for fields,charfield,text in ((v38.NAMES,'current_name_ngrams',query.name),(v38.ADDRESSES,'address_ngrams',query.address)):
        words=v38.tokens(text)
        for field in fields:
            if words: terms.append(tv.Query.term_set_query(backend.index.schema,field,words))
        chars=v38.tokens(text,True)
        if chars: terms.append(tv.Query.term_set_query(backend.index.schema,charfield,chars))
    if not terms: return tv.Query.empty_query()
    doc=tv.Query.const_score_query(tv.Query.term_query(backend.index.schema,'document_type','siret','basic'),0.)
    eligible=tv.Query.const_score_query(tv.Query.boolean_query([(tv.Occur.Should,p) for p in terms]),0.)
    clauses=[(tv.Occur.Must,doc),(tv.Occur.Must,eligible)]
    clauses.extend((tv.Occur.Should,tv.Query.boost_query(p,float(value))) for p,value in zip(parts,w) if p is not None and value>0)
    return tv.Query.boolean_query(clauses)


class LexicalStream(OldLexicalStream):
    def __init__(self,backend,query,weights):
        super().__init__(backend,query,weights)
        if not self.exhausted:
            self.parsed=index_query(backend,query,weights)
            if self.pruned_zero_nonlocal:
                tv=backend._tantivy; place=v42.components(backend,query)[6]
                self.parsed=tv.Query.boolean_query([(tv.Occur.Must,self.parsed),
                    (tv.Occur.Must,tv.Query.const_score_query(place,0.))])


def search(backend,tree,keys,parents,query,geo,weights,max_seconds=20.):
    w=validate_weights(weights); lexical=LexicalStream(backend,query,w[:8]); point,_,reliable=geo_parameters(geo)
    gw=float(w[8]+w[9]*reliable); spatial=SpatialStream(tree,keys,parents,geo,gw); tv=backend._tantivy
    # Any omitted non-local point has lexical score zero by proof, not by
    # convenience or truth status. Its spatial score is still computed.
    parsed_point=None if lexical.exhausted else lexical.parsed
    def score_rows(rows):
        ids=[r['siret'] for r in rows]; unknown=[s for s in ids if s not in lexical.cache]
        lexical.cache.update({s:0. for s in unknown})
        if unknown and parsed_point is not None:
            identity=tv.Query.const_score_query(tv.Query.term_set_query(backend.index.schema,'siret',unknown),0.)
            parsed=tv.Query.boolean_query([(tv.Occur.Must,parsed_point),(tv.Occur.Must,identity)]); found=set()
            for score,loc in backend.searcher.search(parsed,limit=len(unknown),count=False).hits:
                s=backend._first(backend.searcher.doc(loc).to_dict(),'siret')
                if s not in lexical.cache or s in found: raise ValueError('Random access identity drift')
                found.add(s); lexical.cache[s]=float(score)
        g=geo_features(keys,tree.data,ids,geo)[:,0]*gw
        return {s:float(lexical.cache[s]+g[i]) for i,s in enumerate(ids)}
    channels={k:s for k,s in (('lexical',lexical),('spatial',spatial)) if not s.exhausted}
    rows,trace=collect(channels,score_rows,max_seconds=max_seconds)
    trace['zero_nonlocal_lexical_pruning']=lexical.pruned_zero_nonlocal; trace['geo_query_available']=point is not None
    trace['point_score_locality_pruned_by_zero_bound']=lexical.pruned_zero_nonlocal
    trace['eligibility_native_termsets']=True
    return rows,trace
