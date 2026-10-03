"""V90 typed-scalar lookups; physical-only adapter: V87 metric and complete official scope unchanged."""
import time
from src.xgb_matcher.retrieval_scoped_paths_v87 import (OwnerFacts,ScopedPaths,NameStream,NativeStream,literal_scores,
    SpatialStream,geo_features,geo_parameters,native_query)
from src.xgb_matcher.retrieval_ranked_path_v89 import collect,UncertifiedPathBudget

import numpy as np
from src.xgb_matcher.retrieval_scoped_paths_v87 import OwnerFacts as LegacyOwnerFacts
from src.xgb_matcher.retrieval_succession_csr_v84 import SuccessionCSR as LegacySuccessionCSR

def typed_position(keys,number):
    number=int(number)
    if not 0<=number<=np.iinfo(keys.dtype).max: raise ValueError('Join-key range drift')
    return int(np.searchsorted(keys,keys.dtype.type(number)))

class OwnerFacts(LegacyOwnerFacts):
    def references(self,label,key):
        a=self.owner[label]; p=typed_position(a['keys'],key)
        if p==len(a['keys']) or a['keys'][p]!=int(key): return a['facts'][:0]
        lo,hi=a['offsets'][p:p+2]
        return a['facts'][int(lo):int(hi)]

class SuccessionCSR(LegacySuccessionCSR):
    def endpoint_position(self,siret):
        if not (isinstance(siret,str) and len(siret)==14 and siret.isascii() and siret.isdigit()):
            raise ValueError('Invalid endpoint join key')
        keys=self.a['endpoint_keys']; p=typed_position(keys,int(siret))
        return p if p<len(keys) and keys[p]==int(siret) else None

    def parent_endpoint_positions(self,siren):
        if not (isinstance(siren,str) and len(siren)==9 and siren.isascii() and siren.isdigit()):
            raise ValueError('Invalid parent join key')
        keys=self.a['parent_keys']; p=typed_position(keys,int(siren))
        if p==len(keys) or keys[p]!=int(siren): return iter(())
        lo,hi=self.a['parent_offsets'][p:p+2]
        return iter(self.a['parent_endpoints'][int(lo):int(hi)])


class TimedNativeStream(NativeStream):
    def __init__(self,*args):
        super().__init__(*args); self.point_calls=0; self.point_seconds=0.; self.sorted_seconds=0.
    def fetch(self,n):
        began=time.monotonic(); result=super().fetch(n); self.sorted_seconds+=time.monotonic()-began; return result
    def complete(self,ids):
        self.point_calls+=bool(any(s not in self.cache for s in ids)); began=time.monotonic()
        result=super().complete(ids); self.point_seconds+=time.monotonic()-began; return result


def search(backend,tree,keys,parents,word_index,owners,graph,q,geo,weights,terms,alpha,*,path_weight=64.,max_seconds=20.):
    began=time.monotonic(); scores,word_trace=literal_scores(word_index,q.name)
    names=NameStream(scores); paths=ScopedPaths(owners,graph,scores)
    native=TimedNativeStream(backend,native_query(backend,q,weights,terms,alpha))
    _,_,reliable=geo_parameters(geo); reliability=1. if reliable else .5
    spatial=SpatialStream(tree,keys,parents,geo,reliability)
    def complete(rows):
        ids=[r['siret'] for r in rows]; base=native.complete(ids)
        values=geo_features(keys,tree.data,ids,geo)[:,0]*reliability
        return {s:dict(native_score=base[s],path_name_score=paths.best(s)['score'],site_score=float(values[i]))
                for i,s in enumerate(ids)}
    remaining=max_seconds-(time.monotonic()-began)
    if remaining<=0: raise TimeoutError('STOP_WORD_INDEX_PREPARATION_COST')
    try:
        rows,trace=collect(native,names,spatial,paths.targets,complete,path_weight=path_weight,
                           name_score_max=1.,site_score_max=reliability,max_seconds=remaining,page_size=256)
    except UncertifiedPathBudget as exc:
        exc.trace.update(word_index=word_trace,native_point_calls=native.point_calls,
                         native_point_seconds=native.point_seconds,native_sorted_seconds=native.sorted_seconds,
                         seconds_including_preparation=time.monotonic()-began)
        raise
    for row in rows: row['scoped_name_proof']=paths.best(row['siret'])
    trace.update(word_index=word_trace,seconds=time.monotonic()-began,unique_documents_visited=trace['unique_targets_completed'],
                 native_corpus_unchanged=True,graph_scope='V84_ENDPOINTS_ONLY',geo_query_available=spatial.point is not None,
                 ban_reliability_factor=reliability,native_postings_visits_not_instrumented=True,
                 proof_co_validity_unknown=True,identifier_features=False,
                 native_point_calls=native.point_calls,native_point_seconds=native.point_seconds,
                 native_sorted_seconds=native.sorted_seconds,physical_fact_page_completion_batching=True,bound_dominant_source_traversal=True,physical_first_page_size=256,typed_scalar_join_lookups=True)
    return rows,trace
