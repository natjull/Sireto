"""V101 query-term projection and physical batches; V99 full metric unchanged."""
import time
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.xgb_matcher.retrieval_name_postings_v86 import WordPostings
from src.xgb_matcher.retrieval_proof_v56 import signal
from src.xgb_matcher.retrieval_national_paths_v99 import NationalScopedPaths
from src.xgb_matcher.retrieval_scoped_paths_v90 import (literal_scores,NameStream,
    SpatialStream,geo_features,geo_parameters,native_query,TimedNativeStream)
from src.xgb_matcher.retrieval_ranked_path_v101 import collect,UncertifiedPathBudget


class QueryWordPostings(WordPostings):
    """Project dictionary lookup keys, not facts/postings/DF or candidate sites."""
    def __init__(self,stats,names):
        self.facts=stats['facts']; self.postings={}; self.terms={}
        self.allowed=set(t for name in names for t in signal(name).split())
        allowed=pa.array(sorted(self.allowed),type=pa.string()); count=0
        for b in stats['buckets']:
            table=pq.read_table(b['vocabulary'],use_threads=False); bucket=b['bucket']
            if len(table)!=b['words']: raise ValueError('Whole vocabulary count drift')
            count+=len(table)
            self.postings[bucket]=np.load(b['postings_path'],mmap_mode='r',allow_pickle=False)
            selected=table.filter(pc.is_in(table['token'],value_set=allowed))
            for row in selected.to_pylist():
                token=row['token']
                if token in self.terms: raise ValueError('Duplicate projected source word')
                self.terms[token]=(bucket,row['posting_start'],row['posting_stop'])
        if count!=stats['vocabulary_words']: raise ValueError('Full vocabulary denominator drift')

    def scores(self,weighted_terms):
        if not set(weighted_terms)<=self.allowed:
            raise ValueError('Unloaded query word; cannot silently change scores')
        return super().scores(weighted_terms)


def search(backend,tree,keys,parents,word_index,owners,graph,q,geo,weights,terms,alpha,*,path_weight=64.,max_seconds=20.):
    began=time.monotonic(); scores,word_trace=literal_scores(word_index,q.name)
    names=NameStream(scores); paths=NationalScopedPaths(owners,graph,scores)
    native=TimedNativeStream(backend,native_query(backend,q,weights,terms,alpha))
    _,_,reliable=geo_parameters(geo); reliability=1. if reliable else .5
    spatial=SpatialStream(tree,keys,parents,geo,reliability)
    def complete(rows):
        ids=[r['siret'] for r in rows]; base=native.complete(ids)
        values=geo_features(keys,tree.data,ids,geo)[:,0]*reliability
        return {s:dict(native_score=base[s],path_name_score=paths.best(s)['score'],site_score=float(values[i])) for i,s in enumerate(ids)}
    remaining=max_seconds-(time.monotonic()-began)
    if remaining<=0: raise TimeoutError('STOP_WORD_INDEX_PREPARATION_COST')
    try:
        rows,trace=collect(native,names,spatial,paths.targets,complete,path_weight=path_weight,
            name_score_max=1.,site_score_max=reliability,max_seconds=remaining,page_size=256)
    except UncertifiedPathBudget as exc:
        exc.trace.update(word_index=word_trace,native_point_calls=native.point_calls,native_point_seconds=native.point_seconds,
            native_sorted_seconds=native.sorted_seconds,seconds_including_preparation=time.monotonic()-began); raise
    for row in rows: row['scoped_name_proof']=paths.best(row['siret'])
    trace.update(word_index=word_trace,seconds=time.monotonic()-began,unique_documents_visited=trace['unique_targets_completed'],
        native_corpus_unchanged=True,name_site_scope='ALL_NATIONAL_REGISTRY_SITES',succession_hops=1,
        geo_query_available=spatial.point is not None,ban_reliability_factor=reliability,
        native_postings_visits_not_instrumented=True,proof_co_validity_unknown=True,identifier_features=False,
        native_point_calls=native.point_calls,native_point_seconds=native.point_seconds,native_sorted_seconds=native.sorted_seconds,
        physical_completion_buffer=8192,physical_fact_page_size=256,query_term_projection_same_scores=True)
    return rows,trace
