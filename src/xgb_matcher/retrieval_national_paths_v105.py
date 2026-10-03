"""Full-national initial100 with frozen OOF V67 native16 and complete V101 paths."""
import time
from src.xgb_matcher.retrieval_factored_native_v105 import query as native_query,BoundedNativeStream
from src.xgb_matcher.retrieval_national_paths_v101 import QueryWordPostings
from src.xgb_matcher.retrieval_national_paths_v99 import NationalScopedPaths
from src.xgb_matcher.retrieval_scoped_paths_v90 import (literal_scores,NameStream,
    SpatialStream,geo_features,geo_parameters)
from src.xgb_matcher.retrieval_ranked_path_v105 import collect,UncertifiedPathBudget

def search(backend,tree,keys,parents,word_index,owners,graph,q,geo,weights,terms,alpha,*,path_weight=64.,max_seconds=20.):
    if len(weights)!=16 or alpha!=weights[15]: raise ValueError('Frozen V67 semantic coefficient drift')
    began=time.monotonic(); scores,word_trace=literal_scores(word_index,q.name)
    names=NameStream(scores); paths=NationalScopedPaths(owners,graph,scores)
    native=BoundedNativeStream(backend,native_query(backend,q,weights,terms))
    _,_,reliable=geo_parameters(geo); reliability=1. if reliable else .5
    spatial=SpatialStream(tree,keys,parents,geo,reliability)
    def bound_rows(rows):
        ids=[r['siret'] for r in rows]; values=geo_features(keys,tree.data,ids,geo)[:,0]*reliability
        return {s:dict(**native.bound(s),path_name_score=paths.best(s)['score'],site_score=float(values[i])) for i,s in enumerate(ids)}
    def complete(rows):
        ids=[r['siret'] for r in rows]; base=native.complete(ids)
        values=geo_features(keys,tree.data,ids,geo)[:,0]*reliability
        return {s:dict(native_score=base[s],path_name_score=paths.best(s)['score'],site_score=float(values[i])) for i,s in enumerate(ids)}
    remaining=max_seconds-(time.monotonic()-began)
    if remaining<=0: raise TimeoutError('STOP_WORD_INDEX_PREPARATION_COST')
    try:
        rows,trace=collect(native,names,spatial,paths.targets,complete,path_weight=path_weight,
            name_score_max=1.,site_score_max=reliability,max_seconds=remaining,page_size=256,bound_targets=bound_rows)
    except UncertifiedPathBudget as exc:
        exc.trace.update(word_index=word_trace,native_point_calls=native.point_calls,native_point_seconds=native.point_seconds,
            native_sorted_seconds=native.sorted_seconds,seconds_including_preparation=time.monotonic()-began); raise
    for row in rows: row['scoped_name_proof']=paths.best(row['siret'])
    trace.update(word_index=word_trace,seconds=time.monotonic()-began,unique_documents_visited=trace['unique_targets_completed'],
        native_corpus_unchanged=True,name_site_scope='ALL_NATIONAL_REGISTRY_SITES',succession_hops=1,
        geo_query_available=spatial.point is not None,ban_reliability_factor=reliability,
        native_postings_visits_not_instrumented=True,proof_co_validity_unknown=True,identifier_features=False,
        native_point_calls=native.point_calls,native_point_seconds=native.point_seconds,native_sorted_seconds=native.sorted_seconds,
        physical_completion_buffer=8192,physical_fact_page_size=256,query_term_projection_same_scores=True,native_metric='FROZEN_V67_FACTORED_FLOAT32_HEAD_FULL16',native_bounds_not_zero_imputation=True)
    return rows,trace
