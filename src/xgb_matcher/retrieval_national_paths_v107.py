"""Full-national initial100; separate native reads, ONE certified heap100."""
import time
from src.xgb_matcher.retrieval_split_native_v107 import streams
from src.xgb_matcher.retrieval_national_paths_v101 import QueryWordPostings
from src.xgb_matcher.retrieval_national_paths_v99 import NationalScopedPaths
from src.xgb_matcher.retrieval_scoped_paths_v90 import literal_scores,NameStream,SpatialStream,geo_features,geo_parameters
from src.xgb_matcher.retrieval_split_path_v107 import collect,UncertifiedPathBudget


def search(backend,tree,keys,parents,word_index,owners,graph,q,geo,weights,terms,alpha,*,path_weight=64.,max_seconds=20.):
    if len(weights)!=16 or alpha!=weights[15]: raise ValueError('Frozen V67 semantic coefficient drift')
    began=time.monotonic(); scores,word_trace=literal_scores(word_index,q.name)
    names=NameStream(scores); paths=NationalScopedPaths(owners,graph,scores)
    native=streams(backend,q,weights,terms)
    _,_,reliable=geo_parameters(geo); reliability=1. if reliable else .5
    spatial=SpatialStream(tree,keys,parents,geo,reliability)
    def bound_rows(rows):
        ids=[r['siret'] for r in rows]; values=geo_features(keys,tree.data,ids,geo)[:,0]*reliability
        return {s:dict(components={k:dict(score=v.bound(s)['native_score'],upper=v.bound(s)['native_upper_bound'])
            for k,v in native.items()},path_name_score=paths.best(s)['score'],site_score=float(values[i])) for i,s in enumerate(ids)}
    def complete(rows):
        ids=[r['siret'] for r in rows]
        # An absent component after exhaustive sorted traversal is certified0.
        base={k:({s:v.cache.get(s,0.) for s in ids} if v.exhausted else v.complete(ids)) for k,v in native.items()}
        values=geo_features(keys,tree.data,ids,geo)[:,0]*reliability
        return {s:dict(component_scores={k:v[s] for k,v in base.items()},
            path_name_score=paths.best(s)['score'],site_score=float(values[i])) for i,s in enumerate(ids)}
    remaining=max_seconds-(time.monotonic()-began)
    if remaining<=0: raise TimeoutError('STOP_WORD_INDEX_PREPARATION_COST')
    def native_trace():
        return dict(native_point_calls=sum(v.point_calls for v in native.values()),
            native_point_seconds=sum(v.point_seconds for v in native.values()),
            native_sorted_seconds=sum(v.sorted_seconds for v in native.values()),
            native_component_times={k:dict(sorted_seconds=v.sorted_seconds,point_seconds=v.point_seconds,point_calls=v.point_calls)
                                    for k,v in native.items()})
    try:
        rows,trace=collect(native,names,spatial,paths.targets,complete,bound_rows,path_weight=path_weight,
                          name_score_max=1.,site_score_max=reliability,max_seconds=remaining,page_size=256)
    except UncertifiedPathBudget as exc:
        exc.trace.update(word_index=word_trace,**native_trace(),seconds_including_preparation=time.monotonic()-began); raise
    for row in rows: row['scoped_name_proof']=paths.best(row['siret'])
    trace.update(word_index=word_trace,seconds=time.monotonic()-began,**native_trace(),
        unique_documents_visited=trace['unique_targets_completed'],native_corpus_unchanged=True,
        name_site_scope='ALL_NATIONAL_REGISTRY_SITES',succession_hops=1,geo_query_available=spatial.point is not None,
        ban_reliability_factor=reliability,native_postings_visits_not_instrumented=True,
        proof_co_validity_unknown=True,identifier_features=False,physical_completion_buffer=8192,
        native_metric='SUM_FLOAT64_OF_TWO_FROZEN_V67_FLOAT32_COMPONENTS',native_bounds_not_zero_imputation=True)
    return rows,trace
