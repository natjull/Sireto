"""V115 metric unchanged; prevent starvation of an unread spatial frontier.

No native/source score/model/budget changes. Balance completed source reads,
not candidate quotas; strict global stopping certificate remains identical.
"""
import heapq
import math
import time
import numpy as np

from src.xgb_matcher.retrieval_activity_initial_v115 import (
    NativeUpperStream,GeometryUpperStream,OfficialActivityGrades,UncertifiedActivityBudget)
from src.xgb_matcher.retrieval_activity_joint_v114 import vector,linear_weights
from src.xgb_matcher.retrieval_proof_v56 import point_features,geo_features
from src.xgb_matcher.retrieval_ranked_path_v105 import membership,nonnegative


def collect(channels,complete,*,cap=100,page_size=128,max_seconds=20.):
    if (not channels or not isinstance(cap,int) or not 1<=cap<=100 or not isinstance(page_size,int) or page_size<1 or
        not math.isfinite(max_seconds) or max_seconds<=0):raise ValueError('Initial activity collector budget/cap drift')
    started=time.monotonic();frontiers={k:0. if s.exhausted else nonnegative(s.maximum) for k,s in channels.items()}
    maxima=dict(frontiers);calls={k:0 for k in channels};consumed={k:0 for k in channels};heap=[];seen={}
    def bound():return float(np.nextafter(math.fsum(frontiers.values()),np.inf)) if any(frontiers.values()) else 0.
    def trace(certified=False):
        return dict(certified=certified,source_calls=dict(calls),source_rows=dict(consumed),frontiers=dict(frontiers),
            unseen_score_upper_bound=bound(),score_cutoff=heap[0][0] if len(heap)==cap else None,
            unique_documents_visited=len(seen),collector_cap=cap,live_ranked_groups=len(heap),fixed_reservoir=None,
            source_scores_are_upper_bounds=True,ties='GLOBAL_CANONICAL_STRICT_FRONTIER',seconds=time.monotonic()-started,
            traversal='FAIR_COMPLETED_SOURCE_READS_NOT_CANDIDATE_QUOTAS')
    def deadline():
        if time.monotonic()-started>max_seconds:raise UncertifiedActivityBudget(trace())
    while True:
        deadline();unseen=bound()
        if unseen==0 or len(heap)==cap and heap[0][0]>unseen:break
        active=[k for k,s in channels.items() if not s.exhausted and frontiers[k]>0]
        if not active:raise ValueError('Uncertified stalled activity traversal')
        # A lower-amplitude unread source can still prevent certification.
        # Let every active source update its frontier before reading more.
        key=min(active,key=lambda k:(calls[k],-frontiers[k],k));count=min(4096,page_size*2**min(calls[key],5))
        batch=channels[key].fetch(count);calls[key]+=1;consumed[key]+=len(batch);deadline()
        if not batch and not channels[key].exhausted:raise ValueError('Nonprogressing activity bound stream')
        previous=frontiers[key];novel=[]
        for row in batch:
            upper=nonnegative(row['score'])
            if upper>previous:raise ValueError('Sorted activity upper-bound drift')
            previous=upper;s,p=membership(row)
            if s in seen:
                if seen[s]!=p:raise ValueError('Official activity parent conflict')
            else:seen[s]=p;novel.append(row)
        completed=complete(novel) if novel else {};deadline()
        if set(completed)!={r['siret'] for r in novel}:raise ValueError('Whole initial activity score coverage drift')
        for row in novel:
            score=nonnegative(completed[row['siret']])
            if score>sum(maxima.values())+1e-10:raise ValueError('Initial metric exceeds global proof bound')
            if score>row['score']+sum(v for k,v in maxima.items() if k!=key)+1e-10:
                raise ValueError('Completed metric exceeds source proof bound')
            if score==0:continue
            item=(score,-int(row['siret']),row['siret'],row['siren'])
            if len(heap)<cap:heapq.heappush(heap,item)
            elif item[:2]>heap[0][:2]:heapq.heapreplace(heap,item)
        frontiers[key]=0. if channels[key].exhausted else previous
    rows=[dict(siret=s,siren=p,score=score) for score,_,s,p in sorted(heap,key=lambda r:(-r[0],r[2]))]
    return rows,trace(True)


def search(backend,tree,geo_keys,geo_parents,q,geo,weights,activity,*,max_seconds=20.):
    began=time.monotonic();effective=linear_weights(weights,activity.maximum)
    native=NativeUpperStream(backend,q,effective[:11]);spatial=GeometryUpperStream(tree,geo_keys,geo_parents,geo,effective[11:])
    completion_calls=0;completion_seconds=0.;unknown_before=activity.unknown_joins
    def complete(rows):
        nonlocal completion_calls,completion_seconds
        t=time.monotonic();ids=[r['siret'] for r in rows]
        ordered,f=point_features(backend,q,ids);g=geo_features(geo_keys,tree.data,ordered,geo);grades=activity.values(ordered)
        full=np.column_stack((f,g))
        result={s:float(vector(full[i],grades[i])@np.asarray(weights)) for i,s in enumerate(ordered)}
        completion_calls+=1;completion_seconds+=time.monotonic()-t;return result
    remaining=max_seconds-(time.monotonic()-began)
    if remaining<=0:raise TimeoutError('STOP_ACTIVITY_QUERY_PREPARATION_COST')
    try:rows,trace=collect(dict(native=native,geometry=spatial),complete,max_seconds=remaining)
    except UncertifiedActivityBudget as exc:
        exc.trace.update(completion_calls=completion_calls,completion_seconds=completion_seconds);raise
    trace.update(seconds=time.monotonic()-began,completion_calls=completion_calls,completion_seconds=completion_seconds,
        official_unknown_activity_visits=activity.unknown_joins-unknown_before,activity_global_maximum=activity.maximum,
        native_rounding_padding=native.padding,native_rounding_operations=native.operations,
        metric='V114_INITIAL_ACTIVITY_SITE30_FIXED_SCORE',identifiers_as_features=False,
        relation_paths_included=False,full_national_lexical_corpus=True,full_national_activity_corpus=True,
        geometry_missing_is_not_filter=True,native_postings_visits_not_instrumented=True)
    return rows,trace
