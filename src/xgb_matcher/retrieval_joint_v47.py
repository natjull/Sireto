"""Same V46 initial metric; fair geometric-growth physical index traversal.

V46's largest-frontier heuristic may starve a fast spatial source when a
lexical score plateau stays above its frontier. This version alternates
non-exhausted sources and grows fetch pages, never fixing corpus depth.
The unseen-score certificate and <=100 output remain unchanged.
"""
import heapq
import math
import time

import numpy as np

from scripts import retrieval_v42_indexed_metric as v42
from src.xgb_matcher.retrieval_joint_v46 import LexicalStream,SpatialStream,geo_features,geo_parameters,validate_weights


class UncertifiedSearchBudget(TimeoutError):
    def __init__(self,trace):
        self.trace=trace
        super().__init__('STOP_UNCERTIFIED_SEARCH_BUDGET '+str(trace))


def collect(channels,score_rows,cap=100,page_size=100,max_seconds=20.):
    if not 1<=cap<=100 or page_size<1 or max_seconds<=0: raise ValueError('Invalid collector budget')
    started=time.monotonic(); frontiers={k:math.inf for k in channels}
    heap=[]; seen={}; calls={k:0 for k in channels}; consumed={k:0 for k in channels}
    while channels:
        bound=sum(frontiers.values())
        if time.monotonic()-started>max_seconds:
            raise UncertifiedSearchBudget(dict(source_calls=calls,source_rows=consumed,frontiers=frontiers,
                unseen_score_upper_bound=bound,visited=len(seen),cutoff=heap[0][0] if heap else None))
        if len(heap)==cap and heap[0][0]>=bound: break
        active=[k for k in channels if not channels[k].exhausted]
        if not active: break
        # Fairness prevents a high but flat native score from starving another
        # sorted source. The page cap is a physical unit, NOT a pool/depth cap:
        # additional pages continue until the SAME global certificate holds.
        key=min(active,key=lambda k:(calls[k],k))
        requested=min(4096,page_size*(2**min(calls[key],6)))
        batch=channels[key].fetch(requested); calls[key]+=1; consumed[key]+=len(batch)
        previous=frontiers[key]
        for row in batch:
            score=row['score']
            if not math.isfinite(score) or score<0 or score>previous: raise ValueError('Native sorted-source score drift')
            previous=score
        if not batch and not channels[key].exhausted: raise ValueError('Non-progressing index stream')
        frontiers[key]=0. if channels[key].exhausted else previous; novel=[]
        for row in batch:
            s=row['siret']; parent=row['siren']
            if not (len(s)==14 and s.isdigit() and len(parent)==9 and parent.isdigit() and s.startswith(parent)):
                raise ValueError('Official identity drift')
            if s in seen:
                if seen[s]!=parent: raise ValueError('Official membership conflict')
            else: novel.append(row); seen[s]=parent
        completed=score_rows(novel)
        if set(completed)!={r['siret'] for r in novel}: raise ValueError('Random-access metric coverage drift')
        for row in novel:
            score=float(completed[row['siret']])
            if not math.isfinite(score) or score<row['score']: raise ValueError('Joint score/component drift')
            item=(score,-int(row['siret']),row['siret'],row['siren'])
            if len(heap)<cap: heapq.heappush(heap,item)
            elif item[:2]>heap[0][:2]: heapq.heapreplace(heap,item)
    bound=sum(frontiers.values()) if frontiers else 0.
    if not ((len(heap)==cap and heap[0][0]>=bound) or all(s.exhausted for s in channels.values())):
        raise ValueError('Uncertified initial search')
    rows=[dict(siret=s,siren=p,score=float(score)) for score,_,s,p in sorted(heap,key=lambda x:(-x[0],x[2]))]
    return rows,dict(certified=True,unseen_score_upper_bound=bound,score_cutoff=rows[-1]['score'] if rows else None,
        source_calls=calls,source_rows=consumed,unique_documents_visited=len(seen),fixed_reservoir=None,
        ties='CANONICAL_AMONG_VISITED_NATIVE_SCORE_EQUALS',seconds=time.monotonic()-started,
        physical_plan='FAIR_GEOMETRIC_GROWTH_PAGES_NOT_CORPUS_DEPTH_CAP',page_unit_max=4096)


def search(backend,tree,keys,parents,query,geo,weights,max_seconds=20.):
    w=validate_weights(weights); lexical=LexicalStream(backend,query,w[:8]); point,_,reliable=geo_parameters(geo)
    gw=float(w[8]+w[9]*reliable); spatial=SpatialStream(tree,keys,parents,geo,gw); tv=backend._tantivy
    full=None if not np.any(w[:8]>0) else v42.index_query(backend,query,w[:8])
    def score_rows(rows):
        ids=[r['siret'] for r in rows]; unknown=[s for s in ids if s not in lexical.cache]
        lexical.cache.update({s:0. for s in unknown})
        if unknown and full is not None:
            identity=tv.Query.const_score_query(tv.Query.term_set_query(backend.index.schema,'siret',unknown),0.)
            parsed=tv.Query.boolean_query([(tv.Occur.Must,full),(tv.Occur.Must,identity)]); found=set()
            for score,loc in backend.searcher.search(parsed,limit=len(unknown),count=False).hits:
                s=backend._first(backend.searcher.doc(loc).to_dict(),'siret')
                if s not in lexical.cache or s in found: raise ValueError('Random access identity drift')
                found.add(s); lexical.cache[s]=float(score)
        g=geo_features(keys,tree.data,ids,geo)[:,0]*gw
        return {s:float(lexical.cache[s]+g[i]) for i,s in enumerate(ids)}
    channels={k:s for k,s in (('lexical',lexical),('spatial',spatial)) if not s.exhausted}
    rows,trace=collect(channels,score_rows,max_seconds=max_seconds)
    trace['zero_nonlocal_lexical_pruning']=lexical.pruned_zero_nonlocal; trace['geo_query_available']=point is not None
    return rows,trace
