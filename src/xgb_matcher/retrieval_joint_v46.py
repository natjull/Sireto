"""Learned additive initial metric and certified adaptive index traversal.

Not a classifier/reranker on a fixed candidate reservoir. Sorted native
component indexes are traversed until their unseen-score bound certifies the
best100 under the SAME declared metric. IDs only bind source records, output
and deterministic ties among equal scores. No query labels are accepted.
"""
import heapq
import math
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from scripts import retrieval_v42_indexed_metric as v42

FEATURES=(*v42.FEATURES,'geo_proximity','geo_reliable_proximity')
BASE=np.array([1.,1.,0.,0.,0.,0.,1.,0.,1.,0.])


def validate_weights(weights):
    w=np.asarray(weights,dtype=np.float64)
    if w.shape!=(10,) or not np.isfinite(w).all() or np.any(w<0) or not np.any(w>0):
        raise ValueError('Invalid monotone joint metric')
    return w


def geo_parameters(geo):
    if not geo.get('spatial_crs_compatible',False): return None,1.,False
    point=np.array([geo['ban_x_l93'],geo['ban_y_l93']],dtype=np.float64)
    sigma=float(geo['ban_uncertainty_m'])
    if not np.isfinite(point).all() or not math.isfinite(sigma) or sigma<5:
        raise ValueError('Invalid declared query geometry')
    return point,sigma,not geo['ban_declared_weak']


def geo_features(keys,xy,ids,geo):
    result=np.zeros((len(ids),2),dtype=np.float64)
    point,sigma,reliable=geo_parameters(geo)
    if point is None: return result
    numeric=np.asarray([int(s) for s in ids],dtype=np.uint64)
    pos=np.searchsorted(keys,numeric)
    safe=np.minimum(pos,len(keys)-1)
    mask=(pos<len(keys))&(keys[safe]==numeric)
    delta=xy[pos[mask]]-point
    distances=np.sqrt(np.sum(delta*delta,axis=1))
    result[mask,0]=1./(1.+distances/sigma)
    result[:,1]=result[:,0] if reliable else 0.
    return result


def collect(channels,score_rows,cap=100,page_size=100,max_seconds=20.):
    """Blockwise threshold algorithm over full sorted source streams.

    Each source must bound EVERY unread score by its last fetched score,
    returning zero after exhaustion. Random access completes a fixed metric,
    not a learned decision applied to a reservoir. Equality certifies score
    top-k; ties are canonical only among visited equal-scoring records, just
    as a native collector may choose among global boundary ties.
    """
    if not 1<=cap<=100 or page_size<1 or max_seconds<=0: raise ValueError('Invalid collector budget')
    started=time.monotonic(); frontiers={k:math.inf for k in channels}
    heap=[]; seen={}; calls={k:0 for k in channels}; consumed={k:0 for k in channels}
    while channels:
        if time.monotonic()-started>max_seconds: raise TimeoutError('STOP_UNCERTIFIED_SEARCH_BUDGET')
        bound=sum(frontiers.values())
        if len(heap)==cap and heap[0][0]>=bound: break
        active=[k for k in channels if not channels[k].exhausted]
        if not active: break
        key=max(active,key=lambda k:(frontiers[k],k))
        batch=channels[key].fetch(page_size); calls[key]+=1; consumed[key]+=len(batch)
        previous=frontiers[key]
        for row in batch:
            score=row['score']
            if not math.isfinite(score) or score<0 or score>previous:
                raise ValueError('Native sorted-source score drift')
            previous=score
        if not batch and not channels[key].exhausted: raise ValueError('Non-progressing index stream')
        frontiers[key]=0. if channels[key].exhausted else previous
        novel=[]
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
            if not math.isfinite(score) or score<row['score']:
                raise ValueError('Joint score/component drift')
            item=(score,-int(row['siret']),row['siret'],row['siren'])
            if len(heap)<cap: heapq.heappush(heap,item)
            elif item[:2]>heap[0][:2]: heapq.heapreplace(heap,item)
    bound=sum(frontiers.values()) if frontiers else 0.
    certified=(len(heap)==cap and heap[0][0]>=bound) or all(s.exhausted for s in channels.values())
    if not certified: raise ValueError('Uncertified initial search')
    rows=[dict(siret=s,siren=p,score=float(score)) for score,_,s,p in sorted(heap,key=lambda x:(-x[0],x[2]))]
    return rows,dict(certified=True,unseen_score_upper_bound=bound,score_cutoff=rows[-1]['score'] if rows else None,
        source_calls=calls,source_rows=consumed,unique_documents_visited=len(seen),fixed_reservoir=None,
        ties='CANONICAL_AMONG_VISITED_NATIVE_SCORE_EQUALS',seconds=time.monotonic()-started)


class LexicalStream:
    def __init__(self,backend,query,weights):
        self.backend=backend; self.weights=np.asarray(weights); self.offset=0; self.seen=set()
        self.exhausted=not np.any(self.weights>0)
        self.cache={}; self.pruned_zero_nonlocal=False
        self.parsed=None if self.exhausted else v42.index_query(backend,query,self.weights)
        if not self.exhausted and not np.any(self.weights[[0,1,4]]):
            place=v42.components(backend,query)[6]
            if place is None: self.exhausted=True
            else:
                tv=backend._tantivy
                self.parsed=tv.Query.boolean_query([(tv.Occur.Must,self.parsed),
                    (tv.Occur.Must,tv.Query.const_score_query(place,0.))])
                self.pruned_zero_nonlocal=True

    def fetch(self,n):
        if self.exhausted: return []
        hits=self.backend.searcher.search(self.parsed,limit=n,offset=self.offset,count=False).hits
        self.offset+=len(hits); self.exhausted=len(hits)<n; rows=[]
        for score,loc in hits:
            doc=self.backend.searcher.doc(loc).to_dict()
            s,p=(self.backend._first(doc,k) for k in ('siret','siren'))
            if s in self.seen: raise ValueError('Unstable native pagination')
            self.seen.add(s)
            if s in self.cache and self.cache[s]!=float(score): raise ValueError('Native/random-access score drift')
            self.cache[s]=float(score)
            rows.append(dict(siret=s,siren=p,score=float(score)))
        return rows


class SpatialStream:
    def __init__(self,tree,keys,parents,geo,weight):
        self.tree=tree; self.keys=keys; self.parents=parents; self.offset=0; self.weight=float(weight)
        self.point,self.sigma,_=geo_parameters(geo)
        self.exhausted=self.point is None or self.weight==0

    def fetch(self,n):
        if self.exhausted: return []
        count=min(self.tree.n,self.offset+n)
        dist,_=self.tree.query(self.point,k=count,workers=2)
        cutoff=float(np.max(dist))
        pos=np.asarray(self.tree.query_ball_point(self.point,cutoff+max(1e-8,cutoff*1e-12),workers=2),dtype=np.int64)
        delta=self.tree.data[pos]-self.point; d=np.sqrt(np.sum(delta*delta,axis=1))
        order=np.lexsort((self.keys[pos],d))[self.offset:count]
        rows=[dict(siret=f'{int(self.keys[pos[i]]):014d}',siren=f'{int(self.parents[pos[i]]):09d}',
            score=float(self.weight*(1./(1.+float(d[i])/self.sigma)))) for i in order]
        if len(rows)!=count-self.offset: raise ValueError('Spatial pagination drift')
        self.offset=count; self.exhausted=count==self.tree.n
        return rows


def search(backend,tree,keys,parents,query,geo,weights,max_seconds=20.):
    w=validate_weights(weights); lexical=LexicalStream(backend,query,w[:8]); point,_,reliable=geo_parameters(geo)
    gw=float(w[8]+w[9]*reliable)
    spatial=SpatialStream(tree,keys,parents,geo,gw)
    tv=backend._tantivy
    # Full native lexical metric for random access, not the physically pruned
    # stream. A site outside locality may have zero lexical but positive XY.
    full=None if not np.any(w[:8]>0) else v42.index_query(backend,query,w[:8])
    def score_rows(rows):
        ids=[r['siret'] for r in rows]
        unknown=[s for s in ids if s not in lexical.cache]
        lexical.cache.update({s:0. for s in unknown})
        if unknown and full is not None:
            identity=tv.Query.const_score_query(tv.Query.term_set_query(backend.index.schema,'siret',unknown),0.)
            parsed=tv.Query.boolean_query([(tv.Occur.Must,full),(tv.Occur.Must,identity)])
            found=set()
            for score,loc in backend.searcher.search(parsed,limit=len(unknown),count=False).hits:
                s=backend._first(backend.searcher.doc(loc).to_dict(),'siret')
                if s not in lexical.cache or s in found: raise ValueError('Random access identity drift')
                found.add(s); lexical.cache[s]=float(score)
        # Match the spatial stream's float64 arithmetic exactly: query-side
        # reliable flag combines the two monotone coefficients before scale.
        g=geo_features(keys,tree.data,ids,geo)[:,0]*gw
        return {s:float(lexical.cache[s]+g[i]) for i,s in enumerate(ids)}
    channels={k:s for k,s in (('lexical',lexical),('spatial',spatial)) if not s.exhausted}
    rows,trace=collect(channels,score_rows,max_seconds=max_seconds)
    trace['zero_nonlocal_lexical_pruning']=lexical.pruned_zero_nonlocal
    trace['geo_query_available']=point is not None
    return rows,trace


def fit_metric(scenes,heldout_fold,registry,ridge=.0005):
    if heldout_fold not in (2,3,4) or set(registry)!={'2','3','4'}: raise ValueError('OOF registry/fold drift')
    heldout=set(registry[str(heldout_fold)])
    if not heldout: raise ValueError('Empty entity registry')
    x=[]; sw=[]; kept=[]
    for r in scenes:
        if r['fold'] not in (2,3,4) or r['positive_siren'] not in registry[str(r['fold'])]: raise ValueError('Source membership drift')
        if r['fold']==heldout_fold or r['positive_siren'] in heldout or r['positive_features'] is None: continue
        negatives=[n for n in r['negatives'] if n['siren'] not in heldout]
        if not negatives: continue
        x.extend(np.asarray(r['positive_features'])-np.asarray([n['features'] for n in negatives]))
        sw.extend([1./len(negatives)]*len(negatives)); kept.append(r['query_id'])
    x=np.asarray(x,dtype=np.float64); sw=np.asarray(sw)/len(kept) if kept else np.array([])
    if len(kept)<16 or x.ndim!=2 or x.shape[1]!=10 or not np.isfinite(x).all() or ridge<=0:
        raise ValueError('Insufficient/invalid safe OOF fit')
    def objective(w):
        margin=x@w
        return float(sw@np.logaddexp(0.,-margin)+ridge*np.sum((w-BASE)**2)),x.T@(-sw*expit(-margin))+2*ridge*(w-BASE)
    fitted=minimize(objective,BASE,method='L-BFGS-B',jac=True,bounds=[(0.,64.)]*10,
        options=dict(maxiter=500,ftol=1e-12,gtol=1e-8))
    if not fitted.success: raise ValueError('Joint fit failed: '+str(fitted.message))
    validate_weights(fitted.x)
    return dict(weights=fitted.x.tolist(),features=list(FEATURES),heldout_fold=heldout_fold,
        training_folds=sorted({r['fold'] for r in scenes if r['query_id'] in set(kept)}),
        query_ids=kept,training_queries=len(kept),training_pairs=len(x),heldout_entity_overlap=0,
        ridge=ridge,objective_before=objective(BASE)[0],objective_after=float(fitted.fun),identifiers_as_features=False)
