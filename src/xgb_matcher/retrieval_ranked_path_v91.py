"""V91 large physical completion buffers; same V83 complete metric and certificate.

The complete score belongs to the initial metric, not a downstream reranker.
No fixed parent seed/candidate reservoir, no top100 proof rows then dedup.
Adapters must expose exhaustive sorted factors and full random access.
"""
import heapq
import math
import time


class UncertifiedPathBudget(TimeoutError):
    def __init__(self,trace):
        self.trace=trace
        super().__init__('STOP_UNCERTIFIED_PATH_SEARCH_BUDGET')


def nonnegative(value):
    value=float(value)
    if not math.isfinite(value) or value<0: raise ValueError('Nonfinite/negative path score')
    return value


def membership(row):
    s,p=row['siret'],row['siren']
    if not (isinstance(s,str) and isinstance(p,str) and len(s)==14 and s.isascii() and s.isdigit()
            and len(p)==9 and p.isascii() and p.isdigit() and s.startswith(p)):
        raise ValueError('Official endpoint membership drift')
    return s,p


def collect(native,names,sites,targets_for_name,complete_targets,*,path_weight,
            name_score_max=1.,site_score_max=1.,cap=100,page_size=100,max_seconds=20.):
    """Metric max(native(t), weight * best_eligible_path_name(t) * site(t)).

    names is sorted on a scoped name factor, sites on the site factor;
    native is sorted on the SAME frozen native initial metric. For every
    fetched name, targets_for_name must enumerate ALL its eligible one-hop
    endpoints. Each endpoint supplies path_name_score <= fetched name score.
    complete_targets returns native_score, path_name_score (MAX over ALL
    eligible incoming official paths), and site_score for every requested
    target. Path scope/date/site checks belong to the official adapters.

    A target not seen in either graph stream has name <= frontier_name AND
    site <= frontier_site: its path score is bounded by their product.
    Name frontier advances only AFTER the whole fanout is completed.
    Returned groups are distinct SIRET, not distinct proof observations.
    """
    weight=nonnegative(path_weight); nmax=nonnegative(name_score_max); smax=nonnegative(site_score_max)
    if not (isinstance(cap,int) and 1<=cap<=100 and isinstance(page_size,int) and page_size>=1
            and math.isfinite(max_seconds) and max_seconds>0):
        raise ValueError('Invalid initial path collector budget')
    graph_max=nonnegative(weight*nmax*smax)
    channels={'native':native,'names':names,'sites':sites}
    frontiers={'native':0. if native.exhausted else math.inf,
               'names':0. if names.exhausted else nmax,'sites':0. if sites.exhausted else smax}
    calls={k:0 for k in channels}; consumed={k:0 for k in channels}; links=0
    heap=[]; seen={}; started=time.monotonic()

    def bound():
        return max(frontiers['native'],weight*frontiers['names']*frontiers['sites'])

    def trace(certified=False):
        b=bound()
        return dict(certified=certified,unseen_score_upper_bound=b if math.isfinite(b) else None,
            unseen_bound_initially_unbounded=not math.isfinite(b),
            frontiers={k:v if math.isfinite(v) else None for k,v in frontiers.items()},
            source_calls=dict(calls),source_rows=dict(consumed),path_endpoint_observations=links,
            unique_targets_completed=len(seen),collector_cap=cap,live_ranked_groups=len(heap),
            score_cutoff=heap[0][0] if heap else None,fixed_reservoir=None,
            fixed_parent_seed=None,proof_rows_cut_before_projection=False,
            ties='GLOBAL_CANONICAL_POSITIVE_SCORE_STRICT_FRONTIER',
            seconds=time.monotonic()-started)

    def check_deadline():
        if time.monotonic()-started>max_seconds: raise UncertifiedPathBudget(trace())

    def score_features(raw):
        a=nonnegative(raw['native_score']); n=nonnegative(raw['path_name_score']); s=nonnegative(raw['site_score'])
        if n>nmax or s>smax: raise ValueError('Declared global factor bound drift')
        g=nonnegative(weight*n*s)
        if g>graph_max: raise ValueError('Global graph bound drift')
        return a,n,s,max(a,g)

    def check_observed(features,requirements):
        a,n,s,_=features
        if ('native' in requirements and a!=requirements['native']
            or 'sites' in requirements and s!=requirements['sites']
            or n<requirements.get('path_name',0.)):
            raise ValueError('Sorted/complete initial metric drift')

    def complete_pending(pending):
        check_deadline()
        rows=[v['row'] for v in pending.values()]
        completed=complete_targets(rows)
        check_deadline()
        if set(completed)!=set(pending): raise ValueError('Complete incoming-path coverage drift')
        for target,entry in pending.items():
            features=score_features(completed[target]); check_observed(features,entry['requirements'])
            p=entry['row']['siren']; seen[target]=(p,features); score=features[-1]
            # A graph site with no admissible name is not a positive proof;
            # never fill unused slots with zero-score/no-evidence targets.
            if not score: continue
            item=(score,-int(target),target,p)
            if len(heap)<cap: heapq.heappush(heap,item)
            elif item[:2]>heap[0][:2]: heapq.heapreplace(heap,item)

    def consume_targets(rows,kind,name_value=None):
        nonlocal links
        pending={}; since_flush=0
        for row in rows:
            check_deadline(); s,p=membership(row); requirements={}
            if kind=='names':
                links+=1; n=nonnegative(row['path_name_score'])
                if n>name_value: raise ValueError('Path grade exceeds its own scoped name')
                requirements['path_name']=n
            else: requirements[kind]=nonnegative(row['score'])
            if s in seen:
                parent,features=seen[s]
                if parent!=p: raise ValueError('Official parent conflict')
                check_observed(features,requirements)
            else:
                if s not in pending: pending[s]=dict(row=dict(siret=s,siren=p),requirements={})
                entry=pending[s]
                if entry['row']['siren']!=p: raise ValueError('Official parent conflict')
                for k,v in requirements.items():
                    old=entry['requirements'].get(k)
                    if k!='path_name' and old is not None and old!=v:
                        raise ValueError('Duplicate sorted component drift')
                    entry['requirements'][k]=max(old or 0.,v)
            since_flush+=1
            # Completion batching is physical only; continue through ALL
            # targets of this factor before updating its unseen frontier.
            if len(pending)>=512 or since_flush>=4096:
                if pending: complete_pending(pending)
                pending={}; since_flush=0
        if pending: complete_pending(pending)

    while True:
        check_deadline(); unseen=bound()
        if unseen==0 or len(heap)==cap and heap[0][0]>unseen: break
        cutoff=heap[0][0] if len(heap)==cap else 0.
        graph_bound=weight*frontiers['names']*frontiers['sites']
        # Never rescan a source whose own unread score bound is strictly
        # below the current heap cutoff. Equality remains live for global ties.
        active=[k for k,s in channels.items() if not s.exhausted and frontiers[k]>0
                and ((k=='native' and frontiers[k]>=cutoff)
                     or (k!='native' and graph_bound>0 and graph_bound>=cutoff))]
        if not active: raise ValueError('Uncertified stalled path traversal')
        key=min(active,key=lambda k:(calls[k],{'native':0,'sites':1,'names':2}[k]))
        count=min(4096,page_size*(2**min(calls[key],6)))
        batch=channels[key].fetch(count); calls[key]+=1; consumed[key]+=len(batch)
        check_deadline()
        if not batch and not channels[key].exhausted: raise ValueError('Non-progressing sorted path source')
        previous=frontiers[key]
        for row in batch:
            value=nonnegative(row['score'])
            if value>previous: raise ValueError('Sorted factor order drift')
            previous=value
        if key=='names':
            # A page is physical only: consume ALL its fact fanouts before
            # advancing the frontier. Batch completion across facts, not one
            # native point-query per short alias fanout.
            def page_targets():
                for row in batch:
                    if row['score']<=0: continue
                    for target in targets_for_name(row):
                        if nonnegative(target['path_name_score'])>float(row['score']):
                            raise ValueError('Path grade exceeds its own scoped name')
                        yield target
            if batch: consume_targets(page_targets(),key,float(batch[0]['score']))
        else: consume_targets(batch,key)
        frontiers[key]=0. if channels[key].exhausted else previous

    result=[dict(siret=s,siren=p,score=score) for score,_,s,p in sorted(heap,key=lambda r:(-r[0],r[2]))]
    final=trace(True)
    if len(result)>cap or len({r['siret'] for r in result})!=len(result): raise ValueError('Initial group cap drift')
    return result,final
