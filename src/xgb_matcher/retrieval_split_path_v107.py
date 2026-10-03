"""One certified initial top100: max(lexical + initialism, scoped name * site).

Four exhaustive sorted inputs, a single heap100, no intermediate candidate
reservoir. Unknown components carry upper bounds, never predicted zeros.
"""
import heapq
import math
import time
from src.xgb_matcher.retrieval_ranked_path_v105 import nonnegative,membership


class UncertifiedPathBudget(TimeoutError):
    def __init__(self,trace):
        self.trace=trace
        super().__init__('STOP_UNCERTIFIED_SPLIT_PATH_SEARCH_BUDGET')


def collect(components,names,sites,targets_for_name,complete_targets,bound_targets,*,
            path_weight,name_score_max=1.,site_score_max=1.,cap=100,page_size=256,max_seconds=20.):
    if set(components)!={'lexical','initialism'}:
        raise ValueError('Initial component scope drift')
    weight=nonnegative(path_weight); nmax=nonnegative(name_score_max); smax=nonnegative(site_score_max)
    if not (isinstance(cap,int) and 1<=cap<=100 and isinstance(page_size,int) and page_size>=1
            and math.isfinite(max_seconds) and max_seconds>0):
        raise ValueError('Invalid initial split collector budget')
    keys=('lexical','initialism'); channels={**components,'names':names,'sites':sites}
    frontiers={k:0. if s.exhausted else math.inf for k,s in components.items()}
    frontiers.update(names=0. if names.exhausted else nmax,sites=0. if sites.exhausted else smax)
    calls={k:0 for k in channels}; consumed={k:0 for k in channels}; links=0
    shortcuts=dict(exact_components=0,exact_path=0,dominated=0,completion_required=0,proven_zero_components=0)
    heap=[]; seen={}; started=time.monotonic()

    def unseen_bound():
        return max(frontiers['lexical']+frontiers['initialism'],weight*frontiers['names']*frontiers['sites'])

    def trace(certified=False):
        b=unseen_bound()
        return dict(certified=certified,unseen_score_upper_bound=b if math.isfinite(b) else None,
            unseen_bound_initially_unbounded=not math.isfinite(b),
            frontiers={k:v if math.isfinite(v) else None for k,v in frontiers.items()},
            source_calls=dict(calls),source_rows=dict(consumed),path_endpoint_observations=links,
            unique_targets_completed=len(seen),unique_targets_exact_score=sum(r['score'] is not None for r in seen.values()),
            targets_bound_checked=len(seen),component_interval_shortcuts=dict(shortcuts),collector_cap=cap,
            live_ranked_groups=len(heap),score_cutoff=heap[0][0] if heap else None,
            completion_semantics='EXACT_FINAL_SCORE_OR_PROVED_UNADMITTABLE_INTERVAL',
            fixed_reservoir=None,fixed_parent_seed=None,proof_rows_cut_before_projection=False,
            ties='GLOBAL_CANONICAL_POSITIVE_SCORE_STRICT_FRONTIER',seconds=time.monotonic()-started)

    def deadline():
        if time.monotonic()-started>max_seconds: raise UncertifiedPathBudget(trace())

    def exact_path(raw):
        n=nonnegative(raw['path_name_score']); s=nonnegative(raw['site_score'])
        if n>nmax or s>smax: raise ValueError('Declared global factor bound drift')
        return n,s,nonnegative(weight*n*s)

    def check_observed(state,requirements):
        for k in keys:
            if k in requirements:
                actual=state['values'][k]
                if (actual is not None and actual!=requirements[k] or
                        actual is None and requirements[k]>state['upper'][k]):
                    raise ValueError('Sorted/component interval metric drift')
        if (state['name']<requirements.get('path_name',0.) or
                'sites' in requirements and state['site']!=requirements['sites']):
            raise ValueError('Sorted/complete path metric drift')

    def complete_pending(pending):
        deadline(); rows=[r['row'] for r in pending.values()]; bounds=bound_targets(rows)
        if set(bounds)!=set(pending): raise ValueError('Bound incoming-path coverage drift')
        prepared={}; required=[]
        for target,entry in pending.items():
            raw=bounds[target]; n,s,g=exact_path(raw)
            if set(raw['components'])!=set(keys): raise ValueError('Bound component scope drift')
            values={}; upper={}
            for k in keys:
                component=raw['components'][k]; value=component['score']; bound=float(component['upper'])
                if math.isnan(bound) or bound<0: raise ValueError('Invalid component upper bound')
                if value is not None:
                    value=nonnegative(value)
                    if value!=bound: raise ValueError('Known component upper bound drift')
                elif bound==0.:
                    # Zero is PROVEN by the exhaustive component stream,
                    # not imputed because the site was not yet retrieved.
                    value=0.; shortcuts['proven_zero_components']+=1
                values[k]=value; upper[k]=bound
            if all(v is not None for v in values.values()):
                score=max(values['lexical']+values['initialism'],g); shortcuts['exact_components']+=1
            elif g>=upper['lexical']+upper['initialism']:
                score=g; shortcuts['exact_path']+=1
            elif len(heap)==cap and (max(g,upper['lexical']+upper['initialism']),-int(target))<heap[0][:2]:
                score=None; shortcuts['dominated']+=1
            else:
                required.append(entry['row']); score=None
            prepared[target]=dict(values=values,upper=upper,name=n,site=s,score=score)
        completed=complete_targets(required) if required else {}; deadline()
        if set(completed)!={r['siret'] for r in required}: raise ValueError('Complete incoming-path coverage drift')
        shortcuts['completion_required']+=len(required)
        for target,raw in completed.items():
            prior=prepared[target]; n,s,g=exact_path(raw)
            if set(raw['component_scores'])!=set(keys): raise ValueError('Exact component scope drift')
            values={k:nonnegative(raw['component_scores'][k]) for k in keys}
            if (n!=prior['name'] or s!=prior['site'] or
                    any(v>prior['upper'][k] or prior['values'][k] is not None and v!=prior['values'][k]
                        for k,v in values.items())):
                raise ValueError('Component interval/exact completion drift')
            prior.update(values=values,upper=dict(values),score=max(values['lexical']+values['initialism'],g))
        for target,entry in pending.items():
            state=prepared[target]; check_observed(state,entry['requirements'])
            state['parent']=entry['row']['siren']; seen[target]=state; score=state['score']
            if not score: continue
            item=(score,-int(target),target,state['parent'])
            if len(heap)<cap: heapq.heappush(heap,item)
            elif item[:2]>heap[0][:2]: heapq.heapreplace(heap,item)

    def consume(rows,kind):
        nonlocal links
        pending={}; observations=0
        for row in rows:
            deadline(); target,parent=membership(row)
            if kind=='names':
                links+=1; requirements={'path_name':nonnegative(row['path_name_score'])}
            else: requirements={kind:nonnegative(row['score'])}
            if target in seen:
                state=seen[target]
                if state['parent']!=parent: raise ValueError('Official parent conflict')
                check_observed(state,requirements)
            else:
                entry=pending.setdefault(target,dict(row=dict(siret=target,siren=parent),requirements={}))
                if entry['row']['siren']!=parent: raise ValueError('Official parent conflict')
                for k,v in requirements.items():
                    previous=entry['requirements'].get(k)
                    if k!='path_name' and previous is not None and previous!=v:
                        raise ValueError('Duplicate sorted component drift')
                    entry['requirements'][k]=max(previous or 0.,v)
            observations+=1
            if len(pending)>=8192 or observations>=65536:
                if pending: complete_pending(pending)
                pending={}; observations=0
        if pending: complete_pending(pending)

    while True:
        deadline(); unseen=unseen_bound()
        if unseen==0 or len(heap)==cap and heap[0][0]>unseen: break
        cutoff=heap[0][0] if len(heap)==cap else 0.
        native_bound=frontiers['lexical']+frontiers['initialism']
        path_bound=weight*frontiers['names']*frontiers['sites']
        active=[k for k,s in channels.items() if not s.exhausted and frontiers[k]>0
                and ((k in keys and native_bound>=cutoff) or
                     (k not in keys and path_bound>0 and path_bound>=cutoff))]
        if not active: raise ValueError('Uncertified stalled split traversal')
        priority={'lexical':0,'initialism':1,'sites':2,'names':3}
        key=min(active,key=lambda k:(calls[k],priority[k])); count=min(4096,page_size*(2**min(calls[key],6)))
        batch=channels[key].fetch(count); calls[key]+=1; consumed[key]+=len(batch); deadline()
        if not batch and not channels[key].exhausted: raise ValueError('Non-progressing sorted source')
        previous=frontiers[key]
        for row in batch:
            value=nonnegative(row['score'])
            if value>previous: raise ValueError('Sorted factor order drift')
            previous=value
        if key=='names':
            def page_targets():
                for row in batch:
                    if row['score']<=0: continue
                    for target in targets_for_name(row):
                        if nonnegative(target['path_name_score'])>row['score']:
                            raise ValueError('Path grade exceeds its own scoped name')
                        yield target
            if batch: consume(page_targets(),key)
        else: consume(batch,key)
        frontiers[key]=0. if channels[key].exhausted else previous
    result=[dict(siret=s,siren=p,score=v) for v,_,s,p in sorted(heap,key=lambda r:(-r[0],r[2]))]
    if len(result)>cap or len({r['siret'] for r in result})!=len(result): raise ValueError('Initial cap drift')
    return result,trace(True)
