"""Inner entity-disjoint selection of native query metric capacity/penalty."""
import hashlib

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from src.xgb_matcher.retrieval_contrastive_v52 import fit as fit_full
from src.xgb_matcher.retrieval_learned_v41 import unit_vectors,transform_queries


def inner_entity(parent):
    return int(hashlib.sha256(('v53-inner|'+parent).encode()).hexdigest(),16)%5==0


def positions(memberships,held,registry):
    if held not in (2,3,4) or set(registry)!={'2','3','4'} or not registry[str(held)]: raise ValueError('Outer registry drift')
    if len({r['query_id'] for r in memberships})!=len(memberships): raise ValueError('Duplicate training query')
    if any(r['fold'] not in (2,3,4) or r['positive_siren'] not in registry[str(r['fold'])] for r in memberships): raise ValueError('Membership drift')
    protected=set(registry[str(held)])
    eligible=[i for i,r in enumerate(memberships) if r['fold']!=held and r['positive_siren'] not in protected]
    inner={memberships[i]['positive_siren'] for i in eligible if inner_entity(memberships[i]['positive_siren'])}
    train=[i for i in eligible if memberships[i]['positive_siren'] not in inner]
    validation=[i for i in eligible if memberships[i]['positive_siren'] in inner]
    if len(train)<16 or len(validation)<8: raise ValueError('Insufficient entity-disjoint inner data')
    return train,validation,inner


def limited_fit(q,p,negatives,memberships,held,registry,capacity,ridge,temperature,hard_count,maxiter):
    if capacity=='FULL':
        return fit_full(q,p,negatives,memberships,held,registry,ridge,temperature,hard_count,maxiter)
    if capacity not in ('DIAGONAL','FIELD_SCALE'): raise ValueError('Unknown native metric capacity')
    q,p=unit_vectors(q),unit_vectors(p); dimension=q.shape[1]
    if dimension%2 or q.shape!=p.shape: raise ValueError('Structured shape drift')
    coordinate=np.arange(dimension) if capacity=='DIAGONAL' else np.repeat([0,1],dimension//2)
    params=dimension if capacity=='DIAGONAL' else 2
    protected=set(registry[str(held)]); xs=[]; sw=[]; kept=[]; source_pairs=0
    for i,r in enumerate(memberships):
        if r['fold']==held or r['positive_siren'] in protected: continue
        safe=[j for j,s in enumerate(r['negative_sirens']) if s not in protected]
        if not safe: continue
        n=unit_vectors(negatives[i])[safe]; source_pairs+=len(n)
        take=np.argsort(-(n@q[i]),kind='stable')[:hard_count]
        components=(p[i]-n[take])*q[i]
        if capacity=='FIELD_SCALE': components=components.reshape(len(take),2,dimension//2).sum(axis=2)
        xs.extend(components); sw.extend([1./len(take)]*len(take)); kept.append(i)
    if len(kept)<16: raise ValueError('Insufficient safe limited fit')
    x=np.asarray(xs,dtype=np.float64); sw=np.asarray(sw)/len(kept); base=np.ones(params)
    def objective(w):
        margin=temperature*(x@w); delta=w-base
        return float(sw@np.logaddexp(0.,-margin)+ridge*np.mean(delta*delta)),(
            x.T@(-temperature*sw*expit(-margin))+2*ridge*delta/params)
    fitted=minimize(objective,base,method='L-BFGS-B',jac=True,bounds=[(0.,None)]*params,
        options=dict(maxiter=maxiter,ftol=1e-8,gtol=1e-6))
    if not np.isfinite(fitted.x).all() or fitted.fun>objective(base)[0]+1e-8: raise ValueError('Limited fit drift')
    w=np.diag(fitted.x[coordinate]).astype(np.float32)
    return w,dict(capacity=capacity,parameters=params,heldout_fold=held,training_queries=len(kept),training_pairs=len(x),
        query_ids=[memberships[i]['query_id'] for i in kept],safe_source_negative_pairs=source_pairs,
        training_folds=sorted({memberships[i]['fold'] for i in kept}),heldout_entity_overlap=0,
        ridge=ridge,objective_before=objective(base)[0],objective_after=float(fitted.fun),optimizer_converged=bool(fitted.success))


def controls(q,p,negatives,memberships,indices,w,forbidden_parents=()):
    adapted=transform_queries(q[indices],w); q,p=unit_vectors(q),unit_vectors(p); blocked=set(forbidden_parents); rows=[]
    for j,i in enumerate(indices):
        r=memberships[i]; safe=[k for k,parent in enumerate(r['negative_sirens']) if parent not in blocked]
        if not safe: raise ValueError('Control has no safe negatives; do not filter it')
        n=unit_vectors(negatives[i])[safe]; nb=n@q[i]; na=n@adapted[j]; pb=float(q[i]@p[i]); pa=float(adapted[j]@p[i])
        rows.append(dict(query_id=r['query_id'],fold=r['fold'],negative_pairs=len(n),
            negative_outrank_fraction_before=float(np.mean(nb>=pb)),negative_outrank_fraction_after=float(np.mean(na>=pa)),
            hardest_margin_before=float(pb-nb.max()),hardest_margin_after=float(pa-na.max())))
    return rows


def summarize(rows):
    if not rows: raise ValueError('Empty controls')
    keys=('negative_outrank_fraction_before','negative_outrank_fraction_after','hardest_margin_before','hardest_margin_after')
    return dict(queries=len(rows),**{key:float(np.mean([r[key] for r in rows])) for key in keys})


def select_and_refit(q,p,negatives,memberships,held,registry,config):
    train,validation,inner=positions(memberships,held,registry)
    # Adding inner entities to the exclusion registry changes no source fold.
    protected={k:list(v) for k,v in registry.items()}; protected[str(held)]=sorted(set(protected[str(held)])|inner)
    identity=np.eye(q.shape[1],dtype=np.float32)
    baseline=summarize(controls(q,p,negatives,memberships,validation,identity,registry[str(held)]))
    candidates=[dict(capacity='IDENTITY',ridge=None,parameters=0,inner=baseline)]
    for capacity in config['capacities']:
        for ridge in config['ridges']:
            w,report=limited_fit(q,p,negatives,memberships,held,protected,capacity,ridge,
                config['temperature'],config['hard_count'],config['maxiter'])
            if set(report['query_ids'])!={memberships[i]['query_id'] for i in train}:
                raise ValueError('Inner training coverage drift')
            inner_score=summarize(controls(q,p,negatives,memberships,validation,w,registry[str(held)]))
            candidates.append(dict(capacity=capacity,ridge=ridge,parameters=report.get('parameters',q.shape[1]**2),inner=inner_score))
    # All hyperparameters selected from inner data ONLY, including identity.
    selected=min(candidates,key=lambda r:(r['inner']['negative_outrank_fraction_after'],
        r['parameters'],-r['ridge'] if r['ridge'] is not None else 0.))
    if selected['capacity']=='IDENTITY': w=identity; final=dict(capacity='IDENTITY',parameters=0,training_queries=0,query_ids=[])
    else:
        w,final=limited_fit(q,p,negatives,memberships,held,registry,selected['capacity'],selected['ridge'],
            config['temperature'],config['hard_count'],config['maxiter'])
    outer=[i for i,r in enumerate(memberships) if r['fold']==held]
    measured=controls(q,p,negatives,memberships,outer,w)
    return w,dict(heldout_fold=held,inner_validation_queries=len(validation),inner_training_queries=len(train),
        inner_entities=sorted(inner),candidates=candidates,selected=selected,final_fit=final,outer_controls=measured,
        outer=summarize(measured),outer_controls_used_in_selection=False,identifier_features=False,recall_measured=False)
