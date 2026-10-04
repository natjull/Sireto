"""Convex top100 mean-hinge on TRAIN negatives, not a national recall claim.

Native217 features stay exact. Coefficients are learned before serving and
compiled into the global initial query, never applied after a fixed pool.
"""
from copy import deepcopy
import warnings

import numpy as np
from scipy.optimize import linprog, OptimizeWarning
from scipy.sparse import coo_array

from .retrieval_listwise_initial_v233 import design

K = 100
MARGIN = 1.
RIDGE = .005  # L1 departure from the actual228 native head, not L2 of233.


def objective(weights, matrices, offsets, prior):
    losses = []
    for x, o in zip(matrices, offsets):
        z = np.r_[0., MARGIN-(x@weights+o)]
        k = min(K, len(z))
        losses.append(max(0., float(np.partition(z, len(z)-k)[-k:].mean())))
    return float(np.mean(losses)+RIDGE*np.abs(weights-prior).sum())


def solve(matrices, offsets, prior):
    """Exact sparse epigraph: topK(z)=min_tau tau+sum ReLU(z-tau)/K.

    Add the true-class zero coordinate and outer positive part. This is a
    query-balanced topK hinge adaptation with frozen fixed-score offsets.
    No sample/negative truncation is performed by this optimizer.
    """
    prior=np.asarray(prior,dtype=float); n=len(matrices)
    if n<1 or len(offsets)!=n or prior.shape!=(11,) or not np.isfinite(prior).all() or np.any((prior<0)|(prior>64)):
        raise ValueError('Finite complete native11 prior and nonempty TRAIN')
    starts=np.r_[0,np.cumsum([len(o)+1 for o in offsets])]; total=int(starts[-1])
    # w11 | abs_departure11 | tau per query | hinge per query | ReLU per competitor/zero
    tau0=22; hinge0=tau0+n; u0=hinge0+n; size=u0+total
    cost=np.zeros(size);cost[11:22]=RIDGE;cost[hinge0:u0]=1./n
    rr=[];cc=[];vv=[];rhs=[]
    def constraint(terms,bound):
        row=len(rhs);rhs.append(float(bound))
        for col,value in terms:
            if value:
                rr.append(row);cc.append(col);vv.append(float(value))
    for j in range(11):
        constraint([(j,1.),(j+11,-1.)],prior[j])
        constraint([(j,-1.),(j+11,-1.)],-prior[j])
    for q,(x,o) in enumerate(zip(matrices,offsets)):
        x=np.asarray(x,dtype=float);o=np.asarray(o,dtype=float)
        if x.shape!=(len(o),11) or not len(o) or not np.isfinite(x).all() or not np.isfinite(o).all():
            raise ValueError('Actual aligned finite native TRAIN differences')
        begin=int(starts[q]); k=min(K,len(o)+1)
        constraint([(tau0+q,-1.),(u0+begin,-1.)],0.) # true class z=0
        for i in range(len(o)):
            # margin - x*w - fixed_delta - tau <= u
            constraint([*((j,-x[i,j]) for j in range(11)),
                        (tau0+q,-1.),(u0+begin+i+1,-1.)],o[i]-MARGIN)
        constraint([(tau0+q,1.),(hinge0+q,-1.),
                    *((u0+i,1./k) for i in range(begin,int(starts[q+1])))],0.)
    matrix=coo_array((vv,(rr,cc)),shape=(len(rhs),size)).tocsc()
    bounds=[(0.,64.)]*11+[(0.,None)]*11+[(None,None)]*n+[(0.,None)]*(n+total)
    # Installed SciPy forwards threads verbatim to HiGHS; keep one thread.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='Unrecognized options detected:.*',category=OptimizeWarning)
        r=linprog(cost,A_ub=matrix,b_ub=np.asarray(rhs),bounds=bounds,method='highs',
                  options=dict(threads=1,time_limit=20.,dual_feasibility_tolerance=1e-8,
                               primal_feasibility_tolerance=1e-8))
    if not r.success or r.x is None or not np.isfinite(r.x).all():
        raise ValueError('Convex top100 fit failed: '+r.message)
    weights=r.x[:11]
    value=objective(weights,matrices,offsets,prior)
    if abs(value-r.fun)>2e-6 or np.max(matrix@r.x-np.asarray(rhs),initial=0.)>2e-6:
        raise ValueError('LP/direct topK objective or feasible epigraph mismatch')
    return weights,dict(objective_before=objective(prior,matrices,offsets,prior),
                        objective_after=value,lp_objective=float(r.fun),iterations=int(r.nit))


def fit(rows,registry,original,acceptable,held,priors):
    if held not in (2,3,4) or set(registry)!={'2','3','4'} or len(rows)!=64 or len({r['query_id'] for r in rows})!=64:
        raise ValueError('Actual full64 TRAIN and three safe folds')
    m=original[str(held)];p=priors[str(held)]
    if ({k:v for k,v in p.items() if k not in ('weights','place_training')} !=
            {k:v for k,v in m.items() if k!='weights'} or
            any(p['weights'][i]!=m['weights'][i] for i in range(11) if i!=4)):
        raise ValueError('Only actual safe228 place-adjusted prior')
    protected=set(registry[str(held)]);matrices=[];offsets=[];kept=[];excluded=0
    for row in rows:
        if row['fold'] not in (2,3,4) or row['positive_siren'] not in registry[str(row['fold'])]:
            raise ValueError('TRAIN entity registry mismatch')
        if row['fold']==held or row['positive_siren'] in protected:
            excluded+=1;continue
        if row['query_id'] not in m['query_ids']:
            raise ValueError('Preserve original protected OOF query scope')
        x,o=design(row,held,protected,acceptable)
        matrices.append(x);offsets.append(o);kept.append(row['query_id'])
    if len(kept)<16:
        raise ValueError('At least16 safe TRAIN queries')
    weights,details=solve(matrices,offsets,np.asarray(p['weights']))
    model=deepcopy(m);model['weights']=weights.tolist()
    model['native_training']=dict(query_ids=kept,training_queries=len(kept),
        training_pairs=sum(len(o) for o in offsets),excluded_held_fold_or_positive_entities=excluded,
        heldout_entity_overlap=0,objective='EQUAL_QUERY_TOP100_MEAN_HINGE_TRUE_CLASS_ZERO_L1_PRIOR228',
        k=K,margin=MARGIN,ridge=RIDGE,regularizer='L1_DISTANCE_TO_FROZEN228',
        queries_with_fewer_than100_negative_classes=sum(len(o)+1<K for o in offsets),
        criterion_scope='CENSORED_TRAIN_NEGATIVES_NOT_NATIONAL_RECALL',
        actual_native_kernel_features=True,semantic_owners_relations_xy_fixed=True,
        changed_native_indices=list(range(11)),identifiers_as_features=False,
        parameters_selected_on_outer_controls=False,recall_measured=False,**details)
    return model


def controls(rows,registry,original,models,acceptable,priors):
    results=[]
    for row in rows:
        held=row['fold'];f=str(held);model=models[f]
        if row['query_id'] in model['native_training']['query_ids'] or row['positive_siren'] not in registry[f]:
            raise ValueError('Only held TRAIN queries/entities')
        x,o=design(row,held,set(),acceptable)
        prior=np.asarray(priors[f]['weights']);weights=np.asarray(model['weights'])
        before=x@prior+o;after=x@weights+o
        results.append(dict(query_id=row['query_id'],fold=held,negative_references=len(o),
            negative_outrank_count_before=int(np.sum(before<=0)),
            negative_outrank_count_after=int(np.sum(after<=0)),
            hinge_before=objective(prior,[x],[o],prior),
            hinge_after_without_regularizer=objective(weights,[x],[o],weights)))
    return dict(scope='OOF_TRAIN64_CENSORED_MINED_NEGATIVES_NOT_NATIONAL_RECALL',rows=results,
        mean_negative_outrank_count_before=float(np.mean([r['negative_outrank_count_before'] for r in results])),
        mean_negative_outrank_count_after=float(np.mean([r['negative_outrank_count_after'] for r in results])),
        controls_used_in_model_selection=False,recall_measured=False)
