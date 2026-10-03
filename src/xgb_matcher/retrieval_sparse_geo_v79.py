"""Cross-fitted geographic residual for the initial sparse query metric.

The native11/semantic prefix is fixed by V77. Only8 monotone GIS coefficients
are fitted, using base predictions from another permitted source fold.
No reservoir, ranker, decider, state/radius filter or label-side serving data.
"""
import numpy as np
from scipy.optimize import minimize

from src.xgb_matcher.retrieval_native_v57 import fit as native_fit
from src.xgb_matcher.retrieval_nested_v53 import positions
from src.xgb_matcher.retrieval_safe_sparse_v77 import fit_view
from src.xgb_matcher.retrieval_sparse_query_v59 import expand,field_bags,semantic_score,solve


def model_view(q,rows,scenes,documents,registry,indices,blocked,fit_held,config,choice):
    vocab,y,scope=fit_view(rows,documents,indices,blocked,config)
    safe_registry={f:list(parents) for f,parents in registry.items()}
    safe_registry[str(fit_held)]=sorted(set(blocked))
    native_scenes=[dict(s,positive_features=s['positive_features'][:11],
        negatives=[dict(n,features=n['features'][:11]) for n in s['negatives']]) for s in scenes]
    native=native_fit(native_scenes,fit_held,safe_registry,config['native_ridge'])
    if set(native['query_ids'])!={rows[i]['query_id'] for i in indices}:
        raise ValueError('Cross-fit native query scope drift')
    projection=np.zeros((q.shape[1],len(vocab)),dtype=np.float32) if choice['alpha']==0 else solve(q,y,indices,choice['ridge'])
    return dict(weights=native['weights'],alpha=choice['alpha'],projection=projection,vocabulary=vocab,
                scope=scope,query_ids=native['query_ids'])


def crossfit_views(q,rows,scenes,documents,registry,indices,blocked,config,choice):
    result={}; scope={}
    for fold in sorted({rows[i]['fold'] for i in indices}):
        excluded=set(blocked)|set(registry[str(fold)])
        fit=[i for i in indices if rows[i]['fold']!=fold and rows[i]['positive_siren'] not in excluded]
        if len(fit)<16:
            raise ValueError('Insufficient entity-disjoint calibration support')
        result[str(fold)]=model_view(q,rows,scenes,documents,registry,fit,excluded,fold,config,choice)
        scope[str(fold)]=dict(training_queries=len(fit),prediction_fold=fold,
            fitted_folds=sorted({rows[i]['fold'] for i in fit}),protected_entity_overlap=0,
            vocabulary_scope=result[str(fold)]['scope'])
    return result,scope


def design(q,rows,scenes,bags,indices,models,blocked):
    offsets=[]; geometry=[]; sizes=[]; protected=set(blocked)
    for i in indices:
        r,s=rows[i],scenes[i]; m=models[str(r['fold'])]
        if any(r[k]!=s[k] for k in ('query_id','fold','positive_siren')) or len(r['negatives'])!=len(s['negatives']):
            raise ValueError('Training scene alignment drift')
        if any(n['siren']!=nf['siren'] for n,nf in zip(r['negatives'],s['negatives'])):
            raise ValueError('Negative membership drift')
        safe=[j for j,n in enumerate(r['negatives']) if n['siren'] not in protected]
        if not safe:
            raise ValueError('No safe negatives; never remove a query')
        f=np.array(s['positive_features'],dtype=float)
        nf=np.array([s['negatives'][j]['features'] for j in safe],dtype=float)
        if f.shape!=(19,) or nf.shape!=(len(safe),19) or not np.isfinite(f).all() or not np.isfinite(nf).all():
            raise ValueError('Native19/multiscale proof drift')
        terms=expand(q[i],m['projection'],m['vocabulary'],r['crm_name'],8)
        sem_positive=semantic_score(bags[r['positive_siret']],terms,f[4])
        sem_negative=np.array([semantic_score(bags[r['negatives'][j]['siret']],terms,n[4]) for j,n in zip(safe,nf)])
        offsets.extend((f[:11]-nf[:,:11])@np.array(m['weights'])+m['alpha']*(sem_positive-sem_negative))
        geometry.extend(f[11:]-nf[:,11:]); sizes.append(len(safe))
    return np.array(offsets),np.array(geometry),np.array(sizes,dtype=int)


def loss(weights,offsets,x,sizes,ridge):
    starts=np.r_[0,np.cumsum(sizes)[:-1]]
    logits=-(offsets+x@weights); maxima=np.maximum(0.,np.maximum.reduceat(logits,starts))
    exponentials=np.exp(logits-np.repeat(maxima,sizes))
    denominators=np.exp(-maxima)+np.add.reduceat(exponentials,starts)
    value=np.mean(maxima+np.log(denominators))+ridge*np.sum(weights*weights)
    probabilities=exponentials/np.repeat(denominators,sizes)/len(sizes)
    return float(value),-x.T@probabilities+2*ridge*weights


def fit_geometry(offsets,x,sizes,config):
    if x.shape!=(len(offsets),8) or sizes.sum()!=len(offsets) or len(sizes)<16 or np.any(sizes<=0):
        raise ValueError('Geographic calibration denominator drift')
    if not np.isfinite(x).all() or not np.isfinite(offsets).all():
        raise ValueError('Nonfinite geographic design')
    prior=np.zeros(8); ridge=config['geographic_ridge']
    result=minimize(loss,prior,args=(offsets,x,sizes,ridge),jac=True,method='L-BFGS-B',bounds=[(0.,64.)]*8,
        options=dict(maxiter=config['maxiter'],ftol=1e-8,gtol=1e-5))
    if not np.isfinite(result.x).all() or result.fun>loss(prior,offsets,x,sizes,ridge)[0]+1e-8:
        raise ValueError('Geographic optimization drift')
    return result.x,dict(optimizer_converged=bool(result.success),iterations=int(result.nit),
        optimizer_message=str(result.message),objective_before=loss(prior,offsets,x,sizes,ridge)[0],
        objective_after=float(result.fun),parameters=8,training_queries=len(sizes),training_pairs=len(offsets))


def summarize(offsets,x,sizes,weights):
    starts=np.r_[0,np.cumsum(sizes)[:-1]]; margins=offsets+x@weights
    hardest=np.minimum.reduceat(margins,starts); counts=np.add.reduceat((margins<=0).astype(int),starts)
    return dict(queries=len(sizes),pairs=len(margins),head_error=float(np.mean(hardest<=0)),
        mean_outrank_count=float(np.mean(counts)),hardest_margin=float(np.mean(hardest)),
        reciprocal_rank=float(np.mean(1./(1.+counts))))


def select(q,rows,scenes,documents,registry,held,config,choice,final_model):
    membership=[dict(query_id=r['query_id'],fold=r['fold'],positive_siren=r['positive_siren']) for r in rows]
    train,validation,inner=positions(membership,held,registry); outer=set(registry[str(held)])
    blocked=outer|inner; bags={s:field_bags(f) for s,f in documents.items()}
    inner_model=model_view(q,rows,scenes,documents,registry,train,blocked,held,config,choice)
    calibration,inner_scope=crossfit_views(q,rows,scenes,documents,registry,train,blocked,config,choice)
    o,x,n=design(q,rows,scenes,bags,train,calibration,blocked); geo,fit=fit_geometry(o,x,n,config)
    ov,xv,nv=design(q,rows,scenes,bags,validation,{str(f):inner_model for f in (2,3,4)},outer)
    candidates=[dict(family='BASELINE',weights=np.zeros(8).tolist(),inner=summarize(ov,xv,nv,np.zeros(8)))]
    if fit['optimizer_converged']:
        candidates.append(dict(family='ADD_GEO8',weights=geo.tolist(),inner=summarize(ov,xv,nv,geo)))
    selected=min(candidates,key=lambda r:(r['inner']['head_error'],r['inner']['mean_outrank_count'],r['family']!='BASELINE'))
    final_indices=train+validation
    if set(final_model['query_ids'])!={rows[i]['query_id'] for i in final_indices}:
        raise ValueError('Frozen V77 prefix query scope drift')
    if selected['family']=='ADD_GEO8':
        calibration,final_scope=crossfit_views(q,rows,scenes,documents,registry,final_indices,outer,config,choice)
        o,x,n=design(q,rows,scenes,bags,final_indices,calibration,outer)
        weights,final_fit=fit_geometry(o,x,n,config)
        if not final_fit['optimizer_converged']:
            raise ValueError('Selected geographic refit failed convergence')
    else:
        weights=np.zeros(8); final_fit=None; final_scope={}
    test=[i for i,r in enumerate(rows) if r['fold']==held]
    oo,xo,no=design(q,rows,scenes,bags,test,{str(f):final_model for f in (2,3,4)},())
    return weights,dict(heldout_fold=held,selected=selected,candidates=candidates,inner_fit=fit,final_fit=final_fit,
        inner_crossfit_scope=inner_scope,final_crossfit_scope=final_scope,
        frozen_semantic_choice=dict(choice),frozen_semantic_hyperparameters_not_reselected_in_crossfit=True,
        outer_before=summarize(oo,xo,no,np.zeros(8)),outer_after=summarize(oo,xo,no,weights),
        outer_controls_used_in_selection=False,base_predictions_out_of_fit_source_fold=True,
        identifiers_as_features=False,query_ids=[rows[i]['query_id'] for i in final_indices],recall_measured=False)
