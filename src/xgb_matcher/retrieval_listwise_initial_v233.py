"""TRAIN-only listwise calibration of actual native initial-index proofs."""
from copy import deepcopy
import numpy as np
from scipy.optimize import minimize


def design(row,held,protected,acceptable):
    refs=row['references'];ids=sorted(refs)
    if row['positive_siret'] not in acceptable[row['query_id']]:raise ValueError('Exact TRAIN positive cannot be negative')
    x=np.asarray(row['features'],dtype=float);fixed=np.asarray(row['fixed_scores'][str(held)],dtype=float)
    if x.shape!=(len(ids),11) or fixed.shape!=(len(ids),) or not np.isfinite(x).all() or not np.isfinite(fixed).all():
        raise ValueError('Actual finite aligned native217 features and fixed score')
    p=ids.index(row['positive_siret']);safe=[i for i,s in enumerate(ids) if s not in acceptable[row['query_id']] and refs[s] not in protected]
    if not safe:raise ValueError('No safe TRAIN negative; never silently drop query')
    return x[p]-x[safe],fixed[p]-fixed[safe]


def loss(w,x,offsets,sizes,prior,ridge=.005):
    starts=np.r_[0,np.cumsum(sizes)[:-1]];logits=-(x@w+offsets)
    maximum=np.maximum(0.,np.maximum.reduceat(logits,starts))
    e=np.exp(logits-np.repeat(maximum,sizes))
    denominator=np.exp(-maximum)+np.add.reduceat(e,starts)
    probability=e/np.repeat(denominator,sizes)/len(sizes)
    delta=w-prior
    return float(np.mean(maximum+np.log(denominator))+ridge*(delta@delta)), -x.T@probability+2*ridge*delta


def fit(rows,registry,original,acceptable,held):
    if held not in (2,3,4) or set(registry)!={'2','3','4'} or len(rows)!=64 or len({r['query_id'] for r in rows})!=64:
        raise ValueError('Actual full64 safe TRAIN source and full registry required')
    m=original[str(held)];prior=np.asarray(m['weights'],dtype=float);protected=set(registry[str(held)])
    if prior.shape!=(11,) or not np.isfinite(prior).all() or (prior<0).any():raise ValueError('Actual fixed original11 head')
    matrices=[];offsets=[];sizes=[];kept=[];excluded=0
    for r in rows:
        if r['fold'] not in (2,3,4) or r['positive_siren'] not in registry[str(r['fold'])]:raise ValueError('TRAIN entity registry mismatch')
        if r['fold']==held or r['positive_siren'] in protected:excluded+=1;continue
        if r['query_id'] not in m['query_ids']:raise ValueError('Preserve original protected OOF membership')
        x,o=design(r,held,protected,acceptable)
        matrices.append(x);offsets.append(o);sizes.append(len(o));kept.append(r['query_id'])
    if len(kept)<16:raise ValueError('At least16 safe TRAIN queries')
    x=np.vstack(matrices);o=np.concatenate(offsets);sizes=np.asarray(sizes,dtype=int)
    result=minimize(loss,prior,args=(x,o,sizes,prior),method='L-BFGS-B',jac=True,bounds=[(0.,64.)]*11,
        options=dict(maxiter=500,ftol=1e-10,gtol=1e-7))
    if not result.success or not np.isfinite(result.x).all():raise ValueError('Actual native listwise fit failed')
    model=deepcopy(m);model['weights']=result.x.tolist()
    model['native_training']=dict(query_ids=kept,training_queries=len(kept),training_pairs=len(x),
        excluded_held_fold_or_positive_entities=excluded,heldout_entity_overlap=0,ridge=.005,
        objective='EQUAL_QUERY_LOG_ONE_PLUS_SUM_EXP_NEGATIVE_MINUS_POSITIVE_FULL_ACTUAL217',
        objective_before=loss(prior,x,o,sizes,prior)[0],objective_after=float(result.fun),
        actual_native_kernel_features=True,semantic_owners_relations_xy_fixed=True,
        changed_native_indices=list(range(11)),identifiers_as_features=False,
        parameters_selected_on_outer_controls=False,recall_measured=False)
    return model


def controls(rows,registry,original,models,acceptable):
    """OOF TRAIN comparisons on mined negatives only; never national recall."""
    results=[]
    for r in rows:
        held=r['fold'];f=str(held);m=models[f]
        if r['query_id'] in m['native_training']['query_ids'] or r['positive_siren'] not in registry[f]:
            raise ValueError('Only held queries/entities in TRAIN controls')
        # Controls must include all eligible national/local competitors; their
        # entities need not be TRAIN-safe because they do not fit a model.
        x,o=design(r,held,set(),acceptable)
        before=x@np.asarray(original[f]['weights'])+o;after=x@np.asarray(m['weights'])+o
        results.append(dict(query_id=r['query_id'],fold=held,negative_references=len(o),
            negative_outrank_count_before=int(np.sum(before<=0)),negative_outrank_count_after=int(np.sum(after<=0)),
            hardest_margin_before=float(np.min(before)),hardest_margin_after=float(np.min(after))))
    return dict(scope='OOF_TRAIN64_CENSORED_MINED_NEGATIVES_NOT_NATIONAL_RECALL',rows=results,
        mean_negative_outrank_count_before=float(np.mean([r['negative_outrank_count_before'] for r in results])),
        mean_negative_outrank_count_after=float(np.mean([r['negative_outrank_count_after'] for r in results])),
        controls_used_in_model_selection=False,recall_measured=False)
