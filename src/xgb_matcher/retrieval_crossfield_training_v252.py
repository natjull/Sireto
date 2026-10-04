"""TRAIN-only three new proof coefficients, original228 head remains fixed."""
from copy import deepcopy

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .retrieval_crossfield_proofs_v250 import FEATURES,conditional_parity
from .retrieval_listwise_initial_v233 import design as original_design

RIDGE=.005


def design(row,held,protected,acceptable,prior):
    x,old_offset=original_design(row,held,protected,acceptable)
    conditional_parity(row['features'],row['extra_features'])
    ids=sorted(row['references']);p=ids.index(row['positive_siret'])
    safe=[i for i,s in enumerate(ids) if s not in acceptable[row['query_id']] and row['references'][s] not in protected]
    extra=np.asarray(row['extra_features'],dtype=float)
    return extra[p]-extra[safe],x@np.asarray(prior,dtype=float)+old_offset


def loss(w,x,offset,sample_weights):
    margin=x@w+offset
    return float(sample_weights@np.logaddexp(0.,-margin)+RIDGE*(w@w)),x.T@(-sample_weights*expit(-margin))+2*RIDGE*w


def fit(rows,registry,original,acceptable,held,priors):
    if held not in (2,3,4) or set(registry)!={'2','3','4'} or len(rows)!=64 or len({r['query_id'] for r in rows})!=64:
        raise ValueError('Complete actual64 TRAIN and full safe registry')
    m=original[str(held)];prior=priors[str(held)]
    if ({k:v for k,v in prior.items() if k not in ('weights','place_training')}!=
        {k:v for k,v in m.items() if k!='weights'} or
        any(prior['weights'][i]!=m['weights'][i] for i in range(11) if i!=4)):
        raise ValueError('Same safe228 frozen11 prior')
    protected=set(registry[str(held)]);matrices=[];offsets=[];sw=[];kept=[];excluded=0
    for row in rows:
        if row['fold'] not in (2,3,4) or row['positive_siren'] not in registry[str(row['fold'])]:
            raise ValueError('TRAIN entity membership')
        if row['fold']==held or row['positive_siren'] in protected:
            excluded+=1;continue
        if row['query_id'] not in m['query_ids']:raise ValueError('Same original protected OOF membership')
        x,o=design(row,held,protected,acceptable,prior['weights'])
        matrices.append(x);offsets.append(o);sw.extend([1./len(o)]*len(o));kept.append(row['query_id'])
    if len(kept)<16:raise ValueError('At least16 safe TRAIN queries')
    x=np.vstack(matrices);o=np.concatenate(offsets);sw=np.asarray(sw)/len(kept)
    start=np.zeros(3)
    r=minimize(loss,start,args=(x,o,sw),method='L-BFGS-B',jac=True,bounds=[(0.,64.)]*3,
        options=dict(maxiter=500,ftol=1e-10,gtol=1e-7))
    if not r.success or not np.isfinite(r.x).all():raise ValueError('Three-proof fit failed')
    model=deepcopy(prior);model['extra_weights']=r.x.tolist()
    model['extra_proof_training']=dict(query_ids=kept,training_queries=len(kept),training_pairs=len(x),
        excluded_held_fold_or_positive_entities=excluded,heldout_entity_overlap=0,
        features=list(FEATURES),ridge=RIDGE,objective='EQUAL_QUERY_PAIRWISE_LOGISTIC_FIXED228_HEAD_L2_NEW3',
        objective_before=loss(start,x,o,sw)[0],objective_after=float(r.fun),original11_fixed=True,
        conditional_native_parity_verified=True,identifiers_as_features=False,
        parameters_selected_on_outer_controls=False,recall_measured=False)
    return model


def controls(rows,registry,original,models,acceptable,priors):
    output=[]
    for row in rows:
        f=str(row['fold']);m=models[f]
        if row['query_id'] in m['extra_proof_training']['query_ids'] or row['positive_siren'] not in registry[f]:
            raise ValueError('Held TRAIN controls only')
        x,before=design(row,row['fold'],set(),acceptable,priors[f]['weights'])
        after=before+x@np.asarray(m['extra_weights'])
        output.append(dict(query_id=row['query_id'],fold=row['fold'],negative_references=len(before),
            negative_outrank_count_before=int(np.sum(before<=0)),negative_outrank_count_after=int(np.sum(after<=0))))
    return dict(scope='OOF_TRAIN64_CENSORED_MINED_NEGATIVES_NOT_NATIONAL_RECALL',rows=output,
        mean_negative_outrank_count_before=float(np.mean([r['negative_outrank_count_before'] for r in output])),
        mean_negative_outrank_count_after=float(np.mean([r['negative_outrank_count_after'] for r in output])),
        controls_used_in_model_selection=False,recall_measured=False)
