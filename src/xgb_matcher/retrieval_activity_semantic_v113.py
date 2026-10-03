"""Semantic prior from official activity descriptions, TRAIN controls only.

The embedding input is the official LABEL, never its categorical code, site
identifier, CRM alias, or guessed nomenclature. Unknown descriptions score
zero without removal from any denominator or future national site corpus.
"""
import numpy as np

from src.xgb_matcher.retrieval_activity_decoder_v111 import query_design,validate,summary
from src.xgb_matcher.retrieval_nested_v53 import positions


def normalized(values):
    x=np.asarray(values,dtype=np.float64)
    if x.ndim!=2 or not np.isfinite(x).all():raise ValueError('Semantic vector shape/finite drift')
    return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)


def class_lookup(nomenclatures,activities,labels):
    refs={(r['nomenclature'],r['code']):i for i,r in enumerate(labels)}
    if len(refs)!=len(labels):raise ValueError('Duplicate versioned official activity description')
    # -1 is UNKNOWN evidence, not an admissibility/exclusion decision.
    return np.array([[refs.get((nom,code),-1) for code in activities] for nom in nomenclatures],dtype=np.int32)


def validate_semantics(rows,registry,codes,classes):
    if any(v < -1 or v>=classes for v in codes.values()):raise ValueError('Official definition join range drift')
    # Validate every identity/fold/negative; map UNKNOWN only for validator range.
    validate(rows,registry,{k:0 if v==-1 else v+2 for k,v in codes.items()},classes+2)


def fit(x,docs,rows,codes,indices,blocked,family,ridge=None):
    if len(indices)<16 or any(rows[i]['positive_siren'] in blocked for i in indices):
        raise ValueError('Unsafe semantic activity training family/size')
    if family=='TRAIN_FREQUENCY':
        frequencies=np.zeros(len(docs),dtype=np.float32)
        for i in indices:
            code=codes[rows[i]['positive_siret']]
            if code>=0:frequencies[code]+=1./len(indices)
        return frequencies
    if family=='ZERO_SHOT':
        projection=np.zeros((x.shape[1],docs.shape[1]),dtype=np.float32)
        if x.shape[1]!=docs.shape[1]+1:raise ValueError('Official/query encoder dimension drift')
        projection[:-1]=np.eye(docs.shape[1]);return projection
    if family!='SEMANTIC_RIDGE' or ridge is None or ridge<=0:raise ValueError('Unknown semantic activity recipe')
    target=np.zeros((len(indices),docs.shape[1]))
    for j,i in enumerate(indices):
        code=codes[rows[i]['positive_siret']]
        if code>=0:target[j]=docs[code]
    train=x[indices]
    p=np.linalg.solve(train.T@train+ridge*np.eye(x.shape[1]),train.T@target).astype(np.float32)
    if not np.isfinite(p).all():raise ValueError('Nonfinite activity semantic fit')
    return p


def scores(x,docs,parameter,family):
    if family=='TRAIN_FREQUENCY':
        if parameter.shape!=(len(docs),) or not np.isfinite(parameter).all() or np.any(parameter<0):
            raise ValueError('Training frequency shape drift')
        return np.tile(parameter,(len(x),1))
    if parameter.shape!=(x.shape[1],docs.shape[1]) or not np.isfinite(parameter).all():
        raise ValueError('Semantic projection shape/finite drift')
    return np.clip(normalized(x@parameter)@docs.T,0.,1.)


def controls(x,docs,parameter,family,rows,codes,indices,blocked=()):
    values=scores(x[indices],docs,parameter,family);result=[]
    for j,i in enumerate(indices):
        row=rows[i];pc=codes[row['positive_siret']]
        safe=[n for n in row['negatives'] if n['siren'] not in blocked]
        if not safe:raise ValueError('No safe semantic control negatives; no row filtering')
        nc=np.array([codes[n['siret']] for n in safe],dtype=np.int32)
        pos=0. if pc<0 else float(values[j,pc])
        negative=np.zeros(len(nc));known=nc>=0;negative[known]=values[j,nc[known]]
        margin=pos-negative;rank=np.lexsort((np.arange(len(docs)),-values[j]))
        result.append(dict(query_id=row['query_id'],fold=row['fold'],pairs=len(safe),
            outrank=int(np.count_nonzero(margin<=0)),ties=int(np.count_nonzero(margin==0)),
            same_activity=int(np.count_nonzero(nc==pc)) if pc>=0 else 0,unknown_positive=pc<0,
            unknown_negatives=int(np.count_nonzero(nc<0)),
            top1=bool(pc>=0 and values[j,pc]>0 and pc==rank[0]),
            top5=bool(pc>=0 and values[j,pc]>0 and pc in rank[:5]),
            hardest_margin=float(margin.min())))
    return result


def select(vectors,descriptions,rows,registry,codes,held,ridges=(1.,10.,100.)):
    docs=normalized(descriptions);x=query_design(vectors)
    if (len(x)!=len(rows) or docs.shape[1]+1!=x.shape[1] or
        np.any(np.linalg.norm(docs,axis=1)<.999)):
        raise ValueError('Fixed semantic model/complete descriptions drift')
    validate_semantics(rows,registry,codes,len(docs))
    train,inner,inner_parents=positions(rows,held,registry)
    outer_blocked=set(registry[str(held)]);blocked=outer_blocked|inner_parents;candidates=[]
    recipes=[('TRAIN_FREQUENCY',None),('ZERO_SHOT',None),*[('SEMANTIC_RIDGE',r) for r in ridges]]
    for family,ridge in recipes:
        p=fit(x,docs,rows,codes,train,blocked,family,ridge)
        measured=controls(x,docs,p,family,rows,codes,inner,outer_blocked)
        parameters=len(docs) if family=='TRAIN_FREQUENCY' else 0 if family=='ZERO_SHOT' else int(p.size)
        candidates.append(dict(family=family,ridge=ridge,inner=summary(measured),parameters=parameters))
    selected=min(candidates,key=lambda c:(c['inner']['query_equal_pair_error'],c['inner']['head_error'],
        c['parameters'],-(c['ridge'] or 0.)))
    eligible=sorted(train+inner)
    p=fit(x,docs,rows,codes,eligible,outer_blocked,selected['family'],selected['ridge'])
    outer=[i for i,r in enumerate(rows) if r['fold']==held];outer_reports={};outer_rows=None
    # Only the selected recipe is refitted/deployable; fixed controls clarify evidence.
    for key,family,ridge in [('frequency','TRAIN_FREQUENCY',None),('zero_shot','ZERO_SHOT',None),
                             ('selected',selected['family'],selected['ridge'])]:
        parameter=p if (family,ridge)==(selected['family'],selected['ridge']) else fit(x,docs,rows,codes,eligible,outer_blocked,family,ridge)
        measured=controls(x,docs,parameter,family,rows,codes,outer)
        outer_reports[key]=summary(measured)
        if key=='selected':outer_rows=measured
    return p,dict(heldout_fold=held,selected=selected,candidates=candidates,outer=outer_reports,
        query_ids=[rows[i]['query_id'] for i in eligible],training_folds=sorted({rows[i]['fold'] for i in eligible}),
        training_queries=len(eligible),inner_train=len(train),inner_control=len(inner),outer_controls=outer_rows,
        outer_controls_used_in_selection=False,activity_definition_top5_is_not_siret_recall=True,
        identifiers_as_features=False,recall_measured=False,unknown_descriptions_preserved=True,
        all_official_descriptions_scored=True,probabilities_calibrated=False)
