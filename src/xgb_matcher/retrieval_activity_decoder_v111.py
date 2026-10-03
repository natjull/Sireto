"""TRAIN-only query-conditioned raw-activity signal, not a candidate reranker.

Class positions are categorical official codes, never numeric features. Only
fixed name vectors are used; family IDs serve exclusions and joins only.
No national search, top100 recall or deployment claim is made by this probe.
"""
import numpy as np

from src.xgb_matcher.retrieval_nested_v53 import positions


def query_design(vectors):
    x=np.asarray(vectors,dtype=np.float64)
    if x.ndim!=2 or not np.isfinite(x).all() or np.any(np.linalg.norm(x,axis=1)==0):
        raise ValueError('Invalid fixed name vectors')
    return np.column_stack((x/np.linalg.norm(x,axis=1,keepdims=True),np.ones(len(x))))


def validate(rows,registry,codes,classes):
    if set(registry)!={'2','3','4'} or len({r['query_id'] for r in rows})!=len(rows):
        raise ValueError('Training folds/query identities drift')
    for row in rows:
        if row['fold'] not in (2,3,4) or row['positive_siren'] not in registry[str(row['fold'])]:
            raise ValueError('Closed/protected fold in training')
        if not row['negatives']: raise ValueError('No natural negatives; do not drop row')
        for site,parent in [(row['positive_siret'],row['positive_siren']),
                            *[(n['siret'],n['siren']) for n in row['negatives']]]:
            if not (isinstance(site,str) and site.isascii() and site.isdigit() and len(site)==14 and
                    isinstance(parent,str) and parent.isascii() and parent.isdigit() and len(parent)==9 and
                    site.startswith(parent)):
                raise ValueError('Official join grain drift')
            if site not in codes or not isinstance(codes[site],(int,np.integer)) or not 0<=codes[site]<classes:
                raise ValueError('Missing/invalid official raw activity; do not remove row')


def fit(x,rows,codes,indices,classes,blocked,ridge):
    if len(indices)<16 or classes<2 or ridge is not None and ridge<=0:
        raise ValueError('Unsafe activity fit size/ridge')
    if any(rows[i]['positive_siren'] in blocked for i in indices):
        raise ValueError('Protected family in activity fit')
    target=np.zeros((len(indices),classes),dtype=np.float64)
    for j,i in enumerate(indices):
        code=codes[rows[i]['positive_siret']]
        # Unknown and empty cannot become shared-missing identity evidence.
        if code>=2: target[j,code]=1.
    if ridge is None:
        projection=np.zeros((x.shape[1],classes),dtype=np.float64)
        projection[-1]=target.mean(axis=0)
    else:
        train=x[indices]
        projection=np.linalg.solve(train.T@train+ridge*np.eye(x.shape[1]),train.T@target)
    projection=projection.astype(np.float32)
    if not np.isfinite(projection).all() or np.any(projection[:,:2]):
        raise ValueError('Activity coefficients finite/missing-evidence drift')
    return projection


def scores(x,projection):
    if (projection.ndim!=2 or projection.shape[0]!=x.shape[1] or not np.isfinite(projection).all() or
        projection.shape[1]<2 or np.any(projection[:,:2])):
        raise ValueError('Activity score shape or missing-evidence drift')
    # Bounded relevance, NOT a calibrated probability. All classes considered.
    return np.clip(x@projection,0.,1.)


def controls(x,projection,rows,codes,indices,blocked=()):
    predicted=scores(x[indices],projection); measured=[]
    for j,i in enumerate(indices):
        row=rows[i]; pc=codes[row['positive_siret']]
        safe=[n for n in row['negatives'] if n['siren'] not in blocked]
        if not safe: raise ValueError('No safe control negatives; never filter denominator')
        nc=np.array([codes[n['siret']] for n in safe],dtype=np.int64)
        pos=float(predicted[j,pc]); margin=pos-predicted[j,nc]
        rank=np.lexsort((np.arange(predicted.shape[1]),-predicted[j]))
        # Missing target stays in denominator, with no evidence/capture credited.
        measured.append(dict(query_id=row['query_id'],fold=row['fold'],pairs=len(safe),
            outrank=int(np.count_nonzero(margin<=0)),ties=int(np.count_nonzero(margin==0)),
            same_activity=int(np.count_nonzero(nc==pc)),unknown_positive=pc<2,
            top1=bool(pc>=2 and predicted[j,pc]>0 and pc==rank[0]),
            top5=bool(pc>=2 and predicted[j,pc]>0 and pc in rank[:5]),
            hardest_margin=float(margin.min())))
    return measured


def summary(measured):
    if not measured: raise ValueError('Empty controls')
    pairs=sum(r['pairs'] for r in measured)
    return dict(queries=len(measured),pairs=pairs,
        query_equal_pair_error=float(np.mean([r['outrank']/r['pairs'] for r in measured])),
        head_error=float(np.mean([r['outrank']>0 for r in measured])),
        mean_outrank_count=float(np.mean([r['outrank'] for r in measured])),
        raw_activity_top1=float(np.mean([r['top1'] for r in measured])),
        raw_activity_top5=float(np.mean([r['top5'] for r in measured])),
        same_activity_pairs=sum(r['same_activity'] for r in measured),
        tied_pairs=sum(r['ties'] for r in measured),unknown_positives=sum(r['unknown_positive'] for r in measured))


def select(vectors,rows,registry,codes,classes,held,ridges=(1.,10.,100.)):
    validate(rows,registry,codes,classes);x=query_design(vectors)
    if len(x)!=len(rows): raise ValueError('Query vector/row alignment drift')
    train,inner,inner_parents=positions(rows,held,registry)
    outer_blocked=set(registry[str(held)]);train_blocked=outer_blocked|inner_parents
    candidates=[]
    for ridge in (None,*ridges):
        projection=fit(x,rows,codes,train,classes,train_blocked,ridge)
        measured=controls(x,projection,rows,codes,inner,outer_blocked)
        candidates.append(dict(family='TRAIN_FREQUENCY' if ridge is None else 'NAME_ACTIVITY_RIDGE',
            ridge=ridge,inner=summary(measured),parameters=classes if ridge is None else int(projection.size)))
    selected=min(candidates,key=lambda c:(c['inner']['query_equal_pair_error'],c['inner']['head_error'],
        c['parameters'],-(c['ridge'] or 0.)))
    eligible=sorted(train+inner)
    projection=fit(x,rows,codes,eligible,classes,outer_blocked,selected['ridge'])
    frequency=fit(x,rows,codes,eligible,classes,outer_blocked,None)
    outer=[i for i,r in enumerate(rows) if r['fold']==held]
    before=controls(x,frequency,rows,codes,outer);after=controls(x,projection,rows,codes,outer)
    return projection,dict(heldout_fold=held,training_folds=sorted({rows[i]['fold'] for i in eligible}),
        query_ids=[rows[i]['query_id'] for i in eligible],inner_train=len(train),inner_control=len(inner),
        training_queries=len(eligible),selected=selected,candidates=candidates,
        outer_frequency=summary(before),outer_selected=summary(after),outer_controls=after,
        outer_controls_used_in_selection=False,negative_targets_used=False,identifiers_as_features=False,
        all_classes_evaluated=True,activity_top5_is_not_siret_recall=True,recall_measured=False,
        unknown_positive_rows_preserved=True,context_standalone_cannot_disambiguate_same_activity=True)
