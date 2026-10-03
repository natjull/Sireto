"""Initial-score algebra and nested cross-fit probe, not a candidate reranker.

All components are nonnegative official proofs. Activity UNKNOWN is zero
evidence only; native text/address/geometry remain available. TRAIN activity
features exclude both outer/inner protected families AND the evaluated fold.
"""
import numpy as np
from scipy.optimize import minimize

from src.xgb_matcher.retrieval_activity_decoder_v111 import query_design
from src.xgb_matcher.retrieval_activity_semantic_v113 import normalized,validate_semantics,fit as decoder_fit,scores
from src.xgb_matcher.retrieval_nested_v53 import positions
from src.xgb_matcher.retrieval_proof_v56 import FEATURES as NATIVE,BASE
from src.xgb_matcher.retrieval_joint_contrastive_v63 import head_loss

INTERACTIONS=(4,5,7,*range(11,19))
FEATURES=(*NATIVE,*(f'activity_x_{NATIVE[i]}' for i in INTERACTIONS))


def vector(native,activity):
    f=np.asarray(native,dtype=np.float64)
    if (f.shape!=(19,) or not np.isfinite(f).all() or np.any(f<0) or np.any(f>2.000008) or
        not isinstance(activity,(float,int,np.floating,np.integer)) or not np.isfinite(activity) or not 0<=activity<=1):
        raise ValueError('Official native/activity proof range drift')
    if np.any(f[11:]>1.000008) or np.any(f[15:]>f[11:15]+1e-8):
        raise ValueError('BAN geometric/reliable proof drift')
    if np.any(np.delete(f,8)>1.000008):raise ValueError('Individual native proof range drift')
    return np.r_[f,float(activity)*f[list(INTERACTIONS)]]


def linear_weights(weights,activity):
    """Exact native coefficients for one activity grade, also safe upper bounds.

    Nonnegative weights and 0..1 grades make component-wise native maxima
    and an activity maximum an upper bound, not a source/candidate filter.
    """
    w=np.asarray(weights,dtype=np.float64)
    if (w.shape!=(30,) or not np.isfinite(w).all() or np.any(w<0) or np.any(w>64) or
        not np.isscalar(activity) or not np.isfinite(activity) or not 0<=activity<=1):
        raise ValueError('Initial monotone coefficient drift')
    effective=w[:19].copy();effective[list(INTERACTIONS)]+=activity*w[19:]
    return effective


def aligned(rows,scenes,registry,codes,classes):
    validate_semantics(rows,registry,codes,classes)
    if len(rows)!=len(scenes):raise ValueError('All TRAIN scenes required')
    for row,scene in zip(rows,scenes):
        if (any(row[k]!=scene[k] for k in ('query_id','fold','positive_siren')) or
            len(row['negatives'])!=len(scene['negatives'])):
            raise ValueError('Natural teacher alignment drift')
        vector(scene['positive_features'],0.)
        for n,proof in zip(row['negatives'],scene['negatives']):
            if n['siren']!=proof['siren']:raise ValueError('Natural negative order drift')
            vector(proof['features'],0.)


def signals(x,docs,rows,codes,indices,parameter,decoder):
    all_scores=scores(x[indices],docs,parameter,decoder['family']);result={}
    for j,i in enumerate(indices):
        ids=[rows[i]['positive_siret'],*[n['siret'] for n in rows[i]['negatives']]]
        result[i]=np.array([0. if codes[s]<0 else all_scores[j,codes[s]] for s in ids],dtype=np.float64)
    return result


def crossfit(x,docs,rows,registry,codes,indices,blocked,decoder):
    result={};audits=[];forbidden=set(blocked)
    if len(set(indices))!=len(indices) or any(rows[i]['positive_siren'] in forbidden for i in indices):
        raise ValueError('Cross-fit eligible family/duplicates drift')
    for fold in sorted({rows[i]['fold'] for i in indices}):
        controlled=[i for i in indices if rows[i]['fold']==fold]
        protected=forbidden|set(registry[str(fold)])
        train=[i for i in indices if rows[i]['fold']!=fold and rows[i]['positive_siren'] not in protected]
        p=decoder_fit(x,docs,rows,codes,train,protected,decoder['family'],decoder['ridge'])
        if set(train)&set(controlled) or any(rows[i]['positive_siren'] in protected for i in train):
            raise ValueError('Cross-fit decoder evaluated its own family')
        result.update(signals(x,docs,rows,codes,controlled,p,decoder))
        audits.append(dict(controlled_fold=fold,training_folds=sorted({rows[i]['fold'] for i in train}),
            training_query_ids=[rows[i]['query_id'] for i in train],controlled_query_ids=[rows[i]['query_id'] for i in controlled],
            protected_family_overlap=0,self_training_overlap=0))
    if set(result)!=set(indices):raise ValueError('Cross-fit denominator drift')
    return result,audits


def design(rows,scenes,indices,strengths,blocked,codes):
    matrices=[];sizes=[];tags=[];forbidden=set(blocked)
    for i in indices:
        row,scene=rows[i],scenes[i];a=strengths[i]
        if a.shape!=(1+len(row['negatives']),) or not np.isfinite(a).all():
            raise ValueError('Whole activity vector alignment drift')
        safe=[j for j,n in enumerate(row['negatives']) if n['siren'] not in forbidden]
        if not safe:raise ValueError('No safe natural negative; no query filtering')
        positive=vector(scene['positive_features'],a[0])
        matrices.append(positive-np.array([vector(scene['negatives'][j]['features'],a[j+1]) for j in safe]))
        sizes.append(len(safe))
        tags.extend(dict(same_parent=row['negatives'][j]['siren']==row['positive_siren'],
            no_name_proof=scene['positive_features'][0]==0.,
            no_geometry=not np.any(np.asarray(scene['positive_features'])[11:15]),
            unknown_activity=codes[row['positive_siret']]<0,
            zero_activity_evidence=a[0]==0.) for j in safe)
    return np.vstack(matrices),np.array(sizes,dtype=np.int64),tags


def report(matrix,sizes,weights,tags):
    margin=matrix@weights;starts=np.r_[0,np.cumsum(sizes)[:-1]]
    hardest=np.minimum.reduceat(margin,starts);outrank=np.add.reduceat((margin<=0).astype(int),starts)
    result=dict(queries=len(sizes),pairs=len(margin),head_error=float(np.mean(hardest<=0)),
        mean_outrank_count=float(np.mean(outrank)),hardest_margin=float(np.mean(hardest)),
        mrr=float(np.mean(1./(1.+outrank))),query_equal_pair_error=float(np.mean(np.add.reduceat((margin<=0).astype(int),starts)/sizes)))
    for key in ('same_parent','no_name_proof','no_geometry','unknown_activity','zero_activity_evidence'):
        mask=np.array([r[key] for r in tags],dtype=bool)
        result[key]=dict(pairs=int(mask.sum()),pair_error=float(np.mean(margin[mask]<=0)) if np.any(mask) else None)
    return result


def fit_head(matrix,sizes,family,extra_penalty=.05):
    count={'NATIVE11':11,'GEOMETRY19':19,'ACTIVITY30':30}.get(family)
    if (count is None or matrix.ndim!=2 or matrix.shape[1]!=30 or not np.isfinite(matrix).all() or
        sizes.ndim!=1 or sizes.dtype.kind not in 'iu' or not len(sizes) or np.any(sizes<=0) or
        sizes.sum()!=len(matrix) or not np.isfinite(extra_penalty) or extra_penalty<=0):
        raise ValueError('Initial score head family/design drift')
    prior=np.r_[BASE,np.zeros(11)][:count]
    penalty=np.full(count,.005);penalty[19:]=extra_penalty
    x=matrix[:,:count]
    result=minimize(head_loss,prior,args=(x,sizes,prior,penalty),jac=True,method='L-BFGS-B',
        bounds=[(0.,64.)]*count,options=dict(maxiter=200,ftol=1e-8,gtol=1e-5))
    if (not np.isfinite(result.x).all() or np.any(result.x<0) or np.any(result.x>64) or
        result.fun>head_loss(prior,x,sizes,prior,penalty)[0]+1e-8):
        raise ValueError('Nonmonotone/nonfinite/worse initial head')
    weights=np.zeros(30);weights[:count]=result.x
    return weights,dict(converged=bool(result.success),message=str(result.message),parameters=count,
        objective_before=head_loss(prior,x,sizes,prior,penalty)[0],objective_after=float(result.fun),iterations=int(result.nit))


def select(vectors,descriptions,rows,scenes,registry,codes,held,decoder,extra_penalty=.05):
    x=query_design(vectors);docs=normalized(descriptions)
    if len(x)!=len(rows) or docs.shape[1]+1!=x.shape[1] or np.any(np.linalg.norm(docs,axis=1)<.999):
        raise ValueError('Fixed name/official semantic dimension drift')
    aligned(rows,scenes,registry,codes,len(docs))
    train,inner,inner_parents=positions(rows,held,registry)
    outer_blocked=set(registry[str(held)]);blocked=outer_blocked|inner_parents
    train_activity,inner_crossfit=crossfit(x,docs,rows,registry,codes,train,blocked,decoder)
    inner_projection=decoder_fit(x,docs,rows,codes,train,blocked,decoder['family'],decoder['ridge'])
    inner_activity=signals(x,docs,rows,codes,inner,inner_projection,decoder)
    tx,ts,_=design(rows,scenes,train,train_activity,blocked,codes)
    cx,cs,ct=design(rows,scenes,inner,inner_activity,outer_blocked,codes);candidates=[]
    for family in ('NATIVE11','GEOMETRY19','ACTIVITY30'):
        w,fitted=fit_head(tx,ts,family,extra_penalty)
        candidates.append(dict(family=family,fit=fitted,inner=report(cx,cs,w,ct),parameters=fitted['parameters']))
    eligible_candidates=[c for c in candidates if c['fit']['converged']]
    if not eligible_candidates:raise ValueError('No converged initial score head')
    selected=min(eligible_candidates,key=lambda c:(c['inner']['head_error'],c['inner']['mean_outrank_count'],c['parameters']))
    eligible=sorted(train+inner)
    final_activity,final_crossfit=crossfit(x,docs,rows,registry,codes,eligible,outer_blocked,decoder)
    fx,fs,_=design(rows,scenes,eligible,final_activity,outer_blocked,codes)
    projection=decoder_fit(x,docs,rows,codes,eligible,outer_blocked,decoder['family'],decoder['ridge'])
    weights,final_fit=fit_head(fx,fs,selected['family'],extra_penalty)
    if not final_fit['converged']:raise ValueError('Selected final head failed convergence')
    outer=[i for i,r in enumerate(rows) if r['fold']==held]
    outer_activity=signals(x,docs,rows,codes,outer,projection,decoder)
    ox,os,ot=design(rows,scenes,outer,outer_activity,(),codes);controls={}
    # Refitting all frozen control families is diagnostic only, never reselection.
    for family in ('NATIVE11','GEOMETRY19','ACTIVITY30'):
        w,fitted=(weights,final_fit) if family==selected['family'] else fit_head(fx,fs,family,extra_penalty)
        if not fitted['converged']:raise ValueError('Diagnostic final head failed convergence')
        controls[family]=report(ox,os,w,ot)
    model=dict(heldout_fold=held,training_folds=sorted({rows[i]['fold'] for i in eligible}),
        query_ids=[rows[i]['query_id'] for i in eligible],family=selected['family'],weights=weights.tolist(),
        features=list(FEATURES),decoder=decoder,identifiers_as_features=False,serving_integrated=False,
        raw_activity_without_site_proof=False,downstream_models_changed=False)
    return projection,model,dict(heldout_fold=held,selected=selected,candidates=candidates,outer=controls,
        final_fit=final_fit,inner_crossfit=inner_crossfit,final_crossfit=final_crossfit,
        inner_train=len(train),inner_control=len(inner),training_queries=len(eligible),
        outer_controls_used_in_selection=False,activity_train_features_cross_fitted=True,
        source_documents_missing_filtered=False,recall_measured=False)
