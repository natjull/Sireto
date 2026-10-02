"""Official-current-alias initialisms, compiled to the existing national FST.

No CRM alias dictionary, learned identifier feature or candidate reservoir.
Each regex matches ONE complete normalized official alias, not words pooled
across different alias/source observations.
"""
import re
import numpy as np
from scipy.optimize import minimize

from src.xgb_matcher.retrieval_proof_v56 import signal, components, LEXICAL
from src.xgb_matcher.hierarchical_retrieval import normalize_text
from src.xgb_matcher.retrieval_native_v57 import BASE, fit as baseline_fit
from src.xgb_matcher.retrieval_nested_v53 import positions
from src.xgb_matcher.retrieval_joint_contrastive_v63 import head_loss
from src.xgb_matcher.retrieval_sparse_query_v59 import solve, expand, semantic_score, compiled_component, vocabulary, targets, field_bags

EXACT_FIELDS = ('legal_names_exact','trade_names_exact','site_names_exact')
PARTICLES = ('A','AU','AUX','D','DE','DES','DU','ET','L','LA','LE','LES')
LEGAL = ('SA','SAS','SASU','SARL','EURL','SCI','SNC','SELARL','SOCIETE')
STOP = frozenset((*PARTICLES,*LEGAL,'EN','OU','SUR','POUR','AVEC','MAIRIE','ECOLE','VILLE','HOTEL','LYCEE','GROUPE'))
EXTRAS = ('official_initialism','initialism_at_place','initialism_number_at_place',
          'initialism_address_at_place','learned_official_words_at_place')
FEATURES = (*LEXICAL,*EXTRAS)


def patterns(name):
    candidates=[]
    for token in re.findall(r'\b([A-Z]{2,6})[0-9]*\b', signal(name)):
        if token not in STOP and token not in candidates: candidates.append(token)
    prefix='('+'|'.join(v+' ' for v in (*PARTICLES,*LEGAL))+')*'
    separator=' ('+'|'.join(v+' ' for v in PARTICLES)+')*'
    return [prefix+separator.join(c+'[A-Z0-9]*' for c in token)+'( [A-Z0-9]+)*' for token in candidates[:2]]


def membership(name,aliases):
    return float(any(re.fullmatch(p,normalize_text(a)) is not None for p in patterns(name) for a in aliases))


def acronym_components(backend,q):
    tv=backend._tantivy; regexes=[tv.Query.const_score_query(tv.Query.regex_query(backend.index.schema,f,p),1.)
        for p in patterns(q.name) for f in EXACT_FIELDS]
    if not regexes: return [None]*4
    alias=tv.Query.disjunction_max_query(regexes,tie_breaker=0.)
    base=components(backend,q)
    def conjunction(*parts):
        return tv.Query.boolean_query([(tv.Occur.Must,p) for p in parts]) if all(p is not None for p in parts) else None
    gate=tv.Query.const_score_query(base[4],0.) if base[4] is not None else None
    number=conjunction(alias,base[5])
    return [alias,conjunction(alias,gate),tv.Query.const_score_query(number,1.) if number is not None else None,
            conjunction(tv.Query.const_score_query(alias,0.),base[7])]


def query(backend,q,weights,terms):
    weights=np.asarray(weights,dtype=np.float64)
    if weights.shape!=(len(FEATURES),) or not np.isfinite(weights).all() or np.any(weights<0):
        raise ValueError('Initialism native metric drift')
    parts=[*components(backend,q),*acronym_components(backend,q),compiled_component(backend,q,terms)]
    tv=backend._tantivy
    scored=[tv.Query.boost_query(p,float(w)) for p,w in zip(parts,weights) if p is not None and w>0]
    if not scored: return tv.Query.empty_query()
    guard=tv.Query.const_score_query(tv.Query.term_query(backend.index.schema,'document_type','siret','basic'),0.)
    return tv.Query.boolean_query([(tv.Occur.Must,guard),(tv.Occur.Must,tv.Query.boolean_query([(tv.Occur.Should,p) for p in scored]))])


def feature_vector(native,acro,word):
    f=np.asarray(native)[:len(LEXICAL)]
    return np.r_[f,acro,acro*f[4],acro*f[5],acro*f[7],word]


def choose_columns(family):
    return {'HEAD_NATIVE':list(range(11)), 'HEAD_WORD':[*range(11),15],
            'HEAD_ACRONYM':list(range(15)), 'HEAD_FULL':list(range(16))}[family]


def design(q,projection,vocab,rows,scenes,documents,aliases,indices,blocked=()):
    forbid=set(blocked); values=[]; sizes=[]; selected_ids=[]
    for i in indices:
        r=rows[i]; s=scenes[i]; terms=expand(q[i],projection,vocab,r['crm_name'])
        def fv(siret,native):
            a=membership(r['crm_name'],aliases[siret])
            return feature_vector(native,a,semantic_score(documents[siret],terms,native[4]))
        positive=fv(r['positive_siret'],s['positive_features']); negatives=[]
        for n,p in zip(r['negatives'],s['negatives']):
            if n['siren']!=p['siren']: raise ValueError('Negative proof ordering drift')
            if n['siren'] not in forbid: negatives.append(fv(n['siret'],p['features']))
        if not negatives: raise ValueError('No safe negative; do not drop query')
        values.append(positive-np.array(negatives)); sizes.append(len(negatives)); selected_ids.append(r['query_id'])
    return np.vstack(values),np.array(sizes,dtype=int),selected_ids


def summary(x,sizes,w):
    margin=x@w; starts=np.r_[0,np.cumsum(sizes)[:-1]]
    hardest=np.minimum.reduceat(margin,starts); outrank=np.add.reduceat((margin<=0).astype(int),starts)
    return dict(queries=len(sizes),pairs=len(margin),head_error=float(np.mean(hardest<=0)),
        mean_outrank_count=float(np.mean(outrank)),hardest_margin=float(np.mean(hardest)),mrr=float(np.mean(1./(1.+outrank))))


def fit(x,sizes,family):
    cols=choose_columns(family); prior=np.r_[BASE,np.zeros(5)][cols]; penalty=np.full(len(cols),.005)
    result=minimize(head_loss,prior,args=(x[:,cols],sizes,prior,penalty),jac=True,method='L-BFGS-B',
        bounds=[(0.,64.)]*len(cols),options=dict(maxiter=180,ftol=1e-8,gtol=1e-5))
    if not np.isfinite(result.x).all() or result.fun>head_loss(prior,x[:,cols],sizes,prior,penalty)[0]+1e-8:
        raise ValueError('Initialism head fit drift')
    w=np.zeros(len(FEATURES)); w[cols]=result.x
    return w,dict(converged=bool(result.success),message=str(result.message),iterations=int(result.nit),
        parameters=len(cols),objective_before=head_loss(prior,x[:,cols],sizes,prior,penalty)[0],objective_after=float(result.fun))


def word_recipe(q,rows,raw_documents,indices,blocked):
    forbidden=set(blocked); ids=set()
    for i in indices:
        r=rows[i]
        if r['positive_siren'] in forbidden: raise ValueError('Protected positive in word vocabulary')
        ids.add(r['positive_siret'])
        ids.update(n['siret'] for n in r['negatives'] if n['siren'] not in forbidden)
    vocab=vocabulary({s:raw_documents[s] for s in sorted(ids)},2048,3,.15)
    y=targets(rows,raw_documents,vocab)
    return vocab,solve(q,y,indices,10.)


def select(q,rows,scenes,raw_documents,aliases,registry,held):
    train,control,inner=positions(rows,held,registry); outer_blocked=set(registry[str(held)])
    blocked=outer_blocked|inner; expanded={k:list(v) for k,v in registry.items()}; expanded[str(held)]=sorted(blocked)
    native=[dict(s,positive_features=s['positive_features'][:11],
        negatives=[dict(n,features=n['features'][:11]) for n in s['negatives']]) for s in scenes]
    baseline=baseline_fit(native,held,expanded,.005)
    if set(baseline['query_ids'])!={rows[i]['query_id'] for i in train}: raise ValueError('Inner baseline exclusion drift')
    documents={s:field_bags(d) for s,d in raw_documents.items()}
    vocab,projection=word_recipe(q,rows,raw_documents,train,blocked)
    tx,ts,_=design(q,projection,vocab,rows,scenes,documents,aliases,train,blocked)
    cx,cs,_=design(q,projection,vocab,rows,scenes,documents,aliases,control,outer_blocked)
    base=np.r_[baseline['weights'],np.zeros(5)]
    candidates=[dict(family='BASELINE11',parameters=11,converged=True,inner=summary(cx,cs,base))]
    for family in ('HEAD_NATIVE','HEAD_WORD','HEAD_ACRONYM','HEAD_FULL'):
        w,report=fit(tx,ts,family)
        candidates.append(dict(family=family,parameters=report['parameters'],converged=report['converged'],
            fit=report,inner=summary(cx,cs,w)))
    selected=min((c for c in candidates if c['converged']),key=lambda c:(c['inner']['head_error'],
        c['inner']['mean_outrank_count'],c['parameters']))
    eligible=sorted(train+control)
    final_vocab,final_projection=word_recipe(q,rows,raw_documents,eligible,outer_blocked)
    bf=baseline_fit(native,held,registry,.005)
    if set(bf['query_ids'])!={rows[i]['query_id'] for i in eligible}: raise ValueError('Outer baseline exclusion drift')
    before=np.r_[bf['weights'],np.zeros(5)]
    if selected['family']=='BASELINE11': w=before; fitted=bf
    else:
        fx,fs,_=design(q,final_projection,final_vocab,rows,scenes,documents,aliases,eligible,outer_blocked)
        w,fitted=fit(fx,fs,selected['family'])
        if not fitted['converged']: raise ValueError('Selected refit not converged')
    outer=[i for i,r in enumerate(rows) if r['fold']==held]
    ox,os,_=design(q,final_projection,final_vocab,rows,scenes,documents,aliases,outer)
    return final_projection,final_vocab,dict(weights=w.tolist(),features=list(FEATURES),family=selected['family'],heldout_fold=held,
        training_folds=sorted({rows[i]['fold'] for i in eligible}),query_ids=[rows[i]['query_id'] for i in eligible],
        training_queries=len(eligible),decoder_ridge=10.,identifiers_as_features=False),dict(
        heldout_fold=held,selected=selected,candidates=candidates,inner_entities=sorted(inner),
        inner_training_queries=len(train),inner_validation_queries=len(control),final_fit=fitted,
        inner_vocabulary_words=len(vocab),final_vocabulary_words=len(final_vocab),
        outer_before=summary(ox,os,before),outer_after=summary(ox,os,w),outer_controls_used_in_selection=False,recall_measured=False)

