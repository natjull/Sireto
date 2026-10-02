"""Query-side semantic translation into an official word vocabulary.

No CRM alias index, identifier features, target injection or document
re-embedding. Serving compiles <=8 weighted terms into a native component.
This is a small experimental ridge decoder, not a SPLADE reproduction.
"""
from collections import Counter
import numpy as np
from scripts import retrieval_v38_initial_top100 as v38
from src.xgb_matcher.retrieval_proof_v56 import signal
from src.xgb_matcher.retrieval_native_v57 import fit as native_fit
from src.xgb_matcher.retrieval_nested_v53 import positions

STOP=frozenset('A AU AUX AVEC CE CES CET CETTE D DE DES DU EN ET L LA LE LES OU POUR SUR UN UNE'.lower().split())

def words(text):
    return frozenset(w for w in v38.tokens(signal(text)) if w.isalpha() and len(w)>1 and w not in STOP)

def vocabulary(documents,cap=2048,min_df=3,max_df_fraction=.15):
    counts=Counter(w for fields in documents.values() for w in set().union(*[words(v) for values in fields.values() for v in values]))
    return sorted([w for w,n in sorted(counts.items(),key=lambda kv:(-kv[1],kv[0]))
        if min_df<=n<=max_df_fraction*len(documents)][:cap])

def targets(rows,documents,vocab):
    lookup={t:i for i,t in enumerate(vocab)}; y=np.zeros((len(rows),len(vocab)))
    for i,r in enumerate(rows):
        terms=set().union(*[words(v) for values in documents[r['positive_siret']].values() for v in values])
        indices=[lookup[t] for t in terms if t in lookup]
        if indices: y[i,indices]=1./np.sqrt(len(indices))
    return y

def unit(q):
    q=np.asarray(q,dtype=np.float64)
    if q.ndim!=2 or not np.isfinite(q).all() or np.any(np.linalg.norm(q,axis=1)==0): raise ValueError('Query semantic shape drift')
    return q/np.linalg.norm(q,axis=1,keepdims=True)

def solve(q,y,indices,ridge):
    if ridge<=0 or len(indices)<16: raise ValueError('Unsafe sparse-query fit')
    x=unit(q)[indices]; target=y[indices]
    return np.linalg.solve(x.T@x+ridge*np.eye(x.shape[1]),x.T@target).astype(np.float32)

def expand(q,projection,vocab,query_name,count=8):
    if not signal(query_name): return []
    values=unit(np.asarray(q).reshape(1,-1))[0]@projection; excluded=words(query_name)
    if values.shape!=(len(vocab),) or not np.isfinite(values).all(): raise ValueError('Sparse decoder drift')
    ranked=sorted(((-float(v),t) for t,v in zip(vocab,values) if v>0 and t not in excluded))[:count]
    total=sum(-v for v,_ in ranked)
    return [(t,-v/total) for v,t in ranked] if total>0 else []

def semantic_score(fields,terms,place):
    if not place: return 0.
    return max((sum(weight for word,weight in terms if word in (values if isinstance(values,frozenset) else set().union(*[words(v) for v in values])))
        for values in fields.values()),default=0.)

def field_bags(fields):
    return {field:frozenset().union(*[words(v) for v in values]) for field,values in fields.items()}

def compiled_component(backend,q,terms):
    if not terms: return None
    tv=backend._tantivy
    places=[tv.Query.const_score_query(tv.Query.term_query(backend.index.schema,f,v,'basic'),0.)
        for f,v in (('insee',q.insee),('postcode',q.postcode)) if v]
    if not places: return None
    views=[tv.Query.boolean_query([(tv.Occur.Should,tv.Query.const_score_query(
        tv.Query.term_query(backend.index.schema,field,t,'basic'),float(w))) for t,w in terms]) for field in v38.NAMES]
    return tv.Query.boolean_query([(tv.Occur.Must,tv.Query.disjunction_max_query(views,tie_breaker=0.)),
        (tv.Occur.Must,tv.Query.disjunction_max_query(places,tie_breaker=0.))])

def controls(q,projection,vocab,rows,scenes,documents,weights,alpha,indices,blocked=()):
    protected=set(blocked); measured=[]
    for i in indices:
        r=rows[i]; s=scenes[i]; terms=expand(q[i],projection,vocab,r['crm_name'])
        positive=float(np.asarray(s['positive_features'][:11])@weights)+alpha*semantic_score(
            documents[r['positive_siret']],terms,s['positive_features'][4])
        negatives=[]
        for n,p in zip(r['negatives'],s['negatives']):
            if n['siren']!=p['siren']: raise ValueError('Natural negative order drift')
            if n['siren'] in protected: continue
            negatives.append(float(np.asarray(p['features'][:11])@weights)+alpha*semantic_score(documents[n['siret']],terms,p['features'][4]))
        if not negatives: raise ValueError('No safe control negatives; do not remove query')
        count=sum(v>=positive for v in negatives)
        measured.append(dict(query_id=r['query_id'],fold=r['fold'],negative_pairs=len(negatives),negative_outrank_count=count,
            head_error=bool(count),hardest_margin=positive-max(negatives),reciprocal_rank=1./(1.+count),expanded_terms=terms))
    return measured

def summary(rows):
    return dict(queries=len(rows),head_error=float(np.mean([r['head_error'] for r in rows])),
        mean_outrank_count=float(np.mean([r['negative_outrank_count'] for r in rows])),
        hardest_margin=float(np.mean([r['hardest_margin'] for r in rows])),
        reciprocal_rank=float(np.mean([r['reciprocal_rank'] for r in rows])))

def select(q,y,vocab,rows,scenes,documents,registry,held,ridges,alphas,ridge_native=.005):
    membership=[dict(query_id=r['query_id'],fold=r['fold'],positive_siren=r['positive_siren']) for r in rows]
    train,validation,inner=positions(membership,held,registry)
    protected={k:list(v) for k,v in registry.items()}; protected[str(held)]=sorted(set(protected[str(held)])|inner)
    native_scenes=[dict(s,positive_features=s['positive_features'][:11],
        negatives=[dict(n,features=n['features'][:11]) for n in s['negatives']]) for s in scenes]
    baseline=native_fit(native_scenes,held,protected,ridge_native)
    if set(baseline['query_ids'])!={rows[i]['query_id'] for i in train}: raise ValueError('Inner native baseline coverage drift')
    weights=np.array(baseline['weights']); candidates=[]; zero=np.zeros((q.shape[1],len(vocab)),dtype=np.float32)
    before=summary(controls(q,zero,vocab,rows,scenes,documents,weights,0.,validation,registry[str(held)]))
    candidates.append(dict(ridge=None,alpha=0.,inner=before))
    for ridge in ridges:
        projection=solve(q,y,train,ridge)
        for alpha in alphas:
            measured=controls(q,projection,vocab,rows,scenes,documents,weights,alpha,validation,registry[str(held)])
            candidates.append(dict(ridge=ridge,alpha=alpha,inner=summary(measured)))
    selected=min(candidates,key=lambda c:(c['inner']['head_error'],c['inner']['mean_outrank_count'],c['alpha'],-(c['ridge'] or 0.)))
    final_indices=train+validation
    projection=zero if selected['alpha']==0 else solve(q,y,final_indices,selected['ridge'])
    final=native_fit(native_scenes,held,registry,ridge_native); weights=np.array(final['weights'])
    outer=[i for i,r in enumerate(rows) if r['fold']==held]
    base=controls(q,zero,vocab,rows,scenes,documents,weights,0.,outer)
    measured=controls(q,projection,vocab,rows,scenes,documents,weights,selected['alpha'],outer)
    return projection,dict(heldout_fold=held,selected=selected,candidates=candidates,
        inner_training_queries=len(train),inner_validation_queries=len(validation),inner_entities=sorted(inner),
        query_ids=[rows[i]['query_id'] for i in final_indices],native=final,alpha=selected['alpha'],
        outer_before=summary(base),outer_after=summary(measured),outer_controls=measured,
        outer_controls_used_in_selection=False,identifiers_as_features=False,recall_measured=False)
