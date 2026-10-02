"""Native initial proof query:11 learned components, one national top100."""
import math
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from src.xgb_matcher.retrieval_proof_v56 import LEXICAL,index_query,BASE as JOINT_BASE
BASE=JOINT_BASE[:len(LEXICAL)].copy()

def search(backend,q,weights):
    rows=[]
    for score,loc in backend.searcher.search(index_query(backend,q,weights),limit=100,count=False).hits:
        doc=backend.searcher.doc(loc).to_dict()
        s=backend._first(doc,'siret'); p=backend._first(doc,'siren')
        if not (len(s)==14 and s.isdigit() and len(p)==9 and p.isdigit() and s.startswith(p) and math.isfinite(score)):
            raise ValueError('Official initial identity/score drift')
        rows.append(dict(siret=s,siren=p,score=float(score)))
    if len(rows)>100 or len({r['siret'] for r in rows})!=len(rows): raise ValueError('Native cap drift')
    rows.sort(key=lambda r:(-r['score'],r['siret']))
    return rows,dict(certified=True,fixed_reservoir=None,unique_documents_visited=len(rows),
        native_exact_top100=True,metric='NATIVE_11_PROOF_LEARNED_NO_BM25_NO_GEOMETRY',
        score_cutoff=rows[-1]['score'] if rows else None,
        ties='CANONICAL_AMONG_RETURNED_NATIVE_SCORE_EQUALS',
        internal_postings_visits_not_instrumented=True)

def fit(scenes,held,registry,ridge=.005):
    if held not in (2,3,4) or set(registry)!={'2','3','4'} or not registry[str(held)]: raise ValueError('Protected fold drift')
    protected=set(registry[str(held)]); diffs=[]; sw=[]; kept=[]
    for r in scenes:
        if r['fold'] not in (2,3,4) or r['positive_siren'] not in registry[str(r['fold'])]: raise ValueError('OOF membership drift')
        if r['fold']==held or r['positive_siren'] in protected: continue
        neg=[n for n in r['negatives'] if n['siren'] not in protected]
        if not neg: continue
        diffs.extend(np.array(r['positive_features'])-np.array([n['features'] for n in neg])); sw.extend([1./len(neg)]*len(neg)); kept.append(r)
    if len(kept)<16: raise ValueError('Insufficient safe proof scenes')
    x=np.array(diffs); sw=np.array(sw)/len(kept)
    if x.shape[1]!=len(LEXICAL) or not np.isfinite(x).all() or ridge<=0: raise ValueError('Feature shape/finite drift')
    def objective(w):
        margin=x@w; delta=w-BASE
        return float(sw@np.logaddexp(0.,-margin)+ridge*np.sum(delta*delta)),x.T@(-sw*expit(-margin))+2*ridge*delta
    result=minimize(objective,BASE,method='L-BFGS-B',jac=True,bounds=[(0.,64.)]*len(LEXICAL),options=dict(maxiter=500,ftol=1e-10,gtol=1e-7))
    if not result.success or not np.isfinite(result.x).all(): raise ValueError('Proof fit failed')
    return dict(weights=result.x.tolist(),features=list(LEXICAL),heldout_fold=held,training_folds=sorted({r['fold'] for r in kept}),
        query_ids=[r['query_id'] for r in kept],training_queries=len(kept),training_pairs=len(x),heldout_entity_overlap=0,
        identifiers_as_features=False,ridge=ridge,objective_before=objective(BASE)[0],objective_after=float(result.fun))

