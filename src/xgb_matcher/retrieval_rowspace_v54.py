"""Exact full contrastive objective restricted to its nontrivial row space.

For loss L(QW) + lambda||W-I||², the optimum W-I lies in row(Q).
An orthonormal U spanning row(Q) permits W=I+UA, with ||UA||=||A||.
No model-capacity change: only fewer solver variables. Serving remains one
full query transform and native document dot product. Finite optimizer
tolerance need not produce bit-identical parameters to the older solve.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from src.xgb_matcher.retrieval_learned_v41 import unit_vectors
from src.xgb_matcher.retrieval_nested_v53 import limited_fit


def solve(q,p,negatives,memberships,held,registry,capacity,ridge,temperature,hard_count,maxiter):
    if capacity!='FULL': return limited_fit(q,p,negatives,memberships,held,registry,capacity,ridge,temperature,hard_count,maxiter)
    q,p=unit_vectors(q),unit_vectors(p); protected=set(registry[str(held)]); kept=[]; differences=[]; counts=[]; safe_pairs=0
    for i,r in enumerate(memberships):
        if r['fold']==held or r['positive_siren'] in protected: continue
        safe=[j for j,s in enumerate(r['negative_sirens']) if s not in protected]
        if not safe: continue
        n=unit_vectors(negatives[i])[safe]; safe_pairs+=len(n)
        take=np.argsort(-(n@q[i]),kind='stable')[:hard_count]
        differences.extend(p[i]-n[take]); counts.append(len(take)); kept.append(i)
    if len(kept)<16 or not (ridge>0 and temperature>0): raise ValueError('Invalid safe rowspace fit')
    q=q[kept].astype(np.float64); d=np.array(differences,dtype=np.float64); dimension=q.shape[1]
    u=np.linalg.qr(q.T,mode='reduced')[0]; reduced=q@u
    owner=np.repeat(np.arange(len(q)),counts); sw=np.concatenate([np.full(n,1./n/len(q)) for n in counts])
    def objective(flat):
        a=flat.reshape(u.shape[1],dimension); transformed=q+reduced@a
        margin=temperature*np.einsum('ij,ij->i',transformed[owner],d)
        residual=-temperature*sw*expit(-margin); summed=np.zeros_like(q)
        np.add.at(summed,owner,residual[:,None]*d)
        return float(sw@np.logaddexp(0.,-margin)+ridge*np.sum(a*a)/dimension),(
            reduced.T@summed+2*ridge*a/dimension).ravel()
    base=np.zeros(u.shape[1]*dimension); before=objective(base)[0]
    fitted=minimize(objective,base,method='L-BFGS-B',jac=True,
        options=dict(maxiter=maxiter,maxcor=5,ftol=1e-8,gtol=1e-6))
    w=(np.eye(dimension)+u@fitted.x.reshape(u.shape[1],dimension)).astype(np.float32)
    if not np.isfinite(w).all() or not np.isfinite(fitted.fun) or fitted.fun>before+1e-8: raise ValueError('Rowspace fit failed')
    return w,dict(capacity='FULL',parameters=dimension**2,solver_parameters=len(base),rowspace_dimension=u.shape[1],
        heldout_fold=held,training_queries=len(kept),training_pairs=len(d),safe_source_negative_pairs=safe_pairs,
        query_ids=[memberships[i]['query_id'] for i in kept],training_folds=sorted({memberships[i]['fold'] for i in kept}),
        heldout_entity_overlap=0,ridge=ridge,objective_before=before,objective_after=float(fitted.fun),
        optimizer_converged=bool(fitted.success),iterations=int(fitted.nit),same_full_objective=True,
        identifiers_as_features=False,serving_score='NORMALIZED_QUERY_W_DOT_FIXED_DOCUMENT')
