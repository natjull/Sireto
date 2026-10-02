"""Contrastive query adapter compiled to a single indexed dot product.

Document vectors stay fixed. No serving candidate pair features/reservoir.
IDs only select safe OOF training groups, never enter the representation.
"""
import re

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from src.xgb_matcher.retrieval_learned_v41 import unit_vectors


def surface(values,prefix):
    if isinstance(values,str): raise ValueError('Expected separate official views')
    text=' | '.join(dict.fromkeys(' '.join(v.split()) for v in values if v.strip()))
    matches=list(re.finditer(r'(?<!\d)(?:\d{9}|\d{14})(?!\d)',text))
    text=re.sub(r'(?<!\d)(?:\d{9}|\d{14})(?!\d)',' ',text)
    return prefix+': '+' '.join(text.split()),len(matches)


def combine(names,addresses):
    a,b=unit_vectors(names),unit_vectors(addresses)
    if a.shape!=b.shape: raise ValueError('Field alignment drift')
    return np.concatenate([a,b],axis=1)/np.sqrt(2.)


def fit(queries,positives,negatives,memberships,held,registry,ridge=.01,temperature=10.,hard_count=8,maxiter=100):
    q,p=unit_vectors(queries),unit_vectors(positives)
    if held not in (2,3,4) or set(registry)!={'2','3','4'} or not registry[str(held)]: raise ValueError('OOF registry drift')
    if q.shape!=p.shape or len(q)!=len(memberships) or len(negatives)!=len(q): raise ValueError('Training alignment drift')
    if not (ridge>0 and temperature>0 and hard_count>0 and maxiter>0): raise ValueError('Fit budget drift')
    protected=set(registry[str(held)]); kept=[]; differences=[]; pair_counts=[]; source_negatives=0
    for i,m in enumerate(memberships):
        if m['fold'] not in (2,3,4) or m['positive_siren'] not in registry[str(m['fold'])]: raise ValueError('Training membership drift')
        if m['fold']==held or m['positive_siren'] in protected: continue
        if len(negatives[i])!=len(m['negative_sirens']): raise ValueError('Negative alignment drift')
        safe=[j for j,parent in enumerate(m['negative_sirens']) if parent not in protected]
        if not safe: continue
        n=unit_vectors(negatives[i])[safe]; source_negatives+=len(n)
        # Generic fixed metric mining, TRAINING ONLY, after entity exclusion.
        indices=np.argsort(-(n@q[i]),kind='stable')[:hard_count]
        differences.extend(p[i]-n[indices]); pair_counts.append(len(indices)); kept.append(i)
    if len(kept)<16: raise ValueError('Insufficient safe OOF scenes')
    q=q[kept].astype(np.float64); d=np.asarray(differences,dtype=np.float64)
    owner=np.repeat(np.arange(len(q)),pair_counts)
    sw=np.concatenate([np.full(n,1./n/len(q)) for n in pair_counts])
    dimension=q.shape[1]; identity=np.eye(dimension)
    def objective(flat):
        w=flat.reshape(dimension,dimension); transformed=q@w
        margins=temperature*np.einsum('ij,ij->i',transformed[owner],d)
        residual=-temperature*sw*expit(-margins)
        reduced=np.zeros_like(q); np.add.at(reduced,owner,residual[:,None]*d)
        delta=w-identity
        return float(sw@np.logaddexp(0.,-margins)+ridge*np.sum(delta*delta)/dimension),(
            q.T@reduced+2*ridge*delta/dimension).ravel()
    before=objective(identity.ravel())[0]
    result=minimize(objective,identity.ravel(),method='L-BFGS-B',jac=True,
        options=dict(maxiter=maxiter,maxcor=5,ftol=1e-8,gtol=1e-6))
    w=result.x.reshape(dimension,dimension).astype(np.float32)
    if not np.isfinite(w).all() or not np.isfinite(result.fun) or result.fun>before+1e-8: raise ValueError('Invalid/nonimproving fit')
    return w,dict(heldout_fold=held,training_folds=sorted({memberships[i]['fold'] for i in kept}),
        training_queries=len(kept),training_pairs=len(d),safe_source_negative_pairs=source_negatives,
        query_ids=[memberships[i]['query_id'] for i in kept],heldout_entity_overlap=0,
        identifiers_as_features=False,document_transform='IDENTITY',serving_score='NORMALIZED_QUERY_W_DOT_FIXED_DOCUMENT',
        ridge=ridge,temperature=temperature,hard_count=hard_count,maxiter=maxiter,
        objective_before=before,objective_after=float(result.fun),iterations=int(result.nit),
        optimizer_converged=bool(result.success),optimizer_message=str(result.message))
