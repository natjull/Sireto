"""One initial-index place coefficient fitted only on safe TRAIN contrasts."""
from copy import deepcopy
from collections import Counter
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def fit_place(scenes, mined, registry, original, acceptable, held):
    if held not in (2, 3, 4) or set(registry) != {'2', '3', '4'}:
        raise ValueError('Safe folds and original full entity registry required')
    byid = {s['query_id']: s for s in scenes}
    if len(byid) != 512 or len(mined) != 64 or len({r['query_id'] for r in mined}) != 64:
        raise ValueError('Whole original512 and complete TRAIN64 required')
    protected = set(registry[str(held)]); m = original[str(held)]
    weights = np.asarray(m['weights'], dtype=float)
    if weights.shape != (11,) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('Actual original fixed11 initial head required')
    margins=[]; contrast=[]; sample_weights=[]; kept=[]; counts=Counter(); excluded=Counter()
    for row in mined:
        s=byid[row['query_id']]
        if s['fold'] != row['fold'] or s['fold'] not in (2,3,4) or s['positive_siren'] not in registry[str(s['fold'])]:
            raise ValueError('Original source entity/fold alignment required')
        if s['fold'] == held or s['positive_siren'] in protected:
            excluded['held_fold_or_positive_entity'] += 1; continue
        if s['query_id'] not in m['query_ids']:
            raise ValueError('New coefficient must preserve original entity-safe training membership')
        p=np.asarray(s['positive_features'][:11]); local=[]; national=[]
        for n in s['negatives']:
            if n['siren'] not in protected:
                local.append(np.asarray(n['features'][:11]))
        seen=set()
        for values in row['channels'].values():
            for n in values:
                sid=n['siret']
                if sid in seen: continue
                seen.add(sid)
                if sid in acceptable[s['query_id']]: excluded['acceptable_national_references']+=1; continue
                if n['siren'] in protected: excluded['held_national_references']+=1; continue
                national.append(np.asarray(row['features'][sid]))
        groups=[('local',local),('national',national)]
        groups=[(k,v) for k,v in groups if v]
        if not groups: raise ValueError('No safe negative: STOP, never silently omit selected training query')
        for name, values in groups:
            for n in values:
                if p.shape!=(11,) or n.shape!=(11,) or not np.isfinite(p).all() or not np.isfinite(n).all() or p[4] not in (0.,1.) or n[4] not in (0.,1.):
                    raise ValueError('Finite actual native11 and binary place required')
                diff=p-n
                margins.append(float(diff@weights)); contrast.append(float(diff[4]))
                sample_weights.append(1./(len(groups)*len(values)))
                counts[f'{name}_{int(p[4])}_{int(n[4])}']+=1
        kept.append(s['query_id'])
    if len(kept)<16 or not any(contrast): raise ValueError('At least16 safe queries and genuine place contrast required')
    a=np.asarray(margins); d=np.asarray(contrast); sw=np.asarray(sample_weights)/len(kept)
    prior=float(weights[4]); ridge=.005
    def objective(w):
        delta=float(w[0])-prior; z=a+delta*d
        return float(sw@np.logaddexp(0.,-z)+ridge*delta*delta), np.array([float((-sw*expit(-z))@d+2*ridge*delta)])
    result=minimize(objective,[prior],method='L-BFGS-B',jac=True,bounds=[(0.,64.)],
                    options=dict(maxiter=500,ftol=1e-10,gtol=1e-7))
    if not result.success or not np.isfinite(result.x).all(): raise ValueError('Place fit failed')
    updated=deepcopy(m); updated['weights'][4]=float(result.x[0])
    updated['place_training']=dict(query_ids=kept,training_queries=len(kept),training_pairs=len(a),
        pairs_by_channel_and_place=dict(counts), exclusions=dict(excluded),ridge=ridge,
        objective_before=objective([prior])[0], objective_after=float(result.fun),
        heldout_entity_overlap=0,changed_indices=[4], national_negative_source='V227_QUERY_ONLY_TRAIN64',
        fit_score_scope='FIXED11_PRIMITIVES56_NOT_FULL217',identifiers_as_features=False)
    return updated
