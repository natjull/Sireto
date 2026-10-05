"""Query-only current official municipal/EPCI membership relevance.

TRAIN supervises a role, never an entity-name alias dictionary or serving
candidate pair. IDs only join the official role catalogue or block families.
Scores are bounded relevance, NOT calibrated ownership probabilities.
"""
import numpy as np
from .retrieval_activity_decoder_v111 import query_design

ROLES=('commune','epci')
RIDGE=1.


def validate(rows,registry):
 if len(rows)!=512 or len({r['query_id'] for r in rows})!=512 or set(registry)!={'2','3','4'}:
  raise ValueError('Complete protected original TRAIN512 required')
 for r in rows:
  s,p=r['positive_siret'],r['positive_siren']
  if (r['fold'] not in (2,3,4) or p not in registry[str(r['fold'])]
   or not isinstance(s,str) or not isinstance(p,str) or len(s)!=14 or len(p)!=9
   or not s.isascii() or not p.isascii() or not s.isdigit() or not p.isdigit() or not s.startswith(p)):
   raise ValueError('Original human TRAIN entity/fold joins required')


def fit(vectors,rows,registry,safe,role_parents,held):
 validate(rows,registry)
 if held not in (2,3,4) or set(role_parents)!=set(ROLES):raise ValueError('Official two-role scope required')
 x=query_design(vectors)
 if x.shape!=(512,385):raise ValueError('Fixed NOM-only384 design required')
 blocked=set(registry[str(held)]);safe_ids=list(safe['query_ids'])
 if safe['heldout_fold']!=held or set(safe['training_folds'])!={2,3,4}-{held}:
  raise ValueError('Original safe held-fold authority required')
 byid={r['query_id']:i for i,r in enumerate(rows)}
 if len(set(safe_ids))!=len(safe_ids) or set(safe_ids)-set(byid):raise ValueError('Original safe head membership required')
 selected=[byid[q] for q in safe_ids]
 if len(selected)<16 or any(rows[i]['fold']==held or rows[i]['positive_siren'] in blocked for i in selected):
  raise ValueError('Held positive families/fold must not train a query router')
 for role in ROLES:
  if any(not isinstance(p,str) or len(p)!=9 or not p.isascii() or not p.isdigit() for p in role_parents[role]):
   raise ValueError('Official parent membership JOINs, never numeric features')
 y=np.array([[r['positive_siren'] in role_parents[role] for role in ROLES] for r in rows],dtype=float)
 train=x[selected]
 parameter=np.linalg.solve(train.T@train+RIDGE*np.eye(385),train.T@y[selected])
 if parameter.shape!=(385,2) or not np.isfinite(parameter).all():raise ValueError('Finite fixed two-role query model required')
 return parameter,dict(heldout_fold=held,training_folds=sorted({rows[i]['fold'] for i in selected}),
  query_ids=safe_ids,training_queries=len(selected),positive_training_memberships=y[selected].sum(axis=0).astype(int).tolist(),
  heldout_entity_overlap=0,ridge=RIDGE,roles=list(ROLES),feature_dimension=384,identifiers_as_features=False,
  model_family='FIXED_RIDGE_TWO_CURRENT_SOURCE_ROLES',probabilities_calibrated=False,
  crm_alias_dictionary=False,official_list_missing_is_not_a_business_label_error=True)


def scores(vectors,parameter):
 x=query_design(vectors);p=np.asarray(parameter,dtype=float)
 if p.shape!=(385,2) or not np.isfinite(p).all():raise ValueError('Finite whole two-role parameter required')
 return np.clip(x@p,0.,1.)


def controls(vectors,rows,role_parents,models):
 # TRAIN OOF semantic controls ONLY; cannot select retrieval policy/weights.
 output={}
 for held,(parameter,metadata) in models.items():
  pos=[i for i,r in enumerate(rows) if r['fold']==int(held)]
  expected=np.array([[rows[i]['positive_siren'] in role_parents[role] for role in ROLES] for i in pos],dtype=float)
  predicted=scores(np.asarray(vectors)[pos],parameter)
  output[held]=dict(queries=len(pos),observed_positive_memberships=expected.sum(axis=0).astype(int).tolist(),
   mean_squared_error=np.mean((predicted-expected)**2,axis=0).tolist(),
   mean_absolute_error=np.mean(abs(predicted-expected),axis=0).tolist(),
   scope='TRAIN_OOF_QUERY_ROLE_CONTROLS_NOT_RECALL_OR_SELECTION')
 return output
