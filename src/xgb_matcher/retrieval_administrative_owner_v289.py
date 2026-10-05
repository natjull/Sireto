"""Independent uncertain administrative-owner support, not nominal identity.

Unlike282 this does NOT replace/amplify nominal0/6/8. Query-role relevance
grades an independent official parent join corroborated by address OR place.
"""
import math,time
import numpy as np
from .retrieval_corroborated_initial_v257 import Evidence,compose as original_compose
from .retrieval_public_initial_v282 import RoleModels
from .retrieval_public_role_v280 import ROLES
from .retrieval_proof_v56 import components
from .retrieval_structural_channels_v210 import identity


def administrative_support(backend,q,parents,grades):
 if set(parents)!=set(ROLES):raise ValueError('All two official administrative roles required')
 values=np.asarray(grades,dtype=float)
 if values.shape!=(2,) or not np.isfinite(values).all() or np.any((values<0)|(values>1)):
  raise ValueError('Bounded query-only role relevance, not identity probabilities')
 tv=backend._tantivy;parts=components(backend,q)
 supports=[p for p in (parts[1],parts[4]) if p is not None]
 if not supports:return None
 support=tv.Query.disjunction_max_query(supports,tie_breaker=0.);proofs=[]
 for role,grade in zip(ROLES,values):
  ids=parents[role]
  if any(not isinstance(p,str) or len(p)!=9 or not p.isascii() or not p.isdigit() for p in ids):
   raise ValueError('IDs categorical official JOIN only')
  if ids and grade>0:
   predicate=tv.Query.boolean_query([
    (tv.Occur.Must,tv.Query.const_score_query(tv.Query.term_set_query(backend.index.schema,'siren',sorted(ids)),0.)),
    (tv.Occur.Must,support)])
   proofs.append(tv.Query.boost_query(predicate,float(grade)))
 return tv.Query.disjunction_max_query(proofs,tie_breaker=0.) if proofs else None


def compose(backend,q,weights,terms,alpha,parents,sites,related,rings,role_parents,role_grades):
 base=original_compose(backend,q,weights,terms,alpha,parents,sites,related,rings)
 # Same explicit owner relationship is not counted twice. IDs are only
 # relation dedup JOINs; original exact owner support stays untouched.
 unproved={role:set(ids)-set(parents) for role,ids in role_parents.items()}
 public=administrative_support(backend,q,unproved,role_grades)
 if public is None:return base
 tv=backend._tantivy
 leaf=tv.Query.const_score_query(tv.Query.term_query(backend.index.schema,'document_type','siret','basic'),0.)
 return tv.Query.boolean_query([(tv.Occur.Must,leaf),
  (tv.Occur.Must,tv.Query.boolean_query([(tv.Occur.Should,base),(tv.Occur.Should,public)]))])


def search(backend,evidence,q,geo,weights,terms,alpha,municipal,role_grades,max_seconds=20.):
 began=time.monotonic();old_guard=evidence.files.guard
 def guard():
  old_guard()
  if time.monotonic()-began>max_seconds:raise TimeoutError('STOP_ADMINISTRATIVE_QUERY_COST')
 evidence.files.guard=guard
 try:
  parents,sites,nt=evidence.name_owners(q.name)
  related,rt=evidence.related(parents,sites);rings,gt=evidence.radial_proofs(geo)
  role_parents=municipal.parents(q)
  parsed=compose(backend,q,weights,terms,alpha,parents,sites,related,rings,role_parents,role_grades)
  guard();found=[]
  for score,loc in backend.searcher.search(parsed,limit=100,count=False).hits:
   guard();doc=backend.searcher.doc(loc).to_dict()
   row=dict(siret=backend._first(doc,'siret'),siren=backend._first(doc,'siren'),source_score=float(score),channel='joint_proof')
   identity(row)
   if not math.isfinite(score) or score<0:raise ValueError('Finite native initial score required')
   found.append(row)
  found.sort(key=lambda r:(-r['source_score'],r['siret']))
  if len(found)>100 or len({r['siret'] for r in found})!=len(found):raise ValueError('Distinct direct100 native cap required')
  for i,row in enumerate(found,1):row['retrieval_rank']=i
  guard()
  return found,dict(policy='ONE_NATIVE_GLOBAL_TOP100_CORROBORATED_BOUNDED_PROOFS',
   source_admitted={'joint_proof':len(found)},fixed_reservoir=None,score_after_fixed_pool=False,initial_cap=100,
   exact_global_fused_score_top100=True,names=nt,relations=rt,geometry=gt,
   native_leaf_sites=43896818,full_appended_leaf_universe=False,downstream_models_changed=False,
   label_lookup=False,identifiers_as_features=False,recall_measured=False,production_slo_validated=False,
   seconds=time.monotonic()-began,new_fit=False,frozen_oof11_sparse8_head=True,
   initial_place_training='V228_OOF_TRAIN64_NATIONAL_CONTRASTS',actual217_fit_parity_claimed=False,
   source_hypothesis='V289_INDEPENDENT_ADMINISTRATIVE_OWNER_ADDRESS_OR_PLACE_SUPPORT',
   public_role_grades=list(map(float,role_grades)),public_parent_counts={r:len(role_parents[r]) for r in ROLES},
   public_role_calibrated_probability=False,official_administration_is_not_certified_equipment_owner=True,
   public_is_global_hard_filter=False,training_scenes_reopened=False,public_coefficient=1.,
   nominal_components_unchanged_from_v257=True,metric_equivalent_to_v282=False,
   tie_scope='CANONICAL_ONLY_AMONG_RETURNED_NATIVE_SCORE_EQUALS')
 finally:evidence.files.guard=old_guard
