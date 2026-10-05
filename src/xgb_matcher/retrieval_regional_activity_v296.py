"""New quantized activity + official regional qualifier native initial100.

NOT the continuous272 metric retried: fixed score quantization plus regional
286 nominal paths. STOP versions/partials are not read, copied or evaluated.
"""
import math
import time
from .retrieval_regional_initial_v286 import Evidence,compose as original_compose
from .retrieval_activity_native_v264 import ActivityModels
from .retrieval_quantized_activity_v295 import collect_groups,native_proof
from .retrieval_structural_channels_v210 import identity

def search(backend,evidence,q,geo,weights,terms,alpha,grades,mapping,regions,max_seconds=20.):
    began=time.monotonic();old_guard=evidence.files.guard
    def guard():
        old_guard()
        if time.monotonic()-began>max_seconds:raise TimeoutError('STOP_ACTIVITY_AT_PLACE_QUERY_COST')
    evidence.files.guard=guard
    try:
        parents,sites,nt=evidence.name_owners(q.name)
        related,rt=evidence.related(parents,sites)
        rings,gt=evidence.radial_proofs(geo)
        groups,at=collect_groups(evidence,q,grades,mapping)
        tv=backend._tantivy
        base=original_compose(backend,q,weights,terms,alpha,parents,sites,related,rings,regions)
        activity=native_proof(backend,groups)
        parsed=base if activity is None else tv.Query.boolean_query([
            (tv.Occur.Must,tv.Query.const_score_query(tv.Query.term_query(backend.index.schema,'document_type','siret','basic'),0.)),
            (tv.Occur.Must,tv.Query.boolean_query([(tv.Occur.Should,base),(tv.Occur.Should,activity)]))])
        guard();found=[]
        for score,loc in backend.searcher.search(parsed,limit=100,count=False).hits:
            guard();doc=backend.searcher.doc(loc).to_dict()
            row=dict(siret=backend._first(doc,'siret'),siren=backend._first(doc,'siren'),source_score=float(score),channel='joint_proof')
            identity(row)
            if not math.isfinite(score) or score<0:raise ValueError('Finite initial native score required')
            found.append(row)
        found.sort(key=lambda r:(-r['source_score'],r['siret']))
        if len(found)>100 or len({r['siret'] for r in found})!=len(found):raise ValueError('Distinct direct100 required')
        for i,r in enumerate(found,1):r['retrieval_rank']=i
        guard()
        return found,dict(policy='ONE_NATIVE_GLOBAL_TOP100_CORROBORATED_BOUNDED_PROOFS',
            source_admitted={'joint_proof':len(found)},fixed_reservoir=None,score_after_fixed_pool=False,
            initial_cap=100,exact_global_fused_score_top100=True,names=nt,relations=rt,geometry=gt,activity=at,
            activity_grades=len(groups),activity_memberships=sum(len(ids) for ids in groups.values()),
            native_leaf_sites=43896818,full_appended_leaf_universe=False,downstream_models_changed=False,
            label_lookup=False,identifiers_as_features=False,recall_measured=False,production_slo_validated=False,
            seconds=time.monotonic()-began,new_fit=False,initial_place_training='V228_OOF_TRAIN64_NATIONAL_CONTRASTS',
            activity_decoder='V113_FROZEN_OOF_SEMANTIC_RIDGE1_OFFICIAL_DESCRIPTIONS',
            activity_is_not_identity_or_calibrated_probability=True,activity_raw_global_without_place=False,
            activity_coefficient=1.,continuous_grades_quantized=True,activity_score_quantization_levels=128,
            activity_quantization_max_absolute_error=1/128,class_or_site_cap=None,
            source_hypothesis='V296_OFFICIAL_ACTIVITY_FLOOR128_AND_OFFICIAL_REGIONAL_QUALIFICATION',
            regional_names_considered=len(regions),metric_equivalent_to_v272=False,
            metric_equivalent_to_v265=False,previous_stop_outputs_reused=False,
            unknown_and_offzone_sites_remain_reachable=True,metric_equivalent_to_v115_v116=False,
            tie_scope='CANONICAL_ONLY_AMONG_RETURNED_NATIVE_SCORE_EQUALS')
    finally:evidence.files.guard=old_guard
