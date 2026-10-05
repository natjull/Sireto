"""NEW activity-at-place predicate fused before ONE global native top100.

Unlike V115/V116 there is no additive-frontier score completion. This is NOT
the same metric optimized: it adds ONE weak bounded grade to fixed V229 head.
No model training, class/owner cap, output rerank or unknown-site removal.
"""
import json
import math
import time

import numpy as np

from .retrieval_corroborated_initial_v217 import Evidence,compose as original_compose
from .retrieval_activity_semantic_v113 import normalized,class_lookup,scores
from .retrieval_activity_decoder_v111 import query_design
from .retrieval_activity_postings_v263 import collect_groups,native_proof
from .retrieval_structural_channels_v210 import identity


class ActivityModels:
    def __init__(self,files,semantic_ref,old,queries,heads,structured):
        m=json.loads(files.file(semantic_ref).read_text())
        if (m['status']!='SEALED_TRAINING_ONLY_OFFICIAL_ACTIVITY_SEMANTIC_PROBE'
                or any(m[k] is not False for k in ('recall_measured','fold0_opened','fold1_opened',
                    'identifiers_as_features','output_candidates','national_search_performed','downstream_models_changed'))
                or m['no_crm_alias_dictionary'] is not True or not m['resources']['pass_resource_policy']):
            raise ValueError('Existing safely trained semantic activity artifact required')
        launch=json.loads(files.file(m['launch']).read_text())
        for ref in launch['implementation'].values():files.file(ref)
        source=launch['inputs']
        catalog=json.loads(files.file(structured['activity_catalog']).read_text())
        files.file(m['encoding'])
        if (catalog['status']!='SEALED_COMPLETE_EXPLICIT_NATIONAL_NOMENCLATURE'
                or catalog['stats']['sites']!=structured['stats']['sites']
                or source['nomenclatures']!=catalog['stats']['nomenclatures']
                or source['activities']!=catalog['stats']['activity_book']
                or source['nomenclature']!=structured['reused_arrays']['nomenclature_codes']
                or source['original_activity']!=structured['reused_arrays']['original_activity_codes']
                or source['labels']!=catalog['official_labels']):
            raise ValueError('Same versioned current official activity catalog required')
        labels=json.loads(files.file(source['labels']).read_text())['rows']
        if len(labels)!=2140:raise ValueError('ALL official descriptions required')
        models=json.loads(files.file(m['models']).read_text())
        if set(models)!={'2','3','4'} or set(heads)!=set(models):raise ValueError('Three OOF heads required')
        encoded=json.loads(files.file(old['expansions']).read_text())
        bundle=json.loads(files.file(old['bundle']).read_text())
        if (encoded['truth_opened'] is not False or not encoded['resources']['pass_resource_policy']
                or [(r['query_id'],r['fold']) for r in encoded['rows']]!=[(q['query_id'],q['fold']) for q in queries]
                or bundle['model_refs']!=source['model_refs'] or bundle['model_path']!=source['model_path']
                or bundle['max_length']!=source['max_length'] or bundle['encoder_prefix']!='nom'):
            raise ValueError('Original query-only encoder compatibility required')
        for ref in bundle['model_refs'].values():files.file(ref)
        vectors=np.load(files.file(encoded['vectors']),allow_pickle=False)
        if vectors.shape!=(128,384) or not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors,axis=1),1.,atol=1e-6):
            raise ValueError('Full original query-only128 name vectors required')
        self.mapping=class_lookup(source['nomenclatures'],source['activities'],labels)
        self.grades={};description_ref=None
        for f,model in models.items():
            if (model['heldout_fold']!=int(f) or set(model['training_folds'])!={2,3,4}-{int(f)}
                    or model['identifiers_as_features'] is not False or model['family']!='SEMANTIC_RIDGE'
                    or model['ridge']!=1 or model['catalog']!=structured['activity_catalog']
                    or set(model['query_ids'])-set(heads[f]['query_ids'])
                    or set(model['query_ids']) & {q['query_id'] for q in queries}):
                raise ValueError('Frozen OOF semantic decoder, original safe TRAIN families and disjoint queries required')
            if description_ref is not None and description_ref!=model['descriptions']:
                raise ValueError('Same official description universe required')
            description_ref=model['descriptions']
            docs=normalized(np.load(files.file(description_ref),allow_pickle=False))
            parameter=np.load(files.file(model['parameter']),allow_pickle=False)
            if (docs.shape!=(2140,384) or parameter.shape!=(385,384)
                    or np.any(np.linalg.norm(docs,axis=1)<.999)):
                raise ValueError('Official semantic dimensions drift')
            self.grades[f]=scores(query_design(vectors),docs,parameter,model['family'])
        files.check_epochs()


def search(backend,evidence,q,geo,weights,terms,alpha,grades,mapping,max_seconds=20.):
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
        base=original_compose(backend,q,weights,terms,alpha,parents,sites,related,rings)
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
            activity_coefficient=1.,continuous_grades_quantized=False,class_or_site_cap=None,
            unknown_and_offzone_sites_remain_reachable=True,metric_equivalent_to_v115_v116=False,
            tie_scope='CANONICAL_ONLY_AMONG_RETURNED_NATIVE_SCORE_EQUALS')
    finally:evidence.files.guard=old_guard
