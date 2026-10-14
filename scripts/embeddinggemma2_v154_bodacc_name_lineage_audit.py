"""Static business-name source→canonical→portfolio→index lineage audit.

No source row, person/address value, CRM, label, retrieval or fit is read.
Existing bounded source-shape counts are reused only as presence metadata.
"""
from pathlib import Path
import ast
import time

from scripts import embeddinggemma2_night as io
from src.xgb_matcher.retrieval_v20_workbench import record, verify

ROOT=io.ROOT
DATA=io.DATA
OUT=io.HOME/'v154/name_field_lineage_20261014_night'
BUILDER=ROOT/'src/xgb_matcher/official_evidence_builder.py'
INDEX=DATA/'indexes/hierarchical_retrieval_v3/4b332899d8698354/manifest.json'
DICT=DATA/'audits/retrieval_v130_variant_dictionary/compile_20261003_a/manifest.json'
PLAN=DATA/'audits/retrieval_v129_national_variants/projection_20261003_a/plan.json'
FIELDS={'denomination','nomCommercial','enseigne','sigle'}


def business_fields(tree,function):
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==function)
    return sorted({n.elts[0].value for n in ast.walk(node) if isinstance(n,ast.Tuple) and n.elts
        and isinstance(n.elts[0],ast.Constant) and n.elts[0].value in FIELDS})


def main():
    if OUT.exists():raise ValueError('Fresh static audit namespace')
    tree=ast.parse(BUILDER.read_text())
    person_fields=business_fields(tree,'_bodacc_person_names')
    if person_fields!=['denomination','enseigne','nomCommercial']:
        raise ValueError('Known canonical business field mapping drift')
    im=io.read(INDEX);dm=io.read(DICT);plan=io.read(PLAN)
    index_builder=record(ROOT/'scripts/build_hierarchical_retrieval_index.py')
    portfolio_policy=record(ROOT/'config/siren_name_portfolio_v1.json')
    index_source=io.read(verify(im['sources']['dossier_manifest']))
    if index_builder['sha256']!=im['builder_sha256'] or portfolio_policy['sha256']!=index_source['name_portfolio_policy_sha256']:
        raise ValueError('Actual index builder or portfolio-policy lineage drift')
    policy=io.read(ROOT/'config/siren_name_portfolio_v1.json')
    variant_impl=record(ROOT/'src/xgb_matcher/retrieval_variants_v118.py')
    dict_launch=io.read(verify(dm['launch']))
    if variant_impl!=dict_launch['implementation'][str(ROOT/'src/xgb_matcher/retrieval_variants_v118.py')]:
        raise ValueError('Actual all-name dictionary projector version differs')
    source_manifest=DATA/'sources/official_evidence/official_history_2008_2026_v1/manifest.json'
    raw_merged=io.read(source_manifest)
    dossier=DATA/'sources/siren_dossier_v3/01eb624cca97c6f3/manifest.json'
    dossier_manifest=io.read(dossier)
    if raw_merged['artifacts']['official_evidence.parquet']['sha256'] not in {r['sha256'] for r in dossier_manifest['inputs']['official_evidence']}:
        raise ValueError('Actual canonical source→dossier linkage differs')
    current_fields=plan['sources']['current_ul']['source_fields']
    historical_fields=plan['sources']['history_ul']['source_fields']
    prior_counts=io.read(io.HOME/'v152/bodacc_structure_scope_20261014_night_b/counts.json')
    presence={k:prior_counts['field_presence'].get(k,0) for k in
        ('person.denomination','person.nomCommercial','person.enseigne','person.sigle','establishment.enseigne')}
    matrix=[
        dict(field='SIRENE_CURRENT.sigleUniteLegale',canonical_kind='TRADE',
            current_source_schema_present='sigleUniteLegale' in current_fields,
            dictionary_scope='LEGAL_PERSON_COMMERCIAL_NAME_ELIGIBLE_ALL_DECLARED_VARIANTS',
            indexed_role='TRADE_CURRENT',indexed_selection='MAX8_PER_SIREN',absence_not_claimed=True),
        dict(field='SIRENE_HISTORY.sigleUniteLegale',historical_source_schema_present='sigleUniteLegale' in historical_fields,
            behavior='PROJECTOR_SUPPORTS_FIELD_IF_PRESENT_BUT_THIS_PINNED_HISTORY_UL_HAS_NO_SIGLE_COLUMN',
            absence_not_claimed_in_all_other_sources=True),
        dict(field='RNE.personneMorale.identite.description.sigle',canonical_kind='TRADE',
            allowlisted=True,scope='EXPLICIT_OWN_SUBJECT',source_dates_retained=True),
        dict(field='BODACC.personne.sigle',official_schema_declared=True,canonical_allowlisted=False,
            failure_stage='RAW_STRUCTURED_FIELD_IGNORED_BY_CURRENT_CANONICAL_NAME_PARSER',
            not_a_downstream_index_tokenizer_loss=True,not_necessarily_new_vs_current_SIRENE_OR_RNE=True),
        dict(field='BODACC.personne.denomination',canonical_allowlisted=True,canonical_kind='LEGAL',
            binding='SAME_PERSON_OWN_REGISTRATION_SIREN',portfolio_role='SUPPORTING',
            native_lexical='SUPPORTING_FIELD_SELECTED_MAX6_PER_SIREN',
            uncapped_dictionary='ALL_CANONICAL_OBSERVATIONS_LEGAL_PERSON_NAME_SCOPE'),
        dict(field='BODACC.personne.nomCommercial',canonical_allowlisted=True,canonical_kind='TRADE',
            binding='SAME_PERSON_OWN_REGISTRATION_SIREN',portfolio_role='SUPPORTING',
            native_lexical='SUPPORTING_FIELD_SELECTED_MAX6_PER_SIREN',
            uncapped_dictionary='ALL_CANONICAL_OBSERVATIONS_LEGAL_PERSON_COMMERCIAL_NAME_SCOPE',
            no_blanket_nomCommercial_loss=True),
        dict(field='BODACC.personne.enseigne',canonical_allowlisted=True,canonical_kind='SIGN',
            binding='SAME_PERSON_OWN_REGISTRATION_SIREN',portfolio_role='SUPPORTING',
            native_lexical='SUPPORTING_FIELD_SELECTED_MAX6_PER_SIREN',
            uncapped_dictionary='SIREN_SIGN_CONTEXT_ONLY_SITE_SCOPE_UNVERIFIED_NOT_LEXICAL_ELIGIBLE',
            safety_exclusion_not_proven_erroneous=True),
        dict(field='BODACC.etablissement.enseigne/nomCommercial',canonical_allowlisted=True,
            binding='ESTABLISHMENT_NAMES_ONLY_WHEN_CURRENT_STRUCTURED_SIREN_UNIQUE',
            explicit_siret='EXPLICIT_ENDPOINT_ALIAS_ELIGIBLE_WHEN_SIRET_PUBLISHED',
            no_siret='SIREN_TRADE_ELIGIBLE_BUT_SIREN_SIGN_CONTEXT_ONLY_IN_FULL_DICTIONARY',
            ambiguous_multiple_actor='NOT_FANNED_OUT_TO_EVERY_ACTOR',
            no_actor_cross_product=True,no_own_site_assignment_from_parent_address=True),
    ]
    OUT.mkdir(parents=True)
    io.write(OUT/'field_role_lineage.json',matrix)
    io.write(OUT/'report.json',dict(status='COMPLETE_STATIC_BUSINESS_NAME_LINEAGE_NOT_SOURCE_COVERAGE_OR_RECALL',
        implementation=record(Path(__file__)),field_role_lineage=record(OUT/'field_role_lineage.json'),
        current_canonical_builder=record(BUILDER),index_manifest=record(INDEX),actual_index_builder=index_builder,
        actual_policy=portfolio_policy,dictionary_manifest=record(DICT),actual_variant_projector=variant_impl,
        dictionary_source_plan=record(PLAN),canonical_merged_manifest=record(source_manifest),dossier_manifest=record(dossier),
        actual_source_rows_or_index_documents_not_read=True,raw_names_personal_data_addresses_not_read=True,
        canonical_year_manifests_do_not_bind_historical_builder_sha=True,
        historical_code_lineage=dict(person_name_business_allowlist_unchanged_since_commit='12073847',
            source_code_commit_date='2026-08-18',actual_index_creation=im['created_at'],
            omission_is_code_and_lineage_evidence_not_cryptographic_replay_of_every_canonical_row=True),
        bounded_existing_shape_presence=presence,bounded_source_sample_rows=prior_counts['sampled_rows'],
        presence_counts_include_all_sample_rows_not_just_public_pm_not_distinct_aliases=True,
        no_presence_count_extrapolated_to_national_population=True,
        legacy_index_role_policy=dict(supporting=policy['roles']['SUPPORTING'],
            declared_rescue_only_historical=True,
            actual_V115_native_word_query_uses_all5_fields_before_same_collector100=True,
            historic_supporting_names_not_in_current_name_char_ngrams=True),
        temporal_lineage=dict(raw_bodacc_person_valid_from='PUBLICATION_DATE_NOT_ASSERTED_EFFECTIVE_NAME_DATE',
            canonical_and_projected_observations_keep_dates=True,
            existing_index_and_full_dictionary_not_explicitly_filtered_at_2026_08_01=True,
            sqlite_alias_lookup_drops_dates_scope_and_source_after_admitting_typed_facts=True,
            inherited_baseline_is_development_diagnostic_not_asof_certification=True),
        structured_array_limit='DECODER_HANDLES_ROOT_LISTS_ITEMS_LISTE_VALUES_BUT_NOT_PERSONNE_OR_ETABLISSEMENT_WRAPPER_ARRAY; NO_CROSS_PRODUCT_REPAIR',
        one_uniform_future_direction='WHOLE_SOURCE_PUBLIC_CORPORATE_OWN_PERSONNE_SIGLE_AT_CUTOFF_WITH_DATED_SIREN_TRADE_PROVENANCE_DEDUP_VS_EXISTING_SIRENE_RNE_NAMES_NO_SITE_PROMOTION',
        future_source_first_then_unique_ablation_only_if_new_public_owner_names_remain=True,
        no_crm_label_lookup_native_model_fit_alias_creation_or_pipeline_change=True,
        no_claim_all_raw_bodacc_names_absent=True,no_recall_gain_claim=True,closed_sets_opened=False,observed_unix=time.time()))
    print(dict(person_business_fields=person_fields,missing_sigle=True,current_sirene_sigle_present=True,
        history_sirene_sigle_column_present='sigleUniteLegale' in historical_fields,
        bounded_existing_shape_presence=presence,index_builder_exact_hash_match=True),flush=True)


if __name__=='__main__':main()
