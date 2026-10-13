"""Read-only structural diagnosis of all22 already-consumed V115 failures.

No aliases/context or predictions emitted; protected targets and natural-person
names are never reconstructed. Historical dossiers are bounded observations,
not a proof that every raw supplier payload has been inspected.
"""
from collections import Counter
from pathlib import Path
import json
import sqlite3
import time

from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v49_postseal_bodacc_roles as naming
from src.xgb_matcher.retrieval_v20_workbench import record, verify

OUTPUT = io.HOME/'v130/structural_audit_20261013_night'
BASE = io.HOME/'v115/streamed_activity_ablation_20261013_a'
LABELS = io.HOME/'v122/label_clarification_20261013_b'
DOSSIER = io.HOME/'v20/deep_audit_20261011_c/original_source_records.json'
ALIAS_REPORT = io.HOME/'v103/raw_alias_coverage_20261013_a/report.json'
BODACC = io.HOME/'v53/postseal_bodacc_reverse_20261012_a'


def run():
    import pyarrow.parquet as pq
    if OUTPUT.exists():
        raise ValueError('Fresh immutable diagnostic namespace')
    generation, ex = io.read(BASE/'generation.json'), io.read(BASE/'generate_exit.json')
    if generation['status'] != 'COMPLETE_SEALED_ALL526' or ex['exit_code'] or not ex['own_child_exit_confirmed']:
        raise ValueError('Owned complete source before labels')
    verify(generation['outputs'])
    label_authority = io.read(LABELS/'report.json')
    labels = io.read(verify(label_authority['cases']))
    outcomes = io.read(BASE/'outcomes.json')['learned_native_ooc']
    misses = {r['query_id'] for r in outcomes if not r['exact']}
    if len(labels) != 22 or {r['query_id'] for r in labels} != misses:
        raise ValueError('All22 actual failures retained')
    names = io.read(ALIAS_REPORT)['names']
    db = sqlite3.connect('file:'+str(verify(names))+'?mode=ro&immutable=1',uri=True)
    dossier = io.read(DOSSIER)
    OUTPUT.mkdir(parents=True)
    cases = []
    for case in labels:
        protected = case['protected_no_reconstruction']
        category = case['public_legal_category_only'] or {}
        natural_person = category.get('legal_category') == '1000'
        q, target = case['crm'], case['label_siret']
        qt = naming.tokens(q['crm_name'])
        facts = []
        source_missing_aliases = []
        if not protected and not natural_person:
            # Own endpoint plus legal-parent facts ONLY. Sibling facts retained
            # as separate diagnostic context, never copied to the target.
            related = [r for r in dossier['names'] if r['siren'] == target[:9]]
            for fact in related:
                role = 'OWN_PARENT' if fact['subject_kind']=='SIREN' else 'OWN_SITE' if fact['siret']==target else 'OTHER_SITE_SAME_OWNER'
                overlap = sorted(qt & naming.tokens(fact['raw_value']))
                fact_name = naming.norm(fact['raw_value'])
                # Legal/sign source roles are compared to exactly the same
                # endpoint, not a name found elsewhere under the company.
                kind, key = (1,int(target[:9])) if fact['subject_kind']=='SIREN' else (2,int(fact['siret']))
                indexed = bool(db.execute('SELECT 1 FROM aliases WHERE kind=? AND owner_key=? AND name=?',
                    (kind,key,fact_name)).fetchone()) if fact_name else False
                facts.append(dict(source=fact['source'],role=role,name_kind=fact['name_kind'],
                    matching_query_tokens=overlap, in_dictionary_same_role=indexed,
                    published_date=fact['valid_from'],source_record_id=fact['source_record_id']))
                if overlap and not indexed:
                    source_missing_aliases.append(dict(source=fact['source'],role=role,
                        published_name=fact['raw_value'],kind=fact['name_kind'],source_record_id=fact['source_record_id']))
        cases.append(dict(query_id=q['query_id'],crm_name=q['crm_name'],label_siret=target,
            protected_no_lookup=protected,natural_person_names_not_read=natural_person,
            bounded_original_dossier_facts=len(facts),source_fact_roles=dict(Counter(f['role'] for f in facts)),
            matched_fact_roles=dict(Counter(f['role'] for f in facts if f['matching_query_tokens'])),
            own_matching_published_name_lost_from_dictionary=[f for f in source_missing_aliases if f['role']!='OTHER_SITE_SAME_OWNER'],
            sibling_matching_published_name_lost_from_dictionary=[f for f in source_missing_aliases if f['role']=='OTHER_SITE_SAME_OWNER'],
            exact_target_state=next((s['state'] for s in case['same_owner_public_sites_in_query_zone'] if s['siret']==target),None),
            succession_count=len(case['direct_succession_edges']),facts=facts,
            bounded_dossier_not_full_raw_supplier_coverage=True,no_alias_or_label_change=True))
    db.close()
    transfer = io.read(BODACC/'subjects/539891390.json')
    announcement = next(r for r in transfer['records'] if r['id']=='A20180103410')
    structured_parties = [r for r in announcement['parties'] if r['siren'] in ('539891390','819686866')]
    historical_file = Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/sources/sirene/2026-08-01/StockEtablissementHistorique.parquet')
    history_columns = pq.ParquetFile(historical_file).schema_arrow.names
    io.write(OUTPUT/'cases.json',cases)
    io.write(OUTPUT/'report.json',dict(status='COMPLETE_ALL22_POSTSEAL_STRUCTURAL_AUDIT_NOT_RETRIEVAL',
        implementation=record(Path(__file__)), baseline_generation=record(BASE/'generation.json'),
        baseline_exit=record(BASE/'generate_exit.json'), labels=record(LABELS/'report.json'),
        bounded_dossier=record(DOSSIER), dictionary=names,cases=record(OUTPUT/'cases.json'),
        cases_count=22,protected_no_lookup=2,natural_person_names_not_read=sum(r['natural_person_names_not_read'] for r in cases),
        own_overlap_name_missing_dictionary_cases=sum(bool(r['own_matching_published_name_lost_from_dictionary']) for r in cases),
        historical_schema_columns=history_columns,historical_stock_has_addresses=False,
        publication_and_effective_dates_not_equated=True,
        observed_transfer_example=dict(source=record(BODACC/'subjects/539891390.json'),
            announcement_id=announcement['id'],publication_date=announcement['date'],
            structured_corporate_parties=structured_parties,
            separately_published_effective_date='2018-04-01',
            fund_address='4-6 RUE DE LA MARNE CHALONS EN CHAMPAGNE',
            role='BUSINESS_ASSET_PREVIOUS_OWNER_TO_BUYER_NOT_WHOLE_COMPANY_IDENTITY'),
        data_quality_correction='INFODESCA_TARGET_31688191100010_ACTIVE_A_NOT_CLOSED_IN_PUBLIC_STOCK',
        admissibility='DIAGNOSTIC_ONLY_INHERITED_POSTCUTOFF_SOURCES_NO_BENCHMARK_ADMISSION',
        no_labels_changed=True,no_closed_dataset_opened=True,no_new_metrics_or_training=True,
        next_uniform_experiments=[
            'EXPLICIT_PUBLIC_RNA_TITLES_AND_GENERIC_INITIALISMS_ALREADY_PREPARED_WHOLE_SOURCE',
            'OWN_FACILITY_TO_REGISTERED_HQ_TYPED_ADMIN_ALTERNATIVE_V118_READY_MAX_ONE_PRIOR_MISS',
            'DATED_BUSINESS_ASSET_TRANSFER_SITE_BUNDLE_NAMES_ADDRESS_ROLE_NOT_SIREN_ALIAS'],
        limitations=['BODACC_FREE_TEXT_DOES_NOT_ESTABLISH_PRIVATE_CONTRACT_OR_CRM_BILLING_ROLE',
            'SOURCE_NORMALISATION_TEST_ON_CONSUMED_PANEL_IS_EXPLORATORY',
            'NO_GLOBAL_99_2_CLAIM_OR_STRICT_SOURCE_INFORMATION_UPPER_BOUND'],observed_unix=time.time()))
    print('COMPLETE_ALL22_POSTSEAL_STRUCTURAL_AUDIT_NOT_RETRIEVAL',flush=True)


if __name__=='__main__':run()
