"""Exhaustive postseal remaining20: source/representation/score/site limits.

No native search, source alias, label repair, fitting or closed population.
Full native target scores absent from sealed100 are explicitly NOT invented.
"""
from collections import Counter
from pathlib import Path
import gc
import time

import numpy as np
from scripts import embeddinggemma2_night as io
from src.xgb_matcher.retrieval_v20_workbench import record, verify

ROOT = io.HOME/'v140/structural_native_disk_20261013_night'
BASE = io.HOME/'v115/streamed_activity_ablation_20261013_a'
FACTS = io.HOME/'v122/label_clarification_20261013_b'
NUMERIC = io.HOME/'v131/mechanism_20261013_night'
PORTFOLIO = io.HOME/'v103/raw_alias_coverage_20261013_a'
OUT = io.HOME/'v140/remaining_structure_20261014_night'
BITS = (1 << 20)-1


def guard():
    import psutil
    rss = psutil.Process().memory_info().rss
    free, level = io.pressure()
    if rss > 1024**3 or free < 25 or level not in (0, 1, 2):
        raise RuntimeError('Light audit yields on actual resources; preserve outputs')


def sparse(ref, target):
    if ref is None:
        return dict(available=False, leaves=0, target_confidence=0., maximum_confidence=0.)
    with np.load(verify(ref), allow_pickle=False) as f:
        keys, confidence = f['keys'], f['confidence']
        at = int(np.searchsorted(keys, target))
        value = float(confidence[at]) if at < len(keys) and keys[at] == target else 0.
        return dict(available=bool(len(keys)), leaves=len(keys), target_confidence=value,
                    maximum_confidence=float(confidence.max(initial=0.)), source=ref)


def main():
    guard()
    if OUT.exists():
        raise ValueError('Fresh diagnostic namespace')
    top, ex = io.read(ROOT/'report.json'), io.read(ROOT/'owned_tool_exit.json')
    if top['status'] != 'COMPLETE_SIX_PREDECLARED_STRUCTURAL_ABLATIONS_ALL526' or ex['exit_code'] or not ex['own_tool_exit_confirmed']:
        raise ValueError('Complete actual six-source measurement before label diagnosis')
    primary = ROOT/top['primary_predeclared']
    gen, child = io.read(primary/'generation.json'), io.read(primary/'generate_exit.json')
    if child['exit_code'] or not child['own_child_exit_confirmed'] or gen['status'] != 'COMPLETE_SEALED_ALL526':
        raise ValueError('Actual complete primary')
    outputs = io.read(verify(gen['outputs']))['learned_native_ooc']
    outcomes = io.read(primary/'outcomes.json')['learned_native_ooc']
    remaining = [r for r in outcomes if not r['exact']]
    if len(outcomes) != 526 or len(remaining) != 20 or any(r['fold'] not in (2, 3, 4) for r in remaining):
        raise ValueError('Exact exhaustive20, no denominator/closed-set change')
    facts = {r['query_id']: r for r in io.read(verify(io.read(FACTS/'report.json')['cases']))}
    numeric_report = io.read(NUMERIC/'report.json')
    numeric = {r['query_id']: r for r in io.read(verify(numeric_report['cases']))}
    portfolio = {r['query_id']: r for r in io.read(verify(io.read(PORTFOLIO/'report.json')['cases']))}
    launch = io.read(ROOT/'launch.json')
    if numeric_report['baseline_generation'] != launch['baseline_generation']:
        raise ValueError('Numerical score diagnostic from exact unchanged baseline')
    if io.read(ROOT/'features_generation.json') != io.read(BASE/'features_generation.json') or io.read(ROOT/'calibration/generation.json') != io.read(BASE/'calibration/generation.json'):
        raise ValueError('Same51 numerical fields and58 coefficients')
    cal = io.read(ROOT/'calibration/generation.json')
    assignment = io.read(verify(cal['assignment']))
    models = {k: io.read(verify(v)) for k, v in cal['models'].items()}
    sources = {k: io.read(verify(v['generation'])) for k, v in launch['structural_sources'].items()}
    rna = io.read(verify(sources['rna']['context']))
    banatic = io.read(verify(sources['banatic']['context']))
    supplier = io.read(verify(sources['supplier']['extras']))
    address = io.read(verify(sources['address']['extras']))
    cases = []
    for outcome in remaining:
        guard()
        qid = outcome['query_id']
        fact, num, aliases = facts[qid], numeric[qid], portfolio[qid]
        row = dict(query_id=qid, query=fact['crm'], outcome=outcome,
            protected_no_reconstruction=fact['protected_no_reconstruction'],
            label_unchanged=True, no_target_injection=True, classification_not_retrieval_input=True)
        if fact['protected_no_reconstruction']:
            row.update(primary_category='PROTECTED_SOURCE', diagnostic_layer='PUBLIC_INFORMATION_CONSTRAINT_NOT_PROVEN_IMPOSSIBLE',
                       no_target_profile_or_features_reconstructed=True)
            cases.append(row)
            continue
        target = int(fact['label_siret'])
        owner = fact['label_siret'][:9]
        actual_public = next((r for r in fact['same_owner_public_sites_in_query_zone'] if r['siret'] == fact['label_siret']), None)
        if actual_public is None:
            raise ValueError('Already-sealed both-public corporate target record required')
        q = fact['crm']
        admin = bool((q['crm_insee'] and actual_public['insee'] == q['crm_insee']) or
                     (q['crm_postcode'] and actual_public['postcode'] == q['crm_postcode']))
        coefficient = float(models[str(assignment[qid])]['coefficients'][6])
        leaf = dict(own=sparse(supplier[qid]['own'], target), temporal=sparse(supplier[qid]['temporal'], target),
                    continuity_address=sparse(address[qid], target))
        max_conf = max(v['maximum_confidence'] for v in leaf.values())
        target_conf = max(v['target_confidence'] for v in leaf.values())
        leaf_score, bound = coefficient*target_conf, coefficient*max_conf
        effective_leaf = bound*np.floor(np.clip(leaf_score/bound, 0, 1)*BITS)/BITS if bound else 0.
        owner_evidence = {name: [r for r in context[qid] if r['siren_ref'] == owner]
                          for name, context in [('rna', rna), ('banatic', banatic)]}
        owner_conf = max((r['confidence'] for rows in owner_evidence.values() for r in rows), default=0.)
        owner_score = coefficient*np.floor(owner_conf*BITS)/BITS if admin else 0.
        added = float(effective_leaf+owner_score)
        available = bool(rna[qid] or banatic[qid] or any(v['available'] for v in leaf.values()))
        covered = bool(target_conf > 0 or owner_conf > 0)
        if outcome['siren']:
            category, layer = 'SITE_VS_PARENT', 'SITE_REPRESENTATION_OR_ADMISSION_NOT_OWNER_DISCOVERY'
        elif added > 0:
            category, layer = 'POSITIVE_TARGET_SOURCE_SCORE_INSUFFICIENT', 'SOURCE_PRESENT_ADMISSION_REMAINS_MISS'
        elif covered:
            category, layer = 'TARGET_COVERED_BUT_GATE_OR_QUANTIZATION_ZERO', 'REPRESENTATION_GATE_OR_TINY_SCORE'
        elif available:
            category, layer = 'QUERY_SOURCE_EXISTS_TARGET_NOT_COVERED', 'DISCRIMINATING_TARGET_BRIDGE_ABSENT_IN_THESE_ADMITTED_SOURCES'
        else:
            category, layer = 'NEW_SOURCE_ABSENT_FOR_QUERY', 'NEW_FOUR_SOURCE_COVERAGE_LIMIT_NOT_ALL_OPEN_DATA_ABSENCE'
        prior = fact['previous_ledger']
        numeric_score = num['target_numeric_score']
        cut = outputs[qid][-1]['score'] if outputs[qid] else None
        word_score = prior['indexed_word_name_score']
        row.update(primary_category=category, diagnostic_layer=layer, label_siret=fact['label_siret'],
            public_target=actual_public, direct_official_succession_edges=fact['direct_succession_edges'],
            target_in_original_numeric_universe=num['target_in_numeric_universe'],
            source_query_available=available, source_target_covered=covered,
            source_leaf=leaf, source_owner_records=owner_evidence, source_admin_gate=admin,
            source_target_scores=dict(coefficient6=coefficient, unquantized_leaf=leaf_score,
                quantized_leaf=float(effective_leaf), quantized_owner_if_gate=float(owner_score), total_added=added),
            score_accounting=dict(baseline_target_numeric_unquantized=numeric_score,
                primary_native_cut100=cut, numeric_plus_new_source=numeric_score+added,
                cut_minus_numeric_plus_new_source=cut-numeric_score-added if cut is not None else None,
                target_full_native_score=None, target_native_rank=None,
                missing_components='BASELINE_LEXICAL_ADDRESS_PARENT_ACTOR_AND_SOURCE_CONTEXT_NOT_FULLY_DECOMPOSED',
                numeric_gap_not_a_full_native_gap_or_counterfactual_rank=True),
            existing_information=dict(compiled_official_own_alias_counts=aliases['raw_published_legal_and_own_site_alias_counts'],
                compiled_alias_query_word_overlap=aliases['overlapping_query_words'], indexed_word_name_field_max=word_score,
                dictionary_not_all_raw_RNE_BODACC_HISTORY_payloads=True,
                bodacc_announcements=prior['bodacc_announcements'], bodacc_event_passages=prior['bodacc_event_passages'],
                bodacc_nonadmin_nonname_literal_matches=prior['bodacc_nonadmin_nonname_literal_matches'],
                sector_evidence=prior['sector_evidence'], label_error_proven=prior['label_error_proven']),
            target_numeric_proofs=num['target_numeric_features'],
            same_owner_retrieved=[dict(rank=i+1, siret=r['siret'], score=r['score'])
                                  for i, r in enumerate(outputs[qid]) if r['siren'] == owner],
            operational_separate=outcome['operational'], current_addresses_not_full_history=True,
            no_claim_all_possible_official_or_open_data_missing=True)
        cases.append(row)
        gc.collect()
    counts = dict(Counter(r['primary_category'] for r in cases))
    public = [r for r in cases if not r['protected_no_reconstruction']]
    summary = dict(categories=counts, remaining20=len(cases), public_targets=len(public),
        protected_targets=len(cases)-len(public), source_target_covered=sum(r['source_target_covered'] for r in public),
        effective_positive_added_target_score=sum(r['source_target_scores']['total_added'] > 0 for r in public),
        all_public_targets_still_present_original_numeric_universe=all(r['target_in_original_numeric_universe'] for r in public),
        parent_already_found=sum(r['outcome']['siren'] for r in cases), operational_already_found=sum(r['outcome']['operational'] for r in cases),
        public_zero_indexed_word_name=sum(r['existing_information']['indexed_word_name_field_max'] == 0 for r in public),
        public_with_bodacc_records=sum(bool(r['existing_information']['bodacc_announcements']) for r in public),
        proved_human_label_errors=sum(r['existing_information']['label_error_proven'] for r in public),
        target_full_native_scores_not_available_without_forbidden_new_search=True)
    OUT.mkdir(parents=True)
    io.write(OUT/'cases.json', cases)
    io.write(OUT/'report.json', dict(status='COMPLETE_ALL20_POSTSEAL_INFORMATION_REPRESENTATION_SCORE_AUDIT',
        implementation=record(Path(__file__)), primary_generation=record(primary/'generation.json'),
        primary_actual_exit=record(primary/'generate_exit.json'), parent_actual_exit=record(ROOT/'owned_tool_exit.json'),
        public_label_side_facts=record(FACTS/'report.json'), original_numeric_diagnosis=record(NUMERIC/'report.json'),
        compiled_portfolio_audit=record(PORTFOLIO/'report.json'), cases=record(OUT/'cases.json'), summary=summary,
        categories_exhaustive_mutually_exclusive_but_mechanisms_can_overlap=True,
        no_native_query_no_index_scan_no_new_sources=True, no_scoring_or_alias_or_label_change=True,
        exact_all526_denominator_preserved=True, closed_sets_opened=False, full_encoder_oof=False,
        consumed_development_diagnosis_not_independent_validation=True, observed_unix=time.time()))
    print(summary, flush=True)


if __name__ == '__main__':
    main()
