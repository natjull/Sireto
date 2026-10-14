"""Postseal BODACC activity mechanism audit; no search or score modification.

All526 query-only activity shards, all public sealed output100 rows, and the
20 remaining consumed-panel misses are distinct diagnostic populations. This
does NOT scan the25GiB numerical universe or claim its output100 distribution
describes all national distractors. Protected profiles are never reconstructed.
"""
from collections import Counter
from pathlib import Path
import gc
import math
import time

import numpy as np
from scripts import embeddinggemma2_night as io
from src.xgb_matcher.retrieval_v20_workbench import record, verify

BASE = io.HOME/'v115/streamed_activity_ablation_20261013_a'
VOTES = io.HOME/'v131/source_vote_overlap_20261013_night'
REMAIN = io.HOME/'v140/remaining_structure_20261014_night'
XYAUDIT = io.HOME/'v140/remaining_xy_20261014_night'
DOSSIERS = io.HOME/'v52/postseal_bodacc_fulltext_20261012_a'
OUT = io.HOME/'v151/bodacc_context_geo_dependence_20261014_night'
BITS = (1 << 20)-1


def guard():
    import psutil
    free, level = io.pressure()
    if psutil.Process().memory_info().rss > 1024**3 or free < 25 or level not in (0, 1, 2):
        raise RuntimeError('Light read-only audit yields; no active recipe touched')


def main():
    started = time.monotonic()
    guard()
    if OUT.exists():
        raise ValueError('Fresh diagnostic namespace')
    gen, ex = io.read(BASE/'generation.json'), io.read(BASE/'generate_exit.json')
    if gen['status'] != 'COMPLETE_SEALED_ALL526' or ex['exit_code'] or not ex['own_child_exit_confirmed']:
        raise ValueError('Whole sealed baseline with actual owned exit required')
    launch = io.read(BASE/'launch.json')
    panel = io.read(verify(launch['panel']))
    if len(panel) != 526 or any(q['fold'] not in (2, 3, 4) for q in panel):
        raise ValueError('Exact526 development-only inputs')
    fg = io.read(BASE/'features_generation.json')
    source = io.read(verify(fg['activity_generation']))
    source_exit = io.read(verify(fg['activity_exit']))
    if source_exit['exit_code'] or not source_exit['own_tool_exit_confirmed'] or source['query_panel'] != launch['panel']:
        raise ValueError('Actual sealed activity source authority')
    # Global input-only pass FIRST: no target references select source rows.
    input_rows = []
    for q in panel:
        guard()
        qid = q['query_id']
        with np.load(verify(fg['activity_shards'][qid]), allow_pickle=False) as f:
            refs, confidence = f['refs'], f['confidence']
            if len(refs) != len(confidence) or (confidence < 0).any() or (confidence > .500001).any():
                raise ValueError('Unchanged bounded source')
            input_rows.append(dict(query_id=qid, fold=q['fold'], source_owner_references=len(refs),
                positive_confidences=int((confidence > 0).sum()),
                positive_quantized=int((np.floor(confidence*BITS) > 0).sum()),
                maximum_confidence=float(confidence.max(initial=0.)),
                quantization_zero=int(((confidence > 0) & (np.floor(confidence*BITS) == 0)).sum())))
        # Do not keep12.6M owner refs in memory. Source target lookups later
        # reopen just the corresponding already-sealed query shard.
    vr = io.read(VOTES/'report.json')
    votes = io.read(verify(vr['votes']))
    admin_rows = io.read(verify(vr['administrative']))
    admin = {r['siret']:(r['insee'], r['postcode']) for r in admin_rows}
    if vr['baseline_generation'] != record(BASE/'generation.json') or len(votes) != 526:
        raise ValueError('Exact unchanged public output population')
    geometry = {r['query_id']:r['geometry'] for r in io.read(verify(launch['geometry']))}
    keys = np.load(verify(launch['keys']), mmap_mode='r')
    xy = np.load(verify(launch['xy']), mmap_mode='r')
    queries = {q['query_id']:q for q in panel}
    output_rows = []
    for item in votes:
        guard()
        qid = item['query_id']; q = queries[qid]; g = geometry[qid]
        for row in item['rows']:
            s = row['siret']; key = np.uint64(s)
            pos = int(np.searchsorted(keys, key))
            present = pos < len(keys) and keys[pos] == key
            point = xy[pos] if present else np.array([np.nan, np.nan])
            usable = bool(np.isfinite(point).all() and 0 <= point[0] <= 1300000 and 6000000 <= point[1] <= 7300000)
            i, p = admin[s]
            gate = bool((q['crm_insee'] and q['crm_insee'] == i) or (q['crm_postcode'] and q['crm_postcode'] == p))
            own_geo = 0.
            if usable and g['spatial_crs_compatible']:
                distance = float(np.hypot(point[0]-g['ban_x_l93'], point[1]-g['ban_y_l93']))
                own_geo = (.5 if g['ban_declared_weak'] else 1.)/(1+distance/(300+max(0.,g['ban_uncertainty_m'])))
            activity = row['sources']['activity']
            output_rows.append(dict(query_id=qid, rank=row['rank'], siret=s,
                both_public_from_prior_sealed_admin_audit=True, admin_gate=gate,
                own_xy_usable=usable, original_own_geo_recomputed=float(own_geo),
                activity_confidence_after_admin_gate=activity,
                activity_bonus=row['frozen_context_coefficient']*math.floor(activity*BITS)/BITS,
                geo_is_not_an_activity_gate=True))
    del votes, admin_rows, admin, keys, xy
    gc.collect(); guard()
    # ONLY after source-wide and all526 output passes: consumed label-side.
    rr = io.read(REMAIN/'report.json')
    remaining = io.read(verify(rr['cases']))
    xy_facts = {r['query_id']:r for r in io.read(verify(io.read(XYAUDIT/'report.json')['cases']))}
    ds = io.read(DOSSIERS/'summary.json')
    dossiers = {r['query_id']:r for r in io.read(verify(ds['cases']))}
    cal = io.read(BASE/'calibration/generation.json')
    assignment = io.read(verify(cal['assignment']))
    models = {k:io.read(verify(v)) for k,v in cal['models'].items()}
    cases = []
    for row in remaining:
        qid = row['query_id']
        case = dict(query_id=qid, query=row['query'], protected_skip=row['protected_no_reconstruction'],
                    no_label_change=True, source_membership_not_retrieval_input=True)
        if row['protected_no_reconstruction']:
            cases.append(case); continue
        owner = row['label_siret'][:9]
        with np.load(verify(fg['activity_shards'][qid]), allow_pickle=False) as f:
            loc = np.flatnonzero(f['refs'] == owner.encode())
            confidence = float(f['confidence'][loc[0]]) if len(loc) else 0.
        dossier = dossiers[qid]
        matches = [m for m in dossier['literal_matches'] if m['field'] == 'establishment.activite']
        c = float(models[str(assignment[qid])]['coefficients'][6])
        gate = row['source_admin_gate']
        case.update(label_siret=row['label_siret'], announcements=dossier['announcements'],
            actual_activity_literal_matches=matches, source_strict_activity_owner_covered=confidence > 0,
            source_confidence_before_admin_gate=confidence, admin_gate=gate,
            unquantized_activity_bonus=c*confidence,
            effective_activity_bonus=c*math.floor(confidence*BITS)/BITS if gate else 0.,
            own_xy_usable=xy_facts[qid]['original_xy_usable_metropolitan_l93'],
            frozen_own_geo5=row['target_numeric_proofs']['own_geo_quality'],
            frozen_filled_geo42=row['target_numeric_proofs']['filled_own_site_geo'],
            activity_zero_due_to_geo=False,
            activity_zero_due_to_admin=bool(confidence > 0 and not gate),
            activity_field_not_name=True, activity_parent_union_not_own_site_identity=True,
            dossier_join_role='STRUCTURED_SUBJECT_SIREN_NOT_REGISTER_STRING_ALONE',
            dossier_multi_establishment_attribution_not_certified=True,
            source_join_role='STRICT_SINGLE_CORPORATE_SUBJECT_SINGLE_ESTABLISHMENT_PUBLIC_UL',
            dates_publication_at_or_before='2026-08-01', effective_date_not_inferred=True,
            target_full_native_score_not_available=True)
        cases.append(case)
    positive = [r for r in output_rows if r['activity_confidence_after_admin_gate'] > 0]
    public = [r for r in cases if not r['protected_skip']]
    coverage = source['coverage']
    summary = dict(query_only_source_queries=526, queries_with_activity_context=sum(r['source_owner_references'] > 0 for r in input_rows),
        source_owner_references=sum(r['source_owner_references'] for r in input_rows),
        source_positive_quantized=sum(r['positive_quantized'] for r in input_rows),
        source_quantization_zero=sum(r['quantization_zero'] for r in input_rows),
        all526_public_sealed_output_rows=len(output_rows), activity_positive_admin_exact_output_rows=len(positive),
        activity_positive_output_rows_missing_own_xy=sum(not r['own_xy_usable'] for r in positive),
        activity_positive_output_rows_original_geo_zero=sum(r['original_own_geo_recomputed'] == 0 for r in positive),
        activity_positive_output_queries=len({r['query_id'] for r in positive}),
        remaining_exact_misses=len(cases), public_remaining=len(public), protected_remaining=len(cases)-len(public),
        public_remaining_with_announcements=sum(r['announcements'] > 0 for r in public),
        public_remaining_with_literal_activity_words=sum(bool(r['actual_activity_literal_matches']) for r in public),
        public_remaining_with_admitted_activity_owner=sum(r['source_strict_activity_owner_covered'] for r in public),
        public_remaining_with_positive_activity_bonus=sum(r['effective_activity_bonus'] > 0 for r in public),
        public_remaining_activity_geo_annihilated=0,
        before_cutoff_announcement_rows=sum(r['before_cutoff'] for r in coverage),
        strict_single_object_rows=sum(r['strict_single_objects'] for r in coverage),
        eligible_public_corporate_activity_rows=sum(r['eligible_public_activity_rows'] for r in coverage))
    OUT.mkdir(parents=True)
    io.write(OUT/'query_activity_source_counts.json', input_rows)
    io.write(OUT/'public_output_activity_geometry.json', output_rows)
    io.write(OUT/'remaining_activity_cases.json', cases)
    io.write(OUT/'report.json', dict(status='COMPLETE_READONLY_BODACC_ACTIVITY_GEO_DEPENDENCE_AUDIT',
        implementation=record(Path(__file__)), source_generation=fg['activity_generation'], source_actual_exit=fg['activity_exit'],
        source_shards_binding=record(BASE/'features_generation.json'), baseline_generation=record(BASE/'generation.json'),
        baseline_actual_exit=record(BASE/'generate_exit.json'), public_output_audit=record(VOTES/'report.json'),
        postseal_remaining=record(REMAIN/'report.json'), official_text_dossiers=record(DOSSIERS/'summary.json'),
        inspected_scoring_helpers=[record(Path('scripts')/n) for n in ['embeddinggemma2_v112_bodacc_activity_context.py','embeddinggemma2_v114_native_activity_ablation.py','embeddinggemma2_v115_streamed_activity_ablation.py','embeddinggemma2_v23_native_features.py']],
        source_counts=record(OUT/'query_activity_source_counts.json'), public_outputs=record(OUT/'public_output_activity_geometry.json'),
        remaining_cases=record(OUT/'remaining_activity_cases.json'), summary=summary,
        exact_score_path='COEFFICIENT6_TIMES_FLOOR_SOURCE_CONFIDENCE_20BIT_ADMIN_INSEE_OR_POSTCODE_GATE_NO_XY',
        filled_geo42_not_read_by_activity_channel=True, no_proxy_or_geo_fallback_in_activity_channel=True,
        scope_not_all_numeric_or_national_candidate_rows=True, numerical_universe25GiB_not_scanned=True,
        raw_multiactor_sources_not_reparsed=True, no_activity_alias_created=True,
        no_new_recall_or_counterfactual_search=True, no_recipe_fit_or_label_change=True,
        closed_sets_opened=False, development_consumed_not_encoder_full_oof=True,
        parser_exclusion_not_proof_all_raw_source_empty=True,
        direction='AUDIT_TYPED_ACTIVITY_COVERAGE_AND_ROLE_SAFE_MULTI_ACTOR_PARSER_NOT_RELAX_NONEXISTENT_XY_GATE',
        elapsed_seconds=time.monotonic()-started, observed_unix=time.time()))
    print(summary, flush=True)


if __name__ == '__main__':
    main()
