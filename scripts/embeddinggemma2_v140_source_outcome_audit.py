"""All-six post-seal causal audit; no retrieval, recipe or label modification."""
from collections import defaultdict
from pathlib import Path
import time

import numpy as np
from scripts import embeddinggemma2_night as io
from src.xgb_matcher.retrieval_v20_workbench import record, verify

ROOT = io.HOME/'v140/structural_native_disk_20261013_night'
BASE = io.HOME/'v115/streamed_activity_ablation_20261013_a'
FACTS = io.HOME/'v122/label_clarification_20261013_b'
OUT = io.HOME/'v140/source_outcome_audit_20261013_night'


def sparse_value(ref, key):
    if ref is None:
        return 0.
    with np.load(verify(ref), allow_pickle=False) as f:
        at = int(np.searchsorted(f['keys'], key))
        return float(f['confidence'][at]) if at < len(f['keys']) and f['keys'][at] == key else 0.


def main():
    if OUT.exists():
        raise ValueError('Fresh post-seal diagnostic namespace')
    top = io.read(ROOT/'report.json')
    ex = io.read(ROOT/'owned_tool_exit.json')
    if top['status'] != 'COMPLETE_SIX_PREDECLARED_STRUCTURAL_ABLATIONS_ALL526' or ex['exit_code'] or not ex['own_tool_exit_confirmed']:
        raise ValueError('All six actual completed generations before labels/outcomes')
    for variant in top['variants']:
        gen, ex = io.read(ROOT/variant/'generation.json'), io.read(ROOT/variant/'generate_exit.json')
        if gen['status'] != 'COMPLETE_SEALED_ALL526' or ex['exit_code'] or not ex['own_child_exit_confirmed']:
            raise ValueError('Whole observed generation')
        verify(gen['outputs'])
    old = {r['query_id']: r for r in io.read(BASE/'outcomes.json')['learned_native_ooc']}
    fresh, comparisons = {}, {}
    for variant, result in top['variants'].items():
        rows = {r['query_id']: r for r in io.read(ROOT/variant/'outcomes.json')['learned_native_ooc']}
        if len(old) != 526 or set(rows) != set(old) or any(r['fold'] not in (2, 3, 4) for r in rows.values()):
            raise ValueError('Unchanged all526 safe denominator')
        if any(old[q][f] != rows[q][f] for q in old for f in ('fold', 'zone', 'siren_group', 'calibrated_query_weight')):
            raise ValueError('Same evaluation units and weights')
        paired = {m: dict(gains=[q for q in old if rows[q][m] and not old[q][m]],
                          losses=[q for q in old if old[q][m] and not rows[q][m]])
                  for m in ('exact', 'siren', 'operational')}
        segments = {}
        for field in ('fold', 'stratum', 'zone', 'role'):
            groups = defaultdict(lambda: dict(queries=0, baseline_exact_hits=0, new_exact_hits=0))
            for q, r in rows.items():
                segment = groups[str(r[field])]
                segment['queries'] += 1
                segment['baseline_exact_hits'] += int(old[q]['exact'])
                segment['new_exact_hits'] += int(r['exact'])
            for s in groups.values():
                s['delta_exact_hits'] = s['new_exact_hits']-s['baseline_exact_hits']
            segments[field] = dict(groups)
        comparisons[variant] = dict(raw=result['metrics']['raw_sample'],
            weighted={m: result['metrics']['weighted'][m]['poststratified_hajek_estimate'] for m in paired},
            paired=paired, all526_segments=segments)
        fresh[variant] = rows
    primary = top['primary_predeclared']
    altered = set(comparisons[primary]['paired']['exact']['gains']+comparisons[primary]['paired']['exact']['losses'])
    facts = {r['query_id']: r for r in io.read(verify(io.read(FACTS/'report.json')['cases']))}
    launch = io.read(ROOT/'launch.json')
    cal = io.read(ROOT/'calibration/generation.json')
    assignment = io.read(verify(cal['assignment']))
    models = {k: io.read(verify(v)) for k, v in cal['models'].items()}
    controls = set(io.read(verify(cal['encoder_control_queries'])))
    source_generations = {k: io.read(verify(v['generation'])) for k, v in launch['structural_sources'].items()}
    rna = io.read(verify(source_generations['rna']['context']))
    banatic = io.read(verify(source_generations['banatic']['context']))
    supplier = io.read(verify(source_generations['supplier']['extras']))
    address = io.read(verify(source_generations['address']['extras']))
    outputs = {v: io.read(verify(io.read(ROOT/v/'generation.json')['outputs']))['learned_native_ooc'] for v in top['variants']}
    cases = []
    for qid in sorted(altered):
        model = models[str(assignment[qid])]
        row = dict(query_id=qid, baseline=old[qid], primary=fresh[primary][qid],
            calibrator_fold=assignment[qid], calibrator_query_heldout=qid in model['heldout_queries'],
            calibrator_query_not_train=qid not in model['train_queries'],
            encoder_checkpoint_control_query=qid in controls, full_encoder_oof=False,
            prior_development_consumed=True, no_label_correction=True, independent_validation=False)
        fact = facts.get(qid)
        if fact is None or fact['protected_no_reconstruction']:
            row['public_target_profile_not_reconstructed'] = True
            cases.append(row)
            continue
        target, owner = fact['label_siret'], fact['label_siret'][:9]
        row.update(query=fact['crm'], label_siret=target,
            baseline_documented_target=fact['previous_ledger']['public_target'],
            original_label_error_proven=fact['previous_ledger']['label_error_proven'],
            direct_official_economic_links=fact['direct_succession_edges'])
        w = float(model['coefficients'][6])
        own = sparse_value(supplier[qid]['own'], int(target))
        temporal = sparse_value(supplier[qid]['temporal'], int(target))
        continuity = sparse_value(address[qid], int(target))
        owner_records = {name: [r for r in context[qid] if r['siren_ref'] == owner]
                         for name, context in [('rna', rna), ('banatic', banatic)]}
        owner_confidence = max((r['confidence'] for records in owner_records.values() for r in records), default=0.)
        row['source_proof'] = dict(coefficient6=w, leaf_confidence=dict(own=own, temporal=temporal, continuity_address=continuity),
            owner_records=owner_records, unquantized_leaf_increment=w*max(own, temporal, continuity),
            owner_increment_if_admin_gate=w*owner_confidence,
            owner_gate_not_inferred_from_gt_for_generation=True,
            same_source_leaf_max_not_sum=True, contexts_not_identity_or_own_alias=True,
            published_supplier_review=source_generations['supplier']['published_review'],
            published_supplier_brands=source_generations['supplier']['source_brands'])
        row['final100_per_variant'] = {}
        for variant, values in outputs.items():
            hit = next(((rank+1, r['score']) for rank, r in enumerate(values[qid]) if r['siret'] == target), None)
            row['final100_per_variant'][variant] = dict(rank=hit[0], native_score=hit[1]) if hit else dict(rank=None)
        cases.append(row)
    OUT.mkdir(parents=True)
    io.write(OUT/'comparisons.json', comparisons)
    io.write(OUT/'primary_cases.json', cases)
    io.write(OUT/'report.json', dict(status='COMPLETE_POSTSEAL_ALL_SIX_PAIRED_SOURCE_OUTCOME_AUDIT',
        implementation=record(Path(__file__)), complete_comparison=record(ROOT/'report.json'),
        actual_parent_exit=record(ROOT/'owned_tool_exit.json'), comparisons=record(OUT/'comparisons.json'),
        primary_cases=record(OUT/'primary_cases.json'), label_side_authority=record(FACTS/'report.json'),
        primary_predeclared=primary, primary_exact_gains=len(comparisons[primary]['paired']['exact']['gains']),
        primary_exact_losses=len(comparisons[primary]['paired']['exact']['losses']),
        no_new_retrieval_or_recipe=True, no_target_injection=True, no_label_change=True, all526_denominator=True,
        source_presence_score_increment_and_admission_distinct=True, full_encoder_oof=False,
        independent_validation=False, no_national99_certification=True, observed_unix=time.time()))
    print('POSTSEAL_PAIRED_SOURCE_AUDIT_COMPLETE', str(OUT), flush=True)


if __name__ == '__main__':
    main()
