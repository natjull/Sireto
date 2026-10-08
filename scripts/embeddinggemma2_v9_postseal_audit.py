"""Label-side diagnostic ONLY. Never generates or changes retrieval candidates."""
import argparse
from pathlib import Path
from scripts import embeddinggemma2_night as io
from src.xgb_matcher.retrieval_v20_workbench import record, verify

OUTPUT = io.HOME / 'v9'


def rank(candidates, target):
    return next((i + 1 for i, row in enumerate(candidates) if row['siret'] == target), None)


def changes(old, new):
    if set(old) != set(new): raise ValueError('Identical denominators required')
    return dict(gained=sorted(q for q in old if new[q] and not old[q]),
        lost=sorted(q for q in old if old[q] and not new[q]))


def audit(source, out):
    state = io.read(source / 'state.json')
    if state['status'] != 'COMPLETE_MEASURED_TWO_PIPELINES': raise ValueError('Complete sealed evaluation required')
    report = io.read(verify(state['paired'])); geo_report = io.read(verify(state['geo']))
    launch = io.read(source / 'launch.json')
    for ref in launch['implementation'].values(): verify(ref)
    for name in ('v1_native160', 'v3_official512'):
        exited = io.read(source / (name + '_exit.json'))
        if exited['exit_code'] != 0 or not exited['own_child_exit_confirmed']: raise ValueError('Owned complete generation')
    channels = io.read(verify(report['all_outputs']))['candidates']
    geo_generation = io.read(source / 'geo/generation.json')
    channels['geo_only'] = {r['query_id']: r['candidates'] for r in io.read(verify(geo_generation['outputs']))}
    outcomes = io.read(source / 'outcomes.json'); geo_outcomes = io.read(source / 'geo/outcomes.json')
    old = {r['query_id']: r['exact'] for r in outcomes['v1_native160']}
    new = {r['query_id']: r['exact'] for r in outcomes['v3_official512']}
    final = {r['query_id']: r['exact'] for r in geo_outcomes['dense_lexical_geo_rrf']}
    if set(old) != set(new) or set(old) != set(final): raise ValueError('All queries retained')
    panel = {r['query_id']: r for r in io.read(verify(launch['panel']))}
    manifest = io.read(verify(launch['training_manifest']))
    membership = io.HOME / 'night_20261006_a/validation_membership.json'
    if io.digest(membership) != manifest['split_hashes']['validation']: raise ValueError('Label membership binding')
    labels = {r['query_id']: r for r in io.read(membership) if r['query_id'] in old}
    if set(labels) != set(old) or any(r['fold'] not in (2, 3, 4) for r in labels.values()): raise ValueError('Safe existing validation only')
    changed = changes(old, new)
    selected = sorted(set(changed['gained'] + changed['lost'] + [q for q in old if not old[q] or not new[q] or not final[q]]))
    corpus = {}
    for zone in launch['corpora']:
        if any(panel[q]['crm_insee'] == zone['zone'] for q in selected):
            corpus[zone['zone']] = {d['key']: d for d in io.read(verify(zone['file']))}
    geometry = {r['query_id']: r for r in io.read(verify(geo_generation['outputs']))}
    evidence = []
    for q in selected:
        gt = labels[q]; crm = panel[q]; zone = corpus[crm['crm_insee']]
        doc = zone.get(gt['siret'])
        ranks = {name: rank(values[q], gt['siret']) for name, values in channels.items()}
        top = channels['v3_official512'][q][:3]
        evidence.append(dict(query=crm, human_siret=gt['siret'], human_siren=gt['siren'],
            human_target_present_in_complete_commune=doc is not None,
            official_target=None if doc is None else {k: doc[k] for k in ('key', 'siren', 'own_names', 'all_addresses', 'state', 'site_views', 'related_parents')},
            exact_ranks=ranks, geometry=geometry[q]['geometry'],
            new_top3=[dict(candidate=c, official=zone.get(c['siret'])) for c in top],
            final_exact=final[q]))
    oracle = sum(any(rank(channels[name][q], labels[q]['siret']) is not None
        for name in ('v1_native160', 'v3_official512', 'national_lexical', 'geo_only')) for q in old)
    maximum_union = max(len({c['siret'] for name in ('v1_native160', 'v3_official512', 'national_lexical', 'geo_only')
        for c in channels[name][q]}) for q in old)
    out.mkdir(parents=True, exist_ok=False)
    io.write(out / 'audit.json', dict(status='POSTSEALED_DIAGNOSTIC_NEVER_A_RETRIEVAL_RECIPE',
        source_state=record(source / 'state.json'), paired_report=state['paired'], geo_report=state['geo'],
        source_outcomes=record(source / 'outcomes.json'), audit_implementation=record(Path(__file__)),
        queries=len(old), dense_gains_losses=changed, evidence=evidence,
        uncapped_four_channel_union_exact_hits=oracle, uncapped_four_channel_max_sirets=maximum_union,
        union_is_diagnostic_oracle_not_initial100=True, source_final100_exact_hits=sum(final.values()),
        no_new_candidate_generation=True, no_recipe_adjustment=True, no_label_correction=True,
        isolated_fine_tuning_effect_measured=False, national_99percent_validated=False,
        test_opened=False, fold0_opened=False, fold1_opened=False, downstream_models_changed=False,
        hypotheses_not_validated=[
            'Preserve complementary old/new dense evidence without static channel quotas; evaluate an independently frozen policy on unseen validation groups.',
            'Represent missing-number addresses explicitly with official street-level BAN geometry/uncertainty, never invent a precise house coordinate.',
            'Validate any new initial100 fusion on a larger geographically diverse SIREN-disjoint scope, not the postsealed80.',
            'Run the omitted official-text V1 ablation before attributing the measured gain specifically to fine-tuning.']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--source', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(OUTPUT): raise ValueError('Dedicated label-side audit namespace')
    audit(args.source, args.output)
