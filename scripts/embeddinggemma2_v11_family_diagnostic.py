"""Once-only post-hoc family fusion replay. Not new representative evidence."""
import argparse
from pathlib import Path
import subprocess
import sys

from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v10_representative as family
from scripts import embeddinggemma2_v3_compare as paired
from src.xgb_matcher.retrieval_v20_workbench import record, verify

OUTPUT = io.HOME / 'v11'


def generate(out):
    launch = io.read(out / 'launch.json')
    for ref in launch['implementation']: verify(ref)
    source = Path(verify(launch['source_state'])).parent
    state = io.read(verify(launch['source_state']))
    if state['status'] != 'COMPLETE_MEASURED_TWO_PIPELINES': raise ValueError('Complete source only')
    report = io.read(verify(state['paired']))
    channels = io.read(verify(report['all_outputs']))['candidates']
    geo_generation = io.read(source / 'geo/generation.json')
    for variant in ('v1_native160', 'v3_official512'):
        ex = io.read(source / (variant + '_exit.json'))
        if ex['exit_code'] != 0 or not ex['own_child_exit_confirmed']: raise ValueError('Owned source exit')
    ex = io.read(source / 'geo/generation_exit.json')
    if ex['exit_code'] != 0 or not ex['own_child_exit_confirmed']: raise ValueError('Owned geo exit')
    geo = {r['query_id']: r for r in io.read(verify(geo_generation['outputs']))}
    scope = set(channels['national_lexical'])
    if len(scope) != 80 or set(geo) != scope: raise ValueError('Original complete cohort')
    rows = [dict(query_id=q, candidates=family.family_fusion(channels['v1_native160'][q],
        channels['v3_official512'][q], channels['national_lexical'][q],
        geo[q]['candidates'], geo[q]['rrf_geo_weight'])) for q in sorted(scope)]
    io.write(out / 'outputs.json', rows)
    io.write(out / 'generation.json', dict(outputs=record(out / 'outputs.json'), queries=80,
        status='SEALED_POSTHOC80_FAMILY_INITIAL100', truth_opened_by_generation=False,
        outcomes_previously_known=True, independent_validation=False))


def evaluate(out):
    ex = io.read(out / 'owned_exit.json')
    if ex['exit_code'] != 0 or not ex['own_child_exit_confirmed']: raise ValueError('Complete child before labels')
    gen = io.read(out / 'generation.json')
    rows = io.read(verify(gen['outputs'])); scope = {r['query_id'] for r in rows}
    manifest = io.read(family.SOURCE / 'manifest.json')
    membership = family.SOURCE / 'validation_membership.json'
    if io.digest(membership) != manifest['split_hashes']['validation']: raise ValueError('Label membership drift')
    labels = {r['query_id']: r for r in io.read(membership) if r['query_id'] in scope}
    previous = {r['query_id']: r for r in io.read(family.PREVIOUS / 'geo/outcomes.json')['dense_lexical_geo_rrf']}
    if len(scope) != 80 or set(labels) != scope or set(previous) != scope: raise ValueError('Same denominator')
    cfg = io.read(io.ROOT / 'config/retrieval_v358_postseal_evaluation.json')
    ev = io.read(verify(cfg['evaluation'])); side = io.read(verify(ev['label_side']))
    if not side['never_used_in_retrieval'] or side['ledger'] != cfg['ledger']: raise ValueError('Sidecar authority')
    import pyarrow.parquet as pq
    accepted = {q: {labels[q]['siret']} for q in scope}
    for r in pq.read_table(verify(side['files']['acceptable_operational.parquet']),
            filters=[('fold', 'in', [2,3,4]), ('query_id', 'in', sorted(scope))], use_threads=False).to_pylist():
        accepted[r['query_id']].add(r['siret'])
    outcomes = []
    for row in rows:
        q = row['query_id']; cs = row['candidates']; keys = {c['siret'] for c in cs}
        if len(cs)>100 or len(cs)!=len(keys): raise ValueError('Final100 cap')
        outcomes.append(dict(query_id=q, exact=labels[q]['siret'] in keys,
            siren=labels[q]['siren'] in {c['siren'] for c in cs}, operational=bool(keys & accepted[q])))
    io.write(out / 'outcomes.json', outcomes)
    io.write(out / 'report.json', dict(status='POSTHOC80_ONLY_NOT_REPRESENTATIVE_VALIDATION',
        metrics=paired.summarize(outcomes), gained=[r['query_id'] for r in outcomes if r['exact'] and not previous[r['query_id']]['exact']],
        lost=[r['query_id'] for r in outcomes if not r['exact'] and previous[r['query_id']]['exact']],
        outcomes_previously_known=True, no_recipe_retuning=True, national_99percent_validated=False,
        test_opened=False, fold0_opened=False, fold1_opened=False, downstream_models_changed=False))


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    io.write(out / 'launch.json', dict(source_state=record(family.PREVIOUS / 'state.json'),
        implementation=[record(Path(__file__)), record(Path(family.__file__))], frozen_recipe_commit='2358da76'))
    command = [sys.executable, '-m', 'scripts.embeddinggemma2_v11_family_diagnostic', 'generate', '--output', str(out)]
    child = subprocess.run(command, check=False)
    io.write(out / 'owned_exit.json', dict(exit_code=child.returncode, own_child_exit_confirmed=True, command=command))
    child.check_returncode(); evaluate(out)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('action', choices=['run','generate']); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not args.output.resolve().is_relative_to(OUTPUT): raise ValueError('Dedicated V11 namespace')
    (run if args.action=='run' else generate)(args.output)
