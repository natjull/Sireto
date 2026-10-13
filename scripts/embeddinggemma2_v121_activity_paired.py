"""Postseal paired V99/V115 diagnostic; never a retrieval or model selection run."""
from pathlib import Path
import math
import time

from scripts import embeddinggemma2_night as io
from scripts.embeddinggemma2_v102_final_paired import paired_uncertainty
from src.xgb_matcher.retrieval_v20_workbench import record, verify

OUTPUT = io.HOME / 'v121/activity_paired_20261013_a'
ROOTS = {
    'v99': io.HOME / 'v99/initial_alignment_native_20261013_a',
    'v115': io.HOME / 'v115/streamed_activity_ablation_20261013_a',
}


def wilson(hits, n):
    z = 1.959963984540054
    p = hits / n
    den = 1 + z*z/n
    center = (p + z*z/(2*n))/den
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return [center-half, center+half]


def main():
    if OUTPUT.exists():
        raise ValueError('Fresh postseal comparison only')
    refs, launches = {}, {}
    for name, root in ROOTS.items():
        gen = io.read(root / 'generation.json')
        ex = io.read(root / 'generate_exit.json')
        if gen['status'] != 'COMPLETE_SEALED_ALL526' or gen['queries'] != 526:
            raise ValueError('All526 sealed generation required')
        if ex['exit_code'] or not ex['own_child_exit_confirmed']:
            raise ValueError('Owned complete generation required')
        verify(gen['outputs'])
        refs[name] = {k: record(root / (k + '.json')) for k in
                      ('generation', 'generate_exit', 'report', 'outcomes', 'launch')}
        launches[name] = io.read(root / 'launch.json')
    for key in ('panel', 'frozen_models', 'frozen_calibration', 'index_manifest', 'index_meta'):
        if launches['v99'][key] != launches['v115'][key]:
            raise ValueError(f'Frozen shared scientific binding differs: {key}')
    # Labels are read only after BOTH complete generations and their exits.
    rows = {name: {r['query_id']: r for r in
                  io.read(verify(ref['outcomes']))['learned_native_ooc']}
            for name, ref in refs.items()}
    scope = set(rows['v99'])
    if len(scope) != 526 or set(rows['v115']) != scope:
        raise ValueError('Identical526 denominator')
    summaries = {}
    for name, units in rows.items():
        total_weight = sum(r['calibrated_query_weight'] for r in units.values())
        summaries[name] = {
            'queries': len(units),
            'siren_groups': len({r['siren_group'] for r in units.values()}),
            'metrics': {field: {
                'hits': sum(r[field] for r in units.values()),
                'raw_recall': sum(r[field] for r in units.values()) / len(units),
                'weighted_recall': sum(r[field]*r['calibrated_query_weight']
                                       for r in units.values()) / total_weight,
            } for field in ('exact', 'siren', 'operational')},
            'raw_exact_wilson95': wilson(sum(r['exact'] for r in units.values()), len(units)),
        }
    paired = []
    for qid in sorted(scope):
        a, b = rows['v99'][qid], rows['v115'][qid]
        if any(a[k] != b[k] for k in ('zone', 'siren_group', 'fold', 'calibrated_query_weight')):
            raise ValueError('Shared paired-unit authority')
        paired.append(dict(a, difference=int(b['exact'])-int(a['exact'])))
    source = io.read(verify(launches['v99']['source_launch']))
    sampling = io.read(verify(source['sampling']))
    gains = [r for r in paired if r['difference'] == 1]
    losses = [r for r in paired if r['difference'] == -1]
    OUTPUT.mkdir(parents=True)
    io.write(OUTPUT / 'report.json', dict(
        status='COMPLETE_POSTSEAL_PAIRED_V99_V115_ACTIVITY_DIAGNOSTIC',
        implementation=record(Path(__file__)), sources=refs, summaries=summaries,
        exact_gains=[r['query_id'] for r in gains], exact_losses=[r['query_id'] for r in losses],
        gain_siren_groups=sorted({r['siren_group'] for r in gains}),
        loss_siren_groups=sorted({r['siren_group'] for r in losses}),
        paired_weighted=paired_uncertainty(paired, sampling),
        no_new_predictions_or_training=True, no_closed_dataset_opened=True,
        consumed_development_not_confirmatory=True, production_promotion_allowed=False,
        no_99_2_certificate=True, observed_unix=time.time()))
    print(summaries, flush=True)


if __name__ == '__main__':
    main()
