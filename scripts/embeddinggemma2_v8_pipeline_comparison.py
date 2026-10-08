"""Two complete pipelines, original batch8 encoder, NO isolated fit attribution."""
import argparse
from collections import defaultdict
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v4_compare as paired
from scripts import embeddinggemma2_v4_fit as fit
from src.xgb_matcher.retrieval_v20_workbench import record, verify

OUTPUT = io.HOME / 'v8'
VARIANTS = ('v1_native160', 'v3_official512')


def generate_site_channel(out):
    """Exact max SITE-view cosine over ALL sites, not a fixed candidate pool."""
    import torch
    launch = io.read(out / 'launch.json'); paired.verify_launch(launch)
    cfg = io.read(verify(launch['night_config']))
    deadline = io.read(verify(launch['night']))['deadline_unix']
    adapter = record(Path(launch['requested_v3_adapter_path']))
    training_root = Path(adapter['path']).parent
    training = io.read(training_root / 'state.json')
    if training['status'] not in ('FIT_COMPLETE', 'EARLY_STOP', 'TIME_BUDGET_CHECKPOINTED'): raise ValueError('Actual finalized fit required')
    checkpoint = torch.load(verify(adapter), map_location='cpu', weights_only=True)
    if checkpoint['launch_sha256'] != io.digest(training_root / 'launch.json') or checkpoint['step'] > training['step']: raise ValueError('Best adapter binding')
    model, params, encode, inference, tokenizer = fit.load_model(cfg, verify(adapter))
    binding = dict(adapter=adapter, launch=record(out / 'launch.json'), step=checkpoint['step'],
        training_state=record(training_root / 'state.json'), score='MAX_SITE_VIEW_COSINE_NO_PARENT_BLEND')
    binding_path = out / 'v3_official512_binding.json'
    if binding_path.exists() and io.read(binding_path) != binding: raise ValueError('Resume binding')
    io.write(binding_path, binding)
    panel = io.read(verify(launch['panel'])); rows = []; encoded = 0; start = time.monotonic()
    for zone in launch['corpora']:
        path = out / ('site_only_' + zone['zone'] + '.json'); sealed = path.with_suffix('.seal.json')
        if sealed.exists(): rows.extend(io.read(verify(io.read(sealed)))); continue
        queries = [q for q in panel if q['crm_insee'] == zone['zone']]
        docs = io.read(verify(zone['file']))
        query_vectors = inference([io.QUERY_PROMPT + paired.old.query_text(q) for q in queries])
        texts = list(dict.fromkeys(io.DOC_PROMPT + v for d in docs for v in d['site_views']))
        vectors = {}; best = [[] for q in queries]
        for offset in range(0, len(texts), 128):
            fit.resource_guard(deadline, cfg['max_rss_bytes'])
            values = texts[offset:offset + 128]
            vectors.update(zip(values, inference(values))); encoded += len(values)
            io.write(out / 'state.json', dict(status='SITE_INITIAL100_GENERATING', variant='v3_official512',
                score='MAX_SITE_VIEW_COSINE_NO_PARENT_BLEND', zone=zone['zone'],
                encoded_texts=encoded, zone_encoded=min(offset + 128, len(texts)), zone_total=len(texts),
                seconds=time.monotonic() - start))
        for offset in range(0, len(docs), 128):
            chunk = docs[offset:offset + 128]
            scores = np.column_stack([(query_vectors @ np.stack([vectors[io.DOC_PROMPT + v] for v in d['site_views']]).T).max(axis=1) for d in chunk])
            sites = [dict(siret=d['key'], siren=d['siren']) for d in chunk]
            for i in range(len(queries)): best[i] = paired.old.top100(best[i], scores[i], sites)
        result = [dict(query_id=q['query_id'], fold=q['fold'], zone=zone['zone'], variant='v3_official512',
            score_recipe='MAX_SITE_VIEW_COSINE_NO_PARENT_BLEND', candidates=[dict(score=s, siret=t, siren=n) for s, t, n in candidates])
            for q, candidates in zip(queries, best)]
        if any(len(r['candidates']) != min(100, len(docs)) or len({c['siret'] for c in r['candidates']}) != len(r['candidates']) for r in result): raise ValueError('Exact initial100 cap/uniqueness')
        io.write(path, result); io.write(sealed, record(path)); rows.extend(result)
    if len(rows) != len(panel) or {r['query_id'] for r in rows} != {q['query_id'] for q in panel}: raise ValueError('Entire original cohort required')
    paired.verify_launch(launch); verify(adapter)
    io.write(out / 'v3_official512_outputs.json', rows)
    io.write(out / 'v3_official512_generation.json', dict(status='SEALED_COMPLETE_PAIRED_INITIAL100',
        outputs=record(out / 'v3_official512_outputs.json'), binding=record(binding_path), queries=len(rows),
        score='MAX_SITE_VIEW_COSINE_NO_PARENT_BLEND', encoded_texts_this_attempt=encoded,
        seconds_this_attempt=time.monotonic() - start, truth_opened=False, test_opened=False))


def seal(out):
    launch = io.read(out / 'launch.json'); paired.verify_launch(launch)
    panel = io.read(verify(launch['panel'])); scope = {r['query_id'] for r in panel}
    lexical = io.read(verify(launch['lexical']))
    candidates = {'national_lexical': {r['query_id']: r['candidates'] for r in lexical}}
    refs = []
    for variant in VARIANTS:
        generation = io.read(out / (variant + '_generation.json'))
        exited = io.read(out / (variant + '_exit.json'))
        if generation['status'] != 'SEALED_COMPLETE_PAIRED_INITIAL100' or exited.get('exit_code') != 0 or not exited.get('own_child_exit_confirmed'):
            raise ValueError('Complete owned exit0 required before any label')
        rows = io.read(verify(generation['outputs']))
        if len(rows) != len(scope) or {r['query_id'] for r in rows} != scope:
            raise ValueError('No partial denominator or changed cohort')
        candidates[variant] = {r['query_id']: r['candidates'] for r in rows}
        refs.append(record(out / (variant + '_generation.json')))
    for name, values in list(candidates.items()):
        if set(values) != scope: raise ValueError('Full identical query scope')
        for cs in values.values():
            if len(cs) > 100 or len({c['siret'] for c in cs}) != len(cs): raise ValueError('Initial100 cap')
        if name != 'national_lexical':
            candidates[name + '_rrf'] = {q: paired.fused100([candidates['national_lexical'][q], values[q]]) for q in scope}
    io.write(out / 'all_outputs_sealed.json', dict(candidates=candidates, generations=refs,
        variants=list(VARIANTS), omitted_ablation='v1_official512', truth_opened=False))
    return scope, candidates, launch


def evaluate(out):
    scope, candidates, launch = seal(out)
    import pyarrow.parquet as pq
    manifest = io.read(verify(launch['training_manifest']))
    membership = io.HOME / 'night_20261006_a/validation_membership.json'
    if io.digest(membership) != manifest['split_hashes']['validation']: raise ValueError('Membership drift')
    labels = {r['query_id']: r for r in io.read(membership) if r['query_id'] in scope}
    if set(labels) != scope: raise ValueError('Missing human labels')
    cfg = io.read(io.ROOT / 'config/retrieval_v358_postseal_evaluation.json')
    authority = io.read(verify(cfg['evaluation'])); side = io.read(verify(authority['label_side']))
    if side['never_used_in_retrieval'] is not True or side['ledger'] != cfg['ledger']: raise ValueError('Official sealed sidecar authority')
    for ref in (side['policy'], *side['files'].values()): verify(ref)
    filters = [('fold', 'in', [2, 3, 4]), ('query_id', 'in', sorted(scope))]
    accepted = {q: {labels[q]['siret']} for q in scope}
    for r in pq.read_table(side['files']['acceptable_operational.parquet']['path'], filters=filters, use_threads=False).to_pylist():
        accepted[r['query_id']].add(r['siret'])
    meta = {r['query_id']: r for r in pq.read_table(verify(cfg['ledger']), filters=filters, use_threads=False).to_pylist()}
    outcomes = {}; reports = {}
    for variant, values in candidates.items():
        rows = []
        for q in sorted(scope):
            cs = values[q]; gt = labels[q]; ids = {r['siret'] for r in cs}
            rows.append(dict(query_id=q, fold=gt['fold'], zone=gt['crm_insee'],
                exact=gt['siret'] in ids, siren=gt['siren'] in {r['siren'] for r in cs},
                operational=bool(ids & accepted[q]),
                exact_rank=next((i + 1 for i, r in enumerate(cs) if r['siret'] == gt['siret']), None)))
        outcomes[variant] = rows; segments = {}
        for field in ('fold', 'label_kind', 'qualification_v2', 'qualification_v3', 'data_origin', 'ground_truth_state'):
            groups = defaultdict(list)
            for r in rows: groups[str(meta.get(r['query_id'], {}).get(field, 'UNKNOWN'))].append(r)
            segments[field] = {key: paired.summarize(group) for key, group in groups.items()}
        reports[variant] = {**paired.summarize(rows), 'segments': segments}
    baseline = {r['query_id']: r for r in outcomes['v1_native160_rrf']}
    new = {r['query_id']: r for r in outcomes['v3_official512_rrf']}
    io.write(out / 'outcomes.json', outcomes)
    io.write(out / 'report.json', dict(status='MEASURED_DEVELOPMENT_PAIRED_INITIAL100_NOT_NATIONAL_GATE',
        variants=reports, measured_variants=list(VARIANTS), omitted_ablation='v1_official512',
        isolated_fine_tuning_effect_measured=False, texts_and_weights_both_changed=True,
        new_dense_score='MAX_SITE_VIEW_COSINE_NO_PARENT_BLEND', identity_channel_measured=False,
        pipeline_gained=sum(new[q]['exact'] and not baseline[q]['exact'] for q in scope),
        pipeline_lost=sum(baseline[q]['exact'] and not new[q]['exact'] for q in scope),
        all_outputs=record(out / 'all_outputs_sealed.json'), all_queries_in_denominator=True,
        operational_sidecar=authority['label_side'], national_dense_recall_measured=False,
        full_oof_measured=False, production_slo_measured=False,
        test_opened=False, fold0_opened=False, fold1_opened=False, downstream_models_changed=False))


def owned(out, action, deadline, python=None):
    command = [str(python or sys.executable), '-u', '-m', 'scripts.embeddinggemma2_v8_pipeline_comparison', action, '--output', str(out)]
    with (out / (action + '.log')).open('a') as log:
        try:
            child = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=max(1, deadline - time.time()))
            code = child.returncode
        except subprocess.TimeoutExpired: code = -9
    name = 'v3_official512' if action == 'generate' else action
    io.write(out / (name + '_exit.json'), dict(exit_code=code, own_child_exit_confirmed=True,
        command=command, ended_unix=time.time()))
    return code


def run(source, geographic, out):
    old = io.read(source / 'launch.json'); paired.verify_launch(old)
    generation = io.read(source / 'v1_native160_generation.json'); ex = io.read(source / 'v1_native160_exit.json')
    if generation['status'] != 'SEALED_COMPLETE_PAIRED_INITIAL100' or ex['exit_code'] != 0 or not ex['own_child_exit_confirmed']: raise ValueError('Actual native baseline required')
    verify(generation['outputs'])
    out.mkdir(parents=True, exist_ok=False)
    io.write(out / 'launch.json', {**old,
        'implementation': {**old['implementation'], str(Path(__file__)): record(Path(__file__))},
        'pipeline_comparison_source': record(source / 'launch.json'),
        'measured_variants': list(VARIANTS), 'omitted_ablation': 'v1_official512',
        'isolated_fine_tuning_effect_measured': False, 'encoder_execution_unchanged': True,
        'dual_dense_weights': {'site': 1., 'identity': 0.},
        'new_dense_score': 'MAX_SITE_VIEW_COSINE_NO_PARENT_BLEND'})
    io.write(out / 'v1_native160_generation.json', generation); io.write(out / 'v1_native160_exit.json', ex)
    deadline = io.read(verify(old['night']))['deadline_unix']
    while time.time() < deadline:
        code = owned(out, 'generate', deadline)
        if code == 0: break
        if code != 75: raise RuntimeError('Generation exit ' + str(code))
        time.sleep(30)
    else: raise TimeoutError('Original absolute night deadline')
    evaluate(out)
    origin = io.read(geographic / 'launch.json')
    if origin['comparison'] != old['resource_only_source_comparison']: raise ValueError('Original geographic binding')
    if io.read(verify(origin['comparison']))['panel'] != old['panel']: raise ValueError('Cohort drift')
    geo_dir = out / 'geo'; geo_dir.mkdir()
    io.write(geo_dir / 'launch.json', {**origin, 'comparison': record(out / 'launch.json'),
        'reused_sealed_source': record(geographic / 'launch.json'),
        'original_owned_exit': record(geographic / 'generation_exit.json')})
    io.write(geo_dir / 'generation.json', io.read(geographic / 'generation.json'))
    io.write(geo_dir / 'generation_exit.json', io.read(geographic / 'generation_exit.json'))
    if owned(geo_dir, 'evaluate-geo', deadline, python=io.ROOT / '.venv/bin/python'): raise RuntimeError('Geo evaluation failed')
    io.write(geo_dir / 'state.json', dict(status='COMPLETE_GEO_COMPARISON'))
    io.write(out / 'state.json', dict(status='COMPLETE_MEASURED_TWO_PIPELINES',
        paired=record(out / 'report.json'), geo=record(geo_dir / 'report.json'),
        isolated_fine_tuning_effect_measured=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['run', 'generate', 'evaluate-geo'])
    for name in ('source', 'geographic', 'output'): parser.add_argument('--' + name, type=Path, required=name == 'output')
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(OUTPUT): raise ValueError('Dedicated V8 namespace')
    try:
        if args.action == 'generate': generate_site_channel(args.output)
        elif args.action == 'evaluate-geo':
            from scripts import embeddinggemma2_v5_geo as geo
            geo.evaluate(args.output)
        else: run(args.source, args.geographic, args.output)
    except fit.ResourceYield as error:
        io.write(args.output / 'resource_yield.json', dict(reason=str(error))); sys.exit(75)
    except Exception as error:
        if args.output.exists(): io.write(args.output / 'execution_error.json', dict(reason=str(error)))
        raise
