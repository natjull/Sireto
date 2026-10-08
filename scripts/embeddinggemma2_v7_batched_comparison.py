"""Same sealed paired experiment, length-bucketed inference only; no training."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v4_compare as paired
from scripts import embeddinggemma2_v4_fit as fit
from src.xgb_matcher.retrieval_v20_workbench import record, verify

OUTPUT = io.HOME / 'v7'
ORIGINAL_LOADER = fit.load_model


def bucketed_inference(encode, model, tokenizer, texts, batch_size=32):
    import torch
    if not texts: raise ValueError('Nonempty inference required')
    lengths = [len(ids) for ids in tokenizer(texts, truncation=False)['input_ids']]
    order = sorted(range(len(texts)), key=lambda i: (lengths[i], i))
    result = [None] * len(texts)
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(order), batch_size):
            indices = order[offset:offset + batch_size]
            vectors = encode([texts[i] for i in indices]).cpu().numpy()
            for i, vector in zip(indices, vectors): result[i] = vector
    torch.mps.empty_cache()
    return np.stack(result)


def optimized_loader(cfg, adapter):
    model, params, encode, original, tokenizer = ORIGINAL_LOADER(cfg, adapter)
    def inference(texts): return bucketed_inference(encode, model, tokenizer, texts)
    return model, params, encode, inference, tokenizer


def preflight(out):
    launch = io.read(out / 'launch.json'); paired.verify_launch(launch)
    cfg = io.read(verify(launch['night_config']))
    docs = io.read(verify(launch['corpora'][0]['file']))
    # Query-only official texts: never read labels or select by retrieval outcome.
    texts = sorted({io.DOC_PROMPT + v for d in docs for v in d['site_views']},
        key=lambda s: hashlib.sha256(('v7-official-numeric|' + s).encode()).digest())[:128]
    model, params, encode, original, tokenizer = ORIGINAL_LOADER(cfg, verify(launch['v1_adapter']))
    start = time.monotonic(); old = original(texts); old_seconds = time.monotonic() - start
    start = time.monotonic(); new = bucketed_inference(encode, model, tokenizer, texts)
    new_seconds = time.monotonic() - start
    cosine_error = float(np.max(1 - np.sum(old * new, axis=1)))
    maximum_error = float(np.max(np.abs(old - new)))
    finite = bool(np.isfinite(new).all())
    if not finite or maximum_error > .002 or cosine_error > .00001:
        raise ValueError(f'Predeclared numeric tolerance failed: {maximum_error=} {cosine_error=}')
    if new_seconds >= old_seconds:
        raise ValueError('Larger batches did not speed up the measured official texts')
    io.write(out / 'inference_preflight.json', dict(status='PASS_NUMERIC_AND_SPEED_NO_LABELS',
        texts=len(texts), original_seconds=old_seconds, bucketed_seconds=new_seconds,
        maximum_absolute_error=maximum_error, maximum_cosine_error=cosine_error,
        max_absolute_tolerance=.002, max_cosine_tolerance=.00001,
        batch_size=32, mathematical_score_unchanged=True, bitwise_identity_claimed=False,
        truth_opened=False, test_opened=False))


def owned(out, action, deadline, variant=None, python=None):
    command = [str(python or sys.executable), '-u', '-m', 'scripts.embeddinggemma2_v7_batched_comparison',
        action, '--output', str(out)]
    if variant: command += ['--variant', variant]
    name = variant or action
    with (out / (name + '.log')).open('a') as log:
        try:
            child = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                timeout=max(1, deadline - time.time()))
            code = child.returncode
        except subprocess.TimeoutExpired: code = -9
    io.write(out / (name + '_exit.json'), dict(exit_code=code, own_child_exit_confirmed=True,
        command=command, ended_unix=time.time()))
    return code


def continue_run(source, geographic, out):
    # Preparation performs no new corpus scan, mining or training.
    old = io.read(source / 'launch.json'); paired.verify_launch(old)
    generation = io.read(source / 'v1_native160_generation.json'); exit_record = io.read(source / 'v1_native160_exit.json')
    if generation['status'] != 'SEALED_COMPLETE_PAIRED_INITIAL100' or exit_record['exit_code'] != 0 or not exit_record['own_child_exit_confirmed']:
        raise ValueError('Complete native baseline required')
    verify(generation['outputs'])
    out.mkdir(parents=True, exist_ok=False)
    io.write(out / 'launch.json', {**old,
        'implementation': {**old['implementation'], str(Path(__file__)): record(Path(__file__))},
        'execution_only_source': record(source / 'launch.json'), 'inference_batch_size': 32,
        'inference_length_order_restored': True})
    io.write(out / 'v1_native160_generation.json', generation)
    io.write(out / 'v1_native160_exit.json', exit_record)
    deadline = io.read(verify(old['night']))['deadline_unix']
    io.write(out / 'state.json', dict(status='INFERENCE_PREFLIGHT'))
    if owned(out, 'preflight', deadline): raise RuntimeError('Inference preflight failed')
    for variant in paired.VARIANTS[1:]:
        while time.time() < deadline:
            code = owned(out, 'generate', deadline, variant)
            if code == 0: break
            if code != 75: raise RuntimeError('Generation failed ' + variant)
            time.sleep(30)
        else: raise TimeoutError('Original absolute night deadline')
    paired.evaluate(out)
    io.write(out / 'state.json', dict(status='COMPLETE_PAIRED_INITIAL100_COMPARISON'))
    origin = io.read(geographic / 'launch.json')
    source_launch = io.read(verify(old['resource_only_source_comparison']))
    if origin['comparison'] != old['resource_only_source_comparison']:
        raise ValueError('Original geography and cohort binding')
    if source_launch['panel'] != old['panel']: raise ValueError('Cohort drift')
    geo_dir = out / 'geo'; geo_dir.mkdir()
    io.write(geo_dir / 'launch.json', {**origin, 'comparison': record(out / 'launch.json'),
        'reused_sealed_source': record(geographic / 'launch.json'),
        'original_owned_exit': record(geographic / 'generation_exit.json')})
    io.write(geo_dir / 'generation.json', io.read(geographic / 'generation.json'))
    io.write(geo_dir / 'generation_exit.json', io.read(geographic / 'generation_exit.json'))
    code = owned(geo_dir, 'evaluate-geo', deadline, python=io.ROOT / '.venv/bin/python')
    if code: raise RuntimeError('Geo evaluation exit ' + str(code))
    io.write(geo_dir / 'state.json', dict(status='COMPLETE_GEO_COMPARISON'))
    io.write(out / 'completion.json', dict(status='COMPLETE_MEASURED',
        paired=record(out / 'report.json'), geo=record(geo_dir / 'report.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['run', 'preflight', 'generate', 'evaluate-geo'])
    for name in ('source', 'geographic', 'output'): parser.add_argument('--' + name, type=Path, required=name == 'output')
    parser.add_argument('--variant', choices=paired.VARIANTS)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(OUTPUT): raise ValueError('Dedicated V7 namespace')
    try:
        if args.action == 'preflight': preflight(args.output)
        elif args.action == 'generate':
            fit.load_model = optimized_loader
            paired.generate(args.output, args.variant)
        elif args.action == 'evaluate-geo':
            from scripts import embeddinggemma2_v5_geo as geo
            geo.evaluate(args.output)
        else: continue_run(args.source, args.geographic, args.output)
    except fit.ResourceYield as error:
        io.write(args.output / 'resource_yield.json', dict(reason=str(error))); sys.exit(75)
    except Exception as error:
        if args.output.exists(): io.write(args.output / 'execution_error.json', dict(reason=str(error)))
        raise
