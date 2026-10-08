"""Finalize a deadline-killed V4 checkpoint, with ZERO additional gradient steps.

Historical roots remain immutable. The three paired recipes and sealed V5
geographic generation are reused exactly; this is not a new scientific run.
"""
import argparse
from pathlib import Path
import subprocess
import sys
import time

from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v4_fit as fit
from scripts import embeddinggemma2_v4_compare as paired
from scripts import embeddinggemma2_v5_geo as geo
from src.xgb_matcher.retrieval_v20_workbench import record, verify

OUTPUT = io.HOME / 'v6'


def require_deadline_exit(source):
    launch = io.read(source / 'fit/launch.json')
    supervision = io.read(source / 'fit/supervisor.json')
    history = supervision.get('history', [])
    if (supervision['status'] != 'BUDGET_EXHAUSTED' or not history
            or history[-1]['exit_code'] not in (-9, -15, 75)
            or not (history[-1].get('reason') or {}).get('deadline_reached')
            or time.time() < launch['deadline_unix']):
        raise ValueError('Only an actual owned deadline interruption is recoverable')
    import psutil
    for process in psutil.process_iter(['pid', 'cmdline']):
        try:
            command = ' '.join(process.info['cmdline'] or [])
            if 'scripts.embeddinggemma2_v4_fit' in command and str(source) in command:
                raise ValueError('Original training process still active')
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    verify(launch['implementation']); verify(launch['data_implementation'])
    return launch


def migrate_checkpoint(source, out):
    import torch
    original = source / 'fit'
    old_launch = require_deadline_exit(source)
    out.mkdir(parents=True, exist_ok=False)
    target = out / 'fit'; target.mkdir()
    provenance = dict(source_launch=record(original / 'launch.json'),
        source_checkpoint=record(original / 'checkpoint.pt'),
        source_best=record(original / 'best_adapter.pt'),
        source_supervision=record(original / 'supervisor.json'),
        implementation=record(Path(__file__)), no_additional_gradient_steps=True)
    launch = {**old_launch, 'deadline_finalization': provenance}
    io.write(target / 'launch.json', launch)
    states = {}
    for name in ('checkpoint.pt', 'best_adapter.pt'):
        state = torch.load(original / name, map_location='cpu', weights_only=True)
        if state['launch_sha256'] != io.digest(original / 'launch.json'):
            raise ValueError('Original checkpoint binding mismatch')
        states[name] = (state['step'], state['epoch'], state['next_offset'])
        state['launch_sha256'] = io.digest(target / 'launch.json')
        torch.save(state, target / name)
    if states['best_adapter.pt'][0] > states['checkpoint.pt'][0]:
        raise ValueError('Best checkpoint ahead of durable training checkpoint')
    for name in ('preflight.json', 'preflight_exit.json'):
        io.write(target / name, io.read(original / name))
    manifest = io.read(verify(old_launch['data']))
    io.write(out / 'config.json', io.read(verify(manifest['config'])))
    night = io.read(source / 'night.json')
    io.write(out / 'night.json', {**night, 'deadline_recovery': provenance})
    io.write(out / 'migration.json', {**provenance,
        'checkpoint_cursor': states['checkpoint.pt'], 'best_cursor': states['best_adapter.pt'],
        'fit_deadline_preserved': old_launch['deadline_unix'],
        'night_deadline_preserved': night['deadline_unix']})
    for ref in provenance.values():
        if isinstance(ref, dict) and 'sha256' in ref: verify(ref)
    return states['checkpoint.pt']


def owned_call(out, action, deadline, variant=None):
    command = [sys.executable, '-u', '-m', 'scripts.embeddinggemma2_v6_deadline_recovery',
        action, '--output', str(out)]
    if variant: command += ['--variant', variant]
    name = variant or action
    with (out / (name + '.log')).open('a') as log:
        try:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                timeout=max(1, deadline - time.time()))
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = -9
    receipt = dict(own_child_exit_confirmed=True, exit_code=code,
        command=command, ended_unix=time.time())
    io.write(out / (name + '_exit.json'), receipt)
    return code


def recover(source, comparison, geographic, out):
    cursor = migrate_checkpoint(source, out)
    deadline = io.read(out / 'night.json')['deadline_unix']
    io.write(out / 'state.json', dict(status='FINAL_VALIDATION_ONLY', source_cursor=cursor))
    # Worker sees the ORIGINAL already elapsed fit deadline: both training loops
    # are skipped. Only its original final validation/checkpoint logic executes.
    code = owned_call(out / 'fit', 'finalize', deadline)
    if code: raise RuntimeError('Final validation exit ' + str(code))
    final = io.read(out / 'fit/state.json')
    if final['status'] != 'TIME_BUDGET_CHECKPOINTED' or (final['step'], final['epoch']) != cursor[:2]:
        raise ValueError('Finalization must not perform any new training step')
    io.write(out / 'fit/supervisor.json', dict(status='COMPLETE', exit_code=0,
        finalization_only=True, actual_owned_exit=record(out / 'fit/finalize_exit.json')))
    io.write(out / 'state.json', dict(status='FIT_DONE_EVALUATION_PENDING',
        training=record(out / 'fit/state.json'), deadline_unix=deadline))
    old = io.read(comparison / 'launch.json'); paired.verify_launch(old)
    new = out / 'comparison'; new.mkdir()
    launch = {**old, 'night': record(out / 'night.json'),
        'night_config': record(out / 'config.json'),
        'requested_v3_adapter_path': str(out / 'fit/best_adapter.pt'),
        'resource_only_source_comparison': record(comparison / 'launch.json'),
        'implementation': {**old['implementation'], str(Path(__file__)): record(Path(__file__))}}
    io.write(new / 'launch.json', launch)
    io.write(out / 'state.json', dict(status='PAIRED_SEARCH_RUNNING', deadline_unix=deadline))
    for variant in paired.VARIANTS:
        while time.time() < deadline:
            code = owned_call(new, 'generate', deadline, variant)
            if code == 0: break
            if code != 75: raise RuntimeError('Paired generation exit ' + str(code))
            time.sleep(30)
        else: raise TimeoutError('Original absolute night deadline')
    paired.evaluate(new)
    io.write(new / 'state.json', dict(status='COMPLETE_PAIRED_INITIAL100_COMPARISON'))
    old_geo = io.read(geographic / 'launch.json')
    if old_geo['comparison'] != record(comparison / 'launch.json'):
        raise ValueError('Original geographic comparison binding')
    generation = io.read(geographic / 'generation.json')
    receipt = io.read(geographic / 'generation_exit.json')
    if generation['status'] != 'SEALED_ALL80_NATIONAL_GEO_INITIAL100' or receipt.get('exit_code') != 0 or not receipt.get('own_child_exit_confirmed'):
        raise ValueError('No geographic regeneration or fabricated owned exit')
    verify(generation['outputs']); verify(old_geo['implementation'])
    for ref in old_geo['helper_implementations']: verify(ref)
    new_geo = out / 'geo'; new_geo.mkdir()
    io.write(new_geo / 'launch.json', {**old_geo, 'comparison': record(new / 'launch.json'),
        'reused_sealed_source': record(geographic / 'launch.json'),
        'original_owned_exit': record(geographic / 'generation_exit.json')})
    io.write(new_geo / 'generation.json', generation)
    io.write(new_geo / 'generation_exit.json', receipt)
    geo.evaluate(new_geo)
    io.write(new_geo / 'state.json', dict(status='COMPLETE_GEO_COMPARISON'))
    io.write(out / 'state.json', dict(status='COMPLETE_MEASURED_PARTIAL_TRAINING',
        comparison=record(new / 'report.json'), geographic=record(new_geo / 'report.json'),
        no_additional_gradient_steps=True, test_opened=False, downstream_models_changed=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['recover', 'finalize', 'generate'])
    for name in ('source', 'comparison', 'geographic', 'output'):
        parser.add_argument('--' + name, type=Path, required=name == 'output')
    parser.add_argument('--variant', choices=paired.VARIANTS)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(OUTPUT): raise ValueError('Dedicated V6 namespace')
    try:
        if args.action == 'finalize': sys.exit(fit.worker(args.output))
        elif args.action == 'generate': paired.generate(args.output, args.variant)
        else: recover(args.source, args.comparison, args.geographic, args.output)
    except fit.ResourceYield as error:
        io.write(args.output / 'resource_yield.json', dict(reason=str(error))); sys.exit(75)
    except Exception as error:
        if args.output.exists(): io.write(args.output / 'recovery_error.json', dict(reason=str(error)))
        raise
