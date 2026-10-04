"""One exact merge of all successful187 phases, bounded owned metadata memory."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import psutil

from scripts import probe_retrieval_v187_staged_initial as original
from scripts import compile_retrieval_v177_packed_addresses as backend
from src.xgb_matcher.retrieval_memory_policy_v153 import Budget, LIMITS, POLICY, quiet, validate_report
from src.xgb_matcher.retrieval_address_authority_v168 import parent_exit
from src.xgb_matcher.retrieval_supervisor_v164 import supervise
from src.xgb_matcher.retrieval_stream_audit_v189 import audited_plan
from src.xgb_matcher.retrieval_v20_workbench import ROOT, record, verify, write_json

CONFIG = ROOT / 'config/retrieval_v189_stream_merge.json'
STATUS = 'SEALED_V189_ONE_COMPLETE_NATIONAL_INITIAL100_COST_ONLY'
NAMES = ('score000', 'score001', 'score002', 'audit-facts', 'audit-postings', 'audit-context')


def config():
    cfg = json.loads(CONFIG.read_text())
    if (set(cfg) != {'origin_launch', 'stages', 'output_root', 'resources', 'quiet_seconds'}
            or cfg['resources'] != LIMITS or cfg['quiet_seconds'] != 30
            or set(cfg['stages']) != set(NAMES)):
        raise ValueError('Exact full successful phase prefix and unchanged189 budget required')
    return cfg


def origin(cfg):
    launch_path = verify(cfg['origin_launch'])
    old = json.loads(launch_path.read_text())
    if (old['config'] != original.config() or old['old_candidates_opened']
            or any(old.get(k) is not v for k, v in original.FLAGS.items())):
        raise ValueError('Exact unchanged entire187 query/source/recipe required')
    for ref in old['implementation'].values():
        verify(ref)
    # Only actual successful187 components are accepted. The containing root
    # remains STOP; its failed merge/output is NEVER followed or imported.
    for name in NAMES:
        expected = cfg['stages'][name]
        path = verify(expected)
        if path.resolve() != (launch_path.parent / name / 'manifest.json').resolve():
            raise ValueError('Each frozen original own phase path and checksum required')
        m, actual = original.stage(launch_path.parent, name)
        if actual != expected or m['initial100_emitted']:
            raise ValueError('Actual successful original score/audit, never previous output required')
        del m
    original.inputs(old['config'])  # All128 CRM rows validated before frozen prefix.
    return launch_path.parent, old['config']


def implementation():
    refs = original.implementation()
    for path in (Path(__file__), CONFIG, ROOT / 'docs/retrieval_v189_stream_merge_contract.md',
                 ROOT / 'src/xgb_matcher/retrieval_stream_audit_v189.py'):
        refs[str(path)] = record(path)
    return refs


def load(output, dest):
    launch = json.loads((output / 'launch.json').read_text())
    cfg = config()
    if (launch['config'] != cfg or any(launch.get(k) is not v for k, v in original.FLAGS.items())
            or launch['old_candidates_opened'] or (output / 'status.json').exists()
            or any((dest / n).exists() for n in ('worker_stop.json', 'supervisor_stop.json', 'status.json'))
            or not output.resolve().is_relative_to(Path(cfg['output_root']).resolve())):
        raise ValueError('Unchanged safe fresh189 launch; no STOP retry')
    for ref in launch['implementation'].values():
        verify(ref)
    gate = json.loads((dest / 'preflight.json').read_text())
    if gate['policy_id'] != POLICY or not gate['pass_system_quiet_policy'] or not gate['worker_swap_budget_changed']:
        raise ValueError('Actual own authorized189 pressure preflight required')
    return launch


def compact_merge(output, cfg, mark):
    origin_root, source_cfg = origin(cfg)
    gc.collect()
    mark('REQUIRE_ALL_SIX_ACTUAL_SUCCESSFUL_PHASES_AND_FULL_HASH_RECEIPTS')
    value = audited_plan(origin_root, source_cfg, original.stage)
    captured, signature = value['source_epochs'], value['signature']
    del value  # Do not retain three large packet/group copies with identity maps.
    gc.collect()  # Own Python metadata only: no system purge or other apps.
    original.check_epochs(captured)
    mark('LOAD_EXACT_ORIGINAL_CATALOG')
    keys = original.RotatedKeys(source_cfg['catalog'])
    try:
        if keys.sites != source_cfg['sites']:
            raise ValueError('Entire unchanged original site denominator required')
        parts = []
        for name in (n for n in NAMES if n.startswith('score')):
            m, ref = original.stage(origin_root, name)
            if ref != cfg['stages'][name]:
                raise ValueError('Unchanged frozen score manifest required')
            parts.append(json.loads(verify(m['artifacts']['partition']).read_text()))
            del m  # Epochs already retained once in the union certified above.
        mark('SAME_SCORE_FULL_NATIONAL_TOP100_REDUCTION')
        result, trace = original.merge_partitions(SimpleNamespace(sites=keys.sites, keys=keys), source_cfg['facts'], parts)
        if trace['signature'] != signature:
            raise ValueError('Same exact full audit and initial score recipe required')
        keys.verify_after()
    finally:
        keys.close()
    del parts, keys
    gc.collect()
    mark('CHECK_COMPLETE_ORIGINAL_SOURCE_EPOCH_LEDGER_AFTER_REDUCTION')
    # Same full177 ledger validation as187, now AFTER closing owned catalog
    # maps and dropping duplicate score metadata. Never replace it by stat.
    backend.load(Path(source_cfg['addresses']['path']).parent)
    gc.collect()
    original.check_epochs(captured)
    for ref in [cfg['origin_launch'], *cfg['stages'].values()]:
        verify(ref)
    old, rows, models, geo = original.inputs(source_cfg)
    return dict(query_id=rows[0]['query_id'], fold=rows[0]['fold'], candidates=result, trace=trace,
                truth_opened=False, recall_measured=False, ready_for_fold1=False), captured


def worker(output):
    dest = output / 'merge'
    if any(p.name not in ('preflight.json', 'worker.log') for p in dest.iterdir()):
        raise ValueError('Immutable189 worker already started, no retry')
    began = time.monotonic()
    def mark(step):
        print(json.dumps(dict(checkpoint=step, worker_elapsed_seconds=time.monotonic()-began,
                              rss_bytes=psutil.Process().memory_info().rss)), flush=True)
    budget = None
    try:
        with Budget(LIMITS) as budget:
            launch = load(output, dest)
            result, captured = compact_merge(output, launch['config'], mark)
            mark('FINAL_SOURCE_AND_IMPLEMENTATION_CHECK')
            original.check_epochs(captured)
            load(output, dest)
            write_json(dest / 'initial100.json', result)
            artifact = record(dest / 'initial100.json')
        write_json(dest / 'manifest.json', dict(status=STATUS, launch=record(output / 'launch.json'),
            preflight=record(dest / 'preflight.json'), artifacts=dict(initial100=artifact),
            source_epochs=captured, resources=budget.report(), trace=result['trace'],
            inherited_stages=launch['config']['stages'], initial100_emitted=True,
            old_candidates_opened=False, ready_for_fold1=False, **original.FLAGS))
    except BaseException as exc:
        write_json(dest / 'worker_stop.json', dict(status='STOP', reason=str(exc), initial100_emitted=False,
            resources=budget.report() if budget is not None and hasattr(budget, 'elapsed') else None,
            old_candidates_opened=False, **original.FLAGS))
        raise


def stage(output):
    dest = output / 'merge'
    if any((dest / n).exists() for n in ('worker_stop.json', 'supervisor_stop.json', 'status.json')):
        raise ValueError('STOP189_STAGE_DO_NOT_USE')
    parent_exit(dest)
    m = json.loads((dest / 'manifest.json').read_text())
    validate_report(m['resources'])
    if (m['status'] != STATUS or m['launch'] != record(output / 'launch.json')
            or m['inherited_stages'] != config()['stages'] or m['initial100_emitted'] is not True
            or m['old_candidates_opened'] or any(m.get(k) is not v for k, v in original.FLAGS.items())):
        raise ValueError('Actual complete supervised189 merge required')
    if verify(m['preflight']).resolve() != (dest / 'preflight.json').resolve():
        raise ValueError('Actual own189 preflight required')
    load(output, dest)
    original.check_epochs(m['source_epochs'])
    verify(m['artifacts']['initial100'])
    return m


def run(output):
    cfg = config()
    if output.exists() or not output.resolve().is_relative_to(Path(cfg['output_root']).resolve()):
        raise ValueError('Fresh CATNAT-only189 root required')
    origin(cfg)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'launch.json', dict(config=cfg, implementation=implementation(),
                                          old_candidates_opened=False, **original.FLAGS))
    began = time.monotonic()
    try:
        gate = quiet(cfg['quiet_seconds'], 50, 55)
        dest = output / 'merge'
        dest.mkdir()
        write_json(dest / 'preflight.json', gate)
        with (dest / 'worker.log').open('x') as log:
            supervision = supervise([sys.executable, '-u', '-m', 'scripts.probe_retrieval_v189_stream_merge',
                'worker', '--output', str(output)], log,
                {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}, dest / 'supervisor_stop.json')
        write_json(dest / 'supervision.json', supervision)
        m = stage(output)
        source_cfg = original.config()
        prefix_seconds = sum(original.prefix_stage(source_cfg, name)[0]['resources']['wall_seconds']
                             for name in ('names', 'addresses'))
        previous_seconds = sum(json.loads(verify(r).read_text())['resources']['wall_seconds'] for r in cfg['stages'].values())
        write_json(output / 'manifest.json', dict(status=STATUS, launch=record(output / 'launch.json'),
            merge=record(dest / 'manifest.json'), outputs=m['artifacts']['initial100'], queries=1,
            sites=source_cfg['sites'], facts=source_cfg['facts'], inherited_stages=cfg['stages'],
            inherited_proof_worker_seconds=prefix_seconds, inherited_score_and_audit_worker_seconds=previous_seconds,
            all_worker_load_proof_scoring_and_hash_seconds=prefix_seconds+previous_seconds+m['resources']['wall_seconds'],
            current_run_elapsed_including_quiet_seconds=time.monotonic()-began, initial100_emitted=True,
            complete_retrieval_index=False, full_query_population=False, development_panel_reused=True,
            slo_validated=False, ready_for_fold1=False, downstream_models_changed=False,
            old_candidates_opened=False, **original.FLAGS))
        print(json.dumps(dict(status=STATUS, resources=m['resources'], trace=m['trace'])), flush=True)
    except BaseException as exc:
        write_json(output / 'status.json', dict(status='STOP', reason=str(exc), initial100_emitted=False,
                                              old_candidates_opened=False, **original.FLAGS))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('run', 'worker'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    (run(args.output) if args.mode == 'run' else worker(args.output))
