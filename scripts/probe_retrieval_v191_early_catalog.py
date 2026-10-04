"""One exact merge of all successful187 phases, bounded owned metadata memory."""
import argparse
import gc
import json
import os
import resource
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

CONFIG = ROOT / 'config/retrieval_v191_early_catalog.json'
STATUS = 'SEALED_V191_ONE_COMPLETE_NATIONAL_INITIAL100_COST_ONLY'
NAMES = ('score000', 'score001', 'score002', 'audit-facts', 'audit-postings', 'audit-context')


def config():
    cfg = json.loads(CONFIG.read_text())
    if (set(cfg) != {'origin_launch', 'stages', 'output_root', 'resources', 'quiet_seconds', 'source_epoch_count'}
            or cfg['resources'] != LIMITS or cfg['quiet_seconds'] != 30
            or type(cfg['source_epoch_count']) is not int or cfg['source_epoch_count'] != 93507
            or set(cfg['stages']) != set(NAMES)):
        raise ValueError('Exact full successful phase prefix and unchanged191 budget required')
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
    for path in (Path(__file__), CONFIG, ROOT / 'docs/retrieval_v191_early_catalog_contract.md',
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
        raise ValueError('Unchanged safe fresh191 launch; no STOP retry')
    for ref in launch['implementation'].values():
        verify(ref)
    gate = json.loads((dest / 'preflight.json').read_text())
    if gate['policy_id'] != POLICY or not gate['pass_system_quiet_policy'] or not gate['worker_swap_budget_changed']:
        raise ValueError('Actual own authorized191 pressure preflight required')
    return launch


def compact_merge(output, cfg, mark):
    origin_root, source_cfg = origin(cfg)
    gc.collect()
    mark('EARLY_LOAD_EXACT_ORIGINAL_CATALOG')
    keys = original.RotatedKeys(source_cfg['catalog'])
    try:
        if keys.sites != source_cfg['sites']:
            raise ValueError('Entire unchanged original site denominator required')
        mark('REQUIRE_ALL_SIX_ACTUAL_SUCCESSFUL_PHASES_AND_FULL_HASH_RECEIPTS')
        value = audited_plan(origin_root, source_cfg, original.stage)
        captured, signature = value['source_epochs'], value['signature']
        del value
        gc.collect()  # Own Python metadata only: no system purge or other apps.
        original.check_epochs(captured)
        mark('RELOAD_EVERY_SUCCESSFUL_SCORE_BEFORE_EXACT_REDUCTION')
        parts = []
        for name in (n for n in NAMES if n.startswith('score')):
            mark('RELOAD_'+name.upper())
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
    mark('QUARANTINE_UNTIL_INDEPENDENT_COMPLETE_SOURCE_LEDGER')
    original.check_epochs(captured)
    for ref in [cfg['origin_launch'], *cfg['stages'].values()]:
        verify(ref)
    old, rows, models, geo = original.inputs(source_cfg)
    return dict(query_id=rows[0]['query_id'], fold=rows[0]['fold'], candidates=result, trace=trace,
                truth_opened=False, recall_measured=False, ready_for_fold1=False), captured


def ledger(output, cfg, mark):
    merged = stage(output, 'merge')
    draft = merged['artifacts']['draft']
    captured = json.loads(verify(merged['artifacts']['epochs']).read_text())
    original.check_epochs(captured)
    source_cfg = original.config()
    mark('FULL_ORIGINAL_SOURCE_LEDGER_AFTER_ACTUAL_MERGE')
    root = Path(source_cfg['addresses']['path']).parent
    launch, plan, epochs = backend.load(root)  # SAME full177 control, after merge.
    if (len(epochs['epochs']) != cfg['source_epoch_count']
            or plan['sites'] != source_cfg['sites'] or plan['facts'] != source_cfg['facts']):
        raise ValueError('Entire original national source ledger required after reduction')
    authorities = dict(launch=record(root / 'launch.json'), plan=launch['plan'], epochs=launch['epochs'],
                       source=source_cfg['addresses'])
    backend.check_live(launch, plan, epochs)
    original.check_epochs(captured)
    verify(draft)
    return dict(status='VERIFIED_V191_COMPLETE_POSTMERGE_SOURCE_LEDGER',
        merge=record(output / 'merge/manifest.json'), draft=draft, captured_epochs=merged['artifacts']['epochs'],
        authorities=authorities, source_epoch_count=len(epochs['epochs']), sites=plan['sites'], facts=plan['facts'],
        signature=merged['trace']['signature'], initial100_emitted=False, **original.FLAGS)


def worker(output, phase):
    if phase not in ('merge', 'ledger'):
        raise ValueError('Explicit whole physical191 phase required')
    dest = output / phase
    if any(p.name not in ('preflight.json', 'worker.log') for p in dest.iterdir()):
        raise ValueError('Immutable191 worker already started, no retry')
    began = time.monotonic()
    def mark(step):
        print(json.dumps(dict(checkpoint=step, worker_elapsed_seconds=time.monotonic()-began,
                              rss_bytes=psutil.Process().memory_info().rss,
                              peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)), flush=True)
    budget = None
    try:
        with Budget(LIMITS) as budget:
            launch = load(output, dest)
            if phase == 'merge':
                result, captured = compact_merge(output, launch['config'], mark)
                write_json(dest / 'draft_initial100.json', result)
                write_json(dest / 'captured_epochs.json', captured)
                artifacts = dict(draft=record(dest / 'draft_initial100.json'), epochs=record(dest / 'captured_epochs.json'))
                trace = result['trace']
            else:
                result = ledger(output, launch['config'], mark)
                write_json(dest / 'receipt.json', result)
                artifacts = dict(receipt=record(dest / 'receipt.json'))
                trace = dict(signature=result['signature'], source_epoch_count=result['source_epoch_count'])
            mark('FINAL_SOURCE_AND_IMPLEMENTATION_CHECK')
            load(output, dest)
            if phase == 'merge':
                original.check_epochs(captured)
        write_json(dest / 'manifest.json', dict(status='QUARANTINED_V191_'+phase.upper(), launch=record(output / 'launch.json'),
            preflight=record(dest / 'preflight.json'), artifacts=artifacts,
            resources=budget.report(), trace=trace, pending_postmerge_ledger=phase == 'merge',
            inherited_stages=launch['config']['stages'], initial100_emitted=False,
            old_candidates_opened=False, ready_for_fold1=False, **original.FLAGS))
    except BaseException as exc:
        write_json(dest / 'worker_stop.json', dict(status='STOP', reason=str(exc), initial100_emitted=False,
            resources=budget.report() if budget is not None and hasattr(budget, 'elapsed') else None,
            old_candidates_opened=False, **original.FLAGS))
        raise


def stage(output, phase):
    if phase not in ('merge', 'ledger'):
        raise ValueError('Explicit191 physical phase required')
    dest = output / phase
    if any((dest / n).exists() for n in ('worker_stop.json', 'supervisor_stop.json', 'status.json')):
        raise ValueError('STOP191_STAGE_DO_NOT_USE')
    parent_exit(dest)
    m = json.loads((dest / 'manifest.json').read_text())
    validate_report(m['resources'])
    if (m['status'] != 'QUARANTINED_V191_'+phase.upper() or m['launch'] != record(output / 'launch.json')
            or m['inherited_stages'] != config()['stages'] or m['initial100_emitted'] is not False
            or m.get('pending_postmerge_ledger') is not (phase == 'merge')
            or m['old_candidates_opened'] or any(m.get(k) is not v for k, v in original.FLAGS.items())):
        raise ValueError('Actual complete supervised191 merge required')
    if verify(m['preflight']).resolve() != (dest / 'preflight.json').resolve():
        raise ValueError('Actual own191 preflight required')
    load(output, dest)
    for ref in m['artifacts'].values():
        verify(ref)
    if phase == 'merge':
        original.check_epochs(json.loads(verify(m['artifacts']['epochs']).read_text()))
    return m


def admission(output):
    merged, checked = stage(output, 'merge'), stage(output, 'ledger')
    r = json.loads(verify(checked['artifacts']['receipt']).read_text())
    source_cfg = original.config()
    if (r['status'] != 'VERIFIED_V191_COMPLETE_POSTMERGE_SOURCE_LEDGER'
            or r['merge'] != record(output / 'merge/manifest.json')
            or r['draft'] != merged['artifacts']['draft'] or r['captured_epochs'] != merged['artifacts']['epochs']
            or r['source_epoch_count'] != config()['source_epoch_count']
            or r['signature'] != merged['trace']['signature'] or checked['trace']['signature'] != r['signature']
            or r['sites'] != source_cfg['sites'] or r['facts'] != source_cfg['facts']
            or r['initial100_emitted'] is not False or any(r.get(k) is not v for k, v in original.FLAGS.items())):
        raise ValueError('Complete actual postmerge ledger receipt bound to exact current draft required')
    a = r['authorities']
    if a['source'] != source_cfg['addresses']:
        raise ValueError('Same exact source177 authority required')
    launch = json.loads(verify(a['launch']).read_text())
    if a['plan'] != launch['plan'] or a['epochs'] != launch['epochs']:
        raise ValueError('Exact entire source ledger authority hashes required')
    plan = json.loads(verify(a['plan']).read_text())
    epochs = json.loads(verify(a['epochs']).read_text())
    if len(epochs['epochs']) != config()['source_epoch_count']:
        raise ValueError('Entire original source epoch population required')
    backend.check_live(launch, plan, epochs)
    original.check_epochs(json.loads(verify(merged['artifacts']['epochs']).read_text()))
    return merged, checked


def run(output):
    cfg = config()
    if output.exists() or not output.resolve().is_relative_to(Path(cfg['output_root']).resolve()):
        raise ValueError('Fresh CATNAT-only191 root required')
    origin(cfg)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'launch.json', dict(config=cfg, implementation=implementation(),
                                          old_candidates_opened=False, **original.FLAGS))
    began = time.monotonic()
    try:
        for phase in ('merge', 'ledger'):
            gate = quiet(cfg['quiet_seconds'], 50, 55)
            dest = output / phase
            dest.mkdir()
            write_json(dest / 'preflight.json', gate)
            with (dest / 'worker.log').open('x') as log:
                supervision = supervise([sys.executable, '-u', '-m', 'scripts.probe_retrieval_v191_early_catalog',
                    'worker', '--output', str(output), '--phase', phase], log,
                    {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}, dest / 'supervisor_stop.json')
            write_json(dest / 'supervision.json', supervision)
            m = stage(output, phase)
            print(json.dumps(dict(completed=phase, resources=m['resources'], trace=m['trace'])), flush=True)
        m, checked = admission(output)
        source_cfg = original.config()
        prefix_seconds = sum(original.prefix_stage(source_cfg, name)[0]['resources']['wall_seconds']
                             for name in ('names', 'addresses'))
        previous_seconds = sum(json.loads(verify(r).read_text())['resources']['wall_seconds'] for r in cfg['stages'].values())
        write_json(output / 'manifest.json', dict(status=STATUS, launch=record(output / 'launch.json'),
            merge=record(output / 'merge/manifest.json'), ledger=record(output / 'ledger/manifest.json'),
            outputs=m['artifacts']['draft'], queries=1,
            sites=source_cfg['sites'], facts=source_cfg['facts'], inherited_stages=cfg['stages'],
            inherited_proof_worker_seconds=prefix_seconds, inherited_score_and_audit_worker_seconds=previous_seconds,
            all_worker_load_proof_scoring_and_hash_seconds=prefix_seconds+previous_seconds+
                m['resources']['wall_seconds']+checked['resources']['wall_seconds'],
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
    parser.add_argument('--phase', choices=('merge', 'ledger'))
    args = parser.parse_args()
    if args.mode == 'worker' and args.phase is None:
        parser.error('Explicit complete physical phase required')
    (run(args.output) if args.mode == 'run' else worker(args.output, args.phase))
