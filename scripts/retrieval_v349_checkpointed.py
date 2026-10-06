"""Frozen V344 workers, same scores/budgets; append-only validated checkpoints."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from scripts import retrieval_v344_corroborated_population as worker
from src.xgb_matcher.retrieval_checkpoints_v348 import (
    atomic_json, controller_lock, slots, failure_kind, next_slot, MAX_ATTEMPTS, MAX_CONTROLLER_RUNS)
from src.xgb_matcher.retrieval_v20_workbench import ROOT, record, verify, write_json

CONFIG = ROOT/'config/retrieval_v349_checkpointed.json'
STATUS = 'SEALED_V349_ALL2000_CORROBORATED_INITIAL100_PILOT'
SAFE_FALSE = worker.SAFE_FALSE
FALSE = worker.FALSE
backend = worker.backend
population = worker.population
valid_rows = worker.valid_rows


def config():
    cfg = json.loads(CONFIG.read_text())
    if (cfg['resources'] != worker.LIMITS or cfg['queries'] != 2000
            or cfg['max_attempts_per_batch'] != MAX_ATTEMPTS
            or cfg['max_controller_runs'] != MAX_CONTROLLER_RUNS
            or cfg['exhausted_failure_policy'] != 'STOP_KEEP_DENOMINATOR'
            or verify(cfg['worker_config']).resolve() != worker.CONFIG.resolve()
            or not Path(cfg['output_root']).resolve().is_relative_to(Path(worker.config()['output_root']).resolve())):
        raise ValueError('Frozen bounded retries and unchanged worker resources required')
    return cfg


def implementation():
    refs = worker.implementation()
    for p in (Path(__file__), CONFIG, ROOT/'src/xgb_matcher/retrieval_checkpoints_v348.py',
              ROOT/'src/xgb_matcher/retrieval_evaluation_authority_v350.py',
              ROOT/'scripts/evaluate_retrieval_v351_postseal.py',
              ROOT/'config/retrieval_v351_postseal_evaluation.json',
              ROOT/'scripts/launch_retrieval_v352_resumable.py',
              ROOT/'docs/retrieval_v349_checkpoint_contract.md',
              ROOT/'docs/retrieval_v352_detached_launch_contract.md'):
        refs[str(p)] = record(p)
    return refs


def init(root):
    cfg = config()
    if root.exists() or root.parent.resolve() != Path(cfg['output_root']).resolve():
        raise ValueError('Fresh authorized resumable run required')
    _, base, old, _, _ = backend.inputs()
    panel_root = Path(worker.config()['panel_root'])
    panel, _, _ = worker.population.sealed(panel_root)
    queries = worker.population_rows(panel['queries'])
    root.mkdir(parents=True, exist_ok=False)
    launch = dict(config=worker.config(), execution_config=cfg, queries=list(queries),
        implementation=implementation(), query_sources=[record(panel_root/'source/panel.json')],
        query_only_launch=record(panel_root/'population_manifest.json'),
        geo_queries=record(panel_root/'source/panel.json'), panel_manifest=record(panel_root/'manifest.json'),
        inventory=old['inventory'], index_documents=old['index_documents'],
        index_manifest=old['index_manifest'], index_meta=old['index_meta'],
        backend_config=worker.config()['backend_config'], native_leaf_sites=43896818,
        full_appended_leaf_universe=False, development_panel_reused=True,
        panel_scope=panel['metadata'], **FALSE)
    atomic_json(root/'launch.json', launch)
    (root/'checkpoints').mkdir()
    (root/'attempts').mkdir()


def load(root):
    root = Path(root)
    cfg = config()
    if root.parent.resolve() != Path(cfg['output_root']).resolve():
        raise ValueError('Exact authorized resumable root required')
    launch = worker.load(root)  # Original complete query/source/code checks, no targets.
    if launch['execution_config'] != cfg:
        raise ValueError('Frozen original execution policy drift')
    return launch


def attempt_root(root, batch, attempt):
    worker.positions(batch)
    if type(attempt) is not int or not 0 <= attempt < MAX_ATTEMPTS:
        raise ValueError('Bounded typed attempt required')
    return Path(root)/'attempts'/f'batch{batch:03d}'/f'attempt{attempt:02d}'


def checkpoint_path(root, batch):
    worker.positions(batch)
    return Path(root)/'checkpoints'/f'batch{batch:03d}.json'


def validate_attempt(root, batch, attempt):
    launch = load(root)
    shard = attempt_root(root, batch, attempt)
    if json.loads((shard/'launch.json').read_text()) != launch:
        raise ValueError('Every attempt must use exact original launch, controls and hashes')
    m, payload = worker.stage(shard, batch)  # Original actual own exit0/PASS/fullhash/epoch check.
    return shard, m, payload


def commit_checkpoint(root, batch, attempt):
    shard, m, payload = validate_attempt(root, batch, attempt)
    parent = shard.parent
    history = slots(parent)
    if history != list(range(attempt+1)):
        raise ValueError('First completed valid attempt only; no arbitrary best-of selection')
    failures = []
    for previous in history[:-1]:
        old = attempt_root(root, batch, previous)
        directory = old/f'batch{batch:03d}'
        previous_launch = old/'launch.json'
        if previous_launch.exists() and json.loads(previous_launch.read_text()) != load(root):
            raise ValueError('Failed attempt launch drift cannot be hidden by retry')
        if (directory/'supervision.json').exists() and not (directory/'worker_stop.json').exists() and not (directory/'supervisor_stop.json').exists():
            raise ValueError('Cannot select a later attempt over an already successful producer')
        kind = failure_kind(directory)
        if kind.startswith('FATAL'):
            raise ValueError('Fatal failed attempt cannot be hidden by later success')
        failures.append(dict(attempt=previous, kind=kind,
            receipts=[record(p) for p in (previous_launch, directory/'preflight_stop.json', directory/'worker_stop.json', directory/'supervisor_stop.json') if p.exists()]))
    value = dict(status='COMMITTED_V349_BATCH_CHECKPOINT', batch=batch, attempt=attempt,
        launch=record(Path(root)/'launch.json'), shard_launch=record(shard/'launch.json'),
        manifest=record(shard/f'batch{batch:03d}'/'manifest.json'),
        supervision=record(shard/f'batch{batch:03d}'/'supervision.json'),
        payload=m['payload'], failures=failures, **FALSE)
    path = checkpoint_path(root, batch)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('Immutable successful checkpoint drift')
    else:
        atomic_json(path, value)
    return value, payload


def stage(root, batch):
    cp = json.loads(checkpoint_path(root, batch).read_text())
    if (cp['status'] != 'COMMITTED_V349_BATCH_CHECKPOINT' or cp['batch'] != batch or type(cp['batch']) is not int
            or any(cp.get(k) is not False for k in SAFE_FALSE)):
        raise ValueError('Complete protected checkpoint required')
    for ref in (cp['launch'], cp['shard_launch'], cp['manifest'], cp['supervision'], cp['payload']):
        verify(ref)
    for failure in cp['failures']:
        for ref in failure['receipts']: verify(ref)
    current, payload = commit_checkpoint(root, batch, cp['attempt'])
    if current != cp:
        raise ValueError('Original checkpoint and each producer must revalidate')
    return cp, payload


def execute(root, batch):
    path = checkpoint_path(root, batch)
    if path.exists():
        return stage(root, batch)  # No recomputation after verified checkpoint.
    parent = Path(root)/'attempts'/f'batch{batch:03d}'
    parent.mkdir(exist_ok=True)
    # A controller crash may occur AFTER the own exit receipt but BEFORE commit.
    for a in slots(parent):
        shard = attempt_root(root, batch, a)
        directory = shard/f'batch{batch:03d}'
        if (directory/'supervision.json').exists() and not (directory/'worker_stop.json').exists() and not (directory/'supervisor_stop.json').exists():
            return commit_checkpoint(root, batch, a)
        if failure_kind(directory).startswith('FATAL'):
            raise RuntimeError('STOP_FATAL_ATTEMPT_INTEGRITY_OR_UNKNOWN_FAILURE')
    while True:
        a = next_slot(parent)
        shard = attempt_root(root, batch, a); shard.mkdir()
        launch = load(root)
        atomic_json(shard/'launch.json', launch)
        directory = shard/f'batch{batch:03d}'; directory.mkdir()
        env = {**os.environ, 'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
               'RAYON_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'}
        try:
            try:
                write_json(directory/'preflight.json', worker.quiet(30, 50))
            except RuntimeError as exc:
                atomic_json(directory/'preflight_stop.json', dict(reason=str(exc), no_worker_started=True))
                raise
            with (directory/'worker.log').open('x') as log:
                receipt = worker.supervise([sys.executable, '-u', '-m',
                    'scripts.retrieval_v344_corroborated_population', 'worker',
                    '--output', str(shard), '--batch', str(batch)], log, env,
                    directory/'supervisor_stop.json')
            atomic_json(directory/'supervision.json', receipt)
        except (KeyboardInterrupt, SystemExit):
            raise  # Human cancellation is never a transient process failure.
        except Exception:
            load(root)  # Drift is fatal even when it accompanied a resource failure.
            kind = failure_kind(directory)
            if kind.startswith('FATAL') or kind == 'UNCONFIRMED_INTERRUPTED_ATTEMPT_NEVER_ADMIT':
                # Only crash recovery can classify an abandoned unconfirmed
                # attempt. A live unknown exception is NOT permission to retry.
                raise
            print(json.dumps(dict(retry_batch=batch, failed_attempt=a, reason=kind,
                                  attempts_remaining=MAX_ATTEMPTS-a-1)), flush=True)
            if a+1 < MAX_ATTEMPTS:
                time.sleep(10)
            continue
        return commit_checkpoint(root, batch, a)


def admission(root):
    launch = load(root)
    rows, pairs, epochs, reports, checkpoints = [], [], [], [], []
    for b in range(500):
        cp, payload = stage(root, b)
        rows.extend(payload['rows']); epochs.extend(payload['source_epochs'])
        m = json.loads(verify(cp['manifest']).read_text()); reports.append(m['resources'])
        pairs.append(cp['manifest']); checkpoints.append(cp)
    valid_rows(rows, launch['queries'], list(range(2000)))
    epochs = worker.canonical_epochs(epochs); worker.check_epochs(epochs)
    return dict(rows=rows, pairs=pairs, source_epochs=epochs,
        peak_worker_rss_bytes=max(r['peak_rss_bytes'] for r in reports),
        worker_seconds=sum(r['wall_seconds'] for r in reports),
        retry_summary=dict(successful_producers=500,
            failed_attempts=sum(len(c['failures']) for c in checkpoints),
            controller_runs=len(list((Path(root)/'controllers').glob('run*.json'))),
            failure_policy='STOP_KEEP_DENOMINATOR', failed_attempt_time_excluded_from_success_latency=True))


def sealed(root):
    root = Path(root); launch = load(root)
    m = json.loads((root/'manifest.json').read_text())
    if (m.get('status') != STATUS or m.get('queries') != 2000 or type(m.get('queries')) is not int
            or m.get('population') != 2000 or type(m.get('population')) is not int
            or m.get('full_appended_leaf_universe') is not False
            or m.get('all2000_outputs_sealed') is not True or m.get('initial100_emitted') is not True
            or any(m.get(k) is not False for k in SAFE_FALSE)
            or m['launch'] != record(root/'launch.json')):
        raise ValueError('Complete checkpointed2000 generation, never LIVE/STOP/partial required')
    value = admission(root)
    outputs = json.loads(verify(m['outputs']).read_text())
    if (m['outputs'] != record(root/'outputs.json') or outputs['rows'] != value['rows']
            or outputs['source_epochs'] != value['source_epochs'] or m['pairs'] != value['pairs']
            or m['peak_worker_rss_bytes'] != value['peak_worker_rss_bytes']
            or any(outputs.get(k) is not False for k in SAFE_FALSE)
            or m['worker_seconds'] != value['worker_seconds']
            or m['retry_summary'] != value['retry_summary']):
        raise ValueError('Every successful checkpoint and complete output must revalidate')
    return launch, m, outputs


def run(root):
    if not root.exists(): init(root)
    with controller_lock(root):
        load(root)
        for name in ('checkpoints', 'attempts'):
            (root/name).mkdir(exist_ok=True)  # Recover crash just after atomic launch.
        if (root/'manifest.json').exists():
            sealed(root); return
        controllers = root/'controllers'; controllers.mkdir(exist_ok=True)
        index = len(list(controllers.glob('run*.json')))
        if index >= MAX_CONTROLLER_RUNS:
            raise RuntimeError('STOP_CONTROLLER_RESTARTS_EXHAUSTED')
        if index:
            # Only the still-live outer supervisor can authorize restart after
            # cleaning the old child's NEW-session group. Manual resume alone
            # cannot prove the former controller left no orphan producer.
            restart = json.loads((root/'restarts'/f'restart{index:02d}.json').read_text())
            previous = json.loads((controllers/f'run{index-1:02d}.json').read_text())
            if (restart.get('launch') != record(root/'launch.json')
                    or restart.get('supervisor_pid') != os.getppid()
                    or restart.get('previous_controller_pid') != previous['pid']
                    or restart.get('own_child_exit_confirmed') is not True
                    or restart.get('owned_group_cleanup_completed') is not True
                    or restart.get('exit_code') not in (-9, -15)):
                raise ValueError('Owned cleaned controller restart authority required')
        atomic_json(controllers/f'run{index:02d}.json', dict(pid=os.getpid(), time=time.time()))
        began = time.monotonic(); count = 0
        try:
            for b in range(500):
                cp, payload = execute(root, b)
                count += len(payload['rows'])
                print(json.dumps(dict(generated=count, resumed_checkpoint=cp['attempt'], batch=b)), flush=True)
            value = admission(root)
            atomic_json(root/'outputs.json', dict(rows=value['rows'], source_epochs=value['source_epochs'], **FALSE))
            atomic_json(root/'manifest.json', dict(status=STATUS, queries=2000, population=2000,
                all2000_outputs_sealed=True, initial100_emitted=True, launch=record(root/'launch.json'),
                outputs=record(root/'outputs.json'), pairs=value['pairs'],
                peak_worker_rss_bytes=value['peak_worker_rss_bytes'], worker_seconds=value['worker_seconds'],
                retry_summary=value['retry_summary'], wall_seconds=time.monotonic()-began,
                native_leaf_sites=43896818, full_appended_leaf_universe=False,
                native_segments_full_hash_verified=False, **FALSE))
            sealed(root)
        except BaseException as exc:
            # A killed controller cannot write this: outer supervisor may resume.
            # A declared terminal fault MUST NOT be restarted by the supervisor.
            atomic_json(root/'status.json', dict(status='STOP', reason=str(exc),
                completed=count, seconds=time.monotonic()-began, **FALSE))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('run','workflow'))
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.output)
    if args.mode == 'workflow':
        from scripts import evaluate_retrieval_v351_postseal as evaluation
        output = Path(evaluation.config()['output_root'])/args.output.name
        if output.exists():
            evaluation.admission(output)  # Refuse STOP/partial target phase, never rerun it.
        else:
            evaluation.run(args.output, output)
