"""ALL128 exact initial100 metric only AFTER complete V206 and supervised permission."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq

from src.xgb_matcher import retrieval_evaluation_authority_v207 as authority
from src.xgb_matcher import retrieval_pilot_metrics_v200 as metrics
from src.xgb_matcher import retrieval_postscore_audit_v187 as hashes
from src.xgb_matcher.retrieval_extended_catalog_v136 import MISSING
from src.xgb_matcher.retrieval_memory_policy_v153 import Budget, LIMITS, POLICY, quiet, validate_report
from src.xgb_matcher.retrieval_address_authority_v168 import parent_exit
from src.xgb_matcher.retrieval_supervisor_v164 import supervise
from src.xgb_matcher.retrieval_v20_workbench import ROOT, record, verify, write_json

CONFIG = ROOT / 'config/retrieval_v207_postseal_evaluation.json'
STATUS = 'MEASURED_V207_EXACT_INITIAL100_ALL128_REUSED_PILOT'
TARGET_EVALUATION = dict(path='/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v34_fast_resume/full_job/evaluation_resume_20261001/manifest.json',
    sha256='b6d7123d2fb33616c76dc14ed4277552a7696d2bdb17b56f540b7f76edfec3c9')
TARGET_LEDGER = dict(path='/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/retrieval_all_human_v12_label_ledger/1a99b7cb1db5a1692e39/label_ledger.parquet',
    sha256='24f5928cbf2669b0a79ac00a284d14e5fe2288abb5a4ddcbd77a46677c315720')
SAFE_FALSE = ('ready_for_fold1', 'fold0_opened', 'fold1_opened', 'downstream_models_changed', 'old_candidates_opened')
FALSE_FLAGS = {k: False for k in SAFE_FALSE}


def config():
    c = json.loads(CONFIG.read_text())
    if (set(c) != {'generation_config', 'output_root', 'evaluation', 'ledger', 'queries', 'quiet_seconds', 'resources'}
            or c['resources'] != LIMITS or c['quiet_seconds'] != 30
            or type(c['queries']) is not int or c['queries'] != 128
            or c['evaluation'] != TARGET_EVALUATION or c['ledger'] != TARGET_LEDGER
            or verify(c['generation_config']).resolve() != authority.generation.CONFIG.resolve()
            or json.loads(verify(c['generation_config']).read_text()) != authority.generation.config()):
        raise ValueError('Frozen safe207 target specs and actual206 backend with unchanged budgets required')
    return c  # Target paths are specs ONLY; no target verification/read here.


def checked(ref):
    if (set(ref) not in ({'path', 'sha256'}, {'path', 'sha256', 'size_bytes'})
            or ('size_bytes' in ref and (type(ref['size_bytes']) is not int or ref['size_bytes'] < 0))):
        raise ValueError('Typed original reference with optional exact size required')
    p = verify(ref)
    if 'size_bytes' in ref and p.stat().st_size != ref['size_bytes']:
        raise ValueError('Original source/reference size mismatch')
    return p


def implementation():
    paths = (Path(__file__), CONFIG, ROOT / 'docs/retrieval_v207_postseal_evaluation_contract.md',
             ROOT / 'src/xgb_matcher/retrieval_evaluation_authority_v207.py',
             ROOT / 'src/xgb_matcher/retrieval_pilot_metrics_v200.py',
             ROOT / 'docs/siret_operational_equivalence_policy.md')
    return {str(p): record(p) for p in paths}


def load(output, dest=None):
    launch = json.loads((output / 'launch.json').read_text()); c = config()
    if (launch['config'] != c or any(launch.get(k) is not False for k in SAFE_FALSE)
            or launch.get('truth_opened') is not False
            or (output / 'status.json').exists()
            or any(output.glob('*/worker_stop.json')) or any(output.glob('*/supervisor_stop.json'))
            or not output.resolve().is_relative_to(Path(c['output_root']).resolve())):
        raise ValueError('Own fresh closed-label207 launch and no STOP retry required')
    for ref in launch['implementation'].values(): checked(ref)
    root, unused = authority.header(checked(launch['generation']).parent)
    if record(root / 'manifest.json') != launch['generation']:
        raise ValueError('Same complete original206 root required')
    if dest is not None:
        gate = json.loads((dest / 'preflight.json').read_text())
        if gate['policy_id'] != POLICY or gate['pass_system_quiet_policy'] is not True or gate['worker_swap_budget_changed'] is not True:
            raise ValueError('Actual own authorized207 preflight required')
    return launch


def stage(output, phase):
    if phase not in ('authority', 'metrics'): raise ValueError('Explicit isolated evaluation phase required')
    dest = output / phase; parent_exit(dest)
    m = json.loads((dest / 'manifest.json').read_text()); validate_report(m['resources'])
    launch = load(output, dest)
    if (m['status'] != 'QUARANTINED_V207_'+phase.upper()
            or m['launch'] != record(output / 'launch.json') or m['generation'] != launch['generation']
            or checked(m['preflight']).resolve() != (dest / 'preflight.json').resolve()
            or m['truth_opened'] is not (phase == 'metrics')
            or m['recall_measured'] is not (phase == 'metrics')
            or any(m.get(k) is not False for k in SAFE_FALSE)
            or set(m['artifacts']) != ({'permission'} if phase == 'authority' else {'report'})):
        raise ValueError('Actual successful own207 phase bound to complete206 and protected labels required')
    for ref in m['artifacts'].values(): checked(ref)
    return m


def permitted(output):
    m = stage(output, 'authority')  # Actual exit0/PASS/hash, NOT a self-reported permission.
    permission = json.loads(checked(m['artifacts']['permission']).read_text())
    if permission['generation'] != m['generation']:
        raise ValueError('Actual permission for THIS complete generation required')
    authority.recheck(permission)
    return permission, m['artifacts']['permission']


def selected(ref, controls, required, optional=()):
    """Only post-permission caller; push BOTH safe-fold and exact128 filters into Arrow."""
    ids = [r['query_id'] for r in controls]
    if (len(controls) != 128 or len(set(ids)) != 128 or any(type(r['fold']) is not int or r['fold'] not in (2, 3, 4) for r in controls)):
        raise ValueError('Complete128 safe-fold controls required before target table read')
    p = checked(ref); names = pq.read_schema(p).names
    if not set(required) <= set(names): raise ValueError('Original required target columns missing')
    columns = list(dict.fromkeys([*required, *[k for k in optional if k in names]]))
    return pq.read_table(p, columns=columns, filters=[('fold', 'in', [2, 3, 4]), ('query_id', 'in', ids)],
                         use_threads=False).to_pylist()


def targets(c, permission):
    ev = json.loads(checked(c['evaluation']).read_text())
    side = json.loads(checked(ev['label_side']).read_text())
    if (side.get('never_used_in_retrieval') is not True or type(side.get('queries')) is not int
            or side['queries'] != 23571 or hashes.canonical(side['ledger']) != hashes.canonical(c['ledger'])
            or set(side['files']) != {'official_catalog_sidecar.parquet', 'acceptable_operational.parquet'}
            or any(side['files'][k] != ev['files'][k] for k in side['files'])
            or checked(side['policy']).resolve() != (ROOT / 'docs/siret_operational_equivalence_policy.md').resolve()):
        raise ValueError('Original sealed official target-only sidecar/ledger/policy required')
    refs = hashes.references([c['evaluation'], c['ledger'], ev['label_side'], side['policy'], *side['files'].values()])
    epochs = hashes.canonical_epochs(authority.generation.original.epochs(refs))
    for ref in refs: checked(ref)  # Every immutable target/side file FULL hashed, no sampling.
    controls = permission['controls']
    labels = selected(c['ledger'], controls, ('query_id', 'fold', 'human_ground_truth_siret', 'human_ground_truth_siren'), metrics.SEGMENTS)
    official = selected(side['files']['official_catalog_sidecar.parquet'], controls,
        ('query_id', 'fold', 'gt_siret', 'gt_siren', 'official_catalog_present'), ('exact_state',))
    acceptable = selected(side['files']['acceptable_operational.parquet'], controls,
        ('query_id', 'fold', 'siret', 'exact_truth', 'same_site_evidence'), ('administrative_state',))
    hashes.check_epochs(epochs)
    return labels, official, acceptable, refs, epochs, ev.get('excluded_missing_human_labels')


def current_membership(keys, labels, controls):
    scope = {r['query_id']: r['fold'] for r in controls}; rows = metrics.keyed(labels, scope)
    values = np.array([int(metrics.identity(rows[r['query_id']]['human_ground_truth_siret'], 14)) for r in controls], dtype=np.uint64)
    positions = keys.site_positions(values)
    if positions.dtype != np.uint32 or positions.shape != (128,) or np.any((positions != MISSING) & (positions >= keys.sites)):
        raise ValueError('Actual whole current namespace membership, not historical sidecar absence required')
    return {r['query_id']: bool(pos != MISSING) for r, pos in zip(controls, positions)}


def evaluate(output):
    permission, permission_ref = permitted(output)  # FIRST; no target verification/read before this.
    c = config(); keys = authority.generation.original.RotatedKeys(permission['current_catalog'])
    try:
        # Constructor/audit pages rotated before metadata/targets, all44M identities retained.
        if keys.sites != authority.generation.config()['sites']:
            raise ValueError('Same full current generation namespace required')
        labels, official, acceptable, refs, epochs, excluded = targets(c, permission)
        present = current_membership(keys, labels, permission['controls'])
        outputs = json.loads(checked(permission['outputs']).read_text())
        result = metrics.evaluate(outputs['rows'], permission['controls'], labels, official, acceptable, present)
        keys.verify_after(); hashes.check_epochs(epochs)
        for ref in refs: checked(ref)
        authority.recheck(permission)
        result.update(status=STATUS, permission=permission_ref, generation=permission['generation'],
            target_references=refs, target_epochs=epochs, excluded_missing_human_labels_historical_population=excluded,
            truth_opened=True, recall_measured=True, initial100_cap=100, old_candidates_opened=False,
            current_catalog=permission['current_catalog'], current_catalog_membership_checked=True,
            current_catalog_membership_source='EXACT_POSTSEAL_TYPED_LOOKUP_IN_ALL_V141_SITE_IDENTITIES',
            latency_scope='NOT_MEASURED_BY_EVALUATOR; GENERATION_COST_IS_NOT_PRODUCTION_SLO')
        return result
    finally: keys.close()


def worker(output, phase):
    if phase not in ('authority', 'metrics'): raise ValueError('Explicit isolated207 worker required')
    dest = output / phase
    if any(p.name not in ('preflight.json', 'worker.log') for p in dest.iterdir()):
        raise ValueError('Fresh own207 worker, no overwrite or STOP retry')
    budget = None
    try:
        with Budget(LIMITS) as budget:
            launch = load(output, dest)
            if phase == 'authority':
                value = authority.authorize(checked(launch['generation']).parent); artifact = 'permission'
            else:
                value = evaluate(output); artifact = 'report'
            write_json(dest / (artifact+'.json'), value)
            load(output, dest)
        write_json(dest / 'manifest.json', dict(status='QUARANTINED_V207_'+phase.upper(),
            launch=record(output / 'launch.json'), generation=launch['generation'], preflight=record(dest / 'preflight.json'),
            artifacts={artifact: record(dest / (artifact+'.json'))}, resources=budget.report(),
            truth_opened=phase == 'metrics', recall_measured=phase == 'metrics', **FALSE_FLAGS))
    except BaseException as exc:
        write_json(dest / 'worker_stop.json', dict(status='STOP', reason=str(exc),
            resources=budget.report() if budget is not None and hasattr(budget, 'elapsed') else None,
            truth_may_have_been_opened=phase == 'metrics', recall_admitted=False, **FALSE_FLAGS))
        raise


def admission(output):
    permission, permission_ref = permitted(output); measured = stage(output, 'metrics')
    result = json.loads(checked(measured['artifacts']['report']).read_text())
    if (result['status'] != STATUS or result['permission'] != permission_ref or result['generation'] != permission['generation']
            or result['primary']['queries'] != 128 or type(result['primary']['queries']) is not int
            or result['truth_opened'] is not True or result['recall_measured'] is not True
            or any(result.get(k) is not False for k in SAFE_FALSE)
            or result['full_oof_population'] is not False or result['slo_validated'] is not False
            or result['fine_loss_attribution_measured'] is not False
            or result['current_catalog_membership_checked'] is not True or result['initial100_cap'] != 100):
        raise ValueError('Exact allhuman128 current metric and no false READY/OOF/SLO claim required')
    for ref in result['target_references']: checked(ref)
    hashes.check_epochs(result['target_epochs']); authority.recheck(permission)
    return measured


def run(generation, output):
    c = config(); root, unused = authority.header(generation)  # Refuse LIVE/STOP before output or targets.
    if output.exists() or not output.resolve().is_relative_to(Path(c['output_root']).resolve()):
        raise ValueError('Fresh CATNAT-only207 evaluation root required')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'launch.json', dict(config=c, implementation=implementation(), generation=record(root / 'manifest.json'),
        truth_opened=False, **FALSE_FLAGS))
    try:
        for phase in ('authority', 'metrics'):
            gate = quiet(c['quiet_seconds'], 50, 55); dest = output / phase; dest.mkdir()
            write_json(dest / 'preflight.json', gate)
            with (dest / 'worker.log').open('x') as log:
                s = supervise([sys.executable, '-u', '-m', 'scripts.evaluate_retrieval_v207_postseal',
                    'worker', '--output', str(output), '--phase', phase], log,
                    {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}, dest / 'supervisor_stop.json')
            write_json(dest / 'supervision.json', s); m = stage(output, phase)
            print(json.dumps(dict(completed=phase, resources=m['resources'])), flush=True)
        measured = admission(output)
        write_json(output / 'manifest.json', dict(status=STATUS, launch=record(output / 'launch.json'),
            generation=record(root / 'manifest.json'), authority=record(output / 'authority/manifest.json'),
            metrics=record(output / 'metrics/manifest.json'), report=measured['artifacts']['report'],
            queries=128, population=128, truth_opened=True, recall_measured=True, **FALSE_FLAGS))
        report = json.loads(checked(measured['artifacts']['report']).read_text())
        print(json.dumps({k: report[k] for k in ('primary', 'loss_counts', 'decision')}), flush=True)
    except BaseException as exc:
        write_json(output / 'status.json', dict(status='STOP', reason=str(exc), recall_admitted=False,
            truth_may_have_been_opened=(output / 'metrics').exists(), **FALSE_FLAGS)); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('run', 'worker')); p.add_argument('--output', required=True, type=Path)
    p.add_argument('--generation', type=Path); p.add_argument('--phase', choices=('authority', 'metrics')); a = p.parse_args()
    if a.mode == 'run' and a.generation is None: p.error('Explicit complete original generation required')
    if a.mode == 'worker' and a.phase is None: p.error('Explicit isolated phase required')
    run(a.generation, a.output) if a.mode == 'run' else worker(a.output, a.phase)
