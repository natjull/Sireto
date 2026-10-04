"""Two physical full-national name queries, shared reads, complete after-hash."""
import argparse
import gc
import json
import os
from pathlib import Path
import resource
import sys
import time

import psutil

from scripts import probe_retrieval_v187_staged_initial as original
from src.xgb_matcher import retrieval_query_population_v192 as queries
from src.xgb_matcher import retrieval_postscore_audit_v187 as hashes
from src.xgb_matcher.retrieval_memory_policy_v153 import Budget, LIMITS, POLICY, quiet, validate_report
from src.xgb_matcher.retrieval_address_authority_v168 import parent_exit
from src.xgb_matcher.retrieval_supervisor_v164 import supervise
from src.xgb_matcher.retrieval_v20_workbench import ROOT, record, verify, write_json

CONFIG = ROOT / 'config/retrieval_v193_names_pair.json'
STATUS = 'SEALED_V193_TWO_QUERY_COMPLETE_NATIONAL_NAME_PROOFS_COST_ONLY'


def config():
    cfg = json.loads(CONFIG.read_text())
    if (cfg['resources'] != LIMITS or cfg['quiet_seconds'] != 30
            or cfg['positions'] != [0, 1] or any(type(i) is not int for i in cfg['positions'])
            or cfg['queries'] != 128 or type(cfg['queries']) is not int
            or json.loads(verify(cfg['base_config']).read_text()) != original.config()):
        raise ValueError('Exact fixed physical pair inside all128 population and unchanged sources/budgets required')
    return cfg


def inputs(cfg):
    source = original.config()
    old, unused, models, geo = original.inputs(source)  # ALL128 typed/folds/heads before selection.
    pc = json.loads(verify(cfg['population_config']).read_text())
    if pc['query_only_launch'] != source['query_only_launch'] or pc['queries'] != 128 or pc['batch_width'] != 2:
        raise ValueError('Same full128 frozen query population required')
    population = queries.load(old, pc['encoder_launch'], pc['encoded_reference'])
    return source, population


def implementation():
    refs = original.implementation()
    for path in (Path(__file__), CONFIG, ROOT / 'docs/retrieval_v193_names_pair_contract.md',
                 ROOT / 'src/xgb_matcher/retrieval_query_population_v192.py'):
        refs[str(path)] = record(path)
    return refs


def load(output, dest):
    launch = json.loads((output / 'launch.json').read_text())
    cfg = config()
    if (launch['config'] != cfg or any(launch.get(k) is not v for k, v in original.FLAGS.items())
            or launch['old_candidates_opened'] or (output / 'status.json').exists()
            or any(output.glob('*/worker_stop.json')) or any(output.glob('*/supervisor_stop.json'))
            or not output.resolve().is_relative_to(Path(cfg['output_root']).resolve())):
        raise ValueError('Exact fresh safe193 launch; no STOP retry')
    for ref in launch['implementation'].values():
        verify(ref)
    gate = json.loads((dest / 'preflight.json').read_text())
    if gate['policy_id'] != POLICY or not gate['pass_system_quiet_policy'] or not gate['worker_swap_budget_changed']:
        raise ValueError('Actual own193 authorized pressure preflight required')
    return launch


def stage(output, phase):
    if phase not in ('names', 'audit'):
        raise ValueError('Explicit full physical193 phase required')
    dest = output / phase
    if any((dest / n).exists() for n in ('worker_stop.json', 'supervisor_stop.json', 'status.json')):
        raise ValueError('STOP193_STAGE_DO_NOT_USE')
    parent_exit(dest)
    m = json.loads((dest / 'manifest.json').read_text())
    validate_report(m['resources'])
    if (m['status'] != 'QUARANTINED_V193_'+phase.upper() or m['launch'] != record(output / 'launch.json')
            or m['pending_source_hash'] is not (phase == 'names')
            or any(m.get(k) is not v for k, v in original.FLAGS.items())
            or m['old_candidates_opened'] or m['initial100_emitted']):
        raise ValueError('Actual supervised whole pair phase, no previous candidates or GT required')
    if verify(m['preflight']).resolve() != (dest / 'preflight.json').resolve():
        raise ValueError('Own immutable193 preflight required')
    load(output, dest)
    for ref in m['artifacts'].values():
        verify(ref)
    return m


def audit(output, progress):
    named = stage(output, 'names')
    refs = hashes.references(json.loads(verify(named['artifacts']['targets']).read_text()))
    captured = hashes.canonical_epochs(json.loads(verify(named['artifacts']['epochs']).read_text()))
    if {r['path']: r for r in refs} != {e['reference']['path']: e['reference'] for e in captured}:
        raise ValueError('Every full original source/feature file and exact captured epoch required')
    hashes.check_epochs(captured)
    count = 0
    for i, ref in enumerate(refs):
        path = verify(ref)  # FULL bytes, not epochs or sampled hashes.
        count += path.stat().st_size
        if (i+1) % 256 == 0 or i+1 == len(refs):
            progress(dict(files_hashed=i+1, expected_files=len(refs), full_bytes_hashed=count))
    hashes.check_epochs(captured)
    return dict(names=record(output / 'names/manifest.json'), targets=named['artifacts']['targets'],
        epochs=named['artifacts']['epochs'], proofs=named['artifacts']['proofs'],
        files=len(refs), full_bytes_hashed=count, exact_targets_digest=hashes.targets_digest(refs),
        every_expected_source_full_hash_verified=True, initial100_emitted=False, **original.FLAGS)


def admission(output):
    named, checked = stage(output, 'names'), stage(output, 'audit')
    receipt = json.loads(verify(checked['artifacts']['receipt']).read_text())
    refs = hashes.references(json.loads(verify(named['artifacts']['targets']).read_text()))
    captured = hashes.canonical_epochs(json.loads(verify(named['artifacts']['epochs']).read_text()))
    proofs = json.loads(verify(named['artifacts']['proofs']).read_text())
    cfg = config()
    if (receipt['names'] != record(output / 'names/manifest.json')
            or any(receipt[k] != named['artifacts'][k] for k in ('targets', 'epochs', 'proofs'))
            or receipt['exact_targets_digest'] != hashes.targets_digest(refs)
            or receipt['files'] != len(refs) or type(receipt['files']) is not int
            or receipt['full_bytes_hashed'] != sum(Path(r['path']).stat().st_size for r in refs)
            or receipt['every_expected_source_full_hash_verified'] is not True
            or receipt['initial100_emitted'] is not False
            or any(receipt.get(k) is not v for k, v in original.FLAGS.items())
            or [r['position'] for r in proofs['rows']] != cfg['positions']
            or proofs['initial100_emitted'] is not False or proofs['truth_opened'] is not False):
        raise ValueError('Every current full source hash and exact whole pair receipt required')
    source, population = inputs(cfg)
    for result in proofs['rows']:
        row, unused = population.select(result['position'])
        if (result['query_id'] != row['query_id'] or result['fold'] != row['fold']
                or result['gather']['sites'] != source['sites']
                or result['gather']['every_original_site_gathered_once'] is not True
                or result['gather']['candidates_emitted'] is not False):
            raise ValueError('Exact query and ALL national sites for each whole name proof required')
        for ref in result['artifacts'].values():
            verify(ref)
    if proofs['rows'][0]['gather']['original_float64_gather_digest'] != cfg['query0_expected_gather_digest']:
        raise ValueError('Same entire first-query name arithmetic as independently sealed184 required')
    hashes.check_epochs(captured)
    return named, checked


def worker(output, phase):
    dest = output / phase
    if phase not in ('names', 'audit') or any(p.name not in ('preflight.json', 'worker.log') for p in dest.iterdir()):
        raise ValueError('Explicit fresh immutable193 physical worker required')
    owned, budget = [], None
    started = time.monotonic()
    def mark(event):
        print(json.dumps(dict(event, elapsed_seconds=time.monotonic()-started,
            rss_bytes=psutil.Process().memory_info().rss,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)), flush=True)
    try:
        with Budget(LIMITS) as budget:
            launch = load(output, dest)
            cfg = launch['config']
            if phase == 'audit':
                receipt = audit(output, mark)
                write_json(dest / 'receipt.json', receipt)
                artifacts = dict(receipt=record(dest / 'receipt.json'))
                trace = dict(files=receipt['files'], full_bytes_hashed=receipt['full_bytes_hashed'])
            else:
                source, population = inputs(cfg)
                mark(dict(checkpoint='LOAD_COMPLETE_OFFICIAL_NAME_SOURCES'))
                index = original.ScopedNamePages(source['names']); owned.append(index)
                reader = original.PagedReboundNames(source['joins']); owned.append(reader)
                if (reader.sites != source['sites'] or reader.metadata['sources']['catalog'] != source['catalog']
                        or index.names != reader.names or index.source != reader.source):
                    raise ValueError('Same complete official alias/parent/site namespace required')
                pc = json.loads(verify(cfg['population_config']).read_text())
                refs = [index.source, *reader.refs, cfg['base_config'], cfg['population_config'],
                        pc['encoder_launch'], pc['encoded_reference'],
                        *[source[k] for k in ('names', 'joins', 'catalog', 'query_only_launch')]]
                old = json.loads(verify(source['query_only_launch']).read_text())
                safe_models = json.loads(verify(old['models']).read_text())
                refs.extend([old['query_vectors'], old['models'], old['model_manifest'], old['geo_queries'],
                             *[m['decoder_parameter'] for m in safe_models.values()]])
                for access in index.metadata['access']:
                    refs.extend(access['artifacts'].values())
                refs = hashes.references(refs)
                captured = hashes.canonical_epochs(original.epochs(refs))
                outputs = []
                for position in cfg['positions']:
                    row, unused = population.select(position)
                    hashes.check_epochs(captured)
                    directory = dest / ('query'+str(position))
                    mark(dict(checkpoint='WHOLE_NAME_QUERY', position=position))
                    snap, trace = reader.query(index, row['crm_name'], directory / 'scratch',
                                              source['max_proof_query_seconds'], source['lexical_chunk'])
                    owned.append(snap)
                    began = time.monotonic()
                    def guard():
                        if time.monotonic()-began > source['max_proof_query_seconds']:
                            raise TimeoutError('STOP193_WHOLE_FLATTEN_COST')
                    flat, gather = original.flatten_names(snap, directory / 'flat', guard, source['chunk'])
                    owned.append(flat)
                    artifacts = dict(coverage=flat.coverage.reference, exact=flat.exact.reference)
                    outputs.append(dict(position=position, query_id=row['query_id'], fold=row['fold'],
                        artifacts=artifacts, query=trace, gather=gather, flatten_seconds=time.monotonic()-began))
                    refs = hashes.references([*refs, *artifacts.values()])
                    captured = hashes.canonical_epochs([*captured, *original.epochs(artifacts.values())])
                    flat.verify_after(); flat.close(); owned.remove(flat)
                    snap.close(remove=True); owned.remove(snap)  # Own temporary scratch only.
                    rotation = index.rotate(); reader.rotate_keys()
                    hashes.check_epochs(captured)
                    mark(dict(checkpoint='COMPLETE_QUERY_OWN_PAGES_RELEASED', position=position, rotation=rotation))
                    gc.collect()
                mark(dict(checkpoint='QUARANTINE_PENDING_SEPARATE_FULL_SOURCE_HASH'))
                write_json(dest / 'proofs.json', dict(rows=outputs, population=128, initial100_emitted=False,
                    truth_opened=False, recall_measured=False))
                write_json(dest / 'targets.json', refs)
                write_json(dest / 'epochs.json', captured)
                artifacts = {k: record(dest / (k+'.json')) for k in ('proofs', 'targets', 'epochs')}
                trace = dict(positions=cfg['positions'], files=len(refs), sites_per_query=source['sites'],
                             original_aliases=index.names, source_hash_audit_pending=True)
                for owner in reversed(owned): owner.close()
                owned.clear()
            load(output, dest)
        write_json(dest / 'manifest.json', dict(status='QUARANTINED_V193_'+phase.upper(),
            launch=record(output / 'launch.json'), preflight=record(dest / 'preflight.json'),
            pending_source_hash=phase == 'names', artifacts=artifacts, trace=trace, resources=budget.report(),
            initial100_emitted=False, old_candidates_opened=False, ready_for_fold1=False, **original.FLAGS))
    except BaseException as exc:
        write_json(dest / 'worker_stop.json', dict(status='STOP', reason=str(exc), initial100_emitted=False,
            resources=budget.report() if budget is not None and hasattr(budget, 'elapsed') else None,
            old_candidates_opened=False, **original.FLAGS))
        raise
    finally:
        for owner in reversed(owned): owner.close()


def run(output):
    cfg = config()
    inputs(cfg)
    if output.exists() or not output.resolve().is_relative_to(Path(cfg['output_root']).resolve()):
        raise ValueError('Fresh CATNAT-only immutable193 pair root required')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'launch.json', dict(config=cfg, implementation=implementation(),
                                          old_candidates_opened=False, **original.FLAGS))
    began = time.monotonic()
    try:
        for phase in ('names', 'audit'):
            gate = quiet(cfg['quiet_seconds'], 50, 55)
            dest = output / phase; dest.mkdir()
            write_json(dest / 'preflight.json', gate)
            with (dest / 'worker.log').open('x') as log:
                s = supervise([sys.executable, '-u', '-m', 'scripts.probe_retrieval_v193_names_pair',
                    'worker', '--output', str(output), '--phase', phase], log,
                    {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}, dest / 'supervisor_stop.json')
            write_json(dest / 'supervision.json', s)
            m = stage(output, phase)
            print(json.dumps(dict(completed=phase, resources=m['resources'], trace=m['trace'])), flush=True)
        named, checked = admission(output)
        total = named['resources']['wall_seconds']+checked['resources']['wall_seconds']
        write_json(output / 'manifest.json', dict(status=STATUS, launch=record(output / 'launch.json'),
            names=record(output / 'names/manifest.json'), audit=record(output / 'audit/manifest.json'),
            outputs=named['artifacts']['proofs'], queries=2, population=128, initial100_emitted=False,
            all_load_query_flatten_rotation_and_full_hash_worker_seconds=total,
            worker_seconds_per_name_query_including_shared_full_audit=total/2,
            elapsed_including_quiet_and_parent_seconds=time.monotonic()-began,
            source_complete_name_proofs=True, complete_retrieval_index=False, full_query_population=False,
            slo_validated=False, ready_for_fold1=False, downstream_models_changed=False,
            old_candidates_opened=False, **original.FLAGS))
        print(STATUS, flush=True)
    except BaseException as exc:
        write_json(output / 'status.json', dict(status='STOP', reason=str(exc), initial100_emitted=False,
                                              old_candidates_opened=False, **original.FLAGS))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('run', 'worker'))
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--phase', choices=('names', 'audit'))
    args = p.parse_args()
    if args.mode == 'worker' and args.phase is None: p.error('Explicit full phase required')
    (run(args.output) if args.mode == 'run' else worker(args.output, args.phase))
