"""One frozen address query over complete V177 aliases, not initial100/recall."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import duckdb
import numpy as np
import psutil

from scripts.probe_retrieval_v138_national_names import query_rows
from scripts import compile_retrieval_v177_packed_addresses as backend
from src.xgb_matcher.retrieval_address_authority_v168 import parent_exit
from src.xgb_matcher.retrieval_memory_policy_v153 import Budget, quiet, LIMITS, POLICY, validate_report
from src.xgb_matcher.retrieval_paged_addresses_v178 import PagedAddresses
from src.xgb_matcher.retrieval_proof_bridge_v137 import PROTECTED
from src.xgb_matcher.retrieval_supervisor_v164 import supervise
from src.xgb_matcher.retrieval_v20_workbench import ROOT, record, verify, write_json

CONFIG = ROOT / 'config/retrieval_v179_paged_addresses.json'
STATUS = 'SEALED_V179_COMPLETE_OFFICIAL_ADDRESS_SNAPSHOT_COST_ONLY'
FLAGS = {k: k == 'crm_opened' for k in PROTECTED}


def base():
    cfg = json.loads(CONFIG.read_text())
    if (cfg['resources'] != LIMITS or cfg['numpy_version'] != np.__version__
            or cfg['duckdb_version'] != duckdb.__version__ or type(cfg['queries']) is not int or cfg['queries'] != 1
            or any(type(cfg[k]) is not int or cfg[k] <= 0 for k in
                   ('aliases', 'sites', 'facts', 'postings', 'chunk', 'max_query_disk_bytes'))):
        raise ValueError('Exact whole-source single-query authorized memory policy required')
    return cfg


def inputs(cfg):
    original = json.loads(verify(cfg['query_only_launch']).read_text())
    rows = query_rows(original, cfg['queries'])
    # Check ALL128 rows before the fixed prefix, never follow activity proofs.
    strings = ('query_id', 'crm_name', 'crm_number', 'crm_suffix', 'crm_street', 'crm_postcode', 'crm_insee')
    if any(type(q['fold']) is not int or any(not isinstance(q[k], str) for k in strings) for q in original['queries']):
        raise ValueError('Frozen query-only typed address fields required')
    return rows


def freeze():
    cfg = base()
    inputs(cfg)
    access = record(cfg['address_access_path'])
    m = json.loads(verify(access).read_text())
    source = record(verify(cfg['source_launch']))
    if (m['status'] != backend.STATUS or m['launch'] != source
            or any(m[k] != cfg[k] for k in ('aliases', 'sites', 'facts', 'postings'))
            or len(m['access']) != 256 or len(m['phase_authorities']) != 256
            or any(m.get(k) is not False for k in PROTECTED)
            or m['complete_retrieval_index'] or m['ready_for_fold1']):
        raise ValueError('Complete exact pinned V177 source seal required before probe')
    parent_exit(Path(access['path']).parent / 'seal')
    validate_report(m['seal_resources'])
    return {**cfg, 'addresses': access}


def implementation():
    refs = backend.implementation()
    names = ('retrieval_paged_addresses_v178', 'retrieval_paged_names_v170', 'retrieval_address_reverse_v147',
             'retrieval_national_names_v138', 'retrieval_memory_policy_v153', 'retrieval_supervisor_v164')
    files = [Path(__file__), CONFIG, ROOT / 'docs/retrieval_v179_paged_addresses_contract.md',
             ROOT / 'docs/retrieval_v178_paged_addresses_contract.md',
             ROOT / 'scripts/probe_retrieval_v138_national_names.py',
             *[ROOT / ('src/xgb_matcher/'+name+'.py') for name in names]]
    refs.update({str(p): record(p) for p in files})
    return refs


def load(output):
    l = json.loads((output / 'launch.json').read_text())
    cfg = l['config']
    if (cfg != freeze() or not output.resolve().is_relative_to(Path(cfg['output_root']).resolve())
            or any(l.get(k) is not v for k, v in FLAGS.items()) or l['old_candidates_opened']):
        raise ValueError('Unchanged frozen complete source/query-only launch required')
    for ref in l['implementation'].values():
        verify(ref)
    if verify(l['preflight']).resolve() != (output / 'preflight.json').resolve():
        raise ValueError('Actual own bounded preflight required')
    gate = json.loads(verify(l['preflight']).read_text())
    if (gate['policy_id'] != POLICY or not gate['pass_system_quiet_policy']
            or not gate['worker_swap_budget_changed']):
        raise ValueError('Exact authorized pressure policy required')
    return l, inputs(cfg)


def gather(snapshot, chunk, guard):
    digest = hashlib.sha256()
    positive = exact = 0
    for lo in range(0, snapshot.aliases, chunk):
        guard()
        hi = min(snapshot.aliases, lo + chunk)
        cov = snapshot.coverage[lo:hi]
        flags = snapshot.exact[lo:hi]
        if (cov.dtype != np.float64 or flags.dtype != np.bool_ or cov.shape != (hi-lo,)
                or flags.shape != cov.shape or not np.isfinite(cov).all()
                or np.any((cov < 0) | (cov > 1)) or np.any(flags & (cov != 1.))):
            raise ValueError('Every complete coherent original alias proof required')
        digest.update(cov.tobytes())
        digest.update(flags.tobytes())
        positive += int(np.count_nonzero(cov))
        exact += int(np.count_nonzero(flags))
    snapshot.verify_after()
    guard()
    return dict(aliases=snapshot.aliases, positive=positive, exact=exact, sha256=digest.hexdigest(),
                every_alias_gathered_once=True, every_immutable_query_vector_full_hash_verified=True,
                no_candidate_selection=True)


def worker(output):
    if any(p.name not in ('launch.json', 'preflight.json', 'worker.log') for p in output.iterdir()):
        raise ValueError('Immutable address probe already started, no retry')
    budget = index = snapshot = None
    stage = 'INPUTS'
    try:
        with Budget(LIMITS) as budget:
            launch, rows = load(output)
            cfg = launch['config']
            stage = 'LOAD_ADDRESS_INDEX'
            began = time.monotonic()
            index = PagedAddresses(cfg['addresses'])
            load_seconds = time.monotonic() - began
            load_rss = psutil.Process().memory_info().rss
            print(json.dumps(dict(stage='LOAD_COMPLETE', seconds=load_seconds, rss_bytes=load_rss)), flush=True)
            outputs = []
            for i, row in enumerate(rows):
                stage = 'QUERY_'+str(i)
                folder = output / 'queries' / f'query{i:04d}'
                began = time.monotonic()
                def guard():
                    if time.monotonic()-began > cfg['max_query_seconds']:
                        raise TimeoutError('STOP_COMPLETE_ADDRESS_PROBE_QUERY_COST')
                text = ' '.join(row[k] for k in ('crm_number', 'crm_suffix', 'crm_street')).strip()
                snapshot, trace = index.snapshot(text, folder, cfg['max_query_seconds'], cfg['chunk'])
                size = sum(p.stat().st_size for p in folder.iterdir() if p.is_file())
                if size > cfg['max_query_disk_bytes']:
                    raise ValueError('STOP_OWN_QUERY_DISK_BUDGET')
                print(json.dumps(dict(stage='SNAPSHOT_COMPLETE', trace=trace)), flush=True)
                stage = 'FULL_ALIAS_GATHER_'+str(i)
                gathered = gather(snapshot, cfg['chunk'], guard)
                snapshot.close(remove=True)
                snapshot = None
                guard()
                outputs.append(dict(query_id=row['query_id'], fold=row['fold'], trace=trace, gather=gathered,
                                    total_query_and_full_gather_seconds=time.monotonic()-began,
                                    query_disk_bytes=size, successful_owned_query_scratch_removed=True))
                print(json.dumps(outputs[-1]), flush=True)
            stage = 'FULL_ACCESS_REHASH'
            index.verify_after()
            load(output)
            index.close()
            index = None
        write_json(output / 'cost_outputs.json', dict(rows=outputs, initial100_emitted=False,
                                                     recall_measured=False, truth_opened=False))
        write_json(output / 'manifest.json', dict(status=STATUS, launch=record(output / 'launch.json'),
            outputs=record(output / 'cost_outputs.json'), source_addresses=cfg['addresses'], queries=len(outputs),
            aliases=cfg['aliases'], sites=cfg['sites'], facts=cfg['facts'], postings=cfg['postings'],
            resources=budget.report(), load_seconds=load_seconds, load_rss_bytes=load_rss,
            every_address_pack_full_hash_verified_before_and_after=True,
            initial100_emitted=False, complete_retrieval_index=False,
            score_stage_only_not_retrieval_latency=True, query_features_float64_unchanged=True,
            disk_cache_state='UNCONTROLLED_NOT_COLD_DISK', old_candidates_opened=False,
            downstream_models_changed=False, ready_for_fold1=False, **FLAGS))
    except BaseException as exc:
        write_json(output / 'worker_stop.json', dict(status='STOP', reason=str(exc), stage=stage,
            resources=budget.report() if budget is not None and hasattr(budget, 'elapsed') else None,
            initial100_emitted=False, old_candidates_opened=False, **FLAGS))
        raise
    finally:
        if snapshot is not None:
            snapshot.close()  # Failed-query files are preserved.
        if index is not None:
            index.close()


def run(output):
    if output.exists():
        raise FileExistsError('Immutable address probe root exists')
    cfg = freeze()
    if not output.resolve().is_relative_to(Path(cfg['output_root']).resolve()):
        raise ValueError('CATNAT-only fresh probe required')
    gate = quiet(cfg['quiet_seconds'], 50, 55)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'preflight.json', gate)
    write_json(output / 'launch.json', dict(config=cfg, preflight=record(output / 'preflight.json'),
        implementation=implementation(), old_candidates_opened=False, **FLAGS))
    try:
        with (output / 'worker.log').open('x') as log:
            s = supervise([sys.executable, '-u', '-m', 'scripts.probe_retrieval_v179_paged_addresses',
                'worker', '--output', str(output)], log,
                {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}, output / 'supervisor_stop.json')
        write_json(output / 'supervision.json', s)
        parent_exit(output)
        m = json.loads((output / 'manifest.json').read_text())
        validate_report(m['resources'])
        if m['status'] != STATUS or m['launch'] != record(output / 'launch.json'):
            raise ValueError('Actual supervised full address probe required')
        print(json.dumps(dict(status=STATUS, resources=m['resources'])), flush=True)
    except BaseException as exc:
        write_json(output / 'status.json', dict(status='STOP', reason=str(exc), old_candidates_opened=False, **FLAGS))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['run', 'worker'])
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    (run if a.mode == 'run' else worker)(a.output)
