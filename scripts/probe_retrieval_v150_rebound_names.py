"""Bounded frozen-query national name snapshots, not initial100 or recall."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import duckdb
import numpy as np

from scripts.probe_retrieval_v138_national_names import query_rows
from scripts.compile_retrieval_v149_name_rebind import load as source_load,implementation as source_implementation
from src.xgb_matcher.retrieval_national_names_v138 import NationalNames
from src.xgb_matcher.retrieval_rebound_names_v150 import ReboundNames
from src.xgb_matcher.retrieval_system_quiet_v143 import quiet
from src.xgb_matcher.retrieval_proof_bridge_v137 import PROTECTED
from src.xgb_matcher.retrieval_v20_workbench import Budget,ROOT,verify,record,write_json

CONFIG=ROOT/'config/retrieval_v150_rebound_names.json'
FLAGS={k:k=='crm_opened' for k in PROTECTED}
STATUS='SEALED_V150_ALL_OFFICIAL_REBOUND_NAME_SNAPSHOT_COST_ONLY'


def inputs(cfg):
    if (cfg['resources']['max_swapout_pages_delta']!=0 or np.__version__!=cfg['numpy_version']
        or duckdb.__version__!=cfg['duckdb_version']):raise ValueError('Pinned typed/storage versions and unchanged zero swap required')
    verify(cfg['names']);verify(cfg['joins'])
    source_load(Path(cfg['joins']['path']).parent)
    return query_rows(json.loads(verify(cfg['query_only_launch']).read_text()),cfg['queries'])


def implementation():
    refs=source_implementation()
    paths=[Path(__file__),CONFIG,ROOT/'docs/retrieval_v150_rebound_names.md',ROOT/'scripts/probe_retrieval_v138_national_names.py',
        ROOT/'src/xgb_matcher/retrieval_rebound_names_v150.py',ROOT/'src/xgb_matcher/retrieval_national_names_v138.py']
    refs.update({str(p):record(p) for p in paths});return refs


def load(output):
    l=json.loads((output/'launch.json').read_text());cfg=l['config']
    if (not output.resolve().is_relative_to(Path(cfg['output_root']).resolve())
        or any(l.get(k) is not v for k,v in FLAGS.items()) or l['old_candidates_opened']):
        raise ValueError('Protected immutable CATNAT-only query source required')
    for ref in l['implementation'].values():verify(ref)
    g=json.loads(verify(l['preflight']).read_text())
    if (verify(l['preflight']).resolve()!=(output/'preflight.json').resolve()
        or not g['pass_system_quiet_policy'] or g['worker_swap_budget_changed']):
        raise ValueError('Actual unchanged system gate required')
    return l,g,inputs(cfg)


def worker(output):
    if any(p.name not in ('launch.json','preflight.json','worker.log') for p in output.iterdir()):
        raise ValueError('Immutable name snapshot probe already started, no retry')
    budget=None;index=None;names=None
    try:
        cfg=json.loads((output/'launch.json').read_text())['config']
        with Budget(cfg['resources']) as budget:
            launch,gate,rows=load(output)
            if budget.before!=gate['final_snapshot']['swapouts']:raise RuntimeError('STOP_SYSTEM_CHANGED_SINCE_QUIET_GATE')
            began=time.monotonic();index=NationalNames(cfg['names']);names=ReboundNames(cfg['joins'])
            load_seconds=time.monotonic()-began;outputs=[]
            for row in rows:
                snapshot,trace=names.query(index,row['crm_name'],max_seconds=cfg['max_query_seconds'])
                outputs.append(dict(query_id=row['query_id'],fold=row['fold'],trace=trace))
                print(json.dumps(outputs[-1]),flush=True)
                del snapshot;gc.collect()
            for access in index.metadata['access']:
                for ref in access['artifacts'].values():verify(ref)
            names.verify_after();load(output)
            # Queries are complete before output publication; no early metric.
            write_json(output/'cost_outputs.json',dict(rows=outputs,initial100_emitted=False,recall_measured=False,truth_opened=False))
        write_json(output/'manifest.json',dict(status=STATUS,launch=record(output/'launch.json'),outputs=record(output/'cost_outputs.json'),
            source_names=cfg['names'],source_joins=cfg['joins'],queries=len(outputs),load_seconds=load_seconds,resources=budget.report(),
            every_used_input_full_hash_verified_before_and_after=True,initial100_emitted=False,complete_retrieval_index=False,
            national_name_and_graph_corpus=True,score_stage_only_not_retrieval_latency=True,disk_cache_state='UNCONTROLLED_NOT_COLD_DISK',
            downstream_models_changed=False,ready_for_fold1=False,old_candidates_opened=False,**FLAGS))
        print(json.dumps(dict(status=STATUS,load_seconds=load_seconds,resources=budget.report())),flush=True)
    except BaseException as exc:
        write_json(output/'worker_stop.json',dict(status='STOP',reason=str(exc),resources=budget.report() if budget is not None and hasattr(budget,'elapsed') else None,
            initial100_emitted=False,old_candidates_opened=False,**FLAGS));raise
    finally:
        if names is not None:names.close()
        if index is not None:index.close()


def run(output):
    cfg=json.loads(CONFIG.read_text())
    if not output.resolve().is_relative_to(Path(cfg['output_root']).resolve()):raise ValueError('CATNAT-only query probe root required')
    if output.exists():raise FileExistsError('Immutable query probe root exists')
    inputs(cfg)
    gate=quiet(cfg['quiet_seconds'],cfg['min_system_free_percent'],cfg['max_quiet_wait_seconds'])
    output.mkdir(parents=True,exist_ok=False);write_json(output/'preflight.json',gate)
    write_json(output/'launch.json',dict(config=cfg,preflight=record(output/'preflight.json'),implementation=implementation(),old_candidates_opened=False,**FLAGS))
    try:
        with (output/'worker.log').open('x') as log:
            subprocess.run([sys.executable,'-u','-m','scripts.probe_retrieval_v150_rebound_names','worker','--output',str(output)],
                check=True,timeout=90,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
        print((output/'worker.log').read_text()[-8500:],flush=True)
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),old_candidates_opened=False,**FLAGS));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['run','worker']);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    (run if a.mode=='run' else worker)(a.output)
