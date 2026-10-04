"""One frozen query, complete national name/graph namespace, exact V153 guard."""
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
from scripts.compile_retrieval_v149_name_rebind import load as source_load,implementation as source_implementation
from src.xgb_matcher.retrieval_name_pages_v151 import ScopedNamePages
from src.xgb_matcher.retrieval_paged_names_v171 import PagedReboundNames
from src.xgb_matcher.retrieval_memory_policy_v153 import Budget,quiet,LIMITS,POLICY,validate_report
from src.xgb_matcher.retrieval_supervisor_v164 import supervise
from src.xgb_matcher.retrieval_proof_bridge_v137 import PROTECTED
from src.xgb_matcher.retrieval_v20_workbench import ROOT,record,verify,write_json

CONFIG=ROOT/'config/retrieval_v171_paged_names.json'
STATUS='SEALED_V171_COMPLETE_OFFICIAL_PAGED_NAME_SNAPSHOT_COST_ONLY'
FLAGS={k:k=='crm_opened' for k in PROTECTED}


def inputs(cfg):
    if (cfg['resources']!=LIMITS or cfg['numpy_version']!=np.__version__ or cfg['duckdb_version']!=duckdb.__version__
        or type(cfg['chunk']) is not int or cfg['chunk']<=0 or type(cfg['max_query_disk_bytes']) is not int
        or cfg['max_query_disk_bytes']<=0):raise ValueError('Pinned full source and exact authorized memory policy required')
    verify(cfg['names']);verify(cfg['joins']);source_load(Path(cfg['joins']['path']).parent)
    return query_rows(json.loads(verify(cfg['query_only_launch']).read_text()),cfg['queries'])


def implementation():
    names=('retrieval_paged_names_v171','retrieval_compact_coverage_v156','retrieval_name_pages_v151',
        'retrieval_rebound_names_v150','retrieval_national_names_v138','retrieval_memory_policy_v153','retrieval_supervisor_v164')
    refs=source_implementation()
    refs.update({str(p):record(p) for p in [Path(__file__),CONFIG,ROOT/'docs/retrieval_v171_sequential_scratch_contract.md',
        ROOT/'scripts/probe_retrieval_v138_national_names.py',*[ROOT/('src/xgb_matcher/'+n+'.py') for n in names]]})
    return refs


def load(output):
    launch=json.loads((output/'launch.json').read_text());cfg=launch['config']
    if (cfg!=json.loads(CONFIG.read_text()) or not output.resolve().is_relative_to(Path(cfg['output_root']).resolve())
        or any(launch.get(k) is not v for k,v in FLAGS.items()) or launch['old_candidates_opened']):
        raise ValueError('Protected immutable source-bound query probe required')
    for ref in launch['implementation'].values():verify(ref)
    gate=json.loads(verify(launch['preflight']).read_text())
    if (verify(launch['preflight']).resolve()!=(output/'preflight.json').resolve()
        or gate['policy_id']!=POLICY or not gate['pass_system_quiet_policy'] or not gate['worker_swap_budget_changed']):
        raise ValueError('Actual exact authorized memory-pressure gate required')
    return launch,gate,inputs(cfg)


def gather_complete(snapshot,chunk):
    """Consume EVERY opaque site once, bounded copies, no candidate selection."""
    digest=hashlib.sha256();nonzero=exact_count=0
    for start in range(0,snapshot.sites,chunk):
        ids=np.arange(start,min(snapshot.sites,start+chunk),dtype=np.uint32)
        cov,exact=snapshot.gather(ids)
        if (cov.dtype!=np.float64 or exact.dtype!=np.float64 or cov.shape!=ids.shape or exact.shape!=ids.shape
            or not np.isfinite(cov).all() or not np.isfinite(exact).all() or np.any((cov<0)|(cov>1))
            or np.any((exact!=0)&(exact!=1)) or np.any(exact>cov)):
            raise ValueError('Every full source-bound coherent name proof required')
        digest.update(cov.tobytes());digest.update(exact.tobytes())
        nonzero+=int(np.count_nonzero(cov));exact_count+=int(np.count_nonzero(exact))
    return dict(sites=snapshot.sites,nonzero=nonzero,exact=exact_count,sha256=digest.hexdigest(),
        every_site_gathered_once=True,no_candidate_selection=True)


def worker(output):
    if any(p.name not in ('launch.json','preflight.json','worker.log') for p in output.iterdir()):
        raise ValueError('Immutable query probe already started, no retry')
    budget=index=names=snapshot=None;stage='INPUTS'
    try:
        cfg=json.loads((output/'launch.json').read_text())['config']
        with Budget(cfg['resources']) as budget:
            launch,gate,rows=load(output);began=time.monotonic();stage='LOAD_LEXICAL'
            index=ScopedNamePages(cfg['names']);stage='LOAD_OWNERS_AND_GRAPH'
            names=PagedReboundNames(cfg['joins']);load_seconds=time.monotonic()-began;outputs=[]
            load_rss=psutil.Process().memory_info().rss
            print(json.dumps(dict(stage='LOAD_COMPLETE',seconds=load_seconds,rss_bytes=load_rss)),flush=True)
            for i,row in enumerate(rows):
                stage='QUERY_'+str(i);folder=output/'queries'/f'query{i:04d}'
                def checkpoint(value):
                    print(json.dumps(dict(value,rss_bytes=psutil.Process().memory_info().rss)),flush=True)
                snapshot,trace=names.query(index,row['crm_name'],folder,cfg['max_query_seconds'],cfg['chunk'],progress=checkpoint)
                size=sum(p.stat().st_size for p in folder.iterdir() if p.is_file())
                if size>cfg['max_query_disk_bytes']:raise ValueError('STOP_OWN_QUERY_DISK_BUDGET')
                stage='FULL_SITE_GATHER_'+str(i);gather_began=time.monotonic()
                gathered=gather_complete(snapshot,cfg['chunk'])
                outputs.append(dict(query_id=row['query_id'],fold=row['fold'],trace=trace,gather=gathered,
                    gather_seconds=time.monotonic()-gather_began,query_disk_bytes=size))
                print(json.dumps(outputs[-1]),flush=True)
                snapshot.close(remove=True);snapshot=None
            stage='FULL_SOURCE_REHASH'
            for item in index.metadata['access']:
                for ref in item['artifacts'].values():verify(ref)
            verify(index.source);names.verify_after();load(output)
            # Closing only these own source maps before the final RSS check
            # does not erase the process high-water mark or other app pages.
            names.close();names=None;index.close();index=None
        write_json(output/'cost_outputs.json',dict(rows=outputs,initial100_emitted=False,recall_measured=False,truth_opened=False))
        write_json(output/'manifest.json',dict(status=STATUS,launch=record(output/'launch.json'),outputs=record(output/'cost_outputs.json'),
            source_names=cfg['names'],source_joins=cfg['joins'],queries=len(outputs),load_seconds=load_seconds,load_rss_bytes=load_rss,
            resources=budget.report(),every_used_input_full_hash_verified_before_and_after=True,initial100_emitted=False,
            complete_retrieval_index=False,national_name_and_graph_corpus=True,score_stage_only_not_retrieval_latency=True,
            disk_cache_state='UNCONTROLLED_NOT_COLD_DISK',downstream_models_changed=False,ready_for_fold1=False,
            old_candidates_opened=False,query_features_float64_unchanged=True,**FLAGS))
    except BaseException as exc:
        write_json(output/'worker_stop.json',dict(status='STOP',reason=str(exc),stage=stage,
            resources=budget.report() if budget is not None and hasattr(budget,'elapsed') else None,
            initial100_emitted=False,old_candidates_opened=False,**FLAGS));raise
    finally:
        if snapshot is not None:snapshot.close()  # Preserve failed-query files.
        if names is not None:names.close()
        if index is not None:index.close()


def run(output):
    cfg=json.loads(CONFIG.read_text())
    if not output.resolve().is_relative_to(Path(cfg['output_root']).resolve()):raise ValueError('CATNAT-only query probe root required')
    if output.exists():raise FileExistsError('Immutable query probe root exists')
    inputs(cfg);gate=quiet(cfg['quiet_seconds'],cfg['min_system_free_percent'],cfg['max_quiet_wait_seconds'])
    output.mkdir(parents=True,exist_ok=False);write_json(output/'preflight.json',gate)
    write_json(output/'launch.json',dict(config=cfg,preflight=record(output/'preflight.json'),implementation=implementation(),old_candidates_opened=False,**FLAGS))
    try:
        with (output/'worker.log').open('x') as log:
            s=supervise([sys.executable,'-u','-m','scripts.probe_retrieval_v171_paged_names','worker','--output',str(output)],
                log,{**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},output/'supervisor_stop.json')
        write_json(output/'supervision.json',s)
        from src.xgb_matcher.retrieval_address_authority_v166 import parent_exit
        parent_exit(output);m=json.loads((output/'manifest.json').read_text());validate_report(m['resources'])
        if m['status']!=STATUS or m['launch']!=record(output/'launch.json'):raise ValueError('Exact supervised whole name probe required')
        print(json.dumps(dict(status=STATUS,resources=m['resources'],load_seconds=m['load_seconds'])),flush=True)
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),old_candidates_opened=False,**FLAGS));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['run','worker']);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    (run if a.mode=='run' else worker)(a.output)
