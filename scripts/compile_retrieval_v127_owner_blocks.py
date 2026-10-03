"""Complete official parent/block bridge, no name expansion or queries."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from src.xgb_matcher.retrieval_owner_blocks_v127 import build
from src.xgb_matcher.retrieval_v20_workbench import Budget,ROOT,record,verify,write_json

CONFIG=ROOT/'config/retrieval_v127_owner_blocks.json'


def sources(cfg):
    m=json.loads(verify(cfg['membership_manifest']).read_text());l=json.loads(verify(cfg['layout_manifest']).read_text())
    if (m['status']!='SEALED_COMPLETE_NATIONAL_SITE_AND_PARENT_MEMBERSHIP' or l['status']!='SEALED_ALL_NATIONAL_SITE_PHYSICAL_LAYOUT_ONLY'
        or any(x['truth_opened'] or x['crm_opened'] for x in (m,l)) or l['membership']!=cfg['membership_manifest']
        or m['stats']['sites']!=cfg['expected_sites'] or m['stats']['parents']!=cfg['expected_parents']
        or l['stats']['blocks']!=cfg['expected_blocks'] or not l['stats']['all_sites_exactly_once']):
        raise ValueError('Complete official all-site/membership/layout authority required')
    refs={k:m['artifacts'][k+'.npy'] for k in ('endpoint_parents','parent_keys')}
    refs.update({k:l['artifacts'][k+'.npy'] for k in ('block_sites','block_offsets')})
    for ref in refs.values():verify(ref)
    return refs


def worker(output):
    launch=json.loads((output/'launch.json').read_text());cfg=launch['config'];budget=None
    for ref in launch['implementation'].values():verify(ref)
    try:
        with Budget(cfg['resources']) as budget:
            refs=sources(cfg);arrays={k:np.load(verify(r),mmap_mode='r',allow_pickle=False) for k,r in refs.items()}
            stats=build(arrays['endpoint_parents'],arrays['parent_keys'],arrays['block_sites'],arrays['block_offsets'],output,
                lambda row:print(json.dumps(row),flush=True))
            sources(cfg)
            size=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
            if size>cfg['max_output_bytes']:raise ValueError('STOP_OUTPUT_SIZE')
        for ref in launch['implementation'].values():verify(ref)
        write_json(output/'manifest.json',dict(status='SEALED_COMPLETE_NATIONAL_PARENT_BLOCK_BRIDGE_ONLY',
            launch=record(output/'launch.json'),membership=cfg['membership_manifest'],layout=cfg['layout_manifest'],
            input_arrays=refs,stats=stats,artifacts={p.name:record(p) for p in output.glob('*.npy')},resources=budget.report(),
            output_bytes=size,truth_opened=False,crm_opened=False,model_fit=False,recall_measured=False,
            fold0_opened=False,fold1_opened=False,identifiers_as_features=False,downstream_models_changed=False,
            complete_candidate_membership=True,complete_retrieval_index=False,ready_for_fold1=False))
        print(json.dumps(dict(stats=stats,resources=budget.report(),output_bytes=size)),flush=True)
    except BaseException as exc:
        write_json(output/'worker_stop.json',dict(status='STOP',reason=str(exc),truth_opened=False,recall_measured=False,
            resources=budget.report() if budget is not None and hasattr(budget,'elapsed') else None));raise


def run(output):
    output.mkdir(parents=True,exist_ok=False);cfg=json.loads(CONFIG.read_text())
    paths=[Path(__file__),CONFIG,ROOT/'docs/retrieval_v127_owner_blocks.md',ROOT/'src/xgb_matcher/retrieval_owner_blocks_v127.py',
        ROOT/'src/xgb_matcher/retrieval_v20_workbench.py',ROOT/'src/xgb_matcher/retrieval_parent_discovery_v18.py']
    write_json(output/'launch.json',dict(config=cfg,implementation={str(p):record(p) for p in paths},truth_opened=False,crm_opened=False))
    try:
        with (output/'worker.log').open('x') as log:
            subprocess.run([sys.executable,'-u','-m','scripts.compile_retrieval_v127_owner_blocks','worker','--output',str(output)],
                check=True,timeout=90,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
        print((output/'worker.log').read_text()[-7000:],flush=True)
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),truth_opened=False,recall_measured=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['run','worker']);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();(run if a.mode=='run' else worker)(a.output)
