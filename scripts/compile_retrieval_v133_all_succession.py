"""Every raw SIRENE succession observation, without continuity/source caps."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import pyarrow.parquet as pq

from src.xgb_matcher.retrieval_all_succession_v133 import build
from src.xgb_matcher.retrieval_v20_workbench import Budget,ROOT,record,verify,write_json

CONFIG=ROOT/'config/retrieval_v133_all_succession.json'
PROTECTED=('truth_opened','crm_opened','model_fit','recall_measured','fold0_opened','fold1_opened')


def source(cfg):
    path=verify(cfg['source'])
    if pq.ParquetFile(path).metadata.num_rows!=cfg['expected_observations']:raise ValueError('Every raw official succession observation required')
    return path


def worker(output):
    l=json.loads((output/'launch.json').read_text());budget=None
    if (output/'manifest.json').exists() or (output/'worker_stop.json').exists() or any(output.glob('*.npy')):
        raise ValueError('Immutable whole succession phase already started')
    for ref in l['implementation'].values():verify(ref)
    try:
        with Budget(l['config']['resources']) as budget:
            path=source(l['config']);stats=build(path,output,l['config']['expected_observations'],
                progress=lambda v:print(json.dumps(v),flush=True));source(l['config'])
            artifacts={p.name:record(p) for p in output.glob('*.npy')};size=sum(r['size_bytes'] for r in artifacts.values())
            if size>l['config']['max_output_bytes']:raise ValueError('STOP_ALL_SUCCESSION_OUTPUT_SIZE')
        for ref in l['implementation'].values():verify(ref)
        write_json(output/'manifest.json',dict(status='SEALED_V133_COMPLETE_ALL_RAW_OFFICIAL_SUCCESSION',launch=record(output/'launch.json'),
            source=l['config']['source'],stats=stats,artifacts=artifacts,resources=budget.report(),output_bytes=size,
            complete_retrieval_index=False,ready_for_fold1=False,downstream_models_changed=False,**{k:False for k in PROTECTED}))
        print(json.dumps(dict(stats=stats,resources=budget.report(),output_bytes=size)),flush=True)
    except BaseException as exc:
        write_json(output/'worker_stop.json',dict(status='STOP',reason=str(exc),**{k:False for k in PROTECTED},
            resources=budget.report() if budget is not None and hasattr(budget,'elapsed') else None));raise


def init(output):
    output.mkdir(parents=True,exist_ok=False);cfg=json.loads(CONFIG.read_text())
    paths=[Path(__file__),CONFIG,ROOT/'docs/retrieval_v133_all_succession.md',ROOT/'src/xgb_matcher/retrieval_all_succession_v133.py',
        ROOT/'src/xgb_matcher/retrieval_v20_workbench.py',ROOT/'src/xgb_matcher/retrieval_parent_discovery_v18.py']
    write_json(output/'launch.json',dict(config=cfg,implementation={str(p):record(p) for p in paths},**{k:False for k in PROTECTED}))


def run(output):
    init(output)
    try:
        with (output/'worker.log').open('x') as log:
            subprocess.run([sys.executable,'-u','-m','scripts.compile_retrieval_v133_all_succession','worker','--output',str(output)],
                check=True,timeout=90,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
        print((output/'worker.log').read_text()[-7000:],flush=True)
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),**{k:False for k in PROTECTED}));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['run','worker']);p.add_argument('--output',required=True,type=Path);v=p.parse_args()
    (run if v.mode=='run' else worker)(v.output)
