"""Every history-site row, scoped name owner and raw succession endpoint.

Source-only coverage of current candidate identities; no labels or queries.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.xgb_matcher.retrieval_catalog_inventory_v135 import inventory
from src.xgb_matcher.retrieval_v20_workbench import Budget,ROOT,record,verify,write_json

CONFIG=ROOT/'config/retrieval_v135_catalog_inventory.json'
PROTECTED=('truth_opened','crm_opened','model_fit','recall_measured','fold0_opened','fold1_opened')


def sources(cfg):
    ms={key:json.loads(verify(cfg[key]).read_text()) for key in ('membership','owners','graph')}
    statuses={'membership':'SEALED_COMPLETE_NATIONAL_SITE_AND_PARENT_MEMBERSHIP',
        'owners':'SEALED_V132_COMPLETE_ALL_OFFICIAL_NAME_OWNER_CSR','graph':'SEALED_V133_COMPLETE_ALL_RAW_OFFICIAL_SUCCESSION'}
    for key,m in ms.items():
        if m['status']!=statuses[key] or any(m.get(k) is not False for k in PROTECTED):raise ValueError('Protected complete official source required')
        launch=json.loads(verify(m['launch']).read_text())
        for ref in launch['implementation'].values():verify(ref)
    if ms['membership']['stats']['sites']!=cfg['expected_sites'] or ms['owners']['facts']!=cfg['expected_facts'] or ms['graph']['stats']['endpoints']!=cfg['expected_graph_endpoints']:
        raise ValueError('Every official inventory input required')
    refs=[ms['membership']['artifacts']['endpoint_keys.npy'],ms['graph']['artifacts']['endpoint_keys.npy']]
    refs.extend(b['artifacts'][key+'.npy'] for b in ms['owners']['buckets'] for key in ('owner_key','subject_code'))
    for ref in refs:verify(ref)
    history=verify(cfg['history_sites'])
    if pq.ParquetFile(history).metadata.num_rows!=cfg['expected_history_rows']:raise ValueError('Every history-site source row required')
    return ms


def streams(cfg,ms):
    history=pq.ParquetFile(cfg['history_sites']['path']);total=0
    for batch in history.iter_batches(batch_size=131072,columns=['siret'],use_threads=False):
        values=batch.column(0)
        if values.null_count or not pc.all(pc.match_substring_regex(values,'^[0-9]{14}$')).as_py():raise ValueError('Every historical official ASCII14 identity required')
        total+=len(batch);yield 'history_sites',pc.cast(values,pa.uint64()).to_numpy(zero_copy_only=False)
    if total!=cfg['expected_history_rows']:raise ValueError('Whole history-site inventory required')
    keys=np.load(ms['graph']['artifacts']['endpoint_keys.npy']['path'],mmap_mode='r',allow_pickle=False)
    if keys.dtype!=np.uint64 or keys.shape!=(cfg['expected_graph_endpoints'],):raise ValueError('Whole raw graph endpoint array required')
    for begin in range(0,len(keys),131072):yield 'succession_endpoints',keys[begin:begin+131072]
    facts=0;site_facts=0
    for b in ms['owners']['buckets']:
        owners=np.load(b['artifacts']['owner_key.npy']['path'],mmap_mode='r',allow_pickle=False)
        subjects=np.load(b['artifacts']['subject_code.npy']['path'],mmap_mode='r',allow_pickle=False)
        if owners.dtype!=np.uint64 or subjects.dtype!=np.uint8 or owners.shape!=subjects.shape or owners.shape!=(b['facts'],):raise ValueError('Every scoped owner fact required')
        for begin in range(0,len(owners),131072):
            scope=subjects[begin:begin+131072]
            if np.any(~np.isin(scope,[1,2])):raise ValueError('Explicit site/parent proof scope required')
            values=owners[begin:begin+131072][scope==2];facts+=len(scope);site_facts+=len(values);yield 'official_name_sites',values
    if facts!=cfg['expected_facts'] or site_facts!=cfg['expected_site_facts']:raise ValueError('Every site-scoped original name owner required')


def worker(output):
    l=json.loads((output/'launch.json').read_text());budget=None
    if (output/'manifest.json').exists() or (output/'worker_stop.json').exists() or any(output.glob('*.npy')):raise ValueError('Immutable source inventory already started')
    for ref in l['implementation'].values():verify(ref)
    try:
        with Budget(l['config']['resources']) as budget:
            cfg=l['config'];ms=sources(cfg);keys=np.load(ms['membership']['artifacts']['endpoint_keys.npy']['path'],mmap_mode='r',allow_pickle=False)
            if keys.shape!=(cfg['expected_sites'],):raise ValueError('Whole national membership required')
            stats=inventory(keys,streams(cfg,ms),output);sources(cfg)
            artifacts={p.name:record(p) for p in output.glob('*.npy')};size=sum(r['size_bytes'] for r in artifacts.values())
            if size>cfg['max_output_bytes']:raise ValueError('STOP_SOURCE_INVENTORY_SIZE')
        for ref in l['implementation'].values():verify(ref)
        write_json(output/'manifest.json',dict(status='SEALED_V135_COMPLETE_OFFICIAL_SITE_REFERENCE_INVENTORY',launch=record(output/'launch.json'),
            sources={key:cfg[key] for key in ('membership','owners','graph','history_sites')},stats=stats,artifacts=artifacts,
            resources=budget.report(),output_bytes=size,complete_retrieval_index=False,ready_for_fold1=False,**{k:False for k in PROTECTED}))
        print(json.dumps(dict(stats=stats,resources=budget.report(),output_bytes=size)),flush=True)
    except BaseException as exc:
        write_json(output/'worker_stop.json',dict(status='STOP',reason=str(exc),**{k:False for k in PROTECTED},
            resources=budget.report() if budget is not None and hasattr(budget,'elapsed') else None));raise


def init(output):
    output.mkdir(parents=True,exist_ok=False);cfg=json.loads(CONFIG.read_text())
    paths=[Path(__file__),CONFIG,ROOT/'docs/retrieval_v135_catalog_inventory.md',ROOT/'src/xgb_matcher/retrieval_catalog_inventory_v135.py',
        ROOT/'src/xgb_matcher/retrieval_v20_workbench.py',ROOT/'src/xgb_matcher/retrieval_parent_discovery_v18.py']
    write_json(output/'launch.json',dict(config=cfg,implementation={str(p):record(p) for p in paths},**{k:False for k in PROTECTED}))


def run(output):
    init(output)
    try:
        with (output/'worker.log').open('x') as log:
            subprocess.run([sys.executable,'-u','-m','scripts.probe_retrieval_v135_catalog_inventory','worker','--output',str(output)],
                check=True,timeout=90,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
        print((output/'worker.log').read_text()[-6500:],flush=True)
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),**{k:False for k in PROTECTED}));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['run','worker']);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    (run if a.mode=='run' else worker)(a.output)
