"""Current positions preserved, every official V135 missing identity admitted."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from src.xgb_matcher.retrieval_extended_catalog_v136 import build
from src.xgb_matcher.retrieval_v20_workbench import Budget,ROOT,record,verify,write_json

CONFIG=ROOT/'config/retrieval_v136_extended_catalog.json'
PROTECTED=('truth_opened','crm_opened','model_fit','recall_measured','fold0_opened','fold1_opened')


def sources(cfg):
    ms={key:json.loads(verify(cfg[key]).read_text()) for key in ('membership','layout','bridge','inventory')}
    statuses=dict(membership='SEALED_COMPLETE_NATIONAL_SITE_AND_PARENT_MEMBERSHIP',
        layout='SEALED_ALL_NATIONAL_SITE_PHYSICAL_LAYOUT_ONLY',bridge='SEALED_COMPLETE_NATIONAL_PARENT_BLOCK_BRIDGE_ONLY',
        inventory='SEALED_V135_COMPLETE_OFFICIAL_SITE_REFERENCE_INVENTORY')
    for key,m in ms.items():
        if (m['status']!=statuses[key] or any(m.get(k) is not False for k in PROTECTED)
            or not m['resources']['pass_resource_policy']):raise ValueError('Protected complete immutable source required')
        launch=json.loads(verify(m['launch']).read_text())
        for ref in launch['implementation'].values():verify(ref)
    if (ms['layout']['membership']!=cfg['membership'] or ms['bridge']['membership']!=cfg['membership']
        or ms['bridge']['layout']!=cfg['layout'] or ms['inventory']['sources']['membership']!=cfg['membership']
        or ms['membership']['stats']['sites']!=cfg['expected_current_sites'] or ms['membership']['stats']['parents']!=cfg['expected_current_parents']
        or ms['bridge']['stats']['blocks']!=cfg['expected_current_blocks']
        or ms['inventory']['stats']['union_unique_absent_siret']!=cfg['expected_extra_sites']
        or ms['inventory']['stats']['candidate_catalog_changed'] or ms['inventory']['stats']['human_denominator_filtered']):
        raise ValueError('Exact unchanged current corpus and every official extra required')
    refs={key:ms['membership']['artifacts'][key+'.npy'] for key in ('endpoint_keys','endpoint_parents','endpoint_states',
        'endpoint_source_observations','parent_keys','parent_offsets','parent_endpoints')}
    refs.update({key:ms['bridge']['artifacts'][key+'.npy'] for key in ('site_blocks','parent_blocks','parent_block_offsets')})
    refs.update({key:ms['bridge']['input_arrays'][key] for key in ('block_sites','block_offsets')})
    extra=ms['inventory']['artifacts']['all_absent_siret.npy']
    for ref in (*refs.values(),extra):verify(ref)
    return refs,extra


def worker(output):
    l=json.loads((output/'launch.json').read_text());budget=None
    if (output/'manifest.json').exists() or (output/'worker_stop.json').exists() or any(output.glob('*.npy')):
        raise ValueError('Immutable whole extended catalog already started')
    for ref in l['implementation'].values():verify(ref)
    if np.__version__!=l['config']['numpy_version']:raise ValueError('Pinned unsigned array library drift')
    try:
        with Budget(l['config']['resources']) as budget:
            cfg=l['config'];refs,extra=sources(cfg)
            sites=np.load(refs['endpoint_keys']['path'],mmap_mode='r',allow_pickle=False)
            parents=np.load(refs['parent_keys']['path'],mmap_mode='r',allow_pickle=False)
            additions=np.load(extra['path'],mmap_mode='r',allow_pickle=False)
            if sites.shape!=(cfg['expected_current_sites'],) or parents.shape!=(cfg['expected_current_parents'],) or additions.shape!=(cfg['expected_extra_sites'],):
                raise ValueError('Every source identity/current position required')
            stats=build(sites,parents,additions,cfg['expected_current_blocks'],cfg['block_size'],output)
            sources(cfg);arrays={p.name:record(p) for p in output.glob('*.npy')};size=sum(r['size_bytes'] for r in arrays.values())
            if size>cfg['max_output_bytes']:raise ValueError('STOP_EXTENDED_CATALOG_OUTPUT_SIZE')
        for ref in l['implementation'].values():verify(ref)
        write_json(output/'manifest.json',dict(status='SEALED_V136_CURRENT_PLUS_ALL_V135_OFFICIAL_SITE_NAMESPACE',
            launch=record(output/'launch.json'),sources={k:cfg[k] for k in ('membership','layout','bridge','inventory')},
            input_arrays=refs,extra_sites=extra,artifacts=arrays,stats=stats,resources=budget.report(),output_bytes=size,
            extra_missing_attributes=['state','headquarters','xy','postcode','insee','street_number','activity'],
            complete_candidate_namespace=True,complete_retrieval_index=False,ready_for_fold1=False,
            downstream_models_changed=False,**{k:False for k in PROTECTED}))
        print(json.dumps(dict(stats=stats,resources=budget.report(),output_bytes=size)),flush=True)
    except BaseException as exc:
        write_json(output/'worker_stop.json',dict(status='STOP',reason=str(exc),**{k:False for k in PROTECTED},
            resources=budget.report() if budget is not None and hasattr(budget,'elapsed') else None));raise


def init(output):
    output.mkdir(parents=True,exist_ok=False);cfg=json.loads(CONFIG.read_text())
    paths=[Path(__file__),CONFIG,ROOT/'docs/retrieval_v136_extended_catalog.md',ROOT/'src/xgb_matcher/retrieval_extended_catalog_v136.py',
        ROOT/'src/xgb_matcher/retrieval_v20_workbench.py',ROOT/'src/xgb_matcher/retrieval_parent_discovery_v18.py']
    write_json(output/'launch.json',dict(config=cfg,implementation={str(p):record(p) for p in paths},**{k:False for k in PROTECTED}))


def run(output):
    init(output)
    try:
        with (output/'worker.log').open('x') as log:
            subprocess.run([sys.executable,'-u','-m','scripts.compile_retrieval_v136_extended_catalog','worker','--output',str(output)],
                check=True,timeout=90,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
        print((output/'worker.log').read_text()[-6500:],flush=True)
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),**{k:False for k in PROTECTED}));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['run','worker']);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    (run if a.mode=='run' else worker)(a.output)
