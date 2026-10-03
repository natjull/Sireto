"""Admit all official V140 additions with every previous position unchanged."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from src.xgb_matcher.retrieval_append_catalog_v141 import append_build
from src.xgb_matcher.retrieval_proof_bridge_v137 import CatalogKeys,PROTECTED
from src.xgb_matcher.retrieval_v20_workbench import Budget,ROOT,record,verify,write_json

CONFIG=ROOT/'config/retrieval_v141_append_catalog.json'


def load(output):
    l=json.loads((output/'launch.json').read_text())
    for ref in l['implementation'].values():verify(ref)
    if np.__version__!=l['config']['numpy_version']:raise ValueError('Pinned unsigned array library drift')
    return l


def sources(cfg):
    catalog=CatalogKeys(cfg['previous_catalog']);m=json.loads(verify(cfg['inventory']).read_text())
    if (m['status']!='SEALED_V140_ALL_NAME_AND_ADDRESS_OFFICIAL_SITE_IDENTITY_INVENTORY'
        or m['catalog']!=cfg['previous_catalog'] or any(m.get(k) is not False for k in PROTECTED)
        or m['candidate_catalog_changed'] or m['complete_retrieval_index'] or m['ready_for_fold1']
        or m['unique_additional_absent_sites']!=cfg['expected_additional_sites']
        or catalog.sites!=cfg['expected_previous_sites'] or catalog.parents!=cfg['expected_previous_parents']
        or catalog.metadata['stats']['blocks']!=cfg['expected_previous_blocks']):
        raise ValueError('Protected complete official inventory and exact previous namespace required')
    launch=json.loads(verify(m['launch']).read_text())
    for ref in launch['implementation'].values():verify(ref)
    phases=[]
    for ref in m['phase_authorities']:
        phase=json.loads(verify(ref).read_text());s=phase['stats'];verify(phase['artifact'])
        if (phase['status']!='SEALED_V140_ALL_ORIGINAL_CANONICAL_IDENTITY_SCOPE'
            or phase['catalog']!=cfg['previous_catalog'] or phase['launch']!=m['launch']
            or any(phase.get(k) is not False for k in PROTECTED) or not phase['resources']['pass_resource_policy']
            or not s['every_original_source_row_accounted'] or not s['every_absent_site_genuinely_absent']
            or not s['every_present_site_reconstructed_exactly'] or s['candidate_catalog_changed']
            or s['lexical_mask_or_family_filter'] or s['human_denominator_filtered']):
            raise ValueError('Every original canonical observation accounted without filter required')
        phases.append(np.load(phase['artifact']['path'],mmap_mode='r',allow_pickle=False))
    if len(phases)!=2:raise ValueError('Both complete original canonical sources required')
    additional=m['all_absent_sites'];keys=np.load(verify(additional),mmap_mode='r',allow_pickle=False)
    if keys.shape!=(cfg['expected_additional_sites'],) or keys.dtype!=np.uint64 or not np.array_equal(keys,np.unique(np.concatenate(phases))):
        raise ValueError('Every original absent key union required')
    return catalog,additional,keys


def init(output):
    output.mkdir(parents=True,exist_ok=False)
    paths=[Path(__file__),CONFIG,ROOT/'docs/retrieval_v141_append_catalog.md',
        *[ROOT/('src/xgb_matcher/'+k+'.py') for k in ('retrieval_append_catalog_v141','retrieval_proof_bridge_v137',
            'retrieval_extended_catalog_v136','retrieval_canonical_inventory_v140','retrieval_variant_batches_v128',
            'retrieval_variants_v118','retrieval_v20_workbench','retrieval_parent_discovery_v18')],
        ROOT/'scripts/probe_retrieval_v140_canonical_inventory.py']
    write_json(output/'launch.json',dict(config=json.loads(CONFIG.read_text()),implementation={str(p):record(p) for p in paths},
        **{k:False for k in PROTECTED}))


def worker(output):
    budget=None
    if (output/'manifest.json').exists() or (output/'worker_stop.json').exists() or any(output.glob('*.npy')):
        raise ValueError('Immutable append catalog already started')
    try:
        l=load(output);cfg=l['config']
        with Budget(cfg['resources']) as budget:
            catalog,additional,keys=sources(cfg)
            stats=append_build(catalog,keys,cfg['expected_previous_blocks'],cfg['block_size'],output)
            sources(cfg);load(output)
            artifacts={p.name:record(p) for p in output.glob('*.npy')};size=sum(r['size_bytes'] for r in artifacts.values())
            if size>cfg['max_output_bytes']:raise ValueError('STOP_APPEND_CATALOG_OUTPUT_SIZE')
        write_json(output/'manifest.json',dict(status='SEALED_V141_APPEND_ALL_V140_OFFICIAL_SITE_IDENTITIES',
            launch=record(output/'launch.json'),previous_catalog=cfg['previous_catalog'],inventory=cfg['inventory'],
            additional_sites=additional,artifacts=artifacts,stats=stats,resources=budget.report(),output_bytes=size,
            complete_retrieval_index=False,ready_for_fold1=False,downstream_models_changed=False,
            additional_missing_attributes=['xy','state','headquarters','postcode','insee','number','activity','dates'],
            **{k:False for k in PROTECTED}))
        print(json.dumps(dict(stats=stats,resources=budget.report(),output_bytes=size)),flush=True)
    except BaseException as exc:
        write_json(output/'worker_stop.json',dict(status='STOP',reason=str(exc),**{k:False for k in PROTECTED},
            resources=budget.report() if budget is not None and hasattr(budget,'elapsed') else None));raise


def run(output):
    init(output)
    try:
        with (output/'worker.log').open('x') as log:
            subprocess.run([sys.executable,'-u','-m','scripts.compile_retrieval_v141_append_catalog','worker',
                '--output',str(output)],check=True,timeout=90,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
        print((output/'worker.log').read_text()[-6000:],flush=True)
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),**{k:False for k in PROTECTED}));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['run','worker'])
    p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    (run if a.mode=='run' else worker)(a.output)
