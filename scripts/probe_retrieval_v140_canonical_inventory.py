"""Complete original names/address identity coverage, including nonlexical rows."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from src.xgb_matcher.retrieval_canonical_inventory_v140 import inventory
from src.xgb_matcher.retrieval_proof_bridge_v137 import CatalogKeys,PROTECTED
from src.xgb_matcher.retrieval_v20_workbench import Budget,ROOT,record,verify,write_json

CONFIG=ROOT/'config/retrieval_v140_canonical_inventory.json'


def load(output):
    l=json.loads((output/'launch.json').read_text())
    for ref in l['implementation'].values():verify(ref)
    if np.__version__!=l['config']['numpy_version']:raise ValueError('Pinned unsigned library drift')
    return l


def init(output):
    output.mkdir(parents=True,exist_ok=False)
    paths=[Path(__file__),CONFIG,ROOT/'docs/retrieval_v140_canonical_inventory.md',
        *[ROOT/('src/xgb_matcher/'+k+'.py') for k in ('retrieval_canonical_inventory_v140','retrieval_proof_bridge_v137',
            'retrieval_extended_catalog_v136','retrieval_variant_batches_v128','retrieval_variants_v118',
            'retrieval_v20_workbench','retrieval_parent_discovery_v18')]]
    write_json(output/'launch.json',dict(config=json.loads(CONFIG.read_text()),implementation={str(p):record(p) for p in paths},**{k:False for k in PROTECTED}))


def worker(output,kind):
    l=load(output);cfg=l['config'];dest=output/kind;budget=None
    if (kind not in ('names','addresses') or any((dest/k).exists() for k in ('manifest.json','worker_stop.json'))
        or any(dest.glob('*.npy'))):raise ValueError('Immutable actual canonical inventory phase required')
    try:
        with Budget(cfg['resources']) as budget:
            catalog=CatalogKeys(cfg['catalog'])
            if catalog.sites!=cfg['expected_namespace_sites']:raise ValueError('Whole unchanged official namespace required')
            source=verify(cfg[kind]);stats=inventory(source,catalog,dest,cfg['expected_'+kind+'_rows'])
            verify(cfg[kind])
            for ref in catalog.references:verify(ref)
            load(output);artifact=record(dest/'absent_site_keys.npy')
            if artifact['size_bytes']>cfg['max_output_bytes']:raise ValueError('STOP_CANONICAL_INVENTORY_SIZE')
        write_json(dest/'manifest.json',dict(status='SEALED_V140_ALL_ORIGINAL_CANONICAL_IDENTITY_SCOPE',kind=kind,
            launch=record(output/'launch.json'),source=cfg[kind],catalog=cfg['catalog'],stats=stats,artifact=artifact,
            resources=budget.report(),**{k:False for k in PROTECTED}))
        print(json.dumps(dict(kind=kind,stats=stats,resources=budget.report())),flush=True)
    except BaseException as exc:
        write_json(dest/'worker_stop.json',dict(status='STOP',reason=str(exc),**{k:False for k in PROTECTED},
            resources=budget.report() if budget is not None and hasattr(budget,'elapsed') else None));raise


def execute(output,kind):
    if kind not in ('names','addresses'):raise ValueError('Actual canonical source kind required')
    dest=output/kind;dest.mkdir(exist_ok=False)
    try:
        with (dest/'worker.log').open('x') as log:
            subprocess.run([sys.executable,'-u','-m','scripts.probe_retrieval_v140_canonical_inventory','worker',
                '--kind',kind,'--output',str(output)],check=True,timeout=90,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
        print((dest/'worker.log').read_text()[-3500:],flush=True)
    except BaseException as exc:
        write_json(dest/'status.json',dict(status='STOP',reason=str(exc),**{k:False for k in PROTECTED}));raise


def seal(output):
    if (output/'manifest.json').exists():raise ValueError('Immutable canonical inventory already sealed')
    l=load(output);cfg=l['config'];arrays=[];phases=[]
    for kind in ('names','addresses'):
        path=output/kind/'manifest.json'
        if not path.exists():raise ValueError('UNFINISHED_ALL_CANONICAL_SOURCES')
        m=json.loads(path.read_text());s=m['stats']
        if (m['status']!='SEALED_V140_ALL_ORIGINAL_CANONICAL_IDENTITY_SCOPE' or m['kind']!=kind or m['source']!=cfg[kind]
            or m['catalog']!=cfg['catalog'] or m['launch']!=record(output/'launch.json')
            or any(m.get(k) is not False for k in PROTECTED) or not m['resources']['pass_resource_policy']
            or s['rows']!=cfg['expected_'+kind+'_rows'] or not s['every_original_source_row_accounted']
            or not s['every_present_site_reconstructed_exactly'] or not s['every_absent_site_genuinely_absent']
            or s['lexical_mask_or_family_filter'] or s['candidate_catalog_changed'] or s['human_denominator_filtered']):
            raise ValueError('Protected complete original canonical source inventory required')
        a=np.load(verify(m['artifact']),mmap_mode='r',allow_pickle=False)
        if (a.dtype!=np.uint64 or a.shape!=(s['unique_absent_sites'],) or np.any(a[1:]<=a[:-1])
            or Path(m['artifact']['path']).resolve()!=(output/kind/'absent_site_keys.npy').resolve()):
            raise ValueError('All unique official absent identity keys required')
        arrays.append(a);phases.append(record(path))
    union=np.unique(np.concatenate(arrays));catalog=CatalogKeys(cfg['catalog'])
    if np.any(catalog.site_positions(union)!=np.iinfo(np.uint32).max):raise ValueError('Every union absent identity required')
    np.save(output/'all_absent_site_keys.npy',union,allow_pickle=False);ref=record(output/'all_absent_site_keys.npy')
    size=ref['size_bytes']+sum(a.nbytes+128 for a in arrays)
    if size>cfg['max_output_bytes']:raise ValueError('STOP_CANONICAL_UNION_OUTPUT_SIZE')
    write_json(output/'manifest.json',dict(status='SEALED_V140_ALL_NAME_AND_ADDRESS_OFFICIAL_SITE_IDENTITY_INVENTORY',
        launch=record(output/'launch.json'),phase_authorities=phases,catalog=cfg['catalog'],all_absent_sites=ref,
        unique_additional_absent_sites=len(union),output_bytes=size,candidate_catalog_changed=False,
        complete_retrieval_index=False,ready_for_fold1=False,**{k:False for k in PROTECTED}))
    print(json.dumps(dict(phase='seal',unique_additional_absent_sites=len(union),output_bytes=size)),flush=True)


def run(output):
    init(output)
    for kind in ('names','addresses'):execute(output,kind)
    seal(output)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['run','worker','init','phase','seal'])
    p.add_argument('--output',required=True,type=Path);p.add_argument('--kind',choices=['names','addresses']);a=p.parse_args()
    if a.mode in ('worker','phase'):
        if a.kind is None:p.error('Actual original source kind required')
        (worker if a.mode=='worker' else execute)(a.output,a.kind)
    else:{'run':run,'init':init,'seal':seal}[a.mode](a.output)
