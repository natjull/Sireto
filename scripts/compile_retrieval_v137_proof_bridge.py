"""All official name facts and raw relations joined to all V136 candidates."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from src.xgb_matcher.retrieval_proof_bridge_v137 import CatalogKeys,facts_bridge,graph_bridge,PROTECTED
from src.xgb_matcher.retrieval_v20_workbench import Budget,ROOT,record,verify,write_json

CONFIG=ROOT/'config/retrieval_v137_proof_bridge.json'
STATUSES=dict(catalog='SEALED_V136_CURRENT_PLUS_ALL_V135_OFFICIAL_SITE_NAMESPACE',
    owners='SEALED_V132_COMPLETE_ALL_OFFICIAL_NAME_OWNER_CSR',graph='SEALED_V133_COMPLETE_ALL_RAW_OFFICIAL_SUCCESSION')


def sources(cfg):
    if (not isinstance(cfg['owner_group_size'],int) or isinstance(cfg['owner_group_size'],bool)
        or cfg['owner_group_size']<=0):raise ValueError('Positive physical owner group size required')
    result={k:json.loads(verify(cfg[k]).read_text()) for k in STATUSES}
    for k,m in result.items():
        if (m['status']!=STATUSES[k] or any(m.get(f) is not False for f in PROTECTED)
            or m['complete_retrieval_index'] or m['ready_for_fold1']):
            raise ValueError('Protected whole official source required')
        launch=json.loads(verify(m['launch']).read_text())
        for ref in launch['implementation'].values():verify(ref)
    cat,owners,graph=(result[k] for k in STATUSES)
    if (cat['stats']['sites']!=cfg['expected_sites'] or cat['stats']['parents']!=cfg['expected_parents']
        or owners['names']!=cfg['expected_names'] or owners['facts']!=cfg['expected_facts']
        or graph['stats']['endpoints']!=cfg['expected_graph_endpoints']
        or graph['stats']['observations']!=cfg['expected_graph_observations']):
        raise ValueError('Every expected national source row required')
    ns=fs=0
    for i,b in enumerate(owners['buckets']):
        if b['bucket']!=i or b['name_start']!=ns or b['fact_start']!=fs:
            raise ValueError('Every original owner bucket order required')
        ns+=b['names'];fs+=b['facts']
    if ns!=owners['names'] or fs!=owners['facts'] or len(owners['buckets'])!=cfg['expected_owner_buckets']:
        raise ValueError('Complete original owner namespace required')
    return result


def implementation():
    paths=[Path(__file__),CONFIG,ROOT/'docs/retrieval_v137_proof_bridge.md',
        ROOT/'src/xgb_matcher/retrieval_proof_bridge_v137.py',ROOT/'src/xgb_matcher/retrieval_extended_catalog_v136.py',
        ROOT/'src/xgb_matcher/retrieval_v20_workbench.py',ROOT/'src/xgb_matcher/retrieval_parent_discovery_v18.py']
    return {str(p):record(p) for p in paths}


def init(output):
    output.mkdir(parents=True,exist_ok=False);cfg=json.loads(CONFIG.read_text())
    ms=sources(cfg)
    with Budget(cfg['resources']) as budget:
        CatalogKeys(cfg['catalog'])
        for b in ms['owners']['buckets']:
            for ref in b['artifacts'].values():verify(ref)
        for ref in ms['graph']['artifacts'].values():verify(ref)
    write_json(output/'launch.json',dict(config=cfg,implementation=implementation(),source_full_hash_audit=budget.report(),
        **{k:False for k in PROTECTED}))
    print(json.dumps(dict(phase='init',source_audit=budget.report())),flush=True)


def load(output):
    l=json.loads((output/'launch.json').read_text())
    for ref in l['implementation'].values():verify(ref)
    if np.__version__!=l['config']['numpy_version']:raise ValueError('Pinned unsigned library drift')
    return l,sources(l['config'])


def array(ref):return np.load(verify(ref),mmap_mode='r',allow_pickle=False)


def owner_phase(output,index):return output/'owners'/f'group{index:03d}'


def validate_phase(cfg,kind,index):
    if kind=='owners':
        groups=(cfg['expected_owner_buckets']+cfg['owner_group_size']-1)//cfg['owner_group_size']
        if not isinstance(index,int) or isinstance(index,bool) or not 0<=index<groups:
            raise ValueError('Actual complete owner phase required')
    elif kind!='graph' or index is not None:raise ValueError('Explicit official graph phase required')


def worker(output,kind,index):
    l,ms=load(output);cfg=l['config'];budget=None
    validate_phase(cfg,kind,index)
    if kind=='owners':
        dest=owner_phase(output,index)
    elif kind=='graph':dest=output/'graph'
    else:raise ValueError('Explicit official bridge phase required')
    if any((dest/k).exists() for k in ('manifest.json','worker_stop.json')) or any(dest.glob('**/*.npy')):
        raise ValueError('Immutable official proof phase already started')
    try:
        with Budget(cfg['resources']) as budget:
            catalog=CatalogKeys(cfg['catalog']);items=[]
            if kind=='owners':
                start=index*cfg['owner_group_size'];buckets=ms['owners']['buckets'][start:start+cfg['owner_group_size']]
                for b in buckets:
                    folder=dest/f'bucket{b["bucket"]:03d}';folder.mkdir(exist_ok=False)
                    # Retain/check ALL original owner fields, including the
                    # alias CSR and exact-current flag used by query-side scatter.
                    src={k.removesuffix('.npy'):array(ref) for k,ref in b['artifacts'].items()}
                    if any(a.shape!=((b['names']+1,) if k=='name_fact_offsets' else (b['facts'],)) for k,a in src.items()):
                        raise ValueError('Every original owner source array required')
                    stats=facts_bridge(catalog,src,folder)
                    for ref in b['artifacts'].values():verify(ref)
                    items.append(dict(bucket=b,stats=stats,artifacts={p.name:record(p) for p in folder.glob('*.npy')}))
                    del src
                stats=dict(facts=sum(x['stats']['facts'] for x in items),
                    site_facts=sum(x['stats']['site_facts'] for x in items),
                    unbound_parent_facts=sum(x['stats']['unbound_parent_facts'] for x in items))
                artifacts=None;size=sum(r['size_bytes'] for x in items for r in x['artifacts'].values())
            else:
                src={k.removesuffix('.npy'):array(ref) for k,ref in ms['graph']['artifacts'].items()}
                stats=graph_bridge(catalog,src,dest)
                for ref in ms['graph']['artifacts'].values():verify(ref)
                artifacts={p.name:record(p) for p in dest.glob('*.npy')};size=sum(r['size_bytes'] for r in artifacts.values())
            for ref in catalog.references:verify(ref)
            load(output)
            if size>cfg['max_phase_output_bytes']:raise ValueError('STOP_PROOF_BRIDGE_PHASE_SIZE')
        write_json(dest/'manifest.json',dict(status='SEALED_V137_EXACT_COMPLETE_PROOF_PHASE',kind=kind,index=index,
            launch=record(output/'launch.json'),stats=stats,buckets=items,artifacts=artifacts,output_bytes=size,
            resources=budget.report(),complete_retrieval_index=False,**{k:False for k in PROTECTED}))
        print(json.dumps(dict(phase=kind,index=index,stats=stats,resources=budget.report(),output_bytes=size)),flush=True)
    except BaseException as exc:
        write_json(dest/'worker_stop.json',dict(status='STOP',reason=str(exc),**{k:False for k in PROTECTED},
            resources=budget.report() if budget is not None and hasattr(budget,'elapsed') else None));raise


def execute(output,kind,index=None):
    l,_=load(output);validate_phase(l['config'],kind,index)
    dest=output/'graph' if kind=='graph' else owner_phase(output,index)
    dest.mkdir(parents=True,exist_ok=False)
    args=[sys.executable,'-u','-m','scripts.compile_retrieval_v137_proof_bridge','worker','--kind',kind,'--output',str(output)]
    if index is not None:args+=['--index',str(index)]
    try:
        with (dest/'worker.log').open('x') as log:
            subprocess.run(args,check=True,timeout=90,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
        print((dest/'worker.log').read_text()[-3500:],flush=True)
    except BaseException as exc:
        write_json(dest/'status.json',dict(status='STOP',reason=str(exc),**{k:False for k in PROTECTED}));raise


def seal(output):
    l,ms=load(output);cfg=l['config']
    if (output/'manifest.json').exists():raise ValueError('Immutable proof bridge already sealed')
    groups=(cfg['expected_owner_buckets']+cfg['owner_group_size']-1)//cfg['owner_group_size']
    phases=[owner_phase(output,i)/'manifest.json' for i in range(groups)]+[output/'graph/manifest.json']
    buckets=[];refs=[];size=0;graph=None;unbound=site_facts=0
    with Budget(cfg['resources']) as budget:
        for ordinal,path in enumerate(phases):
            if not path.exists():raise ValueError('UNFINISHED_ALL_PROOF_PHASES')
            m=json.loads(path.read_text());kind='owners' if ordinal<groups else 'graph'
            if (m['status']!='SEALED_V137_EXACT_COMPLETE_PROOF_PHASE' or m['launch']!=record(output/'launch.json')
                or m['kind']!=kind or m['index']!=(ordinal if kind=='owners' else None)
                or any(m.get(k) is not False for k in PROTECTED) or not m['resources']['pass_resource_policy']
                or m['complete_retrieval_index']):raise ValueError('Protected complete official proof phase required')
            if kind=='owners':
                expected=ms['owners']['buckets'][ordinal*cfg['owner_group_size']:(ordinal+1)*cfg['owner_group_size']]
                if [x['bucket'] for x in m['buckets']]!=expected:raise ValueError('Every original owner phase/bucket required')
                for x in m['buckets']:
                    b=x['bucket'];s=x['stats']
                    if (s['facts']!=b['facts'] or not s['every_original_fact_reconstructed_exactly'] or not s['every_site_fact_admitted']
                        or not s['unbound_parent_observations_preserved'] or s['site_cartesian_product_materialized']
                        or s['name_or_owner_or_member_cap'] is not None or s['identifiers_as_features']
                        or set(x['artifacts'])!={'fact_parent_positions.npy','fact_site_positions.npy'}):
                        raise ValueError('Every uncapped exact official fact required')
                    for name,ref in x['artifacts'].items():
                        a=array(ref)
                        if a.dtype!=np.uint32 or a.shape!=(b['facts'],) or Path(ref['path']).resolve()!=(path.parent/f'bucket{b["bucket"]:03d}'/name).resolve():
                            raise ValueError('Whole exact exported fact position array required')
                    buckets.append(x);unbound+=s['unbound_parent_facts'];site_facts+=s['site_facts']
            else:
                s=m['stats'];graph=m['artifacts']
                specs={'site_parent_positions':cfg['expected_sites'],'graph_site_positions':cfg['expected_graph_endpoints'],
                    'graph_parent_positions':cfg['expected_graph_endpoints'],
                    'predecessor_graph_positions':cfg['expected_graph_observations'],'successor_graph_positions':cfg['expected_graph_observations']}
                if (s['sites']!=cfg['expected_sites'] or s['endpoints']!=cfg['expected_graph_endpoints']
                    or s['observations']!=cfg['expected_graph_observations'] or not s['every_site_parent_and_graph_endpoint_admitted']
                    or not s['every_original_relation_reconstructed_exactly'] or not s['all_observations_and_provenance_unchanged']
                    or s['relation_or_endpoint_or_parent_cap'] is not None or s['identifiers_as_features'] or s['attributes_inferred_from_relation']
                    or set(graph)!={k+'.npy' for k in specs}):raise ValueError('Every complete official relation required')
                for key,n in specs.items():
                    ref=graph[key+'.npy'];a=array(ref)
                    if a.dtype!=np.uint32 or a.shape!=(n,) or Path(ref['path']).resolve()!=(path.parent/(key+'.npy')).resolve():
                        raise ValueError('Whole exact graph position array required')
            size+=m['output_bytes'];refs.append(record(path))
        if (sum(b['stats']['facts'] for b in buckets)!=cfg['expected_facts'] or site_facts!=cfg['expected_site_facts']
            or size>cfg['max_total_output_bytes']):raise ValueError('Complete national proof namespace/size required')
    write_json(output/'manifest.json',dict(status='SEALED_V137_COMPLETE_OFFICIAL_PROOF_POSITION_BRIDGE',
        launch=record(output/'launch.json'),sources={k:cfg[k] for k in STATUSES},buckets=buckets,graph=graph,phase_authorities=refs,
        stats=dict(names=cfg['expected_names'],facts=cfg['expected_facts'],site_facts=site_facts,unbound_parent_facts=unbound,
            sites=cfg['expected_sites'],parents=cfg['expected_parents'],graph_endpoints=cfg['expected_graph_endpoints'],
            graph_observations=cfg['expected_graph_observations']),resources=budget.report(),output_bytes=size,
        complete_retrieval_index=False,ready_for_fold1=False,downstream_models_changed=False,**{k:False for k in PROTECTED}))
    print(json.dumps(dict(phase='seal',output_bytes=size,unbound_parent_facts=unbound,resources=budget.report())),flush=True)


def run(output):
    began=time.monotonic();init(output);l,_=load(output);cfg=l['config']
    groups=(cfg['expected_owner_buckets']+cfg['owner_group_size']-1)//cfg['owner_group_size']
    for kind,index in [('owners',i) for i in range(groups)]+[('graph',None)]:
        if time.monotonic()-began>cfg['controller_seconds']:raise TimeoutError('STOP_PROOF_CONTROLLER_COST')
        execute(output,kind,index)
    seal(output)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['init','run','worker','phase','seal'])
    p.add_argument('--output',required=True,type=Path);p.add_argument('--kind',choices=['owners','graph']);p.add_argument('--index',type=int);a=p.parse_args()
    if a.mode in ('worker','phase'):
        if a.kind is None:p.error('Explicit planned proof phase required')
        (worker if a.mode=='worker' else execute)(a.output,a.kind,a.index)
    else:{'init':init,'run':run,'seal':seal}[a.mode](a.output)
