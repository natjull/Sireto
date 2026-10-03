"""Audit every official name fact and append exact previously missing UL joins."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from src.xgb_matcher.retrieval_name_rebind_v149 import rebind,patched_positions
from src.xgb_matcher.retrieval_append_catalog_v141 import AppendKeys
from src.xgb_matcher.retrieval_proof_bridge_v137 import PROTECTED
from src.xgb_matcher.retrieval_system_quiet_v143 import quiet
from src.xgb_matcher.retrieval_v20_workbench import Budget,ROOT,verify,record,write_json

CONFIG=ROOT/'config/retrieval_v149_name_rebind.json'
STATUS='SEALED_V149_ALL_OFFICIAL_NAME_PARENT_APPEND_JOINS'


def authority(cfg):
    if np.__version__!=cfg['numpy_version'] or cfg['resources']['max_swapout_pages_delta']!=0:
        raise ValueError('Pinned typed implementation and unchanged zero swap required')
    bridge=json.loads(verify(cfg['bridge']).read_text());cat=json.loads(verify(cfg['catalog']).read_text())
    source=json.loads(verify(bridge['sources']['owners']).read_text())
    for m,status in ((bridge,'SEALED_V137_COMPLETE_OFFICIAL_PROOF_POSITION_BRIDGE'),
        (cat,'SEALED_V141_APPEND_ALL_V140_OFFICIAL_SITE_IDENTITIES'),
        (source,'SEALED_V132_COMPLETE_ALL_OFFICIAL_NAME_OWNER_CSR')):
        if (m['status']!=status or any(m.get(k) is not False for k in PROTECTED)
            or m['complete_retrieval_index'] or m['ready_for_fold1']):
            raise ValueError('Protected complete immutable official source required')
        if m is not source and (not m['resources']['pass_resource_policy'] or m['resources']['swapout_pages_delta']!=0):
            raise ValueError('Original source resource policy must pass')
        l=json.loads(verify(m['launch']).read_text())
        for ref in l['implementation'].values():verify(ref)
    if (cat['previous_catalog']!=bridge['sources']['catalog']
        or cat['stats']['previous_sites']!=bridge['stats']['sites']
        or cat['stats']['previous_parents']!=bridge['stats']['parents']
        or cat['stats']['sites']!=cfg['expected_sites'] or cat['stats']['parents']!=cfg['expected_parents']
        or not cat['stats']['every_previous_site_and_parent_position_unchanged']
        or not cat['stats']['every_new_site_admitted_once']
        or source['names']!=cfg['expected_names'] or source['facts']!=cfg['expected_facts']
        or bridge['stats']['names']!=source['names'] or bridge['stats']['facts']!=source['facts']
        or bridge['stats']['site_facts']!=cfg['expected_site_facts']
        or bridge['stats']['unbound_parent_facts']!=cfg['expected_previous_unbound_facts']
        or [x['bucket'] for x in bridge['buckets']]!=source['buckets']
        or len(bridge['buckets'])!=cfg['expected_buckets']):
        raise ValueError('Every original fact and unchanged append source namespace required')
    if not source['every_owner_fact_reconstructed_exactly'] or len(source['phase_authorities'])!=cfg['expected_buckets']:
        raise ValueError('Every original owner phase required')
    verify(source['source'])
    ns=fs=0
    for i,x in enumerate(bridge['buckets']):
        b=x['bucket'];s=x['stats']
        phase=json.loads(verify(source['phase_authorities'][i]).read_text());ps=phase['stats']
        if (phase['status']!='SEALED_V132_EXACT_ALL_OWNER_BUCKET' or phase['launch']!=source['launch']
            or any(phase.get(k) is not False for k in PROTECTED) or phase['complete_retrieval_index']
            or not phase['resources']['pass_resource_policy'] or phase['resources']['swapout_pages_delta']!=0
            or phase['bucket']['bucket']!=i or phase['bucket']['name_start']!=ns or phase['bucket']['fact_start']!=fs
            or ps['names']!=b['names'] or ps['facts']!=b['facts'] or phase['artifacts']!=b['artifacts']
            or not ps['every_original_fact_reconstructed_exactly'] or not ps['every_raw_name_has_original_owner']
            or not ps['owners_outside_candidate_catalog_preserved'] or ps['identifiers_as_features']
            or ps['site_cartesian_product_materialized'] or ps['owner_or_name_or_period_cap'] is not None):
            raise ValueError('Every complete original owner bucket resource/scope authority required')
        if (b['bucket']!=i or b['name_start']!=ns or b['fact_start']!=fs or s['facts']!=b['facts']
            or not s['every_original_fact_reconstructed_exactly'] or not s['every_site_fact_admitted']
            or not s['unbound_parent_observations_preserved'] or s['name_or_owner_or_member_cap'] is not None
            or s['identifiers_as_features'] or s['site_cartesian_product_materialized']
            or set(b['artifacts'])!={k+'.npy' for k in ('parent_key','owner_key','subject_code','scope_code','current_exact','name_fact_offsets')}
            or set(x['artifacts'])!={'fact_parent_positions.npy','fact_site_positions.npy'}):
            raise ValueError('Complete ordered original proof buckets required')
        ns+=b['names'];fs+=b['facts']
    if (ns!=cfg['expected_names'] or fs!=cfg['expected_facts']
        or sum(x['stats']['unbound_parent_facts'] for x in bridge['buckets'])!=cfg['expected_previous_unbound_facts']):
        raise ValueError('Whole original alias/fact/unbound denominators required')
    return bridge,cat,source


def implementation():
    names=('retrieval_name_rebind_v149','retrieval_append_catalog_v141','retrieval_proof_bridge_v137',
        'retrieval_extended_catalog_v136','retrieval_initial_collector_v146','retrieval_coupled_blocks_v121',
        'retrieval_mapped_blocks_v125','retrieval_site_attributes_v139','retrieval_activity_joint_v114',
        'retrieval_variants_v118','retrieval_system_quiet_v143','retrieval_v20_workbench','retrieval_parent_discovery_v18')
    paths=[Path(__file__),CONFIG,ROOT/'docs/retrieval_v149_name_rebind.md',
        *[ROOT/('src/xgb_matcher/'+n+'.py') for n in names]]
    return {str(p):record(p) for p in paths}


def load(output):
    l=json.loads((output/'launch.json').read_text());cfg=l['config']
    if (not output.resolve().is_relative_to(Path(cfg['output_root']).resolve())
        or any(l.get(k) is not False for k in PROTECTED)):
        raise ValueError('Protected CATNAT-only source audit required')
    for ref in l['implementation'].values():verify(ref)
    gate=json.loads(verify(l['preflight']).read_text())
    if (verify(l['preflight']).resolve()!=(output/'preflight.json').resolve()
        or not gate['pass_system_quiet_policy'] or gate['worker_swap_budget_changed']):
        raise ValueError('Actual unchanged system gate required')
    return l,gate,authority(cfg)


def close(arrays):
    for a in arrays.values():a._mmap.close()


def worker(output):
    if any(p.name not in ('launch.json','preflight.json','worker.log') for p in output.iterdir()):
        raise ValueError('Immutable source audit already started, no retry')
    cfg=json.loads((output/'launch.json').read_text())['config'];budget=None
    try:
        with Budget(cfg['resources']) as budget:
            launch,gate,(bridge,cat,source)=load(output)
            if budget.before!=gate['final_snapshot']['swapouts']:
                raise RuntimeError('STOP_SYSTEM_CHANGED_SINCE_QUIET_GATE')
            keys=AppendKeys(cfg['catalog']);items=[];total_bytes=0
            for x in bridge['buckets']:
                b=x['bucket'];src={};old={}
                folder=output/f'bucket{b["bucket"]:03d}';folder.mkdir(exist_ok=False)
                try:
                    src={k.removesuffix('.npy'):np.load(verify(ref),mmap_mode='r',allow_pickle=False) for k,ref in b['artifacts'].items()}
                    old={k.removesuffix('.npy'):np.load(verify(ref),mmap_mode='r',allow_pickle=False) for k,ref in x['artifacts'].items()}
                    offsets=src['name_fact_offsets'];flags=src['current_exact']
                    if (offsets.dtype!=np.uint64 or offsets.shape!=(b['names']+1,) or offsets[0]!=0
                        or offsets[-1]!=b['facts'] or np.any(offsets[1:]<=offsets[:-1])
                        or flags.dtype!=np.bool_ or flags.shape!=(b['facts'],)):
                        raise ValueError('Whole original name CSR and exact-current flags required')
                    rows,pos,stats=rebind(keys,src,old)
                    if (stats['facts']!=b['facts'] or stats['site_facts']!=x['stats']['site_facts']
                        or stats['previous_unbound_parent_facts']!=x['stats']['unbound_parent_facts']):
                        raise ValueError('Whole original bucket counts required')
                    np.save(folder/'fact_rows.npy',rows,allow_pickle=False)
                    np.save(folder/'parent_positions.npy',pos,allow_pickle=False)
                    rr=np.load(folder/'fact_rows.npy',allow_pickle=False);rp=np.load(folder/'parent_positions.npy',allow_pickle=False)
                    if (rr.dtype!=np.uint32 or rp.dtype!=np.uint32 or not np.array_equal(rr,rows)
                        or not np.array_equal(rp,pos)
                        or not np.array_equal(keys.reconstruct(rp,'parent'),src['parent_key'][rr])):
                        raise ValueError('Every exported sparse patch exact required')
                    # Exercise the actual query-side overlay, not only its hash.
                    patched=patched_positions(old['fact_parent_positions'],rr,rp,keys.previous.parents,keys.parents)
                    changed=np.flatnonzero(patched!=old['fact_parent_positions']).astype(np.uint32)
                    if not np.array_equal(changed,rr) or not np.array_equal(patched[rr],rp):
                        raise ValueError('Exactly the exported missing facts must change')
                    del patched,rr,rp,rows,pos
                    for ref in (*b['artifacts'].values(),*x['artifacts'].values()):verify(ref)
                    artifacts={p.name:record(p) for p in folder.glob('*.npy')}
                    total_bytes+=sum(r['size_bytes'] for r in artifacts.values())
                    if total_bytes>cfg['max_output_bytes']:raise RuntimeError('STOP_NAME_REBIND_OUTPUT_SIZE')
                    items.append(dict(bucket=b['bucket'],source_bucket=b,previous_artifacts=x['artifacts'],stats=stats,artifacts=artifacts))
                    if b['bucket']%8==7:print(json.dumps(dict(buckets_complete=len(items),recovered_facts=sum(i['stats']['recovered_parent_facts'] for i in items))),flush=True)
                finally:close(src);close(old)
            stats={k:sum(i['stats'][k] for i in items) for k in ('facts','site_facts','previous_unbound_parent_facts','recovered_parent_facts','remaining_unbound_parent_facts')}
            if (stats['facts']!=cfg['expected_facts'] or stats['site_facts']!=cfg['expected_site_facts']
                or stats['previous_unbound_parent_facts']!=cfg['expected_previous_unbound_facts']):
                raise ValueError('All original official facts audited required')
            recovered=np.unique(np.array([p for i in items for p in i['stats']['recovered_parent_positions']],dtype=np.uint32))
            members=np.load(verify(cat['artifacts']['new_site_parent_positions.npy']),mmap_mode='r',allow_pickle=False)
            if members.dtype!=np.uint32 or members.shape!=(cat['stats']['additional_sites'],):
                raise ValueError('Whole appended site-parent membership required')
            stats.update(recovered_parents=len(recovered),appended_sites_with_recovered_parent_proofs=int(np.sum(np.isin(members,recovered))),
                every_original_fact_audited=True,every_previous_bound_position_unchanged=True,every_patch_reconstructed_exactly=True,
                all_original_observations_scopes_and_exact_flags_preserved=True,remaining_unbound_facts_preserved=True,
                site_alias_lifted_to_parent=False,name_or_owner_or_member_cap=None,identifiers_as_features=False)
            members._mmap.close()
            for ref in keys.references:verify(ref)
            for i in items:
                for ref in i['artifacts'].values():verify(ref)
            load(output)
        # Success is never published before the resource policy passes.
        write_json(output/'manifest.json',dict(status=STATUS,launch=record(output/'launch.json'),
            sources=dict(bridge=cfg['bridge'],catalog=cfg['catalog'],owners=bridge['sources']['owners']),
            buckets=items,stats=stats,sites=keys.sites,parents=keys.parents,output_bytes=total_bytes,resources=budget.report(),
            complete_retrieval_index=False,ready_for_fold1=False,downstream_models_changed=False,**{k:False for k in PROTECTED}))
        print(json.dumps(dict(status=STATUS,stats=stats,resources=budget.report(),output_bytes=total_bytes)),flush=True)
    except BaseException as exc:
        write_json(output/'worker_stop.json',dict(status='STOP',reason=str(exc),resources=budget.report() if budget is not None and hasattr(budget,'elapsed') else None,
            **{k:False for k in PROTECTED}));raise


def run(output):
    cfg=json.loads(CONFIG.read_text())
    if not output.resolve().is_relative_to(Path(cfg['output_root']).resolve()):raise ValueError('CATNAT-only source audit root required')
    if output.exists():raise FileExistsError('Immutable source audit root exists')
    # Small source manifests only; no full array scan before the quiet gate.
    authority(cfg)
    gate=quiet(cfg['quiet_seconds'],cfg['min_system_free_percent'],cfg['max_quiet_wait_seconds'])
    output.mkdir(parents=True,exist_ok=False);write_json(output/'preflight.json',gate)
    write_json(output/'launch.json',dict(config=cfg,preflight=record(output/'preflight.json'),implementation=implementation(),**{k:False for k in PROTECTED}))
    try:
        with (output/'worker.log').open('x') as log:
            subprocess.run([sys.executable,'-u','-m','scripts.compile_retrieval_v149_name_rebind','worker','--output',str(output)],
                check=True,timeout=90,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
        print((output/'worker.log').read_text()[-6500:],flush=True)
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),**{k:False for k in PROTECTED}));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['run','worker']);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    (run if a.mode=='run' else worker)(a.output)
