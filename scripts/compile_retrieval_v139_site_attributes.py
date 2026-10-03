"""All current CP/INSEE/number evidence; reuse official HQ/state/dates/activity/XY."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from src.xgb_matcher.retrieval_site_attributes_v139 import build,audit,ATTRS,PROTECTED
from src.xgb_matcher.retrieval_v20_workbench import Budget,ROOT,record,verify,write_json

CONFIG=ROOT/'config/retrieval_v139_site_attributes.json'
STATUSES=dict(catalog='SEALED_V136_CURRENT_PLUS_ALL_V135_OFFICIAL_SITE_NAMESPACE',
    context='SEALED_COMPLETE_NATIONAL_OFFICIAL_CONTEXT',activity='SEALED_COMPLETE_EXPLICIT_NATIONAL_NOMENCLATURE',
    layout='SEALED_ALL_NATIONAL_SITE_PHYSICAL_LAYOUT_ONLY')


def inputs(cfg):
    ms={k:json.loads(verify(cfg[k]).read_text()) for k in STATUSES}
    for k,m in ms.items():
        if (m['status']!=STATUSES[k] or any(m.get(f) is not False for f in PROTECTED)
            or m['ready_for_fold1'] or not m['resources']['pass_resource_policy']):
            raise ValueError('Protected complete official attribute source required')
        launch=json.loads(verify(m['launch']).read_text())
        for ref in launch['implementation'].values():verify(ref)
    cat,ctx,act,lay=(ms[k] for k in STATUSES)
    if (cat['stats']['current_sites']!=cfg['expected_current_sites'] or cat['stats']['sites']!=cfg['expected_namespace_sites']
        or ctx['stats']['sites']!=cfg['expected_current_sites'] or not ctx['stats']['full_registry_exact_alignment']
        or act['stats']['sites']!=cfg['expected_current_sites'] or not act['stats']['all_sites_preserved']
        or act['stats']['nomenclature_inferred'] or not lay['stats']['all_sites_exactly_once']
        or lay['stats']['sites']!=cfg['expected_current_sites']
        or lay['source']['sha256']!=cfg['stock']['sha256'] or act['inputs']['stock']['sha256']!=cfg['stock']['sha256']
        or ctx['inputs']['registry_arrays']['endpoint_keys']!=cat['input_arrays']['endpoint_keys']
        or ctx['inputs']['registry_arrays']['endpoint_parents']!=cat['input_arrays']['endpoint_parents']
        or act['inputs']['keys']!=cat['input_arrays']['endpoint_keys'] or act['inputs']['parents']!=cat['input_arrays']['endpoint_parents']):
        raise ValueError('Identical whole current source/opaque position alignment required')
    reused={k:ctx['artifacts'][k+'.npy'] for k in ('headquarters_codes','state_codes','creation_days','current_period_start_days')}
    reused.update({k:act['artifacts'][k+'.npy'] for k in ('nomenclature_codes','original_activity_codes')})
    reused['site_xy']=lay['artifacts']['site_xy.npy']
    refs=dict(keys=cat['input_arrays']['endpoint_keys'],parents=cat['input_arrays']['endpoint_parents'],reused=reused)
    for ref in (refs['keys'],refs['parents'],*reused.values(),cfg['stock'],act['official_labels']):verify(ref)
    return ms,refs


def load(output):
    l=json.loads((output/'launch.json').read_text())
    for ref in l['implementation'].values():verify(ref)
    if np.__version__!=l['config']['numpy_version']:raise ValueError('Pinned unsigned library drift')
    return l


def built(output):
    p=output/'build/manifest.json'
    if not p.exists():raise ValueError('UNFINISHED_CURRENT_ATTRIBUTE_BUILD')
    m=json.loads(p.read_text())
    if (m['status']!='SEALED_V139_ALL_CURRENT_ATTRIBUTE_BUILD_ONLY' or m['launch']!=record(output/'launch.json')
        or any(m.get(k) is not False for k in PROTECTED) or not m['resources']['pass_resource_policy']
        or not m['stats']['every_current_site_once']):raise ValueError('Protected all-current attribute build required')
    for ref in (*m['artifacts'].values(),m['number_dictionary']):verify(ref)
    return m


def worker(output,phase):
    l=load(output);cfg=l['config'];dest=output/phase;budget=None
    if (phase not in ('build','audit') or any((dest/k).exists() for k in ('manifest.json','worker_stop.json'))
        or any(dest.glob('*.npy'))):raise ValueError('Immutable actual structured attribute phase required')
    try:
        with Budget(cfg['resources']) as budget:
            ms,refs=inputs(cfg);keys=np.load(refs['keys']['path'],mmap_mode='r',allow_pickle=False)
            parents=np.load(refs['parents']['path'],mmap_mode='r',allow_pickle=False)
            if phase=='build':
                stats=build(cfg['stock']['path'],keys,parents,dest,progress=lambda row:print(json.dumps(row),flush=True))
                arrays={p.name:record(p) for p in dest.glob('*.npy')};dictionary=record(dest/'number_dictionary.json')
                size=sum(r['size_bytes'] for r in arrays.values())+dictionary['size_bytes']
                if size>cfg['max_output_bytes']:raise ValueError('STOP_STRUCTURED_ATTRIBUTE_SIZE')
            else:
                m=built(output);arrays={k.removesuffix('.npy'):np.load(ref['path'],mmap_mode='r',allow_pickle=False) for k,ref in m['artifacts'].items()}
                book=json.loads(verify(m['number_dictionary']).read_text())['values']
                context={k:np.load(ref['path'],mmap_mode='r',allow_pickle=False) for k,ref in refs['reused'].items()
                    if k in ('headquarters_codes','state_codes','creation_days','current_period_start_days')}
                context['state_book']=ms['context']['stats']['codebooks']['administrative_state']
                stats=audit(cfg['stock']['path'],keys,parents,arrays,book,context)
                if stats['known']!=m['stats']['known'] or stats['sites']!=m['stats']['sites']:raise ValueError('Exact all-source build/audit counts required')
                built(output)
            inputs(cfg);load(output)
        common=dict(launch=record(output/'launch.json'),stats=stats,resources=budget.report(),**{k:False for k in PROTECTED})
        if phase=='build':
            data=dict(status='SEALED_V139_ALL_CURRENT_ATTRIBUTE_BUILD_ONLY',artifacts=arrays,number_dictionary=dictionary,output_bytes=size,**common)
        else:data=dict(status='SEALED_V139_ALL_EXPORTED_AND_REUSED_RAW_ATTRIBUTE_AUDIT',build=record(output/'build/manifest.json'),**common)
        write_json(dest/'manifest.json',data);print(json.dumps(dict(phase=phase,stats=stats,resources=budget.report())),flush=True)
    except BaseException as exc:
        write_json(dest/'worker_stop.json',dict(status='STOP',reason=str(exc),**{k:False for k in PROTECTED},
            resources=budget.report() if budget is not None and hasattr(budget,'elapsed') else None));raise


def execute(output,phase):
    if phase not in ('build','audit'):raise ValueError('Actual structured attribute phase required')
    dest=output/phase;dest.mkdir(exist_ok=False)
    try:
        with (dest/'worker.log').open('x') as log:
            subprocess.run([sys.executable,'-u','-m','scripts.compile_retrieval_v139_site_attributes','worker',
                '--phase',phase,'--output',str(output)],check=True,timeout=90,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
        print((dest/'worker.log').read_text()[-5000:],flush=True)
    except BaseException as exc:
        write_json(dest/'status.json',dict(status='STOP',reason=str(exc),**{k:False for k in PROTECTED}));raise


def init(output):
    output.mkdir(parents=True,exist_ok=False)
    paths=[Path(__file__),CONFIG,ROOT/'docs/retrieval_v139_site_attributes.md',
        *[ROOT/('src/xgb_matcher/'+k+'.py') for k in ('retrieval_site_attributes_v139','retrieval_variant_batches_v128',
            'retrieval_coupled_source_v121','retrieval_coupled_blocks_v121','retrieval_variants_v118','retrieval_official_context_v110',
            'retrieval_activity_semantic_v113','retrieval_activity_decoder_v111','retrieval_proof_bridge_v137',
            'retrieval_extended_catalog_v136','retrieval_v20_workbench','retrieval_parent_discovery_v18')]]
    write_json(output/'launch.json',dict(config=json.loads(CONFIG.read_text()),implementation={str(p):record(p) for p in paths},**{k:False for k in PROTECTED}))


def seal(output):
    if (output/'manifest.json').exists():raise ValueError('Immutable structured evidence already sealed')
    l=load(output);cfg=l['config'];ms,refs=inputs(cfg);b=built(output);p=output/'audit/manifest.json'
    if not p.exists():raise ValueError('UNFINISHED_ALL_RAW_ATTRIBUTE_AUDIT')
    a=json.loads(p.read_text())
    if (a['status']!='SEALED_V139_ALL_EXPORTED_AND_REUSED_RAW_ATTRIBUTE_AUDIT' or a['launch']!=record(output/'launch.json')
        or a['build']!=record(output/'build/manifest.json') or any(a.get(k) is not False for k in PROTECTED)
        or not a['resources']['pass_resource_policy'] or not a['stats']['every_exported_attribute_reconstructed_exactly']
        or not a['stats']['every_reused_headquarters_state_and_date_matches_raw_stock']
        or a['stats']['sites']!=cfg['expected_current_sites'] or b['stats']['sites']!=cfg['expected_current_sites']
        or set(b['artifacts'])!={k+'.npy' for k in ATTRS}):raise ValueError('Protected complete independent all-source audit required')
    for name,ref in b['artifacts'].items():
        values=np.load(verify(ref),mmap_mode='r',allow_pickle=False)
        if values.dtype!=np.uint32 or values.shape!=(cfg['expected_current_sites'],) or Path(ref['path']).resolve()!=(output/'build'/name).resolve():
            raise ValueError('All exact structured source arrays required')
    write_json(output/'manifest.json',dict(status='SEALED_V139_ALL_CURRENT_STRUCTURED_SITE_ATTRIBUTES',launch=record(output/'launch.json'),
        build=record(output/'build/manifest.json'),audit_authority=record(p),sources={k:cfg[k] for k in STATUSES},stock=cfg['stock'],
        artifacts=b['artifacts'],number_dictionary=b['number_dictionary'],reused_arrays=refs['reused'],activity_catalog=cfg['activity'],
        stats={**b['stats'],'independent_all_source_audit_pending':False},audit=a['stats'],namespace_sites=cfg['expected_namespace_sites'],
        extra_missing_attributes=['postcode','insee','number','headquarters','state','dates','activity','xy'],
        resources={'build':b['resources'],'audit':a['resources']},output_bytes=b['output_bytes'],complete_retrieval_index=False,
        downstream_models_changed=False,ready_for_fold1=False,**{k:False for k in PROTECTED}))


def run(output):
    init(output)
    for phase in ('build','audit'):execute(output,phase)
    seal(output)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['run','worker','init','phase','seal'])
    p.add_argument('--output',required=True,type=Path);p.add_argument('--phase',choices=['build','audit']);a=p.parse_args()
    if a.mode in ('worker','phase'):
        if a.phase is None:p.error('Actual structured attribute phase required')
        (worker if a.mode=='worker' else execute)(a.output,a.phase)
    else:{'run':run,'init':init,'seal':seal}[a.mode](a.output)
