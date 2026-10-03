"""Original explicit national nomenclature plus hash-pinned INSEE descriptions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import urlparse

import requests

from src.xgb_matcher.retrieval_activity_catalog_v112 import build,coverage,dbf_labels
from src.xgb_matcher.retrieval_v20_workbench import Budget,ROOT,record,verify,write_json

CONFIG=ROOT/'config/retrieval_v112_activity_catalog.json'


def inputs(cfg):
    m=json.loads(verify(cfg['context']).read_text());launch=json.loads(verify(m['launch']).read_text())
    if (m['status']!='SEALED_COMPLETE_NATIONAL_OFFICIAL_CONTEXT' or m['truth_opened'] or m['crm_opened'] or
        m['stats']['sites']!=cfg['expected_sites'] or m['stats']['closed_sites_removed']):
        raise ValueError('Complete national official context required')
    for ref in launch['implementation'].values():verify(ref)
    verify(cfg['stock'])
    keys=m['inputs']['registry_arrays']['endpoint_keys'];parents=m['inputs']['registry_arrays']['endpoint_parents']
    activity=m['artifacts']['activity_codes.npy']
    for ref in (keys,parents,activity):verify(ref)
    # Original stock authority recorded in the dossier, not an unrelated stock.
    dossier=json.loads(verify(launch['config']['dossier_manifest']).read_text())
    if dossier['inputs']['sirene_establishments']['sha256']!=cfg['stock']['sha256']:
        raise ValueError('Original stock differs from dossier authority')
    return dict(stock=cfg['stock'],keys=keys,parents=parents,activity=activity,activity_book=m['stats']['codebooks']['main_activity'])


def tables(cfg,output):
    labels=[];refs={}
    for nom,source in cfg['tables'].items():
        parsed=urlparse(source['url'])
        if parsed.scheme!='https' or parsed.netloc!='www.insee.fr':raise ValueError('Official INSEE source required')
        response=requests.get(source['url'],timeout=10,stream=True);response.raise_for_status()
        if urlparse(response.url).netloc!='www.insee.fr':raise ValueError('Unexpected official redirect')
        parts=[];size=0
        for part in response.iter_content(65536):
            size+=len(part)
            if size>1048576:raise ValueError('Official table download size drift')
            parts.append(part)
        payload=b''.join(parts)
        if hashlib.sha256(payload).hexdigest()!=source['sha256']:raise ValueError('Official table hash drift')
        rows=dbf_labels(payload,nom)
        path=output/(nom+'.dbf')
        with path.open('xb') as file:file.write(payload)
        labels.extend(dict(r,source_url=source['url']) for r in rows);refs[nom]=record(path)
    return labels,refs


def worker(output):
    launch=json.loads((output/'launch.json').read_text());cfg=launch['config'];budget=None
    for ref in launch['implementation'].values():verify(ref)
    try:
        with Budget(cfg['resources']) as budget:
            refs=inputs(cfg)
            if refs!=launch['inputs']:raise ValueError('Source launch drift')
            labels,table_refs=tables(cfg,output)
            stats=build(refs['stock']['path'],refs['keys']['path'],refs['parents']['path'],refs['activity']['path'],
                        refs['activity_book'],output,lambda r:print(json.dumps(r),flush=True))
            if stats['sites']!=cfg['expected_sites'] or inputs(cfg)!=refs:raise ValueError('Original full source drift')
            cov=coverage(stats,labels)
            write_json(output/'official_labels.json',dict(rows=labels,sources=cfg['tables'],
                encoding='CP850_AND_CP437_IDENTICAL_ON_ALL_USED_OEM_TEXT',crm_aliases=False,nomenclature_inferred=False))
            size=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
            if size>cfg['max_output_bytes']:raise ValueError('STOP_OUTPUT_SIZE')
        for ref in launch['implementation'].values():verify(ref)
        for ref in table_refs.values():verify(ref)
        write_json(output/'manifest.json',dict(status='SEALED_COMPLETE_EXPLICIT_NATIONAL_NOMENCLATURE',
            launch=record(output/'launch.json'),stats=stats,coverage=cov,table_artifacts=table_refs,
            artifacts={p.name:record(p) for p in output.glob('*.npy')},official_labels=record(output/'official_labels.json'),
            inputs=refs,resources=budget.report(),output_bytes=size,truth_opened=False,crm_opened=False,
            fold0_opened=False,fold1_opened=False,model_fit=False,recall_measured=False,
            downstream_models_changed=False,identifiers_as_features=False,ready_for_fold1=False))
        print(json.dumps(dict(complete=True,coverage={k:v for k,v in cov.items() if k!='unresolved'},
                             resources=budget.report(),output_bytes=size)),flush=True)
    except BaseException as exc:
        write_json(output/'worker_stop.json',dict(status='STOP',reason=str(exc),truth_opened=False,
            resources=budget.report() if budget is not None and hasattr(budget,'elapsed') else None));raise


def run(output):
    output.mkdir(parents=True,exist_ok=False);began=time.monotonic();cfg=json.loads(CONFIG.read_text())
    paths=(Path(__file__),CONFIG,ROOT/'docs/retrieval_v112_activity_catalog.md',
        ROOT/'src/xgb_matcher/retrieval_activity_catalog_v112.py',ROOT/'src/xgb_matcher/retrieval_official_context_v110.py',
        ROOT/'src/xgb_matcher/retrieval_v20_workbench.py',ROOT/'src/xgb_matcher/retrieval_parent_discovery_v18.py',
        ROOT/'src/xgb_matcher/siren_dossier.py')
    write_json(output/'launch.json',dict(config=cfg,inputs=inputs(cfg),implementation={str(p):record(p) for p in paths},truth_opened=False))
    try:
        with (output/'worker.log').open('x') as log:
            subprocess.run([sys.executable,'-u','-m','scripts.compile_retrieval_v112_activity_catalog','worker','--output',str(output)],
                check=True,timeout=90,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','ARROW_NUM_THREADS':'2'})
        print((output/'worker.log').read_text()[-4000:],flush=True)
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),seconds=time.monotonic()-began,truth_opened=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['run','worker']);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();(run if a.mode=='run' else worker)(a.output)
