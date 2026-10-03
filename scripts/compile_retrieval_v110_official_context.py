"""Compile complete raw official site context; no learned model or label read."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from src.xgb_matcher.retrieval_official_context_v110 import build
from src.xgb_matcher.retrieval_v20_workbench import Budget,ROOT,record,verify,write_json

CONFIG=ROOT/'config/retrieval_v110_official_context.json'


def inputs(cfg):
    registry=json.loads(verify(cfg['registry_manifest']).read_text())
    if (registry['status']!='SEALED_COMPLETE_NATIONAL_SITE_AND_PARENT_MEMBERSHIP' or
        registry['truth_opened'] or registry['crm_opened'] or registry['stats']['sites']!=cfg['expected_sites'] or
        registry['stats']['closed_sites_removed'] or registry['stats']['graph_membership_required']):
        raise ValueError('Complete official national registry required')
    launch=json.loads(verify(registry['launch']).read_text())
    for ref in launch['implementation'].values(): verify(ref)
    projection=json.loads(verify(registry['source_manifest']).read_text())
    old_launch=json.loads(verify(projection['launch']).read_text())
    if old_launch['config']['sites']!=cfg['sites'] or old_launch['config']['dossier_manifest']!=cfg['dossier_manifest']:
        raise ValueError('Raw source differs from registry authority')
    dossier=json.loads(verify(cfg['dossier_manifest']).read_text())
    path=verify(cfg['sites'])
    if (dossier['files']['establishments.parquet']['sha256']!=cfg['sites']['sha256'] or
        dossier['counts']['establishments']!=cfg['expected_sites']):
        raise ValueError('Complete raw official source required')
    refs={k:registry['artifacts'][k+'.npy'] for k in ('endpoint_keys','endpoint_parents','endpoint_states')}
    for ref in refs.values(): verify(ref)
    dirs={str(Path(ref['path']).parent) for ref in refs.values()}
    if len(dirs)!=1: raise ValueError('Registry array directory drift')
    return dict(source=record(path),registry_arrays=refs,registry_directory=dirs.pop())


def worker(output):
    launch=json.loads((output/'launch.json').read_text());cfg=launch['config'];budget=None
    for ref in launch['implementation'].values(): verify(ref)
    try:
        with Budget(cfg['resources']) as budget:
            refs=inputs(cfg)
            if refs!=launch['inputs']: raise ValueError('Launch source drift')
            stats=build(refs['source']['path'],refs['registry_directory'],output,
                        lambda row:print(json.dumps(row),flush=True))
            if stats['sites']!=cfg['expected_sites'] or inputs(cfg)!=refs:
                raise ValueError('Full context source/cardinality drift')
            size=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
            if size>cfg['max_output_bytes']: raise ValueError('STOP_OUTPUT_SIZE')
        for ref in launch['implementation'].values(): verify(ref)
        write_json(output/'manifest.json',dict(status='SEALED_COMPLETE_NATIONAL_OFFICIAL_CONTEXT',
            launch=record(output/'launch.json'),stats=stats,inputs=refs,resources=budget.report(),output_bytes=size,
            artifacts={p.name:record(p) for p in output.glob('*.npy')},truth_opened=False,crm_opened=False,
            fold0_opened=False,fold1_opened=False,model_fit=False,recall_measured=False,
            downstream_models_changed=False,identifiers_as_features=False,ready_for_fold1=False))
        print(json.dumps(dict(complete=True,sites=stats['sites'],resources=budget.report(),output_bytes=size)),flush=True)
    except BaseException as exc:
        write_json(output/'worker_stop.json',dict(status='STOP',reason=str(exc),truth_opened=False,
            resources=budget.report() if budget is not None and hasattr(budget,'elapsed') else None));raise


def run(output):
    output.mkdir(parents=True,exist_ok=False);began=time.monotonic();cfg=json.loads(CONFIG.read_text())
    paths=(Path(__file__),CONFIG,ROOT/'docs/retrieval_v110_official_context.md',
           ROOT/'src/xgb_matcher/retrieval_official_context_v110.py',
           ROOT/'src/xgb_matcher/retrieval_v20_workbench.py',ROOT/'src/xgb_matcher/retrieval_parent_discovery_v18.py')
    launch=dict(config=cfg,inputs=inputs(cfg),implementation={str(p):record(p) for p in paths},truth_opened=False)
    write_json(output/'launch.json',launch)
    try:
        with (output/'worker.log').open('x') as log:
            subprocess.run([sys.executable,'-u','-m','scripts.compile_retrieval_v110_official_context','worker','--output',str(output)],
                check=True,timeout=90,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','ARROW_NUM_THREADS':'2'})
        print((output/'worker.log').read_text()[-4000:],flush=True)
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),truth_opened=False,seconds=time.monotonic()-began));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['run','worker'])
    p.add_argument('--output',required=True,type=Path);a=p.parse_args();(run if a.mode=='run' else worker)(a.output)
