"""Family-disjoint training-only raw activity prediction and natural controls."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

from src.xgb_matcher.retrieval_activity_decoder_v111 import select
from src.xgb_matcher.retrieval_v20_workbench import Budget,ROOT,record,verify,write_json

CONFIG=ROOT/'config/retrieval_v111_activity_decoder.json'


def inputs(cfg):
    context=json.loads(verify(cfg['context']).read_text())
    if (context['status']!='SEALED_COMPLETE_NATIONAL_OFFICIAL_CONTEXT' or context['truth_opened'] or context['crm_opened'] or
        context['stats']['sites']!=cfg['expected_sites'] or context['stats']['closed_sites_removed'] or
        context['stats']['persisted_context_roundtrip_rows']!=cfg['expected_sites']):
        raise ValueError('Complete raw official context required')
    launch=json.loads(verify(context['launch']).read_text())
    for ref in launch['implementation'].values():verify(ref)
    activity=context['artifacts']['activity_codes.npy'];keys=context['inputs']['registry_arrays']['endpoint_keys']
    verify(activity);verify(keys)
    source=json.loads(verify(cfg['vectors']).read_text())
    if source['recall_measured'] or source['fold0_opened'] or source['fold1_opened']:
        raise ValueError('Training-only open folds required')
    old_launch=json.loads(verify(source['launch']).read_text())
    for ref in old_launch['implementation'].values():verify(ref)
    cache=json.loads(verify(source['worker']).read_text())
    if not cache['training_panel_only'] or cache['output_candidates'] or cache['identifiers_as_features']:
        raise ValueError('Fixed masked training name vectors required')
    verify(old_launch['inputs']);verify(cache['query_vectors'])
    return dict(context=cfg['context'],activity=activity,keys=keys,training=old_launch['inputs'],
                vectors=cache['query_vectors'],query_ids=cache['query_ids'],classes=len(context['stats']['codebooks']['main_activity']))


def training(refs,cfg):
    data=json.loads(verify(refs['training']).read_text());rows=data['rows']
    if (len(rows)!=cfg['expected_training_queries'] or set(data['registry'])!={'2','3','4'} or
        any(r['fold'] not in (2,3,4) for r in rows) or refs['query_ids']!=[r['query_id'] for r in rows]):
        raise ValueError('Training-only query alignment drift')
    x=np.load(verify(refs['vectors']),allow_pickle=False)
    if x.shape!=(len(rows),768):raise ValueError('Fixed NOM/address source vector shape drift')
    keys=np.load(verify(refs['keys']),mmap_mode='r',allow_pickle=False)
    activity=np.load(verify(refs['activity']),mmap_mode='r',allow_pickle=False)
    if keys.dtype!=np.uint64 or activity.dtype!=np.uint16 or keys.shape!=activity.shape or len(keys)!=cfg['expected_sites']:
        raise ValueError('Full registry context dtype/cardinality drift')
    ids=sorted({s for r in rows for s in [r['positive_siret'],*[n['siret'] for n in r['negatives']]]})
    if any(not(isinstance(s,str) and s.isascii() and s.isdigit() and len(s)==14) for s in ids):
        raise ValueError('Invalid official TRAIN join')
    numeric=np.array([int(s) for s in ids],dtype=np.uint64);pos=np.searchsorted(keys,numeric)
    if np.any(pos>=len(keys)) or not np.array_equal(keys[pos],numeric):
        raise ValueError('Missing official training context; no positive insertion or row filtering')
    codes=dict(zip(ids,activity[pos].astype(int).tolist()))
    return data,x[:,:384],codes


def worker(output):
    launch=json.loads((output/'launch.json').read_text());cfg=launch['config'];budget=None
    for ref in launch['implementation'].values():verify(ref)
    try:
        with Budget(cfg['resources']) as budget:
            refs=inputs(cfg)
            if refs!=launch['inputs']:raise ValueError('Training authorities drift')
            data,x,codes=training(refs,cfg);reports={};models={}
            for held in (2,3,4):
                projection,reports[str(held)]=select(x,data['rows'],data['registry'],codes,refs['classes'],held,cfg['ridges'])
                path=output/f'activity_projection_fold{held}.npy';np.save(path,projection,allow_pickle=False)
                report=reports[str(held)]
                models[str(held)]=dict(projection=record(path),heldout_fold=held,training_folds=report['training_folds'],
                    query_ids=report['query_ids'],selected=report['selected'],all_raw_classes=refs['classes'],
                    query_dimension=384,query_prefix='nom',missing_codes_zero=[0,1],probabilities_calibrated=False,
                    context=cfg['context'],identifiers_as_features=False,national_serving_integrated=False)
                print(json.dumps(dict(fold=held,selected=report['selected'],outer=report['outer_selected'])),flush=True)
            if inputs(cfg)!=refs:raise ValueError('Official/training source changed')
        for ref in launch['implementation'].values():verify(ref)
        write_json(output/'models.json',models)
        write_json(output/'manifest.json',dict(status='SEALED_TRAINING_ONLY_RAW_ACTIVITY_PROBE',
            launch=record(output/'launch.json'),models=record(output/'models.json'),reports=reports,
            resources=budget.report(),training_queries=len(data['rows']),training_documents=len(codes),
            truth_scope='TRAINER_ONLY_FOLDS234',fold0_opened=False,fold1_opened=False,recall_measured=False,
            output_candidates=False,national_search_performed=False,no_crm_alias_dictionary=True,
            identifiers_as_features=False,downstream_models_changed=False,ready_for_fold1=False))
    except BaseException as exc:
        write_json(output/'worker_stop.json',dict(status='STOP',reason=str(exc),recall_measured=False,
            resources=budget.report() if budget is not None and hasattr(budget,'elapsed') else None));raise


def run(output):
    output.mkdir(parents=True,exist_ok=False);began=time.monotonic();cfg=json.loads(CONFIG.read_text())
    paths=(Path(__file__),CONFIG,ROOT/'docs/retrieval_v111_activity_decoder.md',
        ROOT/'src/xgb_matcher/retrieval_activity_decoder_v111.py',ROOT/'src/xgb_matcher/retrieval_nested_v53.py',
        ROOT/'src/xgb_matcher/retrieval_v20_workbench.py',ROOT/'src/xgb_matcher/retrieval_parent_discovery_v18.py')
    write_json(output/'launch.json',dict(config=cfg,inputs=inputs(cfg),implementation={str(p):record(p) for p in paths},
                                       truth_scope='TRAINER_ONLY_FOLDS234',recall_measured=False))
    try:
        with (output/'worker.log').open('x') as log:
            subprocess.run([sys.executable,'-u','-m','scripts.probe_retrieval_v111_activity_decoder','worker','--output',str(output)],
                check=True,timeout=90,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','VECLIB_MAXIMUM_THREADS':'2'})
        print((output/'worker.log').read_text()[-6000:],flush=True)
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),seconds=time.monotonic()-began,recall_measured=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['run','worker']);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();(run if a.mode=='run' else worker)(a.output)
