"""Nested source-proof/activity interaction probe with cross-fit on head TRAIN."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

from scripts.probe_retrieval_v113_activity_semantic import inputs as semantic_inputs
from scripts.probe_retrieval_v111_activity_decoder import training
from src.xgb_matcher.retrieval_activity_semantic_v113 import class_lookup
from src.xgb_matcher.retrieval_activity_joint_v114 import select
from src.xgb_matcher.retrieval_v20_workbench import Budget,ROOT,record,verify,write_json

CONFIG=ROOT/'config/retrieval_v114_activity_joint.json'


def inputs(cfg):
    semantic=json.loads(verify(cfg['semantic']).read_text())
    if (semantic['status']!='SEALED_TRAINING_ONLY_OFFICIAL_ACTIVITY_SEMANTIC_PROBE' or semantic['recall_measured'] or
        semantic['fold0_opened'] or semantic['fold1_opened'] or semantic['output_candidates']):
        raise ValueError('Safe fixed semantic TRAIN probe required')
    launch=json.loads(verify(semantic['launch']).read_text())
    for ref in launch['implementation'].values():verify(ref)
    refs=semantic_inputs(launch['config'])
    if refs!=launch['inputs']:raise ValueError('Complete semantic source changed')
    encoding=json.loads(verify(semantic['encoding']).read_text())
    verify(encoding['vectors']);verify(semantic['models'])
    if encoding['descriptions']!=cfg['expected_descriptions'] or not encoding['resources']['pass_resource_policy']:
        raise ValueError('Complete successful descriptions required')
    proof=json.loads(verify(cfg['proof_source']).read_text())
    if (proof['status']!='SEALED_OOF_INITIAL_PROOF_MODELS' or proof['recall_measured'] or proof['fold0_opened'] or
        proof['fold1_opened'] or proof['training_queries']!=cfg['expected_training_queries'] or proof['identifiers_as_features']):
        raise ValueError('Safe real native/geometry TRAIN proof required')
    for ref in proof['implementation'].values():verify(ref)
    for ref in proof['features']:verify(ref)
    return dict(semantic_refs=refs,descriptions=encoding['vectors'],semantic_models=semantic['models'],
                proofs=proof['features'],semantic_config=launch['config'])


def load(refs,cfg):
    sr=refs['semantic_refs'];data,q,old_codes=training(sr['training'],cfg)
    descriptions=np.load(verify(refs['descriptions']),allow_pickle=False)
    labels=json.loads(verify(sr['labels']).read_text())['rows'];mapping=class_lookup(sr['nomenclatures'],sr['activities'],labels)
    keys=np.load(verify(sr['training']['keys']),mmap_mode='r',allow_pickle=False)
    nom=np.load(verify(sr['nomenclature']),mmap_mode='r',allow_pickle=False)
    activity=np.load(verify(sr['original_activity']),mmap_mode='r',allow_pickle=False)
    if nom.dtype!=np.uint8 or activity.dtype!=np.uint16 or nom.shape!=keys.shape or activity.shape!=keys.shape:
        raise ValueError('Complete official activity dtype/cardinality drift')
    ids=sorted(old_codes);numeric=np.array([int(s) for s in ids],dtype=np.uint64);pos=np.searchsorted(keys,numeric)
    if np.any(pos>=len(keys)) or not np.array_equal(keys[pos],numeric):raise ValueError('Missing official TRAIN join')
    codes=dict(zip(ids,mapping[nom[pos],activity[pos]].astype(int).tolist()))
    scenes=[s for ref in refs['proofs'] for s in json.loads(verify(ref).read_text())['scenes']]
    if len(scenes)!=len(data['rows']):raise ValueError('Entire native TRAIN panel required')
    models=json.loads(verify(refs['semantic_models']).read_text())
    if set(models)!={'2','3','4'}:raise ValueError('Semantic decoder fold drift')
    return data,q,descriptions,codes,scenes,models


def worker(output):
    launch=json.loads((output/'launch.json').read_text());cfg=launch['config'];budget=None
    for ref in launch['implementation'].values():verify(ref)
    try:
        with Budget(cfg['resources']) as budget:
            refs=inputs(cfg)
            if refs!=launch['inputs']:raise ValueError('Joint source launch drift')
            data,q,descriptions,codes,scenes,decoders=load(refs,cfg);models={};reports={}
            for held in (2,3,4):
                old=decoders[str(held)]
                if (old['heldout_fold']!=held or set(old['training_folds'])!={2,3,4}-{held} or
                    old['identifiers_as_features'] or old['catalog']!=refs['semantic_config']['catalog']):
                    raise ValueError('Unsafe borrowed semantic decoder selection')
                decoder=dict(family=old['family'],ridge=old['ridge'])
                parameter,models[str(held)],reports[str(held)]=select(q,descriptions,data['rows'],scenes,
                    data['registry'],codes,held,decoder,cfg['extra_penalty'])
                if (set(models[str(held)]['query_ids'])!=set(old['query_ids']) or
                    not np.array_equal(parameter,np.load(verify(old['parameter']),allow_pickle=False))):
                    raise ValueError('Outer decoder exact refit/parity drift')
                path=output/f'decoder_fold{held}.npy';np.save(path,parameter,allow_pickle=False)
                models[str(held)]['decoder_parameter']=record(path);models[str(held)]['descriptions']=refs['descriptions']
                models[str(held)]['official_catalog']=refs['semantic_config']['catalog']
                r=reports[str(held)]
                print(json.dumps(dict(fold=held,selected=r['selected'],outer=r['outer'])),flush=True)
            if inputs(cfg)!=refs:raise ValueError('Joint official/training sources changed')
        for ref in launch['implementation'].values():verify(ref)
        write_json(output/'models.json',models)
        write_json(output/'manifest.json',dict(status='SEALED_TRAINING_ONLY_CROSSFIT_ACTIVITY_SITE_PROBE',
            launch=record(output/'launch.json'),models=record(output/'models.json'),reports=reports,resources=budget.report(),
            training_queries=len(data['rows']),training_documents=len(codes),truth_scope='TRAINER_ONLY_FOLDS234',
            fold0_opened=False,fold1_opened=False,recall_measured=False,output_candidates=False,national_search_performed=False,
            no_crm_alias_dictionary=True,identifiers_as_features=False,downstream_models_changed=False,
            activity_head_training_cross_fitted=True,outer_decoder_parity=True,ready_for_fold1=False))
    except BaseException as exc:
        write_json(output/'worker_stop.json',dict(status='STOP',reason=str(exc),recall_measured=False,
            resources=budget.report() if budget is not None and hasattr(budget,'elapsed') else None));raise


def run(output):
    output.mkdir(parents=True,exist_ok=False);began=time.monotonic();cfg=json.loads(CONFIG.read_text())
    paths=(Path(__file__),CONFIG,ROOT/'docs/retrieval_v114_activity_joint.md',
        ROOT/'src/xgb_matcher/retrieval_activity_joint_v114.py',ROOT/'src/xgb_matcher/retrieval_activity_semantic_v113.py',
        ROOT/'src/xgb_matcher/retrieval_joint_contrastive_v63.py',ROOT/'src/xgb_matcher/retrieval_proof_v56.py',
        ROOT/'src/xgb_matcher/retrieval_nested_v53.py',ROOT/'scripts/probe_retrieval_v113_activity_semantic.py',
        ROOT/'scripts/probe_retrieval_v111_activity_decoder.py',ROOT/'src/xgb_matcher/retrieval_activity_decoder_v111.py',
        ROOT/'src/xgb_matcher/retrieval_v20_workbench.py',ROOT/'src/xgb_matcher/retrieval_parent_discovery_v18.py')
    write_json(output/'launch.json',dict(config=cfg,inputs=inputs(cfg),implementation={str(p):record(p) for p in paths},recall_measured=False))
    try:
        with (output/'worker.log').open('x') as log:
            subprocess.run([sys.executable,'-u','-m','scripts.probe_retrieval_v114_activity_joint','worker','--output',str(output)],
                check=True,timeout=90,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','VECLIB_MAXIMUM_THREADS':'2'})
        print((output/'worker.log').read_text()[-8000:],flush=True)
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),seconds=time.monotonic()-began,recall_measured=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['run','worker']);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();(run if a.mode=='run' else worker)(a.output)
