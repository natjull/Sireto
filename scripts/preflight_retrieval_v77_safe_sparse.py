"""Bounded nested sparse-query fit with per-fold safe official vocabulary."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from scripts import retrieval_v38_initial_top100 as v38
from scripts.retrieval_sparse_training_source import load
from scripts.retrieval_v56_exact_fields import index_authority
from src.xgb_matcher.retrieval_safe_sparse_v77 import select
from src.xgb_matcher.retrieval_sparse_query_v59 import unit
from src.xgb_matcher.retrieval_v20_workbench import Budget,ROOT,record,verify,write_json

CONFIG=ROOT/'config/retrieval_v77_safe_sparse.json'


def authority():
    a=index_authority()
    for p in (Path(__file__),CONFIG,ROOT/'docs/retrieval_v77_safe_sparse_probe.md',
        ROOT/'src/xgb_matcher/retrieval_safe_sparse_v77.py',ROOT/'scripts/retrieval_sparse_training_source.py',
        ROOT/'src/xgb_matcher/retrieval_sparse_query_v59.py',ROOT/'src/xgb_matcher/retrieval_nested_v53.py',
        ROOT/'src/xgb_matcher/retrieval_native_v57.py'):
        a['implementation'][str(p)]=record(p)
    return a


def worker(output):
    launch=json.loads((output/'launch.json').read_text()); v38.verify_launch(launch); cfg=launch['config']
    budget=Budget(cfg['resources'])
    with budget:
        _,rows,scenes,documents,_,registry,counters=load(cfg,launch)
        source=json.loads(verify(cfg['vector_source']).read_text())
        sl=json.loads(verify(source['launch']).read_text()); cache=json.loads(verify(source['worker']).read_text())
        q=unit(np.load(verify(cache['query_vectors']),allow_pickle=False)[:,:384])
        # Reuse the exact V52/V59 name encoder representation, not a new model.
        names={sid:{f:values.get(f,[]) for f in v38.NAMES} for sid,values in documents.items()}
        models={}; reports={}
        for held in (2,3,4):
            projection,vocab,reports[str(held)]=select(q,rows,scenes,names,registry,held,cfg)
            vp=output/f'vocabulary_fold{held}.json'; pp=output/f'projection_fold{held}.npy'
            write_json(vp,vocab); np.save(pp,projection,allow_pickle=False)
            r=reports[str(held)]; native=r['native']
            models[str(held)]=dict(projection=record(pp),vocabulary=record(vp),weights=native['weights'],
                alpha=r['alpha'],features=native['features'],heldout_fold=held,training_folds=native['training_folds'],
                query_ids=r['query_ids'],vocabulary_protected_document_overlap=0,
                heldout_documents_used_for_vocabulary=False)
        write_json(output/'models.json',models)
    v38.verify_launch(launch); verify(cfg['vector_source']); verify(cfg['proof_source']); verify(cfg['alias_validation'])
    write_json(output/'manifest.json',dict(**authority(),status='SEALED_SAFE_NESTED_SPARSE_QUERY_BUNDLE',
        launch=record(output/'launch.json'),models=record(output/'models.json'),reports=reports,counters=counters,
        resources=budget.report(),model_path=sl['config']['model_path'],model_refs=sl['model_refs'],
        max_length=sl['config']['max_length'],encoder_prefix='nom',sparse_terms=cfg['sparse_query_terms'],
        vocabulary_scope='SEPARATE_PER_OUTER_FOLD_AND_INNER_FIT_PROTECTED_PARENTS_EXCLUDED',
        targets_scope='FIT_QUERY_POSITIONS_ONLY_NO_HELDOUT_TARGETS_CONSTRUCTED',
        recall_measured=False,truth_scope='TRAIN_ONLY_NO_HUMAN_LABEL_SIDE_IN_SERVING_MODELS',
        full_national_search_performed=False,ready_for_fold1=False,fold0_opened=False,fold1_opened=False,
        contains_crm_alias_dictionary=False,identifiers_as_features=False,downstream_models_changed=False,
        source_scope='OFFICIAL_TRAIN_DOCUMENTS_NOT_FULL_NATIONAL_VOCABULARY'))
    print(json.dumps(dict(resources=budget.report(),reports={f:dict(selected=r['selected'],
        inner_vocab=r['inner_vocabulary_scope'],final_vocab=r['final_vocabulary_scope'],
        before=r['outer_before'],after=r['outer_after']) for f,r in reports.items()},recall_measured=False)),flush=True)


def run(output):
    output.mkdir(parents=True,exist_ok=False); cfg=json.loads(CONFIG.read_text())
    write_json(output/'launch.json',dict(**authority(),config=cfg,truth_scope='TRAIN_ONLY'))
    try:
        with (output/'worker.log').open('x') as log:
            subprocess.run([sys.executable,'-u','-m','scripts.preflight_retrieval_v77_safe_sparse','worker','--output',str(output)],
                check=True,timeout=90,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2',
                    'VECLIB_MAXIMUM_THREADS':'2','RAYON_NUM_THREADS':'2'})
        print((output/'worker.log').read_text()[-5000:],flush=True)
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),recall_measured=False)); raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=['run','worker'])
    p.add_argument('--output',required=True,type=Path); a=p.parse_args()
    (run if a.mode=='run' else worker)(a.output)
