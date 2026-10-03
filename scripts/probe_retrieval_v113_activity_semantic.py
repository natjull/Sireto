"""Official-description semantic prior with nested protected training controls."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

from scripts.probe_retrieval_v111_activity_decoder import inputs as training_inputs,training
from src.xgb_matcher.retrieval_activity_semantic_v113 import class_lookup,normalized,select
from src.xgb_matcher.retrieval_v20_workbench import Budget,ROOT,record,verify,write_json

CONFIG=ROOT/'config/retrieval_v113_activity_semantic.json'


def inputs(cfg):
    refs=training_inputs(cfg);catalog=json.loads(verify(cfg['catalog']).read_text())
    launch=json.loads(verify(catalog['launch']).read_text())
    if (catalog['status']!='SEALED_COMPLETE_EXPLICIT_NATIONAL_NOMENCLATURE' or catalog['truth_opened'] or
        catalog['crm_opened'] or catalog['stats']['sites']!=cfg['expected_sites'] or
        catalog['stats']['persisted_roundtrip_rows']!=cfg['expected_sites']):
        raise ValueError('Full explicit official catalog required')
    for ref in launch['implementation'].values():verify(ref)
    for ref in catalog['artifacts'].values():verify(ref)
    verify(catalog['official_labels'])
    source=json.loads(verify(cfg['vectors']).read_text());old=json.loads(verify(source['launch']).read_text())
    for ref in old['model_refs'].values():verify(ref)
    return dict(training=refs,nomenclature=catalog['artifacts']['nomenclature_codes.npy'],
        original_activity=catalog['artifacts']['original_activity_codes.npy'],labels=catalog['official_labels'],
        nomenclatures=catalog['stats']['nomenclatures'],activities=catalog['stats']['activity_book'],
        model_refs=old['model_refs'],model_path=old['config']['model_path'],max_length=old['config']['max_length'])


def encode(output):
    launch=json.loads((output/'launch.json').read_text());cfg=launch['config'];refs=launch['inputs'];budget=None
    for ref in launch['implementation'].values():verify(ref)
    try:
        with Budget(cfg['resources']) as budget:
            if inputs(cfg)!=refs:raise ValueError('Official encoder source drift')
            data,q,_=training(refs['training'],cfg)
            labels=json.loads(verify(refs['labels']).read_text())['rows']
            if len(labels)!=cfg['expected_descriptions']:raise ValueError('Complete official description cardinality drift')
            from src.xgb_matcher.retrieval_contrastive_v52 import surface
            from src.xgb_matcher.semantic import _repair_exported_tokenizer,assert_tokenizer_healthy
            import torch
            from sentence_transformers import SentenceTransformer
            torch.set_num_threads(2);torch.set_num_interop_threads(1)
            model=SentenceTransformer(refs['model_path'],device='cpu',local_files_only=True)
            _repair_exported_tokenizer(model,refs['model_path']);assert_tokenizer_healthy(model.tokenizer)
            model.max_seq_length=refs['max_length']
            parity=cfg['encoder_parity_queries']
            text=[data['query_fields'][r['query_id']]['name'] for r in data['rows'][:parity]]
            check=normalized(model.encode(text,batch_size=parity,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False))
            delta=float(np.max(abs(check-normalized(q[:parity]))))
            if delta>cfg['encoder_parity_tolerance']:raise ValueError('Fixed NOM query encoder parity drift')
            # No categorical codes/nomenclature/identifiers enter this text.
            texts=[surface([r['label']],'nom')[0] for r in labels]
            counts=[len(model.tokenizer(t)['input_ids']) for t in texts]
            before=time.monotonic()
            head=model.encode(texts[:128],batch_size=32,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
            estimated=(time.monotonic()-before)*len(texts)/128
            if estimated>cfg['max_estimated_encoding_seconds']:raise ValueError('STOP_ENCODING_COST_PREFLIGHT')
            rest=model.encode(texts[128:],batch_size=32,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
            vectors=np.concatenate([head,rest]).astype(np.float32)
            if vectors.shape!=(len(labels),384) or np.any(abs(np.linalg.norm(vectors,axis=1)-1)>1e-4):
                raise ValueError('Complete description vectors shape/norm drift')
            np.save(output/'description_vectors.npy',vectors,allow_pickle=False)
            if inputs(cfg)!=refs:raise ValueError('Official encoding source changed')
        for ref in launch['implementation'].values():verify(ref)
        write_json(output/'encoding.json',dict(vectors=record(output/'description_vectors.npy'),
            source=refs['labels'],descriptions=len(labels),encoder_prefix='nom',identifiers_in_text=False,
            query_parity_queries=parity,query_parity_max_abs=delta,estimated_seconds=estimated,
            truncated_descriptions=sum(c>refs['max_length'] for c in counts),resources=budget.report(),
            training_truth_used_as_text=False,recall_measured=False))
        print(json.dumps(dict(encoded=len(labels),parity_max_abs=delta,estimated_seconds=estimated,resources=budget.report())),flush=True)
    except BaseException as exc:
        write_json(output/'encoding_stop.json',dict(status='STOP',reason=str(exc),recall_measured=False,
            resources=budget.report() if budget is not None and hasattr(budget,'elapsed') else None));raise


def worker(output):
    launch=json.loads((output/'launch.json').read_text());cfg=launch['config'];refs=launch['inputs'];budget=None
    for ref in launch['implementation'].values():verify(ref)
    try:
        with Budget(cfg['resources']) as budget:
            if inputs(cfg)!=refs:raise ValueError('Semantic training source drift')
            data,q,old_codes=training(refs['training'],cfg);encoding=json.loads(verify(launch['encoding']).read_text())
            if (encoding['descriptions']!=cfg['expected_descriptions'] or not encoding['resources']['pass_resource_policy'] or
                encoding['identifiers_in_text'] or encoding['query_parity_max_abs']>cfg['encoder_parity_tolerance']):
                raise ValueError('Complete successful official-only encoding required')
            descriptions=np.load(verify(encoding['vectors']),allow_pickle=False)
            labels=json.loads(verify(refs['labels']).read_text())['rows'];mapping=class_lookup(refs['nomenclatures'],refs['activities'],labels)
            keys=np.load(verify(refs['training']['keys']),mmap_mode='r',allow_pickle=False)
            nom=np.load(verify(refs['nomenclature']),mmap_mode='r',allow_pickle=False)
            act=np.load(verify(refs['original_activity']),mmap_mode='r',allow_pickle=False)
            if nom.dtype!=np.uint8 or act.dtype!=np.uint16 or nom.shape!=keys.shape or act.shape!=keys.shape:
                raise ValueError('Complete typed nomenclature/registry shape drift')
            ids=sorted(old_codes);numeric=np.array([int(s) for s in ids],dtype=np.uint64);pos=np.searchsorted(keys,numeric)
            if np.any(pos>=len(keys)) or not np.array_equal(keys[pos],numeric):raise ValueError('Official TRAIN site absent')
            codes=dict(zip(ids,mapping[nom[pos],act[pos]].astype(int).tolist()));models={};reports={}
            for held in (2,3,4):
                parameter,reports[str(held)]=select(q,descriptions,data['rows'],data['registry'],codes,held,cfg['ridges'])
                path=output/f'semantic_activity_fold{held}.npy';np.save(path,parameter,allow_pickle=False)
                r=reports[str(held)]
                models[str(held)]=dict(parameter=record(path),heldout_fold=held,training_folds=r['training_folds'],query_ids=r['query_ids'],
                    family=r['selected']['family'],ridge=r['selected']['ridge'],descriptions=encoding['vectors'],
                    catalog=cfg['catalog'],identifiers_as_features=False,probabilities_calibrated=False,national_serving_integrated=False)
                print(json.dumps(dict(fold=held,selected=r['selected'],outer=r['outer'])),flush=True)
            if inputs(cfg)!=refs:raise ValueError('Semantic official/training authorities changed')
        for ref in launch['implementation'].values():verify(ref)
        write_json(output/'models.json',models)
        write_json(output/'manifest.json',dict(status='SEALED_TRAINING_ONLY_OFFICIAL_ACTIVITY_SEMANTIC_PROBE',
            launch=record(output/'launch.json'),models=record(output/'models.json'),reports=reports,encoding=launch['encoding'],
            resources=budget.report(),training_queries=len(data['rows']),training_documents=len(codes),
            unknown_training_documents=sum(c<0 for c in codes.values()),truth_scope='TRAINER_ONLY_FOLDS234',
            fold0_opened=False,fold1_opened=False,recall_measured=False,output_candidates=False,national_search_performed=False,
            no_crm_alias_dictionary=True,identifiers_as_features=False,downstream_models_changed=False,ready_for_fold1=False))
    except BaseException as exc:
        write_json(output/'worker_stop.json',dict(status='STOP',reason=str(exc),recall_measured=False,
            resources=budget.report() if budget is not None and hasattr(budget,'elapsed') else None));raise


def run(output):
    output.mkdir(parents=True,exist_ok=False);began=time.monotonic();cfg=json.loads(CONFIG.read_text())
    paths=(Path(__file__),CONFIG,ROOT/'docs/retrieval_v113_activity_semantic.md',
        ROOT/'src/xgb_matcher/retrieval_activity_semantic_v113.py',ROOT/'src/xgb_matcher/retrieval_activity_decoder_v111.py',
        ROOT/'scripts/probe_retrieval_v111_activity_decoder.py',ROOT/'src/xgb_matcher/retrieval_nested_v53.py',
        ROOT/'src/xgb_matcher/retrieval_contrastive_v52.py',ROOT/'src/xgb_matcher/semantic.py',
        ROOT/'src/xgb_matcher/retrieval_v20_workbench.py',ROOT/'src/xgb_matcher/retrieval_parent_discovery_v18.py')
    launch=dict(config=cfg,inputs=inputs(cfg),implementation={str(p):record(p) for p in paths},recall_measured=False)
    write_json(output/'launch.json',launch)
    try:
        for mode in ('encode','worker'):
            with (output/(mode+'.log')).open('x') as log:
                subprocess.run([sys.executable,'-u','-m','scripts.probe_retrieval_v113_activity_semantic',mode,'--output',str(output)],
                    check=True,timeout=90,stdout=log,stderr=subprocess.STDOUT,
                    env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false',
                        'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','VECLIB_MAXIMUM_THREADS':'2'})
            print((output/(mode+'.log')).read_text()[-7000:],flush=True)
            if mode=='encode':
                launch['encoding']=record(output/'encoding.json');write_json(output/'launch.json',launch)
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),seconds=time.monotonic()-began,recall_measured=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['run','encode','worker']);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();{'run':run,'encode':encode,'worker':worker}[a.mode](a.output)
