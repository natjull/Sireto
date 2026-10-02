"""Isolate actual national initial100 quality/cost of11 native learned proofs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts import retrieval_v38_initial_top100 as v38
from scripts import retrieval_v42_indexed_metric as v42
from scripts.retrieval_v49_primary_probe import evaluate
from scripts.retrieval_v56_exact_fields import index_authority,impl as old_impl
from src.xgb_matcher.hierarchical_retrieval import TantivyBackend
from src.xgb_matcher.retrieval_sparse_flat_v61 import LEXICAL,search
from src.xgb_matcher.retrieval_v20_workbench import Budget,ROOT,record,verify,write_json
CONFIG=ROOT/'config/retrieval_v61_sparse_flat.json'

def authority():
    result=index_authority()
    for p in (Path(__file__),CONFIG,ROOT/'src/xgb_matcher/retrieval_sparse_flat_v61.py',ROOT/'src/xgb_matcher/retrieval_sparse_query_v59.py',ROOT/'src/xgb_matcher/retrieval_contrastive_v52.py',ROOT/'src/xgb_matcher/semantic.py'):
        result['implementation'][str(p)]=record(p)
    return result

def package(output):
    cfg=json.loads(CONFIG.read_text()); output.mkdir(parents=True,exist_ok=False)
    source=json.loads(verify(cfg['training_source']).read_text()); sl=json.loads(verify(source['launch']).read_text()); v38.verify_launch(sl)
    if source['status']!='SEALED_TRAINING_ONLY_SEMANTIC_SPARSE_QUERY_PROBE' or source['recall_measured']: raise ValueError('Training-only decoder authority required')
    vectors=json.loads(verify(cfg['vector_source']).read_text()); vl=json.loads(verify(vectors['launch']).read_text()); v38.verify_launch(vl)
    models={}
    for held in (2,3,4):
        r=source['reports'][str(held)]; native=r['native']; verify(source['projections'][str(held)])
        if r['heldout_fold']!=held or native['heldout_fold']!=held or held in native['training_folds']: raise ValueError('OOF package drift')
        models[str(held)]=dict(projection=source['projections'][str(held)],weights=native['weights'],alpha=r['alpha'],
            features=native['features'],heldout_fold=held,training_folds=native['training_folds'],query_ids=r['query_ids'])
    verify(source['vocabulary']); write_json(output/'models.json',models)
    write_json(output/'manifest.json',dict(**authority(),status='SEALED_OOF_SPARSE_QUERY_SERVING_BUNDLE',models=record(output/'models.json'),
        vocabulary=source['vocabulary'],model_refs=vl['model_refs'],model_path=vl['config']['model_path'],
        max_length=vl['config']['max_length'],encoder_prefix='nom',sparse_terms=8,
        training_source=cfg['training_source'],training_controls_included=False,truth_scope='NO_LABEL_SIDE_IN_SERVING_BUNDLE',
        index_built=False,recall_measured=False,ready_for_fold1=False,fold0_opened=False,fold1_opened=False))

def encode(output):
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    from src.xgb_matcher.semantic import _repair_exported_tokenizer,assert_tokenizer_healthy
    from src.xgb_matcher.retrieval_contrastive_v52 import surface
    from src.xgb_matcher.retrieval_sparse_query_v59 import unit,expand
    launch=json.loads((output/'launch.json').read_text()); v38.verify_launch(launch)
    bundle=json.loads(verify(launch['bundle']).read_text()); models=json.loads(verify(bundle['models']).read_text())
    with Budget(launch['resources']) as budget:
        for ref in bundle['model_refs'].values(): verify(ref)
        model=SentenceTransformer(bundle['model_path'],device='cpu',local_files_only=True)
        _repair_exported_tokenizer(model,bundle['model_path']); assert_tokenizer_healthy(model.tokenizer)
        torch.set_num_threads(2); torch.set_num_interop_threads(1); model.max_seq_length=bundle['max_length']
        texts=[surface([r['crm_name']],bundle['encoder_prefix'])[0] for r in launch['queries']]
        q=unit(model.encode(texts,batch_size=32,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False))
        vocab=json.loads(verify(bundle['vocabulary']).read_text())
        projections={f:np.load(verify(m['projection']),allow_pickle=False) for f,m in models.items()}
        if q.shape!=(128,384) or any(w.shape!=(384,len(vocab)) for w in projections.values()): raise ValueError('Serving semantic shape drift')
        values=[dict(query_id=r['query_id'],fold=r['fold'],terms=expand(q[i],projections[str(r['fold'])],vocab,r['crm_name'],bundle['sparse_terms'])) for i,r in enumerate(launch['queries'])]
        np.save(output/'query_name_vectors.npy',q,allow_pickle=False)
    v38.verify_launch(launch); write_json(output/'expansions.json',dict(rows=values,resources=budget.report(),vectors=record(output/'query_name_vectors.npy'),truth_opened=False))

def worker(output,start):
    launch=json.loads((output/'launch.json').read_text()); v38.verify_launch(launch)
    models=json.loads(verify(launch['models']).read_text())
    expansions={r['query_id']:r for r in json.loads(verify(launch['expansions']).read_text())['rows']}
    with Budget(launch['resources']) as budget:
        backend=TantivyBackend(v38.INDEX); rows=[]
        if backend.searcher.num_docs!=launch['index_documents']: raise ValueError('Official cardinality drift')
        for raw in launch['queries'][start:start+8]:
            m=models[str(raw['fold'])]
            if m['features']!=list(LEXICAL) or m['heldout_fold']!=raw['fold'] or raw['fold'] in m['training_folds']:
                raise ValueError('Serving OOF proof drift')
            t=time.monotonic(); result,trace=search(backend,v42.as_query(raw),m['weights'],expansions[raw['query_id']]['terms'],m['alpha']); elapsed=time.monotonic()-t
            if elapsed>launch['max_query_search_seconds']: raise ValueError('STOP_QUERY_COST')
            rows.append(dict(query_id=raw['query_id'],fold=raw['fold'],results=dict(primary=result),seconds=dict(primary=elapsed),primary_trace=trace))
    v38.verify_launch(launch); write_json(output/f'batch{start:03d}.json',dict(rows=rows,resources=budget.report()))

def run(output,model_manifest):
    cfg=json.loads(CONFIG.read_text()); output.mkdir(parents=True,exist_ok=False); began=time.monotonic()
    try:
        trained=json.loads(model_manifest.read_text()); v38.verify_launch(trained)
        if trained['status']!='SEALED_OOF_SPARSE_QUERY_SERVING_BUNDLE': raise ValueError('Unsealed native model')
        models=json.loads(verify(trained['models']).read_text()); queries,qrefs=v42.input_rows(v38.load(v38.GEN,v38.GEN_SHA),4)
        if any(set(m['query_ids'])&{r['query_id'] for r in queries} for m in models.values()): raise ValueError('Fit/evaluation query reuse')
        launch=dict(**authority(),bundle=record(model_manifest),models=trained['models'],model_manifest=record(model_manifest),queries=queries,query_sources=qrefs,
            resources=cfg['resources'],max_query_search_seconds=cfg['max_query_search_seconds'],
            collector_cap=100,fixed_reservoir=None,truth_opened=False,geometry_as_feature=False,development_panel_reused=True,
            comparator_scope='SEPARATE_EVALUATOR_SEALED_V42_NO_GENERATOR_READ')
        write_json(output/'launch.json',launch)
        with (output/'encode.log').open('x') as log:
            subprocess.run([sys.executable,'-u','-m','scripts.retrieval_v61_sparse_flat','encode','--output',str(output)],
                check=True,stdout=log,stderr=subprocess.STDOUT,timeout=90,
                env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','VECLIB_MAXIMUM_THREADS':'2','RAYON_NUM_THREADS':'2'})
        encoded=json.loads((output/'expansions.json').read_text())
        if not encoded['resources']['pass_resource_policy'] or len(encoded['rows'])!=128: raise ValueError('Query-only encoding scope/resource drift')
        launch['expansions']=record(output/'expansions.json'); write_json(output/'launch.json',launch); rows=[]; refs=[]
        for start in range(0,128,8):
            with (output/f'worker{start:03d}.log').open('x') as log:
                subprocess.run([sys.executable,'-u','-m','scripts.retrieval_v61_sparse_flat','worker','--output',str(output),'--start',str(start)],
                    check=True,stdout=log,stderr=subprocess.STDOUT,timeout=min(90,max(1,cfg['controller_seconds']-(time.monotonic()-began))),
                    env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','VECLIB_MAXIMUM_THREADS':'2','RAYON_NUM_THREADS':'2'})
            path=output/f'batch{start:03d}.json'; part=json.loads(path.read_text())
            if len(part['rows'])!=8 or not part['resources']['pass_resource_policy']: raise ValueError('Native serving scope/resource drift')
            rows.extend(part['rows']); refs.append(record(path)); print(json.dumps(dict(generated=len(rows))),flush=True)
            if len(rows)==16:
                mean=sum(r['seconds']['primary'] for r in rows)/16
                write_json(output/'cost_gate.json',dict(queries=16,mean_primary_seconds=mean,passed=mean<=cfg['primary_mean16_seconds'],
                    quality_opened=False,not_production_slo=True))
                if mean>cfg['primary_mean16_seconds']: raise ValueError('STOP_PRIMARY_COST_PREFLIGHT')
        v38.verify_launch(launch)
        if len(rows)!=128 or len({r['query_id'] for r in rows})!=128: raise ValueError('Serving denominator drift')
        write_json(output/'manifest.json',dict(status='SEALED_INITIAL_TOP100',queries=128,batches=refs,launch=record(output/'launch.json'),
            seconds=time.monotonic()-began,truth_opened=False,model_used=True,full_query_population=False,ready_for_fold1=False,
            metric_family=cfg['metric_family'],full_lexical_corpus=True,full_spatial_corpus=False,geometry_as_feature=False,
            old_candidates_opened=False,comparator_replayed=False))
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),seconds=time.monotonic()-began,truth_opened=False,ready_for_fold1=False)); raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=['package','encode','worker','run','evaluate'])
    p.add_argument('--output',required=True,type=Path); p.add_argument('--models',type=Path); p.add_argument('--start',type=int); a=p.parse_args()
    if a.mode=='package': package(a.output)
    elif a.mode=='encode': encode(a.output)
    elif a.mode=='evaluate': evaluate(a.output)
    elif a.mode=='run':
        if a.models is None: p.error('--models required')
        run(a.output,a.models)
    else:
        if a.start not in range(0,128,8): p.error('Invalid batch')
        worker(a.output,a.start)


