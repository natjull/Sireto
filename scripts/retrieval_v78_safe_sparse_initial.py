"""Initial100 national search with safe per-fold semantic vocabulary V77."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from scripts import retrieval_v38_initial_top100 as v38
from scripts import retrieval_v42_indexed_metric as v42
from scripts import retrieval_v49_primary_probe as evaluator
from scripts.retrieval_v56_exact_fields import index_authority
from src.xgb_matcher.hierarchical_retrieval import TantivyBackend
from src.xgb_matcher.retrieval_sparse_native_v60 import LEXICAL,search
from src.xgb_matcher.retrieval_sparse_query_v59 import expand,unit
from src.xgb_matcher.retrieval_v20_workbench import Budget,ROOT,record,verify,write_json

CONFIG=ROOT/'config/retrieval_v78_safe_sparse_initial.json'


def authority():
    a=index_authority()
    for p in (Path(__file__),CONFIG,ROOT/'docs/retrieval_v78_safe_sparse_initial_probe.md',
        ROOT/'src/xgb_matcher/retrieval_sparse_native_v60.py',ROOT/'src/xgb_matcher/retrieval_sparse_query_v59.py',
        ROOT/'src/xgb_matcher/retrieval_contrastive_v52.py',ROOT/'src/xgb_matcher/semantic.py',
        ROOT/'scripts/retrieval_v49_primary_probe.py',ROOT/'scripts/compare_retrieval_v78_baselines.py'):
        a['implementation'][str(p)]=record(p)
    return a


def validate_models(models):
    if set(models)!={'2','3','4'}:
        raise ValueError('Closed or missing serving fold')
    for f,m in models.items():
        if (m['heldout_fold']!=int(f) or int(f) in m['training_folds'] or
            set(m['training_folds'])!={2,3,4}-{int(f)} or m['features']!=list(LEXICAL) or
            m['heldout_documents_used_for_vocabulary'] or m['vocabulary_protected_document_overlap']!=0):
            raise ValueError('Unsafe serving vocabulary/fold')
        verify(m['projection']); verify(m['vocabulary'])


def expansion_rows(vectors,queries,vocabularies,projections,count):
    rows=[]
    if len(vectors)!=len(queries) or count!=8:
        raise ValueError('Query-only encoder population drift')
    for i,q in enumerate(queries):
        f=str(q['fold'])
        if f not in ('2','3','4'):
            raise ValueError('Closed fold in query encoding')
        rows.append(dict(query_id=q['query_id'],fold=q['fold'],
            terms=expand(vectors[i],projections[f],vocabularies[f],q['crm_name'],count)))
    return rows


def package(output):
    cfg=json.loads(CONFIG.read_text()); trained=json.loads(verify(cfg['trained']).read_text()); v38.verify_launch(trained)
    if trained['status']!='SEALED_SAFE_NESTED_SPARSE_QUERY_BUNDLE' or trained['recall_measured']:
        raise ValueError('Safe training-only bundle required')
    models=json.loads(verify(trained['models']).read_text()); validate_models(models)
    for ref in trained['model_refs'].values():
        verify(ref)
    # No reports, targets, teacher negatives or label sidecar exported.
    write_json(output/'bundle.json',dict(**authority(),status='SEALED_SAFE_SPARSE_SERVING_ONLY',
        training=cfg['trained'],models=trained['models'],model_path=trained['model_path'],
        model_refs=trained['model_refs'],max_length=trained['max_length'],encoder_prefix='nom',sparse_terms=8,
        contains_training_controls=False,truth_opened=False))


def encode(output):
    import torch
    from sentence_transformers import SentenceTransformer
    from src.xgb_matcher.semantic import _repair_exported_tokenizer,assert_tokenizer_healthy
    from src.xgb_matcher.retrieval_contrastive_v52 import surface
    launch=json.loads((output/'launch.json').read_text()); v38.verify_launch(launch)
    b=json.loads(verify(launch['bundle']).read_text()); models=json.loads(verify(b['models']).read_text()); validate_models(models)
    with Budget(launch['resources']) as budget:
        for ref in b['model_refs'].values():
            verify(ref)
        model=SentenceTransformer(b['model_path'],device='cpu',local_files_only=True)
        _repair_exported_tokenizer(model,b['model_path']); assert_tokenizer_healthy(model.tokenizer)
        torch.set_num_threads(2); torch.set_num_interop_threads(1); model.max_seq_length=b['max_length']
        text=[surface([q['crm_name']],b['encoder_prefix'])[0] for q in launch['queries']]
        vectors=unit(model.encode(text,batch_size=32,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False))
        vocabs={f:json.loads(verify(m['vocabulary']).read_text()) for f,m in models.items()}
        projections={f:np.load(verify(m['projection']),allow_pickle=False) for f,m in models.items()}
        if vectors.shape!=(128,384) or any(projections[f].shape!=(384,len(vocabs[f])) for f in models):
            raise ValueError('Safe per-fold semantic shape drift')
        rows=expansion_rows(vectors,launch['queries'],vocabs,projections,b['sparse_terms'])
        np.save(output/'query_vectors.npy',vectors,allow_pickle=False)
    v38.verify_launch(launch)
    write_json(output/'expansions.json',dict(rows=rows,resources=budget.report(),vectors=record(output/'query_vectors.npy'),truth_opened=False))


def worker(output,start):
    launch=json.loads((output/'launch.json').read_text()); v38.verify_launch(launch)
    models=json.loads(verify(launch['models']).read_text()); validate_models(models)
    e=json.loads(verify(launch['expansions']).read_text()); expansions={r['query_id']:r for r in e['rows']}
    if len(expansions)!=128 or [(r['query_id'],r['fold']) for r in e['rows']]!=[(q['query_id'],q['fold']) for q in launch['queries']]:
        raise ValueError('Complete query-only expansion alignment required')
    with Budget(launch['resources']) as budget:
        base=TantivyBackend(v38.INDEX)
        if base.searcher.num_docs!=launch['index_documents']:
            raise ValueError('Full national cardinality drift')
        rows=[]
        for q in launch['queries'][start:start+8]:
            m=models[str(q['fold'])]; began=time.monotonic()
            found,trace=search(base,v42.as_query(q),m['weights'],expansions[q['query_id']]['terms'],m['alpha'])
            elapsed=time.monotonic()-began
            if elapsed>launch['max_query_search_seconds']:
                raise ValueError('STOP_QUERY_COST')
            rows.append(dict(query_id=q['query_id'],fold=q['fold'],results=dict(primary=found),
                             seconds=dict(primary=elapsed),primary_trace=trace))
    v38.verify_launch(launch)
    write_json(output/f'batch{start:03d}.json',dict(rows=rows,resources=budget.report()))


def supervised(output,mode,timeout,start=None):
    args=[sys.executable,'-u','-m','scripts.retrieval_v78_safe_sparse_initial',mode,'--output',str(output)]
    if start is not None:
        args+=['--start',str(start)]
    with (output/f'{mode}{start if start is not None else ""}.log').open('x') as log:
        subprocess.run(args,check=True,timeout=timeout,stdout=log,stderr=subprocess.STDOUT,
            env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2',
                 'VECLIB_MAXIMUM_THREADS':'2','RAYON_NUM_THREADS':'2'})


def run(output):
    output.mkdir(parents=True,exist_ok=False); began=time.monotonic(); cfg=json.loads(CONFIG.read_text())
    try:
        package(output); bundle=record(output/'bundle.json'); b=json.loads(verify(bundle).read_text())
        models=json.loads(verify(b['models']).read_text()); queries,refs=v42.input_rows(v38.load(v38.GEN,v38.GEN_SHA),4)
        if any(set(m['query_ids']) & {q['query_id'] for q in queries} for m in models.values()):
            raise ValueError('Fit/evaluation query overlap')
        launch=dict(**authority(),queries=queries,query_sources=refs,bundle=bundle,models=b['models'],
            model_manifest=cfg['trained'],resources=cfg['resources'],max_query_search_seconds=cfg['max_query_search_seconds'],
            truth_opened=False,fixed_reservoir=None,collector_cap=100,development_panel_reused=True,
            per_fold_vocabulary=True,geometry_as_feature=False,old_candidates_opened=False)
        write_json(output/'launch.json',launch)
        supervised(output,'encode',min(90,max(1,cfg['controller_seconds']-(time.monotonic()-began))))
        e=json.loads((output/'expansions.json').read_text())
        if not e['resources']['pass_resource_policy'] or len(e['rows'])!=128:
            raise ValueError('Encoder scope/resource drift')
        launch['expansions']=record(output/'expansions.json'); write_json(output/'launch.json',launch)
        rows=[]; batches=[]
        for start in range(0,128,8):
            supervised(output,'worker',min(90,max(1,cfg['controller_seconds']-(time.monotonic()-began))),start)
            p=output/f'batch{start:03d}.json'; part=json.loads(p.read_text())
            if (len(part['rows'])!=8 or not part['resources']['pass_resource_policy'] or
                [(r['query_id'],r['fold']) for r in part['rows']]!=[(q['query_id'],q['fold']) for q in queries[start:start+8]]):
                raise ValueError('Serving scope/resource drift')
            rows.extend(part['rows']); batches.append(record(p)); print(json.dumps(dict(generated=len(rows))),flush=True)
            if len(rows)==16:
                mean=sum(r['seconds']['primary'] for r in rows)/16
                write_json(output/'cost_gate.json',dict(mean_seconds=mean,passed=mean<=cfg['primary_mean16_seconds'],truth_opened=False))
                if mean>cfg['primary_mean16_seconds']:
                    raise ValueError('STOP_COST_PREFLIGHT')
        v38.verify_launch(launch); verify(bundle); validate_models(models)
        if len(rows)!=128 or len({r['query_id'] for r in rows})!=128:
            raise ValueError('Complete blind population required')
        write_json(output/'manifest.json',dict(status='SEALED_INITIAL_TOP100',queries=128,batches=batches,
            launch=record(output/'launch.json'),seconds=time.monotonic()-began,truth_opened=False,model_used=True,
            full_query_population=False,full_lexical_corpus=True,ready_for_fold1=False,metric_family=cfg['metric_family'],
            geometry_as_feature=False,per_fold_vocabulary=True,old_candidates_opened=False,comparator_replayed=False))
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),seconds=time.monotonic()-began,
            truth_opened=False,ready_for_fold1=False)); raise


def evaluate(output):
    cfg=json.loads(CONFIG.read_text()); old=evaluator.BASELINE
    try:
        evaluator.BASELINE=cfg['comparators']['V76']; evaluator.evaluate(output)
    finally:
        evaluator.BASELINE=old
    write_json(output/'evaluation_scope.json',dict(evaluation=record(output/'evaluation.json'),
        comparator='SEALED_V76_NOT_V42',per_fold_vocabulary=True,geometry_as_feature=False,
        full_query_population=False,old_evaluator_narrative_overridden=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=['run','encode','worker','evaluate'])
    p.add_argument('--output',type=Path,required=True); p.add_argument('--start',type=int); a=p.parse_args()
    if a.mode=='worker':
        if a.start not in range(0,128,8):
            p.error('Invalid worker start')
        worker(a.output,a.start)
    else:
        {'run':run,'encode':encode,'evaluate':evaluate}[a.mode](a.output)
