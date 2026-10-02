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
from src.xgb_matcher.retrieval_native_v57 import LEXICAL,fit,search
from src.xgb_matcher.retrieval_v20_workbench import Budget,ROOT,record,verify,write_json
CONFIG=ROOT/'config/retrieval_v57_native_proof.json'

def authority():
    result=index_authority()
    for p in (Path(__file__),CONFIG,ROOT/'src/xgb_matcher/retrieval_native_v57.py'):
        result['implementation'][str(p)]=record(p)
    return result

def train(output):
    cfg=json.loads(CONFIG.read_text()); output.mkdir(parents=True,exist_ok=False); began=time.monotonic()
    try:
        source=json.loads(verify(cfg['proof_source']).read_text()); v38.verify_launch(source)
        sl=json.loads(verify(source['launch']).read_text()); v38.verify_launch(sl)
        if source['status']!='SEALED_OOF_INITIAL_PROOF_MODELS' or source['training_queries']!=512 or source['recall_measured']:
            raise ValueError('Training proof authority drift')
        cached=[r for ref in source['features'] for r in json.loads(verify(ref).read_text())['scenes']]
        scenes=[dict(query_id=r['query_id'],fold=r['fold'],positive_siren=r['positive_siren'],
            positive_features=r['positive_features'][:len(LEXICAL)],
            negatives=[dict(siren=n['siren'],features=n['features'][:len(LEXICAL)]) for n in r['negatives']]) for r in cached]
        if len(scenes)!=512 or len({r['query_id'] for r in scenes})!=512: raise ValueError('Training denominator drift')
        launch=dict(**authority(),source=cfg['proof_source'],cached_features=source['features'],registry=sl['registry'],resources=cfg['resources'])
        write_json(output/'launch.json',launch)
        with Budget(cfg['resources']) as budget:
            models={str(f):fit(scenes,f,sl['registry'],cfg['ridge']) for f in (2,3,4)}
        v38.verify_launch(launch); write_json(output/'models.json',models)
        write_json(output/'manifest.json',dict(**authority(),status='SEALED_OOF_NATIVE_PROOF_MODELS',
            models=record(output/'models.json'),launch=record(output/'launch.json'),seconds=time.monotonic()-began,
            training_queries=512,resources=budget.report(),recall_measured=False,
            fold0_opened=False,fold1_opened=False,ready_for_fold1=False,geometry_as_feature=False))
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),recall_measured=False)); raise

def worker(output,start):
    launch=json.loads((output/'launch.json').read_text()); v38.verify_launch(launch)
    models=json.loads(verify(launch['models']).read_text())
    with Budget(launch['resources']) as budget:
        backend=TantivyBackend(v38.INDEX); rows=[]
        if backend.searcher.num_docs!=launch['index_documents']: raise ValueError('Official cardinality drift')
        for raw in launch['queries'][start:start+8]:
            m=models[str(raw['fold'])]
            if m['features']!=list(LEXICAL) or m['heldout_fold']!=raw['fold'] or raw['fold'] in m['training_folds']:
                raise ValueError('Serving OOF proof drift')
            t=time.monotonic(); result,trace=search(backend,v42.as_query(raw),m['weights']); elapsed=time.monotonic()-t
            if elapsed>launch['max_query_search_seconds']: raise ValueError('STOP_QUERY_COST')
            rows.append(dict(query_id=raw['query_id'],fold=raw['fold'],results=dict(primary=result),seconds=dict(primary=elapsed),primary_trace=trace))
    v38.verify_launch(launch); write_json(output/f'batch{start:03d}.json',dict(rows=rows,resources=budget.report()))

def run(output,model_manifest):
    cfg=json.loads(CONFIG.read_text()); output.mkdir(parents=True,exist_ok=False); began=time.monotonic()
    try:
        trained=json.loads(model_manifest.read_text()); v38.verify_launch(trained)
        if trained['status']!='SEALED_OOF_NATIVE_PROOF_MODELS': raise ValueError('Unsealed native model')
        models=json.loads(verify(trained['models']).read_text()); queries,qrefs=v42.input_rows(v38.load(v38.GEN,v38.GEN_SHA),4)
        if any(set(m['query_ids'])&{r['query_id'] for r in queries} for m in models.values()): raise ValueError('Fit/evaluation query reuse')
        launch=dict(**authority(),models=trained['models'],model_manifest=record(model_manifest),queries=queries,query_sources=qrefs,
            resources=cfg['resources'],max_query_search_seconds=cfg['max_query_search_seconds'],
            collector_cap=100,fixed_reservoir=None,truth_opened=False,geometry_as_feature=False,development_panel_reused=True,
            comparator_scope='SEPARATE_EVALUATOR_SEALED_V42_NO_GENERATOR_READ')
        write_json(output/'launch.json',launch); rows=[]; refs=[]
        for start in range(0,128,8):
            with (output/f'worker{start:03d}.log').open('x') as log:
                subprocess.run([sys.executable,'-u','-m','scripts.retrieval_v57_native_proof','worker','--output',str(output),'--start',str(start)],
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
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=['train','worker','run','evaluate'])
    p.add_argument('--output',required=True,type=Path); p.add_argument('--models',type=Path); p.add_argument('--start',type=int); a=p.parse_args()
    if a.mode=='train': train(a.output)
    elif a.mode=='evaluate': evaluate(a.output)
    elif a.mode=='run':
        if a.models is None: p.error('--models required')
        run(a.output,a.models)
    else:
        if a.start not in range(0,128,8): p.error('Invalid batch')
        worker(a.output,a.start)
