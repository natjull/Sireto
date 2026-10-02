"""New physical V46 serving attempt: frozen metric, fair adaptive traversal."""
import argparse
import json
from pathlib import Path
import time

from scripts import retrieval_v38_initial_top100 as v38
from scripts import retrieval_v42_indexed_metric as v42
from scripts import retrieval_v46_joint as v46
from src.xgb_matcher.hierarchical_retrieval import TantivyBackend
from src.xgb_matcher.retrieval_joint_v46 import FEATURES
from src.xgb_matcher.retrieval_joint_v47 import search,UncertifiedSearchBudget
from src.xgb_matcher.retrieval_index_pruning_v44 import search as comparator_search
from src.xgb_matcher.retrieval_v20_workbench import Budget,ROOT,record,verify,write_json

TRAINED=dict(path='/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v46_joint/training_20261002_a/manifest.json',
    sha256='523bce3a4abec664a481ace967b1faa4283dcd9c9d0c9c83c0715887ab7c1495')


def worker(output,start):
    launch=json.loads((output/'launch.json').read_text()); v38.verify_launch(launch)
    models=json.loads(verify(launch['models']).read_text()); old=json.loads(verify(launch['mining_models']).read_text())
    geo={r['query_id']:r for r in json.loads(verify(launch['geo_queries']).read_text())}; rows=[]
    with Budget(launch['resources']) as budget:
        tree,keys,parents=v46.load_tree(launch); backend=TantivyBackend(v38.INDEX)
        if backend.searcher.num_docs!=launch['index_documents']: raise ValueError('National count drift')
        for i,raw in enumerate(launch['queries'][start:start+8],start):
            model=models[str(raw['fold'])]; previous=old[str(raw['fold'])]
            if model['features']!=list(FEATURES) or model['heldout_fold']!=raw['fold'] or raw['fold'] in model['training_folds']:
                raise ValueError('OOF metric drift')
            row=dict(query_id=raw['query_id'],fold=raw['fold'],results={},seconds={})
            for mode in (v38.MODES if i%2==0 else tuple(reversed(v38.MODES))):
                t=time.monotonic()
                if mode=='primary':
                    try:
                        result,trace=search(backend,tree,keys,parents,v42.as_query(raw),geo[raw['query_id']],
                            model['weights'],max_seconds=launch['max_query_search_seconds'])
                    except UncertifiedSearchBudget as exc:
                        write_json(output/f'uncertified{start:03d}.json',dict(query_id=raw['query_id'],fold=raw['fold'],
                            trace=exc.trace,truth_opened=False,status='STOP_UNCERTIFIED_SEARCH_BUDGET'))
                        raise
                else: result,trace=comparator_search(backend,v42.as_query(raw),previous['weights'])
                row['results'][mode]=result; row['seconds'][mode]=time.monotonic()-t; row[mode+'_trace']=trace
            rows.append(row)
    v38.verify_launch(launch); write_json(output/f'batch{start:03d}.json',dict(rows=rows,resources=budget.report()))


def run(output):
    import os
    import subprocess
    import sys
    output.mkdir(parents=True,exist_ok=False); began=time.monotonic(); cfg=json.loads(v46.CONFIG.read_text())
    try:
        trained=json.loads(verify(TRAINED).read_text())
        if trained['status']!='SEALED_OOF_JOINT_INITIAL_METRICS': raise ValueError('Unsealed models')
        v38.verify_launch(trained)
        queries,qrefs=v42.input_rows(v38.load(v38.GEN,v38.GEN_SHA),4); v46.validate_queries(queries)
        gm=json.loads(verify(cfg['geo_evaluation_generation']).read_text()); gl=json.loads(verify(gm['launch']).read_text())
        if gm['status']!='SEALED_INITIAL_TOP100' or queries!=gl['queries']: raise ValueError('Query-only geo authority drift')
        models=json.loads(verify(trained['models']).read_text())
        if any(set(m['query_ids'])&{r['query_id'] for r in queries} for m in models.values()): raise ValueError('Training query reuse')
        sm=v46.spatial(cfg); verify(gl['geo_queries']); launch=v46.index_launch()
        for p in (Path(__file__),ROOT/'src/xgb_matcher/retrieval_joint_v47.py'): launch['implementation'][str(p)]=record(p)
        launch.update(queries=queries,query_sources=qrefs,models=trained['models'],model_manifest=TRAINED,
            spatial_files=sm['files'],spatial_sites=sm['compatible_coordinate_pairs'],spatial_manifest=cfg['spatial_manifest'],
            geo_queries=gl['geo_queries'],query_geo_generation=cfg['geo_evaluation_generation'],mining_models=cfg['mining_models'],
            resources=cfg['resources_worker'],max_query_search_seconds=cfg['max_query_search_seconds'],
            collector_cap=100,fixed_reservoir=None,truth_opened=False,development_panel_reused=True,
            physical_plan='FAIR_GEOMETRIC_GROWTH_PAGES_NOT_CORPUS_DEPTH_CAP',model_parameters_changed=False)
        write_json(output/'launch.json',launch); rows=[]; refs=[]
        for start in range(0,128,8):
            with (output/f'worker{start:03d}.log').open('x') as log:
                subprocess.run([sys.executable,'-u','-m','scripts.retrieval_v47_joint_fair','worker','--output',str(output),'--start',str(start)],
                    check=True,stdout=log,stderr=subprocess.STDOUT,timeout=min(90,max(1,900-(time.monotonic()-began))),
                    env={**os.environ,'RAYON_NUM_THREADS':'2','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','VECLIB_MAXIMUM_THREADS':'2'})
            path=output/f'batch{start:03d}.json'; batch=json.loads(path.read_text())
            if len(batch['rows'])!=8 or not batch['resources']['pass_resource_policy']: raise ValueError('Worker scope/resource drift')
            rows.extend(batch['rows']); refs.append(record(path)); print(json.dumps(dict(generated=len(rows))),flush=True)
            if len(rows)==16:
                cost=sum(sum(r['seconds'].values()) for r in rows)/16
                write_json(output/'cost_gate.json',dict(queries=16,mean_pair_seconds=cost,passed=cost<=4.,quality_opened=False))
                if cost>4.: raise ValueError('STOP_COST_PREFLIGHT')
        v38.verify_launch(launch)
        if len(rows)!=128 or {r['query_id'] for r in rows}!={r['query_id'] for r in queries}: raise ValueError('Serving denominator drift')
        write_json(output/'manifest.json',dict(status='SEALED_INITIAL_TOP100',queries=128,batches=refs,launch=record(output/'launch.json'),
            seconds=time.monotonic()-began,model_used=True,truth_opened=False,full_lexical_corpus=True,full_spatial_corpus=False,
            full_query_population=False,ready_for_fold1=False,certified_initial_score=True,metric_family=cfg['metric_family'],
            physical_plan=launch['physical_plan']))
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),seconds=time.monotonic()-began,truth_opened=False,ready_for_fold1=False)); raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=['run','worker','evaluate'])
    p.add_argument('--output',required=True,type=Path); p.add_argument('--start',type=int); a=p.parse_args()
    if a.mode=='run': run(a.output)
    elif a.mode=='evaluate': v46.evaluate(a.output)
    else:
        if a.start not in range(0,128,8): p.error('Invalid start')
        worker(a.output,a.start)
