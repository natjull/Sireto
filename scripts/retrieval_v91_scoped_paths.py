"""Blind national initial100 with a preregistered scoped name/site path metric."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts import retrieval_v38_initial_top100 as v38
from scripts import retrieval_v42_indexed_metric as v42
from scripts import retrieval_v46_joint as v46
from scripts import retrieval_v49_primary_probe as evaluator
from scripts.retrieval_v56_exact_fields import index_authority
from scripts.retrieval_v78_safe_sparse_initial import validate_models
from src.xgb_matcher.hierarchical_retrieval import TantivyBackend
from src.xgb_matcher.retrieval_name_postings_v86 import WordPostings
from src.xgb_matcher.retrieval_scoped_paths_v91 import OwnerFacts,search
from src.xgb_matcher.retrieval_scoped_paths_v91 import SuccessionCSR
from src.xgb_matcher.retrieval_v20_workbench import Budget,ROOT,record,verify,write_json

CONFIG=ROOT/'config/retrieval_v91_scoped_paths.json'


def authority():
    a=index_authority()
    for path in (Path(__file__),CONFIG,ROOT/'docs/retrieval_v91_scoped_paths_probe.md',
                 ROOT/'src/xgb_matcher/retrieval_scoped_paths_v91.py',
                 ROOT/'src/xgb_matcher/retrieval_ranked_path_v83.py',ROOT/'src/xgb_matcher/retrieval_ranked_path_v91.py',
                 ROOT/'src/xgb_matcher/retrieval_scoped_paths_v87.py',ROOT/'src/xgb_matcher/retrieval_scoped_paths_v90.py',ROOT/'docs/retrieval_v87_scoped_paths_probe.md',
                 ROOT/'src/xgb_matcher/retrieval_name_postings_v86.py',
                 ROOT/'src/xgb_matcher/retrieval_name_facts_v85.py',
                 ROOT/'src/xgb_matcher/retrieval_succession_csr_v84.py',
                 ROOT/'scripts/retrieval_v78_safe_sparse_initial.py',ROOT/'scripts/retrieval_v46_joint.py',
                 ROOT/'src/xgb_matcher/retrieval_sparse_native_v60.py',ROOT/'src/xgb_matcher/retrieval_sparse_query_v59.py'):
        a['implementation'][str(path)]=record(path)
    return a


def official_sources(cfg):
    words=json.loads(verify(cfg['word_manifest']).read_text()); graph=json.loads(verify(cfg['graph_manifest']).read_text())
    for m,status in ((words,'SEALED_COMPLETE_SCOPED_NAME_WORD_ACCESS'),(graph,'SEALED_EXHAUSTIVE_SUCCESSION_CSR')):
        if m['status']!=status or m['truth_opened'] or m['crm_opened']: raise ValueError('Source-only complete index required')
        launch=json.loads(verify(m['launch']).read_text())
        for ref in launch['implementation'].values(): verify(ref)
        for ref in m['artifacts'].values(): verify(ref)
    if words['postings']['facts']!=14162541 or graph['stats']['endpoint_sites']!=12531976:
        raise ValueError('National source cardinality drift')
    # Full original observation catalog and inverse mapping remain referenced;
    # this metric does not assert temporal co-validity or consume those dates.
    verify(words['facts_source'])
    return words,graph


def worker(output,start):
    launch=json.loads((output/'launch.json').read_text()); v38.verify_launch(launch)
    models=json.loads(verify(launch['models']).read_text()); validate_models(models)
    expansions=json.loads(verify(launch['expansions']).read_text())
    if expansions['truth_opened']: raise ValueError('Query-only expansions required')
    erows=expansions['rows']; queries=launch['queries']
    if [(r['query_id'],r['fold']) for r in erows]!=[(r['query_id'],r['fold']) for r in queries]:
        raise ValueError('Expansion alignment drift')
    terms={r['query_id']:r['terms'] for r in erows}
    geos=json.loads(verify(launch['geo_queries']).read_text())
    if {r['query_id'] for r in geos}!={r['query_id'] for r in queries}: raise ValueError('Query geometry denominator drift')
    geometry={r['query_id']:r for r in geos}; rows=[]; raw=None; budget=None
    try:
        with Budget(launch['resources']) as budget:
            words,gm=official_sources(launch)
            index=WordPostings(words['postings']); owners=OwnerFacts(Path(launch['word_manifest']['path']).parent)
            graph=SuccessionCSR(Path(launch['graph_manifest']['path']).parent)
            tree,keys,parents=v46.load_tree(launch); base=TantivyBackend(v38.INDEX)
            if base.searcher.num_docs!=launch['index_documents']: raise ValueError('National native cardinality drift')
            for raw in queries[start:start+8]:
                m=models[str(raw['fold'])]; began=time.monotonic()
                result,trace=search(base,tree,keys,parents,index,owners,graph,v42.as_query(raw),geometry[raw['query_id']],
                                    m['weights'],terms[raw['query_id']],m['alpha'],path_weight=launch['path_weight'],
                                    max_seconds=launch['max_query_search_seconds'])
                elapsed=time.monotonic()-began
                if elapsed>launch['max_query_search_seconds']: raise ValueError('STOP_QUERY_COST')
                rows.append(dict(query_id=raw['query_id'],fold=raw['fold'],results=dict(primary=result),
                                 seconds=dict(primary=elapsed),primary_trace=trace))
                print(json.dumps(dict(query_id=raw['query_id'],seconds=elapsed,trace=trace)),flush=True)
        v38.verify_launch(launch); official_sources(launch)
        write_json(output/f'batch{start:03d}.json',dict(rows=rows,resources=budget.report()))
    except BaseException as exc:
        write_json(output/f'worker_stop{start:03d}.json',dict(status='STOP',reason=str(exc),
            query_id=raw['query_id'] if raw else None,certified_queries_completed=len(rows),
            trace=getattr(exc,'trace',None),resources=budget.report() if budget is not None else None,
            truth_opened=False,recall_measured=False))
        raise


def run(output):
    output.mkdir(parents=True,exist_ok=False); began=time.monotonic(); cfg=json.loads(CONFIG.read_text())
    try:
        old=json.loads(verify(cfg['query_only_launch']).read_text()); v38.verify_launch(old)
        # This source contains query-only model/encoder authority, no candidates.
        queries=old['queries']; v46.validate_queries(queries)
        models=json.loads(verify(old['models']).read_text()); validate_models(models)
        if any(set(m['query_ids'])&{q['query_id'] for q in queries} for m in models.values()):
            raise ValueError('Training/evaluation query overlap')
        verify(old['expansions']); verify(old['bundle'])
        spatial_cfg=json.loads(v46.CONFIG.read_text()); sm=v46.spatial(spatial_cfg)
        geo_generation=json.loads(verify(spatial_cfg['geo_evaluation_generation']).read_text())
        gl=json.loads(verify(geo_generation['launch']).read_text())
        if geo_generation['status']!='SEALED_INITIAL_TOP100' or gl['queries']!=queries:
            raise ValueError('Query-only BAN authority drift')
        verify(gl['geo_queries']); official_sources(cfg)
        launch=dict(**authority(),queries=queries,query_sources=old['query_sources'],models=old['models'],
                    model_manifest=old['model_manifest'],expansions=old['expansions'],bundle=old['bundle'],
                    spatial_files=sm['files'],spatial_sites=sm['compatible_coordinate_pairs'],
                    geo_queries=gl['geo_queries'],word_manifest=cfg['word_manifest'],graph_manifest=cfg['graph_manifest'],
                    resources=cfg['resources'],path_weight=cfg['path_weight'],max_query_search_seconds=cfg['max_query_search_seconds'],
                    truth_opened=False,old_candidates_opened=False,collector_cap=100,fixed_reservoir=None,
                    development_panel_reused=True,geometry_as_feature=True)
        write_json(output/'launch.json',launch); rows=[]; batches=[]
        for start in range(0,128,8):
            with (output/f'worker{start:03d}.log').open('x') as log:
                subprocess.run([sys.executable,'-u','-m','scripts.retrieval_v91_scoped_paths','worker','--output',str(output),'--start',str(start)],
                    stdout=log,stderr=subprocess.STDOUT,check=True,
                    timeout=min(90,max(1,cfg['controller_seconds']-(time.monotonic()-began))),
                    env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','RAYON_NUM_THREADS':'2','VECLIB_MAXIMUM_THREADS':'2'})
            file=output/f'batch{start:03d}.json'; batch=json.loads(file.read_text())
            if (len(batch['rows'])!=8 or not batch['resources']['pass_resource_policy'] or
                [(r['query_id'],r['fold']) for r in batch['rows']]!=[(q['query_id'],q['fold']) for q in queries[start:start+8]] or
                any(not r['primary_trace']['certified'] for r in batch['rows'])):
                raise ValueError('Blind batch scope/resource/certificate drift')
            rows.extend(batch['rows']); batches.append(record(file)); print(json.dumps(dict(generated=len(rows))),flush=True)
            if len(rows)==16:
                mean=sum(r['seconds']['primary'] for r in rows)/16
                write_json(output/'cost_gate.json',dict(mean_seconds=mean,passed=mean<=cfg['primary_mean16_seconds'],truth_opened=False))
                if mean>cfg['primary_mean16_seconds']: raise ValueError('STOP_COST_PREFLIGHT')
        v38.verify_launch(launch); official_sources(cfg)
        write_json(output/'manifest.json',dict(status='SEALED_INITIAL_TOP100',queries=128,batches=batches,launch=record(output/'launch.json'),
            seconds=time.monotonic()-began,truth_opened=False,model_used=True,old_candidates_opened=False,full_query_population=False,
            full_lexical_corpus=True,geometry_as_feature=True,ready_for_fold1=False,metric_family=cfg['metric_family']))
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),seconds=time.monotonic()-began,
            truth_opened=False,ready_for_fold1=False,recall_measured=False)); raise


def evaluate(output):
    cfg=json.loads(CONFIG.read_text()); prior=evaluator.BASELINE
    try:
        evaluator.BASELINE=cfg['comparator']; evaluator.evaluate(output)
    finally: evaluator.BASELINE=prior
    write_json(output/'evaluation_scope.json',dict(evaluation=record(output/'evaluation.json'),comparator='V78_AFTER_SEALED128',
        old_evaluator_narrative_overridden=True,scope='INITIAL100_NATIVE_PLUS_SCOPED_ONE_HOP_NO_POOL',full_query_population=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=['run','worker','evaluate'])
    p.add_argument('--output',required=True,type=Path); p.add_argument('--start',type=int); args=p.parse_args()
    if args.mode=='worker':
        if args.start not in range(0,128,8): p.error('Invalid worker start')
        worker(args.output,args.start)
    else: (run if args.mode=='run' else evaluate)(args.output)
