"""Cost-only repro of the failed V61 worker; no labels/candidate artifact."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from scripts import retrieval_v38_initial_top100 as v38
from scripts import retrieval_v42_indexed_metric as v42
from src.xgb_matcher.hierarchical_retrieval import TantivyBackend
from src.xgb_matcher.retrieval_proof_v56 import signal
from src.xgb_matcher.retrieval_sparse_native_v60 import search as nested
from src.xgb_matcher.retrieval_sparse_flat_v61 import search as flat
from src.xgb_matcher.retrieval_v20_workbench import Budget,record,verify,write_json
SOURCE=Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v61_sparse_flat/pilot_20261002_a')

def worker(output,plan):
    launch=json.loads((SOURCE/'launch.json').read_text()); v38.verify_launch(launch)
    models=json.loads(verify(launch['models']).read_text()); expanded={r['query_id']:r for r in json.loads(verify(launch['expansions']).read_text())['rows']}
    budget=Budget(launch['resources']); rows=[]; failure=None
    try:
        with budget:
            backend=TantivyBackend(v38.INDEX)
            for raw in launch['queries'][96:104]:
                m=models[str(raw['fold'])]; terms=expanded[raw['query_id']]['terms']
                stats={field:{t:backend.searcher.doc_freq(field,t) for t,_ in terms} for field in v38.NAMES}
                write_json(output/(plan+'_progress.json'),dict(query_id=raw['query_id'],completed=len(rows),semantic_document_frequencies=stats,
                    seconds=time.monotonic()-budget.start,no_labels=True,no_candidates_written=True))
                print(json.dumps(dict(query_id=raw['query_id'],started=True)),flush=True)
                before=time.monotonic(); result,_=(nested if plan=='nested' else flat)(backend,v42.as_query(raw),m['weights'],terms,m['alpha'])
                rows.append(dict(query_id=raw['query_id'],seconds=time.monotonic()-before,returned_count=len(result)))
                print(json.dumps(rows[-1]),flush=True)
    except BaseException as exc:
        failure=str(exc)
    finally:
        report=budget.report() if hasattr(budget,'elapsed') else None
        violations=None if report is None else dict(rss=report['peak_rss_bytes']>launch['resources']['max_rss_bytes'],
            swap=report['swapout_pages_delta']>launch['resources']['max_swapout_pages_delta'],
            time=report['wall_seconds']>launch['resources']['max_seconds'])
        write_json(output/(plan+'.json'),dict(status='STOP' if failure else 'SEALED_COST_PROFILE',plan=plan,rows=rows,
            reason=failure,resources=report,resource_violations=violations,no_candidates_written=True,no_labels=True,recall_measured=False))
    if failure: raise RuntimeError(failure)

def run(output):
    output.mkdir(parents=True,exist_ok=False); source_status=json.loads((SOURCE/'status.json').read_text())
    if source_status['status']!='STOP' or source_status['truth_opened']: raise ValueError('Terminal blind source required')
    refs={}
    for plan in ('nested','flat'):
        with (output/(plan+'.log')).open('x') as log:
            try:
                p=subprocess.run([sys.executable,'-u','-m','scripts.profile_retrieval_v62_sparse_cost','worker','--plan',plan,'--output',str(output)],
                    stdout=log,stderr=subprocess.STDOUT,timeout=90,
                    env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','VECLIB_MAXIMUM_THREADS':'2','RAYON_NUM_THREADS':'2'})
                terminal=dict(exit_code=p.returncode)
            except subprocess.TimeoutExpired:
                terminal=dict(external_timeout=True)
        part=output/(plan+'.json')
        refs[plan]=record(part) if part.exists() else terminal
    write_json(output/'manifest.json',dict(status='SEALED_COST_ONLY_DIAGNOSTICS',profiles=refs,source_launch=record(SOURCE/'launch.json'),
        implementation=record(Path(__file__)),no_candidate_artifact=True,no_labels=True,recall_measured=False,
        selection='FAILED_WORKER96_COST_ONLY_NOT_LABEL_SELECTED',fold0_opened=False,fold1_opened=False,ready_for_fold1=False))
    print(json.dumps({plan:json.loads((output/(plan+'.json')).read_text()) if (output/(plan+'.json')).exists() else refs[plan] for plan in refs}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=['run','worker']); p.add_argument('--output',required=True,type=Path)
    p.add_argument('--plan',choices=['nested','flat']); a=p.parse_args()
    if a.mode=='run': run(a.output)
    else:
        if a.plan is None: p.error('--plan required')
        worker(a.output,a.plan)
