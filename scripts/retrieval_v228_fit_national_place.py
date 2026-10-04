"""Supervised TRAIN only after complete V227; no pilot labels or recall."""
import argparse
import json
import os
from pathlib import Path
import sys
import pyarrow.parquet as pq
from scripts import retrieval_v227_national_training_negatives as mining
from src.xgb_matcher.retrieval_national_place_v228 import fit_place
from src.xgb_matcher.retrieval_structural_channels_v210 import SourceFiles
from src.xgb_matcher.retrieval_memory_policy_v214 import Budget,LIMITS,quiet,validate_report,parent_exit
from src.xgb_matcher.retrieval_supervisor_v214 import supervise
from src.xgb_matcher.retrieval_v20_workbench import ROOT,record,verify,write_json

OUTPUT=Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v228_national_place_training')
MINING=Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v227_national_training_negatives/run_20261004_a')
SIDE=dict(path='/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v34_fast_resume/full_job/evaluation_resume_20261001/label_side_manifest.json',
          sha256='42e15d604624364a59690fffa2a97c50552ac51a61789ed3347e16401bd6bf13')
STATUS='SEALED_V228_TRAIN_ONLY_OOF_INITIAL_INDEX_PLACE_COEFFICIENT'


def implementation():
    return {str(p):record(p) for p in (Path(__file__),ROOT/'src/xgb_matcher/retrieval_national_place_v228.py',
        ROOT/'docs/retrieval_v227_national_training_contract.md',Path(mining.__file__),
        ROOT/'src/xgb_matcher/retrieval_memory_policy_v214.py',ROOT/'src/xgb_matcher/retrieval_supervisor_v214.py')}


def worker(root):
    l=json.loads((root/'launch.json').read_text())
    for ref in l['implementation'].values():verify(ref)
    with Budget(LIMITS) as budget:
        def guard():
            if budget.failure:raise RuntimeError(budget.failure)
        files=SourceFiles(guard); value=mining.sealed(MINING)
        if l['mining']!=record(MINING/'manifest.json'):raise ValueError('Same full mining prerequisite')
        a=json.loads(files.file(json.loads((MINING/'manifest.json').read_text())['training_authority']).read_text())
        proof=json.loads(files.file(a['source_proof']).read_text()); old=json.loads(files.file(a['source_launch']).read_text())
        if a['source_launch']!=proof['launch'] or old['truth_scope']!='TRAINER_ONLY':raise ValueError('Original supervised TRAIN authority')
        scenes=[s for ref in proof['features'] for s in json.loads(files.file(ref).read_text())['scenes']]
        raw=json.loads(files.file(a['raw_queries']).read_text())
        if mining.project(old['training_rows'],[])!=raw:raise ValueError('Same label-blind first64')
        launch=json.loads((MINING/'launch.json').read_text()); original=json.loads(files.file(launch['models']).read_text())
        side=json.loads(files.file(SIDE).read_text())
        if side['never_used_in_retrieval'] is not True or side['queries']!=23571:raise ValueError('Sealed official equivalence sidecar')
        ref=side['files']['acceptable_operational.parquet']; path=files.file(ref)
        ids=[r['query_id'] for r in raw]; scope={r['query_id']:r['fold'] for r in raw}
        rows=pq.read_table(path,columns=['query_id','fold','siret','exact_truth','same_site_evidence'],
            filters=[('fold','in',[2,3,4]),('query_id','in',ids)],use_threads=False).to_pylist()
        acceptable={qid:set() for qid in ids}
        for row in rows:
            if row['query_id'] not in scope or row['fold']!=scope[row['query_id']] or not (row['exact_truth'] or row['same_site_evidence']):
                raise ValueError('Actual TRAIN-only official acceptable references')
            acceptable[row['query_id']].add(row['siret'])
        source={r['query_id']:r for r in old['training_rows']}
        for qid in ids:
            if source[qid]['positive_siret'] not in acceptable[qid]:raise ValueError('Exact positive must not be a negative')
        models={str(f):fit_place(scenes,value['rows'],old['registry'],original,acceptable,f) for f in (2,3,4)}
        mining.base.validate_models(models);files.verify_after(); mining.sealed(MINING)
        write_json(root/'worker/models.json',models)
        write_json(root/'worker/report.json',dict(folds={f:dict(place_weight=m['weights'][4],**m['place_training']) for f,m in models.items()},
            source_epochs=list(files.entries.values()),truth_scope='TRAINER_ONLY',recall_measured=False,fold0_opened=False,fold1_opened=False,
            downstream_models_changed=False,ready_for_fold1=False,supervised_training_sidecar_not_serving_feature=True))
    validate_report(budget.report())
    write_json(root/'worker/manifest.json',dict(launch=record(root/'launch.json'),resources=budget.report(),
        models=record(root/'worker/models.json'),report=record(root/'worker/report.json')))


def sealed(root):
    root=Path(root);parent_exit(root/'worker')
    if (root/'status.json').exists():raise ValueError('No STOP model reuse')
    m=json.loads((root/'manifest.json').read_text());l=json.loads(verify(m['launch']).read_text())
    for ref in l['implementation'].values():verify(ref)
    w=json.loads(verify(m['worker']).read_text());validate_report(w['resources'])
    gate=json.loads((root/'worker/preflight.json').read_text())
    if (m['status']!=STATUS or w['launch']!=record(root/'launch.json') or m['models']!=w['models']
            or m['launch']!=record(root/'launch.json') or m['worker']!=record(root/'worker/manifest.json')
            or m['models']!=record(root/'worker/models.json') or m['report']!=w['report']
            or gate['pass_system_quiet_policy'] is not True or gate['quiet_seconds']<30):
        raise ValueError('Actual successful same TRAIN-only model producer')
    mining.sealed(MINING)
    report=json.loads(verify(w['report']).read_text());mining.base.check_epochs(report['source_epochs'])
    if (report['truth_scope']!='TRAINER_ONLY' or report['recall_measured'] is not False
            or report['fold0_opened'] is not False or report['fold1_opened'] is not False):
        raise ValueError('Supervised training scope is not pilot evaluation')
    return json.loads(verify(m['models']).read_text()),m


def run(root):
    mining.sealed(MINING)
    if root.exists() or not root.resolve().is_relative_to(OUTPUT):raise ValueError('Fresh TRAIN-only root')
    root.mkdir(parents=True);dest=root/'worker';dest.mkdir()
    write_json(root/'launch.json',dict(implementation=implementation(),mining=record(MINING/'manifest.json')))
    try:
        write_json(dest/'preflight.json',quiet(30,50))
        with (dest/'worker.log').open('x') as log:
            s=supervise([sys.executable,'-u','-m','scripts.retrieval_v228_fit_national_place','worker','--output',str(root)],
                log,{**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','RAYON_NUM_THREADS':'1'},dest/'supervisor_stop.json')
        write_json(dest/'supervision.json',s);parent_exit(dest)
        w=json.loads((dest/'manifest.json').read_text());validate_report(w['resources'])
        write_json(root/'manifest.json',dict(status=STATUS,launch=record(root/'launch.json'),worker=record(dest/'manifest.json'),
            models=w['models'],report=w['report'],truth_scope='TRAINER_ONLY',recall_measured=False,ready_for_fold1=False))
        models,_=sealed(root);print(json.dumps({f:m['weights'][4] for f,m in models.items()}),flush=True)
    except BaseException as exc:
        write_json(root/'status.json',dict(status='STOP',reason=str(exc),recall_measured=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('run','worker'))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    run(a.output) if a.mode=='run' else worker(a.output)
