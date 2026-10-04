"""Fit initial-index11 on complete actual TRAIN64 scores; no national recall."""
import argparse
import json
import os
from pathlib import Path
import sys
import pyarrow.parquet as pq
from scripts import retrieval_v232_actual_train_features as features
from scripts import retrieval_v228_fit_national_place as previous
from src.xgb_matcher.retrieval_topk_initial_v239 import fit,controls
from src.xgb_matcher.retrieval_structural_channels_v210 import SourceFiles
from src.xgb_matcher.retrieval_memory_policy_v214 import Budget,LIMITS,quiet,validate_report,parent_exit
from src.xgb_matcher.retrieval_supervisor_v214 import supervise
from src.xgb_matcher.retrieval_v20_workbench import ROOT,record,verify,write_json

OUTPUT=Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v240_actual_initial_training')
FEATURES=Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v232_actual_train_features/run_20261004_a')
PRIOR=dict(path='/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v228_national_place_training/run_20261004_a/manifest.json',
           sha256='05966487d7d26b2438f6d5072917a53169691bca402df4a481ceb678921ec44c',size_bytes=1191)
STATUS='SEALED_V240_TRAIN_ONLY_ACTUAL_NATIVE_INITIAL_INDEX11_TOP100_HINGE'


def implementation():
    refs={**features.implementation(),**previous.implementation()}
    for p in (Path(__file__),ROOT/'src/xgb_matcher/retrieval_topk_initial_v239.py',
        ROOT/'docs/retrieval_v240_topk_initial_contract.md'):
        refs[str(p)]=record(p)
    return refs


def worker(root):
    l=json.loads((root/'launch.json').read_text())
    for ref in l['implementation'].values():verify(ref)
    with Budget(LIMITS) as budget:
        def guard():
            if budget.failure:raise RuntimeError(budget.failure)
        files=SourceFiles(guard);features.sealed(FEATURES)
        fm=json.loads(files.file(l['training_features']).read_text())
        dataset=json.loads(files.file(fm['outputs']).read_text());rows=dataset['rows']
        a=json.loads(files.file(json.loads((features.MINING/'manifest.json').read_text())['training_authority']).read_text())
        old=json.loads(files.file(a['source_launch']).read_text())
        ml=json.loads((features.MINING/'launch.json').read_text());original=json.loads(files.file(ml['models']).read_text())
        side=json.loads(files.file(previous.SIDE).read_text())
        if side['never_used_in_retrieval'] is not True or side['queries']!=23571:raise ValueError('Sealed official supervised equivalence')
        ids=[r['query_id'] for r in rows];scope={r['query_id']:r['fold'] for r in rows};acceptable={qid:set() for qid in ids}
        allowed=pq.read_table(files.file(side['files']['acceptable_operational.parquet']),
            columns=['query_id','fold','siret','exact_truth','same_site_evidence'],
            filters=[('fold','in',[2,3,4]),('query_id','in',ids)],use_threads=False).to_pylist()
        for r in allowed:
            if r['query_id'] not in scope or r['fold']!=scope[r['query_id']] or not (r['exact_truth'] or r['same_site_evidence']):
                raise ValueError('Only natural exact or officially equivalent TRAIN references')
            acceptable[r['query_id']].add(r['siret'])
        if l['prior_training']!=PRIOR:raise ValueError('Pinned actual228 prior source')
        prior_path=files.file(PRIOR)
        priors,pm=previous.sealed(prior_path.parent)
        files.file(pm['models'])
        models={str(f):fit(rows,old['registry'],original,acceptable,f,priors) for f in (2,3,4)}
        features.mining.base.validate_models(models)
        # Model bytes frozen BEFORE observing the OOF TRAIN comparisons;
        # neither a family selection nor fit is performed after controls.
        write_json(root/'worker/models.json',models);model_ref=record(root/'worker/models.json')
        control=controls(rows,old['registry'],original,models,acceptable,priors)
        if model_ref!=record(root/'worker/models.json'):raise ValueError('No parameter changes after TRAIN controls')
        files.verify_after();features.sealed(FEATURES)
        write_json(root/'worker/report.json',dict(folds={f:dict(weights=m['weights'],**m['native_training']) for f,m in models.items()},
            train_controls=control,source_epochs=list(files.entries.values()),truth_scope='TRAINER_ONLY',recall_measured=False,
            ready_for_fold1=False,fold0_opened=False,fold1_opened=False,downstream_models_changed=False))
    validate_report(budget.report())
    write_json(root/'worker/manifest.json',dict(launch=record(root/'launch.json'),resources=budget.report(),
        models=model_ref,report=record(root/'worker/report.json'),preflight=record(root/'worker/preflight.json')))


def sealed(root,serving=False):
    """Serving checks model-only producer authority, never opens TRAIN targets."""
    root=Path(root);parent_exit(root/'worker')
    if (root/'status.json').exists():raise ValueError('No STOP model reuse')
    m=json.loads((root/'manifest.json').read_text());l=json.loads(verify(m['launch']).read_text())
    for ref in l['implementation'].values():verify(ref)
    w=json.loads(verify(m['worker']).read_text());validate_report(w['resources'])
    gate=json.loads(verify(w['preflight']).read_text())
    if (m['status']!=STATUS or m['launch']!=record(root/'launch.json') or m['worker']!=record(root/'worker/manifest.json')
            or w['launch']!=m['launch'] or m['models']!=w['models'] or m['models']!=record(root/'worker/models.json')
            or m['report']!=w['report'] or gate['pass_system_quiet_policy'] is not True or gate['quiet_seconds']<30
            or l['training_features']!=record(FEATURES/'manifest.json') or l['prior_training']!=PRIOR):
        raise ValueError('Actual successful same full actual-score TRAIN producer')
    report=json.loads(verify(w['report']).read_text());features.mining.base.check_epochs(report['source_epochs'])
    if report['truth_scope']!='TRAINER_ONLY' or any(report[k] is not False for k in ('recall_measured','ready_for_fold1','fold0_opened','fold1_opened','downstream_models_changed')):
        raise ValueError('Initial supervised learning is not product evaluation')
    verify(PRIOR)
    if not serving:features.sealed(FEATURES)
    return json.loads(verify(m['models']).read_text()),m


def run(root):
    features.sealed(FEATURES)
    if root.exists() or not root.resolve().is_relative_to(OUTPUT):raise ValueError('Fresh dedicated TRAIN-only233 root')
    root.mkdir(parents=True);dest=root/'worker';dest.mkdir()
    write_json(root/'launch.json',dict(implementation=implementation(),training_features=record(FEATURES/'manifest.json'),prior_training=PRIOR))
    try:
        write_json(dest/'preflight.json',quiet(30,50))
        with (dest/'worker.log').open('x') as log:
            s=supervise([sys.executable,'-u','-m','scripts.retrieval_v240_fit_topk_initial','worker','--output',str(root)],
                log,{**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','RAYON_NUM_THREADS':'1'},dest/'supervisor_stop.json')
        write_json(dest/'supervision.json',s);parent_exit(dest)
        w=json.loads((dest/'manifest.json').read_text());validate_report(w['resources'])
        write_json(root/'manifest.json',dict(status=STATUS,launch=record(root/'launch.json'),worker=record(dest/'manifest.json'),
            models=w['models'],report=w['report'],truth_scope='TRAINER_ONLY',recall_measured=False,ready_for_fold1=False))
        models,_=sealed(root);print(json.dumps({f:m['weights'] for f,m in models.items()}),flush=True)
    except BaseException as exc:
        write_json(root/'status.json',dict(status='STOP',reason=str(exc),recall_measured=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('run','worker'))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    run(a.output) if a.mode=='run' else worker(a.output)
