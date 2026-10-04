"""Post-V225 bounded audit of frozen OOF training negatives. No fit or recall."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys

from scripts import evaluate_retrieval_v225_postseal as evaluation
from src.xgb_matcher.retrieval_structural_channels_v210 import SourceFiles
from src.xgb_matcher.retrieval_memory_policy_v214 import Budget,LIMITS,quiet,parent_exit,validate_report
from src.xgb_matcher.retrieval_supervisor_v214 import supervise
from src.xgb_matcher.retrieval_v20_workbench import ROOT,record,verify,write_json

EVALUATION=Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v225_postseal_evaluation/run_20261004_a')
OUTPUT=Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v226_training_geography')
SOURCE_CONFIG=ROOT/'config/retrieval_v77_safe_sparse.json'


def summarize(scenes,registry,models,controls):
    if len(scenes)!=512 or len({s['query_id'] for s in scenes})!=512 or set(registry)!={'2','3','4'}:
        raise ValueError('Original complete512 safe training source required')
    byid={s['query_id']:s for s in scenes}
    if set(byid)&{r['query_id'] for r in controls}:
        raise ValueError('Training source must not reuse current128 diagnostic labels')
    def place(vector):
        if len(vector)!=19 or any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in vector) or vector[4] not in (0.,1.):
            raise ValueError('Actual original19 proof features and binary place required')
        return int(vector[4])
    for s in scenes:
        if s['fold'] not in (2,3,4) or s['positive_siren'] not in registry[str(s['fold'])]:
            raise ValueError('Safe official training membership required')
    result={}
    for held in (2,3,4):
        m=models[str(held)];protected=set(registry[str(held)]);pairs=Counter();queries=0;zero_contrast=0
        if m['heldout_fold']!=held or not set(m['query_ids'])<=set(byid):
            raise ValueError('Same original OOF model training queries required')
        for qid in m['query_ids']:
            s=byid[qid]
            if s['fold']==held or s['positive_siren'] in protected:
                raise ValueError('Held fold/entity cannot enter training audit population')
            p=place(s['positive_features']);negative=[n for n in s['negatives'] if n['siren'] not in protected]
            if not negative: raise ValueError('No safe original training negative; no query filtering')
            locations=[place(n['features']) for n in negative]
            pairs.update(f'{p}_{n}' for n in locations);queries+=1
            zero_contrast+=all(n==p for n in locations)
        result[str(held)]=dict(training_queries=queries,negative_pairs=sum(pairs.values()),
            positive_place_negative_place_counts=dict(pairs),
            queries_with_no_place_contrast=zero_contrast,place_weight=m['weights'][4],
            feature_definition='CRM_INSEE_OR_POSTCODE_MATCH_NOT_DISTANCE_OR_ALL_GEOGRAPHY')
    return result


def worker(root):
    launch=json.loads((root/'launch.json').read_text())
    for ref in launch['implementation'].values():verify(ref)
    evaluation.admission(EVALUATION)
    permission,_=evaluation.permitted(EVALUATION)
    with Budget(LIMITS) as budget:
        def guard():
            if budget.failure:raise RuntimeError(budget.failure)
        files=SourceFiles(guard)
        cfg=json.loads(files.file(launch['source_config']).read_text())
        proof=json.loads(files.file(cfg['proof_source']).read_text())
        old=json.loads(files.file(proof['launch']).read_text())
        if proof['training_queries']!=512 or proof['recall_measured'] is not False or old['truth_scope']!='TRAINER_ONLY':
            raise ValueError('Original training-only complete proof source required')
        scenes=[s for ref in proof['features'] for s in json.loads(files.file(ref).read_text())['scenes']]
        serving=json.loads(files.file(evaluation.authority.generation.config()['serving_query_launch']).read_text())
        models=json.loads(files.file(serving['models']).read_text())
        evaluation.authority.generation.validate_models(models)
        result=summarize(scenes,old['registry'],models,permission['controls'])
        files.verify_after();evaluation.admission(EVALUATION)
        write_json(root/'worker/report.json',dict(folds=result,source_epochs=list(files.entries.values()),
            scope='FROZEN_MODEL_ACTUAL_TRAINING_PAIRS_ONLY_NOT_GLOBAL_RETRIEVAL_RECALL',
            model_fit=False,model_changed=False,recall_measured=False,fold0_opened=False,fold1_opened=False))
    validate_report(budget.report())
    write_json(root/'worker/manifest.json',dict(resources=budget.report(),launch=record(root/'launch.json'),
        report=record(root/'worker/report.json')))


def run(root):
    evaluation.admission(EVALUATION)  # No TRAIN GT either before complete current experiment.
    if root.exists() or not root.resolve().is_relative_to(OUTPUT):raise ValueError('Fresh dedicated audit root required')
    root.mkdir(parents=True);dest=root/'worker';dest.mkdir()
    paths=[Path(__file__),ROOT/'docs/retrieval_v226_training_geography_contract.md',
        SOURCE_CONFIG,Path(evaluation.__file__),ROOT/'src/xgb_matcher/retrieval_memory_policy_v214.py',
        ROOT/'src/xgb_matcher/retrieval_supervisor_v214.py']
    write_json(root/'launch.json',dict(implementation={str(p):record(p) for p in paths},
        source_config=record(SOURCE_CONFIG),evaluation=record(EVALUATION/'manifest.json')))
    write_json(dest/'preflight.json',quiet(30,50))
    with (dest/'worker.log').open('x') as log:
        s=supervise([sys.executable,'-u','-m','scripts.audit_retrieval_v226_training_geography','worker',
            '--output',str(root)],log,{**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},dest/'supervisor_stop.json')
    write_json(dest/'supervision.json',s);parent_exit(dest)
    m=json.loads((dest/'manifest.json').read_text());validate_report(m['resources'])
    if m['launch']!=record(root/'launch.json'):raise ValueError('Own original supervised launch required')
    report=json.loads(verify(m['report']).read_text())
    evaluation.hashes.check_epochs(report['source_epochs']);evaluation.admission(EVALUATION)
    write_json(root/'manifest.json',dict(status='SEALED_V226_FROZEN_OOF_TRAIN_GEOGRAPHY_NOT_RECALL',
        launch=record(root/'launch.json'),worker=record(dest/'manifest.json'),report=m['report']))
    print(json.dumps(report['folds']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('run','worker'))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    run(a.output) if a.mode=='run' else worker(a.output)
