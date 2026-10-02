"""Nested contrastive capacity selection using sealed training vectors only."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from scripts import retrieval_v38_initial_top100 as v38
from src.xgb_matcher.retrieval_nested_v53 import select_and_refit
from src.xgb_matcher.retrieval_v20_workbench import Budget,ROOT,record,verify,write_json

CONFIG=ROOT/'config/retrieval_v53_nested_training.json'


def run(output,config_path=CONFIG,solver=None,extra_implementation=()):
    output.mkdir(parents=True,exist_ok=False); began=time.monotonic(); cfg=json.loads(config_path.read_text()); budget=None
    try:
        source=json.loads(verify(cfg['source']).read_text())
        if source['status']!='SEALED_TRAINING_ONLY_STRUCTURED_CONTRASTIVE_PROBE' or source['recall_measured'] or source['index_built']:
            raise ValueError('Training-only vectors required')
        launch=json.loads(verify(source['launch']).read_text()); v38.verify_launch(launch)
        data=json.loads(verify(launch['inputs']).read_text()); worker=json.loads(verify(source['worker']).read_text())
        if not worker['resources']['pass_resource_policy']: raise ValueError('Unsafe training source')
        rows=data['rows']; memberships=[dict(query_id=r['query_id'],fold=r['fold'],positive_siren=r['positive_siren'],negative_sirens=[n['siren'] for n in r['negatives']]) for r in rows]
        q=np.load(verify(worker['query_vectors']),allow_pickle=False)
        docs=np.load(verify(worker['document_vectors']),mmap_mode='r',allow_pickle=False); byid={s:i for i,s in enumerate(worker['document_ids'])}
        if [r['query_id'] for r in rows]!=worker['query_ids'] or len(q)!=512 or len(docs)!=len(byid): raise ValueError('Vector authority alignment drift')
        p=np.array([docs[byid[r['positive_siret']]] for r in rows]); negatives=[np.array([docs[byid[n['siret']]] for n in r['negatives']]) for r in rows]
        reports={}; refs={}
        with Budget(cfg['resources']) as budget:
            for held in (2,3,4):
                options={} if solver is None else dict(solver=solver)
                w,reports[str(held)]=select_and_refit(q,p,negatives,memberships,held,data['registry'],cfg,**options)
                path=output/f'query_transform_fold{held}.npy'; np.save(path,w,allow_pickle=False); refs[str(held)]=record(path)
        v38.verify_launch(launch); verify(cfg['source']); verify(source['worker']); verify(launch['inputs'])
        paths=[Path(__file__),config_path,ROOT/'src/xgb_matcher/retrieval_nested_v53.py',ROOT/'src/xgb_matcher/retrieval_contrastive_v52.py',
            ROOT/'src/xgb_matcher/retrieval_learned_v41.py',ROOT/'src/xgb_matcher/retrieval_v20_workbench.py',Path(v38.__file__),*extra_implementation]
        write_json(output/'manifest.json',dict(status='SEALED_NESTED_TRAINING_ONLY_METRIC_SELECTION',source=cfg['source'],
            config=record(config_path),implementation={str(p):record(p) for p in paths},reports=reports,transforms=refs,
            seconds=time.monotonic()-began,resources=budget.report(),training_queries=512,recall_measured=False,index_built=False,
            outer_controls_used_in_selection=False,fold0_opened=False,fold1_opened=False,ready_for_fold1=False,downstream_models_unchanged=True))
        print(json.dumps(dict(reports={f:dict(selected=r['selected']['capacity'],ridge=r['selected']['ridge'],outer=r['outer']) for f,r in reports.items()},
            seconds=time.monotonic()-began,recall_measured=False)))
    except BaseException as exc:
        details=budget.report() if budget is not None and hasattr(budget,'elapsed') else None
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),recall_measured=False,resources=details,seconds=time.monotonic()-began)); raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',required=True,type=Path)
    run(p.parse_args().output)
