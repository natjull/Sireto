"""Refit only frozen safe OOF features; no evaluation query/label is opened."""
import argparse
import json
from pathlib import Path
import time

from scripts import retrieval_v38_initial_top100 as v38
from src.xgb_matcher.retrieval_training_v50 import fit_aligned
from src.xgb_matcher.retrieval_v20_workbench import Budget,ROOT,record,verify,write_json

SOURCE=dict(path='/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v46_joint/training_20261002_a/manifest.json',
    sha256='523bce3a4abec664a481ace967b1faa4283dcd9c9d0c9c83c0715887ab7c1495')
RESOURCES=dict(max_seconds=60,max_rss_bytes=1024**3,max_swapout_pages_delta=0)


def run(output):
    output.mkdir(parents=True,exist_ok=False); began=time.monotonic()
    try:
        source=json.loads(verify(SOURCE).read_text()); v38.verify_launch(source)
        if source['status']!='SEALED_OOF_JOINT_INITIAL_METRICS': raise ValueError('Unsealed cached features')
        launch=json.loads(verify(source['launch']).read_text()); v38.verify_launch(launch)
        if set(source['features'])!={'2','3','4'}: raise ValueError('Feature fold drift')
        models={}; counts={}; refs=[]
        with Budget(RESOURCES) as budget:
            for held in (2,3,4):
                scenes=[]
                for ref in source['features'][str(held)]:
                    batch=json.loads(verify(ref).read_text())
                    if not batch['resources']['pass_resource_policy']: raise ValueError('Unsafe cached mining')
                    scenes.extend(batch['scenes']); refs.append(ref)
                expected=launch['rows_by_held'][str(held)]
                if len(scenes)!=len(expected) or {r['query_id'] for r in scenes}!={r['query_id'] for r in expected}:
                    raise ValueError('Cached training denominator drift')
                models[str(held)],counts[str(held)]=fit_aligned(scenes,held,launch['registry'])
        for ref in refs: verify(ref)
        v38.verify_launch(source); v38.verify_launch(launch)
        write_json(output/'models.json',models)
        impl={str(p):record(p) for p in (Path(__file__),ROOT/'src/xgb_matcher/retrieval_training_v50.py')}
        write_json(output/'manifest.json',dict(status='SEALED_OOF_JOINT_INITIAL_METRICS',source=SOURCE,
            launch=source['launch'],models=record(output/'models.json'),features=source['features'],counts=counts,
            training_queries=128,seconds=time.monotonic()-began,resources=budget.report(),
            implementation={**source['implementation'],**impl},index_manifest=source['index_manifest'],
            index_meta=source['index_meta'],inventory=source['inventory'],evaluation_opened=False,
            fold0_opened=False,fold1_opened=False,identifiers_as_features=False,
            downstream_models_unchanged=True,queries_or_documents_dropped=0,recall_measured=False,
            architecture_changed=False,ready_for_fold1=False))
        print(json.dumps(dict(counts=counts,seconds=time.monotonic()-began,recall_measured=False)))
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),recall_measured=False)); raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',required=True,type=Path)
    run(p.parse_args().output)
