"""Nested entity-disjoint identity/site contrastive probe on cached evidence."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from scripts import retrieval_v38_initial_top100 as v38
from scripts.retrieval_v56_exact_fields import index_authority
from src.xgb_matcher.retrieval_joint_contrastive_v63 import unit_parts, context, select
from src.xgb_matcher.retrieval_v20_workbench import Budget, ROOT, record, verify, write_json

CONFIG = ROOT / 'config/retrieval_v63_joint_contrastive.json'


def worker(output):
    launch = json.loads((output/'launch.json').read_text()); cfg = launch['config']
    v38.verify_launch(launch)
    source = json.loads(verify(cfg['vector_source']).read_text())
    sl = json.loads(verify(source['launch']).read_text()); v38.verify_launch(sl)
    inputs = json.loads(verify(sl['inputs']).read_text())
    cache = json.loads(verify(source['worker']).read_text())
    proof = json.loads(verify(cfg['proof_source']).read_text()); v38.verify_launch(proof)
    pl = json.loads(verify(proof['launch']).read_text()); v38.verify_launch(pl)
    old = json.loads(verify(pl['cached_proof_source']).read_text()); v38.verify_launch(old)
    geo_launch = json.loads(verify(old['launch']).read_text()); v38.verify_launch(geo_launch)
    rows = inputs['rows']
    scenes = [s for ref in proof['features'] for s in json.loads(verify(ref).read_text())['scenes']]
    if len(rows) != 512 or len(scenes) != 512 or cache['query_ids'] != [r['query_id'] for r in rows]:
        raise ValueError('Training-only source denominator/order drift')
    if any(s['recall_measured'] for s in (source, proof)):
        raise ValueError('Not a cached training source')
    q = unit_parts(np.load(verify(cache['query_vectors']), allow_pickle=False))
    docs = unit_parts(np.load(verify(cache['document_vectors']), allow_pickle=False))
    # The encoder prefix itself must not manufacture proof for an empty field.
    for i, row in enumerate(rows):
        for j, field in enumerate(('name', 'address')):
            if not inputs['query_fields'][row['query_id']][field].split(':', 1)[-1].strip():
                q[i, j] = 0.
    for i, siret in enumerate(cache['document_ids']):
        for j, field in enumerate(('name', 'address')):
            if not inputs['document_fields'][siret][field].split(':', 1)[-1].strip():
                docs[i, j] = 0.
    contexts = np.array([context(r, geo_launch['geo'][r['query_id']]) for r in rows])
    models = {}; reports = {}; budget = Budget(cfg['resources'])
    try:
        with budget:
            for held in (2, 3, 4):
                print(json.dumps(dict(starting_fold=held)), flush=True)
                began = time.monotonic()
                models[str(held)], reports[str(held)] = select(rows, scenes, q, docs, cache['document_ids'],
                    contexts, inputs['registry'], held, cfg['ridges'], cfg['maxiter'])
                reports[str(held)]['fit_seconds'] = time.monotonic()-began
                print(json.dumps(dict(fold=held, selected=reports[str(held)]['selected'],
                    before=reports[str(held)]['outer_before'], after=reports[str(held)]['outer_after'])), flush=True)
        v38.verify_launch(launch)
        for ref in (cfg['vector_source'], cfg['proof_source'], sl['inputs'], source['worker']):
            verify(ref)
        write_json(output/'models.json', models)
        write_json(output/'manifest.json', dict(status='SEALED_TRAINING_ONLY_JOINT_CONTRASTIVE_PROBE',
            models=record(output/'models.json'), reports=reports, launch=record(output/'launch.json'),
            resources=budget.report(), seconds=time.monotonic()-launch['monotonic_start'],
            training_queries=512, official_training_documents=len(docs),
            recall_measured=False, full_national_search_performed=False, output_candidates=False,
            index_built=False, no_identifiers_as_features=True, no_crm_alias_dictionary=True,
            fold0_opened=False, fold1_opened=False, ready_for_fold1=False, downstream_models_unchanged=True,
            limitations=['LIMITED_NATURAL_TRAINING_NEGATIVES_NOT_NATIONAL_RECALL100',
                'NO_NATIONAL_INDEX_EXECUTOR_OR_SLO_PROOF', 'RELATIONS_LEGAL_CATEGORY_TIME_NOT_ADDED',
                'CACHED_EMBEDDINGS_COLLAPSE_ALIAS_PROVENANCE_WITHIN_EACH_VIEW',
                'OPERATIONAL_EQUIVALENTS_EXCLUDED_FROM_CACHED_TRAINING_NEGATIVES_NOT_FROM_EVALUATION_DENOMINATOR']))
    except BaseException as exc:
        write_json(output/'worker_stop.json', dict(status='STOP', reason=str(exc),
            resources=budget.report() if hasattr(budget, 'elapsed') else None,
            completed_fold_reports=reports, recall_measured=False, models_deployable=False))
        raise


def run(output):
    output.mkdir(parents=True, exist_ok=False); began = time.monotonic()
    cfg = json.loads(CONFIG.read_text())
    try:
        launch = dict(**index_authority(), config=cfg, monotonic_start=began, truth_scope='TRAINER_ONLY')
        paths = [Path(__file__), CONFIG, ROOT/'docs/retrieval_v63_joint_contrastive_probe.md',
                 ROOT/'src/xgb_matcher/retrieval_joint_contrastive_v63.py',
                 ROOT/'src/xgb_matcher/retrieval_native_v57.py', ROOT/'src/xgb_matcher/retrieval_nested_v53.py']
        launch['implementation'].update({str(p): record(p) for p in paths})
        write_json(output/'launch.json', launch)
        with (output/'worker.log').open('x') as log:
            subprocess.run([sys.executable, '-u', '-m', 'scripts.preflight_retrieval_v63_joint_contrastive',
                'worker', '--output', str(output)], check=True, stdout=log, stderr=subprocess.STDOUT,
                timeout=90, env={**os.environ, 'OMP_NUM_THREADS':'2', 'OPENBLAS_NUM_THREADS':'2',
                               'VECLIB_MAXIMUM_THREADS':'2', 'RAYON_NUM_THREADS':'2'})
        print((output/'worker.log').read_text()[-10000:], flush=True)
    except BaseException as exc:
        write_json(output/'status.json', dict(status='STOP', reason=str(exc),
            seconds=time.monotonic()-began, recall_measured=False, models_deployable=False))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['run', 'worker']); p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    (run if a.mode == 'run' else worker)(a.output)
