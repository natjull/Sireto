"""Bounded same-objective conditioned sparse training, no retrieval recall."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts import retrieval_v38_initial_top100 as v38
from scripts.retrieval_sparse_training_source import load
from scripts.retrieval_v56_exact_fields import index_authority
from src.xgb_matcher.retrieval_conditioned_sparse_v71 import select
from src.xgb_matcher.retrieval_v20_workbench import Budget, ROOT, record, verify, write_json

CONFIG = ROOT/'config/retrieval_v71_conditioned_sparse.json'


def worker(output):
    launch = json.loads((output/'launch.json').read_text()); cfg = launch['config']
    v38.verify_launch(launch); reports = {}; models = {}; budget = Budget(cfg['resources'])
    try:
        with budget:
            q, rows, scenes, documents, aliases, registry, counters = load(cfg, launch)
            for held in (2, 3, 4):
                print(json.dumps(dict(starting_fold=held)), flush=True); began = time.monotonic()
                models[str(held)], reports[str(held)] = select(q, rows, scenes, documents, aliases, registry, held, cfg)
                reports[str(held)]['fit_seconds'] = time.monotonic()-began
                print(json.dumps(dict(fold=held, selected=reports[str(held)]['selected'],
                    before=reports[str(held)]['outer_before'], after=reports[str(held)]['outer_after'])), flush=True)
        v38.verify_launch(launch)
        for ref in (cfg['vector_source'], cfg['proof_source'], cfg['alias_validation']):
            verify(ref)
        write_json(output/'models.json', models)
        write_json(output/'manifest.json', dict(status='SEALED_TRAINING_ONLY_CONDITIONED_SPARSE_PROBE',
            models=record(output/'models.json'), reports=reports, launch=record(output/'launch.json'),
            resources=budget.report(), seconds=time.monotonic()-launch['monotonic_start'], training_queries=512,
            **counters, recall_measured=False, full_national_search_performed=False, output_candidates=False,
            index_built=False, ready_for_fold1=False, fold0_opened=False, fold1_opened=False,
            no_crm_alias_dictionary=True, no_identifiers_as_features=True, downstream_models_unchanged=True,
            limitations=['NATURAL_SOURCE_NEGATIVES_NOT_RETRIEVAL_RECALL', 'NO_NATIONAL_JOINT_EXECUTOR_OR_SLO',
                'NO_TIME_SOURCE_BINDING_GRAPH_OR_LEGAL_ACTIVITY', 'FIXED_DOCUMENT_ENCODER',
                'TEACHER_EXCLUDES_OPERATIONAL_EQUIVALENTS_NOT_RETRIEVAL_DENOMINATOR']))
    except BaseException as exc:
        write_json(output/'worker_stop.json', dict(status='STOP', reason=str(exc), reports=reports,
            resources=budget.report() if hasattr(budget, 'elapsed') else None, recall_measured=False, models_deployable=False))
        raise


def run(output):
    output.mkdir(parents=True, exist_ok=False); began = time.monotonic(); cfg = json.loads(CONFIG.read_text())
    try:
        launch = dict(**index_authority(), config=cfg, monotonic_start=began, truth_scope='TRAINER_ONLY')
        paths = [Path(__file__), CONFIG, ROOT/'docs/retrieval_v71_conditioned_sparse_probe.md',
            ROOT/'scripts/retrieval_sparse_training_source.py', ROOT/'scripts/preflight_retrieval_v69_sparse_contrastive.py',
            *(ROOT/f'src/xgb_matcher/{n}.py' for n in ('retrieval_conditioned_sparse_v71', 'retrieval_continuous_sparse_v70',
                'retrieval_sparse_contrastive_v69', 'retrieval_impact_v68', 'retrieval_initialisms_v67',
                'retrieval_joint_contrastive_v63', 'retrieval_nested_v53', 'retrieval_sparse_query_v59'))]
        launch['implementation'].update({str(p):record(p) for p in paths}); write_json(output/'launch.json', launch)
        with (output/'worker.log').open('x') as log:
            subprocess.run([sys.executable, '-u', '-m', 'scripts.preflight_retrieval_v71_conditioned_sparse',
                'worker', '--output', str(output)], check=True, stdout=log, stderr=subprocess.STDOUT,
                timeout=90, env={**os.environ, 'OMP_NUM_THREADS':'2', 'OPENBLAS_NUM_THREADS':'2',
                    'VECLIB_MAXIMUM_THREADS':'2', 'RAYON_NUM_THREADS':'2'})
        print((output/'worker.log').read_text()[-10000:], flush=True)
    except BaseException as exc:
        write_json(output/'status.json', dict(status='STOP', reason=str(exc), seconds=time.monotonic()-began,
            recall_measured=False, models_deployable=False)); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('mode', choices=['run', 'worker'])
    p.add_argument('--output', required=True, type=Path); a = p.parse_args()
    (run if a.mode == 'run' else worker)(a.output)
