"""Bounded TRAIN-only geographic calibration of the frozen V77 initial metric."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from scripts import retrieval_v38_initial_top100 as v38
from scripts.retrieval_sparse_training_source import load
from scripts.retrieval_v56_exact_fields import index_authority
from src.xgb_matcher.retrieval_proof_v56 import FEATURES, LEXICAL
from src.xgb_matcher.retrieval_sparse_geo_v79 import select
from src.xgb_matcher.retrieval_sparse_query_v59 import unit
from src.xgb_matcher.retrieval_v20_workbench import Budget, ROOT, record, verify, write_json

CONFIG = ROOT / 'config/retrieval_v79_sparse_geo.json'


def authority():
    result = index_authority()
    for path in (Path(__file__), CONFIG, ROOT/'docs/retrieval_v79_sparse_geo_probe.md',
                 ROOT/'src/xgb_matcher/retrieval_sparse_geo_v79.py',
                 ROOT/'scripts/retrieval_sparse_training_source.py',
                 ROOT/'src/xgb_matcher/retrieval_safe_sparse_v77.py',
                 ROOT/'src/xgb_matcher/retrieval_sparse_query_v59.py',
                 ROOT/'src/xgb_matcher/retrieval_native_v57.py',
                 ROOT/'src/xgb_matcher/retrieval_nested_v53.py',
                 ROOT/'src/xgb_matcher/retrieval_proof_v56.py'):
        result['implementation'][str(path)] = record(path)
    return result


def frozen_bundle(config):
    trained = json.loads(verify(config['trained_sparse']).read_text())
    v38.verify_launch(trained)
    if trained['status'] != 'SEALED_SAFE_NESTED_SPARSE_QUERY_BUNDLE' or trained['recall_measured']:
        raise ValueError('Frozen V77 training authority drift')
    models = json.loads(verify(trained['models']).read_text())
    if set(models) != {'2', '3', '4'} or trained['sparse_terms'] != 8:
        raise ValueError('Frozen model fold/expansion scope drift')
    for ref in trained['model_refs'].values():
        verify(ref)
    for fold, model in models.items():
        words = json.loads(verify(model['vocabulary']).read_text())
        projection = np.load(verify(model['projection']), allow_pickle=False)
        if (model['heldout_fold'] != int(fold) or tuple(model['features']) != LEXICAL
                or projection.shape != (384, len(words)) or not np.isfinite(projection).all()
                or model['heldout_documents_used_for_vocabulary']
                or model['vocabulary_protected_document_overlap'] != 0):
            raise ValueError('Frozen sparse prefix authority drift')
    return trained, models


def worker(output):
    launch = json.loads((output/'launch.json').read_text())
    v38.verify_launch(launch); config = launch['config']; budget = Budget(config['resources'])
    with budget:
        trained, originals = frozen_bundle(config)
        _, rows, scenes, documents, _, registry, counters = load(config, launch)
        source = json.loads(verify(config['vector_source']).read_text())
        cache = json.loads(verify(source['worker']).read_text())
        q = unit(np.load(verify(cache['query_vectors']), allow_pickle=False)[:, :384])
        if q.shape != (512, 384) or not np.isfinite(q).all():
            raise ValueError('Cached name-encoder query matrix drift')
        names = {sid: {f: values.get(f, []) for f in v38.NAMES} for sid, values in documents.items()}
        models = {}; reports = {}
        for held in (2, 3, 4):
            fold = str(held); original = originals[fold]
            model = dict(original, projection=np.load(verify(original['projection']), allow_pickle=False),
                         vocabulary=json.loads(verify(original['vocabulary']).read_text()))
            choice = trained['reports'][fold]['selected']
            weights, report = select(q, rows, scenes, names, registry, held, config, choice, model)
            models[fold] = dict(original, geo_weights=weights.tolist(), geo_features=list(FEATURES[11:]),
                                geo_family=report['selected']['family'], frozen_v77_prefix=True)
            if any(models[fold][key] != value for key, value in original.items()):
                raise ValueError('Frozen V77 serving prefix modified')
            reports[fold] = report
        if sum(r['outer_after']['queries'] for r in reports.values()) != 512:
            raise ValueError('Outer TRAIN denominator drift')
        write_json(output/'models.json', models)
    v38.verify_launch(launch); frozen_bundle(config)
    for key in ('vector_source', 'proof_source', 'alias_validation'):
        verify(config[key])
    write_json(output/'manifest.json', dict(**authority(),
        status='SEALED_OOF_SPARSE_GEOGRAPHIC_RESIDUAL_MODELS', launch=record(output/'launch.json'),
        models=record(output/'models.json'), frozen_prefix=config['trained_sparse'], reports=reports,
        counters=counters, resources=budget.report(), model_path=trained['model_path'],
        model_refs=trained['model_refs'], max_length=trained['max_length'], encoder_prefix='nom',
        sparse_terms=8, geographic_features=list(FEATURES[11:]),
        truth_scope='TRAIN_ONLY_NO_HUMAN_LABEL_SIDE_IN_SERVING_MODELS', recall_measured=False,
        full_national_search_performed=False, ready_for_fold1=False, fold0_opened=False, fold1_opened=False,
        contains_crm_alias_dictionary=False, identifiers_as_features=False, downstream_models_changed=False,
        base_predictions_out_of_fit_source_fold=True,
        semantic_hyperparameters_frozen_not_reselected_within_crossfit=True,
        inner_validation_reused_from_v77_hyperparameter_selection=True))
    print(json.dumps(dict(resources=budget.report(), reports={f: dict(
        family=r['selected']['family'], inner=r['selected']['inner'], weights=models[f]['geo_weights'],
        fit=r['final_fit'], before=r['outer_before'], after=r['outer_after']) for f, r in reports.items()},
        recall_measured=False)), flush=True)


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads(CONFIG.read_text())
    write_json(output/'launch.json', dict(**authority(), config=config, truth_scope='TRAIN_ONLY'))
    try:
        with (output/'worker.log').open('x') as log:
            subprocess.run([sys.executable, '-u', '-m', 'scripts.preflight_retrieval_v79_sparse_geo',
                            'worker', '--output', str(output)], check=True, timeout=90,
                           stdout=log, stderr=subprocess.STDOUT,
                           env={**os.environ, 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
                                'VECLIB_MAXIMUM_THREADS': '2', 'RAYON_NUM_THREADS': '2'})
        print((output/'worker.log').read_text()[-6500:], flush=True)
    except BaseException as exc:
        write_json(output/'status.json', dict(status='STOP', reason=str(exc), recall_measured=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['run', 'worker'])
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    (run if args.mode == 'run' else worker)(args.output)
