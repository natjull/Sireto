"""Bounded, nested learned-sparse probe; no national candidate evaluation."""
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
from src.xgb_matcher.hierarchical_retrieval import TantivyBackend, normalize_text
from src.xgb_matcher.retrieval_proof_v56 import signal
from src.xgb_matcher.retrieval_sparse_contrastive_v69 import select, unit_parts
from src.xgb_matcher.retrieval_v20_workbench import Budget, ROOT, record, verify, write_json

CONFIG = ROOT/'config/retrieval_v69_sparse_contrastive.json'


def worker(output):
    launch = json.loads((output/'launch.json').read_text()); cfg = launch['config']
    v38.verify_launch(launch)
    reports = {}; models = {}; budget = Budget(cfg['resources'])
    try:
        with budget:
            source = json.loads(verify(cfg['vector_source']).read_text())
            sl = json.loads(verify(source['launch']).read_text()); v38.verify_launch(sl)
            data = json.loads(verify(sl['inputs']).read_text())
            cache = json.loads(verify(source['worker']).read_text())
            proof = json.loads(verify(cfg['proof_source']).read_text()); v38.verify_launch(proof)
            alias_proof = json.loads(verify(cfg['alias_validation']).read_text()); v38.verify_launch(alias_proof)
            if not alias_proof['native_current_alias_parity'] or alias_proof['recall_measured']:
                raise ValueError('Missing corrected alias validation')
            rows = data['rows']
            scenes = [s for ref in proof['features'] for s in json.loads(verify(ref).read_text())['scenes']]
            if source['recall_measured'] or proof['recall_measured'] or len(rows) != 512 or len(scenes) != 512:
                raise ValueError('Training-only denominator drift')
            if cache['query_ids'] != [r['query_id'] for r in rows]:
                raise ValueError('Cached query order drift')
            q = unit_parts(np.load(verify(cache['query_vectors']), allow_pickle=False))
            for i, r in enumerate(rows):
                for view, field in enumerate(('name', 'address')):
                    if not data['query_fields'][r['query_id']][field].split(':', 1)[-1].strip():
                        q[i, view] = 0.
            backend = TantivyBackend(v38.INDEX); tv = backend._tantivy
            if backend.searcher.num_docs != launch['index_documents']:
                raise ValueError('National official index count drift')
            ids = cache['document_ids']; documents = {}; aliases = {}
            if len(ids) != len(set(ids)):
                raise ValueError('Duplicate official training document')
            for start in range(0, len(ids), 2048):
                subset = ids[start:start+2048]
                hits = backend.searcher.search(tv.Query.term_set_query(backend.index.schema, 'siret', subset),
                                               limit=len(subset)+1, count=False).hits
                for _, loc in hits:
                    d = backend.searcher.doc(loc).to_dict(); sid = backend._first(d, 'siret')
                    if sid not in subset or sid in documents or not sid.startswith(backend._first(d, 'siren')):
                        raise ValueError('Official source identity drift')
                    documents[sid] = {f: d.get(f, []) for f in (*v38.NAMES, *v38.ADDRESSES)}
                    aliases[sid] = [v for f in ('legal_names', 'trade_names', 'site_names') for v in d.get(f, [])]
            if set(documents) != set(ids):
                raise ValueError('Official training source missing; no filtering')
            exact_audited = 0
            for r, s in zip(rows, scenes):
                name = signal(r['crm_name'])
                refs = [(r['positive_siret'], s['positive_features']),
                        *[(n['siret'], p['features']) for n, p in zip(r['negatives'], s['negatives'])]]
                for sid, features in refs:
                    exact = float(bool(name and name in {normalize_text(a) for a in aliases[sid]}))
                    if exact != features[2]:
                        raise ValueError('Stored alias/native exact proof mismatch')
                    exact_audited += 1
            for held in (2, 3, 4):
                print(json.dumps(dict(starting_fold=held)), flush=True); began = time.monotonic()
                models[str(held)], reports[str(held)] = select(q, rows, scenes, documents, aliases,
                                                               data['registry'], held, cfg)
                reports[str(held)]['fit_seconds'] = time.monotonic()-began
                print(json.dumps(dict(fold=held, selected=reports[str(held)]['selected'],
                    before=reports[str(held)]['outer_before'], after=reports[str(held)]['outer_after'])), flush=True)
        v38.verify_launch(launch)
        for ref in (cfg['vector_source'], cfg['proof_source'], cfg['alias_validation']):
            verify(ref)
        write_json(output/'models.json', models)
        write_json(output/'manifest.json', dict(status='SEALED_TRAINING_ONLY_SPARSE_CONTRASTIVE_PROBE',
            models=record(output/'models.json'), reports=reports, launch=record(output/'launch.json'),
            resources=budget.report(), seconds=time.monotonic()-launch['monotonic_start'],
            training_queries=512, official_training_documents=len(ids), current_exact_vectors_audited=exact_audited,
            recall_measured=False, full_national_search_performed=False, output_candidates=False,
            index_built=False, ready_for_fold1=False, fold0_opened=False, fold1_opened=False,
            no_crm_alias_dictionary=True, no_identifiers_as_features=True, downstream_models_unchanged=True,
            limitations=['NATURAL_SOURCE_NEGATIVES_NOT_RETRIEVAL_RECALL', 'NO_NATIONAL_JOINT_EXECUTOR_OR_SLO',
                'NO_TIME_SOURCE_BINDING_GRAPH_OR_LEGAL_ACTIVITY', 'FIXED_DOCUMENT_ENCODER',
                'TEACHER_EXCLUDES_OPERATIONAL_EQUIVALENTS_NOT_RETRIEVAL_DENOMINATOR']))
    except BaseException as exc:
        write_json(output/'worker_stop.json', dict(status='STOP', reason=str(exc), reports=reports,
            resources=budget.report() if hasattr(budget, 'elapsed') else None,
            recall_measured=False, models_deployable=False))
        raise


def run(output):
    output.mkdir(parents=True, exist_ok=False); began = time.monotonic(); cfg = json.loads(CONFIG.read_text())
    try:
        launch = dict(**index_authority(), config=cfg, monotonic_start=began, truth_scope='TRAINER_ONLY')
        paths = [Path(__file__), CONFIG, ROOT/'docs/retrieval_v69_sparse_contrastive_probe.md',
                 *(ROOT/f'src/xgb_matcher/{name}.py' for name in (
                    'retrieval_sparse_contrastive_v69', 'retrieval_impact_v68', 'retrieval_initialisms_v67',
                    'retrieval_joint_contrastive_v63', 'retrieval_nested_v53', 'retrieval_sparse_query_v59'))]
        launch['implementation'].update({str(p): record(p) for p in paths}); write_json(output/'launch.json', launch)
        with (output/'worker.log').open('x') as log:
            subprocess.run([sys.executable, '-u', '-m', 'scripts.preflight_retrieval_v69_sparse_contrastive',
                            'worker', '--output', str(output)], check=True, stdout=log, stderr=subprocess.STDOUT,
                           timeout=90, env={**os.environ, 'OMP_NUM_THREADS':'2', 'OPENBLAS_NUM_THREADS':'2',
                                           'VECLIB_MAXIMUM_THREADS':'2', 'RAYON_NUM_THREADS':'2'})
        print((output/'worker.log').read_text()[-10000:], flush=True)
    except BaseException as exc:
        write_json(output/'status.json', dict(status='STOP', reason=str(exc), seconds=time.monotonic()-began,
                                              recall_measured=False, models_deployable=False))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('mode', choices=['run', 'worker'])
    p.add_argument('--output', required=True, type=Path); a = p.parse_args()
    (run if a.mode == 'run' else worker)(a.output)
