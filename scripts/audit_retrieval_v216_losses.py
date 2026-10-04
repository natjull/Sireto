"""ALL21 post-V215 losses: official names, point XY and pure text counterfactual.

Human IDs are accepted ONLY after actual V215 all128 admission, for diagnostics.
Counterfactual candidate lists are not saved or reusable as generation outputs.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from scripts import evaluate_retrieval_v215_postseal as evaluation
from scripts import retrieval_v38_initial_top100 as v38
from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery, TantivyBackend
from src.xgb_matcher.retrieval_loss_evidence_v216 import geometry, nominal_evidence, summarize
from src.xgb_matcher.retrieval_memory_policy_v214 import Budget, LIMITS, quiet, parent_exit, validate_report
from src.xgb_matcher.retrieval_postscore_audit_v187 import canonical_epochs, check_epochs
from src.xgb_matcher.retrieval_structural_channels_v210 import SourceFiles
from src.xgb_matcher.retrieval_supervisor_v214 import supervise
from src.xgb_matcher.retrieval_variants_v118 import normalized
from src.xgb_matcher.retrieval_v20_workbench import ROOT, record, verify, write_json

EVALUATION = Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v215_postseal_evaluation/run_20261004_a')
NAMES = dict(path='/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v101_national_paths/name_catalog_v103_20261003_a/report.json',
    sha256='1c3d335521d5d2653055a64c55c865437fbf5633dfa8bd4a59f366e7c9323aab')
OUTPUT = Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v216_loss_evidence')
STATUS = 'SEALED_V216_ALL21_POSTSEAL_LOSS_DIAGNOSTICS_NOT_RECALL'


def implementation():
    paths = (Path(__file__), ROOT/'src/xgb_matcher/retrieval_loss_evidence_v216.py',
        ROOT/'docs/retrieval_v216_loss_evidence_contract.md', Path(evaluation.__file__),
        ROOT/'src/xgb_matcher/hierarchical_retrieval.py', Path(v38.__file__),
        ROOT/'src/xgb_matcher/retrieval_structural_channels_v210.py',
        ROOT/'src/xgb_matcher/retrieval_variants_v118.py',
        ROOT/'src/xgb_matcher/retrieval_memory_policy_v214.py',
        ROOT/'src/xgb_matcher/retrieval_supervisor_v214.py')
    return {str(p): record(p) for p in paths}


def inputs():
    admitted = evaluation.admission(EVALUATION)  # Complete actual exits/seals BEFORE any target facts.
    report = json.loads(verify(admitted['artifacts']['report']).read_text())
    permission, _ = evaluation.permitted(EVALUATION)
    labels, _, _, refs, epochs, _ = evaluation.targets(evaluation.config(), permission)
    names = json.loads(verify(NAMES).read_text())
    nl = json.loads(verify(names['launch']).read_text())
    old_evaluation = json.loads(verify(json.loads(verify(nl['generation']).read_text())['launch']).read_text())
    old_metric = json.loads(verify(nl['point_report']).read_text())['evaluation']
    old_metric = json.loads(verify(old_metric).read_text())
    if old_metric['ledger']['sha256'] != evaluation.TARGET_LEDGER['sha256']:
        raise ValueError('Same unchanged human ledger required for diagnostic names')
    launch, _, outputs = evaluation.authority.generation.sealed(verify(permission['generation']).parent)
    if (names['status'] != 'SEALED_ALL128_LABEL_SIDE_OFFICIAL_NAME_CATALOG'
            or names['queries'] != 128 or names['not_a_serving_source'] is not True
            or names['outputs_promoted_to_features'] is not False or names['generation_modified'] is not False
            or names['fold0_opened'] is not False or names['fold1_opened'] is not False
            or old_evaluation['queries'] != launch['queries']):
        raise ValueError('Sealed same128 names strictly diagnostic, not serving input')
    truth = {r['query_id']: r for r in labels}
    nominal = {r['query_id']: r for r in names['rows']}
    if len(nominal) != 128 or set(nominal) != set(truth):
        raise ValueError('Complete consistent128 diagnostic owners required')
    for q in launch['queries']:
        n, t = nominal[q['query_id']], truth[q['query_id']]
        if (n['fold'] != q['fold'] or n['crm_name'] != q['crm_name'] or
            n['human_siret'] != t['human_ground_truth_siret'] or n['human_siren'] != t['human_ground_truth_siren']):
            raise ValueError('Unchanged exact GT and CRM diagnostic membership required')
    misses = [r for r in report['outcomes'] if not r['exact_hit']]
    if len(misses) != 21: raise ValueError('Original ALL21 misses required')
    geo = {r['query_id']: r for r in json.loads(verify(launch['geo_queries']).read_text())}
    source = json.loads(verify(json.loads(verify(launch['backend_config']).read_text())['source_config']).read_text())
    attributes = json.loads(verify(source['structured']).read_text())
    return launch, outputs, truth, nominal, misses, geo, permission['native_keys'], attributes['reused_arrays']['site_xy'], refs, epochs


def worker(root, batch):
    launch = json.loads((root/'launch.json').read_text())
    for ref in launch['implementation'].values(): verify(ref)
    actual, outputs, truth, nominal, misses, geo, key_ref, xy_ref, refs, epochs = inputs()
    if launch['evaluation'] != record(EVALUATION/'manifest.json') or launch['names'] != NAMES:
        raise ValueError('Own original sealed216 authority required')
    dest = root/f'batch{batch:03d}'
    preflight = json.loads((dest/'preflight.json').read_text())
    if (preflight['pass_system_quiet_policy'] is not True or preflight['quiet_seconds'] < 30
            or preflight['worker_swap_budget_changed'] is not True):
        raise ValueError('Actual own authorized quiet preflight required')
    if any(p.name not in ('worker.log', 'preflight.json') for p in dest.iterdir()):
        raise ValueError('Fresh diagnostic worker; no overwrite/retry')
    rows, budget, native = [], None, None
    try:
        with Budget(LIMITS) as budget:
            def guard():
                if budget.failure: raise RuntimeError(budget.failure)
                if time.monotonic()-budget.start > 80: raise TimeoutError('STOP_DIAGNOSTIC_WORKER_COST')
            files = SourceFiles(guard)
            keys = files.mapped(key_ref, np.uint64, (43896818,))
            xy_path = files.file(xy_ref)
            emitted = {r['query_id']: r for r in outputs['rows']}
            try:
                for outcome in misses[7*batch:7*(batch+1)]:
                    began = time.monotonic(); qid = outcome['query_id']; raw = actual['queries'][outcome['position']]
                    target = truth[qid]['human_ground_truth_siret']
                    p = int(np.searchsorted(keys, np.uint64(int(target))))
                    if p >= len(keys) or keys[p] != int(target): raise ValueError('Admitted native presence conflict')
                    xy = np.load(xy_path, mmap_mode='r', allow_pickle=False)
                    try:
                        if xy.dtype != np.float64 or xy.shape != (43896818, 2) or xy.flags.writeable:
                            raise ValueError('Complete actual native XY required')
                        point = xy[p].tolist()
                    finally: xy._mmap.close()
                    q = RetrievalQuery(name=normalized(raw['crm_name']),
                        address=normalized(' '.join(raw[k] for k in ('crm_number', 'crm_suffix', 'crm_street'))),
                        number=raw['crm_number'], postcode=raw['crm_postcode'], insee=raw['crm_insee'])
                    native = TantivyBackend(v38.INDEX)
                    if native.searcher.num_docs != actual['index_documents']: raise ValueError('Native corpus drift')
                    tv = native._tantivy
                    docgate = tv.Query.boolean_query([
                        (tv.Occur.Must, tv.Query.term_query(native.index.schema, 'siret', target, 'basic')),
                        (tv.Occur.Must, tv.Query.term_query(native.index.schema, 'document_type', 'siret', 'basic'))])
                    docs = native.searcher.search(docgate, limit=2, count=False).hits
                    if len(docs) != 1: raise ValueError('Exactly one original native GT doc required')
                    doc = native.searcher.doc(docs[0][1]).to_dict()
                    # Original text-only top100, not an expanded pool or injected positive.
                    candidates = v38.search(native, q, 'primary')
                    rank = next((i+1 for i, c in enumerate(candidates) if c['siret'] == target), None)
                    counters = dict(rank=rank, cutoff_score=candidates[-1]['score'] if candidates else None,
                        diagnostic_subset_only=True, promoted_to_generation=False)
                    row = dict(query_id=qid, fold=outcome['fold'], position=outcome['position'], crm=raw,
                        human_siret=target, original_operational_hit=outcome['operational_hit'],
                        names=nominal_evidence(raw['crm_name'], nominal[qid]['facts']),
                        geometry=geometry(geo[qid], point), native_gt_fields={f: doc.get(f, []) for f in
                            (*v38.NAMES, *v38.ADDRESSES, 'postcode', 'insee')},
                        original_emitted_by_channel=emitted[qid]['primary_trace']['source_admitted'],
                        text_counterfactual=counters, seconds=time.monotonic()-began)
                    if row['seconds'] > 20: raise TimeoutError('STOP_DIAGNOSTIC_QUERY_COST')
                    rows.append(row); native = None; candidates = None; gc.collect(); guard()
                    print(json.dumps(dict(audited=outcome['position'], text_counterfactual_rank=rank,
                        geometry=row['geometry'], seconds=row['seconds'])), flush=True)
            finally: keys._mmap.close()
            files.verify_after()
            for ref in refs: evaluation.checked(ref)
            evaluation.hashes.check_epochs(epochs)
            if v38.inventory() != actual['inventory']: raise ValueError('Native inventory drift')
            for ref in launch['implementation'].values(): verify(ref)
            verify(NAMES)
            write_json(dest/'payload.json', dict(rows=rows, source_epochs=list(files.entries.values()),
                truth_scope='POST_ALL128_LABEL_SELECTED_DIAGNOSTIC_ONLY', not_a_serving_source=True))
        validate_report(budget.report())
        write_json(dest/'manifest.json', dict(status='QUARANTINED_V216_DIAGNOSTIC', resources=budget.report(),
            launch=record(root/'launch.json'), preflight=record(dest/'preflight.json'), payload=record(dest/'payload.json')))
    except BaseException as exc:
        write_json(dest/'worker_stop.json', dict(status='STOP', reason=str(exc), completed=len(rows),
            resources=budget.report() if budget is not None and hasattr(budget, 'elapsed') else None,
            recall_measured=False, promoted_to_generation=False)); raise


def run(root):
    if root.exists() or not root.resolve().is_relative_to(OUTPUT): raise ValueError('Fresh dedicated216 root required')
    inputs()  # Refuse incomplete/STOP generation before creating a diagnostic run.
    root.mkdir(parents=True, exist_ok=False)
    write_json(root/'launch.json', dict(evaluation=record(EVALUATION/'manifest.json'), names=NAMES,
        implementation=implementation(), truth_scope='POST_ALL128_LABEL_SELECTED_DIAGNOSTIC_ONLY',
        not_a_serving_source=True, resources=LIMITS, queries=21, batch_width=7))
    rows, batches, epochs = [], [], []
    try:
        for b in range(3):
            dest = root/f'batch{b:03d}'; dest.mkdir()
            write_json(dest/'preflight.json', quiet(30, 50))
            with (dest/'worker.log').open('x') as log:
                s = supervise([sys.executable, '-u', '-m', 'scripts.audit_retrieval_v216_losses', 'worker',
                    '--output', str(root), '--batch', str(b)], log,
                    {**os.environ, 'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1',
                     'RAYON_NUM_THREADS':'1', 'VECLIB_MAXIMUM_THREADS':'1'}, dest/'supervisor_stop.json')
            write_json(dest/'supervision.json', s); parent_exit(dest)
            m = json.loads((dest/'manifest.json').read_text()); validate_report(m['resources'])
            if (m['status'] != 'QUARANTINED_V216_DIAGNOSTIC' or
                    m['launch'] != record(root/'launch.json') or m['preflight'] != record(dest/'preflight.json')):
                raise ValueError('Actual own supervised diagnostic manifest required')
            payload = json.loads(verify(m['payload']).read_text())
            if (len(payload['rows']) != 7 or payload['not_a_serving_source'] is not True or
                    payload['truth_scope'] != 'POST_ALL128_LABEL_SELECTED_DIAGNOSTIC_ONLY'):
                raise ValueError('Whole diagnostic batch required')
            epochs.extend(payload['source_epochs'])
            rows.extend(payload['rows']); batches.append(record(dest/'manifest.json'))
            print(json.dumps(dict(completed=len(rows), resources=m['resources'])), flush=True)
        _, _, _, _, misses, _, _, _, _, _ = inputs()
        if [(r['query_id'], r['fold']) for r in rows] != [(r['query_id'], r['fold']) for r in misses]:
            raise ValueError('ALL21 unchanged miss controls required')
        report = summarize(rows)
        epochs = canonical_epochs(epochs); check_epochs(epochs)
        report.update(status=STATUS, launch=record(root/'launch.json'), batches=batches, source_epochs=epochs)
        write_json(root/'report.json', report)
        print(json.dumps(report['categories_nonexclusive']), flush=True)
    except BaseException as exc:
        write_json(root/'status.json', dict(status='STOP', reason=str(exc), completed=len(rows), recall_measured=False)); raise


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode', choices=('run','worker'))
    p.add_argument('--output', type=Path, required=True); p.add_argument('--batch', type=int); a=p.parse_args()
    if a.mode == 'worker' and a.batch not in range(3): p.error('Exact diagnostic batch0..2 required')
    run(a.output) if a.mode == 'run' else worker(a.output, a.batch)
