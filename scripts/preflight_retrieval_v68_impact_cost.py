"""Bounded official-corpus physical-plan probe. No labels, training or recall."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from scripts import retrieval_v38_initial_top100 as v38
from scripts.retrieval_v56_exact_fields import index_authority
from src.xgb_matcher.hierarchical_retrieval import TantivyBackend
from src.xgb_matcher.retrieval_initialisms_v67 import membership
from src.xgb_matcher.retrieval_impact_v68 import ImpactIndex, document_impacts, query_impacts, query_initialisms
from src.xgb_matcher.retrieval_v20_workbench import ROOT, Budget, record, verify, write_json

CONFIG = ROOT / 'config/retrieval_v68_impact_cost.json'


def matrix_bytes(matrix):
    return sum(a.nbytes for a in (matrix.data, matrix.indices, matrix.indptr))


def worker(output):
    launch = json.loads((output / 'launch.json').read_text()); cfg = launch['config']
    v38.verify_launch(launch)
    for ref in launch['query_sources']: verify(ref)
    budget = Budget(cfg['resources']); reports = []
    try:
        with budget:
            backend = TantivyBackend(v38.INDEX); tv = backend._tantivy
            if backend.searcher.num_docs != launch['index_documents']:
                raise ValueError('National official corpus count drift')
            before = time.monotonic()
            hits = backend.searcher.search(tv.Query.term_query(backend.index.schema, 'document_type', 'siret', 'basic'),
                limit=cfg['documents'], count=False).hits
            if len(hits) != cfg['documents']: raise ValueError('Independent corpus sample cardinality drift')
            documents = {}; digest = hashlib.sha256(); aliases_audited = 0
            probes = sorted({code for row in launch['queries'] for code in query_initialisms(row['crm_name'])})
            for i, (_, location) in enumerate(hits):
                doc = backend.searcher.doc(location).to_dict()
                key = backend._first(doc, 'siret'); parent = backend._first(doc, 'siren')
                if not (len(key) == 14 and key.isdigit() and key.startswith(parent) and len(parent) == 9):
                    raise ValueError('Official site identity drift')
                if key in documents: raise ValueError('Duplicate official site, not deduplicated silently')
                impacts = document_impacts(doc)
                aliases = [v for field in ('legal_names', 'trade_names', 'site_names') for v in doc.get(field, [])]
                for code in probes:
                    if ('I:' + code in impacts) != bool(membership(code, aliases)):
                        raise ValueError('Index-time initialisms vs official alias regex drift')
                    aliases_audited += 1
                documents[key] = impacts
                digest.update(json.dumps([key, sorted(impacts.items())], ensure_ascii=True).encode() + b'\n')
                if (i + 1) % 8192 == 0:
                    print(json.dumps(dict(official_sites_encoded=i+1, seconds=time.monotonic()-before)), flush=True)
            read_encode_seconds = time.monotonic() - before
            before = time.monotonic(); index = ImpactIndex.build(documents, cfg['block_size'])
            build_seconds = time.monotonic() - before
            del documents, hits
            term_df = np.diff(index.postings.indptr)
            common = sorted(((int(term_df[col]), term) for term, col in index.terms.items() if term.startswith(('N:', 'I:'))),
                            key=lambda r: (-r[0], r[1]))[:cfg['common_term_stress_queries']]
            if len(common) != cfg['common_term_stress_queries']: raise ValueError('Insufficient common term stress coverage')
            queries = [(row['query_id'], 'CRM_QUERY_ONLY', query_impacts(row)) for row in launch['queries']]
            queries += [(f'common-term-{i}', 'OFFICIAL_COMMON_TERM_COST_SENTINEL_NOT_CRM', {term: 1.})
                        for i, (_, term) in enumerate(common)]
            print(json.dumps(dict(index_ready=True, build_seconds=build_seconds, terms=len(index.terms),
                postings=index.postings.nnz, initialism_vectors_audited=aliases_audited)), flush=True)
            for repeat in range(cfg['repetitions']):
                for query_id, kind, query in queries:
                    values = {}; results = {}; trace = None
                    for method in (('pruned', 'exhaustive') if repeat % 2 == 0 else ('exhaustive', 'pruned')):
                        start = time.monotonic()
                        if method == 'pruned': results[method], trace = index.search(query)
                        else: results[method] = index.exhaustive(query)
                        values[method] = time.monotonic() - start
                    if results['pruned'] != results['exhaustive']:
                        raise ValueError('STOP_TOP100_SCORE_OR_TIE_PARITY')
                    reports.append(dict(query_id=query_id, kind=kind, repeat=repeat, times=values,
                        trace=trace, returned_count=len(results['pruned']), exact_score_and_order_parity=True))
            summary = {}
            for kind in sorted({r['kind'] for r in reports}):
                subset = [r for r in reports if r['kind'] == kind]
                summary[kind] = dict(searches=len(subset), nonempty_searches=sum(r['returned_count'] > 0 for r in subset),
                    mean_scored_fraction=float(np.mean([r['trace']['scored_documents'] / len(index.keys) for r in subset])),
                    methods={method: dict(mean=float(np.mean([r['times'][method] for r in subset])),
                        p50=float(np.percentile([r['times'][method] for r in subset], 50)),
                        p95=float(np.percentile([r['times'][method] for r in subset], 95)))
                        for method in ('pruned', 'exhaustive')})
            metrics = dict(summary=summary, rows=reports, independent_official_sites=len(index.keys),
                source_document_impacts_sha256=digest.hexdigest(), vocabulary_terms=len(index.terms),
                document_postings=index.postings.nnz, block_maxima_postings=index.maxima.nnz,
                sparse_arrays_bytes=matrix_bytes(index.postings)+matrix_bytes(index.maxima),
                read_encode_seconds=read_encode_seconds, build_seconds=build_seconds,
                initialism_vectors_audited=aliases_audited)
        v38.verify_launch(launch)
        write_json(output / 'metrics.json', metrics)
        write_json(output / 'manifest.json', dict(status='SEALED_COST_ONLY_EXACT_DIRECT_TOP100_PARITY',
            launch=record(output/'launch.json'), metrics=record(output/'metrics.json'), resources=budget.report(),
            seconds=time.monotonic()-launch['monotonic_start'], recall_measured=False, truth_opened=False,
            no_candidates_written=True, no_training_performed=True, full_national_search_performed=False,
            ready_for_fold1=False, fold0_opened=False, fold1_opened=False, downstream_models_unchanged=True,
            limitations=['QUERY_INDEPENDENT_PHYSICAL_PREFIX_NOT_RANDOM_NATIONAL_CORPUS',
                'UNTRAINED_COST_SCORE_NOT_V67_LEARNED_SCORE', 'ALL_DATA_MEMORY_RESIDENT_NO_COLD_SSD_SLO',
                'PYTHON_BLOCK_BOUNDS_REFERENCE_NOT_PRODUCTION_BMW', 'BAN_XY_GRAPH_TIME_NOT_SERVED',
                'BAG_WORD_VIEWS_NO_FULL_SOURCE_TIME_COHERENCE', 'NO_NATIONAL_LATENCY_OR_RECALL_CONCLUSION']))
        print(json.dumps(dict(summary=summary, resources=budget.report(), read_encode_seconds=read_encode_seconds,
            build_seconds=build_seconds, recall_measured=False)), flush=True)
    except BaseException as exc:
        write_json(output / 'worker_stop.json', dict(status='STOP', reason=str(exc), completed_searches=len(reports),
            resources=budget.report() if hasattr(budget, 'elapsed') else None, recall_measured=False))
        raise


def run(output):
    output.mkdir(parents=True, exist_ok=False); began = time.monotonic()
    try:
        cfg = json.loads(CONFIG.read_text())
        generation = json.loads(verify(cfg['generation_source']).read_text())
        queries, refs = v38.input_rows(generation)
        launch = dict(**index_authority(), config=cfg, queries=queries[:cfg['crm_queries']], query_sources=refs,
            monotonic_start=began, truth_scope='NO_LABELS_OR_TRAINING_FILES',
            corpus_selection='FIRST32768_NATIVE_SITE_DOCS_UNCONDITIONED_ON_QUERY_OR_LABEL')
        for path in (Path(__file__), CONFIG, ROOT/'src/xgb_matcher/retrieval_impact_v68.py',
            ROOT/'src/xgb_matcher/retrieval_initialisms_v67.py', ROOT/'src/xgb_matcher/retrieval_sparse_query_v59.py',
            ROOT/'docs/retrieval_v68_impact_cost_probe.md'):
            launch['implementation'][str(path)] = record(path)
        write_json(output/'launch.json', launch)
        with (output/'worker.log').open('x') as log:
            subprocess.run([sys.executable, '-u', '-m', 'scripts.preflight_retrieval_v68_impact_cost', 'worker', '--output', str(output)],
                check=True, stdout=log, stderr=subprocess.STDOUT, timeout=90,
                env={**os.environ, 'OMP_NUM_THREADS':'2', 'OPENBLAS_NUM_THREADS':'2',
                     'VECLIB_MAXIMUM_THREADS':'2', 'RAYON_NUM_THREADS':'2'})
        print((output/'worker.log').read_text()[-6000:], flush=True)
    except BaseException as exc:
        write_json(output/'status.json', dict(status='STOP', reason=str(exc), seconds=time.monotonic()-began,
            recall_measured=False, truth_opened=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['run', 'worker']); parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(); (run if args.mode == 'run' else worker)(args.output)
