"""Global historical SITE overlay and query-only initial100 pilot."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pyarrow.parquet as pq
import tantivy

from scripts import retrieval_v38_initial_top100 as v38
from scripts import retrieval_v42_indexed_metric as v42
from scripts import retrieval_v49_primary_probe as evaluator
from scripts.retrieval_v56_exact_fields import index_authority
from src.xgb_matcher.hierarchical_retrieval import TantivyBackend
from src.xgb_matcher.retrieval_history_site_v75 import document, grouped, schema, search
from src.xgb_matcher.retrieval_native_v57 import LEXICAL
from src.xgb_matcher.retrieval_v20_workbench import Budget, ROOT, record, verify, write_json

CONFIG = ROOT/'config/retrieval_v75_history_site.json'


def authority():
    a = index_authority()
    for p in (Path(__file__), CONFIG, ROOT/'docs/retrieval_v75_history_site_probe.md',
              ROOT/'src/xgb_matcher/retrieval_history_site_v75.py',
              ROOT/'scripts/retrieval_v40_initial_top100.py', ROOT/'scripts/retrieval_v49_primary_probe.py'):
        a['implementation'][str(p)] = record(p)
    return a


def sources(cfg):
    found = []
    for p in Path(cfg['projection_root']).glob('*/manifest.json'):
        m = json.loads(p.read_text()); path = p.parent/'historical_siret_identity.parquet'
        if (m['schema_version'] != 'sireto-sirene-temporal-v1' or m['contains_crm_labels']
            or m['contains_synthetic_rows'] or m['semantics']['historical_addresses_claimed']):
            raise ValueError('Historical projection scope drift')
        rows = pq.ParquetFile(path).metadata.num_rows
        if rows != m['counts']['historical_siret_identity']:
            raise ValueError('Projection source count drift')
        found.append(dict(prefix=m['siren_prefix'], manifest=record(p), parquet=record(path),
                          rows=rows, raw_inputs=m['inputs']))
    if (len(found) != cfg['expected_partitions'] or
        {s['prefix'] for s in found} != {f'{i:02d}' for i in range(100)} or
        sum(s['rows'] for s in found) != cfg['expected_observations'] or
        any(s['raw_inputs'] != found[0]['raw_inputs'] for s in found)):
        raise ValueError('Global prefix/source inventory drift; no subset build')
    return sorted(found, key=lambda s:s['prefix'])


def verify_sources(records):
    for s in records:
        verify(s['manifest']); verify(s['parquet'])


def history_files(folder):
    return {str(p.relative_to(folder)):record(p) for p in sorted((folder/'tantivy').rglob('*'))
            if p.is_file() and not p.name.endswith('.lock')}


def verify_history(ref):
    h = json.loads(verify(ref).read_text()); v38.verify_launch(json.loads(verify(h['launch']).read_text()))
    if h['status'] != 'SEALED_GLOBAL_HISTORICAL_SITE_SOURCE' or h['truth_opened']:
        raise ValueError('Historical source not sealed/no-label')
    for r in h['files'].values():
        verify(r)
    return h


def build(output):
    launch = json.loads((output/'build_launch.json').read_text()); v38.verify_launch(launch)
    budget = Budget(launch['config']['resources'])
    try:
        with budget:
            verify_sources(launch['sources'])
            folder = output/'history'; folder.mkdir(exist_ok=False)
            (folder/'tantivy').mkdir()
            index = tantivy.Index(schema(), path=str(folder/'tantivy'))
            writer = index.writer(heap_size=128000000, num_threads=2)
            count = observations = 0
            for source in launch['sources']:
                rows = pq.read_table(verify(source['parquet'])).to_pylist()
                if len(rows) != source['rows']:
                    raise ValueError('Partial official projection read')
                groups = grouped(rows, source['prefix'])
                for records in groups.values():
                    writer.add_document(document(records)); count += 1
                observations += len(rows)
            if observations != launch['config']['expected_observations'] or not count:
                raise ValueError('Empty/partial historical index forbidden')
            writer.commit(); writer.wait_merging_threads(); index.reload()
            if index.searcher().num_docs != count:
                raise ValueError('Historical live index count mismatch')
            verify_sources(launch['sources']); files = history_files(folder)
        v38.verify_launch(launch)
        write_json(folder/'manifest.json', dict(status='SEALED_GLOBAL_HISTORICAL_SITE_SOURCE',
            launch=record(output/'build_launch.json'), files=files, documents=count,
            observations=observations, partitions=100, resources=budget.report(), truth_opened=False,
            historical_addresses_claimed=False, contains_crm_labels=False, historical_parent_names_included=False,
            succession_paths_included=False, source_selection='ALL_PREEXISTING_OFFICIAL_PREFIXES_NO_QUERY_INPUT'))
        print(json.dumps(dict(build_documents=count, observations=observations, resources=budget.report())), flush=True)
    except BaseException as exc:
        write_json(output/'build_stop.json', dict(status='STOP', reason=str(exc), truth_opened=False,
            resources=budget.report() if hasattr(budget, 'elapsed') else None)); raise


def worker(output, start):
    launch = json.loads((output/'launch.json').read_text()); v38.verify_launch(launch)
    h = verify_history(launch['history']); models = json.loads(verify(launch['models']).read_text())
    budget = Budget(launch['resources'])
    with budget:
        base = TantivyBackend(v38.INDEX)
        history = tantivy.Index.open(str(Path(launch['history']['path']).parent/'tantivy'))
        if base.searcher.num_docs != launch['index_documents'] or history.searcher().num_docs != h['documents']:
            raise ValueError('Live source count drift')
        rows = []
        for raw in launch['queries'][start:start+8]:
            m = models[str(raw['fold'])]
            if m['features'] != list(LEXICAL) or m['heldout_fold'] != raw['fold'] or raw['fold'] in m['training_folds']:
                raise ValueError('OOF model/feature scope drift')
            began = time.monotonic(); found, trace = search(base, history, v42.as_query(raw), m['weights'])
            seconds = time.monotonic()-began
            if seconds > launch['max_query_search_seconds']:
                raise ValueError('STOP_QUERY_COST')
            rows.append(dict(query_id=raw['query_id'], fold=raw['fold'], results=dict(primary=found),
                             seconds=dict(primary=seconds), primary_trace=trace))
    v38.verify_launch(launch); verify_history(launch['history'])
    write_json(output/f'batch{start:03d}.json', dict(rows=rows, resources=budget.report()))


def supervised(output, mode, timeout, start=None):
    args = [sys.executable, '-u', '-m', 'scripts.retrieval_v75_history_site', mode, '--output', str(output)]
    if start is not None:
        args += ['--start', str(start)]
    with (output/f'{mode}{start if start is not None else ""}.log').open('x') as log:
        subprocess.run(args, check=True, timeout=timeout, stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, 'OMP_NUM_THREADS':'2', 'OPENBLAS_NUM_THREADS':'2',
                 'VECLIB_MAXIMUM_THREADS':'2', 'RAYON_NUM_THREADS':'2'})


def run(output):
    output.mkdir(parents=True, exist_ok=False); began = time.monotonic(); cfg = json.loads(CONFIG.read_text())
    try:
        write_json(output/'build_launch.json', dict(**authority(), config=cfg, sources=sources(cfg),
            truth_scope='NO_CRM_NO_LABELS_NO_TEACHER_NO_PREVIOUS_CANDIDATES'))
        supervised(output, 'build', 90)
        print((output/'build.log').read_text()[-1500:], flush=True)
        href = record(output/'history/manifest.json'); verify_history(href)
        trained = json.loads(verify(cfg['trained']).read_text()); v38.verify_launch(trained)
        if trained['status'] != 'SEALED_OOF_NATIVE_PROOF_MODELS' or trained['recall_measured']:
            raise ValueError('Frozen training model scope drift')
        models = json.loads(verify(trained['models']).read_text())
        queries, refs = v42.input_rows(v38.load(v38.GEN, v38.GEN_SHA), 4)
        if any(set(m['query_ids']) & {q['query_id'] for q in queries} for m in models.values()):
            raise ValueError('Training/evaluation query overlap')
        launch = dict(**authority(), models=trained['models'], model_manifest=cfg['trained'],
            queries=queries, query_sources=refs, history=href, resources=cfg['resources'],
            max_query_search_seconds=cfg['max_query_search_seconds'], truth_opened=False,
            model_parameters_changed=False, fixed_reservoir=None, development_panel_reused=True)
        write_json(output/'launch.json', launch); rows = []; batches = []
        for start in range(0, 128, 8):
            supervised(output, 'worker', min(90, max(1, cfg['controller_seconds']-(time.monotonic()-began))), start)
            p = output/f'batch{start:03d}.json'; b = json.loads(p.read_text())
            if (len(b['rows']) != 8 or not b['resources']['pass_resource_policy'] or
                [(r['query_id'],r['fold']) for r in b['rows']] != [(r['query_id'],r['fold']) for r in queries[start:start+8]]):
                raise ValueError('Generation batch scope/resources drift')
            rows.extend(b['rows']); batches.append(record(p)); print(json.dumps(dict(generated=len(rows))), flush=True)
            if len(rows) == 16:
                mean = sum(r['seconds']['primary'] for r in rows)/16
                write_json(output/'cost_gate.json', dict(mean_seconds=mean, passed=mean<=cfg['primary_mean16_seconds'], truth_opened=False))
                if mean > cfg['primary_mean16_seconds']:
                    raise ValueError('STOP_COST_PREFLIGHT')
        v38.verify_launch(launch); verify_history(href)
        if len(rows) != 128 or len({r['query_id'] for r in rows}) != 128:
            raise ValueError('Complete population required')
        write_json(output/'manifest.json', dict(status='SEALED_INITIAL_TOP100', queries=128,
            batches=batches, launch=record(output/'launch.json'), truth_opened=False,
            seconds=time.monotonic()-began, model_used=True, ready_for_fold1=False,
            metric_family=cfg['metric_family'], full_query_population=False, full_lexical_corpus=True,
            old_candidates_opened=False, historical_parent_names_included=False, successions_included=False))
    except BaseException as exc:
        write_json(output/'status.json', dict(status='STOP', reason=str(exc), truth_opened=False,
            ready_for_fold1=False, seconds=time.monotonic()-began)); raise


def evaluate(output):
    cfg = json.loads(CONFIG.read_text()); launch = json.loads((output/'launch.json').read_text())
    verify_history(launch['history'])
    old = evaluator.BASELINE
    try:
        evaluator.BASELINE = cfg['baseline']; evaluator.evaluate(output)
    finally:
        evaluator.BASELINE = old
    write_json(output/'evaluation_scope.json', dict(evaluation=record(output/'evaluation.json'),
        comparator='SEALED_V57_NOT_V42', history=launch['history'],
        source='GLOBAL_HISTORY_SITE_NAMES_PLUS_CURRENT_SCAFFOLD_NOT_HISTORICAL_ADDRESS',
        historical_parent_names_included=False, successions_included=False,
        scope='128_REUSED_DEVELOPMENT_QUERIES_NOT_FINAL_OOF_GATE'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('mode', choices=['run','build','worker','evaluate'])
    p.add_argument('--output', type=Path, required=True); p.add_argument('--start', type=int); a = p.parse_args()
    if a.mode == 'worker':
        if a.start not in range(0,128,8):
            p.error('Invalid worker start')
        worker(a.output, a.start)
    else:
        {'run':run, 'build':build, 'evaluate':evaluate}[a.mode](a.output)
