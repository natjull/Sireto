"""Global official LEGAL historical equality join and initial100 pilot."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import tantivy

from scripts import retrieval_v38_initial_top100 as v38
from scripts import retrieval_v42_indexed_metric as v42
from scripts import retrieval_v49_primary_probe as evaluator
from scripts import retrieval_v75_history_site as v75
from src.xgb_matcher.hierarchical_retrieval import TantivyBackend
from src.xgb_matcher.retrieval_history_parent_v76 import alias_pairs, search
from src.xgb_matcher.retrieval_native_v57 import LEXICAL
from src.xgb_matcher.retrieval_v20_workbench import Budget, ROOT, record, verify, write_json

CONFIG = ROOT/'config/retrieval_v76_history_parent.json'


def authority():
    a = v75.authority()
    for p in (Path(__file__), CONFIG, ROOT/'docs/retrieval_v76_history_parent_probe.md',
              ROOT/'src/xgb_matcher/retrieval_history_parent_v76.py'):
        a['implementation'][str(p)] = record(p)
    return a


def sources(cfg):
    records = []
    for p in Path(cfg['projection_root']).glob('*/manifest.json'):
        m = json.loads(p.read_text()); path = p.parent/'historical_siren_identity.parquet'
        rows = pq.ParquetFile(path).metadata.num_rows
        if (m['schema_version'] != 'sireto-sirene-temporal-v1' or m['contains_crm_labels'] or
            m['contains_synthetic_rows'] or m['semantics']['historical_addresses_claimed'] or
            rows != m['counts']['historical_siren_identity']):
            raise ValueError('Official LEGAL history source drift')
        records.append(dict(prefix=m['siren_prefix'], manifest=record(p), parquet=record(path),
                            rows=rows, raw_inputs=m['inputs']))
    if (len(records) != cfg['expected_partitions'] or
        {r['prefix'] for r in records} != {f'{i:02d}' for i in range(100)} or
        sum(r['rows'] for r in records) != cfg['expected_observations'] or
        any(r['raw_inputs'] != records[0]['raw_inputs'] for r in records)):
        raise ValueError('Full official prefix inventory required')
    return sorted(records, key=lambda r:r['prefix'])


def verify_aliases(ref):
    m = json.loads(verify(ref).read_text()); v38.verify_launch(json.loads(verify(m['launch']).read_text()))
    if m['status'] != 'SEALED_GLOBAL_LEGAL_ALIAS_EQUALITY_SOURCE' or m['truth_opened']:
        raise ValueError('Alias source must be sealed and label-free')
    verify(m['database'])
    return m


def build(output):
    launch = json.loads((output/'build_launch.json').read_text()); v38.verify_launch(launch)
    budget = Budget(launch['config']['resources'])
    try:
        with budget:
            v75.verify_sources(launch['sources'])
            c = duckdb.connect(); c.execute("SET threads=2"); c.execute("SET memory_limit='2GB'")
            paths = [r['parquet']['path'] for r in launch['sources']]
            counts = c.execute('SELECT count(*),count(DISTINCT siren),count(DISTINCT (siren,normalized_name)) FROM read_parquet(?)', [paths]).fetchone()
            if counts[0] != launch['config']['expected_observations']:
                raise ValueError('Full projection read required')
            records = c.execute('SELECT DISTINCT siren,normalized_name FROM read_parquet(?)', [paths]).fetch_arrow_table().to_pylist()
            c.close(); pairs = alias_pairs(records); del records
            table = pa.table(dict(name=[r[0] for r in pairs], siren=[r[1] for r in pairs])); del pairs
            destination = output/'aliases.duckdb'
            c = duckdb.connect(str(destination)); c.execute("SET threads=2"); c.execute("SET memory_limit='2GB'")
            c.register('official_pairs', table)
            c.execute('CREATE TABLE aliases AS SELECT * FROM official_pairs ORDER BY name,siren')
            c.execute('CREATE INDEX alias_equality ON aliases(name)')
            actual = c.execute('SELECT count(*),count(DISTINCT siren) FROM aliases').fetchone()
            if actual[0] != table.num_rows or actual[1] > counts[1] or not actual[0]:
                raise ValueError('Normalized membership drift')
            c.unregister('official_pairs'); c.execute('CHECKPOINT'); c.close(); del table
            if Path(str(destination)+'.wal').exists():
                raise ValueError('Uncheckpointed database')
            v75.verify_sources(launch['sources'])
        v38.verify_launch(launch)
        write_json(output/'aliases_manifest.json', dict(status='SEALED_GLOBAL_LEGAL_ALIAS_EQUALITY_SOURCE',
            launch=record(output/'build_launch.json'), database=record(destination),
            observations=counts[0], parents=counts[1], original_unique_alias_parent_pairs=counts[2],
            normalized_unique_alias_parent_pairs=actual[0], partitions=100, resources=budget.report(),
            parents_with_nonempty_signal=actual[1], parents_without_nonempty_signal=counts[1]-actual[1],
            truth_opened=False, contains_crm_labels=False, contains_synthetic_rows=False,
            source_selection='ALL_PREEXISTING_OFFICIAL_PREFIXES_NO_QUERY_INPUT',
            historical_addresses_claimed=False, historical_alias_current_site_coexistence_proven=False))
        print(json.dumps(dict(counts=counts, normalized_pairs=actual[0], resources=budget.report())), flush=True)
    except BaseException as exc:
        write_json(output/'build_stop.json', dict(status='STOP',reason=str(exc),truth_opened=False)); raise


def worker(output, start):
    launch = json.loads((output/'launch.json').read_text()); v38.verify_launch(launch)
    a = verify_aliases(launch['aliases']); h = v75.verify_history(launch['history'])
    models = json.loads(verify(launch['models']).read_text()); budget = Budget(launch['resources'])
    with budget:
        base = TantivyBackend(v38.INDEX)
        history = tantivy.Index.open(str(Path(launch['history']['path']).parent/'tantivy'))
        aliases = duckdb.connect(a['database']['path'], read_only=True); aliases.execute('SET threads=2')
        if base.searcher.num_docs != launch['index_documents'] or history.searcher().num_docs != h['documents']:
            raise ValueError('National live source cardinality drift')
        if aliases.execute('SELECT count(*) FROM aliases').fetchone()[0] != a['normalized_unique_alias_parent_pairs']:
            raise ValueError('Alias database cardinality drift')
        rows = []
        for raw in launch['queries'][start:start+8]:
            m = models[str(raw['fold'])]
            if m['features'] != list(LEXICAL) or m['heldout_fold'] != raw['fold'] or raw['fold'] in m['training_folds']:
                raise ValueError('OOF metric drift')
            began = time.monotonic(); found, trace = search(base,history,aliases,v42.as_query(raw),m['weights'])
            seconds = time.monotonic()-began
            if seconds > launch['max_query_search_seconds']:
                raise ValueError('STOP_QUERY_COST')
            rows.append(dict(query_id=raw['query_id'],fold=raw['fold'],results=dict(primary=found),
                             seconds=dict(primary=seconds),primary_trace=trace))
        aliases.close()
    v38.verify_launch(launch); verify_aliases(launch['aliases']); v75.verify_history(launch['history'])
    write_json(output/f'batch{start:03d}.json',dict(rows=rows,resources=budget.report()))


def supervised(output, mode, timeout, start=None):
    args = [sys.executable,'-u','-m','scripts.retrieval_v76_history_parent',mode,'--output',str(output)]
    if start is not None:
        args += ['--start',str(start)]
    with (output/f'{mode}{start if start is not None else ""}.log').open('x') as log:
        subprocess.run(args,check=True,timeout=timeout,stdout=log,stderr=subprocess.STDOUT,
            env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2',
                 'VECLIB_MAXIMUM_THREADS':'2','RAYON_NUM_THREADS':'2'})


def run(output):
    output.mkdir(parents=True,exist_ok=False); cfg=json.loads(CONFIG.read_text())
    try:
        write_json(output/'build_launch.json',dict(**authority(),config=cfg,sources=sources(cfg),
            truth_scope='NO_CRM_NO_LABELS_NO_TEACHER_NO_PREVIOUS_CANDIDATES'))
        supervised(output,'build',cfg['build_supervisor_seconds']); print((output/'build.log').read_text()[-1500:],flush=True)
        a = record(output/'aliases_manifest.json'); verify_aliases(a); v75.verify_history(cfg['history'])
        trained=json.loads(verify(cfg['trained']).read_text()); v38.verify_launch(trained)
        if trained['status']!='SEALED_OOF_NATIVE_PROOF_MODELS' or trained['recall_measured']:
            raise ValueError('Frozen training model scope drift')
        models=json.loads(verify(trained['models']).read_text())
        queries,refs=v42.input_rows(v38.load(v38.GEN,v38.GEN_SHA),4)
        if any(set(m['query_ids']) & {q['query_id'] for q in queries} for m in models.values()):
            raise ValueError('Training/evaluation query overlap')
        launch=dict(**authority(),models=trained['models'],model_manifest=cfg['trained'],
            queries=queries,query_sources=refs,aliases=a,history=cfg['history'],resources=cfg['resources'],
            max_query_search_seconds=cfg['max_query_search_seconds'],truth_opened=False,
            model_parameters_changed=False,fixed_reservoir=None,parent_limit=None,development_panel_reused=True)
        write_json(output/'launch.json',launch); rows=[]; batches=[]; began=time.monotonic()
        for start in range(0,128,8):
            supervised(output,'worker',min(90,max(1,cfg['generation_controller_seconds']-(time.monotonic()-began))),start)
            p=output/f'batch{start:03d}.json'; b=json.loads(p.read_text())
            if (len(b['rows'])!=8 or not b['resources']['pass_resource_policy'] or
                [(r['query_id'],r['fold']) for r in b['rows']]!=[(q['query_id'],q['fold']) for q in queries[start:start+8]]):
                raise ValueError('Batch population/resource drift')
            rows.extend(b['rows']); batches.append(record(p)); print(json.dumps(dict(generated=len(rows))),flush=True)
            if len(rows)==16:
                mean=sum(r['seconds']['primary'] for r in rows)/16
                write_json(output/'cost_gate.json',dict(mean_seconds=mean,passed=mean<=cfg['primary_mean16_seconds'],truth_opened=False))
                if mean>cfg['primary_mean16_seconds']:
                    raise ValueError('STOP_COST_PREFLIGHT')
        v38.verify_launch(launch); verify_aliases(a); v75.verify_history(cfg['history'])
        if len(rows)!=128 or len({r['query_id'] for r in rows})!=128:
            raise ValueError('Complete population required')
        write_json(output/'manifest.json',dict(status='SEALED_INITIAL_TOP100',queries=128,batches=batches,
            launch=record(output/'launch.json'),truth_opened=False,seconds=time.monotonic()-began,
            model_used=True,ready_for_fold1=False,metric_family=cfg['metric_family'],full_query_population=False,
            full_lexical_corpus=True,old_candidates_opened=False,historical_parent_names_included=True,
            historical_parent_match='EXACT_ONLY',successions_included=False))
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),truth_opened=False,ready_for_fold1=False)); raise


def evaluate(output):
    cfg=json.loads(CONFIG.read_text()); launch=json.loads((output/'launch.json').read_text())
    verify_aliases(launch['aliases']); v75.verify_history(launch['history']); old=evaluator.BASELINE
    try:
        evaluator.BASELINE=cfg['baseline']; evaluator.evaluate(output)
    finally:
        evaluator.BASELINE=old
    write_json(output/'evaluation_scope.json',dict(evaluation=record(output/'evaluation.json'),
        comparator='SEALED_V75_NOT_V42',aliases=launch['aliases'],history=launch['history'],
        historical_parent_match='EXACT_ONLY_NO_RANKED_PARENT_POOL',successions_included=False,
        scope='128_REUSED_DEVELOPMENT_QUERIES_NOT_FINAL_OOF_GATE'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=['run','build','worker','evaluate'])
    p.add_argument('--output',type=Path,required=True); p.add_argument('--start',type=int); a=p.parse_args()
    if a.mode=='worker':
        if a.start not in range(0,128,8):
            p.error('Invalid worker start')
        worker(a.output,a.start)
    else:
        {'run':run,'build':build,'evaluate':evaluate}[a.mode](a.output)
