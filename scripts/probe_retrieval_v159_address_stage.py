"""One actual V154 dictionary bucket, exact V159 staging under unchanged 1GiB.

Resource probe ONLY, not full reverse storage or a retrieval/recall result.
The stopped V154 reverse stage is never imported or reclassified as success.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import duckdb

from scripts import compile_retrieval_v154_address_store as original
from src.xgb_matcher.retrieval_address_stage_v159 import stage
from src.xgb_matcher.retrieval_append_catalog_v141 import AppendKeys
from src.xgb_matcher.retrieval_memory_policy_v153 import Budget, LIMITS, quiet, validate_report
from src.xgb_matcher.retrieval_proof_bridge_v137 import PROTECTED
from src.xgb_matcher.retrieval_supervisor_v157 import supervise
from src.xgb_matcher.retrieval_v20_workbench import ROOT, record, verify, write_json


def source(reference, bucket):
    path = verify(reference)
    root = path.parent
    if path != root/'dictionary.json':
        raise ValueError('Exact complete dictionary authority required')
    launch, plan = original.load(root)
    dictionary = original.authority(root, 'dictionary', 'SEALED_V154_COMPLETE_ALL_ADDRESS_DICTIONARY')
    if (not isinstance(bucket, int) or isinstance(bucket, bool)
            or not 0 <= bucket < launch['config']['dictionary_buckets']):
        raise ValueError('Exact original physical bucket required')
    # Only a SEALED dictionary phase may supply the cost probe, NOT the
    # orphan/STOP reverse stage, even if its staging files are present.
    phase = original.phase_record(root, 'dictionary', bucket)
    b = dictionary['buckets'][bucket]
    if (b['bucket'] != bucket or b['artifacts'] != phase['artifacts']
            or b['facts'] != phase['stats']['facts'] or b['aliases'] != phase['stats']['addresses']):
        raise ValueError('Whole original successful dictionary phase required')
    return launch, plan, b


def worker(output):
    launch = json.loads((output/'launch.json').read_text())
    if (launch['resources'] != LIMITS or any(launch.get(k) is not False for k in PROTECTED)
            or launch['complete_retrieval_index'] is not False
            or launch['only_single_real_physical_bucket'] is not True):
        raise ValueError('Protected resource-only launch and unchanged limits required')
    for ref in launch['implementation'].values():
        verify(ref)
    budget = None
    c = None
    keys = None
    try:
        with Budget(LIMITS) as budget:
            old, plan, b = source(launch['dictionary'], launch['bucket'])
            cfg = old['config']
            keys = AppendKeys(plan['catalog'])
            c = duckdb.connect()
            c.execute('SET threads=1')
            c.execute('SET memory_limit=?', [cfg['duckdb_memory']])
            c.execute('SET temp_directory=?', [str(output/'spill')])
            c.execute('SET max_temp_directory_size=?', [cfg['spill_limit']])
            stats = stage(c, verify(b['artifacts']['facts.parquet']), output, keys, b['bucket'],
                          b['alias_start'], b['aliases'], b['facts'], cfg['site_quantum'], 16)
            artifacts = [record(p) for p in map(Path, stats.pop('files'))]
            c.close()
            c = None
            source(launch['dictionary'], launch['bucket'])
        validate_report(budget.report())
        write_json(output/'manifest.json', dict(status='SEALED_V159_SINGLE_REAL_BUCKET_RESOURCE_PROBE',
            launch=record(output/'launch.json'), source_dictionary=launch['dictionary'], bucket=launch['bucket'],
            full_namespace_sites=plan['sites'], stats=stats, artifacts=artifacts, resources=budget.report(),
            complete_reverse_store=False, complete_retrieval_index=False, ready_for_fold1=False,
            **{k: False for k in PROTECTED}))
        print(json.dumps(dict(stats=stats, resources=budget.report())), flush=True)
    except BaseException as exc:
        if c is not None:
            c.close()
        write_json(output/'worker_stop.json', dict(status='STOP', reason=str(exc),
            resources=budget.report() if budget is not None and hasattr(budget, 'elapsed') else None,
            **{k: False for k in PROTECTED}))
        raise
    finally:
        if keys is not None:
            for holder in (keys, keys.previous):
                for name in ('site', 'parent', 'extra_site', 'extra_parent', 'new_site', 'new_parent'):
                    a = getattr(holder, name, None)
                    if hasattr(a, '_mmap'):
                        a._mmap.close()


def run(output, reference, bucket):
    expected = Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v159_address_stage')
    if output.parent.resolve() != expected.resolve() or output.exists():
        raise ValueError('Fresh immutable CATNAT resource-probe root required')
    old, _, _ = source(reference, bucket)
    gate = quiet(30, 50, 55)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'preflight.json', gate)
    implementation = dict(old['implementation'])
    for path in (Path(__file__), ROOT/'src/xgb_matcher/retrieval_address_stage_v159.py',
                 ROOT/'src/xgb_matcher/retrieval_supervisor_v157.py',
                 ROOT/'src/xgb_matcher/retrieval_memory_policy_v153.py'):
        implementation[str(path)] = record(path)
    write_json(output/'launch.json', dict(dictionary=reference, bucket=bucket, implementation=implementation,
        resources=LIMITS, only_single_real_physical_bucket=True, complete_retrieval_index=False,
        **{k: False for k in PROTECTED}))
    try:
        with (output/'worker.log').open('x') as log:
            result = supervise([sys.executable, '-u', '-m', 'scripts.probe_retrieval_v159_address_stage',
                                'worker', '--output', str(output)], log,
                               {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
        write_json(output/'supervision.json', result)
        manifest = json.loads((output/'manifest.json').read_text())
        validate_report(manifest['resources'])
        print(json.dumps(dict(manifest=record(output/'manifest.json'), resources=manifest['resources'],
                              supervision=result)), flush=True)
    except BaseException as exc:
        write_json(output/'status.json', dict(status='STOP', reason=str(exc), **{k: False for k in PROTECTED}))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['run', 'worker'])
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--dictionary', type=Path)
    parser.add_argument('--sha256')
    parser.add_argument('--bucket', default=0, type=int)
    a = parser.parse_args()
    if a.mode == 'worker':
        worker(a.output)
    else:
        if a.dictionary is None or a.sha256 is None:
            parser.error('Exact complete original dictionary and its SHA required')
        run(a.output, dict(path=str(a.dictionary.resolve()), sha256=a.sha256), a.bucket)
