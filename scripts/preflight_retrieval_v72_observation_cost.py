"""Read official observation prefixes; group-direct100 parity/cost, no labels."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scripts import retrieval_v38_initial_top100 as v38
from scripts.retrieval_v56_exact_fields import index_authority
from src.xgb_matcher.retrieval_observation_v72 import Observation, ObservationIndex
from src.xgb_matcher.retrieval_v20_workbench import Budget, ROOT, record, verify, write_json

CONFIG = ROOT/'config/retrieval_v72_observation_cost.json'


def authority(source):
    folder = Path(source['folder'])
    manifest = json.loads(verify(dict(path=str(folder/'manifest.json'), sha256=source['manifest_sha256'])).read_text())
    if 'crm_labels' not in manifest['exclusions']:
        raise ValueError('Official source must exclude CRM labels')
    path = folder/'official_evidence.parquet'; st = path.stat(); pf = pq.ParquetFile(path)
    if st.st_size != source['size_bytes'] or pf.metadata.num_rows != source['rows']:
        raise ValueError('Official source size/count drift')
    with path.open('rb') as handle:
        handle.seek(-8, 2); tail = handle.read(8)
        length = int.from_bytes(tail[:4], 'little')
        if tail[4:] != b'PAR1' or length > 32*1024*1024:
            raise ValueError('Invalid/bloated Parquet footer')
        handle.seek(-8-length, 2); footer = handle.read(length+8)
    return dict(path=str(path), size_bytes=st.st_size, mtime_ns=st.st_mtime_ns,
                inode=st.st_ino, rows=pf.metadata.num_rows, row_groups=pf.metadata.num_row_groups,
                footer_sha256=hashlib.sha256(footer).hexdigest(), full_file_rehashed=False)


def worker(output):
    launch = json.loads((output/'launch.json').read_text()); cfg = launch['config']
    v38.verify_launch(launch); budget = Budget(cfg['resources']); reports = []
    pa.set_cpu_count(2)
    try:
        with budget:
            observations = []; sources = []; all_digest = hashlib.sha256(); began = time.monotonic()
            for spec, expected in zip(cfg['sources'], launch['source_authority']):
                if authority(spec) != expected:
                    raise ValueError('Official source changed before read')
                count = 0; counters = Counter(); source_digest = hashlib.sha256()
                for batch in pq.ParquetFile(expected['path']).iter_batches(batch_size=4096):
                    remaining = cfg['rows_per_source']-count
                    rows = batch.slice(0, min(len(batch), remaining)).to_pylist()
                    for row in rows:
                        digest = json.dumps(row, sort_keys=True, ensure_ascii=True).encode()+b'\n'
                        source_digest.update(digest); all_digest.update(digest)
                        o = Observation.from_row(row); observations.append(o)
                        counters['source:'+o.source] += 1; counters['subject:'+o.subject_kind] += 1
                        counters['with_names'] += bool(o.names); counters['with_addresses'] += bool(o.addresses)
                        counters['same_observation_name_address'] += bool(o.names and o.addresses)
                        counters['with_valid_from'] += bool(o.valid_from); counters['with_valid_to'] += bool(o.valid_to)
                        counters['with_observed_at'] += bool(o.observed_at)
                        counters['with_explicit_observation_id'] += bool(o.evidence_id)
                        count += 1
                    if count == cfg['rows_per_source']:
                        break
                if count != cfg['rows_per_source']:
                    raise ValueError('Independent prefix cardinality drift')
                sources.append(dict(path=expected['path'], rows=count, counters=dict(counters),
                                    read_prefix_sha256=source_digest.hexdigest()))
                print(json.dumps(dict(source=spec['folder'], count=count, counters=dict(counters))), flush=True)
            read_seconds = time.monotonic()-began; began = time.monotonic()
            index = ObservationIndex(observations); build_seconds = time.monotonic()-began
            common = sorted(((len(v), k[2:]) for k, v in index.postings.items() if k.startswith('N:')),
                            key=lambda x:(-x[0], x[1]))[:cfg['common_term_queries']]
            if len(common) != cfg['common_term_queries']:
                raise ValueError('Insufficient official nonempty cost sentinels')
            queries = [(r['query_id'], 'CRM_QUERY_ONLY', r) for r in launch['queries']]
            queries += [(f'common{i}', 'OFFICIAL_COST_SENTINEL_NOT_CRM', dict(crm_name=t)) for i, (_, t) in enumerate(common)]
            for repeat in range(cfg['repetitions']):
                for query_id, kind, row in queries:
                    results = {}; times = {}; trace = None
                    for method in (('indexed', 'oracle') if repeat%2 == 0 else ('oracle', 'indexed')):
                        began = time.monotonic()
                        if method == 'indexed':
                            results[method], trace = index.search(row)
                        else:
                            results[method] = index.exhaustive(row)
                        times[method] = time.monotonic()-began
                    if results['indexed'] != results['oracle']:
                        raise ValueError('Distinct SIRET top100 score/order drift')
                    reports.append(dict(query_id=query_id, kind=kind, repeat=repeat, times=times,
                        returned_count=len(results['indexed']), trace=trace, exact_score_order_parity=True))
            for spec, expected in zip(cfg['sources'], launch['source_authority']):
                if authority(spec) != expected:
                    raise ValueError('Source changed during probe')
        v38.verify_launch(launch)
        for ref in launch['query_sources']:
            verify(ref)
        summary = {}
        for kind in sorted({r['kind'] for r in reports}):
            selected = [r for r in reports if r['kind'] == kind]
            summary[kind] = dict(searches=len(selected), nonempty=sum(r['returned_count'] > 0 for r in selected),
                mean_scored_observations=float(np.mean([r['trace']['scored_observations'] for r in selected])),
                methods={m:dict(mean=float(np.mean([r['times'][m] for r in selected])),
                    p95=float(np.percentile([r['times'][m] for r in selected], 95))) for m in ('indexed', 'oracle')})
        write_json(output/'metrics.json', dict(sources=sources, read_prefix_sha256=all_digest.hexdigest(),
            input_observations=len(observations), site_observations=len(index.observations),
            distinct_site_keys=len({o.siret for o in index.observations}), parent_only_not_projected=index.parent_only_count,
            read_encode_seconds=read_seconds, build_seconds=build_seconds, searches=reports, summary=summary))
        write_json(output/'manifest.json', dict(status='SEALED_COST_ONLY_OBSERVATION_GROUP_TOP100_PARITY',
            launch=record(output/'launch.json'), metrics=record(output/'metrics.json'), resources=budget.report(),
            seconds=time.monotonic()-launch['monotonic_start'], recall_measured=False, truth_opened=False,
            no_candidates_written=True, no_training_performed=True, full_national_search_performed=False,
            ready_for_fold1=False, fold0_opened=False, fold1_opened=False, downstream_models_unchanged=True,
            limitations=['PHYSICAL_PREFIX_NOT_RANDOM_OR_NATIONAL', 'FIXED_COST_SCORE_NOT_LEARNED',
                'PARENT_ONLY_NOT_PROJECTED_TO_SITES', 'NO_GRAPH_XY_BAN_OR_LEGAL_PARENT_INHERITANCE',
                'DATES_PRESERVED_NOT_SCORED', 'ALL_RAM_PYTHON_NO_NATIONAL_SLO', 'FULL_SOURCE_FILES_NOT_REHASHED']))
        print(json.dumps(dict(summary=summary, resources=budget.report(), read_encode_seconds=read_seconds,
                              build_seconds=build_seconds)), flush=True)
    except BaseException as exc:
        write_json(output/'worker_stop.json', dict(status='STOP', reason=str(exc), completed_searches=len(reports),
            resources=budget.report() if hasattr(budget, 'elapsed') else None, recall_measured=False)); raise


def run(output):
    output.mkdir(parents=True, exist_ok=False); began = time.monotonic(); cfg = json.loads(CONFIG.read_text())
    try:
        generation = json.loads(verify(cfg['generation_source']).read_text()); rows, refs = v38.input_rows(generation)
        launch = dict(**index_authority(), config=cfg, queries=rows[:cfg['crm_queries']], query_sources=refs,
            source_authority=[authority(s) for s in cfg['sources']], monotonic_start=began,
            truth_scope='NO_LABELS_NO_TEACHER_NO_PREVIOUS_CANDIDATES', corpus_selection='FIXED_OFFICIAL_PHYSICAL_PREFIXES')
        for p in (Path(__file__), CONFIG, ROOT/'docs/retrieval_v72_observation_cost_probe.md',
                  ROOT/'src/xgb_matcher/retrieval_observation_v72.py', ROOT/'src/xgb_matcher/official_evidence.py'):
            launch['implementation'][str(p)] = record(p)
        write_json(output/'launch.json', launch)
        with (output/'worker.log').open('x') as log:
            subprocess.run([sys.executable, '-u', '-m', 'scripts.preflight_retrieval_v72_observation_cost',
                'worker', '--output', str(output)], check=True, stdout=log, stderr=subprocess.STDOUT,
                timeout=90, env={**os.environ, 'OMP_NUM_THREADS':'2', 'OPENBLAS_NUM_THREADS':'2',
                    'VECLIB_MAXIMUM_THREADS':'2', 'RAYON_NUM_THREADS':'2'})
        print((output/'worker.log').read_text()[-6000:], flush=True)
    except BaseException as exc:
        write_json(output/'status.json', dict(status='STOP', reason=str(exc), seconds=time.monotonic()-began,
                                             recall_measured=False, truth_opened=False)); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('mode', choices=['run', 'worker'])
    p.add_argument('--output', required=True, type=Path); a = p.parse_args()
    (run if a.mode == 'run' else worker)(a.output)
