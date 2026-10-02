"""Bounded explicit owner binding audit and group-top100 parity; no GT."""
import argparse
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
from scripts.preflight_retrieval_v72_observation_cost import authority
from scripts.retrieval_v56_exact_fields import index_authority
from src.xgb_matcher.retrieval_observation_v72 import Observation, ObservationIndex
from src.xgb_matcher.retrieval_owner_observation_v73 import compose
from src.xgb_matcher.retrieval_v20_workbench import Budget, ROOT, record, verify, write_json

CONFIG = ROOT/'config/retrieval_v73_owner_observation.json'


def worker(output):
    launch = json.loads((output/'launch.json').read_text()); cfg = launch['config']
    v38.verify_launch(launch); budget = Budget(cfg['resources']); reports = []; pa.set_cpu_count(2)
    try:
        with budget:
            verify(cfg['structure_source'])
            if authority(launch['source_spec']) != launch['source_authority']:
                raise ValueError('Source changed before scan')
            rows = []; digest = hashlib.sha256(); began = time.monotonic()
            for batch in pq.ParquetFile(launch['source_authority']['path']).iter_batches(batch_size=4096):
                rows.extend(batch.slice(0, min(len(batch), cfg['rows']-len(rows))).to_pylist())
                if len(rows) == cfg['rows']:
                    break
            if len(rows) != cfg['rows']:
                raise ValueError('Official prefix denominator drift')
            for row in rows:
                digest.update(json.dumps(row, sort_keys=True, ensure_ascii=True).encode()+b'\n')
            before = [Observation.from_row(r) for r in rows]
            after, bindings, counts = compose(rows); read_compose_seconds = time.monotonic()-began
            began = time.monotonic(); indexes = dict(raw=ObservationIndex(before), owner_bound=ObservationIndex(after))
            build_seconds = time.monotonic()-began
            common = sorted(((len(v), k[2:]) for k, v in indexes['owner_bound'].postings.items() if k.startswith('N:')),
                key=lambda x:(-x[0], x[1]))[:cfg['common_term_queries']]
            if len(common) != cfg['common_term_queries']:
                raise ValueError('Insufficient nonempty cost sentinels')
            queries = [(r['query_id'], 'CRM_QUERY_ONLY', r) for r in launch['queries']]
            queries += [(f'common{i}', 'OFFICIAL_COST_SENTINEL_NOT_CRM', dict(crm_name=t)) for i, (_, t) in enumerate(common)]
            for query_id, kind, row in queries:
                for recipe, index in indexes.items():
                    began = time.monotonic(); indexed, trace = index.search(row); indexed_seconds = time.monotonic()-began
                    began = time.monotonic(); oracle = index.exhaustive(row); oracle_seconds = time.monotonic()-began
                    if indexed != oracle:
                        raise ValueError('Exact distinct100 parity drift')
                    reports.append(dict(query_id=query_id, kind=kind, recipe=recipe, returned_count=len(indexed),
                        indexed_seconds=indexed_seconds, oracle_seconds=oracle_seconds, trace=trace, exact_score_order_parity=True))
            if authority(launch['source_spec']) != launch['source_authority']:
                raise ValueError('Source changed during probe')
        v38.verify_launch(launch)
        for ref in launch['query_sources']:
            verify(ref)
        write_json(output/'bindings.json', bindings)
        summary = {recipe:dict(nonempty=sum(r['returned_count']>0 for r in reports if r['recipe']==recipe),
            indexed_mean=float(np.mean([r['indexed_seconds'] for r in reports if r['recipe']==recipe])),
            indexed_p95=float(np.percentile([r['indexed_seconds'] for r in reports if r['recipe']==recipe],95)))
            for recipe in indexes}
        write_json(output/'metrics.json', dict(counts=counts, read_prefix_sha256=digest.hexdigest(), searches=reports,
            summary=summary, source_rows=len(rows), site_observations=len(indexes['raw'].observations),
            read_compose_seconds=read_compose_seconds, build_seconds=build_seconds))
        write_json(output/'manifest.json', dict(status='SEALED_STRUCTURE_COST_ONLY_EXPLICIT_OWNER_BINDING',
            launch=record(output/'launch.json'), bindings=record(output/'bindings.json'), metrics=record(output/'metrics.json'),
            resources=budget.report(), seconds=time.monotonic()-launch['monotonic_start'],
            recall_measured=False, truth_opened=False, no_candidates_written=True, no_training_performed=True,
            full_national_search_performed=False, ready_for_fold1=False, fold0_opened=False, fold1_opened=False,
            downstream_models_unchanged=True, limitations=['PHYSICAL_RNE_PREFIX_NOT_NATIONAL',
                'OWNER_SNAPSHOT_BINDING_NOT_TEMPORAL_EQUIVALENCE', 'NO_BODACC_PROJECTION_GRAPH_XY_BAN',
                'FIXED_COST_SCORE_NO_QUALITY_OR_NATIONAL_SLO', 'FULL_SOURCE_FILE_NOT_REHASHED']))
        print(json.dumps(dict(counts=counts, summary=summary, resources=budget.report(),
            read_compose_seconds=read_compose_seconds, build_seconds=build_seconds)), flush=True)
    except BaseException as exc:
        write_json(output/'worker_stop.json', dict(status='STOP', reason=str(exc), completed_searches=len(reports),
            resources=budget.report() if hasattr(budget, 'elapsed') else None, recall_measured=False)); raise


def run(output):
    output.mkdir(parents=True, exist_ok=False); began = time.monotonic(); cfg = json.loads(CONFIG.read_text())
    try:
        parent = json.loads(verify(cfg['structure_source']).read_text())
        pl = json.loads(verify(parent['launch']).read_text()); v38.verify_launch(pl)
        if parent['truth_opened'] or parent['recall_measured']:
            raise ValueError('Not the no-label official structure source')
        launch = dict(**index_authority(), config=cfg, monotonic_start=began, query_sources=pl['query_sources'],
            queries=pl['queries'][:cfg['crm_queries']], source_spec=pl['config']['sources'][0],
            source_authority=pl['source_authority'][0], truth_scope='NO_LABELS_NO_TEACHER_NO_PREVIOUS_CANDIDATES')
        for p in (Path(__file__), CONFIG, ROOT/'docs/retrieval_v73_owner_observation_probe.md',
            ROOT/'src/xgb_matcher/retrieval_owner_observation_v73.py', ROOT/'src/xgb_matcher/retrieval_observation_v72.py',
            ROOT/'scripts/preflight_retrieval_v72_observation_cost.py', ROOT/'src/xgb_matcher/official_evidence.py'):
            launch['implementation'][str(p)] = record(p)
        write_json(output/'launch.json', launch)
        with (output/'worker.log').open('x') as log:
            subprocess.run([sys.executable, '-u', '-m', 'scripts.preflight_retrieval_v73_owner_observation',
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
