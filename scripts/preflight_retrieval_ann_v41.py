"""Read-only cost probe on existing national ANN, NOT SIRET recall validation.

The historical index contains SIREN vectors. It is a scale/cost reference only:
neither its entity coverage nor its semantic quality validates a future site
index. No label, CRM query, identifier map or downstream model is opened.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from src.xgb_matcher.retrieval_v20_workbench import Budget, ROOT, record, verify, write_json

INDEX = Path('/Volumes/CATNAT_DATA/SIRETO_V9/dense/global_siren_minilm867_5c17354c_2d74b2b')
INDEX_MANIFEST_SHA = '10eee7dba997db6d25083b2d0b6c3b279ef777f72bdf79bedfdb11dd48cf64f9'
ENCODING = Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v41_learned_initial/cost_20261002_a/manifest.json')
ENCODING_SHA = '18e84e901f034867116d38e6da94b17217505ffeaee9286ee4fa75b758c7d09f'
RESOURCES = dict(max_seconds=90, max_rss_bytes=4*1024**3, max_swapout_pages_delta=0)


def worker(output):
    launch = json.loads((output/'launch.json').read_text())
    for ref in (launch['vectors'], launch['index']):
        verify(ref)
    import faiss
    faiss.omp_set_num_threads(2)
    with Budget(RESOURCES) as budget:
        started = time.monotonic()
        index = faiss.read_index(launch['index']['path'])
        load_seconds = time.monotonic()-started
        if index.ntotal != launch['index_entities'] or index.d != launch['dimension']:
            raise ValueError('Index cardinality/dimension drift')
        index.nprobe = launch['nprobe']
        queries = np.load(launch['vectors']['path'], allow_pickle=False)[:16]
        if queries.shape != (16, index.d):
            raise ValueError('Vector shape drift')
        seconds, hit_counts = [], []
        for query in queries:
            began = time.monotonic()
            scores, offsets = index.search(query[None], 100)
            seconds.append(time.monotonic()-began)
            valid = offsets[0] >= 0
            if not np.isfinite(scores[0, valid]).all() or len(set(offsets[0, valid])) != valid.sum():
                raise ValueError('Invalid native ANN output')
            hit_counts.append(int(valid.sum()))
    write_json(output/'worker.json', dict(
        rows=16, index_entities=index.ntotal, dimension=index.d, nprobe=index.nprobe,
        index_type=type(index).__name__, collector_limit=100, valid_hit_counts=hit_counts,
        load_seconds=load_seconds, first_query_seconds=seconds[0],
        subsequent_query_p50=float(np.quantile(seconds[1:], .5)),
        subsequent_query_p95=float(np.quantile(seconds[1:], .95)), seconds=seconds,
        resources=budget.report(), truth_opened=False, recall_measured=False,
        site_retrieval_validated=False, cold_cache_controlled=False))


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    try:
        source = json.loads(verify(dict(path=str(ENCODING), sha256=ENCODING_SHA)).read_text())
        cost = json.loads(verify(source['worker']).read_text())
        manifest_path = INDEX/'manifest.json'
        manifest = json.loads(verify(dict(path=str(manifest_path), sha256=INDEX_MANIFEST_SHA)).read_text())
        index_ref = dict(path=str(INDEX/'siren_faiss.index'), sha256=manifest['outputs']['siren_faiss.index'])
        verify(index_ref)
        index_stat = (INDEX/'siren_faiss.index').stat()
        launch = dict(encoding=record(ENCODING), vectors=cost['vectors'], index=index_ref,
                      index_manifest=record(manifest_path), index_entities=manifest['entity_count'],
                      dimension=manifest['dimension'], nprobe=manifest['nprobe'], resources=RESOURCES,
                      sample='FIRST16_LABEL_BLIND_OFFICIAL_ENCODING_COST_TEXTS',
                      identifiers_opened=False, implementation=record(Path(__file__)),
                      index_stat=dict(size=index_stat.st_size, mtime=index_stat.st_mtime_ns))
        write_json(output/'launch.json', launch)
        with (output/'worker.log').open('x') as log:
            subprocess.run([sys.executable, '-u', '-m', 'scripts.preflight_retrieval_ann_v41',
                            'worker', '--output', str(output)], check=True, timeout=90,
                           stdout=log, stderr=subprocess.STDOUT,
                           env={**os.environ, 'OMP_NUM_THREADS':'2', 'OPENBLAS_NUM_THREADS':'2'})
        for ref in (launch['encoding'], launch['vectors'], launch['index_manifest'], launch['implementation']):
            verify(ref)
        after = Path(index_ref['path']).stat()
        if dict(size=after.st_size, mtime=after.st_mtime_ns) != launch['index_stat']:
            raise ValueError('Index changed after full hash verification')
        write_json(output/'manifest.json', dict(status='SEALED_NATIONAL_PARENT_ANN_COST_ONLY',
                   launch=record(output/'launch.json'), worker=record(output/'worker.json'),
                   truth_opened=False, recall_measured=False, site_retrieval_validated=False,
                   ready_for_fold1=False))
        print((output/'worker.json').read_text())
    except BaseException as exc:
        write_json(output/'status.json', dict(status='STOP', reason=str(exc), recall_measured=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('run', 'worker'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.output) if args.mode == 'run' else worker(args.output)
