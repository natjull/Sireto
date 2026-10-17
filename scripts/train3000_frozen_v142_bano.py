"""Rebuild BANO query contexts using the historical body, under the owned VM guard.

This stage is NOT the four-family fidelity gate, the BANO weighted leaf scores,
Gemma preparation, native smoke, or the full3000 generator.
"""
import argparse
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import signal
import sqlite3
import sys
import textwrap
import time

os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import ijson
from scripts import train3000_frozen_v142_cohort as cohort
from scripts.retrieval_site_operational_metric import read, record
from scripts import retrieval_site_target_full as supervision

OUT = cohort.OUT/'bano'
SOURCE = ROOT/'scripts/embeddinggemma2_v141_bano_national_source.py'
EXPECTED = 'bbc819028db034bbbec59a4510be2b0853beb1cf16f960867e87abacbf5a5d14'
# Recomputed against the historical V142 recipe commit ae7c3e05, not current HEAD.
HELPERS = {
    'scripts/embeddinggemma2_v141_bano_osm_source.py': '8b1e2496be9e9d7da57817a892f4e4b0bba4e38d5db0e5d76f26ec36216aa97c',
    'scripts/embeddinggemma2_v129_osm_coverage_audit.py': '5f83be1bdc5a3053db664d6f8a7245b256d64967bee4a235b4c5350fd84befa9',
    'scripts/embeddinggemma2_v17_structural_retrieval.py': '71ddfee9043e0a799a7ad39f76ec60c2f86d3558ce6382de5c044f0ff7ea08ea',
    'src/xgb_matcher/retrieval_geo_v16.py': 'd8997434c26150607b1bc37072185c98bb5965155a7a99c73b4ffa4aee0665b0',
}


def context_builder():
    source = record(SOURCE, EXPECTED)
    text = SOURCE.read_text()
    start = '    communes=base.io.read(base.COMMUNES);current='
    end = '    con.close();base.guard()'
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError('Exact historical BANO context body required')
    body = text[text.index(start):text.index(end)]
    from scripts import embeddinggemma2_v141_bano_osm_source as base
    from src.xgb_matcher.retrieval_geo_v16 import lonlat_to_l93
    namespace = dict(base=base, lonlat_to_l93=lonlat_to_l93)
    # Only provide explicit inputs and return value around the unchanged body.
    exec(compile('def build(panel, con):\n'+textwrap.indent(textwrap.dedent(body), '    ')+
        '\n    return context\n', str(SOURCE), 'exec'), namespace)
    return namespace['build'], source, base, hashlib.sha256(body.encode()).hexdigest()


def worker():
    if supervision.hydraulic()['status'] != 'Stopped':
        raise RuntimeError('Worker cannot bypass live hydraulic admission')
    generation_path = OUT/'generation.json'
    if generation_path.exists():
        for ref in read(generation_path)['inputs']:
            record(ref['path'], ref['sha256'])
        return
    started = time.monotonic()
    sample = read(cohort.OUT/'sample.json')
    if read(ROOT/'docs/train3000_frozen_v142_sample.json') != sample:
        raise ValueError('Committed draw binding differs')
    build, source, base, kernel_sha = context_builder()
    launch = read(cohort.preflight.RUN/'launch.json')
    ref = launch['bano_generation']
    bg = read(record(ref['path'], ref['sha256'])['path'])
    bl = read(record(bg['launch']['path'], bg['launch']['sha256'])['path'])
    refs = [source, record(ref['path'], ref['sha256'])]
    for item in (bg['launch'], bg['index'], bl['commune_reference'], bl['delegated_reference'],
                 bl['shared_preparation_module'], bg['query_context']):
        refs.append(record(item['path'], item['sha256']))
    refs.extend(record(ROOT/path, expected) for path, expected in HELPERS.items())
    if str(base.COMMUNES) != bl['commune_reference']['path'] or str(base.DELEGATES) != bl['delegated_reference']['path']:
        raise ValueError('Historical commune authorities differ')
    if bl['retained_source_codes'] != ['OSM'] or not bg['no_BAN_seed_used']:
        raise ValueError('Frozen OSM-only national BANO authority required')
    pilot = read(cohort.OUT/'pilot_fidelity_queries.json')
    queries = read(cohort.OUT/'queries.json')
    if len(pilot) != 20 or len(queries) != 3000 or any(set(q) != set(cohort.old.CRM) for q in [*pilot, *queries]):
        raise ValueError('Strict query-only inputs, no label/model input')
    wanted = {q['query_id'] for q in pilot}
    if wanted != set(sample['fidelity_query_ids']):
        raise ValueError('Fidelity pilot IDs differ from committed draw')
    frozen_panel_ref = launch['panel']
    frozen_panel = read(record(frozen_panel_ref['path'], frozen_panel_ref['sha256'])['path'])
    expected_pilot = {q['query_id']:q for q in frozen_panel if q['query_id'] in wanted}
    if {q['query_id']:q for q in pilot} != expected_pilot:
        raise ValueError('Fidelity CRM differs from frozen original')
    import pyarrow.parquet as pq
    ids = sorted(sample['selected_query_ids'])
    originals = {}
    for item in sample['query_sources']:
        record(item['path'], item['sha256'])
        for q in pq.read_table(item['path'], columns=list(cohort.old.CRM),
                filters=[('fold', 'in', [2, 3, 4]), ('query_id', 'in', ids)], use_threads=False).to_pylist():
            if q['query_id'] in originals:
                raise ValueError('Duplicate original query')
            originals[q['query_id']] = q
    if {q['query_id']:q for q in queries} != originals:
        raise ValueError('New CRM differs from sealed original query sources')
    for path in (cohort.OUT/'sample.json', cohort.OUT/'queries.json', cohort.OUT/'pilot_fidelity_queries.json'):
        refs.append(record(path))
    frozen_context = {}
    with Path(bg['query_context']['path']).open('rb') as stream:
        for qid, value in ijson.kvitems(stream, '', use_float=True):
            if qid in wanted:
                frozen_context[qid] = value
    if set(frozen_context) != wanted:
        raise ValueError('Fidelity scope differs from frozen pilot')
    database = Path(bg['index']['path'])
    connection = sqlite3.connect(f'file:{database}?mode=ro&immutable=1', uri=True)
    connection.execute('PRAGMA query_only=ON')
    try:
        fresh = build(pilot, connection)
        canonical = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()
        old_bytes, new_bytes = canonical(frozen_context), canonical(fresh)
        if new_bytes != old_bytes:
            differences = [qid for qid in wanted if fresh[qid] != frozen_context[qid]]
            cohort.save(OUT/'fidelity_failure.json', dict(status='STOP_BANO_CONTEXT_NONFIDELITY', queries=differences,
                source=source, context_body_sha256=kernel_sha))
            raise ValueError('STOP_BANO_CONTEXT_NONFIDELITY: preserve outputs, no further phase')
        cohort.save(OUT/'pilot_contexts.json', fresh)
        new = build(queries, connection)
    finally:
        connection.close()
    if set(new) != set(sample['selected_query_ids']):
        raise ValueError('All3000 query contexts required')
    for ref in refs:
        record(ref['path'], ref['sha256'])
    cohort.save(OUT/'query_context.json', new)
    cohort.save(generation_path, dict(status='COMPLETE_BANO_QUERY_CONTEXTS_NOT_WEIGHTED_LEAF_SCORES',
        inputs=refs, context=record(OUT/'query_context.json'), pilot_contexts=record(OUT/'pilot_contexts.json'),
        fidelity_queries=20, exact_context_matches=20, new_queries=3000, source=source,
        fidelity_canonical_sha256=hashlib.sha256(new_bytes).hexdigest(),
        reference_canonical_sha256=hashlib.sha256(old_bytes).hexdigest(),
        context_body_sha256=kernel_sha, full_four_family_fidelity_complete=False,
        weighted_bano_scores_rebuilt=False, raw_labels_read=False, native_queries=0,
        elapsed_seconds=time.monotonic()-started))


def configure():
    # Reuse supervised(), VM/disk/RSS/owned-exit/restart logic verbatim. Only
    # private process-local paths, worker entry point and publication are bound.
    original = record(Path(supervision.__file__))
    supervision.OUT = OUT
    supervision.__file__ = str(Path(__file__).resolve())
    supervision.atomic_json = cohort.save
    supervision.work_ref = lambda *args, **kwargs: OUT/'generation.json'
    signal.signal(signal.SIGTERM, supervision.signal_stop)
    signal.signal(signal.SIGINT, supervision.signal_stop)
    return original


def run():
    original = configure()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'owned.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cohort.save(OUT/'controllers'/(supervision.stamp()+'.json'), dict(pid=os.getpid(),
            command=sys.argv, started_unix=time.time(), implementation=record(Path(__file__)),
            reused_supervision=original, function_sha256=hashlib.sha256(inspect.getsource(supervision.supervised).encode()).hexdigest(),
            scope='BANO contexts only, not full3000 or full fidelity gate'))
        supervision.supervised('prepare', stage='bano')
        print('COMPLETE_BANO_CONTEXT_STAGE_ONLY', flush=True)


def status():
    if (OUT/'generation.json').exists():
        result = read(OUT/'generation.json')
        return {k:result[k] for k in ('status', 'fidelity_queries', 'exact_context_matches', 'new_queries',
            'full_four_family_fidelity_complete', 'weighted_bano_scores_rebuilt', 'native_queries', 'elapsed_seconds')}
    events = sorted((OUT/'progress').glob('*.json'))
    return dict(status=read(events[-1])['phase'] if events else 'NOT_STARTED',
        stage='BANO_QUERY_CONTEXTS_ONLY', full_four_family_fidelity_complete=False,
        native_queries=0, namespace=str(OUT), latest_event=read(events[-1]) if events else None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--run', action='store_true')
    actions.add_argument('--worker', choices=['prepare'])
    actions.add_argument('--status', action='store_true')
    parser.add_argument('--stage', choices=['bano'])
    args = parser.parse_args()
    if args.worker:
        worker()
    elif args.run:
        run()
    else:
        print(json.dumps(status(), indent=2, ensure_ascii=False))
