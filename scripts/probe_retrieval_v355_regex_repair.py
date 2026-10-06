"""Label-blind full-panel compiler audit and one real failed-batch regression.

Fresh probe ONLY. No old checkpoints/payloads, target sidecars, recall or full
generation authority; no automatic long run and no terminal STOP alteration.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import tantivy

from scripts import retrieval_v354_regex_worker as repaired
from scripts import retrieval_v281_official_region_snapshots as official
from src.xgb_matcher.retrieval_nominal_abbreviation_v256 import patterns
from src.xgb_matcher.retrieval_regional_abbreviation_v285 import patterns as regional_patterns, names
from src.xgb_matcher.retrieval_regex_compiler_v353 import Backend
from src.xgb_matcher.retrieval_identifier_surface_v329 import feature_query
from src.xgb_matcher.retrieval_corroborated_initial_v217 import EXACT_FIELDS
from src.xgb_matcher.retrieval_v20_workbench import ROOT, record, verify, write_json

OUTPUT = Path(repaired.original.config()['output_root'])/'probes_v355_regex_repair'


def run(source_launch, root, batch):
    if root.exists() or root.parent.resolve() != OUTPUT.resolve():
        raise ValueError('One fresh dedicated regex regression probe required')
    original_ref = record(source_launch)
    source = json.loads(verify(original_ref).read_text())  # QUERY-only launch, never old outputs.
    repaired.original.positions(batch)
    refs = repaired.implementation()
    refs[str(Path(__file__))] = record(Path(__file__))
    refs['compiler-test'] = record(ROOT/'tests/test_retrieval_regex_compiler_v353.py')
    root.mkdir(parents=True, exist_ok=False)
    launch = {**source, 'implementation': {**source['implementation'], **refs},
              'execution_config': dict(mode='V355_PROBE_ONLY', batch=batch),
              'regex_compiler_policy': repaired.COMPILER_POLICY,
              'probe_source_query_launch': original_ref, 'old_outputs_imported': False}
    write_json(root/'launch.json', launch)
    rows = []
    try:
        repaired.load(root)
        _, snapshots = official.sealed(Path(launch['config']['regional_source']['path']).parent)
        regions = names(snapshots['snapshots'])
        builder = tantivy.SchemaBuilder()
        for field in EXACT_FIELDS:
            builder.add_text_field(field, tokenizer_name='raw')
        schema = builder.build()
        backend = Backend(SimpleNamespace(_tantivy=tantivy))
        began = time.monotonic()
        expressions, calls, affected = 0, 0, []
        for pos, raw in enumerate(launch['queries']):
            q = feature_query(raw)
            before = len(backend.queries.repairs)
            ps = patterns(q.name) + regional_patterns(q.name, regions)
            expressions += len(ps)
            for field in EXACT_FIELDS:
                for pattern in ps:
                    backend.queries.regex_query(schema, field, pattern)
                    calls += 1
            if len(backend.queries.repairs) > before:
                affected.append(dict(position=pos, field_patterns_repaired=len(backend.queries.repairs)-before))
        repaired.load(root)
        if record(source_launch) != original_ref:
            raise ValueError('Source query launch drift')
        write_json(root/'compiler_audit.json', dict(status='PASS_COMPILER_ONLY_NOT_RETRIEVAL',
            launch=record(root/'launch.json'), queries=len(launch['queries']), expressions=expressions,
            field_compilations=calls, repaired_field_patterns=len(backend.queries.repairs),
            affected=affected, seconds=time.monotonic()-began, patterns_discarded=0,
            native_state_limit_changed=False, **repaired.original.FALSE))
        directory = root/f'batch{batch:03d}'
        directory.mkdir()
        write_json(directory/'preflight.json', repaired.original.quiet(30, 50))
        env = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
               'RAYON_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1'}
        with (directory/'worker.log').open('x') as log:
            receipt = repaired.original.supervise([sys.executable, '-u', '-m',
                'scripts.retrieval_v354_regex_worker', 'worker', '--output', str(root),
                '--batch', str(batch)], log, env, directory/'supervisor_stop.json')
        write_json(directory/'supervision.json', receipt)
        manifest, payload = repaired.stage(root, batch)
        rows = payload['rows']
        write_json(root/'probe_manifest.json', dict(status='PASS_V355_REAL_BATCH_ONLY_NOT_ALL2000',
            launch=record(root/'launch.json'), compiler_audit=record(root/'compiler_audit.json'),
            producer=record(directory/'manifest.json'), supervision=record(directory/'supervision.json'),
            payload=manifest['payload'], positions=repaired.original.positions(batch),
            resources=manifest['resources'], old_outputs_imported=False, all2000_outputs_sealed=False,
            **repaired.original.FALSE))
        print(json.dumps(dict(status='PASS_V355_REAL_BATCH_ONLY_NOT_ALL2000',
            rows=[dict(position=r['position'], candidates=len(r['results']['primary']),
                       seconds=r['seconds']['primary'], repaired=r['primary_trace']['regex_patterns_repaired'])
                  for r in rows], resources=manifest['resources'], truth_opened=False)), flush=True)
    except BaseException as exc:
        write_json(root/'status.json', dict(status='STOP', reason=str(exc),
            completed=len(rows), **repaired.original.FALSE))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-launch', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--batch', type=int, default=97)
    args = parser.parse_args()
    run(args.source_launch, args.output, args.batch)
