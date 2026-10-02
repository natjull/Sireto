"""Bounded RNE extraction and nested TRAIN proof control, no retrieval eval."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scripts import retrieval_v38_initial_top100 as v38
from scripts.preflight_retrieval_v72_observation_cost import authority
from scripts.retrieval_sparse_training_source import load
from scripts.retrieval_v56_exact_fields import index_authority
from src.xgb_matcher.retrieval_coherent_proof_v74 import coherent, select, views
from src.xgb_matcher.retrieval_sparse_contrastive_v69 import proof
from src.xgb_matcher.retrieval_v20_workbench import Budget, ROOT, record, verify, write_json

CONFIG = ROOT/'config/retrieval_v74_coherent_proof.json'


def training_rows(cfg):
    source = json.loads(verify(cfg['vector_source']).read_text())
    sl = json.loads(verify(source['launch']).read_text()); v38.verify_launch(sl)
    data = json.loads(verify(sl['inputs']).read_text())
    rows = data['rows']
    if source['recall_measured'] or len(rows) != 512 or any(r['fold'] not in (2, 3, 4) for r in rows):
        raise ValueError('Training source scope drift')
    if len({r['query_id'] for r in rows}) != 512:
        raise ValueError('Training query uniqueness drift')
    return rows


def keys(rows):
    sites = set(); parents = set()
    for r in rows:
        for siret, siren in [(r['positive_siret'], r['positive_siren']),
                            *[(n['siret'], n['siren']) for n in r['negatives']]]:
            if len(siret) != 14 or len(siren) != 9 or not siret.isdecimal() or not siren.isdecimal() or not siret.startswith(siren):
                raise ValueError('Training official lookup identity drift')
            sites.add(siret); parents.add(siren)
    return sorted(sites), sorted(parents)


def extract(output, launch, cfg):
    rows = training_rows(cfg); sites, parents = keys(rows)
    expected = launch['source_authority']
    if authority(cfg['rne_source']) != expected:
        raise ValueError('RNE source changed before extraction')
    connection = duckdb.connect(':memory:')
    connection.execute("SET threads=2")
    connection.execute("SET memory_limit='1GB'")
    connection.execute("SET preserve_insertion_order=false")
    connection.execute("SET temp_directory=?", [str(output/'duckdb_scratch')])
    connection.register('wanted_parent', pa.table({'siren': parents}))
    connection.register('wanted_site', pa.table({'siret': sites}))
    cursor = connection.execute('''SELECT * FROM read_parquet(?)
        WHERE siren IN (SELECT siren FROM wanted_parent)
          AND (subject_kind='SIREN' OR
               (subject_kind='SIRET' AND siret IN (SELECT siret FROM wanted_site)))
        ORDER BY siren, subject_kind, source_record_id, evidence_id''', [expected['path']])
    selected = []; began = time.monotonic(); site_set = set(sites); parent_set = set(parents)
    for batch in cursor.fetch_record_batch(4096):
        selected.extend(batch.to_pylist())
        if len(selected) > cfg['extraction_row_cap']:
            raise ValueError('STOP extraction cap exceeded; do not truncate')
    connection.close()
    source_digest = hashlib.sha256()
    for row in selected:
        if row['source'] != 'RNE' or row['siren'] not in parent_set or (
            row['subject_kind'] == 'SIRET' and row['siret'] not in site_set):
            raise ValueError('Official TRAIN extraction selection drift')
        source_digest.update(json.dumps(row, sort_keys=True, default=str).encode()+b'\n')
    bound, counters = views(selected)
    features = []; coverage_counts = Counter()
    for row in rows:
        ordered = [row['positive_siret'], *[n['siret'] for n in row['negatives']]]
        values = [coherent(row, bound.get(sid, ())) for sid in ordered]
        flags = [sid in bound for sid in ordered]
        coverage_counts['queries'] += 1
        coverage_counts['positive_rne_present'] += flags[0]
        coverage_counts['positive_bonus_nonzero'] += values[0] > 0
        coverage_counts['negative_pairs'] += len(flags)-1
        coverage_counts['negative_rne_present'] += sum(flags[1:])
        coverage_counts['negative_bonus_nonzero'] += sum(v > 0 for v in values[1:])
        features.append(dict(query_id=row['query_id'], fold=row['fold'], sirets=ordered,
                             bonus=values, rne_present=flags))
    write_json(output/'coherent_features.json', features)
    pq.write_table(pa.Table.from_pylist(selected, schema=pq.ParquetFile(expected['path']).schema_arrow),
                   output/'training_only_rne_observations.parquet', compression='zstd')
    if authority(cfg['rne_source']) != expected:
        raise ValueError('Official source changed during extraction')
    return dict(status='SEALED_COMPLETE_TRAIN_ONLY_EXTRACTION', features=record(output/'coherent_features.json'),
        observations=record(output/'training_only_rne_observations.parquet'), rows=len(selected),
        requested_sites=len(sites), requested_parents=len(parents), returned_site_keys=len(bound),
        coverage=dict(coverage_counts), binding=counters, selected_rows_sha256=source_digest.hexdigest(),
        materialize_and_feature_seconds=time.monotonic()-began, source_authority=expected,
        label_selected_training_only=True, forbidden_as_national_index_source=True,
        all_512_queries_preserved=True, recall_measured=False)


def train(output, launch, cfg):
    extraction = json.loads((output/'extraction_manifest.json').read_text())
    if extraction['status'] != 'SEALED_COMPLETE_TRAIN_ONLY_EXTRACTION' or not extraction['forbidden_as_national_index_source']:
        raise ValueError('Unsealed/incorrect-scope TRAIN feature extraction')
    verify(extraction['observations'])
    records = json.loads(verify(extraction['features']).read_text())
    _, rows, scenes, _, aliases, registry, counters = load(cfg, launch)
    if len(records) != len(rows):
        raise ValueError('Missing TRAIN scene; do not filter')
    features = []
    for row, scene, new in zip(rows, scenes, records):
        ordered = [row['positive_siret'], *[n['siret'] for n in row['negatives']]]
        if (new['sirets'] != ordered or new['query_id'] != row['query_id'] or new['fold'] != row['fold']
            or len(new['bonus']) != len(ordered) or len(new['rne_present']) != len(ordered)):
            raise ValueError('Coupled feature alignment drift')
        native = [scene['positive_features'], *[n['features'] for n in scene['negatives']]]
        features.append(np.asarray([np.r_[proof(row, f, aliases[sid]), bonus]
                                    for sid, f, bonus in zip(ordered, native, new['bonus'])]))
    reports = {}; models = {}
    for held in (2, 3, 4):
        models[str(held)], reports[str(held)] = select(rows, features, registry, held)
        print(json.dumps(dict(fold=held, selected=reports[str(held)]['selected'],
            before={k:v for k,v in reports[str(held)]['outer_before'].items() if k != 'per_query'},
            after={k:v for k,v in reports[str(held)]['outer_after'].items() if k != 'per_query'})), flush=True)
    if sum(r['outer_after']['queries'] for r in reports.values()) != 512:
        raise ValueError('Outer denominator drift')
    write_json(output/'models.json', models)
    return dict(status='SEALED_COHERENT_RNE_TRAIN_CONTROL', models=record(output/'models.json'),
        extraction=record(output/'extraction_manifest.json'), reports=reports, training_queries=512,
        **counters, recall_measured=False, full_national_search_performed=False,
        output_candidates=False, index_built=False, ready_for_fold1=False, fold0_opened=False,
        fold1_opened=False, downstream_models_unchanged=True,
        limitations=['SOURCE_NEGATIVE_CONTROL_NOT_RETRIEVAL_RECALL',
            'TRAIN_ONLY_LABEL_SELECTED_EXTRACTION_FORBIDDEN_AS_NATIONAL_CORPUS',
            'NATIVE_COMPARATOR_STILL_AGGREGATES_OLD_VIEWS', 'NO_NATIONAL_EXECUTOR_OR_SLO',
            'RNE_VALIDITY_DATES_NOT_FABRICATED_OR_SCORED', 'NO_BODACC_PROJECTION_OR_TYPED_GRAPH'])


def worker(output, mode):
    launch = json.loads((output/'launch.json').read_text()); cfg = launch['config']
    v38.verify_launch(launch); budget = Budget(cfg['resources'])
    pa.set_cpu_count(2)
    try:
        with budget:
            result = (extract if mode == 'extract' else train)(output, launch, cfg)
        v38.verify_launch(launch)
        for key in ('vector_source', 'proof_source', 'alias_validation'):
            verify(cfg[key])
        result.update(resources=budget.report(), launch=record(output/'launch.json'))
        write_json(output/('extraction_manifest.json' if mode == 'extract' else 'manifest.json'), result)
        print(json.dumps(dict(mode=mode, resources=budget.report(), status=result['status'])), flush=True)
    except BaseException as exc:
        write_json(output/f'{mode}_stop.json', dict(status='STOP', reason=str(exc), recall_measured=False,
            resources=budget.report() if hasattr(budget, 'elapsed') else None)); raise


def run(output):
    output.mkdir(parents=True, exist_ok=False); cfg = json.loads(CONFIG.read_text())
    launch = dict(**index_authority(), config=cfg, source_authority=authority(cfg['rne_source']),
                  truth_scope='TRAINER_ONLY_NO_RETRIEVAL_EVALUATION')
    paths = [Path(__file__), CONFIG, ROOT/'docs/retrieval_v74_coherent_proof_probe.md',
        ROOT/'scripts/retrieval_sparse_training_source.py', ROOT/'scripts/preflight_retrieval_v69_sparse_contrastive.py',
        ROOT/'scripts/preflight_retrieval_v72_observation_cost.py', ROOT/'scripts/audit_retrieval_v73_alias_counters.py',
        *(ROOT/f'src/xgb_matcher/{n}.py' for n in ('retrieval_coherent_proof_v74', 'official_evidence',
            'retrieval_observation_v72', 'retrieval_owner_observation_v73', 'retrieval_sparse_contrastive_v69',
            'retrieval_initialisms_v67', 'retrieval_joint_contrastive_v63', 'retrieval_nested_v53'))]
    launch['implementation'].update({str(p):record(p) for p in paths})
    write_json(output/'launch.json', launch)
    try:
        for mode in ('extract', 'train'):
            print(json.dumps(dict(starting=mode)), flush=True)
            with (output/f'{mode}.log').open('x') as log:
                subprocess.run([sys.executable, '-u', '-m', 'scripts.preflight_retrieval_v74_coherent_proof',
                    mode, '--output', str(output)], check=True, stdout=log, stderr=subprocess.STDOUT,
                    timeout=90, env={**os.environ, 'OMP_NUM_THREADS':'2', 'OPENBLAS_NUM_THREADS':'2',
                        'VECLIB_MAXIMUM_THREADS':'2', 'RAYON_NUM_THREADS':'2'})
            print((output/f'{mode}.log').read_text()[-6000:], flush=True)
    except BaseException as exc:
        write_json(output/'status.json', dict(status='STOP', reason=str(exc), recall_measured=False)); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['run', 'extract', 'train'])
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.output) if args.mode == 'run' else worker(args.output, args.mode)
