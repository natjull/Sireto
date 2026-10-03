"""Bounded official-source variant prototype; cannot be a recall evaluation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from src.xgb_matcher.retrieval_variants_v118 import OBS_SCHEMA, factorize, observations
from src.xgb_matcher.retrieval_v20_workbench import Budget, ROOT, record, verify, write_json


CONFIG = ROOT / 'config/retrieval_v118_official_variants.json'


def fingerprint(path):
    """Stat + complete footer, explicitly NOT the full input content hash."""
    path = Path(path); stat = path.stat()
    with path.open('rb') as stream:
        stream.seek(-8, 2); tail = stream.read(8)
        if tail[4:] != b'PAR1':
            raise ValueError('Invalid Parquet trailer')
        size = int.from_bytes(tail[:4], 'little')
        if not 0 < size < min(stat.st_size - 8, 64 * 1024 * 1024):
            raise ValueError('Unexpected footer size')
        stream.seek(-size - 8, 2); footer = stream.read(size)
    return dict(size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, inode=stat.st_ino,
                footer_sha256=hashlib.sha256(footer).hexdigest(), footer_bytes=size,
                full_input_content_sha256_verified=False)


def sample_batches(path, max_rows):
    if not isinstance(max_rows, int) or isinstance(max_rows, bool) or max_rows <= 0:
        raise ValueError('Positive sample limit required')
    file = pq.ParquetFile(path)
    groups = sorted({0, file.metadata.num_row_groups // 2, file.metadata.num_row_groups - 1})
    if groups[0] < 0:
        raise ValueError('No physical row group')
    offsets = [0]
    for i in range(file.metadata.num_row_groups):
        offsets.append(offsets[-1] + file.metadata.row_group(i).num_rows)
    for index, group in enumerate(groups):
        quota = max_rows // len(groups) + (index < max_rows % len(groups))
        if quota == 0:
            continue
        batch = next(file.iter_batches(batch_size=quota, row_groups=[group], use_threads=False))
        yield group, offsets[group], batch


def required_fields(role):
    common = {'siren'}
    if role == 'current_ul':
        return common | {'nomUniteLegale', 'nomUsageUniteLegale', 'prenomUsuelUniteLegale',
                         'prenom1UniteLegale', 'statutDiffusionUniteLegale'}
    if role == 'history_ul':
        return common | {'nomUniteLegale', 'denominationUniteLegale', 'dateDebut', 'dateFin'}
    if role in ('current_site', 'history_site'):
        return common | {'siret', 'enseigne1Etablissement', 'enseigne2Etablissement',
                         'enseigne3Etablissement', 'denominationUsuelleEtablissement'}
    if role == 'official_names':
        return common | {'siret', 'subject_kind', 'source', 'name_kind', 'raw_value',
                         'normalized_value', 'valid_from', 'valid_to'}
    if role == 'successions':
        return {'siretEtablissementPredecesseur', 'siretEtablissementSuccesseur',
                'dateLienSuccession', 'transfertSiege', 'continuiteEconomique'}
    raise ValueError('Unknown source role')


def project_samples(cfg, output):
    sources = {}; phases = {}; raw_path = output / 'raw_observations.parquet'
    with pq.ParquetWriter(raw_path, OBS_SCHEMA, compression='zstd') as observation_writer:
        for role, ref in cfg['sources'].items():
            began = time.monotonic(); path = Path(ref['path']); before = fingerprint(path)
            file = pq.ParquetFile(path)
            if file.metadata.num_rows != ref['expected_rows']:
                raise ValueError('Full source footer count drift: ' + role)
            missing = required_fields(role) - set(file.schema_arrow.names)
            if missing:
                raise ValueError('Source field missing: ' + role + ' ' + str(sorted(missing)))
            rows = 0; obs_count = 0; sampled = []
            sample_path = output / ('sample_' + role + '.parquet')
            sample_schema = file.schema_arrow.append(pa.field('source_row_ordinal', pa.uint64()))
            with pq.ParquetWriter(sample_path, sample_schema, compression='zstd') as sample_writer:
                for group, start, batch in sample_batches(path, cfg['sample_rows_per_source']):
                    table = pa.Table.from_batches([batch]).append_column(
                        'source_row_ordinal', pa.array(range(start, start + batch.num_rows), type=pa.uint64()))
                    sample_writer.write_table(table)
                    sampled.append(dict(row_group=group, source_row_start=start, rows=batch.num_rows))
                    rows += batch.num_rows
                    if role != 'successions':
                        buffer = []
                        for row in table.to_pylist():
                            buffer.extend(observations(row, role, row['source_row_ordinal']))
                            if len(buffer) >= 8192:
                                observation_writer.write_table(pa.Table.from_pylist(buffer, schema=OBS_SCHEMA))
                                obs_count += len(buffer); buffer.clear()
                        if buffer:
                            observation_writer.write_table(pa.Table.from_pylist(buffer, schema=OBS_SCHEMA))
                            obs_count += len(buffer)
            if fingerprint(path) != before:
                raise ValueError('Source changed during sampling: ' + role)
            sources[role] = dict(path=str(path), declared_full_sha256=ref['sha256'],
                fingerprint=before, full_rows=file.metadata.num_rows, row_groups=file.metadata.num_row_groups,
                schema_fields=file.schema_arrow.names, sampled_groups=sampled, sampled_rows=rows,
                observations=obs_count, full_sample_record=record(sample_path))
            phases[role] = time.monotonic() - began
            print(json.dumps(dict(phase=role, sampled_rows=rows, observations=obs_count,
                                  seconds=phases[role])), flush=True)
    c = duckdb.connect(); c.execute('SET threads=1'); c.execute('SET preserve_insertion_order=false')
    c.execute('SET memory_limit=?', [cfg['duckdb_memory']])
    c.execute('SET temp_directory=?', [str(output / 'spill')])
    c.execute('SET max_temp_directory_size=?', [cfg['spill_limit']])
    began = time.monotonic(); stats = factorize(c, raw_path, output)
    phases['factorize_and_complete_roundtrip'] = time.monotonic() - began
    # Every sampled input row must survive in the observation/context ledger.
    for role, source in sources.items():
        if role == 'successions':
            continue
        survived = c.execute('SELECT count(DISTINCT source_row_ordinal) FROM read_parquet(?) '
                             'WHERE input_role=?', [str(raw_path), role]).fetchone()[0]
        if survived != source['sampled_rows']:
            raise ValueError('Sample source row ledger drift: ' + role)
    result = c.execute('SELECT input_role,source,derivation,count(*) observations, '
        'count(*) FILTER(WHERE lexical_eligible) lexical_eligible '
        'FROM read_parquet(?) GROUP BY ALL ORDER BY ALL', [str(raw_path)])
    columns = [field[0] for field in result.description]
    stats['segments'] = [dict(zip(columns, row)) for row in result.fetchall()]
    c.close()
    for source in sources.values():
        if fingerprint(source['path']) != source['fingerprint']:
            raise ValueError('Source changed after factorization')
    return dict(sources=sources, phases=phases, factorization=stats,
                sampled_original_records_preserved=True, selected_by_crm_or_label=False,
                successor_edges_retained_as_original_records=True,
                succession_expansion_built=False, national_publication=False,
                full_source_content_hash_verified=False, cost_sample_representative=False)


def worker(output):
    launch = json.loads((output / 'launch.json').read_text()); budget = None
    try:
        for ref in launch['implementation'].values():
            verify(ref)
        with Budget(launch['config']['resources']) as budget:
            stats = project_samples(launch['config'], output)
            output_bytes = sum(path.stat().st_size for path in output.rglob('*') if path.is_file())
            if output_bytes > launch['config']['max_output_bytes']:
                raise ValueError('STOP_OUTPUT_SIZE')
        for ref in launch['implementation'].values():
            verify(ref)
        write_json(output / 'manifest.json', dict(status='SEALED_SOURCE_VARIANT_COST_PROTOTYPE_ONLY',
            launch=record(output / 'launch.json'), stats=stats, resources=budget.report(),
            output_bytes=output_bytes, artifacts={p.name:record(p) for p in output.glob('*.parquet')},
            truth_opened=False, crm_opened=False, fold0_opened=False, fold1_opened=False,
            model_fit=False, recall_measured=False, index_built=False, identifiers_as_features=False,
            downstream_models_changed=False, ready_for_fold1=False))
        print(json.dumps(dict(stats=stats['factorization'], phases=stats['phases'],
                              resources=budget.report(), output_bytes=output_bytes)), flush=True)
    except BaseException as exc:
        write_json(output / 'worker_stop.json', dict(status='STOP', reason=str(exc),
            recall_measured=False, truth_opened=False, resources=budget.report()
            if budget is not None and hasattr(budget, 'elapsed') else None))
        raise


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(CONFIG.read_text())
    paths = [Path(__file__), CONFIG, ROOT / 'src/xgb_matcher/retrieval_variants_v118.py',
             ROOT / 'src/xgb_matcher/retrieval_v20_workbench.py',
             ROOT / 'src/xgb_matcher/retrieval_parent_discovery_v18.py',
             ROOT / 'docs/retrieval_v118_official_variants.md']
    write_json(output / 'launch.json', dict(config=cfg,
        implementation={str(path):record(path) for path in paths}, truth_opened=False))
    try:
        with (output / 'worker.log').open('x') as log:
            subprocess.run([sys.executable, '-u', '-m', 'scripts.probe_retrieval_v118_official_variants',
                'worker', '--output', str(output)], check=True, timeout=90, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, 'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1'})
        print((output / 'worker.log').read_text()[-10000:], flush=True)
    except BaseException as exc:
        write_json(output / 'status.json', dict(status='STOP', reason=str(exc), recall_measured=False,
                                               truth_opened=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['run', 'worker']); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); (run if args.mode == 'run' else worker)(args.output)
