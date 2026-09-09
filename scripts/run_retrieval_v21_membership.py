#!/usr/bin/env python3
"""Rebuild a single truth-blind development Master from pinned official inputs."""
import argparse
import json
from pathlib import Path
import sys

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.xgb_matcher.retrieval_v20_workbench import Budget, record, verify, write_json, sql_path
from src.xgb_matcher.retrieval_v20_workbench import load_config as load_v20
from src.xgb_matcher.retrieval_v21_membership import load_config, manifest_files, build_tables, implementation, open_workbench_db


def run(config_path, output, preflight=None):
    config = load_config(config_path)
    baseline, files = manifest_files(config['baseline_generation'])
    if (baseline['schema'] != 'sireto-v20-master-generation-1' or baseline['preflight']
        or baseline['truth_columns_opened'] or baseline['query_count'] != 1034
        or baseline['config']['sha256'] != config['source_config']['sha256']):
        raise ValueError('Baseline generation authority')
    source, sources = load_v20(verify(config['source_config']))
    a = sources['v19']['authorities']
    signals = [verify(a[k]) for k in ('structured_signals', 'siret_native_signals')]
    sites = verify(source['sites'])
    if preflight is not None and not 1 <= preflight <= 16:
        raise ValueError('Preflight must be 1..16')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    with Budget(config['resources']) as budget:
        c = open_workbench_db(output)
        c.execute('''CREATE TEMP TABLE q AS SELECT query_id,fold,crm_number,crm_suffix,
          crm_street,crm_postcode,crm_insee FROM read_parquet(?) ORDER BY fold,query_id LIMIT ?''',
          [str(files['queries.parquet']), preflight or config['expected_queries']])
        if c.execute('SELECT count(*),count(DISTINCT query_id) FROM q').fetchone() != (preflight or 1034, preflight or 1034):
            raise ValueError('Query population drift')
        c.execute('''CREATE TEMP TABLE raw AS SELECT p.query_id,p.fold,p.siren,p.siret,p.base_rank
          FROM read_parquet(?) p SEMI JOIN q USING(query_id,fold) WHERE base_rank IS NOT NULL''',
          [str(files['candidates.parquet'])])
        c.execute('''CREATE TEMP TABLE signals AS SELECT s.query_id,s.fold,s.siren,
          s.source_siret,s.channel,s.channel_rank FROM read_parquet(?) s SEMI JOIN q USING(query_id,fold)''',
          [[str(p) for p in signals]])
        c.execute('''CREATE TEMP TABLE sites AS SELECT s.siren,s.siret,s.insee,s.postcode,
          s.street_number,s.street_number_suffix,s.street_normalized,s.administrative_state
          FROM read_parquet(?) s WHERE s.siren IN (SELECT siren FROM signals UNION SELECT siren FROM raw)''', [str(sites)])
        build_tables(c)
        tables = {'queries': 'q', 'candidates': 'candidates', 'master': 'master', 'witnesses': 'witnesses'}
        for name, table in tables.items():
            order = {'master':'query_id,candidate_rank','q':'query_id',
                     'candidates':'query_id,siret','witnesses':'query_id,siret,channel,channel_rank'}[table]
            c.execute(f'COPY (SELECT * FROM {table} ORDER BY {order}) TO {sql_path(output/(name+".parquet"))} (FORMAT PARQUET,COMPRESSION ZSTD)')
        counts = {t: c.execute(f'SELECT count(*) FROM {t}').fetchone()[0] for t in ('q','parents','address_sites','candidates','master')}
        c.close()
    m = {'schema': 'sireto-v21-generation-1', 'config': record(config_path),
         'baseline_generation': config['baseline_generation'], 'source_config': config['source_config'],
         'inputs': {str(p): record(p) for p in [files['queries.parquet'], files['candidates.parquet'], sites, *signals]},
         'files': {p.name: record(p) for p in output.iterdir() if p.is_file()},
         'preflight': preflight is not None, 'counts': counts, 'resources': budget.report(),
         'truth_columns_opened': False, 'fold0_opened': False, 'fold1_opened': False,
         'positive_injection': False, 'learned_components': False, 'external_validation': False,
         'implementation': implementation()}
    write_json(output/'manifest.json', m)
    print(json.dumps(m), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--preflight', type=int)
    a = p.parse_args(); run(a.config, a.output, a.preflight)
