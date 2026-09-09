#!/usr/bin/env python3
"""Evaluate once against sealed truths; never claim this reused panel is unseen."""
import argparse
import json
from pathlib import Path
import sys

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.xgb_matcher.retrieval_v20_workbench import Budget, record, write_json, sql_path
from src.xgb_matcher.retrieval_v21_membership import load_config, manifest_files, implementation, open_workbench_db


def summarize(df):
    return {'queries': len(df), 'exact_hits': int(df.exact_hit.sum()),
            'exact_recall': float(df.exact_hit.mean()) if len(df) else None,
            'operational_hits': int(df.operational_hit.sum()),
            'operational_recall': float(df.operational_hit.mean()) if len(df) else None,
            'operational_only': int((df.operational_hit & ~df.exact_hit).sum()),
            'active_successor_present': int(df.active_successor_present.sum()),
            'multiple_active_equivalents_present': int(df.multiple_active_equivalents_present.sum()),
            'gains_vs_v20': int((df.exact_hit & ~df.previous_exact_hit).sum()),
            'losses_vs_v20': int((~df.exact_hit & df.previous_exact_hit).sum())}


def evaluate_tables(c):
    """Tables t (truth), official_loss, acceptable, q, master, candidates."""
    if c.execute('SELECT count(*),count(DISTINCT query_id) FROM t').fetchone() != c.execute('SELECT count(*),count(DISTINCT query_id) FROM q').fetchone():
        raise ValueError('Truth population drift')
    if c.execute('SELECT count(*) FROM t ANTI JOIN q USING(query_id,fold)').fetchone()[0]:
        raise ValueError('Truth/query membership mismatch')
    c.execute('''CREATE TEMP TABLE operational AS
      SELECT t.query_id,bool_or(m.siret IS NOT NULL) operational_hit,
        coalesce(bool_or(m.siret IS NOT NULL AND a.administrative_state='A' AND NOT a.exact_truth AND cat.exact_state='F'),false) active_successor_present,
        count(DISTINCT CASE WHEN m.siret IS NOT NULL AND a.administrative_state='A' THEN a.siret END)>1 multiple_active_equivalents_present
      FROM t JOIN official_loss cat USING(query_id)
      LEFT JOIN acceptable a USING(query_id)
      LEFT JOIN master m ON a.query_id=m.query_id AND a.siret=m.siret GROUP BY t.query_id''')
    c.execute('''CREATE TEMP TABLE outcomes AS
      SELECT t.*,cat.official_catalog_present,cat.parent_in_evidence,
        p.siret IS NOT NULL site_in_reservoir,m.siret IS NOT NULL exact_hit,m.candidate_rank,
        (m.siret IS NOT NULL OR coalesce(op.operational_hit,false)) operational_hit,
        op.active_successor_present,op.multiple_active_equivalents_present,
        coalesce(p.base_rank<=1750,false) baseline_hit,
        CASE WHEN m.siret IS NOT NULL THEN 'HIT'
             WHEN NOT cat.official_catalog_present THEN 'OFFICIAL_CATALOG_ABSENT'
             WHEN NOT cat.parent_in_evidence THEN 'PARENT_NOT_RETRIEVED'
             WHEN p.siret IS NULL THEN 'SITE_NOT_EXPANDED'
             ELSE 'EVICTED_RESERVOIR_TO_MASTER2000' END loss_stage
      FROM t JOIN official_loss cat USING(query_id)
      LEFT JOIN candidates p ON t.query_id=p.query_id AND t.gt_siret=p.siret
      LEFT JOIN master m ON t.query_id=m.query_id AND t.gt_siret=m.siret
      JOIN operational op ON t.query_id=op.query_id''')
    df = c.execute('SELECT * FROM outcomes ORDER BY fold,query_id').df()
    if df.query_id.duplicated().any():
        raise ValueError('Duplicate outcomes')
    if not df.baseline_hit.eq(df.cohort_kind.eq('BASE_TOP1750_CONTROL')).all() or not df.loc[df.baseline_hit,'exact_hit'].all():
        raise ValueError('Baseline cohort drift or regression')
    return df


def run(config_path, generation, output):
    config = load_config(config_path)
    gm, gf = manifest_files(record(Path(generation)/'manifest.json'))
    if (gm['schema'] != 'sireto-v21-generation-1' or gm['preflight'] or gm['truth_columns_opened']
        or gm['config']['sha256'] != record(config_path)['sha256'] or gm['counts']['q'] != 1034
        or gm['baseline_generation'] != config['baseline_generation']):
        raise ValueError('Generation authority')
    if gm['implementation'] != implementation():
        raise ValueError('Implementation changed after generation')
    bm, bf = manifest_files(config['baseline_evaluation'])
    if (bm['schema'] != 'sireto-v20-master-evaluation-1'
        or bm['generation_manifest']['sha256'] != config['baseline_generation']['sha256']
        or bm['generation_manifest']['path'] != config['baseline_generation']['path']):
        raise ValueError('Truth authority')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    with Budget(config['resources']) as budget:
        c = open_workbench_db(output)
        for name, path in [('q',gf['queries.parquet']),('master',gf['master.parquet']),
                           ('candidates',gf['candidates.parquet']),('official_loss',bf['official_loss_sidecar.parquet']),
                           ('acceptable',bf['acceptable_operational.parquet'])]:
            c.execute(f'CREATE TEMP TABLE {name} AS SELECT * FROM read_parquet(?)', [str(path)])
        c.execute('''CREATE TEMP TABLE t AS SELECT query_id,fold,gt_siret,gt_siren,
          qualification_v2,qualification_v3,label_kind,data_origin,ground_truth_state,cohort_kind,
          exact_hit previous_exact_hit FROM read_parquet(?)''', [str(bf['outcomes.parquet'])])
        if c.execute('SELECT count(*) FROM t WHERE fold NOT IN (2,3,4)').fetchone()[0]:
            raise ValueError('Forbidden fold')
        df = evaluate_tables(c)
        if len(df) != 1034:
            raise ValueError('Incomplete evaluation')
        cap = c.execute('SELECT max(n) FROM (SELECT query_id,count(*) n FROM master GROUP BY query_id)').fetchone()[0]
        if cap > 2000:
            raise ValueError('Master cap')
        c.execute(f'COPY outcomes TO {sql_path(output/"outcomes.parquet")} (FORMAT PARQUET,COMPRESSION ZSTD)')
        dimensions = ('fold','data_origin','qualification_v2','qualification_v3','label_kind','ground_truth_state','cohort_kind')
        segments = {d: {str(k): summarize(v) for k,v in df.fillna({d:'UNKNOWN'}).groupby(d,dropna=False)} for d in dimensions}
        losses = {str(k):int(v) for k,v in df.loss_stage.value_counts().items()}
        rescues = int(df.loc[~df.baseline_hit,'exact_hit'].sum())
        reservoir = int(df.loc[~df.baseline_hit,'site_in_reservoir'].sum())
        c.close()
    report = {'schema': 'sireto-v21-evaluation-1', 'decision': 'PIVOT_GENERATION',
      'scope': config['scope'], 'config': record(config_path), 'generation_manifest': record(Path(generation)/'manifest.json'),
      'truth_authority': config['baseline_evaluation'], 'official_loss_sidecar': bm['files']['official_loss_sidecar.parquet'],
      'acceptable_operational_sidecar': bm['files']['acceptable_operational.parquet'],
      'primary': summarize(df), 'historical_all_human': summarize(df), 'segments': segments,
      'rescues_of_522': rescues, 'reservoir_exacts_of_522': reservoir, 'max_master_sirets': int(cap), 'loss_counts': losses,
      'conditional_projection': {'recall': (23049+rescues)/23571, 'reservoir_ceiling': (23049+reservoir)/23571,
        'measured_national_recall': False, 'assumption': 'same prefix preserved on all national queries'},
      'generalization': {'unseen_test': False, 'reused_development_labels': True,
        'no_fitted_parameters_in_this_recipe': True, 'cross_crm_claim_allowed': False,
        'external_shadow_test': 'MISSING_REQUIRED_BEFORE_CROSS_CRM_CLAIM'},
      'prerank500_measured': False, 'recall100_measured': False, 'ready_for_fold1': False,
      'fold0_opened': False, 'fold1_opened': False, 'resources': budget.report(),
      'files': {p.name:record(p) for p in output.iterdir() if p.is_file()}, 'implementation': implementation()}
    write_json(output/'manifest.json', report); print(json.dumps(report), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config',type=Path,required=True)
    p.add_argument('--generation',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a = p.parse_args(); run(a.config,a.generation,a.output)
