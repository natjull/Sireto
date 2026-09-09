#!/usr/bin/env python3
"""Frozen nested parent calibration and separate all-human evaluation."""
import argparse
import gc
import json
from pathlib import Path
import sys

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.xgb_matcher.retrieval_v20_workbench import Budget, record, verify, write_json, sql_path
from src.xgb_matcher.retrieval_v21_membership import manifest_files, open_workbench_db
from src.xgb_matcher.retrieval_parent_fusion_v19 import validate_fold_isolation
from src.xgb_matcher.retrieval_v24_nested_fusion import (
    config_at, code_records, inner_jobs, check_split, calibrate, parent_probabilities, make_master,
)
from scripts.retrieval_v22_global_budget import outcomes, summary as old_summary


def versions():
    return dict(duckdb=duckdb.__version__, numpy=np.__version__, pandas=pd.__version__, pyarrow=pa.__version__)


def save_manifest(output, result):
    result['implementation'] = code_records()
    result['runtime_versions'] = versions()
    result['files'] = {p.name: record(p) for p in output.iterdir() if p.is_file()}
    write_json(output/'manifest.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('files', 'implementation')}), flush=True)


def generate(config_path, output):
    config = config_at(config_path)
    gm, gf = manifest_files(config['source_generation'])
    fm, ff = manifest_files(config['features'])
    mm, mf = manifest_files(config['models'])
    bm, bf = manifest_files(config['baseline_evaluation'])
    if (gm['schema'] != 'sireto-v21-generation-1' or gm['preflight'] or gm['truth_columns_opened']
        or fm['schema'] != 'sireto-v23-features-1' or fm['preflight'] or fm['truth_columns_opened']
        or fm['source_generation'] != config['source_generation']
        or mm['schema'] != 'sireto-v23-oof-master-1' or mm['target_labels_used_for_selection']
        or mm['positive_injection'] or mm['features_manifest']['sha256'] != config['features']['sha256']
        or bm['schema'] != 'sireto-v23-evaluation-1'
        or bm['generation_manifest']['sha256'] != config['models']['sha256']):
        raise ValueError('Sealed source lineage mismatch')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    with Budget(config['resources']) as budget:
        c = open_workbench_db(output)
        truth = pq.read_table(bf['outcomes.parquet'], columns=['query_id', 'fold', 'gt_siren']).to_pandas()
        validate_fold_isolation(truth)
        q = pq.read_table(ff['queries.parquet'])
        if (len(truth) != 1034 or set(map(tuple, q.select(['query_id', 'fold']).to_pandas().to_numpy()))
            != set(map(tuple, truth[['query_id', 'fold']].to_numpy()))):
            raise ValueError('Query population drift')
        c.register('q', q)
        c.execute('''CREATE TEMP TABLE pool AS SELECT query_id,fold,siren,siret,parent_rank,
          direct_rrf,rescue_score FROM read_parquet(?)''', [str(gf['candidates.parquet'])])
        if c.execute('SELECT count(*) FROM pool ANTI JOIN q USING(query_id,fold)').fetchone()[0]:
            raise ValueError('Foreign pool query')
        if c.execute('''SELECT count(*) FROM (SELECT query_id,fold,siren,count(DISTINCT parent_rank) n
          FROM pool GROUP BY ALL HAVING n<>1)''').fetchone()[0]:
            raise ValueError('Nonconstant parent evidence rank')
        c.execute('CREATE TEMP TABLE base_parents AS SELECT DISTINCT query_id,fold,siren,parent_rank FROM pool')
        c.execute('CREATE TEMP TABLE site_features AS SELECT * FROM read_parquet(?)', [str(ff['site_features.parquet'])])
        splits = {}
        for f in (2, 3, 4):
            for name, train, target in [(f'outer{f}', [x for x in (2, 3, 4) if x != f], [f]),
                                         (f'inner_train{f}', [f], [x for x in (2, 3, 4) if x != f])]:
                splits[name] = json.loads(mf[name+'_split.json'].read_text())
                check_split(splits[name], train, target, truth)
        weights = []; records = []; outer_scores = []
        for outer in (2, 3, 4):
            tables = []
            for train, target in inner_jobs(outer):
                name = f'inner_train{train}'
                if outer in splits[name]['train_folds'] or target not in splits[name]['target_folds']:
                    raise ValueError('Outer fold entered inner model')
                table = pq.read_table(mf[name+'_scores.parquet'], filters=[('fold', '=', target)])
                if set(table['query_id'].to_pylist()) != set(truth.loc[truth.fold.eq(target), 'query_id']):
                    raise ValueError('Inner prediction query population')
                tables.append(table)
            probabilities = parent_probabilities(c, pa.concat_tables(tables))
            fit = calibrate(probabilities, truth, outer)
            fit['inner_jobs'] = [{'model': f'inner_train{train}', 'train_fold': train, 'score_fold': target}
                                 for train, target in inner_jobs(outer)]
            # Truth-side audit only; this table is not an input to site selection.
            positive = probabilities.merge(truth.loc[truth.fold.ne(outer)], on=['query_id', 'fold'], validate='many_to_one')
            positive = positive.loc[positive.siren.eq(positive.gt_siren)].sort_values('query_id')
            pq.write_table(pa.Table.from_pandas(positive, preserve_index=False), output/f'outer{outer}_calibration_truth.parquet', compression='zstd')
            write_json(output/f'outer{outer}_calibration.json', fit)
            records.append({k: v for k, v in fit.items() if not k.endswith('_ids')})
            weights.append(dict(fold=outer, learned_weight=fit['learned_weight']))
            predicted = pq.read_table(mf[f'outer{outer}_scores.parquet'])
            if (set(predicted['fold'].to_pylist()) != {outer}
                or set(predicted['query_id'].to_pylist()) != set(truth.loc[truth.fold.eq(outer), 'query_id'])):
                raise ValueError('Outer prediction population')
            parent_probabilities(c, predicted)  # complete/unique natural parent validation
            outer_scores.append(predicted)
            print(json.dumps(records[-1]), flush=True)
            del probabilities, positive, tables; gc.collect()
        c.register('parent_scores', pa.concat_tables(outer_scores))
        c.register('fold_weights', pa.Table.from_pylist(weights))
        make_master(c)
        for name, table, order in [('queries', 'q', 'query_id'), ('master', 'master', 'query_id,candidate_rank')]:
            c.execute(f'COPY (SELECT * FROM {table} ORDER BY {order}) TO {sql_path(output/(name+".parquet"))} (FORMAT PARQUET,COMPRESSION ZSTD)')
        counts = {t: c.execute(f'SELECT count(*) FROM {t}').fetchone()[0] for t in ('q', 'pool', 'base_parents', 'master')}
        c.close()
    save_manifest(output, dict(schema='sireto-v24-generation-1', config=record(config_path),
        authorities={k: config[k] for k in ('source_generation', 'features', 'models', 'baseline_evaluation')},
        counts=counts, calibrations=records, target_labels_used_for_selection=False,
        fold0_opened=False, fold1_opened=False, positive_injection=False,
        site_model_scoring_performed=False, resources=budget.report()))


def summarize(df):
    result = old_summary(df)
    result['gains_vs_v23'] = result.pop('gains_vs_v21')
    result['losses_vs_v23'] = result.pop('losses_vs_v21')
    return result


def evaluate(config_path, generation, output):
    config = config_at(config_path)
    gm, gf = manifest_files(record(Path(generation)/'manifest.json'))
    if (gm['schema'] != 'sireto-v24-generation-1' or gm['counts']['q'] != 1034
        or gm['target_labels_used_for_selection'] or gm['positive_injection']
        or gm['config']['sha256'] != record(config_path)['sha256'] or gm['implementation'] != code_records()
        or gm['runtime_versions'] != versions()
        or gm['authorities'] != {k: config[k] for k in ('source_generation', 'features', 'models', 'baseline_evaluation')}):
        raise ValueError('Generation authority mismatch')
    bm, bf = manifest_files(config['baseline_evaluation'])
    side = verify(bm['official_loss_sidecar']); acceptable = verify(bm['acceptable_operational_sidecar'])
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    with Budget(config['resources']) as budget:
        c = open_workbench_db(output)
        for name, path in [('q', gf['queries.parquet']), ('master', gf['master.parquet']),
                           ('official_loss', side), ('acceptable', acceptable)]:
            c.execute(f'CREATE TEMP TABLE {name} AS SELECT * FROM read_parquet(?)', [str(path)])
        c.execute('''CREATE TEMP TABLE t AS SELECT query_id,fold,gt_siret,gt_siren,cohort_kind,
          qualification_v2,qualification_v3,label_kind,data_origin,ground_truth_state,site_in_reservoir,
          exact_hit previous_exact_hit FROM read_parquet(?)''', [str(bf['outcomes.parquet'])])
        df = outcomes(c)
        if len(df) != 1034:
            raise ValueError('Incomplete evaluation')
        cap = c.execute('SELECT max(n) FROM (SELECT query_id,count(*) n FROM master GROUP BY 1)').fetchone()[0]
        if cap > 2000:
            raise ValueError('Master cap')
        c.execute(f'COPY outcomes TO {sql_path(output/"outcomes.parquet")} (FORMAT PARQUET,COMPRESSION ZSTD)')
        segments = {d: {str(k): summarize(v) for k, v in df.fillna({d: 'UNKNOWN'}).groupby(d, dropna=False)}
                    for d in ('fold', 'cohort_kind', 'data_origin', 'qualification_v2', 'qualification_v3', 'label_kind', 'ground_truth_state')}
        c.close()
    save_manifest(output, dict(schema='sireto-v24-evaluation-1', decision='PIVOT_GENERATION', config=record(config_path),
        generation_manifest=record(Path(generation)/'manifest.json'), baseline_evaluation=config['baseline_evaluation'],
        official_loss_sidecar=record(side), acceptable_operational_sidecar=record(acceptable),
        primary=summarize(df), historical_all_human=summarize(df), segments=segments, max_master_sirets=int(cap),
        loss_counts={str(k): int(v) for k, v in df.loss_stage.value_counts().items()},
        national_projection=None, unseen_external_validation=False, reused_development_panel=True,
        prerank500_measured=False, recall100_measured=False, fold0_opened=False, fold1_opened=False,
        ready_for_fold1=False, resources=budget.report()))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['generate', 'evaluate'])
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--generation', type=Path)
    a = p.parse_args()
    if a.action == 'generate':
        if a.generation:
            p.error('generation is only for evaluation')
        generate(a.config, a.output)
    else:
        if not a.generation:
            p.error('evaluate requires generation')
        evaluate(a.config, a.generation, a.output)
