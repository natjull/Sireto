"""Frozen v28 membership replacement and frozen v23 inference; labels only in evaluate."""
import argparse
import gc
import json
from pathlib import Path
import sys

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.xgb_matcher.retrieval_v20_workbench import ROOT, Budget, record, verify, write_json, sql_path
from src.xgb_matcher.retrieval_v21_membership import manifest_files, open_workbench_db, build_tables
from src.xgb_matcher.retrieval_v23_joint_parent import FEATURES, NAME_SELECTION_SQL, build_features, make_master, code_records
from src.xgb_matcher.retrieval_parent_fusion_v19 import validate_fold_isolation
from scripts.retrieval_v22_global_budget import outcomes, summary


def config_at(path):
    cfg = json.loads(Path(path).read_text())
    if (cfg['schema'] != 'sireto-v29-consolidated-1'
        or cfg['recipe'] != 'REPLACE_V17_GEO_MEMBERSHIP_WITH_FROZEN_V28_THEN_FROZEN_V23'
        or cfg['folds'] != [2, 3, 4] or cfg['master_cap'] != 2000 or cfg['geo_parent_cap'] != 512
        or any(cfg[k] for k in ('retrain', 'national_run', 'production_activation'))):
        raise ValueError('Frozen recipe drift')
    return cfg


def implementation():
    return {**code_records(), 'scripts/retrieval_v29_consolidated.py': record(Path(__file__)),
            'docs/retrieval_v29_consolidated_contract.md': record(ROOT/'docs/retrieval_v29_consolidated_contract.md')}


def versions():
    return dict(duckdb=duckdb.__version__, numpy=np.__version__, pyarrow=pa.__version__, xgboost=xgb.__version__)


def save(output, config, result):
    result.update(config=record(config), implementation=implementation(), versions=versions(),
                  fold0_opened=False, fold1_opened=False, production_activation=False)
    result['files'] = {p.name: record(p) for p in output.iterdir() if p.is_file()}
    write_json(output/'manifest.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('files', 'implementation', 'segments')}), flush=True)


def equal_sql(c, left, right):
    if c.execute(f'SELECT count(*) FROM ((({left}) EXCEPT ALL ({right})) UNION ALL (({right}) EXCEPT ALL ({left})))').fetchone()[0]:
        raise ValueError('Replay/set equality failed')


def check_q(c, expected):
    if c.execute('SELECT count(*),count(DISTINCT query_id) FROM q').fetchone() != (expected, expected):
        raise ValueError('Query population')
    if c.execute('SELECT count(*) FROM q WHERE fold NOT IN (2,3,4) OR fold IS NULL').fetchone()[0]:
        raise ValueError('Forbidden fold')


def replacement_union(c):
    """Only source-membership flags and frozen OOF ranks; no truth-side input."""
    c.execute('''CREATE TEMP TABLE new_union AS
      SELECT query_id,fold,siren FROM presence WHERE from_text OR from_address
      UNION SELECT query_id,fold,siren FROM selected''')
    c.execute('''CREATE TEMP TABLE replacement_signals AS
      SELECT * FROM signals WHERE channel<>'v17_union'
      UNION ALL SELECT query_id,fold,siren,'' source_siret,'v17_union' channel,
        row_number() OVER(PARTITION BY query_id ORDER BY siren)::INTEGER channel_rank FROM new_union''')
    equal_sql(c, "SELECT * FROM signals WHERE channel<>'v17_union'",
              "SELECT * FROM replacement_signals WHERE channel<>'v17_union'")
    c.execute('DROP TABLE signals')
    c.execute('ALTER TABLE replacement_signals RENAME TO signals')


def predict(c, models, output=None, reference=None):
    predictions = []
    for fold in (2, 3, 4):
        split = json.loads(models[f'outer{fold}_split.json'].read_text())
        if split['train_folds'] != [f for f in (2, 3, 4) if f != fold] or split['target_folds'] != [fold]:
            raise ValueError('Model split')
        frame = c.execute('SELECT * FROM parent_features WHERE fold=? ORDER BY query_id,siren', [fold]).df()
        if not set(frame.query_id) <= set(split['target_query_ids']):
            raise ValueError('Foreign model target')
        if not len(frame):
            continue
        values = frame[list(FEATURES)].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite features')
        model = xgb.XGBRanker(n_jobs=2); model.load_model(models[f'outer{fold}.json'])
        scores = model.predict(values)
        if not np.isfinite(scores).all():
            raise ValueError('Nonfinite scores')
        pred = frame[['query_id', 'fold', 'siren']].copy(); pred['parent_score'] = scores
        if reference:
            old = pq.read_table(models[f'outer{fold}_scores.parquet']).to_pandas()
            old = old[old.query_id.isin(frame.query_id)].sort_values(['query_id', 'siren'])
            if (list(zip(old.query_id, old.siren)) != list(zip(pred.query_id, pred.siren))
                or not np.array_equal(old.parent_score.to_numpy(), scores)):
                raise ValueError('Frozen model score replay')
        tab = pa.Table.from_pandas(pred, preserve_index=False); predictions.append(tab)
        if output:
            pq.write_table(tab, output/f'outer{fold}_scores.parquet', compression='zstd')
        del frame, values, model, pred; gc.collect()
    c.register('parent_scores', pa.concat_tables(predictions))


def generate(config, output, preflight=None):
    cfg = config_at(config)
    if preflight is not None and not 1 <= preflight <= 16:
        raise ValueError('Preflight1..16')
    v23 = json.loads(verify(cfg['v23_config']).read_text())
    v19 = json.loads(verify(cfg['v19_config']).read_text())
    v18 = json.loads(verify(v23['v18_config']).read_text())['generation_authorities']
    gm, gf = manifest_files(v23['source_generation'])
    fm, ff = manifest_files(cfg['features']); mm, mf = manifest_files(cfg['models'])
    geo, geof = manifest_files(cfg['geo'])
    if (gm['truth_columns_opened'] or gm['preflight'] or fm['truth_columns_opened'] or fm['preflight']
        or fm['implementation'] != code_records() or fm['feature_order'] != list(FEATURES)
        or mm['features_manifest']['sha256'] != cfg['features']['sha256']
        or mm['target_labels_used_for_selection'] or mm['positive_injection']
        or geo['target_labels_used_for_selection'] or geo['positive_injection']
        or geo['schema'] != 'sireto-v28-oof-evaluation-1'):
        raise ValueError('Source lineage')
    sources = {k: verify(v19['authorities'][k]) for k in ('structured_signals', 'siret_native_signals')}
    sources.update(presence=verify(v18['v17_parent_presence']), names=verify(v23['names']), sites=verify(v23['sites']))
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    with Budget(cfg['resources']) as budget:
        c = open_workbench_db(output); c.execute("SET memory_limit='8GB'"); c.execute('SET threads=2')
        c.execute('CREATE TEMP TABLE q AS SELECT * FROM read_parquet(?) ORDER BY query_id LIMIT ?',
                  [str(ff['queries.parquet']), preflight or 1034]); check_q(c, preflight or 1034)
        c.execute('''CREATE TEMP TABLE raw AS SELECT p.query_id,p.fold,p.siren,p.siret,p.base_rank
          FROM read_parquet(?) p SEMI JOIN q USING(query_id,fold) WHERE base_rank IS NOT NULL''', [str(gf['candidates.parquet'])])
        c.execute('''CREATE TEMP TABLE signals AS SELECT s.query_id,s.fold,s.siren,s.source_siret,s.channel,s.channel_rank
          FROM read_parquet(?) s SEMI JOIN q USING(query_id,fold)''', [[str(sources[k]) for k in ('structured_signals', 'siret_native_signals')]])
        c.execute('CREATE TEMP TABLE presence AS SELECT p.* FROM read_parquet(?) p SEMI JOIN q USING(query_id,fold)', [str(sources['presence'])])
        equal_sql(c, "SELECT query_id,fold,siren FROM signals WHERE channel='v17_union'", 'SELECT query_id,fold,siren FROM presence')
        c.execute('CREATE TEMP TABLE geo_scores AS SELECT s.* FROM read_parquet(?) s SEMI JOIN q USING(query_id,fold)',
                  [[str(geof[f'outer{f}_scores.parquet']) for f in (2, 3, 4)]])
        c.execute('''CREATE TEMP TABLE selected AS SELECT * FROM (SELECT *,row_number() OVER(PARTITION BY query_id
          ORDER BY score DESC,legacy_parent_rank,siren) selected_rank FROM geo_scores) WHERE selected_rank<=512''')
        equal_sql(c, 'SELECT * FROM selected', f'SELECT s.* FROM read_parquet({sql_path(geof["selected.parquet"])}) s SEMI JOIN q USING(query_id,fold)')
        for f in (2, 3, 4):
            split = json.loads(geof[f'outer{f}_split.json'].read_text())
            if split['train_folds'] != [x for x in (2, 3, 4) if x != f] or split['target_folds'] != [f]:
                raise ValueError('Geo OOF split')
            target = {r[0] for r in c.execute('SELECT DISTINCT query_id FROM geo_scores WHERE fold=?', [f]).fetchall()}
            if not target <= set(split['target_query_ids']):
                raise ValueError('Geo foreign target')
        c.execute('''CREATE TEMP TABLE sites AS SELECT s.siren,s.siret,s.insee,s.postcode,s.street_number,
          s.street_number_suffix,s.street_normalized,s.administrative_state FROM read_parquet(?) s
          SEMI JOIN (SELECT siren FROM signals UNION SELECT siren FROM raw UNION SELECT siren FROM selected) p USING(siren)''', [str(sources['sites'])])
        # Original reservoir and frozen inference are replayed before any modification.
        build_tables(c)
        equal_sql(c, 'SELECT * FROM candidates', f'SELECT p.* FROM read_parquet({sql_path(gf["candidates.parquet"])}) p SEMI JOIN q USING(query_id,fold)')
        equal_sql(c, 'SELECT * FROM witnesses', f'SELECT p.* FROM read_parquet({sql_path(gf["witnesses.parquet"])}) p SEMI JOIN q USING(query_id,fold)')
        c.execute('DROP TABLE master')
        c.execute('CREATE TEMP TABLE parent_features AS SELECT p.* FROM read_parquet(?) p SEMI JOIN q USING(query_id,fold)', [str(ff['parent_features.parquet'])])
        c.execute('CREATE TEMP TABLE site_features AS SELECT p.* FROM read_parquet(?) p SEMI JOIN q USING(query_id,fold)', [str(ff['site_features.parquet'])])
        predict(c, mf, reference=True); make_master(c)
        equal_sql(c, 'SELECT * FROM master', f'SELECT m.* FROM read_parquet({sql_path(mf["master.parquet"])}) m SEMI JOIN q USING(query_id,fold)')
        print('PASS_BASELINE_RESERVOIR_SCORES_MASTER_REPLAY', flush=True)
        for t in ('parents', 'witnesses', 'direct', 'address_sites', 'candidates', 'master', 'parent_features', 'site_features', 'ranked_parents'):
            c.execute(f'DROP TABLE {t}')
        c.unregister('parent_scores')
        replacement_union(c); build_tables(c); c.execute('DROP TABLE master')
        c.execute('CREATE TEMP TABLE pool AS SELECT query_id,fold,siren,siret,base_rank,parent_rrf,direct_rrf FROM candidates')
        c.execute(f'CREATE TEMP VIEW name_source AS SELECT * FROM read_parquet({sql_path(sources["names"])})')
        c.execute(NAME_SELECTION_SQL); build_features(c)
        predict(c, mf, output=output); make_master(c)
        if c.execute('SELECT count(*) FROM master ANTI JOIN candidates USING(query_id,fold,siren,siret)').fetchone()[0]:
            raise ValueError('Non-natural output')
        tables = dict(queries='q', candidates='candidates', parents='parents', witnesses='witnesses',
                      parent_features='parent_features', site_features='site_features', selected='selected', master='master')
        for name, table in tables.items():
            order = 'query_id,candidate_rank' if name == 'master' else 'query_id'
            c.execute(f'COPY (SELECT * FROM {table} ORDER BY {order}) TO {sql_path(output/(name+".parquet"))} (FORMAT PARQUET,COMPRESSION ZSTD)')
        counts = {name: c.execute(f'SELECT count(*) FROM {table}').fetchone()[0] for name, table in tables.items()}
        c.close()
    save(output, config, dict(schema='sireto-v29-generation-1', preflight=preflight is not None,
         counts=counts, sources={k: record(p) for k, p in sources.items()},
         authority_manifests={k: cfg[k] for k in ('features', 'models', 'geo')},
         baseline_reservoir_replayed=True, baseline_scores_master_replayed=True,
         truth_columns_opened=False, positive_injection=False, model_fit_performed=False,
         site_model_scoring_performed=False, rich_hydration=False, resources=budget.report()))


def summarize(df):
    result = summary(df)
    result['gains_vs_v23'] = result.pop('gains_vs_v21'); result['losses_vs_v23'] = result.pop('losses_vs_v21')
    result['net_vs_v23'] = result['gains_vs_v23'] - result['losses_vs_v23']
    for col in ('old_reservoir_hit', 'site_in_reservoir', 'geo_exact_present', 'geo_exact_new_to_old_reservoir'):
        result[col] = int(df[col].sum())
    result['generation_gains'] = int((df.site_in_reservoir & ~df.old_reservoir_hit).sum())
    result['generation_losses'] = int((~df.site_in_reservoir & df.old_reservoir_hit).sum())
    return result


def evaluate(config, generation, output):
    cfg = config_at(config); gm, gf = manifest_files(record(Path(generation)/'manifest.json'))
    if (gm['schema'] != 'sireto-v29-generation-1' or gm['preflight'] or gm['truth_columns_opened']
        or gm['config'] != record(config) or gm['implementation'] != implementation() or gm['versions'] != versions()
        or gm['counts']['queries'] != 1034 or gm['positive_injection'] or gm['model_fit_performed']):
        raise ValueError('Generation authority')
    bm, bf = manifest_files(cfg['baseline'])
    if bm['generation_manifest']['sha256'] != cfg['models']['sha256']:
        raise ValueError('Baseline mismatch')
    side = verify(bm['official_loss_sidecar']); op = verify(bm['acceptable_operational_sidecar'])
    geo, geof = manifest_files(cfg['geo']); geom, geop = manifest_files(geo['features_manifest'])
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    with Budget(cfg['resources']) as budget:
        c = open_workbench_db(output); c.execute("SET memory_limit='8GB'"); c.execute('SET threads=2')
        for name, path in [('q', gf['queries.parquet']), ('master', gf['master.parquet']), ('pool', gf['candidates.parquet']),
                           ('parents', gf['parents.parquet']), ('catalog', side), ('acceptable', op), ('selected', gf['selected.parquet'])]:
            c.execute(f'CREATE TEMP TABLE {name} AS SELECT * FROM read_parquet(?)', [str(path)])
        check_q(c, 1034)
        c.execute('''CREATE TEMP TABLE baseline AS SELECT query_id,fold,gt_siret,gt_siren,cohort_kind,
          qualification_v2,qualification_v3,label_kind,data_origin,ground_truth_state,site_in_reservoir old_reservoir_hit,
          exact_hit previous_exact_hit,operational_hit previous_operational_hit FROM read_parquet(?) WHERE fold IN (2,3,4)''', [str(bf['outcomes.parquet'])])
        truth = c.execute('SELECT query_id,fold,gt_siren FROM baseline').df(); validate_fold_isolation(truth)
        if c.execute('SELECT count(*) FROM baseline ANTI JOIN catalog USING(query_id)').fetchone()[0]:
            raise ValueError('Missing official sidecar')
        if c.execute('SELECT count(*) FROM catalog GROUP BY query_id HAVING count(*)<>1').fetchall():
            raise ValueError('Duplicate official sidecar')
        missing = c.execute('''SELECT count(*) FROM baseline t JOIN catalog cat USING(query_id)
          ANTI JOIN acceptable a ON a.query_id=t.query_id AND a.siret=t.gt_siret
          WHERE cat.official_catalog_present OR EXISTS(SELECT 1 FROM pool p WHERE p.query_id=t.query_id AND p.siret=t.gt_siret)''').fetchone()[0]
        if missing:
            raise ValueError('Missing operational exact beyond catalog absence')
        c.execute('''CREATE TEMP TABLE geo_sites AS SELECT p.query_id,p.fold,p.siren,p.siret FROM read_parquet(?) p
          JOIN selected s USING(query_id,fold,siren)''', [str(geop['pool.parquet'])])
        c.execute('''CREATE TEMP TABLE t AS SELECT b.*,
          EXISTS(SELECT 1 FROM pool p WHERE p.query_id=b.query_id AND p.siret=b.gt_siret) site_in_reservoir,
          EXISTS(SELECT 1 FROM geo_sites p WHERE p.query_id=b.query_id AND p.siret=b.gt_siret) geo_exact_present,
          (EXISTS(SELECT 1 FROM geo_sites p WHERE p.query_id=b.query_id AND p.siret=b.gt_siret) AND NOT b.old_reservoir_hit) geo_exact_new_to_old_reservoir
          FROM baseline b''')
        c.execute('''CREATE TEMP TABLE official_loss AS SELECT cat.* EXCLUDE(parent_in_evidence),
          EXISTS(SELECT 1 FROM parents p WHERE p.query_id=cat.query_id AND p.siren=t.gt_siren) parent_in_evidence
          FROM catalog cat JOIN t USING(query_id)''')
        df = outcomes(c)
        if len(df) != 1034:
            raise ValueError('Denominator drift')
        cap = c.execute('SELECT max(n) FROM (SELECT query_id,count(*) n FROM master GROUP BY 1)').fetchone()[0]
        if cap > 2000:
            raise ValueError('Master cap')
        for name in ('outcomes', 'official_loss'):
            c.execute(f'COPY {name} TO {sql_path(output/(name+".parquet"))} (FORMAT PARQUET,COMPRESSION ZSTD)')
        segments = {dim: {str(k): summarize(v) for k, v in df.groupby(dim, dropna=False)} for dim in
                    ('fold', 'cohort_kind', 'qualification_v2', 'qualification_v3', 'data_origin', 'label_kind', 'ground_truth_state')}
        primary = summarize(df); c.close()
    save(output, config, dict(schema='sireto-v29-evaluation-1', generation_manifest=record(Path(generation)/'manifest.json'),
         baseline=cfg['baseline'], official_loss_sidecar=record(side), operational_sidecar=record(op),
         primary=primary, historical_all_human=primary, segments=segments,
         loss_counts={str(k): int(v) for k, v in df.loss_stage.value_counts().items()}, max_master_sirets=int(cap),
         experiment_status='NET_GAIN_REUSED_PANEL' if primary['net_vs_v23'] > 0 else 'NO_NET_GAIN_REUSED_PANEL',
         decision='PIVOT_GENERATION', ready_for_fold1=False, recall100_measured=False, prerank500_measured=False,
         national_projection=None, reused_development_panel=True, resources=budget.report()))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['generate', 'evaluate'])
    p.add_argument('--config', type=Path, default=Path('config/retrieval_v29_consolidated.json'))
    p.add_argument('--output', type=Path, required=True); p.add_argument('--generation', type=Path); p.add_argument('--preflight', type=int)
    a = p.parse_args()
    if a.action == 'generate':
        if a.generation: p.error('No generation input for generate')
        generate(a.config, a.output, a.preflight)
    else:
        if not a.generation or a.preflight: p.error('Full generation required')
        evaluate(a.config, a.generation, a.output)
