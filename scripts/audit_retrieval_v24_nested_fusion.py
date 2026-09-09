#!/usr/bin/env python3
"""Read-only recipe replay and independent label-ledger check; never tunes."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.xgb_matcher.retrieval_v20_workbench import Budget, record, verify, write_json, sql_path
from src.xgb_matcher.retrieval_v21_membership import manifest_files, open_workbench_db
from src.xgb_matcher.retrieval_v24_nested_fusion import config_at, code_records


def audit(config_path, generation, evaluation, output):
    config = config_at(config_path)
    gm, gf = manifest_files(record(Path(generation)/'manifest.json'))
    em, ef = manifest_files(record(Path(evaluation)/'manifest.json'))
    _, pool = manifest_files(config['source_generation'])
    _, features = manifest_files(config['features'])
    _, models = manifest_files(config['models'])
    _, baseline = manifest_files(config['baseline_evaluation'])
    if gm['implementation'] != code_records() or em['generation_manifest'] != record(Path(generation)/'manifest.json'):
        raise ValueError('Implementation or evaluation lineage drift')
    ledger = verify(dict(path='/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/retrieval_all_human_v12_label_ledger/1a99b7cb1db5a1692e39/label_ledger.parquet',
                         sha256='24f5928cbf2669b0a79ac00a284d14e5fe2288abb5a4ddcbd77a46677c315720'))
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    with Budget(config['resources']) as budget:
        c = open_workbench_db(output)
        for name, path in [('q', gf['queries.parquet']), ('m', gf['master.parquet']), ('e', ef['outcomes.parquet']),
                           ('pool', pool['candidates.parquet']), ('sf', features['site_features.parquet']),
                           ('previous', baseline['outcomes.parquet']),
                           ('official', verify(em['official_loss_sidecar'])), ('acceptable', verify(em['acceptable_operational_sidecar']))]:
            c.execute(f'CREATE TEMP VIEW {name} AS SELECT * FROM read_parquet({sql_path(path)})')
        # Explicit projection and workbench-only selection; folds0/1 never enter audit.
        c.execute('''CREATE TEMP TABLE labels AS SELECT l.query_id,l.fold,
          l.human_ground_truth_siret gt_siret,l.human_ground_truth_siren gt_siren
          FROM read_parquet(?) l SEMI JOIN q USING(query_id,fold) WHERE l.fold IN (2,3,4)''', [str(ledger)])
        assert c.execute('SELECT count(*),count(DISTINCT query_id) FROM labels').fetchone() == (1034, 1034)
        assert c.execute('SELECT count(*) FROM labels l JOIN e USING(query_id,fold) WHERE l.gt_siret<>e.gt_siret OR l.gt_siren<>e.gt_siren').fetchone()[0] == 0
        assert c.execute('SELECT count(*) FROM m ANTI JOIN pool USING(query_id,fold,siren,siret)').fetchone()[0] == 0
        cap = c.execute('''SELECT max(n),max(mx),sum(n) FROM
          (SELECT query_id,count(*) n,max(candidate_rank) mx FROM m GROUP BY 1)''').fetchone()
        assert cap == (2000, 2000, 1973172)
        assert c.execute('SELECT count(DISTINCT (query_id,siret)),count(*) FROM m').fetchone() == (1973172, 1973172)
        assert c.execute('SELECT count(*) FROM q WHERE fold NOT IN (2,3,4)').fetchone()[0] == 0
        weights = []; calibrations = []
        for outer in (2, 3, 4):
            fit = json.loads(gf[f'outer{outer}_calibration.json'].read_text())
            other = [f for f in (2, 3, 4) if f != outer]
            tables = []
            for target in other:
                train = next(f for f in other if f != target)
                split = json.loads(models[f'inner_train{train}_split.json'].read_text())
                assert split['train_folds'] == [train] and target in split['target_folds']
                train_ids = {row[0] for row in c.execute('SELECT query_id FROM labels WHERE fold=?', [train]).fetchall()}
                assert set(split['trained_query_ids']) <= train_ids
                tables.append(c.execute('SELECT * FROM read_parquet(?) WHERE fold=?',
                                        [str(models[f'inner_train{train}_scores.parquet']), target]).df())
            c.register('inner_scores', pa.Table.from_pandas(pd.concat(tables), preserve_index=False))
            recomputed = c.execute('''WITH ranks AS (
              SELECT s.*,b.parent_rank,dense_rank() OVER(PARTITION BY s.query_id ORDER BY s.parent_score DESC) lr
              FROM inner_scores s JOIN (SELECT DISTINCT query_id,fold,siren,parent_rank FROM pool) b USING(query_id,fold,siren)
            ), probs AS (
              SELECT *, (1.0/(60+parent_rank))/sum(1.0/(60+parent_rank)) OVER(PARTITION BY query_id) bp,
                (1.0/(60+lr))/sum(1.0/(60+lr)) OVER(PARTITION BY query_id) lp FROM ranks
            ) SELECT p.query_id,p.fold,p.siren,p.bp,p.lp FROM probs p JOIN labels l USING(query_id,fold)
              WHERE p.siren=l.gt_siren ORDER BY query_id''').df()
            saved = c.execute('SELECT * FROM read_parquet(?) ORDER BY query_id',
                              [str(gf[f'outer{outer}_calibration_truth.parquet'])]).df()
            assert list(saved.query_id) == list(recomputed.query_id) == fit['fitted_query_ids']
            assert not (recomputed.fold == outer).any()
            assert np.allclose(saved.base_probability, recomputed.bp, rtol=1e-12, atol=1e-15)
            assert np.allclose(saved.learned_probability, recomputed.lp, rtol=1e-12, atol=1e-15)
            a = fit['learned_weight']; bp = recomputed.bp.to_numpy(); lp = recomputed.lp.to_numpy()
            derivative = float(-np.mean((lp-bp)/(bp+a*(lp-bp))) - .01/a + .01/(1-a))
            assert abs(derivative) < 1e-10  # strict convexity makes this the unique optimum
            assert set(fit['calibration_query_ids']) == {row[0] for row in c.execute('SELECT query_id FROM labels WHERE fold<>?', [outer]).fetchall()}
            weights.append(dict(fold=outer, learned_weight=a))
            calibrations.append(dict(outer=outer, fitted_queries=len(saved), weight=a, independent_derivative=derivative))
            split = json.loads(models[f'outer{outer}_split.json'].read_text())
            assert split['train_folds'] == other and split['target_folds'] == [outer]
            train_ids = {row[0] for row in c.execute('SELECT query_id FROM labels WHERE fold<>?', [outer]).fetchall()}
            target_ids = {row[0] for row in c.execute('SELECT query_id FROM labels WHERE fold=?', [outer]).fetchall()}
            assert set(split['trained_query_ids']) <= train_ids and set(split['target_query_ids']) == target_ids
        c.register('weights', pa.Table.from_pylist(weights))
        c.execute('''CREATE TEMP TABLE outer_scores AS SELECT *,dense_rank() OVER(
          PARTITION BY query_id ORDER BY parent_score DESC) lr FROM read_parquet(?)''',
                  [[str(models[f'outer{f}_scores.parquet']) for f in (2, 3, 4)]])
        c.execute('''CREATE TEMP TABLE replay AS WITH scored AS (
          SELECT s.query_id,s.fold,s.siren,s.siret,s.direct_rrf,p.parent_score,
            s.direct_rrf+(1-w.learned_weight)*(b.rescue_score-s.direct_rrf)
            +w.learned_weight*CASE WHEN s.local_site THEN 1.0/(60+p.lr+s.joint_site_rank) ELSE 0 END score
          FROM sf s JOIN pool b USING(query_id,fold,siren,siret)
          JOIN outer_scores p USING(query_id,fold,siren) JOIN weights w USING(fold)
        ) SELECT *,row_number() OVER(PARTITION BY query_id ORDER BY score DESC,direct_rrf DESC,parent_score DESC,siret) r
        FROM scored QUALIFY r<=2000''')
        assert c.execute('SELECT count(*) FROM replay').fetchone()[0] == 1973172
        assert c.execute('''SELECT count(*) FROM replay FULL JOIN m USING(query_id,fold,siren,siret)
          WHERE replay.r IS DISTINCT FROM m.candidate_rank OR abs(replay.score-m.rescue_score)>1e-14''').fetchone()[0] == 0
        c.execute('''CREATE TEMP TABLE independent AS SELECT l.*,
          EXISTS(SELECT 1 FROM m WHERE m.query_id=l.query_id AND m.siret=l.gt_siret) exact,
          EXISTS(SELECT 1 FROM pool p WHERE p.query_id=l.query_id AND p.siret=l.gt_siret) in_pool,
          EXISTS(SELECT 1 FROM m JOIN acceptable a USING(query_id,siret) WHERE m.query_id=l.query_id) acceptable_hit
          FROM labels l''')
        assert c.execute('SELECT count(*) FROM independent i JOIN e USING(query_id) WHERE i.exact<>e.exact_hit OR (i.exact OR i.acceptable_hit)<>e.operational_hit').fetchone()[0] == 0
        counts = c.execute('SELECT count(*),sum(CAST(exact AS INT)),sum(CAST(exact OR acceptable_hit AS INT)),sum(CAST(in_pool AS INT)) FROM independent').fetchone()
        assert counts[1:3] == (em['primary']['exact_hits'], em['primary']['operational_hits'])
        assert c.execute('''SELECT count(*) FROM independent i JOIN e USING(query_id) JOIN official o USING(query_id)
          WHERE e.loss_stage <> CASE WHEN i.exact THEN 'HIT' WHEN NOT o.official_catalog_present THEN 'OFFICIAL_CATALOG_ABSENT'
            WHEN NOT o.parent_in_evidence THEN 'PARENT_NOT_RETRIEVED' WHEN NOT i.in_pool THEN 'SITE_NOT_EXPANDED'
            ELSE 'EVICTED_RESERVOIR_TO_MASTER2000' END''').fetchone()[0] == 0
        changes = c.execute('''SELECT sum(CAST(i.exact AND NOT p.exact_hit AS INT)),sum(CAST(NOT i.exact AND p.exact_hit AS INT))
          FROM independent i JOIN previous p USING(query_id)''').fetchone()
        assert changes == (em['primary']['gains_vs_v23'], em['primary']['losses_vs_v23'])
        c.close()
    result = dict(schema='sireto-v24-independent-audit-1', passed=True, config=record(config_path),
        generation_manifest=record(Path(generation)/'manifest.json'), evaluation_manifest=record(Path(evaluation)/'manifest.json'),
        ledger=record(ledger), implementation=record(Path(__file__)), calibrations=calibrations,
        counts=dict(queries=counts[0], exact=counts[1], operational=counts[2], natural_pool_hits=counts[3],
                    gains=changes[0], losses=changes[1], master_rows=cap[2], max_candidates=cap[0]),
        folds_checked=[2, 3, 4], recipe_tuned=False, resources=budget.report())
    write_json(output/'manifest.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('config', 'generation', 'evaluation', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); audit(a.config, a.generation, a.evaluation, a.output)
