"""Independent replay of the frozen v28 OOF predictions and diagnostic metrics."""
import json
from pathlib import Path

import duckdb
import numpy as np
import pyarrow.parquet as pq
import xgboost as xgb

from src.xgb_matcher.retrieval_v20_workbench import record,verify
from src.xgb_matcher.retrieval_v21_membership import manifest_files
from scripts.retrieval_v28_geo_selection import FEATURES,training_sample

ROOT=Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100')
RUN='1f604bd8a92e9fd9'


def main():
    fm,ff=manifest_files(record(ROOT/'audits/retrieval_v28_geo_features'/RUN/'manifest.json'))
    em,ef=manifest_files(record(ROOT/'audits/retrieval_v28_geo_oof'/(RUN+'-sealed-fits')/'manifest.json'))
    recovery=json.loads(verify(em['completed_fits_recovery']).read_text())
    for name,source in recovery['files'].items():
        assert record(ef[name])['sha256']==source['sha256']
    assert not em['models_retrained'] and em['known_catalog_absence_retained']==1
    assert fm['config']['sha256']==em['config']['sha256']=='1f604bd8a92e9fd923d4792e228d414d44d8e224ec559cb7eb22bdd9a5446394'
    assert not fm['truth_columns_opened'] and not em['target_labels_used_for_selection'] and not em['positive_injection']
    assert fm['feature_order']==em['feature_order']==list(FEATURES)
    assert fm['implementation']==em['implementation']
    c=duckdb.connect();c.execute("SET threads=2;SET memory_limit='2GB'")
    for table,path in [('q',ff['queries.parquet']),('pool',ff['pool.parquet']),('features',ff['parent_features.parquet']),
                       ('old',ff['baseline_parents.parquet']),('selected',ef['selected.parquet']),('outcomes',ef['outcomes.parquet'])]:
        c.execute(f'CREATE TEMP TABLE {table} AS SELECT * FROM read_parquet(?)',[str(path)])
    assert c.execute('SELECT count(*),count(DISTINCT query_id) FROM q').fetchone()==(1034,1034)
    assert c.execute('SELECT DISTINCT fold FROM q ORDER BY 1').fetchall()==[(2,),(3,),(4,)]
    assert c.execute('SELECT count(*) FROM (SELECT query_id,count(*) n,count(DISTINCT siret) nd FROM pool GROUP BY query_id HAVING n>4096 OR n<>nd)').fetchone()[0]==0
    assert c.execute('SELECT count(*) FROM ((SELECT query_id,fold,siren FROM pool EXCEPT SELECT query_id,fold,siren FROM features) UNION ALL (SELECT query_id,fold,siren FROM features EXCEPT SELECT query_id,fold,siren FROM pool))').fetchone()[0]==0
    for file,table in [('queries','q'),('pool','pool'),('parent_features','features')]:
        pre=ROOT/'audits/retrieval_v28_geo_features/preflight16'/(file+'.parquet')
        c.execute('CREATE OR REPLACE TEMP TABLE pre AS SELECT * FROM read_parquet(?)',[str(pre)])
        assert c.execute(f'''SELECT count(*) FROM ((SELECT * FROM pre EXCEPT SELECT * FROM {table}) UNION ALL
          (SELECT * FROM {table} WHERE query_id IN (SELECT query_id FROM pre) EXCEPT SELECT * FROM pre))''').fetchone()[0]==0
    ledger=verify(dict(path=str(ROOT/'datasets/retrieval_all_human_v12_label_ledger/1a99b7cb1db5a1692e39/label_ledger.parquet'),sha256='24f5928cbf2669b0a79ac00a284d14e5fe2288abb5a4ddcbd77a46677c315720'))
    c.execute('''CREATE TEMP TABLE labels AS SELECT query_id,fold,human_ground_truth_siret gt_siret,human_ground_truth_siren gt_siren
      FROM read_parquet(?) SEMI JOIN q USING(query_id) WHERE fold IN (2,3,4)''',[str(ledger)])
    assert c.execute('SELECT count(*) FROM labels').fetchone()[0]==1034
    assert c.execute('''SELECT count(*) FROM labels l JOIN outcomes o USING(query_id) WHERE l.fold IS DISTINCT FROM o.fold
      OR l.gt_siret IS DISTINCT FROM o.gt_siret OR l.gt_siren IS DISTINCT FROM o.gt_siren''').fetchone()[0]==0
    frame=c.execute('SELECT f.*,l.gt_siren FROM features f JOIN labels l USING(query_id,fold) ORDER BY query_id,siren').df()
    for fold in (2,3,4):
        split=json.loads(ef[f'outer{fold}_split.json'].read_text())
        assert split['train_folds']==[f for f in (2,3,4) if f!=fold] and split['target_folds']==[fold]
        assert not set(split['trained_query_ids']) & set(split['target_query_ids'])
        X,y,groups,queries=training_sample(frame,split['train_folds'])
        assert queries==split['trained_query_ids'] and len(y)==split['training_rows'] and int(y.sum())==len(groups)
        del X,y,groups
        pred=pq.read_table(ef[f'outer{fold}_scores.parquet']).to_pandas()
        target=frame[frame.fold.eq(fold)]
        joined=pred.merge(target,on=['query_id','fold','siren','legacy_parent_rank'],validate='one_to_one')
        assert len(joined)==len(pred)==len(target)
        model=xgb.XGBRanker();model.load_model(ef[f'outer{fold}.json'])
        replay=model.predict(joined[list(FEATURES)].to_numpy(dtype=np.float32))
        assert np.array_equal(replay,joined.score.to_numpy())
        del model,target,joined,pred
    del frame
    c.execute('CREATE TEMP TABLE scores AS SELECT * FROM read_parquet(?)',[[str(ef[f'outer{f}_scores.parquet']) for f in (2,3,4)]])
    c.execute('''CREATE TEMP TABLE reconstructed AS SELECT * FROM (SELECT *,row_number() OVER(PARTITION BY query_id
      ORDER BY score DESC,legacy_parent_rank,siren) selected_rank FROM scores) WHERE selected_rank<=512''')
    assert c.execute('SELECT count(*) FROM ((SELECT * FROM reconstructed EXCEPT SELECT * FROM selected) UNION ALL (SELECT * FROM selected EXCEPT SELECT * FROM reconstructed))').fetchone()[0]==0
    assert c.execute('SELECT count(*) FROM selected ANTI JOIN features USING(query_id,fold,siren)').fetchone()[0]==0
    op=verify(em['operational_sidecar']);c.execute('CREATE TEMP TABLE op AS SELECT * FROM read_parquet(?)',[str(op)])
    for name,parent in [('baseline','old'),('oof','selected')]:
        c.execute(f'''CREATE OR REPLACE TEMP TABLE sites AS SELECT p.* FROM pool p JOIN {parent} s USING(query_id,siren)''')
        assert c.execute(f'''SELECT count(*) FROM outcomes o WHERE o.{name}_parent_hit IS DISTINCT FROM
          EXISTS(SELECT 1 FROM {parent} s WHERE s.query_id=o.query_id AND s.siren=o.gt_siren)
          OR o.{name}_expansion_exact IS DISTINCT FROM EXISTS(SELECT 1 FROM sites s WHERE s.query_id=o.query_id AND s.siret=o.gt_siret)
          OR o.{name}_expansion_operational IS DISTINCT FROM EXISTS(SELECT 1 FROM sites s JOIN op USING(query_id,siret) WHERE s.query_id=o.query_id)''').fetchone()[0]==0
    primary=c.execute('''SELECT count(*) queries,sum(baseline_parent_hit::INT) baseline,sum(oof_parent_hit::INT) oof,
      sum((oof_parent_hit AND NOT baseline_parent_hit)::INT) gains,sum((baseline_parent_hit AND NOT oof_parent_hit)::INT) losses,
      sum(exact_hit::INT) previous_master_exact,sum(operational_hit::INT) previous_master_operational FROM outcomes''').df().to_dict('records')[0]
    assert primary['baseline']==em['primary']['baseline_parent_hits'] and primary['oof']==em['primary']['oof_parent_hits']
    assert (primary['previous_master_exact'],primary['previous_master_operational'])==(823,825)
    control_losses=c.execute("SELECT count(*) FROM outcomes WHERE cohort_kind='BASE_TOP1750_CONTROL' AND baseline_parent_hit AND NOT oof_parent_hit").fetchone()[0]
    expected='ELIGIBLE_FOR_SEPARATE_END_TO_END_TEST' if primary['gains']>primary['losses'] and control_losses==0 else 'REJECT_NO_RETUNING'
    assert em['pilot_status']==expected
    print(json.dumps(dict(decision='PASS_V28_OUTPUT_RECONCILIATION',primary=primary,control_parent_losses=control_losses,
      pilot_status=expected,all_three_models_replayed_exactly=True,new_recall100_measured=False)))
    c.close()


if __name__=='__main__':main()
