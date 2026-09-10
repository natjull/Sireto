"""Independent read-only reconciliation of the frozen v29 run and outcome tables."""
import argparse
import gc
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq
import xgboost as xgb

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.xgb_matcher.retrieval_v20_workbench import Budget,record,verify,write_json,sql_path
from src.xgb_matcher.retrieval_v21_membership import manifest_files,open_workbench_db
from src.xgb_matcher.retrieval_v23_joint_parent import FEATURES
from src.xgb_matcher.retrieval_parent_fusion_v19 import validate_fold_isolation


def same(c,a,b):
    assert c.execute(f'SELECT count(*) FROM ((({a}) EXCEPT ALL ({b})) UNION ALL (({b}) EXCEPT ALL ({a})))').fetchone()[0]==0


def audit(config,generation,evaluation,preflight,output):
    cfg=json.loads(Path(config).read_text())
    gm,gf=manifest_files(record(Path(generation)/'manifest.json'))
    em,ef=manifest_files(record(Path(evaluation)/'manifest.json'))
    pm,pf=manifest_files(record(Path(preflight)/'manifest.json'))
    bm,bf=manifest_files(cfg['baseline']);mm,mf=manifest_files(cfg['models']);geo,geof=manifest_files(cfg['geo'])
    v23=json.loads(verify(cfg['v23_config']).read_text());oldm,oldf=manifest_files(v23['source_generation'])
    v19=json.loads(verify(cfg['v19_config']).read_text())
    ledger=verify(v19['authorities']['label_ledger'])
    side=verify(em['official_loss_sidecar']);acceptable=verify(em['operational_sidecar'])
    sources={k:verify(v) for k,v in gm['sources'].items()}
    assert gm['config']==em['config']==pm['config']==record(config)
    assert gm['implementation']==em['implementation']==pm['implementation']
    assert not gm['preflight'] and pm['preflight'] and gm['counts']['queries']==1034
    assert em['generation_manifest']==record(Path(generation)/'manifest.json')
    assert em['baseline']==cfg['baseline'] and em['decision']=='PIVOT_GENERATION'
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    with Budget(cfg['resources']) as budget:
        c=open_workbench_db(output);c.execute("SET memory_limit='8GB'");c.execute('SET threads=2')
        for name,path in [('q',gf['queries.parquet']),('master',gf['master.parquet']),('pool',gf['candidates.parquet']),
                          ('parents',gf['parents.parquet']),('witnesses',gf['witnesses.parquet']),('selected',gf['selected.parquet']),
                          ('sitesf',gf['site_features.parquet']),('t',ef['outcomes.parquet']),('catalog',side),('acceptable',acceptable),
                          ('oldpool',oldf['candidates.parquet']),('oldmaster',mf['master.parquet'])]:
            c.execute(f'CREATE TEMP TABLE {name} AS SELECT * FROM read_parquet(?)',[str(path)])
        assert c.execute('SELECT count(*),count(DISTINCT query_id) FROM q').fetchone()==(1034,1034)
        assert c.execute('SELECT count(*) FROM q WHERE fold NOT IN (2,3,4)').fetchone()[0]==0
        same(c,'SELECT query_id,fold FROM q','SELECT query_id,fold FROM t')
        # Explicitly predicate the ledger before reading permitted human labels.
        c.execute('''CREATE TEMP TABLE ledger AS SELECT l.query_id,l.fold,l.human_ground_truth_siret
          FROM read_parquet(?) l SEMI JOIN q USING(query_id,fold) WHERE l.fold IN (2,3,4)''',[str(ledger)])
        same(c,'SELECT query_id,fold,gt_siret FROM t','SELECT query_id,fold,human_ground_truth_siret FROM ledger')
        truth=c.execute('SELECT query_id,fold,gt_siren FROM t').df();validate_fold_isolation(truth)
        c.execute('CREATE TEMP TABLE signals AS SELECT s.* FROM read_parquet(?) s SEMI JOIN q USING(query_id,fold)',
                  [[str(sources[k]) for k in ('structured_signals','siret_native_signals')]])
        c.execute('CREATE TEMP TABLE presence AS SELECT p.* FROM read_parquet(?) p SEMI JOIN q USING(query_id,fold)',[str(sources['presence'])])
        c.execute('CREATE TEMP TABLE geoscores AS SELECT * FROM read_parquet(?)',[[str(geof[f'outer{f}_scores.parquet']) for f in (2,3,4)]])
        same(c,'SELECT query_id,fold,siren,selected_rank FROM selected',
             '''SELECT query_id,fold,siren,row_number() OVER(PARTITION BY query_id ORDER BY score DESC,legacy_parent_rank,siren) r
                FROM geoscores QUALIFY r<=512''')
        same(c,'SELECT query_id,fold,siren FROM parents',
             '''SELECT query_id,fold,siren FROM presence WHERE from_text OR from_address
                UNION SELECT query_id,fold,siren FROM selected
                UNION SELECT query_id,fold,siren FROM signals WHERE channel<>'v17_union'
                UNION SELECT query_id,fold,siren FROM oldpool WHERE base_rank IS NOT NULL''')
        same(c,'SELECT query_id,fold,siren,siret FROM pool',
             f'''SELECT query_id,fold,siren,siret FROM witnesses UNION
                SELECT p.query_id,p.fold,p.siren,s.siret FROM parents p JOIN q USING(query_id,fold)
                JOIN read_parquet({sql_path(sources['sites'])}) s ON s.siren=p.siren AND
                ((q.crm_insee<>'' AND s.insee=q.crm_insee) OR (q.crm_postcode<>'' AND s.postcode=q.crm_postcode))''')
        same(c,'SELECT * FROM witnesses',f'SELECT * FROM read_parquet({sql_path(oldf["witnesses.parquet"])})')
        assert c.execute('SELECT count(*) FROM master ANTI JOIN pool USING(query_id,fold,siren,siret)').fetchone()[0]==0
        assert not c.execute('''SELECT query_id FROM master GROUP BY query_id HAVING count(*)>2000
          OR count(*)<>count(DISTINCT siret) OR min(candidate_rank)<>1 OR max(candidate_rank)<>count(*)''').fetchall()
        # Outer model artifacts/splits and exact replay on generated natural features.
        for fold in (2,3,4):
            train=[f for f in (2,3,4) if f!=fold]
            for files in (mf,geof):
                split=json.loads(files[f'outer{fold}_split.json'].read_text())
                assert split['train_folds']==train and split['target_folds']==[fold]
                assert set(split['trained_query_ids'])<=set(truth.loc[truth.fold.isin(train),'query_id'])
                assert set(split['target_query_ids'])==set(truth.loc[truth.fold.eq(fold),'query_id'])
            frame=pq.read_table(gf['parent_features.parquet'],filters=[('fold','=',fold)]).to_pandas().sort_values(['query_id','siren'])
            pred=pq.read_table(gf[f'outer{fold}_scores.parquet']).to_pandas().sort_values(['query_id','siren'])
            assert list(zip(frame.query_id,frame.siren))==list(zip(pred.query_id,pred.siren))
            model=xgb.Booster();model.load_model(mf[f'outer{fold}.json']);model.set_param({'nthread':2})
            actual=model.predict(xgb.DMatrix(frame[list(FEATURES)].to_numpy(dtype=np.float32)))
            assert np.array_equal(actual,pred.parent_score.to_numpy())
            del frame,pred,model,actual;gc.collect()
        c.execute('CREATE TEMP TABLE scores AS SELECT * FROM read_parquet(?)',[[str(gf[f'outer{f}_scores.parquet']) for f in (2,3,4)]])
        same(c,'SELECT query_id,fold,siren,siret,candidate_rank,rescue_score FROM master',
             '''WITH ranks AS (SELECT *,dense_rank() OVER(PARTITION BY query_id ORDER BY parent_score DESC) pr FROM scores),
                scored AS (SELECT s.*,p.parent_score,s.direct_rrf+CASE WHEN s.local_site THEN
                  1.0/(60+p.pr+s.joint_site_rank) ELSE 0 END score FROM sitesf s JOIN ranks p USING(query_id,fold,siren))
                SELECT query_id,fold,siren,siret,row_number() OVER(PARTITION BY query_id
                  ORDER BY score DESC,direct_rrf DESC,parent_score DESC,siret) r,score FROM scored QUALIFY r<=2000''')
        # Every preflight artifact agrees with the same queries in the full run.
        c.execute('CREATE TEMP TABLE preq AS SELECT query_id FROM read_parquet(?)',[str(pf['queries.parquet'])])
        for key in pf:
            if key.endswith('.parquet'):
                same(c,f'SELECT p.* FROM read_parquet({sql_path(gf[key])}) p SEMI JOIN preq USING(query_id)',
                     f'SELECT * FROM read_parquet({sql_path(pf[key])})')
        verify_metrics(c,em)
        totals=c.execute('''SELECT count(*) pairs,count(*) FILTER(WHERE o.siret IS NULL) new_pairs
          FROM pool p LEFT JOIN oldpool o USING(query_id,siret)''').fetchone()
        dropped=c.execute('SELECT count(*) FROM oldpool ANTI JOIN pool USING(query_id,siret)').fetchone()[0]
        movement=c.execute('''SELECT count(*) FILTER(WHERE o.siret IS NULL) admitted,count(*) FILTER(WHERE p.siret IS NULL) displaced
          FROM master p FULL JOIN oldmaster o USING(query_id,siret)''').fetchone()
        c.close()
    result=dict(verdict='PASS_V29_OUTPUT_RECONCILIATION',config=record(config),generation=record(Path(generation)/'manifest.json'),
                evaluation=record(Path(evaluation)/'manifest.json'),preflight=record(Path(preflight)/'manifest.json'),
                implementation=record(Path(__file__)),counts=dict(candidate_pairs=totals[0],new_candidate_pairs=totals[1],
                removed_candidate_pairs=dropped,admitted_master_pairs=movement[0],displaced_master_pairs=movement[1]),resources=budget.report())
    write_json(output/'manifest.json',result);print(json.dumps(result),flush=True)


def verify_metrics(c,em):
    """Separate per-query set membership calculation, not the evaluation helper."""
    c.execute('''CREATE TEMP TABLE independent AS SELECT t.query_id,
      EXISTS(SELECT 1 FROM master m WHERE m.query_id=t.query_id AND m.siret=t.gt_siret) exact_hit,
      (EXISTS(SELECT 1 FROM master m WHERE m.query_id=t.query_id AND m.siret=t.gt_siret) OR
       EXISTS(SELECT 1 FROM master m JOIN acceptable a USING(query_id,siret) WHERE m.query_id=t.query_id)) operational_hit,
      EXISTS(SELECT 1 FROM oldmaster m WHERE m.query_id=t.query_id AND m.siret=t.gt_siret) previous_exact_hit,
      EXISTS(SELECT 1 FROM pool p WHERE p.query_id=t.query_id AND p.siret=t.gt_siret) site_in_reservoir,
      EXISTS(SELECT 1 FROM oldpool p WHERE p.query_id=t.query_id AND p.siret=t.gt_siret) old_reservoir_hit,
      EXISTS(SELECT 1 FROM parents p WHERE p.query_id=t.query_id AND p.siren=t.gt_siren) parent_hit,
      cat.official_catalog_present FROM t JOIN catalog cat USING(query_id)''')
    same(c,'SELECT query_id,exact_hit,operational_hit,previous_exact_hit,site_in_reservoir,old_reservoir_hit FROM t',
         'SELECT query_id,exact_hit,operational_hit,previous_exact_hit,site_in_reservoir,old_reservoir_hit FROM independent')
    same(c,'SELECT query_id,loss_stage FROM t', '''SELECT query_id,CASE WHEN exact_hit THEN 'HIT'
      WHEN NOT official_catalog_present THEN 'OFFICIAL_CATALOG_ABSENT' WHEN NOT parent_hit THEN 'PARENT_NOT_RETRIEVED'
      WHEN NOT site_in_reservoir THEN 'SITE_NOT_EXPANDED' ELSE 'EVICTED_RESERVOIR_TO_MASTER2000' END FROM independent''')
    counts=c.execute('''SELECT count(*),sum(exact_hit::INT),sum(operational_hit::INT),
      sum((exact_hit AND NOT previous_exact_hit)::INT),sum((NOT exact_hit AND previous_exact_hit)::INT) FROM independent''').fetchone()
    assert counts==tuple(em['primary'][k] for k in ('queries','exact_hits','operational_hits','gains_vs_v23','losses_vs_v23'))
    assert em['historical_all_human']==em['primary']
    for dim,segments in em['segments'].items():
        for label,summary in segments.items():
            rows=c.execute(f'''SELECT count(*),sum(i.exact_hit::INT),sum(i.operational_hit::INT),
              sum((i.exact_hit AND NOT i.previous_exact_hit)::INT),sum((NOT i.exact_hit AND i.previous_exact_hit)::INT)
              FROM independent i JOIN t USING(query_id) WHERE coalesce(cast(t.{dim} AS VARCHAR),'nan')=?''',[label]).fetchone()
            assert rows==tuple(summary[k] for k in ('queries','exact_hits','operational_hits','gains_vs_v23','losses_vs_v23'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path('config/retrieval_v29_consolidated.json'))
    for name in ('generation','evaluation','preflight','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();audit(a.config,a.generation,a.evaluation,a.preflight,a.output)
