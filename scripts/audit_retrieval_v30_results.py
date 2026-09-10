"""Reconcile joint retrieval, natural expansion, frozen inference and exact/op metrics."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import gc
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.xgb_matcher.retrieval_v20_workbench import Budget,record,verify,write_json,sql_path
from src.xgb_matcher.retrieval_v21_membership import manifest_files,open_workbench_db
from src.xgb_matcher.retrieval_v23_joint_parent import FEATURES
from src.xgb_matcher.retrieval_parent_fusion_v19 import validate_fold_isolation
from src.xgb_matcher.hierarchical_retrieval import TantivyBackend
from src.xgb_matcher.retrieval_v25_conjunctive import index_inventory
from scripts.retrieval_v30_joint_search import search,CHANNEL,SIGNALS
from scripts.run_retrieval_parent_discovery_v18 import _load_retrieval_queries
from scripts.audit_retrieval_v29_results import same,verify_metrics


def audit(config,generation,evaluation,preflight,output):
    cfg=json.loads(Path(config).read_text());gm,gf=manifest_files(record(Path(generation)/'manifest.json'))
    em,ef=manifest_files(record(Path(evaluation)/'manifest.json'));pm,pf=manifest_files(record(Path(preflight)/'manifest.json'))
    dm,df=manifest_files(gm['discovery_manifest']);pdm,pdf=manifest_files(pm['discovery_manifest'])
    bm,bf=manifest_files(cfg['baseline']);mm,mf=manifest_files(cfg['models'])
    v23=json.loads(verify(cfg['v23_config']).read_text());oldm,oldf=manifest_files(v23['source_generation'])
    v18=json.loads(verify(v23['v18_config']).read_text())['generation_authorities']
    v19=json.loads(verify(cfg['v19_config']).read_text());ledger=verify(v19['authorities']['label_ledger'])
    for m in (gm,em,pm,dm,pdm):
        assert m['config']==record(config) and m['implementation']==gm['implementation'] and m['versions']==gm['versions']
        assert not m['fold0_opened'] and not m['fold1_opened'] and not m['production_activation']
    assert not gm['preflight'] and not dm['preflight'] and pm['preflight'] and pdm['preflight']
    assert em['generation_manifest']==record(Path(generation)/'manifest.json')
    assert em['baseline']==cfg['baseline'] and bm['generation_manifest']['sha256']==cfg['models']['sha256']
    sources={k:verify(v) for k,v in gm['sources'].items()}
    for r in dm['sources']:verify(r)
    index=Path(v18['hierarchical_index']['path']);assert index_inventory(index/'tantivy')==dm['index_inventory']
    verify(dm['index_meta']);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    with Budget(cfg['resources']) as budget:
        # Replay queries from their original truth-free SceneIR, before loading labels.
        qtable=pq.read_table(df['queries.parquet']);stored=qtable.to_pylist()
        queries=_load_retrieval_queries(tuple((r['query_id'],r['fold']) for r in stored),Path(v18['scene_ir']['path']))
        assert [dict(query_id=i,fold=f,**asdict(q)) for i,f,q in queries]==stored
        backend=TantivyBackend(index)
        for _,_,q in queries:
            if q.insee:backend._has_insee(q.insee)
        def one(item):
            i,f,q=item;return [dict(query_id=i,fold=f,**row) for row in search(backend,q)]
        replay=[]
        with ThreadPoolExecutor(max_workers=4) as workers:
            for rows in workers.map(one,queries):replay.extend(rows)
        assert replay==pq.read_table(df['joint_signals.parquet']).to_pylist()
        assert index_inventory(index/'tantivy')==dm['index_inventory'];verify(dm['index_meta'])
        del replay,backend,queries;gc.collect()
        print('PASS_JOINT_CHANNEL_EXACT_REPLAY',flush=True)
        c=open_workbench_db(output);c.execute("SET memory_limit='8GB'");c.execute('SET threads=2')
        for name,path in [('q',gf['queries.parquet']),('master',gf['master.parquet']),('pool',gf['candidates.parquet']),
                          ('parents',gf['parents.parquet']),('witnesses',gf['witnesses.parquet']),('joint',df['joint_signals.parquet']),
                          ('sitesf',gf['site_features.parquet']),('t',ef['outcomes.parquet']),('catalog',verify(em['official_loss_sidecar'])),
                          ('acceptable',verify(em['operational_sidecar'])),('oldpool',oldf['candidates.parquet']),('oldmaster',mf['master.parquet'])]:
            c.execute(f'CREATE TEMP TABLE {name} AS SELECT * FROM read_parquet(?)',[str(path)])
        assert c.execute('SELECT count(*),count(DISTINCT query_id) FROM q').fetchone()==(1034,1034)
        assert c.execute('SELECT count(*) FROM q WHERE fold NOT IN (2,3,4)').fetchone()[0]==0
        same(c,'SELECT query_id,fold FROM q','SELECT query_id,fold FROM t')
        same(c,'SELECT query_id,fold,gt_siret,gt_siren FROM t',f'''SELECT l.query_id,l.fold,l.human_ground_truth_siret,l.human_ground_truth_siren
          FROM read_parquet({sql_path(ledger)}) l SEMI JOIN q USING(query_id,fold) WHERE l.fold IN (2,3,4)''')
        truth=c.execute('SELECT query_id,fold,gt_siren FROM t').df();validate_fold_isolation(truth)
        c.execute('CREATE TEMP TABLE signals AS SELECT * FROM read_parquet(?)',[[str(sources[k]) for k in ('structured_signals','siret_native_signals')]])
        same(c,'SELECT query_id,fold,siren FROM parents','''SELECT query_id,fold,siren FROM signals UNION
          SELECT query_id,fold,siren FROM oldpool WHERE base_rank IS NOT NULL UNION SELECT query_id,fold,siren FROM joint''')
        same(c,'SELECT * FROM witnesses',f'''SELECT * FROM read_parquet({sql_path(oldf['witnesses.parquet'])}) UNION
          SELECT query_id,fold,siren,siret,'{CHANNEL}' channel,evidence_rank channel_rank FROM joint''')
        same(c,'SELECT query_id,fold,siren,siret FROM pool',f'''SELECT query_id,fold,siren,siret FROM witnesses UNION
          SELECT p.query_id,p.fold,p.siren,s.siret FROM parents p JOIN q USING(query_id,fold)
          JOIN read_parquet({sql_path(sources['sites'])}) s ON p.siren=s.siren AND
          ((q.crm_insee<>'' AND q.crm_insee=s.insee) OR (q.crm_postcode<>'' AND q.crm_postcode=s.postcode))''')
        assert not c.execute('SELECT 1 FROM oldpool ANTI JOIN pool USING(query_id,fold,siren,siret) LIMIT 1').fetchall()
        assert not c.execute('SELECT 1 FROM master ANTI JOIN pool USING(query_id,fold,siren,siret) LIMIT 1').fetchall()
        for table,cap,rank in [('joint',512,'candidate_rank'),('master',2000,'candidate_rank')]:
            assert not c.execute(f'''SELECT query_id FROM {table} GROUP BY query_id HAVING count(*)>{cap} OR
              count(*)<>count(DISTINCT siret) OR min({rank})<>1 OR max({rank})<>count(*)''').fetchall()
        for fold in (2,3,4):
            split=json.loads(mf[f'outer{fold}_split.json'].read_text());train=[f for f in (2,3,4) if f!=fold]
            assert split['train_folds']==train and split['target_folds']==[fold]
            assert set(split['trained_query_ids'])<=set(truth.loc[truth.fold.isin(train),'query_id'])
            assert set(split['target_query_ids'])==set(truth.loc[truth.fold.eq(fold),'query_id'])
            frame=pq.read_table(gf['parent_features.parquet'],filters=[('fold','=',fold)]).to_pandas().sort_values(['query_id','siren'])
            pred=pq.read_table(gf[f'outer{fold}_scores.parquet']).to_pandas().sort_values(['query_id','siren'])
            assert list(zip(frame.query_id,frame.siren))==list(zip(pred.query_id,pred.siren))
            model=xgb.Booster();model.load_model(mf[f'outer{fold}.json']);model.set_param({'nthread':2})
            values=model.predict(xgb.DMatrix(frame[list(FEATURES)].to_numpy(dtype=np.float32)))
            assert np.array_equal(values,pred.parent_score.to_numpy())
            del frame,pred,model,values;gc.collect()
        c.execute('CREATE TEMP TABLE scores AS SELECT * FROM read_parquet(?)',[[str(gf[f'outer{f}_scores.parquet']) for f in (2,3,4)]])
        same(c,'SELECT query_id,fold,siren,siret,candidate_rank,rescue_score FROM master',
             '''WITH ranks AS (SELECT *,dense_rank() OVER(PARTITION BY query_id ORDER BY parent_score DESC) pr FROM scores),
                scored AS (SELECT s.*,p.parent_score,s.direct_rrf+CASE WHEN s.local_site THEN
                  1.0/(60+p.pr+s.joint_site_rank) ELSE 0 END score FROM sitesf s JOIN ranks p USING(query_id,fold,siren))
                SELECT query_id,fold,siren,siret,row_number() OVER(PARTITION BY query_id
                  ORDER BY score DESC,direct_rrf DESC,parent_score DESC,siret) r,score FROM scored QUALIFY r<=2000''')
        c.execute('CREATE TEMP TABLE preq AS SELECT query_id FROM read_parquet(?)',[str(pf['queries.parquet'])])
        for small,full in [(pf,gf),(pdf,df)]:
            for key in small:
                if key.endswith('.parquet') and key!='timings.parquet':
                    same(c,f'SELECT p.* FROM read_parquet({sql_path(full[key])}) p SEMI JOIN preq USING(query_id)',f'SELECT * FROM read_parquet({sql_path(small[key])})')
        verify_metrics(c,em)
        same(c,'SELECT query_id,channel_exact,channel_parent FROM t', '''SELECT t.query_id,
          EXISTS(SELECT 1 FROM joint p WHERE p.query_id=t.query_id AND p.siret=t.gt_siret),
          EXISTS(SELECT 1 FROM joint p WHERE p.query_id=t.query_id AND p.siren=t.gt_siren) FROM t''')
        details=dict(new_candidate_pairs=c.execute('SELECT count(*) FROM pool ANTI JOIN oldpool USING(query_id,siret)').fetchone()[0],
                     total_candidate_pairs=c.execute('SELECT count(*) FROM pool').fetchone()[0],
                     direct_joint_pairs=c.execute('SELECT count(*) FROM joint').fetchone()[0])
        c.close()
    result=dict(verdict='PASS_V30_OUTPUT_RECONCILIATION',config=record(config),generation=record(Path(generation)/'manifest.json'),
                evaluation=record(Path(evaluation)/'manifest.json'),preflight=record(Path(preflight)/'manifest.json'),
                implementation=record(Path(__file__)),metric_verifier=record(Path('scripts/audit_retrieval_v29_results.py')),
                counts=details,resources=budget.report())
    write_json(output/'manifest.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path('config/retrieval_v30_joint_search.json'))
    for key in ('generation','evaluation','preflight','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();audit(a.config,a.generation,a.evaluation,a.preflight,a.output)
