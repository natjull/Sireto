"""Finalize pinned completed v28 fits; no fit call and no candidate injection."""
import argparse
import json
from pathlib import Path
import shutil

import pyarrow.parquet as pq

from src.xgb_matcher.retrieval_v20_workbench import Budget,record,verify,write_json,sql_path
from src.xgb_matcher.retrieval_v21_membership import manifest_files,open_workbench_db
from scripts.retrieval_v28_geo_selection import config_at,implementation,versions,summary,save


def missing_op_allowed(official_present, exact_in_pool):
    return not official_present and not exact_in_pool


def main(recovery,output):
    seal=json.loads(Path(recovery).read_text())
    if seal['schema']!='sireto-v28-completed-fits-recovery-1' or seal['models_retrained']:raise ValueError('Recovery drift')
    config=verify(seal['config']);cfg=config_at(config)
    fm,ff=manifest_files(seal['features_manifest'])
    if fm['implementation']!=implementation() or fm['versions']!=versions() or fm['preflight']:raise ValueError('Frozen features drift')
    trained={k:verify(v) for k,v in seal['files'].items()}
    expected={f'outer{fold}{suffix}' for fold in (2,3,4) for suffix in ('.json','_scores.parquet','_split.json')}
    assert set(trained)==expected
    tm,tf=manifest_files(cfg['truth_audit']);side=verify(cfg['official_loss_sidecar']);op=verify(cfg['operational_sidecar'])
    model_config=json.loads(verify(cfg['model_config']).read_text())['model']
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    with Budget(cfg['resources']) as budget:
        # Byte-identical copies of generated artifacts, never edits or refits.
        for name,path in trained.items():shutil.copyfile(path,output/name)
        c=open_workbench_db(output)
        for table,path in [('t',tf['result.parquet']),('pool',ff['pool.parquet']),('features',ff['parent_features.parquet']),
                           ('old',ff['baseline_parents.parquet']),('op',op),('official',side)]:
            c.execute(f'CREATE TEMP TABLE {table} AS SELECT * FROM read_parquet(?)',[str(path)])
        assert c.execute('SELECT count(*),count(DISTINCT query_id) FROM t WHERE fold IN (2,3,4)').fetchone()==(1034,1034)
        assert c.execute('SELECT count(*) FROM t ANTI JOIN official USING(query_id)').fetchone()[0]==0
        missing=c.execute('''SELECT t.query_id,o.official_catalog_present,
          EXISTS(SELECT 1 FROM pool p WHERE p.query_id=t.query_id AND p.siret=t.gt_siret) exact_in_pool
          FROM t JOIN official o USING(query_id) WHERE NOT EXISTS
          (SELECT 1 FROM op WHERE op.query_id=t.query_id AND op.siret=t.gt_siret)''').fetchall()
        assert len(missing)==seal['expected_missing_operational_exact']==1
        assert all(missing_op_allowed(present,in_pool) for _,present,in_pool in missing)
        c.execute('CREATE TEMP TABLE scores AS SELECT * FROM read_parquet(?)',[[str(trained[f'outer{f}_scores.parquet']) for f in (2,3,4)]])
        assert c.execute('SELECT count(*) FROM scores WHERE fold NOT IN (2,3,4) OR NOT isfinite(score)').fetchone()[0]==0
        assert c.execute('SELECT count(*) FROM (SELECT query_id,siren,count(*) n FROM scores GROUP BY ALL HAVING n<>1)').fetchone()[0]==0
        assert c.execute('''SELECT count(*) FROM ((SELECT query_id,fold,siren FROM scores EXCEPT SELECT query_id,fold,siren FROM features)
          UNION ALL (SELECT query_id,fold,siren FROM features EXCEPT SELECT query_id,fold,siren FROM scores))''').fetchone()[0]==0
        records=[]
        for fold in (2,3,4):
            rec=json.loads(trained[f'outer{fold}_split.json'].read_text())
            assert rec['train_folds']==[f for f in (2,3,4) if f!=fold] and rec['target_folds']==[fold]
            assert not set(rec['trained_query_ids']) & set(rec['target_query_ids'])
            records.append({k:v for k,v in rec.items() if not k.endswith('query_ids')})
        c.execute('''CREATE TEMP TABLE selected AS SELECT * FROM (SELECT *,row_number() OVER(PARTITION BY query_id
          ORDER BY score DESC,legacy_parent_rank,siren) selected_rank FROM scores) WHERE selected_rank<=512''')
        assert c.execute('SELECT count(*) FROM (SELECT query_id,count(*) n,count(DISTINCT siren) nd FROM selected GROUP BY query_id HAVING n>512 OR n<>nd)').fetchone()[0]==0
        c.execute('CREATE TEMP TABLE old_sites AS SELECT p.* FROM pool p JOIN old o USING(query_id,siren)')
        c.execute('CREATE TEMP TABLE new_sites AS SELECT p.* FROM pool p JOIN selected s USING(query_id,siren)')
        c.execute('''CREATE TEMP TABLE outcomes AS SELECT t.*,
          EXISTS(SELECT 1 FROM old WHERE old.query_id=t.query_id AND old.siren=t.gt_siren) baseline_parent_hit,
          EXISTS(SELECT 1 FROM selected s WHERE s.query_id=t.query_id AND s.siren=t.gt_siren) oof_parent_hit,
          EXISTS(SELECT 1 FROM pool p WHERE p.query_id=t.query_id AND p.siren=t.gt_siren) pool_parent_hit,
          EXISTS(SELECT 1 FROM old_sites p WHERE p.query_id=t.query_id AND p.siret=t.gt_siret) baseline_expansion_exact,
          EXISTS(SELECT 1 FROM new_sites p WHERE p.query_id=t.query_id AND p.siret=t.gt_siret) oof_expansion_exact,
          EXISTS(SELECT 1 FROM old_sites p JOIN op USING(query_id,siret) WHERE p.query_id=t.query_id) baseline_expansion_operational,
          EXISTS(SELECT 1 FROM new_sites p JOIN op USING(query_id,siret) WHERE p.query_id=t.query_id) oof_expansion_operational FROM t''')
        df=c.execute('SELECT * FROM outcomes ORDER BY query_id').df();assert len(df)==1034
        assert c.execute('SELECT count(*) FROM outcomes WHERE query_id=? AND (baseline_expansion_exact OR oof_expansion_exact OR baseline_expansion_operational OR oof_expansion_operational)',[missing[0][0]]).fetchone()[0]==0
        for table in ('selected','outcomes'):
            c.execute(f'COPY {table} TO {sql_path(output/(table+".parquet"))} (FORMAT PARQUET,COMPRESSION ZSTD)')
        seg={dim:{str(k):summary(v) for k,v in df.groupby(dim,dropna=False)} for dim in
             ('fold','cohort_kind','qualification_v2','qualification_v3','data_origin','label_kind','ground_truth_state','loss_stage')}
        primary=summary(df);controls=df[df.cohort_kind.eq('BASE_TOP1750_CONTROL')]
        eligible=primary['parent_gains']>primary['parent_losses'] and not ((~controls.oof_parent_hit)&controls.baseline_parent_hit).any()
        c.close()
    save(output,config,dict(schema='sireto-v28-oof-evaluation-1',decision='PIVOT_GENERATION',pilot_status='ELIGIBLE_FOR_SEPARATE_END_TO_END_TEST' if eligible else 'REJECT_NO_RETUNING',
        features_manifest=seal['features_manifest'],truth_authority=cfg['truth_audit'],official_loss_sidecar=record(side),operational_sidecar=record(op),
        completed_fits_recovery=record(recovery),finalizer_implementation=record(Path(__file__)),models_retrained=False,known_catalog_absence_retained=1,
        records=records,model_parameters=model_config,primary=primary,segments=seg,focus=summary(df[df.focus_missing_near500]),
        recall100_measured=False,master2000_measured=False,rich_hydration=False,positive_injection=False,target_labels_used_for_selection=False,
        raw_coordinates_persisted=False,training_resource_report_available=False,finalization_resources=budget.report()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--recovery',type=Path,default=Path('config/retrieval_v28_completed_fits.json'))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.recovery,a.output)
