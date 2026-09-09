#!/usr/bin/env python3
"""Separate generate/evaluate commands for one frozen global-budget experiment."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.xgb_matcher.retrieval_v20_workbench import ROOT, Budget, record, verify, write_json, sql_path
from src.xgb_matcher.retrieval_v21_membership import manifest_files, open_workbench_db


def config_at(path):
    c=json.loads(Path(path).read_text())
    if (c['schema']!='sireto-v22-global-budget-1' or c['recipe']!='V21_SCORE_GLOBAL_TOP2000_NO_RESERVED_PREFIX'
        or (c['master_cap'],c['reserved_prefix'])!=(2000,0)
        or any(c[k] for k in ('national_projection_allowed','fold1_allowed','external_generalization_claim_allowed'))):
        raise ValueError('Frozen v22 contract drift')
    return c


def code_records():
    paths=[Path(__file__),ROOT/'src/xgb_matcher/retrieval_v21_membership.py',
           ROOT/'src/xgb_matcher/retrieval_v20_workbench.py',ROOT/'src/xgb_matcher/retrieval_parent_discovery_v18.py']
    return {str(p.relative_to(ROOT)):record(p) for p in paths}


def select_master(c):
    if c.execute('SELECT count(*) FROM candidates WHERE fold NOT IN (2,3,4) OR rescue_score IS NULL OR NOT isfinite(rescue_score)').fetchone()[0]:
        raise ValueError('Invalid candidate scores/folds')
    if c.execute('SELECT count(*) FROM (SELECT query_id,siret,count(*) n FROM candidates GROUP BY ALL HAVING n>1)').fetchone()[0]:
        raise ValueError('Duplicate SIRET')
    c.execute('''CREATE TEMP TABLE master AS
      SELECT * EXCLUDE(global_rank),global_rank candidate_rank,'GLOBAL_SCORE_NO_RESERVED_PREFIX' selection_reason
      FROM (SELECT *,row_number() OVER(PARTITION BY query_id ORDER BY
        rescue_score DESC,direct_rrf DESC,parent_rrf DESC,siret) global_rank FROM candidates)
      WHERE global_rank<=2000''')
    if c.execute('SELECT count(*) FROM (SELECT query_id,count(*) n,count(DISTINCT siret) nd,max(candidate_rank) mx FROM master GROUP BY query_id HAVING n>2000 OR n<>nd OR n<>mx)').fetchone()[0]:
        raise ValueError('Master cap or uniqueness')


def generate(config_path, output, preflight=None):
    config=config_at(config_path); gm,gf=manifest_files(config['source_generation'])
    if gm['schema']!='sireto-v21-generation-1' or gm['preflight'] or gm['truth_columns_opened'] or gm['counts']['q']!=1034:
        raise ValueError('Source generation authority')
    if preflight is not None and not 1<=preflight<=16:raise ValueError('Preflight 1..16')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    with Budget(config['resources']) as budget:
        c=open_workbench_db(output)
        c.execute('CREATE TEMP TABLE q AS SELECT * FROM read_parquet(?) ORDER BY query_id LIMIT ?', [str(gf['queries.parquet']),preflight or 1034])
        if c.execute('SELECT count(*),count(DISTINCT query_id) FROM q').fetchone()!=(preflight or 1034,preflight or 1034):raise ValueError('Query drift')
        c.execute('''CREATE TEMP TABLE candidates AS SELECT p.query_id,p.fold,p.siret,p.siren,
          p.base_rank,p.rescue_score,p.direct_rrf,p.parent_rrf FROM read_parquet(?) p SEMI JOIN q USING(query_id,fold)''', [str(gf['candidates.parquet'])])
        select_master(c)
        for name,table,order in [('queries','q','query_id'),('master','master','query_id,candidate_rank')]:
            c.execute(f'COPY (SELECT * FROM {table} ORDER BY {order}) TO {sql_path(output/(name+".parquet"))} (FORMAT PARQUET,COMPRESSION ZSTD)')
        counts={t:c.execute(f'SELECT count(*) FROM {t}').fetchone()[0] for t in ('q','candidates','master')}
        c.close()
    result={'schema':'sireto-v22-generation-1','config':record(config_path),'source_generation':config['source_generation'],
      'candidates':gm['files']['candidates.parquet'],'counts':counts,'preflight':preflight is not None,
      'truth_columns_opened':False,'learned_components':False,'fold0_opened':False,'fold1_opened':False,
      'resources':budget.report(),'implementation':code_records(),
      'files':{p.name:record(p) for p in output.iterdir() if p.is_file()}}
    write_json(output/'manifest.json',result);print(json.dumps(result),flush=True)


def outcomes(c):
    if c.execute('SELECT count(*) FROM t WHERE fold NOT IN (2,3,4)').fetchone()[0]:raise ValueError('Forbidden truth fold')
    if c.execute('SELECT count(*),count(DISTINCT query_id) FROM t').fetchone()!=c.execute('SELECT count(*),count(DISTINCT query_id) FROM q').fetchone():raise ValueError('Population mismatch')
    if c.execute('SELECT count(*) FROM t ANTI JOIN q USING(query_id,fold)').fetchone()[0]:raise ValueError('Query mismatch')
    c.execute('''CREATE TEMP TABLE op AS
      SELECT t.query_id,bool_or(m.siret IS NOT NULL) operational_hit,
        coalesce(bool_or(m.siret IS NOT NULL AND a.administrative_state='A' AND NOT a.exact_truth AND cat.exact_state='F'),false) active_successor_present,
        count(DISTINCT CASE WHEN m.siret IS NOT NULL AND a.administrative_state='A' THEN a.siret END)>1 multiple_active_equivalents_present
      FROM t JOIN official_loss cat USING(query_id) LEFT JOIN acceptable a USING(query_id)
      LEFT JOIN master m ON a.query_id=m.query_id AND a.siret=m.siret GROUP BY t.query_id''')
    c.execute('''CREATE TEMP TABLE outcomes AS
      SELECT t.*,m.siret IS NOT NULL exact_hit,m.candidate_rank,
        (m.siret IS NOT NULL OR coalesce(op.operational_hit,false)) operational_hit,
        op.active_successor_present,op.multiple_active_equivalents_present,
        CASE WHEN m.siret IS NOT NULL THEN 'HIT'
          WHEN NOT cat.official_catalog_present THEN 'OFFICIAL_CATALOG_ABSENT'
          WHEN NOT cat.parent_in_evidence THEN 'PARENT_NOT_RETRIEVED'
          WHEN NOT t.site_in_reservoir THEN 'SITE_NOT_EXPANDED'
          ELSE 'EVICTED_RESERVOIR_TO_MASTER2000' END loss_stage
      FROM t JOIN official_loss cat USING(query_id)
      LEFT JOIN master m ON t.query_id=m.query_id AND t.gt_siret=m.siret JOIN op ON t.query_id=op.query_id''')
    df=c.execute('SELECT * FROM outcomes ORDER BY fold,query_id').df()
    if df.query_id.duplicated().any() or len(df)!=c.execute('SELECT count(*) FROM t').fetchone()[0]:raise ValueError('Incomplete outcomes')
    return df


def summary(df):
    return {'queries':len(df),'exact_hits':int(df.exact_hit.sum()),'exact_recall':float(df.exact_hit.mean()) if len(df) else None,
      'operational_hits':int(df.operational_hit.sum()),'operational_recall':float(df.operational_hit.mean()) if len(df) else None,
      'operational_only':int((df.operational_hit & ~df.exact_hit).sum()),
      'active_successor_present':int(df.active_successor_present.sum()),
      'multiple_active_equivalents_present':int(df.multiple_active_equivalents_present.sum()),
      'gains_vs_v21':int((df.exact_hit & ~df.previous_exact_hit).sum()),
      'losses_vs_v21':int((~df.exact_hit & df.previous_exact_hit).sum())}


def evaluate(config_path, generation, output):
    config=config_at(config_path);gm,gf=manifest_files(record(Path(generation)/'manifest.json'))
    if (gm['schema']!='sireto-v22-generation-1' or gm['preflight'] or gm['truth_columns_opened']
        or gm['config']['sha256']!=record(config_path)['sha256'] or gm['counts']['q']!=1034
        or gm['source_generation']!=config['source_generation'] or gm['implementation']!=code_records()):raise ValueError('Generation authority')
    bm,bf=manifest_files(config['baseline_evaluation'])
    if bm['schema']!='sireto-v21-evaluation-1' or bm['generation_manifest']['sha256']!=config['source_generation']['sha256']:raise ValueError('Baseline authority')
    side=verify(bm['official_loss_sidecar']);acceptable=verify(bm['acceptable_operational_sidecar'])
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    with Budget(config['resources']) as budget:
        c=open_workbench_db(output)
        for name,path in [('q',gf['queries.parquet']),('master',gf['master.parquet']),('official_loss',side),('acceptable',acceptable)]:
            c.execute(f'CREATE TEMP TABLE {name} AS SELECT * FROM read_parquet(?)',[str(path)])
        c.execute('''CREATE TEMP TABLE t AS SELECT query_id,fold,gt_siret,gt_siren,cohort_kind,
          qualification_v2,qualification_v3,label_kind,data_origin,ground_truth_state,
          site_in_reservoir,exact_hit previous_exact_hit FROM read_parquet(?)''',[str(bf['outcomes.parquet'])])
        df=outcomes(c)
        if len(df)!=1034:raise ValueError('Incomplete panel')
        cap=c.execute('SELECT max(n) FROM (SELECT query_id,count(*) n FROM master GROUP BY query_id)').fetchone()[0]
        if cap>2000:raise ValueError('Master cap')
        c.execute(f'COPY outcomes TO {sql_path(output/"outcomes.parquet")} (FORMAT PARQUET,COMPRESSION ZSTD)')
        segments={d:{str(k):summary(v) for k,v in df.fillna({d:'UNKNOWN'}).groupby(d,dropna=False)} for d in ('fold','cohort_kind','data_origin','qualification_v2','qualification_v3','label_kind','ground_truth_state')}
        c.close()
    result={'schema':'sireto-v22-evaluation-1','decision':'PIVOT_GENERATION',
      'config':record(config_path),'generation_manifest':record(Path(generation)/'manifest.json'),
      'baseline_evaluation':config['baseline_evaluation'],'official_loss_sidecar':record(side),'acceptable_operational_sidecar':record(acceptable),
      'primary':summary(df),'historical_all_human':summary(df),'segments':segments,'max_master_sirets':int(cap),
      'loss_counts':{str(k):int(v) for k,v in df.loss_stage.value_counts().items()},
      'national_projection':None,'national_projection_prohibited_reason':'BASELINE_PREFIX_NO_LONGER_PRESERVED; BIASED_1034_PANEL',
      'unseen_external_validation':False,'reused_development_labels':True,'prerank500_measured':False,'recall100_measured':False,
      'ready_for_fold1':False,'fold0_opened':False,'fold1_opened':False,'resources':budget.report(),
      'implementation':code_records(),'files':{p.name:record(p) for p in output.iterdir() if p.is_file()}}
    write_json(output/'manifest.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['generate','evaluate']);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--generation',type=Path);p.add_argument('--preflight',type=int)
    a=p.parse_args()
    if a.action=='generate':
        if a.generation:p.error('generation is only an evaluation argument')
        generate(a.config,a.output,a.preflight)
    else:
        if not a.generation or a.preflight:p.error('evaluation requires generation and disallows preflight')
        evaluate(a.config,a.generation,a.output)
