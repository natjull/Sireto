#!/usr/bin/env python3
"""Truth-side Master recall and official loss/operational sidecars for v20."""
from pathlib import Path
import argparse
import json
import sys

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.xgb_matcher.retrieval_v20_workbench import Budget, load_config, verify, record, write_json, sql_path, implementation


def summarize(df):
    return {'queries':len(df),'exact_hits':int(df.exact_hit.sum()),'exact_recall':float(df.exact_hit.mean()) if len(df) else None,
      'operational_hits':int(df.operational_hit.sum()),'operational_recall':float(df.operational_hit.mean()) if len(df) else None,
      'operational_only':int((df.operational_hit & ~df.exact_hit).sum()),
      'active_equivalent_when_exact_closed':int(df.active_equivalent_when_exact_closed.sum())}


def run(config_path,generation,output):
    config,sources=load_config(config_path);output=Path(output)
    code = implementation(__file__)
    mp=Path(generation)/'manifest.json';m=json.loads(mp.read_text())
    if m['schema']!='sireto-v20-master-generation-1' or m['config']['sha256']!=record(config_path)['sha256'] or m['truth_columns_opened'] or m['preflight'] or m['query_count']!=1034:
        raise ValueError('Generation authority')
    files={k:verify(v) for k,v in m['files'].items()}
    # Require internal artifacts; the generation is not allowed to swap in an external result.
    if any(p.parent.resolve()!=mp.parent.resolve() for p in files.values()):raise ValueError('External generation artifact')
    a=sources['v19']['authorities'];labels=verify(a['cohort_labels']);ledger=verify(a['label_ledger']);sites=verify(config['sites'])
    signals=[verify(a[k]) for k in ('structured_signals','siret_native_signals')]
    output.mkdir(parents=True,exist_ok=False)
    with Budget(config['resources']) as budget:
        c=duckdb.connect();c.execute("SET memory_limit='4GB'");c.execute('SET threads=4')
        c.execute('''CREATE TEMP TABLE truth AS SELECT l.query_id,l.fold,l.human_ground_truth_siret gt_siret,
          l.human_ground_truth_siren gt_siren,l.qualification_v2,l.qualification_v3,l.label_kind,l.data_origin,
          l.ground_truth_state,l.official_catalog_present previous_catalog_flag,x.cohort_kind
          FROM read_parquet(?) l JOIN read_parquet(?) x USING(query_id,fold)
          WHERE l.human_ground_truth_siret=x.human_ground_truth_siret''',[str(ledger),str(labels)])
        if c.execute('SELECT count(*),count(DISTINCT query_id) FROM truth').fetchone()!=(1034,1034):raise ValueError('Truth population drift')
        if c.execute('SELECT count(*) FROM truth WHERE fold NOT IN (2,3,4)').fetchone()[0]:raise ValueError('Forbidden fold')
        c.execute('CREATE TEMP TABLE master AS SELECT * FROM read_parquet(?)',[str(files['master.parquet'])])
        c.execute('CREATE TEMP TABLE candidates AS SELECT * FROM read_parquet(?)',[str(files['candidates.parquet'])])
        c.execute('CREATE TEMP TABLE q AS SELECT * FROM read_parquet(?)',[str(files['queries.parquet'])])
        c.execute('''CREATE TEMP TABLE official AS SELECT s.* FROM read_parquet(?) s
          WHERE siren IN (SELECT gt_siren FROM truth) OR siret IN (SELECT gt_siret FROM truth)''',[str(sites)])
        c.execute('''CREATE TEMP TABLE official_loss_sidecar AS
          SELECT t.query_id,t.fold,t.gt_siret,t.gt_siren,
            (s.siret IS NOT NULL) official_catalog_present,s.administrative_state exact_state,
            t.previous_catalog_flag,
            (p.query_id IS NOT NULL) parent_in_evidence
          FROM truth t LEFT JOIN official s ON t.gt_siret=s.siret
          LEFT JOIN (SELECT DISTINCT query_id,siren FROM read_parquet(?)) p ON t.query_id=p.query_id AND t.gt_siren=p.siren''',[[str(p) for p in signals]])
        c.execute('''CREATE TEMP TABLE acceptable_operational AS
          SELECT DISTINCT t.query_id,t.fold,s.siret,s.administrative_state,
            (s.siret=t.gt_siret) exact_truth
          FROM truth t JOIN q USING(query_id,fold) JOIN official s ON t.gt_siren=s.siren
          WHERE s.siret=t.gt_siret OR (
            q.crm_number<>'' AND q.crm_number=s.street_number
            AND (q.crm_suffix='' OR q.crm_suffix=upper(coalesce(s.street_number_suffix,'')))
            AND q.crm_street<>'' AND q.crm_street=s.street_normalized
            AND ((q.crm_postcode<>'' AND q.crm_postcode=s.postcode) OR (q.crm_insee<>'' AND q.crm_insee=s.insee)))''')
        c.execute('''CREATE TEMP TABLE outcomes AS
          WITH op AS (SELECT t.query_id,bool_or(m.siret IS NOT NULL) operational_hit,
            bool_or(m.siret IS NOT NULL AND a.administrative_state='A' AND NOT a.exact_truth AND cat.exact_state='F') active_equivalent_when_exact_closed
            FROM truth t JOIN official_loss_sidecar cat USING(query_id)
            LEFT JOIN acceptable_operational a USING(query_id) LEFT JOIN master m ON a.query_id=m.query_id AND a.siret=m.siret GROUP BY t.query_id)
          SELECT t.*,cat.official_catalog_present,cat.parent_in_evidence,
            (pool.siret IS NOT NULL) site_in_reservoir,(m.siret IS NOT NULL) exact_hit,m.candidate_rank,
            (m.siret IS NOT NULL OR coalesce(op.operational_hit,false)) operational_hit,
            coalesce(op.active_equivalent_when_exact_closed,false) active_equivalent_when_exact_closed,
            (pool.base_rank<=1750) baseline_hit,
            CASE WHEN m.siret IS NOT NULL THEN 'HIT'
                 WHEN NOT cat.official_catalog_present THEN 'OFFICIAL_CATALOG_ABSENT'
                 WHEN NOT cat.parent_in_evidence THEN 'PARENT_NOT_RETRIEVED'
                 WHEN pool.siret IS NULL THEN 'SITE_NOT_EXPANDED'
                 ELSE 'EVICTED_RESERVOIR_TO_MASTER2000' END loss_stage
          FROM truth t JOIN official_loss_sidecar cat USING(query_id)
          LEFT JOIN candidates pool ON t.query_id=pool.query_id AND t.gt_siret=pool.siret
          LEFT JOIN master m ON t.query_id=m.query_id AND t.gt_siret=m.siret JOIN op USING(query_id)''')
        df=c.execute('SELECT * FROM outcomes ORDER BY fold,query_id').df()
        if len(df)!=1034 or df.query_id.duplicated().any():raise ValueError('Duplicated outcome')
        baseline=df.baseline_hit.fillna(False)
        if not baseline.eq(df.cohort_kind.eq('BASE_TOP1750_CONTROL')).all():raise ValueError('Sealed baseline/cohort mismatch')
        if not df.loc[baseline,'exact_hit'].all():raise ValueError('Baseline regression')
        cap=c.execute('SELECT max(n) FROM (SELECT query_id,count(*) n FROM master GROUP BY query_id)').fetchone()[0]
        if cap>2000:raise ValueError('Master exceeds cap')
        segments={}
        for dimension in ('fold','data_origin','qualification_v2','qualification_v3','label_kind','ground_truth_state','cohort_kind'):
            segments[dimension]={str(key):summarize(group) for key,group in df.fillna({dimension:'UNKNOWN'}).groupby(dimension,dropna=False)}
        misses=df.loc[~baseline];rescues=int(misses.exact_hit.sum())
        for name in ('official_loss_sidecar','acceptable_operational','outcomes'):
            c.execute(f'COPY (SELECT * FROM {name} ORDER BY query_id) TO {sql_path(output/(name+".parquet"))} (FORMAT PARQUET,COMPRESSION ZSTD)')
        c.close()
    report={'schema':'sireto-v20-master-evaluation-1','decision':'PIVOT_GENERATION','scope':'1034_QUERY_DEVELOPMENT_WORKBENCH',
      'config':record(config_path),'generation_manifest':record(mp),'label_ledger':record(ledger),'official_sites':record(sites),
      'primary':summarize(df),'rescues_of_522':rescues,'control_hits':int(df.loc[baseline,'exact_hit'].sum()),'max_master_sirets':int(cap),
      'loss_counts':{str(k):int(v) for k,v in df.loss_stage.value_counts().items()},'segments':segments,
      'conditional_projection':{'hits':23049+rescues,'denominator':23571,'recall':(23049+rescues)/23571,'measured_national_recall':False,'assumption':'same prefix preserved on all national queries'},
      'historical_all_human':summarize(df),'prerank500_measured':False,'recall100_measured':False,'ready_for_fold1':False,
      'fold0_opened':False,'fold1_opened':False,'positive_injection':False,'resources':budget.report(),
      'files':{p.name:record(p) for p in output.iterdir() if p.is_file()}}
    report['implementation'] = code
    write_json(output/'manifest.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--generation',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.config,a.generation,a.output)
