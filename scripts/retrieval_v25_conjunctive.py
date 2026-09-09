#!/usr/bin/env python3
"""Separate truth-free discovery and truth-side diagnostic union evaluation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.xgb_matcher.hierarchical_retrieval import TantivyBackend
from src.xgb_matcher.retrieval_v20_workbench import Budget, record, verify, write_json, sql_path
from src.xgb_matcher.retrieval_v21_membership import manifest_files,open_workbench_db
from src.xgb_matcher.retrieval_v25_conjunctive import config_at,code_records,index_inventory,search
from scripts.run_retrieval_parent_discovery_v18 import _load_retrieval_queries


SIGNALS=pa.schema([('query_id',pa.string()),('fold',pa.int8()),('siren',pa.string()),('siret',pa.string()),
                   ('score',pa.float64()),('candidate_rank',pa.int32()),('evidence_rank',pa.int32())])


def save(output,result):
    result['implementation']=code_records()
    result['files']={p.name:record(p) for p in output.iterdir() if p.is_file()}
    write_json(output/'manifest.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('implementation','files','index_inventory')}),flush=True)


def generate(config_path,output,preflight=None):
    config=config_at(config_path);source=json.loads(verify(config['source_config']).read_text())['generation_authorities']
    queries_path=verify(source['queries']);raw=verify(source['raw_union'])
    index=Path(source['hierarchical_index']['path']);scene=Path(source['scene_ir']['path'])
    sources=[queries_path,raw]
    for key,root in [('hierarchical_index',index),('scene_ir',scene)]:
        sources.append(verify(dict(path=str(root/'manifest.json'),sha256=source[key]['manifest_sha256'])))
    if preflight is not None and not 1<=preflight<=16:
        raise ValueError('Technical preflight must be 1..16')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    inventory=index_inventory(index/'tantivy');meta=record(index/'tantivy/meta.json')
    with Budget(config['resources']) as budget:
        ids=pq.read_table(queries_path).to_pylist()
        if len(ids)!=1034 or len({r['query_id'] for r in ids})!=1034 or {r['fold'] for r in ids}!={2,3,4}:
            raise ValueError('Query population drift')
        ids=sorted(ids,key=lambda r:r['query_id'])[:preflight or 1034]
        queries=_load_retrieval_queries(tuple((r['query_id'],r['fold']) for r in ids),scene)
        backend=TantivyBackend(index)
        # Cache geographic existence before worker access; searches only read index.
        for _,_,q in queries:
            if q.insee:backend._has_insee(q.insee)
        def one(item):
            qid,fold,q=item
            return [dict(query_id=qid,fold=fold,**r) for r in search(backend,q)]
        allrows=[]
        with ThreadPoolExecutor(max_workers=4) as pool:
            for i,rows in enumerate(pool.map(one,queries),1):
                allrows.extend(rows)
                if i%100==0:print(json.dumps(dict(processed=i,total=len(queries))),flush=True)
        qrows=[dict(query_id=qid,fold=fold,**asdict(q)) for qid,fold,q in queries]
        pq.write_table(pa.Table.from_pylist(qrows),output/'queries.parquet',compression='zstd')
        pq.write_table(pa.Table.from_pylist(allrows,schema=SIGNALS),output/'signals.parquet',compression='zstd')
        if index_inventory(index/'tantivy')!=inventory or record(index/'tantivy/meta.json')!=meta:
            raise ValueError('Index changed during generation')
    save(output,dict(schema='sireto-v25-conjunctive-generation-1',config=record(config_path),sources=[record(p) for p in sources],
        index_meta=meta,index_inventory=inventory,index_full_bytes_rehashed=False,preflight=preflight is not None,
        query_count=len(qrows),signal_count=len(allrows),eligible_queries=sum(bool(r['number'] and r['address']) for r in qrows),
        queries_with_hits=len({r['query_id'] for r in allrows}),truth_columns_opened=False,positive_injection=False,
        fold0_opened=False,fold1_opened=False,model_trained=False,site_expansion=False,resources=budget.report()))


def summarize(df):
    r={'queries':len(df)}
    for stage in ('base','channel','union'):
        for target in ('parent','exact','operational'):
            r[f'{stage}_{target}']=int(df[f'{stage}_{target}'].sum())
    r['new_parents']=int((df.union_parent & ~df.base_parent).sum())
    r['new_exact_sites']=int((df.union_exact & ~df.base_exact).sum())
    r['lost_exact_sites']=int((~df.union_exact & df.base_exact).sum())
    return r


def evaluate(config_path,generation,output):
    config=config_at(config_path);gm,gf=manifest_files(record(Path(generation)/'manifest.json'))
    if (gm['schema']!='sireto-v25-conjunctive-generation-1' or gm['preflight'] or gm['truth_columns_opened']
        or gm['query_count']!=1034 or gm['config']['sha256']!=record(config_path)['sha256'] or gm['implementation']!=code_records()):
        raise ValueError('Generation authority mismatch')
    bm,bf=manifest_files(config['baseline_evaluation']);pm,pf=manifest_files(config['baseline_generation'])
    if bm['schema']!='sireto-v23-evaluation-1' or pm['schema']!='sireto-v21-generation-1':
        raise ValueError('Baseline authority')
    official=verify(bm['official_loss_sidecar']);acceptable=verify(bm['acceptable_operational_sidecar'])
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    with Budget(config['resources']) as budget:
        c=open_workbench_db(output)
        for name,path in [('q',gf['queries.parquet']),('s',gf['signals.parquet']),('b',pf['candidates.parquet']),
                          ('t',bf['outcomes.parquet']),('official',official),('acceptable',acceptable)]:
            c.execute(f'CREATE TEMP TABLE {name} AS SELECT * FROM read_parquet(?)',[str(path)])
        if c.execute('SELECT count(*),count(DISTINCT query_id) FROM t WHERE fold IN (2,3,4)').fetchone()!=(1034,1034):
            raise ValueError('Truth population')
        if c.execute('SELECT count(*) FROM t ANTI JOIN q USING(query_id,fold)').fetchone()[0]:
            raise ValueError('Query mismatch')
        cap=c.execute('SELECT coalesce(max(n),0) FROM (SELECT query_id,count(*) n FROM s GROUP BY 1)').fetchone()[0]
        if cap>512 or c.execute('SELECT count(*),count(DISTINCT (query_id,siret)) FROM s').fetchone()[0]!=c.execute('SELECT count(DISTINCT (query_id,siret)) FROM s').fetchone()[0]:
            raise ValueError('Channel cap or uniqueness')
        c.execute('''CREATE TEMP TABLE u AS SELECT query_id,fold,siren,siret FROM b UNION SELECT query_id,fold,siren,siret FROM s''')
        c.execute('''CREATE TEMP TABLE presence AS SELECT t.query_id,t.fold,t.gt_siren,t.gt_siret,t.cohort_kind,t.data_origin,
          t.label_kind,t.qualification_v2,t.qualification_v3,t.ground_truth_state,t.exact_hit baseline_master_exact,
          t.operational_hit baseline_master_operational FROM t''')
        for stage,table in [('base','b'),('channel','s'),('union','u')]:
            c.execute(f'''CREATE OR REPLACE TEMP TABLE presence AS SELECT p.*,
              EXISTS(SELECT 1 FROM {table} x WHERE x.query_id=p.query_id AND x.siren=p.gt_siren) {stage}_parent,
              EXISTS(SELECT 1 FROM {table} x WHERE x.query_id=p.query_id AND x.siret=p.gt_siret) {stage}_exact,
              (EXISTS(SELECT 1 FROM {table} x WHERE x.query_id=p.query_id AND x.siret=p.gt_siret) OR
               EXISTS(SELECT 1 FROM {table} x JOIN acceptable a USING(query_id,siret) WHERE x.query_id=p.query_id)) {stage}_operational
              FROM presence p''')
        df=c.execute('SELECT * FROM presence ORDER BY fold,query_id').df()
        if len(df)!=1034 or ((df.base_exact & ~df.union_exact)|(df.base_parent & ~df.union_parent)).any():
            raise ValueError('Nonmonotone diagnostic union')
        c.execute(f'COPY presence TO {sql_path(output/"outcomes.parquet")} (FORMAT PARQUET,COMPRESSION ZSTD)')
        union_rows=c.execute('SELECT count(*) FROM u').fetchone()[0]
        segments={dim:{str(k):summarize(v) for k,v in df.groupby(dim,dropna=False)}
                  for dim in ('fold','cohort_kind','data_origin','label_kind','qualification_v2','qualification_v3','ground_truth_state')}
        c.close()
    save(output,dict(schema='sireto-v25-conjunctive-evaluation-1',decision='PIVOT_GENERATION',config=record(config_path),
        generation_manifest=record(Path(generation)/'manifest.json'),baseline_generation=config['baseline_generation'],
        baseline_evaluation=config['baseline_evaluation'],official_loss_sidecar=record(official),acceptable_operational_sidecar=record(acceptable),
        primary=summarize(df),historical_all_human=summarize(df),segments=segments,union_reference_rows=union_rows,
        max_channel_sites=int(cap),master_measured=False,recall100_measured=False,ready_for_fold1=False,
        national_projection=None,fold0_opened=False,fold1_opened=False,reused_development_panel=True,resources=budget.report()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['generate','evaluate'])
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--generation',type=Path);p.add_argument('--preflight',type=int)
    a=p.parse_args()
    if a.action=='generate':generate(a.config,a.output,a.preflight)
    else:
        if not a.generation or a.preflight:p.error('Evaluation requires a full generation')
        evaluate(a.config,a.generation,a.output)
