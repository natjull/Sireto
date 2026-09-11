"""Bounded multi-process verification of the frozen v32 succession repair.

prepare/run/compare are separate processes. Validation uses the truth-free
v31 reconstructed channels, not a claimed full production replay.
"""
import argparse
import json
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.retrieval_v31_replay_preflight import DATA,ROOT,RESOURCE,MODEL,input_rows,normalize_input
from src.xgb_matcher.retrieval_v20_workbench import Budget,verify,record,write_json,sql_path
from src.xgb_matcher.retrieval_succession_v32 import build,reference,CHANNEL
from src.xgb_matcher.retrieval_v21_membership import open_workbench_db,build_tables
from src.xgb_matcher.retrieval_v23_joint_parent import NAME_SELECTION_SQL,build_features,make_master
from src.xgb_matcher.retrieval_geo_v16 import normalize_street
from src.xgb_matcher.retrieval_site_inventory_v17 import split_street_number
from scripts.retrieval_v29_consolidated import predict
from scripts.retrieval_v30_joint_search import CHANNEL as JOINT_CHANNEL

SOURCE=DATA/'audits/retrieval_v31_replay_preflight/channels16'


def code():
    return {str(p.relative_to(ROOT)):record(p) for p in [Path(__file__),ROOT/'src/xgb_matcher/retrieval_succession_v32.py',
      ROOT/'docs/retrieval_v32_stable_succession_contract.md',ROOT/'scripts/retrieval_v29_consolidated.py',
      ROOT/'src/xgb_matcher/retrieval_v23_joint_parent.py',ROOT/'src/xgb_matcher/retrieval_v21_membership.py']}


def prepare(output):
    a=json.loads((ROOT/'config/retrieval_all_human_v18_parent_discovery_generation.json').read_text())['generation_authorities']['temporal_sources']
    manifest=json.loads(verify(dict(path=Path(a['index_path'])/'manifest.json',sha256=a['manifest_sha256'])).read_text())
    edges=[verify(dict(path=Path(a['source_root'])/s['projection_build_id']/'succession_edges.parquet',sha256=s['successions_sha256'])) for s in manifest['sources']]
    c=open_workbench_db(output)
    files={name:record(SOURCE/name) for name in ('raw.parquet','structured.parquet','direct.parquet','joint.parquet')}
    c.execute('CREATE TABLE raw AS SELECT * FROM read_parquet(?)',[str(verify(files['raw.parquet']))])
    ids=c.execute('SELECT DISTINCT query_id,fold FROM raw ORDER BY query_id').fetchall()
    if len(ids)!=16 or any(f not in (2,3,4) for _,f in ids):raise ValueError('Bounded scope')
    c.execute('''CREATE TABLE edges AS SELECT DISTINCT e.* FROM read_parquet(?) e
      SEMI JOIN (SELECT siret FROM raw WHERE union_rank<=250) r
      ON e.predecessor_siret=r.siret OR e.successor_siret=r.siret''',[[str(p) for p in edges]])
    c.execute(f'COPY edges TO {sql_path(output/"edges.parquet")} (FORMAT PARQUET)')
    rows={r['query_id']:r for r in input_rows() if r['query_id'] in dict(ids)};q=[]
    for qid,fold in ids:
        r=normalize_input(rows[qid]);number,suffix=split_street_number(r.number)
        q.append(dict(query_id=qid,fold=fold,crm_name=r.name,crm_number=number,crm_suffix=suffix,
          crm_street=normalize_street(r.address,r.number),crm_insee=r.insee,crm_postcode=r.postcode))
    pq.write_table(pa.Table.from_pylist(q),output/'queries.parquet')
    v23=json.loads((ROOT/'config/retrieval_v23_joint_parent.json').read_text())
    for key in ('names','sites'):files[key]=record(verify(v23[key]))
    models=json.loads((MODEL/'manifest.json').read_text())['files']
    for fold in (2,3,4):
        for ext in ('.json','_split.json'):
            name=f'outer{fold}{ext}';files[name]=record(verify(models[name]))
    build(c)
    actual=c.execute('SELECT * FROM v32_proof ORDER BY query_id,fold,channel_rank').fetch_arrow_table().to_pylist()
    expected=reference(c.execute('SELECT * FROM raw').fetch_arrow_table().to_pylist(),c.execute('SELECT * FROM edges').fetch_arrow_table().to_pylist())
    if actual!=expected:raise ValueError('Independent reference mismatch')
    parent_delta=c.execute('''SELECT count(*) FROM (
      (SELECT query_id,fold,siren,seed_rank,channel_rank FROM v32_signals)
      EXCEPT ALL
      (SELECT query_id,fold,siren,seed_rank,channel_rank FROM read_parquet(?) WHERE channel=?))''',
      [str(SOURCE/'structured.parquet'),CHANNEL]).fetchone()[0]
    old_count=c.execute('SELECT count(*) FROM read_parquet(?) WHERE channel=?',[str(SOURCE/'structured.parquet'),CHANNEL]).fetchone()[0]
    if parent_delta or old_count!=len(actual):raise ValueError('Parent support/ranking changed')
    result=dict(status='PASS_PREPARED_REFERENCE',queries=16,proof_count=len(actual),source_files=files,
      edges=record(output/'edges.parquet'),queries_file=record(output/'queries.parquet'),
      temporal_files=[record(p) for p in edges],code=code(),truth_columns_opened=False,quality_measured=False)
    c.close();return result


def run(prepared,output,threads,shard_size,reverse):
    m=json.loads((prepared/'manifest.json').read_text())
    if m['status']!='PASS_PREPARED_REFERENCE' or m['code']!=code():raise ValueError('Preparation drift')
    files={k:verify(v) for k,v in m['source_files'].items()}
    q=pq.read_table(verify(m['queries_file'])).to_pylist();q.sort(key=lambda r:r['query_id'],reverse=reverse)
    emitted=[];masters={v:[] for v in ('v23_stable','v30_stable')}
    proof_tables=[]
    for offset in range(0,len(q),shard_size):
        shard=q[offset:offset+shard_size];stage=output/f'shard{offset:02d}';stage.mkdir()
        c=open_workbench_db(stage);c.execute(f'SET threads={threads}');c.register('q',pa.Table.from_pylist(shard))
        c.execute('CREATE TABLE raw AS SELECT r.* FROM read_parquet(?) r SEMI JOIN q USING(query_id,fold)',[str(files['raw.parquet'])])
        order='DESC' if reverse else 'ASC'
        c.execute(f'CREATE TABLE edges AS SELECT * FROM read_parquet(?) ORDER BY relation_id {order}',[str(verify(m['edges']))])
        build(c);signals=c.execute('SELECT * FROM v32_signals ORDER BY query_id,fold,channel_rank').fetch_arrow_table()
        proof_tables.append(c.execute('SELECT * FROM v32_proof ORDER BY query_id,fold,channel_rank').fetch_arrow_table())
        emitted.append(signals);c.close()
        for version in masters:
            vdir=stage/version;vdir.mkdir();c=open_workbench_db(vdir);c.execute(f'SET threads={threads}')
            c.register('q',pa.Table.from_pylist(shard));c.register('stable',signals)
            c.execute('CREATE TABLE raw AS SELECT query_id,fold,siren,siret,union_rank base_rank FROM read_parquet(?) SEMI JOIN q USING(query_id,fold)',[str(files['raw.parquet'])])
            c.execute('''CREATE TABLE signals AS SELECT query_id,fold,siren,source_siret,channel,channel_rank
              FROM read_parquet(?) SEMI JOIN q USING(query_id,fold) WHERE channel<>?''',[[str(files['structured.parquet']),str(files['direct.parquet'])],CHANNEL])
            c.execute('INSERT INTO signals SELECT query_id,fold,siren,source_siret,channel,channel_rank FROM stable')
            if version=='v30_stable':c.execute('INSERT INTO signals SELECT query_id,fold,siren,siret,?,evidence_rank FROM read_parquet(?) SEMI JOIN q USING(query_id,fold)',[JOINT_CHANNEL,str(files['joint.parquet'])])
            c.execute('CREATE TABLE sites AS SELECT s.* FROM read_parquet(?) s SEMI JOIN (SELECT siren FROM signals UNION SELECT siren FROM raw) p USING(siren)',[str(files['sites'])])
            build_tables(c);c.execute('DROP TABLE master')
            c.execute('CREATE TABLE pool AS SELECT query_id,fold,siren,siret,base_rank,parent_rrf,direct_rrf FROM candidates')
            c.execute(f'CREATE VIEW name_source AS SELECT * FROM read_parquet({sql_path(files["names"])})')
            c.execute(NAME_SELECTION_SQL);build_features(c);predict(c,files);make_master(c)
            masters[version].append(c.execute('SELECT * FROM master ORDER BY query_id,candidate_rank').fetch_arrow_table());c.close()
    pq.write_table(pa.concat_tables(emitted).sort_by([('query_id','ascending'),('channel_rank','ascending')]),output/'signals.parquet')
    pq.write_table(pa.concat_tables(proof_tables).sort_by([('query_id','ascending'),('channel_rank','ascending')]),output/'proof.parquet')
    for v,tables in masters.items():pq.write_table(pa.concat_tables(tables).sort_by([('query_id','ascending'),('candidate_rank','ascending')]),output/f'{v}.parquet')
    return dict(status='SEALED_STABILITY_RUN',threads=threads,shard_size=shard_size,reverse=reverse,
      code=code(),prepared=record(prepared/'manifest.json'),files={p.name:record(p) for p in output.glob('*.parquet')},
      truth_columns_opened=False,quality_measured=False)


def compare_runs(paths,output):
    manifests=[json.loads((p/'manifest.json').read_text()) for p in paths]
    if len(paths)<3 or any(m['status']!='SEALED_STABILITY_RUN' or m['code']!=code() or m['prepared']!=manifests[0]['prepared'] for m in manifests):raise ValueError('Runs drift')
    c=duckdb.connect();checks={}
    for name in ('proof.parquet','signals.parquet','v23_stable.parquet','v30_stable.parquet'):
        c.execute('CREATE OR REPLACE TABLE base AS SELECT * FROM read_parquet(?)',[str(verify(manifests[0]['files'][name]))])
        counts=[]
        for m in manifests[1:]:
            c.execute('CREATE OR REPLACE TABLE other AS SELECT * FROM read_parquet(?)',[str(verify(m['files'][name]))])
            count=c.execute('SELECT count(*) FROM ((TABLE base EXCEPT ALL TABLE other) UNION ALL (TABLE other EXCEPT ALL TABLE base))').fetchone()[0]
            if count:raise ValueError(f'Unstable {name}: {count}')
            counts.append(count)
        checks[name]=counts
    return dict(status='PASS_STABLE_WITNESSES_AND_MASTERS',checks=checks,runs=[record(p/'manifest.json') for p in paths],
      code=code(),queries=16,quality_measured=False,truth_columns_opened=False,full_replay_performed=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','run','compare']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prepared',type=Path);p.add_argument('--threads',type=int,choices=[1,2,8],default=2);p.add_argument('--shard-size',type=int,choices=[1,4,16],default=16);p.add_argument('--reverse',action='store_true');p.add_argument('--runs',type=Path,nargs='+');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    with Budget(RESOURCE) as budget:
        if a.phase=='prepare':result=prepare(a.output)
        elif a.phase=='run':result=run(a.prepared,a.output,a.threads,a.shard_size,a.reverse)
        else:result=compare_runs(a.runs,a.output)
    result['resources']=budget.report();write_json(a.output/'manifest.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('code','temporal_files','source_files')}),flush=True)


if __name__=='__main__':main()
