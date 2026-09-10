"""Reconstruct all non-legacy channels and frozen Masters from CRM inputs.

Only technical parity on <=16 preregistered inputs; never read human truths.
Old artifacts are comparison targets, not retrieval inputs.
"""
import argparse
from collections import defaultdict
from dataclasses import asdict
import gc
import json
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.retrieval_v31_replay_preflight import DATA,ROOT,PANEL,RESOURCE,input_rows,normalize_input
from src.xgb_matcher.retrieval_v20_workbench import Budget,record,verify,write_json,sql_path
from src.xgb_matcher.retrieval_geo_v16 import normalize_street
from src.xgb_matcher.retrieval_site_inventory_v17 import split_street_number,build_parent_rows
from src.xgb_matcher.retrieval_v21_membership import build_tables,open_workbench_db
from src.xgb_matcher.retrieval_v23_joint_parent import NAME_SELECTION_SQL,build_features,make_master
from scripts.retrieval_v29_consolidated import predict


def cfg(name):return json.loads((ROOT/'config'/name).read_text())


def compare(c,actual,expected,columns):
    cols=','.join(columns)
    a=f'SELECT {cols} FROM read_parquet({sql_path(actual)})'
    b=f'SELECT {cols} FROM read_parquet({sql_path(expected)}) SEMI JOIN chosen USING(query_id,fold)'
    missing=c.execute(f'SELECT count(*) FROM (({b}) EXCEPT ALL ({a}))').fetchone()[0]
    added=c.execute(f'SELECT count(*) FROM (({a}) EXCEPT ALL ({b}))').fetchone()[0]
    return dict(missing_rows=missing,added_rows=added,pass_parity=missing==added==0)


def run(output,legacy,count):
    from scripts.run_retrieval_parent_v16_workbench import _build_backends
    from src.xgb_matcher.retrieval_parent_v16 import generate_batch_signals,ParentChannelSpec
    from scripts.run_retrieval_geo_v16_workbench import _resolve_ban,_write_geo_neighbors
    from scripts.run_retrieval_site_inventory_v17 import _write_address_signals,_text_orders
    from scripts.run_retrieval_parent_discovery_v18 import _write_structured_signals,_write_direct_signals
    from scripts.retrieval_v30_joint_search import search,CHANNEL
    from src.xgb_matcher.hierarchical_retrieval import TantivyBackend
    if not 1<=count<=16:raise ValueError('Preflight limit')
    lm=json.loads((legacy/'manifest.json').read_text())
    if lm['status']!='PASS_LEGACY_RECONSTRUCTION' or lm['queries']!=count:raise ValueError('Legacy prerequisite')
    records=json.loads((legacy/'legacy_rows.json').read_text())
    ids=[(r['query_id'],int(r['fold'])) for r in records]
    selected=dict(ids);rows={r['query_id']:r for r in input_rows() if r['query_id'] in selected}
    queries=[(qid,fold,normalize_input(rows[qid])) for qid,fold in ids]
    q=[];geo_q=[];address_q=[];raw=[]
    for i,(qid,fold,r) in enumerate(queries):
        number,suffix=split_street_number(r.number);street=normalize_street(r.address,r.number)
        q.append(dict(query_id=qid,fold=fold,crm_name=r.name,crm_number=number,crm_suffix=suffix,crm_street=street,crm_insee=r.insee,crm_postcode=r.postcode))
        geo_q.append(dict(query_id=qid,fold=fold,crm_number=r.number,crm_street_norm=street,crm_insee=r.insee,crm_postcode=r.postcode))
        address_q.append(dict(query_id=qid,fold=fold,crm_number=number,crm_suffix=suffix,crm_street_norm=street,crm_insee=r.insee,crm_postcode=r.postcode))
        for x in pq.read_table(legacy/f'legacy_{i:02d}.parquet').to_pylist():
            raw.append(dict(query_id=qid,fold=fold,siret=x['siret'],siren=x['siren'],union_rank=x['rank']))
    pq.write_table(pa.Table.from_pylist(raw),output/'raw.parquet')
    v16=cfg('retrieval_all_human_v16_parent_workbench.json');v17=cfg('retrieval_all_human_v17_site_inventory_oracle.json')
    v18=cfg('retrieval_all_human_v18_parent_discovery_generation.json');v23=cfg('retrieval_v23_joint_parent.json');v30=cfg('retrieval_v30_joint_search.json')
    backend=_build_backends(v16);specs=tuple(ParentChannelSpec.from_mapping(r) for r in v16['channels'])
    text_signals=generate_batch_signals(queries=queries,backends=backend,specs=specs,workers=4)
    pq.write_table(pa.Table.from_pylist(text_signals),output/'parent_signals.parquet')
    del backend;gc.collect();print('REBUILT_NATIVE_PARENT_CHANNELS',flush=True)
    geo=cfg('retrieval_geo_v16_sidequest.json')['authorities']
    for pathkey,hashkey in [('sirene_snapshot_path','sirene_snapshot_sha256'),('ban_silver_path','ban_silver_sha256')]:
        verify(dict(path=geo[pathkey],sha256=geo[hashkey]))
    c=open_workbench_db(output);c.register('chosen',pa.Table.from_pylist([dict(query_id=k,fold=f) for k,f in ids]))
    qt=pa.Table.from_pylist(geo_q);ban=_resolve_ban(c,qt,Path(geo['ban_silver_path']))
    pq.write_table(ban,output/'ban.parquet')
    _write_geo_neighbors(c,snapshot_path=Path(geo['sirene_snapshot_path']),query_surface=qt,query_geo=ban,
        output_path=output/'geo_sites.parquet',parent_output_path=output/'geo_parents.parquet',neighbor_cap=512)
    _write_address_signals(c,queries=pa.Table.from_pylist(address_q),sidecar_path=Path(v23['sites']['path']),
        output_path=output/'address.parquet',minimum_similarity=.84,cap=512)
    text_orders=_text_orders(text_signals,v16);geo_orders=defaultdict(list);address_orders=defaultdict(list)
    for r in pq.read_table(output/'geo_parents.parquet').to_pylist():geo_orders[r['query_id']].append(r['siren'])
    for r in pq.read_table(output/'address.parquet').to_pylist():address_orders[r['query_id']].append(r['siren'])
    parents=[];presence=[]
    for qid,fold in ids:
        p,s=build_parent_rows(query_id=qid,fold=fold,source_orders=dict(TEXT=text_orders.get(qid,()),ADDRESS=address_orders[qid],GEO=geo_orders[qid]),
          variants=v17['recipe']['variants'],primary_variant=v17['recipe']['primary_variant'],cap=512)
        parents+=p;presence+=s
    pq.write_table(pa.Table.from_pylist(parents),output/'parent_candidates.parquet')
    pq.write_table(pa.Table.from_pylist(presence),output/'parent_presence.parquet')
    paths=dict(raw_union=output/'raw.parquet',v17_parent_presence=output/'parent_presence.parquet',v17_parent_candidates=output/'parent_candidates.parquet')
    for key in ('rne_roles','bodacc_relations'):paths[key]=verify(v18['generation_authorities'][key])
    paths['temporal_glob']=Path(v18['generation_authorities']['temporal_sources']['source_root'])/'*'/'succession_edges.parquet'
    v18['resources']['duckdb_threads']=2;v18['resources']['search_workers']=4
    _write_structured_signals(config=v18,query_rows=tuple(ids),paths=paths,output_path=output/'structured.parquet')
    _write_direct_signals(config=v18,retrieval_queries=queries,index_root=Path(v18['generation_authorities']['hierarchical_index']['path']),output_path=output/'direct.parquet')
    backend=TantivyBackend(v18['generation_authorities']['hierarchical_index']['path']);joint=[]
    for qid,fold,query in queries:
        joint.extend(dict(query_id=qid,fold=fold,**r) for r in search(backend,query))
    pq.write_table(pa.Table.from_pylist(joint),output/'joint.parquet');del backend;gc.collect()
    print('REBUILT_ALL_DISCOVERY_CHANNELS',flush=True)
    va=v17['authorities'];v19=cfg('retrieval_all_human_v19_parent_fusion.json')['authorities']
    checks={}
    for name,old,cols in [
      ('parent_signals',Path(va['v16_parent_generation_dir'])/'parent_signals.parquet',['query_id','fold','channel','channel_rank','siren','score']),
      ('parent_presence',Path(va['v16_parent_generation_dir']).parent.parent/'retrieval_site_inventory_v17_generation/69ddd52660b41f68/parent_presence.parquet',['query_id','fold','siren']),
      ('structured',Path(v19['structured_signals']['path']),['query_id','fold','channel','channel_rank','siren','source_siret']),
      ('direct',Path(v19['siret_native_signals']['path']),['query_id','fold','channel','channel_rank','siren','source_siret']),
      ('joint',DATA/'audits/retrieval_v30_discovery/8cb4ad779f11ac84/joint_signals.parquet',['query_id','fold','siret','siren','candidate_rank','evidence_rank','score'])]:
        checks[name]=compare(c,output/f'{name}.parquet',old,cols)
    c.close()
    for version in ('v23','v30'):
        stage=output/version;stage.mkdir();c=open_workbench_db(stage)
        c.register('chosen',pa.Table.from_pylist([dict(query_id=k,fold=f) for k,f in ids]));c.register('q',pa.Table.from_pylist(q))
        c.execute('CREATE TABLE raw AS SELECT query_id,fold,siren,siret,union_rank base_rank FROM read_parquet(?)',[str(output/'raw.parquet')])
        c.execute('CREATE TABLE signals AS SELECT query_id,fold,siren,source_siret,channel,channel_rank FROM read_parquet(?)',[[str(output/'structured.parquet'),str(output/'direct.parquet')]])
        if version=='v30':c.execute('INSERT INTO signals SELECT query_id,fold,siren,siret,?,evidence_rank FROM read_parquet(?)',[CHANNEL,str(output/'joint.parquet')])
        c.execute('''CREATE TABLE sites AS SELECT s.* FROM read_parquet(?) s SEMI JOIN
          (SELECT siren FROM signals UNION SELECT siren FROM raw) p USING(siren)''',[str(verify(v23['sites']))])
        build_tables(c);c.execute('DROP TABLE master')
        c.execute('CREATE TABLE pool AS SELECT query_id,fold,siren,siret,base_rank,parent_rrf,direct_rrf FROM candidates')
        c.execute(f'CREATE VIEW name_source AS SELECT * FROM read_parquet({sql_path(verify(v23["names"]))})')
        c.execute(NAME_SELECTION_SQL);build_features(c)
        model_paths={p.name:p for p in (DATA/'audits/retrieval_v23_oof_master/bb12058773f80d6f').iterdir() if p.is_file()}
        predict(c,model_paths);make_master(c)
        c.execute(f'COPY master TO {sql_path(stage/"master.parquet")} (FORMAT PARQUET)')
        old=DATA/('audits/retrieval_v23_oof_master/bb12058773f80d6f/master.parquet' if version=='v23' else 'audits/retrieval_v30_master/8cb4ad779f11ac84/master.parquet')
        cols=c.execute('DESCRIBE master').fetchall();checks[version]=compare(c,stage/'master.parquet',old,[x[0] for x in cols]);c.close()
    return dict(status='PASS_CHANNELS_AND_MASTER_PARITY' if all(v['pass_parity'] for v in checks.values()) else 'STOP_CHANNEL_OR_MASTER_PARITY',
       checks=checks,queries=count,quality_measured=False,truth_columns_opened=False,legacy_manifest=record(legacy/'manifest.json'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--legacy',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--count',type=int,default=16);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    with Budget(RESOURCE) as budget:result=run(a.output,a.legacy,a.count)
    result.update(resources=budget.report(),implementation=record(Path(__file__)))
    write_json(a.output/'manifest.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
