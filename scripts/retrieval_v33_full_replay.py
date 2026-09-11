"""Truth-blind, bounded CRM -> paired stable Master replay. No fit or evaluation.

Preparation exports an explicit inference scope from the sealed v31 isolation
audit. Generation never reads a human label or a historical candidate cache.
Each completed shard is immutable; failures keep partial evidence and STOP.
"""
import argparse
from collections import defaultdict
from dataclasses import replace
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb

from scripts.retrieval_v31_replay_preflight import DATA, ROOT, INPUT, MODEL, SAFE, input_rows, normalize_input
from src.xgb_matcher.retrieval_v20_workbench import Budget, record, verify, write_json, sql_path
from src.xgb_matcher.retrieval_v21_membership import open_workbench_db, build_tables
from src.xgb_matcher.retrieval_v23_joint_parent import FEATURES, NAME_SELECTION_SQL, build_features, make_master
from src.xgb_matcher.retrieval_succession_v32 import build as succession, CHANNEL as SUCCESSION
from src.xgb_matcher.retrieval_geo_v16 import normalize_street
from src.xgb_matcher.retrieval_site_inventory_v17 import split_street_number, build_parent_rows
from src.xgb_matcher.retrieval_parent_v16 import SIGNAL_SCHEMA as PARENT_SCHEMA, ParentChannelSpec, generate_batch_signals
from scripts.run_retrieval_parent_v16_workbench import _build_backends
from scripts.run_retrieval_geo_v16_workbench import _resolve_ban, _write_geo_neighbors
from scripts.run_retrieval_site_inventory_v17 import _write_address_signals, _text_orders
from scripts.run_retrieval_parent_discovery_v18 import _write_structured_signals, _write_direct_signals
from scripts.retrieval_v30_joint_search import search, CHANNEL as JOINT
from src.xgb_matcher.hierarchical_retrieval import TantivyBackend
from src.xgb_matcher.official_evidence_retrieval import OfficialEvidenceRetriever, OfficialEvidenceRetrievalConfig
from src.xgb_matcher.retrieval_all_human_v11 import AllHumanV11Retriever, RetrievalV11Config
from src.xgb_matcher.retrieval_sidecars_v11 import RetrievalSidecarReader

RESOURCE = dict(max_seconds=1800, max_rss_bytes=14*1024**3, max_swapout_pages_delta=0)
POP = DATA/'audits/retrieval_v31_replay_preflight/population/manifest.json'
STABLE = DATA/'audits/retrieval_v32_stability/verification/manifest.json'
CONFIGS = ['retrieval_all_human_v11.json', 'retrieval_ltr_admission_dossier_v2.json',
 'retrieval_all_human_v15_master_feasibility.json', 'retrieval_all_human_v16_parent_workbench.json',
 'retrieval_geo_v16_sidequest.json', 'retrieval_all_human_v17_site_inventory_oracle.json',
 'retrieval_all_human_v18_parent_discovery_generation.json', 'retrieval_v23_joint_parent.json']
SCRIPT_NAMES = ['retrieval_v33_full_replay.py', 'retrieval_v31_replay_preflight.py',
 'run_retrieval_parent_v16_workbench.py', 'run_retrieval_geo_v16_workbench.py',
 'run_retrieval_site_inventory_v17.py', 'run_retrieval_parent_discovery_v18.py',
 'retrieval_v30_joint_search.py', 'retrieval_v29_consolidated.py',
 'retrieval_v22_global_budget.py', '__init__.py']
RAW_SCHEMA = pa.schema([(k, pa.int64() if k in ('fold','union_rank') else pa.string())
                       for k in ('query_id','fold','siren','siret','union_rank')])
PRESENCE_SCHEMA = pa.schema([(k, pa.bool_() if k.startswith('from_') else pa.int64() if k=='fold' else pa.string())
                            for k in ('query_id','fold','siren','from_text','from_address','from_geo')])
PARENTS_SCHEMA = pa.schema([(k, pa.bool_() if k.startswith('from_') or k=='is_primary' else
                            pa.int64() if k in ('fold','parent_rank') else pa.string())
 for k in ('query_id','fold','variant','is_primary','parent_rank','siren','from_text','from_address','from_geo')])
JOINT_SCHEMA = pa.schema([(k, pa.float64() if k=='score' else pa.int64() if k in
 ('fold','candidate_rank','evidence_rank') else pa.string())
 for k in ('query_id','fold','siren','siret','score','candidate_rank','evidence_rank')])


def cfg(name): return json.loads((ROOT/'config'/name).read_text())


def code():
    paths = list((ROOT/'src/xgb_matcher').glob('*.py'))
    paths += [ROOT/'scripts'/n for n in SCRIPT_NAMES]
    paths += [ROOT/'config'/n for n in CONFIGS]
    paths += [ROOT/'docs/retrieval_v33_full_replay_contract.md']
    return {str(p.relative_to(ROOT)): record(p) for p in sorted(paths)}


def stat_record(path):
    p=Path(path);s=p.stat()
    return dict(path=str(p.resolve()), size=s.st_size, mtime_ns=s.st_mtime_ns, inode=s.st_ino)


def select_edges(c,paths):
    # Equivalent to DISTINCT(LEFT SEMI JOIN on predecessor OR successor), but
    # permits two hash joins instead of a blockwise nested-loop over all seeds.
    c.execute('''CREATE TABLE edges AS
      SELECT e.* FROM read_parquet(?) e SEMI JOIN (SELECT siret FROM raw WHERE union_rank<=250) r
        ON e.predecessor_siret=r.siret
      UNION
      SELECT e.* FROM read_parquet(?) e SEMI JOIN (SELECT siret FROM raw WHERE union_rank<=250) r
        ON e.successor_siret=r.siret''',[paths,paths])


def ordered_inputs(rows):
    ids=[r['query_id'] for r in rows]
    if len(ids)!=len(set(ids)) or any(not isinstance(q,str) or not q for q in ids):
        raise ValueError('Duplicate/invalid query identity')
    if any(r['oof_fold'] not in (2,3,4) for r in rows): raise ValueError('Forbidden fold')
    return sorted(rows,key=lambda r:(hashlib.sha256(('v33-cost|'+r['query_id']).encode()).hexdigest(),r['query_id']))


def prepare(output):
    pop=json.loads(verify(dict(path=POP,sha256='f2232c960a67334dd96b518b6a0e0abbe27915773915f3532aae7d195a3e316d')).read_text())
    stable=json.loads(verify(dict(path=STABLE,sha256='d5c25efeb10a92930683b96b3110a6527a2f08d2994165da6f2f20641ac76422')).read_text())
    if pop['status']!='PASS_POPULATION_AND_INPUT_PARITY' or stable['status']!='PASS_STABLE_WITNESSES_AND_MASTERS':
        raise ValueError('Prerequisites')
    source_files={}; indexes={}
    def pin(key,r): source_files[key]=record(verify(r));return Path(source_files[key]['path'])
    pin('input',dict(path=INPUT,sha256='fb6d29ee1bc44d88162266485bb916d5cc178b46809ef9222b5d45fc80915705'))
    rows=ordered_inputs(input_rows())
    if len(rows)!=23571 or {f:sum(r['oof_fold']==f for r in rows) for f in (2,3,4)}!={2:7767,3:7967,4:7837}:
        raise ValueError('Population drift')
    pq.write_table(pa.Table.from_pylist(rows).select(SAFE),output/'inputs.parquet',compression='zstd')
    pq.write_table(pa.table({'query_id':[r['query_id'] for r in rows], 'fold':[r['oof_fold'] for r in rows]}),output/'scope.parquet')
    mm=json.loads(verify(dict(path=MODEL/'manifest.json',sha256='ac4b279be0f30fbee5cfe7fc0c205c31b9cf50fcc23351795ceb55423e05de81')).read_text())
    audited={r['path']:r for r in pop['sources']}
    for f in (2,3,4):
        for ext in ('.json','_split.json'):
            n=f'outer{f}{ext}';p=pin(n,mm['files'][n])
            if record(p)!=audited[str(p)]: raise ValueError('Model not covered by isolation audit')
        s=json.loads((MODEL/f'outer{f}_split.json').read_text())
        target={r['query_id'] for r in rows if r['oof_fold']==f}
        if set(s['trained_query_ids']) & target or s['target_folds']!=[f] or s['train_folds']!=[x for x in (2,3,4) if x!=f]:
            raise ValueError('Inference isolation')
    v15=cfg(CONFIGS[2])['authorities'];v16=cfg(CONFIGS[3])['authorities']['backends']
    index_authorities=list(v16.values())+[v15['sidecars'],
      dict(path=str(DATA/'indices/dossier_temporal_tail_v1/66f38cc146c51da6'),manifest_sha256='e568e5975940a70952fdd9033d69c208727c015455104d4869eabf89670a46b6'),
      dict(path=str(DATA/'indices/official_sector_data_es_v1/37ea905013a83660'),manifest_sha256='9278c317be6c0f8f92dde7ecd894a8a4002f11f34c176f754f03980e9fb0acf0')]
    for a in index_authorities:
        root=Path(a['path']);p=pin(str(root/'manifest.json'),dict(path=root/'manifest.json',sha256=a['manifest_sha256']))
        indexes[str(root)]=[stat_record(x) for x in sorted(root.rglob('*')) if x.is_file() and not x.name.endswith('.lock')]
        for meta in root.rglob('meta.json'): pin(str(meta),record(meta))
    v23=cfg(CONFIGS[7]);v18=cfg(CONFIGS[6])['generation_authorities'];geo=cfg(CONFIGS[4])['authorities']
    for k in ('sites','names'):pin(k,v23[k])
    for k in ('rne_roles','bodacc_relations'):pin(k,v18[k])
    pin('stock',dict(path=geo['sirene_snapshot_path'],sha256=geo['sirene_snapshot_sha256']))
    pin('ban',dict(path=geo['ban_silver_path'],sha256=geo['ban_silver_sha256']))
    a=v18['temporal_sources'];tm=json.loads((Path(a['index_path'])/'manifest.json').read_text());edges=[]
    for i,s in enumerate(tm['sources']):
        edges.append(str(pin(f'edge{i}',dict(path=Path(a['source_root'])/s['projection_build_id']/'succession_edges.parquet',sha256=s['successions_sha256']))))
    if len(edges)!=100:raise ValueError('Temporal file count')
    return dict(status='SEALED_FULL_CRM_INFERENCE_SCOPE',population=record(POP),stability=record(STABLE),
      inputs=record(output/'inputs.parquet'),scope=record(output/'scope.parquet'),sources=source_files,
      stats={k:stat_record(r['path']) for k,r in source_files.items()},indexes=indexes,edges=edges,
      queries=23571,code=code(),quality_measured=False,truth_columns_opened=False,
      inference_authority='OUTER_FOLD_SCOPE_EXTENDED_BY_SEALED_V31_SIREN_ISOLATION',
      index_integrity='PINNED_MANIFEST_AND_META_PLUS_FILE_STAT_INVENTORY_NOT_FULL_SEGMENT_REHASH')


def authority(prepared):
    m=json.loads((prepared/'manifest.json').read_text())
    if m['status']!='SEALED_FULL_CRM_INFERENCE_SCOPE' or m['code']!=code():raise ValueError('Code/scope drift')
    for r in m['stats'].values():
        if stat_record(r['path'])!=r:raise ValueError('Source changed since full hash')
    for root,inventory in m['indexes'].items():
        now=[stat_record(p) for p in sorted(Path(root).rglob('*')) if p.is_file() and not p.name.endswith('.lock')]
        if now!=inventory:raise ValueError('Index inventory changed')
    verify(m['scope']);verify(m['inputs'])
    return m


def predict(c,m):
    scope=pq.read_table(verify(m['scope'])).to_pylist();allowed={(r['query_id'],r['fold']) for r in scope}
    frames=[]
    for fold in (2,3,4):
        frame=c.execute('SELECT * FROM parent_features WHERE fold=? ORDER BY query_id,siren',[fold]).df()
        if not set(zip(frame.query_id,frame.fold))<=allowed:raise ValueError('Foreign inference target')
        if not len(frame):continue
        values=frame[list(FEATURES)].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():raise ValueError('Nonfinite features')
        model=xgb.XGBRanker(n_jobs=2);model.load_model(verify(m['sources'][f'outer{fold}.json']))
        scores=model.predict(values)
        if not np.isfinite(scores).all():raise ValueError('Nonfinite prediction')
        frame=frame[['query_id','fold','siren']].copy();frame['parent_score']=scores
        frames.append(pa.Table.from_pandas(frame,preserve_index=False));del model,frame,values;gc.collect()
    schema=pa.schema([('query_id',pa.string()),('fold',pa.int64()),('siren',pa.string()),('parent_score',pa.float32())])
    c.register('parent_scores',pa.concat_tables(frames) if frames else pa.Table.from_pylist([],schema=schema))


def legacy(queries,output):
    v15=cfg(CONFIGS[2])['authorities'];b=cfg(CONFIGS[3])['authorities']['backends']
    vc=RetrievalV11Config.load(ROOT/'config'/CONFIGS[0])
    oc=replace(OfficialEvidenceRetrievalConfig.load(ROOT/'config'/CONFIGS[1]),
      enable_direct_parent_expansion=False,enable_consensus_parent_expansion=False,
      relation_seed_limit=vc.relation_seed_limit,search_workers=4)
    raw=[];latencies=[]
    with RetrievalSidecarReader(Path(v15['sidecars']['path'])) as side:
        side._connection.execute("SET memory_limit='2GB'");side._connection.execute('SET threads=2')
        official=OfficialEvidenceRetriever.from_index_paths(v15['base_index']['path'],
          [b['overlay_official']['path'],str(DATA/'indices/dossier_temporal_tail_v1/66f38cc146c51da6')],oc,
          temporal_overlay_index_path=b['temporal']['path'],legal_entity_role_index_path=b['roles']['path'],
          sector_official_index_path=[b['sector']['path'],str(DATA/'indices/official_sector_data_es_v1/37ea905013a83660')])
        retriever=AllHumanV11Retriever(official,side,vc)
        try:
            for qid,fold,query in queries:
                start=time.monotonic();result=retriever.retrieve(query)
                raw.extend(dict(query_id=qid,fold=fold,siren=r.siren,siret=r.siret,union_rank=r.union_rank) for r in result.candidates)
                latencies.append(dict(query_id=qid,seconds=time.monotonic()-start,candidates=len(result.candidates)))
                print(json.dumps(dict(stage='legacy',done=len(latencies),total=len(queries),seconds=latencies[-1]['seconds'])),flush=True)
        finally:official._search_pool.shutdown(wait=True)
    pq.write_table(pa.Table.from_pylist(raw,schema=RAW_SCHEMA),output/'raw.parquet',compression='zstd')
    write_json(output/'legacy_timing.json',latencies)


def shard(rows,output,m):
    output.mkdir(parents=True,exist_ok=False);start=time.monotonic();timing={}
    queries=[(r['query_id'],int(r['oof_fold']),normalize_input(r)) for r in rows]
    pq.write_table(pa.Table.from_pylist(rows).select(SAFE),output/'inputs.parquet')
    q=[];geo_q=[];address_q=[]
    for qid,fold,r in queries:
        number,suffix=split_street_number(r.number);street=normalize_street(r.address,r.number)
        q.append(dict(query_id=qid,fold=fold,crm_name=r.name,crm_number=number,crm_suffix=suffix,crm_street=street,crm_insee=r.insee,crm_postcode=r.postcode))
        geo_q.append(dict(query_id=qid,fold=fold,crm_number=r.number,crm_street_norm=street,crm_insee=r.insee,crm_postcode=r.postcode))
        address_q.append(dict(query_id=qid,fold=fold,crm_number=number,crm_suffix=suffix,crm_street_norm=street,crm_insee=r.insee,crm_postcode=r.postcode))
    pq.write_table(pa.Table.from_pylist(q),output/'queries.parquet')
    legacy(queries,output);gc.collect();timing['legacy']=time.monotonic()-start;t=time.monotonic()
    v16=cfg(CONFIGS[3]);v17=cfg(CONFIGS[5]);v18=cfg(CONFIGS[6]);b=_build_backends(v16)
    signals=generate_batch_signals(queries=queries,backends=b,specs=tuple(ParentChannelSpec.from_mapping(x) for x in v16['channels']),workers=4)
    pq.write_table(pa.Table.from_pylist(signals,schema=PARENT_SCHEMA),output/'parent_signals.parquet',compression='zstd')
    del b;gc.collect();timing['parent_search']=time.monotonic()-t;t=time.monotonic()
    paths={k:Path(r['path']) for k,r in m['sources'].items()};c=open_workbench_db(output);c.execute('SET threads=2')
    qt=pa.Table.from_pylist(geo_q);ban=_resolve_ban(c,qt,paths['ban']);pq.write_table(ban,output/'ban.parquet')
    _write_geo_neighbors(c,snapshot_path=paths['stock'],query_surface=qt,query_geo=ban,
      output_path=output/'geo_sites.parquet',parent_output_path=output/'geo_parents.parquet',neighbor_cap=512)
    # Historical geo helper derives a key; validate it against the stored official
    # parent for EVERY considered neighbor before its membership is consumed.
    if c.execute('''SELECT count(*) FROM geo_v16_nearest_sites g LEFT JOIN read_parquet(?) s USING(siret)
      WHERE s.siren IS NULL OR s.siren<>g.siren''',[str(paths['stock'])]).fetchone()[0]:raise ValueError('Geo official parent mismatch')
    _write_address_signals(c,queries=pa.Table.from_pylist(address_q),sidecar_path=paths['sites'],
      output_path=output/'address.parquet',minimum_similarity=.84,cap=512)
    text=_text_orders(signals,v16);geo=defaultdict(list);addr=defaultdict(list)
    for r in pq.read_table(output/'geo_parents.parquet').to_pylist():geo[r['query_id']].append(r['siren'])
    for r in pq.read_table(output/'address.parquet').to_pylist():addr[r['query_id']].append(r['siren'])
    parents=[];presence=[]
    for qid,fold,_ in queries:
        p,s=build_parent_rows(query_id=qid,fold=fold,source_orders=dict(TEXT=text.get(qid,()),ADDRESS=addr[qid],GEO=geo[qid]),
          variants=v17['recipe']['variants'],primary_variant=v17['recipe']['primary_variant'],cap=512)
        parents+=p;presence+=s
    pq.write_table(pa.Table.from_pylist(parents,schema=PARENTS_SCHEMA),output/'parent_candidates.parquet',compression='zstd')
    pq.write_table(pa.Table.from_pylist(presence,schema=PRESENCE_SCHEMA),output/'parent_presence.parquet',compression='zstd')
    c.execute('CREATE TABLE raw AS SELECT * FROM read_parquet(?)',[str(output/'raw.parquet')])
    select_edges(c,m['edges'])
    c.execute(f'COPY edges TO {sql_path(output/"edges.parquet")} (FORMAT PARQUET,COMPRESSION ZSTD)')
    succession(c)
    c.execute(f'COPY (SELECT * FROM v32_proof ORDER BY query_id,channel_rank) TO {sql_path(output/"proof.parquet")} (FORMAT PARQUET,COMPRESSION ZSTD)')
    v18['resources'].update(duckdb_threads=2,duckdb_memory_gib=2,search_workers=4)
    _write_structured_signals(config=v18,query_rows=tuple((a,b) for a,b,_ in queries),
      paths=dict(raw_union=output/'raw.parquet',v17_parent_presence=output/'parent_presence.parquet',
       v17_parent_candidates=output/'parent_candidates.parquet',rne_roles=paths['rne_roles'],
       bodacc_relations=paths['bodacc_relations'],temporal_glob=output/'edges.parquet'),output_path=output/'structured_intermediate.parquet')
    # The old unstable succession is discarded unconditionally before scoring.
    c.execute('CREATE TABLE stable_structured AS SELECT * FROM read_parquet(?) WHERE channel<>? UNION ALL SELECT * FROM v32_signals',
      [str(output/'structured_intermediate.parquet'),SUCCESSION])
    c.execute(f'COPY (SELECT * FROM stable_structured ORDER BY query_id,channel,channel_rank) TO {sql_path(output/"structured.parquet")} (FORMAT PARQUET,COMPRESSION ZSTD)')
    c.close();timing['geography_relations']=time.monotonic()-t;t=time.monotonic()
    _write_direct_signals(config=v18,retrieval_queries=queries,index_root=Path(v18['generation_authorities']['hierarchical_index']['path']),output_path=output/'direct.parquet')
    backend=TantivyBackend(v18['generation_authorities']['hierarchical_index']['path']);joint=[]
    for qid,fold,query in queries:joint.extend(dict(query_id=qid,fold=fold,**r) for r in search(backend,query))
    pq.write_table(pa.Table.from_pylist(joint,schema=JOINT_SCHEMA),output/'joint.parquet',compression='zstd');del backend;gc.collect()
    timing['direct_joint_search']=time.monotonic()-t
    for version in ('v23_stable','v30_stable'):
        t=time.monotonic();stage=output/version;stage.mkdir();c=open_workbench_db(stage);c.execute('SET threads=2');c.register('q',pa.Table.from_pylist(q))
        c.execute('CREATE TABLE raw AS SELECT query_id,fold,siren,siret,union_rank base_rank FROM read_parquet(?)',[str(output/'raw.parquet')])
        c.execute('CREATE TABLE signals AS SELECT query_id,fold,siren,source_siret,channel,channel_rank FROM read_parquet(?)',[[str(output/'structured.parquet'),str(output/'direct.parquet')]])
        if version=='v30_stable':c.execute('INSERT INTO signals SELECT query_id,fold,siren,siret,?,evidence_rank FROM read_parquet(?)',[JOINT,str(output/'joint.parquet')])
        c.execute('CREATE TABLE sites AS SELECT s.* FROM read_parquet(?) s SEMI JOIN (SELECT siren FROM raw UNION SELECT siren FROM signals) p USING(siren)',[str(paths['sites'])])
        build_tables(c);c.execute('DROP TABLE master');c.execute('CREATE TABLE pool AS SELECT query_id,fold,siren,siret,base_rank,parent_rrf,direct_rrf FROM candidates')
        c.execute(f'CREATE VIEW name_source AS SELECT * FROM read_parquet({sql_path(paths["names"])})');c.execute(NAME_SELECTION_SQL);build_features(c);predict(c,m);make_master(c)
        if c.execute('SELECT count(*) FROM master ANTI JOIN candidates USING(query_id,fold,siren,siret)').fetchone()[0]:raise ValueError('Injected master')
        for name,sql in [('master','SELECT * FROM master ORDER BY query_id,candidate_rank'),
          ('pool','SELECT query_id,fold,siren,siret FROM candidates ORDER BY query_id,siret'),
          ('parents','SELECT query_id,fold,siren FROM parents ORDER BY query_id,siren')]:
            c.execute(f'COPY ({sql}) TO {sql_path(stage/(name+".parquet"))} (FORMAT PARQUET,COMPRESSION ZSTD)')
        c.close();gc.collect();timing[version]=time.monotonic()-t
    write_json(output/'manifest.json',dict(status='SEALED_PAIRED_SHARD',queries=len(rows),timing=timing,
      input_scope_sha=m['scope']['sha256'],truth_columns_opened=False,quality_measured=False,
      files={str(p.relative_to(output)):record(p) for p in sorted(output.rglob('*.parquet'))}))
    print(json.dumps(dict(stage='paired_shard_complete',queries=len(rows),timing=timing)),flush=True)


def batch(prepared,output,start,count,subshard,reverse,parity=False):
    m=authority(prepared);rows=pq.read_table(verify(m['inputs'])).to_pylist()
    if parity:
        # Only a validation target, never a candidate-generation input.
        ids={r['query_id'] for r in pq.read_table(DATA/'audits/retrieval_v32_stability/prepared/queries.parquet',columns=['query_id']).to_pylist()}
        rows=[r for r in rows if r['query_id'] in ids]
        if len(rows)!=16:raise ValueError('Parity scope')
    else:
        if start<0 or count<1 or count>64 or start+count>len(rows):raise ValueError('Batch scope')
        rows=rows[start:start+count]
    if reverse:rows.reverse()
    if subshard not in (4,8,16,32,64):raise ValueError('Subshard size')
    output.mkdir(parents=True,exist_ok=False)
    with Budget(RESOURCE) as budget:
        for off in range(0,len(rows),subshard):shard(rows[off:off+subshard],output/f'shard{off:05d}',m)
        authority(prepared)
    result=dict(status='SEALED_PAIRED_BATCH',prepared=record(prepared/'manifest.json'),code=code(),
      start=start,count=len(rows),subshard=subshard,reverse=reverse,parity=parity,resources=budget.report(),
      quality_measured=False,truth_columns_opened=False,shards=[record(p) for p in sorted(output.glob('shard*/manifest.json'))])
    write_json(output/'manifest.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('code','shards')}),flush=True)


def manifests(path):
    m=json.loads((path/'manifest.json').read_text())
    if m['status']!='SEALED_PAIRED_BATCH' or m['code']!=code():raise ValueError('Batch drift')
    shards=[json.loads(verify(r).read_text()) for r in m['shards']]
    return m,shards


def files_for(shards,name):return [str(verify(s['files'][name])) for s in shards]


def same(c,left,right):
    a='SELECT * FROM read_parquet(?)';b='SELECT * FROM read_parquet(?)'
    n=c.execute(f'SELECT count(*) FROM (({a} EXCEPT ALL {b}) UNION ALL ({b} EXCEPT ALL {a}))',[left,right,right,left]).fetchone()[0]
    if n:raise ValueError(f'Nonreproducible: {n} different rows')


def check(prepared,output,parity,pilot,repeated):
    m=authority(prepared);pm,ps=manifests(parity);am,aa=manifests(pilot);bm,bb=manifests(repeated)
    if not pm['parity'] or am['parity'] or bm['parity'] or am['start']!=0 or bm['start']!=0 or am['count']!=32 or bm['count']!=32:
        raise ValueError('Expected parity16 and fresh pilot32')
    if any(x['prepared']!=record(prepared/'manifest.json') for x in (pm,am,bm)):raise ValueError('Source drift')
    c=duckdb.connect();c.execute("SET memory_limit='2GB'")
    for name in ('raw.parquet','proof.parquet','structured.parquet','direct.parquet','joint.parquet',
      'v23_stable/master.parquet','v30_stable/master.parquet','v23_stable/pool.parquet','v30_stable/pool.parquet'):
        same(c,files_for(aa,name),files_for(bb,name))
    for version in ('v23_stable','v30_stable'):
        same(c,files_for(ps,version+'/master.parquet'),str(DATA/'audits/retrieval_v32_stability/run_a'/(version+'.parquet')))
    elapsed=am['resources']['wall_seconds'];size=sum(Path(p['path']).stat().st_size for s in aa for p in s['files'].values())
    projected_seconds=elapsed/32*23571;projected_bytes=size/32*23571
    # Conservative stop thresholds, not promises or product SLOs.
    cost_ok=projected_seconds<=36*3600 and projected_bytes<=100*1024**3 and shutil.disk_usage(DATA).free>=max(50*1024**3,2*projected_bytes)
    output.mkdir(parents=True,exist_ok=False)
    result=dict(status='PASS_READY_FOR_FULL_REPLAY' if cost_ok else 'STOP_COST_PROJECTION',
      prepared=record(prepared/'manifest.json'),batches=[record(p/'manifest.json') for p in (parity,pilot,repeated)],
      code=code(),technical_queries=32,parity_queries=16,quality_measured=False,truth_columns_opened=False,
      cost=dict(pilot_seconds=elapsed,linear_projection_hours=projected_seconds/3600,linear_projection_gib=projected_bytes/1024**3,
        uncertainty='SMALL_PILOT_NOT_A_DURATION_GUARANTEE',maximum_full_hours=36,maximum_output_gib=100))
    write_json(output/'manifest.json',result);print(json.dumps({k:v for k,v in result.items() if k!='code'}),flush=True)


def full(prepared,output,gate):
    m=authority(prepared);g=json.loads((gate/'manifest.json').read_text())
    if g['status']!='PASS_READY_FOR_FULL_REPLAY' or g['prepared']!=record(prepared/'manifest.json') or g['code']!=code():raise ValueError('Full gate absent/drifted')
    output.mkdir(parents=True,exist_ok=False);write_json(output/'RUNNING.json',dict(pid=os.getpid(),gate=record(gate/'manifest.json'),queries=23571))
    start_time=time.monotonic();done=[]
    try:
        for start in range(0,23571,32):
            if time.monotonic()-start_time>36*3600 or shutil.disk_usage(DATA).free<30*1024**3:raise RuntimeError('STOP_FULL_BUDGET')
            stage=output/f'batch{start:05d}';cmd=[sys.executable,str(Path(__file__).resolve()),'batch','--prepared',str(prepared),
              '--output',str(stage),'--start',str(start),'--count',str(min(32,23571-start)),'--subshard','32']
            with (output/f'batch{start:05d}.log').open('x') as log:
                subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1900)
            done.append(record(stage/'manifest.json'))
            write_json(output/'progress.json',dict(completed_queries=min(start+32,23571),total_queries=23571,seconds=time.monotonic()-start_time))
        authority(prepared)
        write_json(output/'manifest.json',dict(status='SEALED_FULL_PAIRED_GENERATION',queries=23571,prepared=record(prepared/'manifest.json'),
          gate=record(gate/'manifest.json'),batches=done,code=code(),quality_measured=False,truth_columns_opened=False,
          seconds=time.monotonic()-start_time,evaluation_pending=True))
    except BaseException as exc:
        write_json(output/'STOP.json',dict(status='STOP',error=repr(exc),completed_batches=len(done),quality_measured=False));raise


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','batch','check','full']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prepared',type=Path);p.add_argument('--start',type=int,default=0);p.add_argument('--count',type=int,default=32)
    p.add_argument('--subshard',type=int,default=32);p.add_argument('--reverse',action='store_true');p.add_argument('--parity',action='store_true')
    p.add_argument('--parity-run',type=Path);p.add_argument('--pilot',type=Path);p.add_argument('--repeated',type=Path);p.add_argument('--gate',type=Path);a=p.parse_args()
    if a.phase=='prepare':
        a.output.mkdir(parents=True,exist_ok=False)
        with Budget(RESOURCE) as b:r=prepare(a.output)
        r['resources']=b.report();write_json(a.output/'manifest.json',r);print(r['status'],flush=True)
    elif a.phase=='batch':batch(a.prepared,a.output,a.start,a.count,a.subshard,a.reverse,a.parity)
    elif a.phase=='check':check(a.prepared,a.output,a.parity_run,a.pilot,a.repeated)
    else:full(a.prepared,a.output,a.gate)


if __name__=='__main__':main()
