#!/usr/bin/env python3
"""Generate a proof-preserving 1750+250 SIRET Master, never opening labels."""
from pathlib import Path
import argparse
import json
import sys

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.xgb_matcher.retrieval_v20_workbench import Budget, load_config, verify, record, write_json, sql_path, implementation
from src.xgb_matcher.retrieval_parent_discovery_v18 import load_query_selection
from src.xgb_matcher.retrieval_site_inventory_v17 import split_street_number
from src.xgb_matcher.retrieval_geo_v16 import normalize_street
from src.xgb_matcher.retrieval_v12_scene_ir import SceneIRCache


def build_tables(c):
    """Q, RAW, SIGNALS and SITES are explicit truth-free Arrow/SQL surfaces."""
    if c.execute('SELECT count(*) FROM (SELECT query_id,siret,count(*) n FROM raw GROUP BY ALL HAVING n>1)').fetchone()[0]:
        raise ValueError('Duplicate raw SIRET')
    if c.execute('SELECT count(*) FROM raw WHERE base_rank<1 OR base_rank>2000 OR fold NOT IN (2,3,4)').fetchone()[0]:
        raise ValueError('Forbidden raw ranks/folds')
    c.execute('''CREATE TEMP TABLE parents AS
        SELECT *,row_number() OVER(PARTITION BY query_id ORDER BY parent_rrf DESC,siren) parent_rank
        FROM (SELECT query_id,fold,siren,sum(1.0/(60+channel_rank)) parent_rrf
          FROM signals WHERE channel<>'v17_union' GROUP BY query_id,fold,siren)''')
    # Site witnesses survive parent aggregation. Each channel contributes once
    # to one candidate, not once for every sibling in the parent.
    c.execute('''CREATE TEMP TABLE witnesses AS
        SELECT query_id,fold,siren,siret,'legacy_site_pool' channel,base_rank channel_rank FROM raw
        UNION ALL
        SELECT query_id,fold,siren,source_siret siret,channel,channel_rank
        FROM signals WHERE length(source_siret)=14 AND channel NOT IN ('legacy_site_pool','v17_union')''')
    conflicts = c.execute('''SELECT count(*) FROM witnesses w JOIN sites s USING(siret) WHERE w.siren<>s.siren''').fetchone()[0]
    if conflicts:
        raise ValueError('Source SIRET/official parent mismatch')
    c.execute('''CREATE TEMP TABLE direct AS
        SELECT query_id,fold,siren,siret,sum(1.0/(60+r)) direct_rrf,count(*) direct_channels
        FROM (SELECT query_id,fold,siren,siret,channel,min(channel_rank) r FROM witnesses GROUP BY ALL)
        GROUP BY query_id,fold,siren,siret''')
    # All geographically compatible siblings of all evidence-supported parents,
    # not a fixed number of siblings. Geography is an expansion route only;
    # direct witnesses outside it remain in the candidate set.
    c.execute('''CREATE TEMP TABLE address_sites AS
        SELECT *, row_number() OVER(PARTITION BY query_id,siren ORDER BY
          (number_exact AND suffix_ok AND street_similarity=1) DESC,
          same_insee DESC,number_exact DESC,suffix_ok DESC,street_similarity DESC,
          (administrative_state='A') DESC,siret) site_address_rank
        FROM (
          SELECT p.query_id,p.fold,p.siren,s.siret,s.administrative_state,
            (q.crm_insee<>'' AND q.crm_insee=s.insee) same_insee,
            (q.crm_number<>'' AND q.crm_number=s.street_number) number_exact,
            (q.crm_suffix='' OR q.crm_suffix=upper(coalesce(s.street_number_suffix,''))) suffix_ok,
            CASE WHEN q.crm_street<>'' AND coalesce(s.street_normalized,'')<>''
              THEN jaro_winkler_similarity(q.crm_street,s.street_normalized) ELSE 0 END street_similarity
          FROM parents p JOIN q USING(query_id,fold) JOIN sites s ON p.siren=s.siren
          AND ((q.crm_insee<>'' AND q.crm_insee=s.insee) OR (q.crm_postcode<>'' AND q.crm_postcode=s.postcode))
        )''')
    c.execute('''CREATE TEMP TABLE candidates AS
        SELECT u.*,r.base_rank,coalesce(d.direct_rrf,0) direct_rrf,
          coalesce(p.parent_rrf,0) parent_rrf,p.parent_rank,a.site_address_rank,
          coalesce(d.direct_channels,0) direct_channels,
          coalesce(a.number_exact AND a.suffix_ok AND a.street_similarity=1,false) joint_address,
          coalesce(d.direct_rrf,0) + CASE WHEN a.site_address_rank IS NOT NULL THEN
          1.0/(60+p.parent_rank+a.site_address_rank) ELSE 0 END rescue_score
        FROM (SELECT query_id,fold,siren,siret FROM raw UNION
          SELECT query_id,fold,siren,siret FROM direct UNION
          SELECT query_id,fold,siren,siret FROM address_sites) u
        LEFT JOIN raw r USING(query_id,fold,siren,siret)
        LEFT JOIN direct d USING(query_id,fold,siren,siret)
        LEFT JOIN parents p USING(query_id,fold,siren)
        LEFT JOIN address_sites a USING(query_id,fold,siren,siret)''')
    c.execute('''CREATE TEMP TABLE master AS
        WITH counts AS (SELECT query_id,count(*) n FROM candidates WHERE base_rank<=1750 GROUP BY query_id),
        rescue AS (SELECT c.*,row_number() OVER(PARTITION BY query_id ORDER BY
          rescue_score DESC,direct_rrf DESC,parent_rrf DESC,siret) rescue_rank
          FROM candidates c WHERE base_rank IS NULL OR base_rank>1750),
        selected AS (
          SELECT c.*,base_rank candidate_rank,'PRESERVED_BASE1750' selection_reason FROM candidates c WHERE base_rank<=1750
          UNION ALL
          SELECT r.* EXCLUDE(rescue_rank),coalesce(n,0)+rescue_rank candidate_rank,'DIRECT_OR_SIBLING_RESCUE' selection_reason
          FROM rescue r LEFT JOIN counts USING(query_id) WHERE rescue_rank<=2000-coalesce(n,0)
        ) SELECT * FROM selected''')
    if c.execute('SELECT count(*) FROM (SELECT query_id,count(*) n,count(DISTINCT siret) nd,max(candidate_rank) mx FROM master GROUP BY query_id HAVING n<>nd OR n>2000 OR n<>mx)').fetchone()[0]:
        raise ValueError('Master cap, uniqueness or ranks')
    if c.execute('''SELECT count(*) FROM raw r ANTI JOIN master m USING(query_id,siret) WHERE r.base_rank<=1750''').fetchone()[0]:
        raise ValueError('Baseline preservation failed')


def run(config_path, output, limit=None):
    config,sources=load_config(config_path)
    code = implementation(__file__)
    if config['master']['cap_sirets']!=2000 or config['master']['protected_baseline_prefix']!=1750 or config['master']['rrf_k']!=60:
        raise ValueError('Frozen caps drifted')
    a=sources['v18']['generation_authorities']
    inputs={key:verify(a[key]) for key in ('queries','raw_union','raw_union_manifest')}
    sitepath=verify(config['sites'])
    va=sources['v19']['authorities']
    signals=[verify(va[key]) for key in ('structured_signals','siret_native_signals')]
    scene_root=Path(a['scene_ir']['path'])
    verify({'path':str(scene_root/'manifest.json'),'sha256':a['scene_ir']['manifest_sha256']})
    queries=load_query_selection(inputs['queries'],expected_count=1034)
    if limit is not None:
        if not 1<=limit<=16:raise ValueError('Preflight 1..16')
        queries=queries[:limit]
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    with Budget(config['resources']) as budget:
        c=duckdb.connect();c.execute("SET memory_limit='4GB'");c.execute('SET threads=4')
        query_rows=[];raw_tables=[]
        # Explicit projection: neither qualifications nor target fields are read.
        cols=['query_id','fold','union_rank','siret','siren','crm_number','crm_address','crm_postcode','crm_insee','postcode','insee']
        cache=SceneIRCache(scene_root,verify_source_hash=False)
        if cache.source.resolve()!=inputs['raw_union'].resolve():raise ValueError('SceneIR source path drift')
        for scene in cache.iter_selected_scenes(dict(queries),columns=cols):
            if scene.fold!=dict(queries)[scene.query_id]:raise ValueError('SceneIR fold drift')
            first=scene.rows.slice(0,1).to_pylist()[0]
            num,suffix=split_street_number(first['crm_number'])
            query_rows.append(dict(query_id=scene.query_id,fold=scene.fold,crm_number=num,crm_suffix=suffix,
                crm_street=normalize_street(first['crm_address'] or '',first['crm_number'] or ''),
                crm_postcode=str(first['crm_postcode'] or first['postcode'] or ''),crm_insee=str(first['crm_insee'] or first['insee'] or '')))
            raw_tables.append(scene.rows.select(['query_id','fold','siret','siren','union_rank']).rename_columns(['query_id','fold','siret','siren','base_rank']))
        if len(query_rows)!=len(queries):raise ValueError('Query surface incomplete')
        qt=pa.Table.from_pylist(query_rows);rt=pa.concat_tables(raw_tables)
        pq.write_table(qt,output/'queries.parquet',compression='zstd')
        c.register('q',qt);c.register('raw',rt)
        c.execute('CREATE TEMP TABLE signals AS SELECT s.* FROM read_parquet(?) s JOIN q USING(query_id,fold)',[[str(p) for p in signals]])
        # Scan official metadata once. Keep only relevant parents and direct sites.
        c.execute('''CREATE TEMP TABLE sites AS SELECT s.* FROM read_parquet(?) s
          WHERE s.siren IN (SELECT siren FROM signals UNION SELECT siren FROM raw)''',[str(sitepath)])
        build_tables(c)
        names={'master.parquet':"SELECT * FROM master ORDER BY fold,query_id,candidate_rank",
            'witnesses.parquet':"SELECT * FROM witnesses ORDER BY fold,query_id,siret,channel",
            'candidates.parquet':"SELECT * FROM candidates ORDER BY fold,query_id,siret"}
        for name,sql in names.items():
            c.execute(f"COPY ({sql}) TO {sql_path(output/name)} (FORMAT PARQUET,COMPRESSION ZSTD,COMPRESSION_LEVEL 3,ROW_GROUP_SIZE 65536)")
        counts={name:c.execute(f'SELECT count(*) FROM {name}').fetchone()[0] for name in ('parents','direct','address_sites','candidates','master')}
        c.close()
    files={p.name:record(p) for p in output.iterdir() if p.is_file()}
    m={'schema':'sireto-v20-master-generation-1','config':record(config_path),'inputs':{str(p):record(p) for p in [*inputs.values(),sitepath,*signals,scene_root/'manifest.json']},'files':files,'query_count':len(queries),'preflight':limit is not None,'counts':counts,'resources':budget.report(),
       'truth_columns_opened':False,'positive_injection':False,'fold0_opened':False,'fold1_opened':False,'all_base1750_preserved':True,'siret_cap':2000,'v17_identifier_sorted_rank_used':False,'new_model_trained':False}
    m['implementation'] = code
    write_json(output/'manifest.json',m);print(json.dumps(m),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--preflight',type=int);a=p.parse_args();run(a.config,a.output,a.preflight)
