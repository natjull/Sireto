"""One frozen joint lexical discovery channel, then frozen v23 Master inference."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.xgb_matcher.hierarchical_retrieval import TantivyBackend,character_ngrams
from src.xgb_matcher.retrieval_v20_workbench import ROOT,Budget,record,verify,write_json,sql_path
from src.xgb_matcher.retrieval_v21_membership import manifest_files,open_workbench_db,build_tables
from src.xgb_matcher.retrieval_v23_joint_parent import FEATURES,NAME_SELECTION_SQL,build_features,make_master,code_records
from src.xgb_matcher.retrieval_v25_conjunctive import index_inventory
from src.xgb_matcher.retrieval_parent_fusion_v19 import validate_fold_isolation
from scripts.run_retrieval_parent_discovery_v18 import _load_retrieval_queries
from scripts.retrieval_v29_consolidated import predict,equal_sql,check_q,versions as base_versions
from scripts.retrieval_v22_global_budget import outcomes,summary

CHANNEL='siret_joint_name_address_char'
SIGNALS=pa.schema([('query_id',pa.string()),('fold',pa.int8()),('siren',pa.string()),('siret',pa.string()),
                   ('score',pa.float64()),('candidate_rank',pa.int32()),('evidence_rank',pa.int32())])


def config_at(path):
    c=json.loads(Path(path).read_text())
    if (c['schema']!='sireto-v30-joint-search-1' or c['recipe']!='CURRENT_NAME_CHAR_AND_CURRENT_ADDRESS_CHAR_MEAN_BM25_THEN_V23'
        or c['folds']!=[2,3,4] or (c['channel_cap'],c['master_cap'],c['workers'])!=(512,2000,4)
        or any(c[k] for k in ('require_exact_number','retrain','national_run'))):
        raise ValueError('Frozen recipe drift')
    return c


def implementation():
    paths=[Path(__file__),ROOT/'scripts/retrieval_v29_consolidated.py',ROOT/'src/xgb_matcher/retrieval_v25_conjunctive.py',
           ROOT/'scripts/run_retrieval_parent_discovery_v18.py',ROOT/'docs/retrieval_v30_joint_search_contract.md']
    return {**code_records(),**{str(p.relative_to(ROOT)):record(p) for p in paths}}


def versions():
    return {**base_versions(),'tantivy':importlib.metadata.version('tantivy')}


def save(output,config,result):
    result.update(config=record(config),implementation=implementation(),versions=versions(),
                  fold0_opened=False,fold1_opened=False,production_activation=False)
    result['files']={p.name:record(p) for p in output.iterdir() if p.is_file()}
    write_json(output/'manifest.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('files','implementation','segments','index_inventory')}),flush=True)


def terms(text):
    return sorted({g.split('_',1)[1].lower() for g in character_ngrams(text) if '_' in g})


def joint_query(backend,query):
    tv=backend._tantivy;groups=[]
    for field,text in [('current_name_ngrams',query.name),('address_ngrams',query.address)]:
        words=terms(text)
        if not words:return None
        group=tv.Query.boolean_query([(tv.Occur.Should,tv.Query.term_query(backend.index.schema,field,w,'basic')) for w in words])
        groups.append((tv.Occur.Must,tv.Query.boost_query(group,1.0/len(words))))
    return backend._combined_query(query,tv.Query.boolean_query(groups),document_type='siret')


def search(backend,query,cap=512):
    if not 1<=cap<=512:raise ValueError('Channel cap')
    parsed=joint_query(backend,query)
    if parsed is None:return []
    rows=[]
    for score,location in backend.searcher.search(parsed,limit=cap,count=False).hits:
        doc=backend.searcher.doc(location).to_dict()
        siren=backend._first(doc,'siren');siret=backend._first(doc,'siret')
        if not isinstance(siren,str) or not siren.isdigit() or len(siren)!=9 or not isinstance(siret,str) or not siret.isdigit() or len(siret)!=14:
            raise ValueError('Invalid stored official identity')
        if not math.isfinite(score):raise ValueError('Nonfinite search score')
        rows.append(dict(siren=siren,siret=siret,score=float(score)))
    rows.sort(key=lambda r:(-r['score'],r['siret']))
    if len({r['siret'] for r in rows})!=len(rows):raise ValueError('Duplicate indexed site')
    dense=0;previous=None
    for pos,row in enumerate(rows,1):
        if row['score']!=previous:dense+=1
        row.update(candidate_rank=pos,evidence_rank=dense);previous=row['score']
    return rows


def discover(config,output,preflight=None):
    cfg=config_at(config);v23=json.loads(verify(cfg['v23_config']).read_text())
    source=json.loads(verify(v23['v18_config']).read_text())['generation_authorities']
    querypath=verify(source['queries']);raw=verify(source['raw_union'])
    index=Path(source['hierarchical_index']['path']);scene=Path(source['scene_ir']['path'])
    authorities=[record(querypath),record(raw)]
    for key,root in [('hierarchical_index',index),('scene_ir',scene)]:
        authorities.append(record(verify(dict(path=str(root/'manifest.json'),sha256=source[key]['manifest_sha256']))))
    if preflight is not None and not 1<=preflight<=16:raise ValueError('Preflight1..16')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    inventory=index_inventory(index/'tantivy');meta=record(index/'tantivy/meta.json')
    with Budget(cfg['resources']) as budget:
        ids=pq.read_table(querypath).to_pylist()
        if len(ids)!=1034 or len({r['query_id'] for r in ids})!=1034 or {r['fold'] for r in ids}!={2,3,4}:
            raise ValueError('Query authority')
        ids=sorted(ids,key=lambda r:r['query_id'])[:preflight or 1034]
        queries=_load_retrieval_queries(tuple((r['query_id'],r['fold']) for r in ids),scene)
        backend=TantivyBackend(index)
        if not backend.manifest.get('typed_name_portfolio'):raise ValueError('Typed current names required')
        for _,_,q in queries:
            if q.insee:backend._has_insee(q.insee)
        def one(item):
            qid,fold,q=item;start=time.monotonic();rows=search(backend,q)
            return [dict(query_id=qid,fold=fold,**r) for r in rows],dict(query_id=qid,fold=fold,seconds=time.monotonic()-start)
        allrows=[];timings=[]
        with ThreadPoolExecutor(max_workers=4) as pool:
            for i,(rows,timing) in enumerate(pool.map(one,queries),1):
                allrows.extend(rows);timings.append(timing)
                if i%100==0:print(json.dumps(dict(processed=i,total=len(queries))),flush=True)
        qrows=[dict(query_id=qid,fold=fold,**asdict(q)) for qid,fold,q in queries]
        pq.write_table(pa.Table.from_pylist(qrows),output/'queries.parquet',compression='zstd')
        pq.write_table(pa.Table.from_pylist(allrows,schema=SIGNALS),output/'joint_signals.parquet',compression='zstd')
        pq.write_table(pa.Table.from_pylist(timings),output/'timings.parquet',compression='zstd')
        if inventory!=index_inventory(index/'tantivy') or meta!=record(index/'tantivy/meta.json'):
            raise ValueError('Index changed')
    save(output,config,dict(schema='sireto-v30-discovery-1',query_count=len(queries),signal_count=len(allrows),
         eligible_queries=sum(bool(terms(q.name) and terms(q.address)) for _,_,q in queries),preflight=preflight is not None,
         sources=authorities,index_inventory=inventory,index_meta=meta,index_full_bytes_rehashed=False,
         truth_columns_opened=False,positive_injection=False,model_fit_performed=False,resources=budget.report()))


def generate(config,discovery,output,preflight=None):
    cfg=config_at(config);dm,df=manifest_files(record(Path(discovery)/'manifest.json'))
    if (dm['schema']!='sireto-v30-discovery-1' or dm['truth_columns_opened'] or dm['config']!=record(config)
        or dm['implementation']!=implementation() or dm['versions']!=versions() or dm['preflight']!=(preflight is not None)):
        raise ValueError('Discovery authority')
    if preflight is not None and not 1<=preflight<=16:raise ValueError('Preflight1..16')
    v23=json.loads(verify(cfg['v23_config']).read_text());v19=json.loads(verify(cfg['v19_config']).read_text())
    gm,gf=manifest_files(v23['source_generation']);fm,ff=manifest_files(cfg['features']);mm,mf=manifest_files(cfg['models'])
    if (fm['implementation']!=code_records() or fm['feature_order']!=list(FEATURES) or fm['preflight'] or gm['preflight']
        or mm['features_manifest']['sha256']!=cfg['features']['sha256'] or mm['target_labels_used_for_selection']):
        raise ValueError('Baseline lineage')
    sources={k:verify(v19['authorities'][k]) for k in ('structured_signals','siret_native_signals')}
    sources.update(names=verify(v23['names']),sites=verify(v23['sites']))
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    with Budget(cfg['resources']) as budget:
        c=open_workbench_db(output);c.execute("SET memory_limit='8GB'");c.execute('SET threads=2')
        c.execute('CREATE TEMP TABLE q AS SELECT * FROM read_parquet(?) ORDER BY query_id LIMIT ?',[str(ff['queries.parquet']),preflight or 1034])
        check_q(c,preflight or 1034)
        equal_sql(c,'SELECT query_id,fold FROM q',f'SELECT query_id,fold FROM read_parquet({sql_path(df["queries.parquet"])})')
        c.execute('CREATE TEMP TABLE joint AS SELECT * FROM read_parquet(?)',[str(df['joint_signals.parquet'])])
        if c.execute('SELECT count(*) FROM joint ANTI JOIN q USING(query_id,fold)').fetchone()[0]:raise ValueError('Foreign signal')
        if c.execute('SELECT query_id FROM joint GROUP BY query_id HAVING count(*)>512 OR count(*)<>count(DISTINCT siret)').fetchall():raise ValueError('Signal cap/uniqueness')
        c.execute('''CREATE TEMP TABLE raw AS SELECT p.query_id,p.fold,p.siren,p.siret,p.base_rank FROM read_parquet(?) p
          SEMI JOIN q USING(query_id,fold) WHERE base_rank IS NOT NULL''',[str(gf['candidates.parquet'])])
        c.execute('''CREATE TEMP TABLE signals AS SELECT s.query_id,s.fold,s.siren,s.source_siret,s.channel,s.channel_rank
          FROM read_parquet(?) s SEMI JOIN q USING(query_id,fold)''',[[str(sources[k]) for k in ('structured_signals','siret_native_signals')]])
        if c.execute('SELECT count(*) FROM signals WHERE channel=?',[CHANNEL]).fetchone()[0]:raise ValueError('Channel already present')
        c.execute('INSERT INTO signals SELECT query_id,fold,siren,siret,?,evidence_rank FROM joint',[CHANNEL])
        c.execute('''CREATE TEMP TABLE sites AS SELECT s.siren,s.siret,s.insee,s.postcode,s.street_number,s.street_number_suffix,
          s.street_normalized,s.administrative_state FROM read_parquet(?) s
          SEMI JOIN (SELECT siren FROM signals UNION SELECT siren FROM raw) p USING(siren)''',[str(sources['sites'])])
        build_tables(c);c.execute('DROP TABLE master')
        if c.execute('''SELECT count(*) FROM read_parquet(?) p SEMI JOIN q USING(query_id,fold)
          ANTI JOIN candidates USING(query_id,fold,siren,siret)''',[str(gf['candidates.parquet'])]).fetchone()[0]:raise ValueError('Nonadditive reservoir')
        # Replay frozen baseline inference on its own feature snapshot, not a second recipe.
        c.execute('CREATE TEMP TABLE parent_features AS SELECT p.* FROM read_parquet(?) p SEMI JOIN q USING(query_id,fold)',[str(ff['parent_features.parquet'])])
        c.execute('CREATE TEMP TABLE site_features AS SELECT p.* FROM read_parquet(?) p SEMI JOIN q USING(query_id,fold)',[str(ff['site_features.parquet'])])
        predict(c,mf,reference=True);make_master(c)
        equal_sql(c,'SELECT * FROM master',f'SELECT m.* FROM read_parquet({sql_path(mf["master.parquet"])}) m SEMI JOIN q USING(query_id,fold)')
        print('PASS_V23_FROZEN_SCORES_MASTER_REPLAY',flush=True)
        for t in ('master','parent_features','site_features','ranked_parents'):c.execute(f'DROP TABLE {t}')
        c.unregister('parent_scores')
        c.execute('CREATE TEMP TABLE pool AS SELECT query_id,fold,siren,siret,base_rank,parent_rrf,direct_rrf FROM candidates')
        c.execute(f'CREATE TEMP VIEW name_source AS SELECT * FROM read_parquet({sql_path(sources["names"])})')
        c.execute(NAME_SELECTION_SQL);build_features(c);predict(c,mf,output=output);make_master(c)
        tables=dict(queries='q',candidates='candidates',parents='parents',witnesses='witnesses',parent_features='parent_features',site_features='site_features',master='master')
        for name,table in tables.items():
            c.execute(f'COPY (SELECT * FROM {table} ORDER BY query_id) TO {sql_path(output/(name+".parquet"))} (FORMAT PARQUET,COMPRESSION ZSTD)')
        counts={k:c.execute(f'SELECT count(*) FROM {v}').fetchone()[0] for k,v in tables.items()};c.close()
    save(output,config,dict(schema='sireto-v30-generation-1',discovery_manifest=record(Path(discovery)/'manifest.json'),
         sources={k:record(p) for k,p in sources.items()},counts=counts,preflight=preflight is not None,
         truth_columns_opened=False,positive_injection=False,model_fit_performed=False,site_model_scoring_performed=False,
         baseline_scores_master_replayed=True,rich_hydration=False,resources=budget.report()))


def summarize(df):
    r=summary(df);r['gains_vs_v23']=r.pop('gains_vs_v21');r['losses_vs_v23']=r.pop('losses_vs_v21')
    r['net_vs_v23']=r['gains_vs_v23']-r['losses_vs_v23']
    for col in ('old_reservoir_hit','site_in_reservoir','channel_exact','channel_parent'):r[col]=int(df[col].sum())
    r['new_exact_channel']=int((df.channel_exact & ~df.old_reservoir_hit).sum())
    r['new_exact_expanded_reservoir']=int((df.site_in_reservoir & ~df.old_reservoir_hit).sum())
    return r


def evaluate(config,generation,output):
    cfg=config_at(config);gm,gf=manifest_files(record(Path(generation)/'manifest.json'))
    if (gm['schema']!='sireto-v30-generation-1' or gm['preflight'] or gm['truth_columns_opened'] or gm['counts']['queries']!=1034
        or gm['config']!=record(config) or gm['implementation']!=implementation() or gm['versions']!=versions()):raise ValueError('Generation authority')
    dm,df=manifest_files(gm['discovery_manifest']);bm,bf=manifest_files(cfg['baseline'])
    side=verify(bm['official_loss_sidecar']);acceptable=verify(bm['acceptable_operational_sidecar'])
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    with Budget(cfg['resources']) as budget:
        c=open_workbench_db(output);c.execute("SET memory_limit='8GB'");c.execute('SET threads=2')
        for name,path in [('q',gf['queries.parquet']),('master',gf['master.parquet']),('pool',gf['candidates.parquet']),('parents',gf['parents.parquet']),
                          ('joint',df['joint_signals.parquet']),('catalog',side),('acceptable',acceptable)]:
            c.execute(f'CREATE TEMP TABLE {name} AS SELECT * FROM read_parquet(?)',[str(path)])
        check_q(c,1034)
        c.execute('''CREATE TEMP TABLE baseline AS SELECT query_id,fold,gt_siret,gt_siren,cohort_kind,qualification_v2,qualification_v3,
          label_kind,data_origin,ground_truth_state,site_in_reservoir old_reservoir_hit,exact_hit previous_exact_hit,
          operational_hit previous_operational_hit FROM read_parquet(?) WHERE fold IN (2,3,4)''',[str(bf['outcomes.parquet'])])
        validate_fold_isolation(c.execute('SELECT query_id,fold,gt_siren FROM baseline').df())
        if c.execute('SELECT count(*) FROM baseline ANTI JOIN catalog USING(query_id)').fetchone()[0]:raise ValueError('Missing official sidecar')
        if c.execute('SELECT query_id FROM catalog GROUP BY query_id HAVING count(*)<>1').fetchall():raise ValueError('Duplicate official sidecar')
        if c.execute('''SELECT count(*) FROM baseline t JOIN catalog cat USING(query_id) ANTI JOIN acceptable a
          ON a.query_id=t.query_id AND a.siret=t.gt_siret WHERE cat.official_catalog_present OR
          EXISTS(SELECT 1 FROM pool p WHERE p.query_id=t.query_id AND p.siret=t.gt_siret)''').fetchone()[0]:raise ValueError('Missing operational exact')
        c.execute('''CREATE TEMP TABLE t AS SELECT b.*,
          EXISTS(SELECT 1 FROM pool p WHERE p.query_id=b.query_id AND p.siret=b.gt_siret) site_in_reservoir,
          EXISTS(SELECT 1 FROM joint p WHERE p.query_id=b.query_id AND p.siret=b.gt_siret) channel_exact,
          EXISTS(SELECT 1 FROM joint p WHERE p.query_id=b.query_id AND p.siren=b.gt_siren) channel_parent FROM baseline b''')
        c.execute('''CREATE TEMP TABLE official_loss AS SELECT cat.* EXCLUDE(parent_in_evidence),
          EXISTS(SELECT 1 FROM parents p WHERE p.query_id=cat.query_id AND p.siren=t.gt_siren) parent_in_evidence
          FROM catalog cat JOIN t USING(query_id)''')
        out=outcomes(c);assert len(out)==1034
        cap=c.execute('SELECT max(n) FROM (SELECT query_id,count(*) n FROM master GROUP BY 1)').fetchone()[0];assert cap<=2000
        for name in ('outcomes','official_loss'):
            c.execute(f'COPY {name} TO {sql_path(output/(name+".parquet"))} (FORMAT PARQUET,COMPRESSION ZSTD)')
        segments={dim:{str(k):summarize(v) for k,v in out.groupby(dim,dropna=False)} for dim in
                  ('fold','cohort_kind','data_origin','ground_truth_state','label_kind','qualification_v2','qualification_v3')}
        primary=summarize(out);c.close()
    save(output,config,dict(schema='sireto-v30-evaluation-1',generation_manifest=record(Path(generation)/'manifest.json'),baseline=cfg['baseline'],
         official_loss_sidecar=record(side),operational_sidecar=record(acceptable),primary=primary,historical_all_human=primary,segments=segments,
         loss_counts={str(k):int(v) for k,v in out.loss_stage.value_counts().items()},max_master_sirets=int(cap),
         experiment_status='NET_GAIN_REUSED_PANEL' if primary['net_vs_v23']>0 else 'NO_NET_GAIN_REUSED_PANEL',
         decision='PIVOT_GENERATION',ready_for_fold1=False,recall100_measured=False,prerank500_measured=False,
         national_projection=None,reused_development_panel=True,resources=budget.report()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['discover','generate','evaluate'])
    p.add_argument('--config',type=Path,default=Path('config/retrieval_v30_joint_search.json'));p.add_argument('--output',type=Path,required=True)
    p.add_argument('--discovery',type=Path);p.add_argument('--generation',type=Path);p.add_argument('--preflight',type=int)
    a=p.parse_args()
    if a.action=='discover':discover(a.config,a.output,a.preflight)
    elif a.action=='generate':
        if not a.discovery:p.error('Discovery required')
        generate(a.config,a.discovery,a.output,a.preflight)
    else:
        if not a.generation or a.preflight:p.error('Full generation required')
        evaluate(a.config,a.generation,a.output)
