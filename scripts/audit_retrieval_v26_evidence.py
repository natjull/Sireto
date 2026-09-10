#!/usr/bin/env python3
"""Two sealed truth-side audits; no generation and no raw BAN coordinate export."""
import argparse
import json
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.xgb_matcher.retrieval_v20_workbench import ROOT,Budget,record,verify,write_json,sql_path
from src.xgb_matcher.retrieval_v21_membership import manifest_files,open_workbench_db
from src.xgb_matcher.retrieval_v25_conjunctive import index_inventory
from src.xgb_matcher.hierarchical_retrieval import TantivyBackend,RetrievalQuery,normalize_text
from scripts.run_retrieval_geo_v16_workbench import _query_surface,_resolve_ban


FIELDS={'LEGAL_CURRENT':('legal_current_names','legal_names'),
        'TRADE_CURRENT':('trade_current_names','trade_names'),
        'SITE_CURRENT':('site_current_names','site_names'),
        'HISTORICAL':('historical_names','historical_names')}


def load(path):
    cfg=json.loads(Path(path).read_text())
    if (cfg['schema']!='sireto-v26-evidence-audit-1' or cfg['folds']!=[2,3,4]
        or cfg['distance_bins_m']!=[50,100,250,500] or cfg['name_roles']!=list(FIELDS)
        or not cfg['truth_side_only'] or cfg['persist_raw_coordinates'] or cfg['new_candidates']):
        raise ValueError('Frozen audit drift')
    _,files=manifest_files(cfg['diagnoses'])
    return cfg,files


def implementation():
    paths=[Path(__file__),ROOT/'scripts/run_retrieval_geo_v16_workbench.py',
           ROOT/'src/xgb_matcher/retrieval_geo_v16.py',ROOT/'src/xgb_matcher/hierarchical_retrieval.py',
           ROOT/'src/xgb_matcher/retrieval_v20_workbench.py',ROOT/'src/xgb_matcher/retrieval_v21_membership.py',
           ROOT/'src/xgb_matcher/retrieval_v25_conjunctive.py']
    return {str(p.relative_to(ROOT)):record(p) for p in paths}


def selected_labels(c,files,preflight):
    if preflight is not None and not 1<=preflight<=16:raise ValueError('Preflight1..16')
    c.execute('CREATE TEMP TABLE t AS SELECT * FROM read_parquet(?) ORDER BY query_id LIMIT ?',
              [str(files['label_side_diagnoses.parquet']),preflight or 1034])
    if c.execute('SELECT count(*),count(DISTINCT query_id) FROM t WHERE fold IN (2,3,4)').fetchone()!=(preflight or 1034,preflight or 1034):
        raise ValueError('Query/fold population')


def save(output,result):
    result['implementation']=implementation()
    result['files']={p.name:record(p) for p in output.iterdir() if p.is_file()}
    write_json(output/'manifest.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('files','implementation','index_inventory','segments')}),flush=True)


def segments(df,summary):
    return {dim:{str(k):summary(v) for k,v in df.groupby(dim,dropna=False)} for dim in
            ('loss_stage','cohort_kind','fold','data_origin','label_kind','qualification_v2','qualification_v3')}


def names_summary(df):
    r=dict(queries=len(df),baseline_exact=int(df.exact_hit.sum()),baseline_operational=int(df.operational_hit.sum()))
    for scope in ('portfolio','global_docs','local_docs'):
        for key in ('name_count','exact_name','any_token'):
            r[f'{scope}_{key}_queries']=int(df[f'{scope}_{key}'].gt(0).sum())
        for key in ('jw','tokens'):
            r[f'{scope}_{key}_median']=float(df[f'{scope}_{key}'].median())
    r['exact_name_lost_global']=int((df.portfolio_exact_name.gt(0)&df.global_docs_exact_name.eq(0)).sum())
    r['exact_name_lost_local']=int((df.portfolio_exact_name.gt(0)&df.local_docs_exact_name.eq(0)).sum())
    r['token_overlap_lost_local']=int((df.portfolio_any_token.gt(0)&df.local_docs_any_token.eq(0)).sum())
    r['queries_with_unrepresented_names']=int(df.unrepresented_names.gt(0).sum())
    return r


def names(config_path,output,preflight=None):
    cfg,files=load(config_path)
    v23=json.loads(verify(cfg['names_config']).read_text())
    portfolio=verify(v23['names']);impath=verify(cfg['index_manifest']);im=json.loads(impath.read_text())
    sp=verify(im['sources']['dossier_siret_documents']);pp=verify(im['sources']['dossier_siren_documents'])
    indexroot=impath.parent;inv=index_inventory(indexroot/'tantivy');meta=record(indexroot/'tantivy/meta.json')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    with Budget(cfg['resources']) as budget:
        c=open_workbench_db(output);selected_labels(c,files,preflight)
        c.execute("CREATE MACRO norm(x) AS trim(regexp_replace(upper(strip_accents(coalesce(x,''))),'[^A-Z0-9]+',' ','g'))")
        c.execute('''CREATE TEMP TABLE portfolio AS SELECT DISTINCT n.siren,n.siret,n.subject_kind,n.name_role,
          norm(n.normalized_value) AS "name",n.selected_for_subject,n.selected_for_siren
          FROM read_parquet(?) n SEMI JOIN (SELECT DISTINCT gt_siren siren FROM t) s USING(siren)
          WHERE n.name_role IN ('LEGAL_CURRENT','TRADE_CURRENT','SITE_CURRENT','HISTORICAL') AND norm(n.normalized_value)<>'' ''',[str(portfolio)])
        cols='siren,document_kind,insee,postcode,'+','.join(v[0] for v in FIELDS.values())
        c.execute(f'''CREATE TEMP TABLE docs AS SELECT {cols} FROM read_parquet(?)
          SEMI JOIN (SELECT DISTINCT gt_siren siren FROM t) s USING(siren)''',[[str(sp),str(pp)]])
        branches=[]
        for role,(source_field,_) in FIELDS.items():
            branches.append(f"SELECT DISTINCT siren,document_kind,insee,postcode,'{role}' name_role,norm(x) AS \"name\" FROM docs,unnest(string_split({source_field},'|')) v(x) WHERE norm(x)<>''")
        c.execute('CREATE TEMP TABLE doc_names AS '+' UNION ALL '.join(branches))
        c.execute('''CREATE TEMP TABLE local_names AS SELECT DISTINCT t.query_id,t.fold,n.document_kind,n.name_role,n.name
          FROM t JOIN doc_names n ON n.siren=t.gt_siren AND
          CASE WHEN t.effective_geo_field='insee' THEN n.insee=t.query_insee ELSE n.postcode=t.query_postcode END''')
        backend=TantivyBackend(indexroot);tv=backend._tantivy
        actual=[];index_documents=0
        for i,row in enumerate(c.execute('SELECT query_id,fold,gt_siren,query_name,query_address,query_insee,query_postcode FROM t ORDER BY query_id').fetchall(),1):
            qid,fold,siren,name,address,insee,postcode=row
            query=RetrievalQuery(name=name,address=address,insee=insee,postcode=postcode)
            term=tv.Query.term_query(backend.index.schema,'siren',siren,'basic')
            for kind in ('SIREN','SIRET'):
                parsed=backend._combined_query(query,term,document_type=kind.lower())
                offset=0;total=None;found=set()
                while total is None or offset<total:
                    page=backend.searcher.search(parsed,limit=4096,offset=offset)
                    total=page.count
                    if not page.hits:break
                    for _,loc in page.hits:
                        doc=backend.searcher.doc(loc).to_dict();index_documents+=1
                        for role,(_,field) in FIELDS.items():
                            for value in doc.get(field,[]):
                                normalized=normalize_text(value)
                                if normalized:found.add((role,normalized))
                    offset+=len(page.hits)
                if offset!=total:raise ValueError('Incomplete physical-index pagination')
                actual.extend(dict(query_id=qid,fold=fold,document_kind=kind,name_role=role,name=name) for role,name in sorted(found))
            if i%100==0:print(json.dumps(dict(names_queries=i)),flush=True)
        schema=pa.schema([('query_id',pa.string()),('fold',pa.int64()),('document_kind',pa.string()),('name_role',pa.string()),('name',pa.string())])
        c.register('actual_index_names',pa.Table.from_pylist(actual,schema=schema))
        missing=c.execute('SELECT count(*) FROM local_names ANTI JOIN actual_index_names USING(query_id,fold,document_kind,name_role,name)').fetchone()[0]
        extra=c.execute('SELECT count(*) FROM actual_index_names ANTI JOIN local_names USING(query_id,fold,document_kind,name_role,name)').fetchone()[0]
        if missing or extra:raise ValueError(f'Source/index name discrepancy: missing={missing}, extra={extra}')
        c.execute('''CREATE TEMP TABLE views AS
          SELECT DISTINCT t.query_id,'portfolio' AS "scope",n.name_role,n.name FROM t JOIN portfolio n ON n.siren=t.gt_siren
          UNION SELECT DISTINCT t.query_id,'global_docs',n.name_role,n.name FROM t JOIN doc_names n ON n.siren=t.gt_siren
          UNION SELECT DISTINCT query_id,'local_docs',name_role,name FROM local_names''')
        c.execute('''CREATE TEMP TABLE similarities AS SELECT v.*,CASE WHEN t.query_name='' THEN 0 ELSE jaro_winkler_similarity(t.query_name,v.name) END jw,
          CASE WHEN t.query_name='' THEN 0 ELSE len(list_intersect(string_split(t.query_name,' '),string_split(v.name,' ')))::DOUBLE /
            len(list_distinct(list_concat(string_split(t.query_name,' '),string_split(v.name,' ')))) END tokens,
          CAST(t.query_name<>'' AND t.query_name=v.name AS INT) exact_name FROM views v JOIN t USING(query_id)''')
        c.execute('''CREATE TEMP TABLE per_role AS SELECT query_id,scope,name_role,count(DISTINCT name) name_count,
          max(jw) jw,max(tokens) tokens,max(exact_name) exact_name,max(CAST(tokens>0 AS INT)) any_token
          FROM similarities GROUP BY query_id,scope,name_role''')
        c.execute('''CREATE TEMP TABLE missing_names AS SELECT DISTINCT p.query_id,p.name_role,p.name FROM views p
          ANTI JOIN (SELECT DISTINCT query_id,name FROM views WHERE scope='global_docs') d USING(query_id,name)
          WHERE p.scope='portfolio' ''')
        c.execute('CREATE TEMP TABLE result AS SELECT * FROM t')
        for scope in ('portfolio','global_docs','local_docs'):
            c.execute(f'''CREATE OR REPLACE TEMP TABLE result AS SELECT r.*,coalesce(x.name_count,0) {scope}_name_count,
              coalesce(x.jw,0) {scope}_jw,coalesce(x.tokens,0) {scope}_tokens,coalesce(x.exact_name,0) {scope}_exact_name,
              coalesce(x.any_token,0) {scope}_any_token FROM result r LEFT JOIN
              (SELECT query_id,count(DISTINCT name) name_count,max(jw) jw,max(tokens) tokens,max(exact_name) exact_name,
               max(CAST(tokens>0 AS INT)) any_token FROM similarities WHERE scope='{scope}' GROUP BY query_id) x USING(query_id)''')
        c.execute('''CREATE OR REPLACE TEMP TABLE result AS SELECT r.*,coalesce(m.n,0) unrepresented_names
          FROM result r LEFT JOIN (SELECT query_id,count(*) n FROM missing_names GROUP BY query_id) m USING(query_id)''')
        df=c.execute('SELECT * FROM result ORDER BY query_id').df()
        for table in ('result','per_role','missing_names'):
            c.execute(f'COPY {table} TO {sql_path(output/(table+".parquet"))} (FORMAT PARQUET,COMPRESSION ZSTD)')
        c.close()
        if index_inventory(indexroot/'tantivy')!=inv or record(indexroot/'tantivy/meta.json')!=meta:raise ValueError('Index mutation')
    save(output,dict(schema='sireto-v26-names-audit-1',decision='PIVOT_GENERATION',config=record(config_path),preflight=preflight is not None,
        primary=None if preflight else names_summary(df),segments={} if preflight else segments(df,names_summary),queries=len(df),
        actual_index_documents_read=index_documents,local_materialization_matches_physical_index=True,index_meta=meta,index_inventory=inv,
        sources=[record(p) for p in (portfolio,sp,pp,impath)],truth_side_only=True,new_candidates=False,fold0_opened=False,fold1_opened=False,
        baseline_metrics_unchanged=True,resources=budget.report()))


def geo_summary(df):
    r=dict(queries=len(df),baseline_exact=int(df.exact_hit.sum()),baseline_operational=int(df.operational_hit.sum()),
           ban_resolved=int(df.ban_resolved.sum()),ban_not_declared_weak=int((df.ban_resolved & ~df.ban_declared_weak).sum()),
           distance_available=int(df.nearest_parent_m.notna().sum()),exact_distance_available=int(df.exact_distance_m.notna().sum()),
           any_parent_missing_coordinates=int(df.valid_coordinate_sites.lt(df.parent_site_count).sum()),
           previous_geo_parent_present=int(df.previous_geo_parent_present.sum()))
    for n in (50,100,250,500):
        for field in ('nearest_parent_m','exact_distance_m','nearest_active_m','nearest_different_street_m'):
            r[f'{field}_le_{n}']=int(df[field].le(n).sum())
        r[f'parent_le_{n}_ban_not_weak']=int((df.nearest_parent_m.le(n)&~df.ban_declared_weak).sum())
    r['other_site_strictly_closer']=int((df.nearest_parent_m<df.exact_distance_m).sum())
    return r


def geo(config_path,output,preflight=None):
    cfg,files=load(config_path)
    gcpath=verify(cfg['geo_config']);gc=json.loads(gcpath.read_text());a=gc['authorities']
    sources=[gcpath]
    for key in ('sirene_snapshot','ban_silver','ban_catalog'):
        sources.append(verify(dict(path=a[key+'_path'],sha256=a[key+'_sha256'])))
    v17=json.loads(verify(cfg['v17_config']).read_text())['authorities']
    oldgeo=verify(dict(path=str(Path(v17['v16_geo_generation_dir'])/'geo_neighbor_parents.parquet'),sha256=v17['v16_geo_parent_sha256']))
    sources.append(oldgeo)
    scene=verify(dict(path=str(Path(a['scene_ir_dir'])/'manifest.json'),sha256=a['scene_ir_manifest_sha256']));sources.append(scene)
    # SceneIR's raw data authority is independently pinned through the names config/v18 config lineage.
    v18=json.loads(verify(dict(path='config/retrieval_all_human_v18_parent_discovery_generation.json',
        sha256='e270a60d520d7c7be7928988382ce91fc8f326c7bc1541a7c53eff0ce1aa80e6')).read_text())
    sources.append(verify(v18['generation_authorities']['raw_union']))
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    with Budget(cfg['resources']) as budget:
        c=open_workbench_db(output);selected_labels(c,files,preflight)
        ids=tuple(c.execute('SELECT query_id,fold FROM t ORDER BY query_id').fetchall())
        qs=_query_surface(gc,ids);qg=_resolve_ban(c,qs,sources[2])
        c.register('qs',qs);c.register('qg',qg)
        c.execute('''CREATE TEMP TABLE sites AS SELECT s.siret,s.siren,s.etatAdministratifEtablissement state,
          s.codeCommuneEtablissement insee,s.numeroVoieEtablissement number,
          trim(regexp_replace(upper(strip_accents(concat_ws(' ',s.typeVoieEtablissement,s.libelleVoieEtablissement))),'[^A-Z0-9]+',' ','g')) street,
          try_cast(s.coordonneeLambertAbscisseEtablissement AS DOUBLE) x,
          try_cast(s.coordonneeLambertOrdonneeEtablissement AS DOUBLE) y
          FROM read_parquet(?) s SEMI JOIN (SELECT DISTINCT gt_siren siren FROM t) p USING(siren)''',[str(sources[1])])
        c.execute('''CREATE TEMP TABLE distances AS SELECT t.query_id,s.siret,s.state,
          s.siret=t.gt_siret is_exact,s.street<>qs.crm_street_norm different_street,
          coalesce(s.x BETWEEN 0 AND 1300000 AND s.y BETWEEN 6000000 AND 7300000,false) valid_coordinate,
          CASE WHEN qg.ban_resolved AND qg.ban_x_l93 BETWEEN 0 AND 1300000 AND qg.ban_y_l93 BETWEEN 6000000 AND 7300000
            AND s.x BETWEEN 0 AND 1300000 AND s.y BETWEEN 6000000 AND 7300000
            THEN sqrt(pow(s.x-qg.ban_x_l93,2)+pow(s.y-qg.ban_y_l93,2)) END distance_m
          FROM t JOIN qs USING(query_id,fold) JOIN qg USING(query_id,fold) JOIN sites s ON s.siren=t.gt_siren''')
        c.execute('''CREATE TEMP TABLE result AS SELECT t.*,qg.* EXCLUDE(query_id,fold,ban_x_l93,ban_y_l93),
          coalesce(qg.ban_x_l93 BETWEEN 0 AND 1300000 AND qg.ban_y_l93 BETWEEN 6000000 AND 7300000,false) query_coordinate_valid,
          coalesce(d.n,0) parent_site_count,coalesce(d.ncoord,0) valid_coordinate_sites,d.nearest_parent_m,d.exact_distance_m,
          d.nearest_active_m,d.nearest_different_street_m,d.nearest_siret,
          EXISTS(SELECT 1 FROM read_parquet(?) g WHERE g.query_id=t.query_id AND g.siren=t.gt_siren) previous_geo_parent_present
          FROM t JOIN qg USING(query_id,fold) LEFT JOIN (
            SELECT query_id,count(*) n,sum(CAST(valid_coordinate AS INT)) ncoord,min(distance_m) nearest_parent_m,
              min(distance_m) FILTER(WHERE is_exact) exact_distance_m,min(distance_m) FILTER(WHERE state='A') nearest_active_m,
              min(distance_m) FILTER(WHERE different_street) nearest_different_street_m,arg_min(siret,distance_m) nearest_siret
            FROM distances GROUP BY query_id) d USING(query_id)''',[str(oldgeo)])
        df=c.execute('SELECT * FROM result ORDER BY query_id').df()
        c.execute(f'COPY result TO {sql_path(output/"result.parquet")} (FORMAT PARQUET,COMPRESSION ZSTD)')
        # Explicit allowlist: never serialize the in-memory query coordinate surface or raw site coordinates.
        c.execute(f'''COPY (SELECT query_id,siret,state,is_exact,different_street,valid_coordinate,distance_m FROM distances)
          TO {sql_path(output/'site_distances.parquet')} (FORMAT PARQUET,COMPRESSION ZSTD)''')
        site_pairs=c.execute('SELECT count(*) FROM distances').fetchone()[0]
        c.close()
    save(output,dict(schema='sireto-v26-geo-audit-1',decision='PIVOT_GENERATION',config=record(config_path),preflight=preflight is not None,
        queries=len(df),site_pairs=site_pairs,primary=None if preflight else geo_summary(df),segments={} if preflight else segments(df,geo_summary),
        ban_quality_segments={} if preflight else {str(k):geo_summary(v) for k,v in df.groupby('ban_declared_weak')},
        sources=[record(p) for p in sources],truth_side_only=True,raw_coordinates_persisted=False,all_parent_sites_no_geography_cap=True,
        new_candidates=False,fold0_opened=False,fold1_opened=False,baseline_metrics_unchanged=True,resources=budget.report()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['names','geo']);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--preflight',type=int)
    a=p.parse_args();{'names':names,'geo':geo}[a.action](a.config,a.output,a.preflight)
