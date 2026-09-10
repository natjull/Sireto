"""Bounded v27 truth-side audits. No new retrieval recipe or coordinate export."""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.xgb_matcher.retrieval_v20_workbench import ROOT, Budget, record, verify, write_json, sql_path
from src.xgb_matcher.retrieval_v21_membership import manifest_files, open_workbench_db
from scripts.audit_retrieval_v26_evidence import segments
from scripts.run_retrieval_geo_v16_workbench import _query_surface, _resolve_ban, _write_geo_neighbors

UL_FIELDS = {'denominationUniteLegale':'LEGAL', 'denominationUsuelle1UniteLegale':'USUAL',
             'denominationUsuelle2UniteLegale':'USUAL', 'denominationUsuelle3UniteLegale':'USUAL',
             'sigleUniteLegale':'TRADE'}
SITE_FIELDS = {'enseigne1Etablissement':'SIGN', 'enseigne2Etablissement':'SIGN',
               'enseigne3Etablissement':'SIGN', 'denominationUsuelleEtablissement':'USUAL'}


def load(path):
    cfg = json.loads(Path(path).read_text())
    if (cfg['schema'] != 'sireto-v27-upstream-audit-1' or cfg['folds'] != [2,3,4]
        or not cfg['truth_side_only'] or cfg['new_candidates'] or cfg['persist_raw_coordinates']
        or (cfg['geo_site_cap'],cfg['geo_parent_cap'],cfg['focus_distance_m']) != (4096,512,500)):
        raise ValueError('Frozen audit drift')
    return cfg


def select(c, path, preflight):
    if preflight is not None and not 1 <= preflight <= 16:
        raise ValueError('Technical preflight must be 1..16')
    c.execute('CREATE TEMP TABLE t AS SELECT * FROM read_parquet(?) ORDER BY query_id LIMIT ?',
              [str(path), preflight or 1034])
    assert c.execute('SELECT count(*),count(DISTINCT query_id) FROM t WHERE fold IN (2,3,4)').fetchone() == (preflight or 1034,)*2


def save(output, config, sources, result):
    result.update(config=record(config), sources=sources, truth_side_only=True,
                  new_candidates=False, raw_coordinates_persisted=False,
                  fold0_opened=False, fold1_opened=False, baseline_metrics_unchanged=True,
                  decision='PIVOT_GENERATION')
    paths = [Path(__file__),ROOT/'scripts/run_retrieval_geo_v16_workbench.py',
             ROOT/'src/xgb_matcher/retrieval_geo_v16.py',ROOT/'src/xgb_matcher/retrieval_v20_workbench.py',
             ROOT/'src/xgb_matcher/retrieval_v21_membership.py',ROOT/'scripts/audit_retrieval_v26_evidence.py']
    result['implementation'] = {str(p.relative_to(ROOT)):record(p) for p in paths}
    result['files'] = {p.name:record(p) for p in output.iterdir() if p.is_file()}
    write_json(output/'manifest.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('files','sources','implementation','segments')}),flush=True)


def geo_reason(resolved, valid_sites, local_sites, site_rank, parent_rank):
    if not resolved: return 'BAN_UNRESOLVED'
    if not valid_sites: return 'NO_VALID_PARENT_COORDINATE'
    if not local_sites: return 'COMMUNE_BOUNDARY'
    if pd.isna(site_rank): return 'SITE_CAP_4096'
    if pd.isna(parent_rank): raise ValueError('Parent ranking integrity failure')
    return 'PARENT_CAP_512' if parent_rank > 512 else 'PRESENT'


def geo_summary(d):
    return dict(queries=len(d),baseline_exact=int(d.exact_hit.sum()),baseline_operational=int(d.operational_hit.sum()),
                reasons={str(k):int(v) for k,v in d.geo_reason.value_counts().items()},
                focus_missing_near500=int(d.focus_missing_near500.sum()),
                nearest_excluded_by_commune=int(d.nearest_excluded_by_commune.sum()))


def geo(config, output, preflight):
    cfg=load(config);m,files=manifest_files(cfg['v26_geo'])
    sources=[record(verify(cfg['v26_geo']))]
    for src in m['sources']: sources.append(record(verify(src)))
    gc=json.loads(Path(m['sources'][0]['path']).read_text())
    stock=Path(gc['authorities']['sirene_snapshot_path']);ban=Path(gc['authorities']['ban_silver_path'])
    oldmanifest=verify(cfg['v16_manifest']);old=json.loads(oldmanifest.read_text());sources.append(record(oldmanifest))
    oldfiles={name:verify(dict(path=str(oldmanifest.parent/name),sha256=r['sha256'])) for name,r in old['files'].items()
              if name in ('query_geo.parquet','geo_neighbor_candidates.parquet','geo_neighbor_parents.parquet')}
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    with Budget(cfg['resources']) as budget:
        c=open_workbench_db(output);select(c,files['result.parquet'],preflight)
        ids=tuple(c.execute('SELECT query_id,fold FROM t ORDER BY query_id').fetchall())
        qs=_query_surface(gc,ids);qg=_resolve_ban(c,qs,ban)
        c.register('qs',qs);c.register('qg',qg)
        original=c.execute('SELECT q.* FROM read_parquet(?) q SEMI JOIN t USING(query_id) ORDER BY query_id',[str(oldfiles['query_geo.parquet'])]).df()
        replay=c.execute('SELECT * EXCLUDE(ban_x_l93,ban_y_l93) FROM qg ORDER BY query_id').df()
        pd.testing.assert_frame_equal(original,replay,check_dtype=False)
        _write_geo_neighbors(c,snapshot_path=stock,query_surface=qs,query_geo=qg,
                            output_path=output/'replay_sites.parquet',parent_output_path=output/'replay_parents.parquet',neighbor_cap=512)
        for key,filename,sort,exclude in [('geo_neighbor_candidates.parquet','replay_sites.parquet','query_id,geo_neighbor_rank',[]),
                                         ('geo_neighbor_parents.parquet','replay_parents.parquet','query_id,geo_parent_rank',['nearest_siret'])]:
            a=c.execute(f'SELECT p.* FROM read_parquet(?) p SEMI JOIN t USING(query_id) ORDER BY {sort}',[str(oldfiles[key])]).df()
            b=c.execute(f'SELECT * FROM read_parquet(?) ORDER BY {sort}',[str(output/filename)]).df()
            pd.testing.assert_frame_equal(a.drop(columns=exclude),b.drop(columns=exclude),check_dtype=False)
        # The legacy prefix-derived SIREN must agree with the source's official parent.
        assert c.execute('''SELECT count(*) FROM geo_v16_nearest_sites n JOIN read_parquet(?) s USING(siret)
          WHERE n.siren IS DISTINCT FROM s.siren''',[str(stock)]).fetchone()[0] == 0
        c.execute('''CREATE TEMP TABLE parent_ranks AS WITH p AS (
          SELECT query_id,siren,min(geo_distance_m) distance,min(geo_neighbor_rank) site_rank,
            max(candidate_number_exact) num,max(candidate_street_similarity) street
          FROM geo_v16_nearest_sites GROUP BY query_id,siren)
          SELECT *,row_number() OVER(PARTITION BY query_id ORDER BY distance,num DESC,street DESC,siren) parent_rank FROM p''')
        c.execute('''CREATE TEMP TABLE gt_sites AS SELECT t.query_id,s.siret,s.siren,s.codeCommuneEtablissement insee,
          try_cast(s.coordonneeLambertAbscisseEtablissement AS DOUBLE) x,
          try_cast(s.coordonneeLambertOrdonneeEtablissement AS DOUBLE) y
          FROM read_parquet(?) s JOIN t ON s.siren=t.gt_siren''',[str(stock)])
        c.execute('''CREATE TEMP TABLE gd AS SELECT s.query_id,
          count(*) FILTER(WHERE s.x BETWEEN 0 AND 1300000 AND s.y BETWEEN 6000000 AND 7300000) valid_sites,
          count(*) FILTER(WHERE s.insee=q.crm_insee AND s.x BETWEEN 0 AND 1300000 AND s.y BETWEEN 6000000 AND 7300000) local_valid_sites,
          min(CASE WHEN s.x BETWEEN 0 AND 1300000 AND s.y BETWEEN 6000000 AND 7300000 AND g.ban_resolved
            THEN sqrt(pow(s.x-g.ban_x_l93,2)+pow(s.y-g.ban_y_l93,2)) END) nearest_m,
          min(CASE WHEN s.insee=q.crm_insee AND s.x BETWEEN 0 AND 1300000 AND s.y BETWEEN 6000000 AND 7300000 AND g.ban_resolved
            THEN sqrt(pow(s.x-g.ban_x_l93,2)+pow(s.y-g.ban_y_l93,2)) END) local_nearest_m
          FROM gt_sites s JOIN qs q USING(query_id) JOIN qg g USING(query_id) GROUP BY s.query_id''')
        assert c.execute('SELECT count(*) FROM t LEFT JOIN gd USING(query_id) WHERE nearest_parent_m IS DISTINCT FROM nearest_m').fetchone()[0] == 0
        df=c.execute('''SELECT t.*,coalesce(g.valid_sites,0) valid_sites,coalesce(g.local_valid_sites,0) local_valid_sites,
          g.local_nearest_m,p.site_rank first_local_site_rank,p.parent_rank local_parent_rank
          FROM t LEFT JOIN gd g USING(query_id) LEFT JOIN parent_ranks p ON p.query_id=t.query_id AND p.siren=t.gt_siren ORDER BY t.query_id''').df()
        df['geo_reason']=[geo_reason(r.ban_resolved,r.valid_sites,r.local_valid_sites,r.first_local_site_rank,r.local_parent_rank) for r in df.itertuples()]
        assert (df.geo_reason.eq('PRESENT') == df.previous_geo_parent_present).all()
        df['nearest_excluded_by_commune']=df.nearest_parent_m.notna() & (df.local_nearest_m.isna() | df.local_nearest_m.gt(df.nearest_parent_m))
        df['focus_missing_near500']=df.loss_stage.eq('PARENT_NOT_RETRIEVED') & df.nearest_parent_m.le(500)
        if preflight is None: assert int(df.focus_missing_near500.sum()) == 12
        c.register('final',pa.Table.from_pandas(df,preserve_index=False));c.execute(f'COPY final TO {sql_path(output/"result.parquet")} (FORMAT PARQUET,COMPRESSION ZSTD)')
        c.close()
    save(output,config,sources,dict(schema='sireto-v27-geo-attribution-1',preflight=preflight is not None,
        primary=None if preflight else geo_summary(df),segments={} if preflight else segments(df,geo_summary),
        focus=None if preflight else geo_summary(df[df.focus_missing_near500]),old_replay_equal=True,resources=budget.report()))


def names_summary(d):
    return dict(queries=len(d),baseline_exact=int(d.exact_hit.sum()),baseline_operational=int(d.operational_hit.sum()),
                raw_current_exact=int(d.current_exact.sum()),raw_history_exact=int(d.history_exact.sum()),
                current_missing_portfolio_exact=int(d.current_missing_exact.sum()),
                history_missing_portfolio_exact=int(d.history_missing_exact.sum()),
                history_adds_exact_over_portfolio=int((d.history_missing_exact & ~d.portfolio_exact_name).sum()))


def names(config, output, preflight):
    cfg=load(config);m,files=manifest_files(cfg['v26_names']);sources=[record(verify(cfg['v26_names']))]
    portfolio=verify(m['sources'][0]);sources.append(record(portfolio))
    dm=verify(cfg['dossier_manifest']);dossier=json.loads(dm.read_text());sources.append(record(dm))
    evidence=verify(dict(path=str(dm.parent/'name_evidence.parquet'),sha256=dossier['files']['name_evidence.parquet']['sha256']));sources.append(record(evidence))
    inputs={key:verify(cfg[key]) for key in ('sirene_current_ul','sirene_current_site','sirene_history_ul','sirene_history_site','rne_canonical')}
    sources.extend(record(p) for p in inputs.values())
    assert cfg['sirene_current_ul']['sha256']==dossier['inputs']['sirene_legal_units']['sha256']
    assert cfg['sirene_current_site']['sha256']==dossier['inputs']['sirene_establishments']['sha256']
    assert cfg['rne_canonical']['sha256'] in {r['sha256'] for r in dossier['inputs']['official_evidence']}
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    with Budget(cfg['resources']) as budget:
        c=open_workbench_db(output);select(c,files['result.parquet'],preflight)
        c.execute("CREATE MACRO norm(v) AS trim(regexp_replace(upper(strip_accents(coalesce(v,''))),'[^A-Z0-9]+',' ','g'))")
        c.execute('''CREATE TEMP TABLE e AS SELECT siren,coalesce(siret,'') siret,subject_kind,source,name_kind,norm(normalized_value) AS name
          FROM read_parquet(?) SEMI JOIN (SELECT DISTINCT gt_siren siren FROM t) p USING(siren)''',[str(evidence)])
        c.execute('''CREATE TEMP TABLE p AS SELECT siren,coalesce(siret,'') siret,subject_kind,name_role,norm(normalized_value) AS name
          FROM read_parquet(?) SEMI JOIN (SELECT DISTINCT gt_siren siren FROM t) t USING(siren)
          WHERE name_role IN ('LEGAL_CURRENT','TRADE_CURRENT','SITE_CURRENT','HISTORICAL')''',[str(portfolio)])
        c.execute('''CREATE TEMP TABLE raw_names(source_group VARCHAR,source VARCHAR,siren VARCHAR,siret VARCHAR,
          subject_kind VARCHAR,name_kind VARCHAR,field VARCHAR,name VARCHAR)''')
        for key,path in inputs.items():
            if key=='rne_canonical': continue
            fields=UL_FIELDS if key.endswith('_ul') else SITE_FIELDS
            available={x[0] for x in c.execute('DESCRIBE SELECT * FROM read_parquet(?)',[str(path)]).fetchall()}
            cols=[f for f in fields if f in available]
            siret="''" if key.endswith('_ul') else 's.siret'
            subject='SIREN' if key.endswith('_ul') else 'SIRET'
            source='SIRENE_HISTORY' if 'history' in key else 'SIRENE_CURRENT'
            values=','.join(f"('{col}','{fields[col]}',s.{col}::VARCHAR)" for col in cols)
            c.execute(f'''INSERT INTO raw_names SELECT DISTINCT '{key}','{source}',s.siren,{siret},'{subject}',v.kind,v.field,norm(v.value)
              FROM read_parquet(?) s SEMI JOIN (SELECT DISTINCT gt_siren siren FROM t) t USING(siren),
              LATERAL (VALUES {values}) v(field,kind,value) WHERE norm(v.value)<>'' ''',[str(path)])
        c.execute('''CREATE TEMP TABLE audited_names AS SELECT r.*,
          EXISTS(SELECT 1 FROM e WHERE e.siren=r.siren AND e.siret=r.siret AND e.subject_kind=r.subject_kind
            AND e.source=r.source AND e.name_kind=r.name_kind AND e.name=r.name) in_evidence,
          EXISTS(SELECT 1 FROM p WHERE p.siren=r.siren AND p.name=r.name) in_portfolio_global
          FROM raw_names r''')
        c.execute('''CREATE TEMP TABLE name_details AS SELECT t.query_id,t.fold,a.source_group,a.field,a.siret,a.subject_kind,
          a.name,a.in_evidence,a.in_portfolio_global,(a.name=norm(t.query_name)) exact_name,
          (len(list_intersect(string_split(a.name,' '),string_split(norm(t.query_name),' ')))>0) any_token
          FROM audited_names a JOIN t ON t.gt_siren=a.siren''')
        c.execute('''CREATE TEMP TABLE result AS SELECT t.*,coalesce(d.current_exact,false) current_exact,
          coalesce(d.history_exact,false) history_exact,coalesce(d.current_missing_exact,false) current_missing_exact,
          coalesce(d.history_missing_exact,false) history_missing_exact FROM t LEFT JOIN (
            SELECT query_id,bool_or(exact_name AND source_group LIKE '%current%') current_exact,
              bool_or(exact_name AND source_group LIKE '%history%') history_exact,
              bool_or(exact_name AND NOT in_portfolio_global AND source_group LIKE '%current%') current_missing_exact,
              bool_or(exact_name AND NOT in_portfolio_global AND source_group LIKE '%history%') history_missing_exact
            FROM name_details GROUP BY query_id) d USING(query_id)''')
        # Structural RNE comparison: no raw names or CRM similarity exported.
        c.execute('''CREATE TEMP TABLE rne AS SELECT DISTINCT x.siren,coalesce(x.siret,'') siret,x.subject_kind,
          n.kind name_kind,norm(n.normalized_value) AS name FROM read_parquet(?) x
          SEMI JOIN (SELECT DISTINCT gt_siren siren FROM t) t USING(siren),unnest(x.names) item(n)
          WHERE norm(n.normalized_value)<>'' ''',[str(inputs['rne_canonical'])])
        rne=c.execute('''SELECT count(*) value_subject_pairs,
          count(*) FILTER(WHERE NOT EXISTS(SELECT 1 FROM e WHERE e.source='RNE' AND e.siren=r.siren AND e.siret=r.siret
            AND e.subject_kind=r.subject_kind AND e.name_kind=r.name_kind AND e.name=r.name)) missing_evidence,
          count(*) FILTER(WHERE NOT EXISTS(SELECT 1 FROM p WHERE p.siren=r.siren AND p.name=r.name)) missing_portfolio_global FROM rne r''').df().to_dict('records')[0]
        per=c.execute('''SELECT source_group,count(*) field_value_subject_pairs,count(DISTINCT siren) parents,
          count(*) FILTER(WHERE NOT in_evidence) missing_evidence,
          count(*) FILTER(WHERE NOT in_portfolio_global) missing_portfolio_global FROM audited_names GROUP BY source_group ORDER BY source_group''').df().to_dict('records')
        df=c.execute('SELECT * FROM result ORDER BY query_id').df()
        for table in ('result','name_details'):
            c.execute(f'COPY {table} TO {sql_path(output/(table+".parquet"))} (FORMAT PARQUET,COMPRESSION ZSTD)')
        c.close()
    save(output,config,sources,dict(schema='sireto-v27-name-import-1',preflight=preflight is not None,
        primary=None if preflight else names_summary(df),segments={} if preflight else segments(df,names_summary),
        per_source=None if preflight else per,rne_structural=None if preflight else rne,
        raw_rne_population_audited=False,person_name_features=False,resources=budget.report()))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['geo','names'])
    ap.add_argument('--config',type=Path,default=Path('config/retrieval_v27_upstream_audit.json'))
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--preflight',type=int)
    a=ap.parse_args();{'geo':geo,'names':names}[a.action](a.config,a.output,a.preflight)
