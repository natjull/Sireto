"""One frozen OOF parent-selection pilot over the unchanged v16 spatial pool."""
import argparse
import gc
import json
from pathlib import Path
import sys

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.xgb_matcher.retrieval_v20_workbench import ROOT,Budget,record,verify,write_json,sql_path,negative_positions
from src.xgb_matcher.retrieval_v21_membership import manifest_files,open_workbench_db
from src.xgb_matcher.retrieval_parent_fusion_v19 import validate_fold_isolation,assert_identifier_free_features
from src.xgb_matcher.retrieval_v12_scene_ir import SceneIRCache
from scripts.run_retrieval_geo_v16_workbench import _query_surface,_resolve_ban,_write_geo_neighbors
from scripts.audit_retrieval_v27_upstream import UL_FIELDS,SITE_FIELDS

FEATURES=('distance_log','spatial_rank_log','site_count_log','parent_name_jw','parent_name_tokens',
          'site_name_jw','site_name_tokens','joint_name_street','joint_tokens_street','street','number_exact',
          'same_postcode','active_share','name_missing','crm_name_missing','crm_number_missing',
          'ban_street_similarity','ban_certification_commune','ban_uncertainty_log','ban_declared_weak','distance_uncertainty_ratio')


def config_at(path):
    c=json.loads(Path(path).read_text())
    if (c['schema']!='sireto-v28-geo-selection-1' or c['recipe']!='SINGLE_OOF_PARENT_RANKER_BEFORE_GEO_TOP512'
        or c['folds']!=[2,3,4] or (c['site_pool_cap'],c['parent_cap'])!=(4096,512)
        or c['new_geographic_candidates'] or c['production_activation']
        or c['sampler']!='HEAD256_PLUS_STRATIFIED_TAIL256'):
        raise ValueError('Frozen recipe drift')
    assert_identifier_free_features(FEATURES)
    return c


def implementation():
    paths=[Path(__file__),ROOT/'scripts/run_retrieval_geo_v16_workbench.py',ROOT/'scripts/audit_retrieval_v27_upstream.py',
           ROOT/'src/xgb_matcher/retrieval_v20_workbench.py',ROOT/'src/xgb_matcher/retrieval_v21_membership.py',
           ROOT/'src/xgb_matcher/retrieval_v12_scene_ir.py',ROOT/'src/xgb_matcher/retrieval_parent_fusion_v19.py',
           ROOT/'src/xgb_matcher/retrieval_geo_v16.py']
    return {str(p.relative_to(ROOT)):record(p) for p in paths}


def versions():
    return dict(duckdb=duckdb.__version__,numpy=np.__version__,pandas=pd.__version__,pyarrow=pa.__version__,xgboost=xgb.__version__)


def save(output,config,result):
    result.update(config=record(config),implementation=implementation(),versions=versions(),
                  feature_order=list(FEATURES),fold0_opened=False,fold1_opened=False,production_activation=False)
    result['files']={p.name:record(p) for p in output.iterdir() if p.is_file()}
    write_json(output/'manifest.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('implementation','files','sources','segments')}),flush=True)


def build_features(c):
    """Inputs q, pool, names. Never joins a label or a cohort field."""
    c.execute("CREATE MACRO norm(v) AS trim(regexp_replace(upper(strip_accents(coalesce(v,''))),'[^A-Z0-9]+',' ','g'))")
    c.execute("CREATE MACRO jw(a,b) AS CASE WHEN coalesce(a,'')='' OR coalesce(b,'')='' THEN 0.0 ELSE jaro_winkler_similarity(a,b) END")
    c.execute("CREATE MACRO tokens(a,b) AS CASE WHEN coalesce(a,'')='' OR coalesce(b,'')='' THEN 0.0 ELSE len(list_intersect(string_split(a,' '),string_split(b,' ')))::DOUBLE/len(list_distinct(list_concat(string_split(a,' '),string_split(b,' ')))) END")
    c.execute('''CREATE TEMP TABLE parent_names AS SELECT p.query_id,p.siren,
      max(jw(q.crm_name,n.name)) parent_name_jw,max(tokens(q.crm_name,n.name)) parent_name_tokens,
      max((n.name IS NOT NULL)::INT) has_parent_name
      FROM (SELECT DISTINCT query_id,siren FROM pool) p JOIN q USING(query_id)
      LEFT JOIN names n ON n.siren=p.siren AND n.subject_kind='SIREN' GROUP BY p.query_id,p.siren''')
    c.execute('''CREATE TEMP TABLE site_names AS SELECT p.query_id,p.siret,
      max(jw(q.crm_name,n.name)) site_name_jw,max(tokens(q.crm_name,n.name)) site_name_tokens,
      max((n.name IS NOT NULL)::INT) has_site_name
      FROM pool p JOIN q USING(query_id) LEFT JOIN names n ON n.siren=p.siren AND n.siret=p.siret
      AND n.subject_kind='SIRET' GROUP BY p.query_id,p.siret''')
    c.execute('''CREATE TEMP TABLE site_features AS SELECT p.*,pn.parent_name_jw,pn.parent_name_tokens,pn.has_parent_name,
      sn.site_name_jw,sn.site_name_tokens,sn.has_site_name,
      greatest(pn.parent_name_jw,sn.site_name_jw)*coalesce(p.candidate_street_similarity,0) joint_name_street,
      greatest(pn.parent_name_tokens,sn.site_name_tokens)*coalesce(p.candidate_street_similarity,0) joint_tokens_street,
      q.crm_name='' crm_name_missing,q.crm_number='' crm_number_missing
      FROM pool p JOIN q USING(query_id,fold) JOIN parent_names pn USING(query_id,siren) JOIN site_names sn USING(query_id,siret)''')
    c.execute('''CREATE TEMP TABLE parent_features AS WITH p AS (
      SELECT query_id,fold,siren,ln(1+min(geo_distance_m)) distance_log,
        ln(1+min(geo_neighbor_rank)) spatial_rank_log,ln(1+count(*)) site_count_log,
        max(parent_name_jw) parent_name_jw,max(parent_name_tokens) parent_name_tokens,
        max(site_name_jw) site_name_jw,max(site_name_tokens) site_name_tokens,
        max(joint_name_street) joint_name_street,max(joint_tokens_street) joint_tokens_street,
        max(coalesce(candidate_street_similarity,0)) street,max(coalesce(candidate_number_exact,0)) number_exact,
        max(coalesce(candidate_same_postcode,0)) same_postcode,avg((candidate_state='A')::INT) active_share,
        1-max(greatest(has_parent_name,has_site_name)) name_missing,
        max(crm_name_missing::INT) crm_name_missing,max(crm_number_missing::INT) crm_number_missing,
        max(ban_street_similarity) ban_street_similarity,max(ban_certification_commune::INT) ban_certification_commune,
        ln(1+max(ban_uncertainty_m)) ban_uncertainty_log,max(ban_declared_weak::INT) ban_declared_weak,
        min(geo_distance_m)/max(ban_uncertainty_m) distance_uncertainty_ratio,
        min(geo_distance_m) geo_parent_distance_m,max(candidate_number_exact) legacy_number,
        max(candidate_street_similarity) legacy_street
      FROM site_features GROUP BY query_id,fold,siren)
      SELECT *,row_number() OVER(PARTITION BY query_id ORDER BY geo_parent_distance_m,
        legacy_number DESC,legacy_street DESC,siren) legacy_parent_rank FROM p''')


def generate(config,output,preflight=None):
    cfg=config_at(config);v18=json.loads(verify(cfg['v18_config']).read_text())['generation_authorities']
    geo=json.loads(verify(cfg['geo_config']).read_text());a=geo['authorities']
    sources=[record(verify(v18[k])) for k in ('queries','raw_union')]
    scene=verify(dict(path=str(Path(a['scene_ir_dir'])/'manifest.json'),sha256=a['scene_ir_manifest_sha256']));sources.append(record(scene))
    for key in ('sirene_snapshot','ban_silver','ban_catalog'):
        sources.append(record(verify(dict(path=a[key+'_path'],sha256=a[key+'_sha256']))))
    ul=verify(cfg['current_ul']);sites=verify(cfg['current_sites']);sources.extend([record(ul),record(sites)])
    oldpath=verify(cfg['baseline_geo']);old=json.loads(oldpath.read_text());sources.append(record(oldpath))
    oldparents=verify(dict(path=str(oldpath.parent/'geo_neighbor_parents.parquet'),sha256=old['files']['geo_neighbor_parents.parquet']['sha256']))
    if preflight is not None and not 1<=preflight<=16:raise ValueError('Preflight1..16')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    with Budget(cfg['resources']) as budget:
        c=open_workbench_db(output);c.execute("SET memory_limit='8GB'");c.execute('SET threads=2');c.execute('SET preserve_insertion_order=false')
        ids=pq.read_table(v18['queries']['path']);assert set(ids.column_names)=={'query_id','fold'}
        rows=sorted((str(r['query_id']),int(r['fold'])) for r in ids.to_pylist())[:preflight or 1034]
        assert len(rows)==(preflight or 1034) and len(set(r[0] for r in rows))==len(rows) and {r[1] for r in rows}<={2,3,4}
        qs=_query_surface(geo,tuple(rows));qg=_resolve_ban(c,qs,Path(a['ban_silver_path']))
        _write_geo_neighbors(c,snapshot_path=Path(a['sirene_snapshot_path']),query_surface=qs,query_geo=qg,
                            output_path=output/'baseline_sites.parquet',parent_output_path=output/'baseline_parents.parquet',neighbor_cap=512)
        c.register('qs',qs)
        assert c.execute('''SELECT count(*) FROM ((SELECT b.* EXCLUDE(nearest_siret) FROM read_parquet(?) b SEMI JOIN qs USING(query_id)
          EXCEPT SELECT * EXCLUDE(nearest_siret) FROM read_parquet(?)) UNION ALL
          (SELECT * EXCLUDE(nearest_siret) FROM read_parquet(?) EXCEPT SELECT b.* EXCLUDE(nearest_siret) FROM read_parquet(?) b SEMI JOIN qs USING(query_id)))''',
          [str(oldparents),str(output/'baseline_parents.parquet'),str(output/'baseline_parents.parquet'),str(oldparents)]).fetchone()[0]==0
        cache=SceneIRCache(Path(a['scene_ir_dir']),verify_source_hash=False)
        assert cache.source.resolve()==Path(v18['raw_union']['path']).resolve()
        crm=[]
        for s in cache.iter_selected_scenes(dict(rows),columns=['query_id','fold','crm_name']):
            names=set(s.rows['crm_name'].to_pylist());assert len(names)==1
            crm.append(dict(query_id=s.query_id,crm_name=next(iter(names)) or ''))
        c.register('crm',pa.Table.from_pylist(crm))
        c.execute("CREATE TEMP TABLE q AS SELECT qs.*,trim(regexp_replace(upper(strip_accents(crm.crm_name)),'[^A-Z0-9]+',' ','g')) crm_name FROM qs JOIN crm USING(query_id)")
        assert c.execute('SELECT count(*) FROM q').fetchone()[0]==len(rows)
        # Official parent mapping is rejoined, not inferred for the new features.
        c.execute('''CREATE TEMP TABLE pool AS SELECT n.* EXCLUDE(siren),s.siren FROM geo_v16_nearest_sites n
          JOIN read_parquet(?) s USING(siret)''',[a['sirene_snapshot_path']])
        assert c.execute('SELECT count(*) FROM pool').fetchone()==c.execute('SELECT count(*) FROM geo_v16_nearest_sites').fetchone()
        assert c.execute('SELECT count(*) FROM pool p JOIN geo_v16_nearest_sites n USING(query_id,siret) WHERE p.siren<>n.siren').fetchone()[0]==0
        c.execute('CREATE TEMP TABLE names(siren VARCHAR,siret VARCHAR,subject_kind VARCHAR,name VARCHAR)')
        for path,fields,subject in [(ul,UL_FIELDS,'SIREN'),(sites,SITE_FIELDS,'SIRET')]:
            values=','.join(f'(s.{col}::VARCHAR)' for col in fields)
            siret="''" if subject=='SIREN' else 's.siret'
            on='siren' if subject=='SIREN' else 'siren,siret'
            c.execute(f'''INSERT INTO names SELECT DISTINCT s.siren,{siret},'{subject}',
              trim(regexp_replace(upper(strip_accents(v.value)),'[^A-Z0-9]+',' ','g')) FROM read_parquet(?) s
              SEMI JOIN (SELECT DISTINCT {on} FROM pool) p USING({on}),LATERAL (VALUES {values}) v(value)
              WHERE coalesce(v.value,'')<>'' ''',[str(path)])
        build_features(c)
        assert c.execute('SELECT count(*) FROM (SELECT query_id,siren,count(*) n FROM parent_features GROUP BY ALL HAVING n>1)').fetchone()[0]==0
        for name,table,order in [('queries','q','query_id'),('pool','pool','query_id,geo_neighbor_rank'),('parent_features','parent_features','query_id,siren')]:
            c.execute(f'COPY (SELECT * FROM {table} ORDER BY {order}) TO {sql_path(output/(name+".parquet"))} (FORMAT PARQUET,COMPRESSION ZSTD)')
        counts={t:c.execute(f'SELECT count(*) FROM {t}').fetchone()[0] for t in ('q','pool','parent_features')}
        c.close()
    save(output,config,dict(schema='sireto-v28-features-1',preflight=preflight is not None,sources=sources,counts=counts,
         truth_columns_opened=False,new_geographic_candidates=False,raw_coordinates_persisted=False,resources=budget.report()))


def training_sample(frame,folds):
    if not folds or not set(folds)<={2,3,4}:raise ValueError('Invalid training folds')
    train=frame[frame.fold.isin(folds)].sort_values(['query_id','legacy_parent_rank','siren'])
    indices=[];sizes=[];queries=[]
    for query,g in train.groupby('query_id',sort=True):
        pos=g.index[g.siren.eq(g.gt_siren)].to_numpy();neg=g.index[~g.siren.eq(g.gt_siren)].to_numpy()
        if len(pos)>1:raise ValueError('Duplicate positive parent')
        if len(pos)!=1 or not len(neg):continue
        chosen=np.concatenate([pos,neg[negative_positions(len(neg),'HEAD256_PLUS_STRATIFIED_TAIL256')]])
        indices.extend(chosen);sizes.append(len(chosen));queries.append(str(query))
    if not sizes:raise ValueError('STOP_NO_NATURAL_TRAINING_GROUP')
    sampled=frame.loc[indices]
    return sampled[list(FEATURES)].to_numpy(dtype=np.float32),sampled.siren.eq(sampled.gt_siren).to_numpy(dtype=np.int8),np.array(sizes,dtype=np.uint32),queries


def summary(d):
    return dict(queries=len(d),baseline_parent_hits=int(d.baseline_parent_hit.sum()),oof_parent_hits=int(d.oof_parent_hit.sum()),
                pool_parent_ceiling=int(d.pool_parent_hit.sum()),parent_gains=int((d.oof_parent_hit & ~d.baseline_parent_hit).sum()),
                parent_losses=int((~d.oof_parent_hit & d.baseline_parent_hit).sum()),
                new_vs_multichannel_missing=int((d.oof_parent_hit & ~d.parent_in_evidence).sum()),
                baseline_expansion_exact=int(d.baseline_expansion_exact.sum()),oof_expansion_exact=int(d.oof_expansion_exact.sum()),
                baseline_expansion_operational=int(d.baseline_expansion_operational.sum()),oof_expansion_operational=int(d.oof_expansion_operational.sum()),
                baseline_master_exact_unchanged=int(d.exact_hit.sum()),baseline_master_operational_unchanged=int(d.operational_hit.sum()))


def fit_evaluate(config,features,output):
    cfg=config_at(config);fm,ff=manifest_files(record(Path(features)/'manifest.json'))
    if (fm['schema']!='sireto-v28-features-1' or fm['preflight'] or fm['truth_columns_opened'] or fm['counts']['q']!=1034
        or fm['config']!=record(config) or fm['implementation']!=implementation() or fm['versions']!=versions()):raise ValueError('Feature authority')
    tm,tf=manifest_files(cfg['truth_audit']);side=verify(cfg['official_loss_sidecar']);op=verify(cfg['operational_sidecar'])
    model_config=json.loads(verify(cfg['model_config']).read_text())['model']
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    with Budget(cfg['resources']) as budget:
        truth=pq.read_table(tf['result.parquet'],columns=['query_id','fold','gt_siren']).to_pandas();validate_fold_isolation(truth)
        queries=pq.read_table(ff['queries.parquet'],columns=['query_id','fold']).to_pandas()
        if set(map(tuple,queries.to_numpy()))!=set(map(tuple,truth[['query_id','fold']].to_numpy())):raise ValueError('Query population mismatch')
        features_frame=pq.read_table(ff['parent_features.parquet']).to_pandas()
        frame=features_frame.merge(truth,on=['query_id','fold'],validate='many_to_one')
        if len(frame)!=len(features_frame):raise ValueError('Unlabelled feature query')
        del features_frame
        if not len(frame) or not np.isfinite(frame[list(FEATURES)].to_numpy(dtype=np.float32)).all():raise ValueError('Invalid features')
        if frame[['parent_name_jw','site_name_jw','street','number_exact']].nunique().max()<=1:raise ValueError('STOP_NO_FEATURE_VARIATION')
        records=[]
        for fold in (2,3,4):
            train_folds=[f for f in (2,3,4) if f!=fold];target=frame[frame.fold.eq(fold)]
            if set(truth.loc[truth.fold.isin(train_folds),'gt_siren']) & set(truth.loc[truth.fold.eq(fold),'gt_siren']):raise ValueError('Shared human SIREN')
            X,y,groups,queries=training_sample(frame,train_folds)
            model=xgb.XGBRanker(**model_config);model.fit(X,y,group=groups,verbose=False)
            model.save_model(output/f'outer{fold}.json')
            pred=target[['query_id','fold','siren','legacy_parent_rank']].copy()
            pred['score']=model.predict(target[list(FEATURES)].to_numpy(dtype=np.float32));assert np.isfinite(pred.score).all()
            pq.write_table(pa.Table.from_pandas(pred,preserve_index=False),output/f'outer{fold}_scores.parquet',compression='zstd')
            rec=dict(fold=fold,train_folds=train_folds,target_folds=[fold],trained_query_ids=queries,
                     target_query_ids=sorted(truth.loc[truth.fold.eq(fold),'query_id']),training_rows=len(y),natural_positive_groups=len(groups),
                     nontrainable_queries=int(truth.fold.isin(train_folds).sum()-len(groups)),target_parent_rows=len(pred))
            write_json(output/f'outer{fold}_split.json',rec);records.append({k:v for k,v in rec.items() if not k.endswith('query_ids')})
            print(json.dumps(records[-1]),flush=True);del X,y,model,target,pred;gc.collect()
        del frame;gc.collect()
        c=open_workbench_db(output)
        c.execute('CREATE TEMP TABLE scores AS SELECT * FROM read_parquet(?)',[[str(output/f'outer{f}_scores.parquet') for f in (2,3,4)]])
        c.execute('''CREATE TEMP TABLE selected AS SELECT * FROM (SELECT *,row_number() OVER(PARTITION BY query_id
          ORDER BY score DESC,legacy_parent_rank,siren) selected_rank FROM scores) WHERE selected_rank<=512''')
        assert c.execute('SELECT count(*) FROM (SELECT query_id,count(*) n,count(DISTINCT siren) nd FROM selected GROUP BY query_id HAVING n>512 OR n<>nd)').fetchone()[0]==0
        for table,path in [('t',tf['result.parquet']),('pool',ff['pool.parquet']),('old',ff['baseline_parents.parquet']),('op',op),('official',side)]:
            c.execute(f'CREATE TEMP TABLE {table} AS SELECT * FROM read_parquet(?)',[str(path)])
        assert c.execute('SELECT count(*) FROM t ANTI JOIN official USING(query_id)').fetchone()[0]==0
        assert c.execute('SELECT count(*) FROM t ANTI JOIN op ON t.query_id=op.query_id AND t.gt_siret=op.siret').fetchone()[0]==0
        c.execute('''CREATE TEMP TABLE old_sites AS SELECT p.* FROM pool p JOIN old o USING(query_id,siren)''')
        c.execute('''CREATE TEMP TABLE new_sites AS SELECT p.* FROM pool p JOIN selected s USING(query_id,siren)''')
        c.execute('''CREATE TEMP TABLE outcomes AS SELECT t.*,
          EXISTS(SELECT 1 FROM old WHERE old.query_id=t.query_id AND old.siren=t.gt_siren) baseline_parent_hit,
          EXISTS(SELECT 1 FROM selected s WHERE s.query_id=t.query_id AND s.siren=t.gt_siren) oof_parent_hit,
          EXISTS(SELECT 1 FROM pool p WHERE p.query_id=t.query_id AND p.siren=t.gt_siren) pool_parent_hit,
          EXISTS(SELECT 1 FROM old_sites p WHERE p.query_id=t.query_id AND p.siret=t.gt_siret) baseline_expansion_exact,
          EXISTS(SELECT 1 FROM new_sites p WHERE p.query_id=t.query_id AND p.siret=t.gt_siret) oof_expansion_exact,
          EXISTS(SELECT 1 FROM old_sites p JOIN op USING(query_id,siret) WHERE p.query_id=t.query_id) baseline_expansion_operational,
          EXISTS(SELECT 1 FROM new_sites p JOIN op USING(query_id,siret) WHERE p.query_id=t.query_id) oof_expansion_operational
          FROM t''')
        df=c.execute('SELECT * FROM outcomes ORDER BY query_id').df();assert len(df)==1034
        for table in ('selected','outcomes'):
            c.execute(f'COPY {table} TO {sql_path(output/(table+".parquet"))} (FORMAT PARQUET,COMPRESSION ZSTD)')
        seg={dim:{str(k):summary(v) for k,v in df.groupby(dim,dropna=False)} for dim in
             ('fold','cohort_kind','qualification_v2','qualification_v3','data_origin','label_kind','ground_truth_state','loss_stage')}
        primary=summary(df);controls=df[df.cohort_kind.eq('BASE_TOP1750_CONTROL')]
        eligible=primary['parent_gains']>primary['parent_losses'] and not ((~controls.oof_parent_hit)&controls.baseline_parent_hit).any()
        c.close()
    save(output,config,dict(schema='sireto-v28-oof-evaluation-1',decision='PIVOT_GENERATION',pilot_status='ELIGIBLE_FOR_SEPARATE_END_TO_END_TEST' if eligible else 'REJECT_NO_RETUNING',
        features_manifest=record(Path(features)/'manifest.json'),truth_authority=cfg['truth_audit'],official_loss_sidecar=record(side),operational_sidecar=record(op),
        records=records,model_parameters=model_config,primary=primary,segments=seg,focus=summary(df[df.focus_missing_near500]),
        recall100_measured=False,master2000_measured=False,rich_hydration=False,positive_injection=False,target_labels_used_for_selection=False,
        raw_coordinates_persisted=False,resources=budget.report()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['generate','fit-evaluate']);p.add_argument('--config',type=Path,default=Path('config/retrieval_v28_geo_selection.json'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--features',type=Path);p.add_argument('--preflight',type=int)
    a=p.parse_args()
    if a.action=='generate':generate(a.config,a.output,a.preflight)
    else:
        if not a.features or a.preflight:p.error('Full features required')
        fit_evaluate(a.config,a.features,a.output)
