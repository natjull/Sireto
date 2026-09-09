#!/usr/bin/env python3
"""Feature generation, fixed OOF parent training, and separate final evaluation."""
import argparse
import gc
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb
import duckdb

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.xgb_matcher.retrieval_v20_workbench import Budget, record, verify, write_json, sql_path, negative_positions
from src.xgb_matcher.retrieval_v21_membership import manifest_files, open_workbench_db
from src.xgb_matcher.retrieval_v23_joint_parent import FEATURES, NAME_SELECTION_SQL, config_at, code_records, build_features, make_master
from src.xgb_matcher.hierarchical_retrieval import normalize_text
from src.xgb_matcher.retrieval_v12_scene_ir import SceneIRCache
from src.xgb_matcher.retrieval_parent_fusion_v19 import validate_fold_isolation
from scripts.retrieval_v22_global_budget import outcomes, summary as old_summary


def runtime_versions():
    return {'xgboost':xgb.__version__,'duckdb':duckdb.__version__,'numpy':np.__version__,
            'pandas':pd.__version__,'pyarrow':pa.__version__}


def save_manifest(output, result):
    result['implementation']=code_records()
    result['runtime_versions']=runtime_versions()
    result['files']={p.name:record(p) for p in output.iterdir() if p.is_file()}
    write_json(output/'manifest.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('implementation','files')}),flush=True)


def generate(config_path,output,preflight=None):
    config=config_at(config_path);gm,gf=manifest_files(config['source_generation'])
    if gm['schema']!='sireto-v21-generation-1' or gm['preflight'] or gm['truth_columns_opened']:raise ValueError('Pool authority')
    v18=json.loads(verify(config['v18_config']).read_text())['generation_authorities']
    raw=verify(v18['raw_union']);names=verify(config['names']);sites=verify(config['sites'])
    scene_root=Path(v18['scene_ir']['path']);verify({'path':str(scene_root/'manifest.json'),'sha256':v18['scene_ir']['manifest_sha256']})
    if preflight is not None and not 1<=preflight<=16:raise ValueError('Preflight 1..16')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    with Budget(config['resources']) as budget:
        c=open_workbench_db(output)
        q=pq.read_table(gf['queries.parquet']).to_pandas().sort_values('query_id').head(preflight or 1034)
        if len(q)!=(preflight or 1034) or q.query_id.duplicated().any():raise ValueError('Query population')
        cache=SceneIRCache(scene_root,verify_source_hash=False)
        if cache.source.resolve()!=raw.resolve():raise ValueError('Raw/SceneIR mismatch')
        extracted=[]
        for scene in cache.iter_selected_scenes(dict(zip(q.query_id,q.fold)),columns=['query_id','fold','crm_name']):
            values=set(scene.rows['crm_name'].to_pylist())
            if len(values)!=1:raise ValueError('CRM name not query-constant')
            extracted.append({'query_id':scene.query_id,'fold':scene.fold,'crm_name':normalize_text(next(iter(values)))})
        q=q.merge(pd.DataFrame(extracted),on=['query_id','fold'],validate='one_to_one')
        if len(q)!=(preflight or 1034):raise ValueError('Name extraction lost a query')
        c.register('q',pa.Table.from_pandas(q,preserve_index=False))
        c.execute('''CREATE TEMP TABLE pool AS SELECT p.query_id,p.fold,p.siren,p.siret,p.base_rank,
          p.parent_rrf,p.direct_rrf FROM read_parquet(?) p SEMI JOIN q USING(query_id,fold)''',[str(gf['candidates.parquet'])])
        c.execute(f'CREATE TEMP VIEW name_source AS SELECT * FROM read_parquet({sql_path(names)})')
        c.execute(NAME_SELECTION_SQL)
        c.execute('''CREATE TEMP TABLE sites AS SELECT s.siren,s.siret,s.street_normalized,s.street_number,
          s.street_number_suffix,s.insee,s.postcode,s.administrative_state FROM read_parquet(?) s
          SEMI JOIN (SELECT DISTINCT siren,siret FROM pool) p USING(siren,siret)''',[str(sites)])
        build_features(c)
        for filename,table,order in [('queries','q','query_id'),('parent_features','parent_features','query_id,siren'),('site_features','site_features','query_id,siret')]:
            c.execute(f'COPY (SELECT * FROM {table} ORDER BY {order}) TO {sql_path(output/(filename+".parquet"))} (FORMAT PARQUET,COMPRESSION ZSTD)')
        counts={t:c.execute(f'SELECT count(*) FROM {t}').fetchone()[0] for t in ('q','pool','names','parent_features','site_features')}
        c.close()
    save_manifest(output,{'schema':'sireto-v23-features-1','config':record(config_path),'source_generation':config['source_generation'],
      'sources':{str(p):record(p) for p in (raw,names,sites,scene_root/'manifest.json')},'feature_order':list(FEATURES),
      'preflight':preflight is not None,'counts':counts,'truth_columns_opened':False,'fold0_opened':False,'fold1_opened':False,'resources':budget.report()})


def training_sample(frame,train_folds):
    if not train_folds or not set(train_folds)<=set((2,3,4)):raise ValueError('Forbidden training folds')
    train=frame.loc[frame.fold.isin(train_folds)].sort_values(['query_id','parent_rrf','parent_name_jw','siren'],ascending=[True,False,False,True])
    indices=[];sizes=[];groups=[]
    for query,group in train.groupby('query_id',sort=True):
        pos=group.index[group.siren.eq(group.gt_siren)].to_numpy();neg=group.index[~group.siren.eq(group.gt_siren)].to_numpy()
        if len(pos)>1:raise ValueError('Duplicate positive parent')
        if len(pos)!=1 or len(neg)==0:continue
        chosen=np.concatenate([pos,neg[negative_positions(len(neg),'HEAD256_PLUS_STRATIFIED_TAIL256')]])
        indices.extend(chosen);sizes.append(len(chosen));groups.append(str(query))
    if not sizes:raise ValueError('No trainable natural groups')
    sampled=frame.loc[indices]
    return sampled[list(FEATURES)].to_numpy(dtype=np.float32),sampled.siren.eq(sampled.gt_siren).to_numpy(dtype=np.int8),np.array(sizes,dtype=np.uint32),groups


def fit(config_path,features,output):
    config=config_at(config_path);fm,ff=manifest_files(record(Path(features)/'manifest.json'))
    if (fm['schema']!='sireto-v23-features-1' or fm['preflight'] or fm['truth_columns_opened'] or fm['counts']['q']!=1034
        or fm['config']['sha256']!=record(config_path)['sha256'] or fm['feature_order']!=list(FEATURES)
        or fm['implementation']!=code_records() or fm['runtime_versions']!=runtime_versions()):raise ValueError('Feature authority')
    bm,bf=manifest_files(config['baseline_evaluation'])
    if bm['schema']!='sireto-v22-evaluation-1':raise ValueError('Training truth authority')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    with Budget(config['resources']) as budget:
        truth=pq.read_table(bf['outcomes.parquet'],columns=['query_id','fold','gt_siren']).to_pandas()
        validate_fold_isolation(truth)
        frame=pq.read_table(ff['parent_features.parquet']).to_pandas().merge(truth,on=['query_id','fold'],validate='many_to_one')
        if set(frame.query_id)!=set(truth.query_id):raise ValueError('Training query population')
        outer_predictions=[];records=[]
        jobs=[(f'outer{f}',[x for x in (2,3,4) if x!=f],[f],True) for f in (2,3,4)]
        # These fixed single-fold fits provide inner-OOF scores for the training
        # folds of each outer model. They never choose a threshold or a model.
        if config['single_fold_inner_models']:
            jobs += [(f'inner_train{f}',[f],[x for x in (2,3,4) if x!=f],False) for f in (2,3,4)]
        for name,train_folds,target_folds,is_outer in jobs:
            if set(train_folds)&set(target_folds):raise ValueError('OOF contamination')
            X,y,sizes,groups=training_sample(frame,train_folds)
            target=frame.loc[frame.fold.isin(target_folds)]
            if set(frame.loc[frame.fold.isin(train_folds),'gt_siren']) & set(target.gt_siren):raise ValueError('Shared target SIREN')
            model=xgb.XGBRanker(**config['model']);model.fit(X,y,group=sizes,verbose=False)
            model.save_model(output/(name+'.json'))
            pred=target[['query_id','fold','siren']].copy();pred['parent_score']=model.predict(target[list(FEATURES)].to_numpy(dtype=np.float32))
            if not np.isfinite(pred.parent_score).all():raise ValueError('Nonfinite prediction')
            pq.write_table(pa.Table.from_pandas(pred,preserve_index=False),output/(name+'_scores.parquet'),compression='zstd')
            rec={'model':name,'train_folds':train_folds,'target_folds':target_folds,'natural_positive_groups':len(groups),'training_rows':len(y),
              'skipped_no_positive_or_no_negative':int(frame.loc[frame.fold.isin(train_folds),'query_id'].nunique()-len(groups)),
              'target_queries':int(target.query_id.nunique()),'target_parent_rows':len(pred),'features':list(FEATURES)}
            write_json(output/(name+'_split.json'),{**rec,'trained_query_ids':groups,'target_query_ids':sorted(set(target.query_id))})
            records.append(rec);print(json.dumps(rec),flush=True)
            if is_outer:outer_predictions.append(pred)
            del model,X,y,target;gc.collect()
        del frame;gc.collect()
        c=open_workbench_db(output)
        c.execute('CREATE TEMP TABLE site_features AS SELECT * FROM read_parquet(?)',[str(ff['site_features.parquet'])])
        scores=pd.concat(outer_predictions,ignore_index=True)
        if scores.duplicated(['query_id','siren']).any():raise ValueError('Duplicate OOF prediction')
        c.register('parent_scores',pa.Table.from_pandas(scores,preserve_index=False));make_master(c)
        c.execute(f'COPY (SELECT * FROM master ORDER BY query_id,candidate_rank) TO {sql_path(output/"master.parquet")} (FORMAT PARQUET,COMPRESSION ZSTD)')
        counts={'queries':len(truth),'parents_scored_oof':len(scores),'master':c.execute('SELECT count(*) FROM master').fetchone()[0]}
        c.close()
    save_manifest(output,{'schema':'sireto-v23-oof-master-1','config':record(config_path),'features_manifest':record(Path(features)/'manifest.json'),
      'training_truth_authority':config['baseline_evaluation'],'records':records,'counts':counts,'target_labels_used_for_selection':False,
      'fold0_opened':False,'fold1_opened':False,'positive_injection':False,'site_lambda_scoring_performed':False,'resources':budget.report()})


def summarize(df):
    r=old_summary(df);r['gains_vs_v22']=r.pop('gains_vs_v21');r['losses_vs_v22']=r.pop('losses_vs_v21');return r


def evaluate(config_path,generation,output):
    config=config_at(config_path);gm,gf=manifest_files(record(Path(generation)/'manifest.json'))
    if (gm['schema']!='sireto-v23-oof-master-1' or gm['counts']['queries']!=1034 or gm['target_labels_used_for_selection']
        or gm['config']['sha256']!=record(config_path)['sha256'] or gm['implementation']!=code_records()
        or gm['runtime_versions']!=runtime_versions()):raise ValueError('OOF master authority')
    fm,ff=manifest_files(gm['features_manifest']);bm,bf=manifest_files(config['baseline_evaluation'])
    if gm['training_truth_authority']!=config['baseline_evaluation'] or bm['schema']!='sireto-v22-evaluation-1':raise ValueError('Truth drift')
    side=verify(bm['official_loss_sidecar']);acceptable=verify(bm['acceptable_operational_sidecar'])
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    with Budget(config['resources']) as budget:
        c=open_workbench_db(output)
        for name,path in [('q',ff['queries.parquet']),('master',gf['master.parquet']),('official_loss',side),('acceptable',acceptable)]:
            c.execute(f'CREATE TEMP TABLE {name} AS SELECT * FROM read_parquet(?)',[str(path)])
        c.execute('''CREATE TEMP TABLE t AS SELECT query_id,fold,gt_siret,gt_siren,cohort_kind,
          qualification_v2,qualification_v3,label_kind,data_origin,ground_truth_state,site_in_reservoir,
          exact_hit previous_exact_hit FROM read_parquet(?)''',[str(bf['outcomes.parquet'])])
        df=outcomes(c)
        if len(df)!=1034:raise ValueError('Evaluation population')
        cap=c.execute('SELECT max(n) FROM (SELECT query_id,count(*) n FROM master GROUP BY 1)').fetchone()[0]
        if cap>2000:raise ValueError('Master cap')
        c.execute(f'COPY outcomes TO {sql_path(output/"outcomes.parquet")} (FORMAT PARQUET,COMPRESSION ZSTD)')
        segments={d:{str(k):summarize(v) for k,v in df.fillna({d:'UNKNOWN'}).groupby(d,dropna=False)} for d in ('fold','cohort_kind','data_origin','qualification_v2','qualification_v3','label_kind','ground_truth_state')}
        c.close()
    save_manifest(output,{'schema':'sireto-v23-evaluation-1','decision':'PIVOT_GENERATION','config':record(config_path),
      'generation_manifest':record(Path(generation)/'manifest.json'),'baseline_evaluation':config['baseline_evaluation'],
      'official_loss_sidecar':record(side),'acceptable_operational_sidecar':record(acceptable),'primary':summarize(df),
      'historical_all_human':summarize(df),'segments':segments,'max_master_sirets':int(cap),
      'loss_counts':{str(k):int(v) for k,v in df.loss_stage.value_counts().items()},'national_projection':None,
      'unseen_external_validation':False,'reused_development_panel':True,'prerank500_measured':False,'recall100_measured':False,
      'fold0_opened':False,'fold1_opened':False,'ready_for_fold1':False,'resources':budget.report()})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['generate','fit','evaluate']);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--features',type=Path);p.add_argument('--generation',type=Path);p.add_argument('--preflight',type=int)
    a=p.parse_args()
    if a.action=='generate':generate(a.config,a.output,a.preflight)
    elif a.action=='fit':
        if not a.features or a.preflight:p.error('fit requires full feature generation')
        fit(a.config,a.features,a.output)
    else:
        if not a.generation or a.preflight:p.error('evaluate requires full OOF generation')
        evaluate(a.config,a.generation,a.output)
