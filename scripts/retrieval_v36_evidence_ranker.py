"""One preregistered official-evidence/full-list ranking experiment plus ablation."""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb

from scripts import retrieval_v35_site_fusion as v35
from src.xgb_matcher.retrieval_v20_workbench import Budget, ROOT, record, verify, write_json, sql_path

BASE = v35.DATA/'audits/retrieval_v35_site_fusion/pilot_20261001_a'
BASE_SHA = '8000f1f8851d28af788708007b14c7fc2e2e5735d7fa8bbd34f46c27ca37112a'
CHANNELS = ('bodacc_asset_transfer','rne_subject_to_related','sirene_economic_continuity',
            'siret_address_char','siret_address_word','siret_current_name_char','siret_fielded_name_bm25',
            'siret_historical_address_word','siret_historical_name_word','siret_site_name_word',
            'siret_supporting_address_word','siret_supporting_name_word','siret_trade_name_word',
            'siret_joint_name_address_char')
NAME_GROUPS = ('parent_current','parent_support','site_current','site_history')
NAME_FEATURES = tuple(f'{g}_{m}' for g in NAME_GROUPS for m in ('jw','tokens','coverage','available'))
GEO_FEATURES = ('xy_available','xy_distance_log','xy_lower_log','xy_upper_log','xy_uncertainty_ratio',
                'parent_xy_available','parent_xy_min_log','xy_gap_to_parent_min_log')
CHANNEL_FEATURES = tuple(f'{level}_{channel}_{stat}' for level in ('parent','site')
                         for channel in CHANNELS for stat in ('rank_rrf','relative_score'))
EXTRA = NAME_FEATURES + GEO_FEATURES + CHANNEL_FEATURES
FEATURES = v35.FEATURES + EXTRA
PARAMS = dict(v35.PARAMS, lambdarank_pair_method='topk', lambdarank_num_pair_per_sample=100)
RESOURCE = dict(max_seconds=1200, max_rss_bytes=6*1024**3, max_swapout_pages_delta=0)


def project_sources(c, sources, source_shards, output):
    paths = [s['files']['v30_stable/pool.parquet']['path'] for s in source_shards]
    c.execute('CREATE TEMP TABLE keys AS SELECT DISTINCT siren,siret FROM read_parquet(?)',[paths])
    verify(sources['stock']); verify(sources['names'])
    c.execute('''CREATE TEMP TABLE xy AS SELECT s.siren,s.siret,
      try_cast(s.coordonneeLambertAbscisseEtablissement AS DOUBLE) x,
      try_cast(s.coordonneeLambertOrdonneeEtablissement AS DOUBLE) y
      FROM read_parquet(?) s SEMI JOIN keys k USING(siret)''',[sources['stock']['path']])
    c.execute('''CREATE TEMP TABLE all_names AS SELECT DISTINCT n.siren,n.siret,
      CASE WHEN n.subject_kind='SIREN' AND n.name_role IN ('LEGAL_CURRENT','TRADE_CURRENT') THEN 'parent_current'
        WHEN n.subject_kind='SIREN' AND n.name_role='SUPPORTING' THEN 'parent_support'
        WHEN n.subject_kind='SIRET' AND n.name_role='SITE_CURRENT' THEN 'site_current'
        WHEN n.subject_kind='SIRET' AND n.name_role='HISTORICAL' THEN 'site_history' END name_group,
      trim(regexp_replace(upper(strip_accents(n.normalized_value)),'[^A-Z0-9]+',' ','g')) AS name
      FROM read_parquet(?) n WHERE n.selected_for_subject AND n.normalized_value<>'' AND
      ((n.subject_kind='SIREN' AND n.name_role IN ('LEGAL_CURRENT','TRADE_CURRENT','SUPPORTING') AND n.siren IN (SELECT siren FROM keys))
       OR (n.subject_kind='SIRET' AND n.name_role IN ('SITE_CURRENT','HISTORICAL') AND n.siret IN (SELECT siret FROM keys)))''',[sources['names']['path']])
    result = {}
    for table in ('xy','all_names'):
        dest = output/(table+'.parquet')
        c.execute(f'COPY {table} TO {sql_path(dest)} (FORMAT PARQUET,COMPRESSION ZSTD)')
        result[table] = record(dest)
    return result


def extra_evidence(c, source, cache):
    paths = {k:r['path'] for k,r in source['files'].items()}
    c.execute('CREATE TEMP TABLE q AS SELECT * FROM read_parquet(?)',[paths['queries.parquet']])
    c.execute('CREATE TEMP TABLE pool AS SELECT * FROM read_parquet(?)',[paths['v30_stable/pool.parquet']])
    c.execute('CREATE TEMP TABLE entities AS SELECT DISTINCT query_id,fold,siren FROM pool')
    c.execute('''CREATE TEMP TABLE names AS SELECT n.* FROM read_parquet(?) n WHERE
      (starts_with(name_group,'parent') AND siren IN (SELECT siren FROM pool)) OR
      (starts_with(name_group,'site') AND siret IN (SELECT siret FROM pool))''',[cache['all_names']['path']])
    c.execute('''CREATE MACRO jw(a,b) AS CASE WHEN coalesce(a,'')='' OR coalesce(b,'')='' THEN 0.
      ELSE jaro_winkler_similarity(a,b) END''')
    c.execute('''CREATE MACRO tok(a,b) AS CASE WHEN coalesce(a,'')='' OR coalesce(b,'')='' THEN 0.
      ELSE len(list_intersect(string_split(a,' '),string_split(b,' ')))::DOUBLE /
      len(list_distinct(list_concat(string_split(a,' '),string_split(b,' ')))) END''')
    c.execute('''CREATE MACRO cov(a,b) AS CASE WHEN coalesce(a,'')='' OR coalesce(b,'')='' THEN 0.
      ELSE len(list_intersect(string_split(a,' '),string_split(b,' ')))::DOUBLE /
      len(list_distinct(string_split(a,' '))) END''')
    for level in ('parent','site'):
        source_table = 'entities' if level=='parent' else 'pool'
        key = 'siren' if level=='parent' else 'siret'
        cols = 'p.query_id,p.fold,p.siren'+(',p.siret' if level=='site' else '')
        group_cols = cols+',n.name_group'
        c.execute(f'''CREATE TEMP TABLE {level}_names AS SELECT {group_cols},max(jw(q.crm_name,n.name)) jw,
          max(tok(q.crm_name,n.name)) tokens,max(cov(q.crm_name,n.name)) coverage,1.0 available
          FROM {source_table} p JOIN q USING(query_id,fold) JOIN names n ON p.{key}=n.{key}
          AND starts_with(n.name_group,'{level}') GROUP BY {group_cols}''')
    c.execute('''CREATE TEMP TABLE distances AS SELECT p.*,CASE WHEN b.ban_resolved
      AND isfinite(x.x) AND isfinite(x.y) AND x.x>0 AND x.y>0
      AND isfinite(b.ban_x_l93) AND isfinite(b.ban_y_l93) AND b.ban_x_l93>0 AND b.ban_y_l93>0
      THEN sqrt(pow(x.x-b.ban_x_l93,2)+pow(x.y-b.ban_y_l93,2)) END distance,
      greatest(coalesce(b.ban_uncertainty_m,0),0) uncertainty
      FROM pool p LEFT JOIN read_parquet(?) x USING(siren,siret)
      LEFT JOIN read_parquet(?) b USING(query_id,fold)''',[cache['xy']['path'],paths['ban.parquet']])
    c.execute('''CREATE TEMP TABLE geo AS SELECT *,min(distance) OVER(PARTITION BY query_id,siren) parent_distance FROM distances''')
    c.execute('''CREATE TEMP TABLE signals AS SELECT query_id,fold,siren,source_siret siret,channel,channel_rank,score
      FROM read_parquet(?) WHERE channel NOT IN ('legacy_site_pool','v17_union') UNION ALL
      SELECT query_id,fold,siren,siret,'siret_joint_name_address_char',evidence_rank,score FROM read_parquet(?)''',
      [[paths['direct.parquet'],paths['structured.parquet']],paths['joint.parquet']])
    observed={r[0] for r in c.execute('SELECT DISTINCT channel FROM signals').fetchall()}
    if not observed<=set(CHANNELS): raise ValueError('Unknown official channel')
    c.execute('''CREATE TEMP TABLE normalized_signals AS SELECT *,
      greatest(coalesce(score,0),0)/nullif(max(greatest(coalesce(score,0),0)) OVER(PARTITION BY query_id,channel),0) relative_score
      FROM signals''')
    channel_selects = {}
    for level in ('parent','site'):
        cols='query_id,fold,siren'+(',siret' if level=='site' else '')
        fields=[]
        for channel in CHANNELS:
            fields.extend([f"coalesce(max(1.0/(60+channel_rank)) FILTER(WHERE channel='{channel}'),0) {level}_{channel}_rank_rrf",
                           f"coalesce(max(relative_score) FILTER(WHERE channel='{channel}'),0) {level}_{channel}_relative_score"])
        c.execute(f'CREATE TEMP TABLE {level}_channels AS SELECT {cols},'+','.join(fields)+
                  ' FROM normalized_signals GROUP BY '+cols)
        channel_selects[level]=[f'coalesce({level[0]}c.{n},0) {n}' for n in CHANNEL_FEATURES if n.startswith(level+'_')]
    selections=[]; joins=[]
    for i,g in enumerate(NAME_GROUPS):
        table='parent_names' if g.startswith('parent') else 'site_names'
        keys='query_id,fold,siren'+(',siret' if g.startswith('site') else '')
        joins.append(f"LEFT JOIN (SELECT * FROM {table} WHERE name_group='{g}') n{i} USING({keys})")
        selections.extend([f'coalesce(n{i}.{stat},0) {g}_{stat}' for stat in ('jw','tokens','coverage','available')])
    geo_sql=['distance IS NOT NULL xy_available','coalesce(ln(1+distance),0) xy_distance_log',
             'coalesce(ln(1+greatest(distance-uncertainty,0)),0) xy_lower_log',
             'CASE WHEN distance IS NOT NULL THEN ln(1+distance+uncertainty) ELSE 0 END xy_upper_log',
             'coalesce(distance/greatest(uncertainty,1),0) xy_uncertainty_ratio',
             'parent_distance IS NOT NULL parent_xy_available','coalesce(ln(1+parent_distance),0) parent_xy_min_log',
             'coalesce(ln(1+greatest(distance-parent_distance,0)),0) xy_gap_to_parent_min_log']
    # Parent and site names share e.g. site_current_* only in the extra schema;
    # all selectors are numeric, identities stay outside the feature allowlist.
    c.execute('CREATE TEMP TABLE extra AS SELECT g.query_id,g.fold,g.siren,g.siret,'+
              ','.join(selections+geo_sql+channel_selects['parent']+channel_selects['site'])+
              ' FROM geo g '+' '.join(joins)+
              ' LEFT JOIN parent_channels pc USING(query_id,fold,siren) LEFT JOIN site_channels sc USING(query_id,fold,siren,siret)')
    if c.execute('SELECT count(*) FROM extra').fetchone()!=c.execute('SELECT count(*) FROM pool').fetchone():
        raise ValueError('Extra evidence join multiplied population')


def enrich(source, base_shard, cache, output):
    output.mkdir()
    c=v35.db(output)
    try:
        extra_evidence(c,source,cache)
        files={}
        for name,ref in base_shard['files'].items():
            original=verify(ref)
            additional=','.join(f'CAST(e.{f} AS FLOAT) {f}' for f in EXTRA)
            frame=c.execute('SELECT b.*,'+additional+' FROM read_parquet(?) b JOIN extra e USING(query_id,fold,siren,siret) ORDER BY query_id,candidate_rank',[str(original)]).fetch_arrow_table()
            if frame.num_rows!=pq.read_metadata(original).num_rows:raise ValueError('Context Master changed')
            array=np.column_stack([frame[f].to_numpy() for f in EXTRA])
            if not np.isfinite(array).all():raise ValueError('Nonfinite extra evidence')
            pq.write_table(frame,output/name,compression='zstd');files[name]=record(output/name)
        manifest=dict(status='SEALED_TRUTH_BLIND_ENRICHED_CONTEXTS',files=files,features=list(FEATURES),base=base_shard,
                      membership_unchanged=True,truth_opened=False)
        write_json(output/'manifest.json',manifest)
        return manifest
    finally:
        c.close();gc.collect()


def full_training(frame,labels,neutral):
    parts=[];stats=dict(queries=0,trainable=0,absent_positive=0,neutralized=0,rows=0)
    for qid,g in frame.groupby('query_id',sort=False):
        stats['queries']+=1
        target=(g.siret==labels[qid])
        if target.sum()!=1:
            if target.sum()>1:raise ValueError('Duplicate truth')
            stats['absent_positive']+=1;continue
        keep=target | ~g.siret.isin(neutral.get(qid,set()))
        stats['neutralized']+=int((~keep).sum());stats['trainable']+=1
        g=g.loc[keep].copy();g['target']=target.loc[keep].astype(np.float32);parts.append(g)
    result=pd.concat(parts,ignore_index=True) if parts else frame.iloc[:0].assign(target=pd.Series(dtype='float32'))
    stats['rows']=len(result)
    return result,stats


class Progress(xgb.callback.TrainingCallback):
    def __init__(self,variant,fold):self.variant=variant;self.fold=fold;self.start=time.monotonic()
    def after_iteration(self,model,epoch,evals_log):
        if epoch%50==0:print(json.dumps(dict(stage='fit',variant=self.variant,fold=self.fold,trees=epoch+1,seconds=time.monotonic()-self.start)),flush=True)
        return False


def evaluate(shards,scope,output):
    prereg=json.loads((output/'preregistration.json').read_text())
    for ref in prereg['implementation'].values():verify(ref)
    evaluator=v35.load_manifest(v35.EVAL,v35.EVAL_SHA)
    for ref in evaluator['files'].values():verify(ref)
    verify(dict(path=v35.LEDGER,sha256=v35.LEDGER_SHA))
    ids={q for q,f in scope};folds=dict(scope)
    baseline={r['query_id']:r for r in pq.read_table(evaluator['files']['v30_stable_outcomes.parquet']['path'],filters=[('fold','in',[2,3,4])]).to_pylist() if r['query_id'] in ids}
    prior={r['query_id']:r for r in pq.read_table(verify(v35.load_manifest(BASE/'manifest.json',BASE_SHA)['outcomes'])).to_pylist()}
    ledger={r['query_id']:r for r in pq.read_table(v35.LEDGER,filters=[('fold','in',[2,3,4])],columns=['query_id','fold','human_ground_truth_siret']).to_pylist() if r['query_id'] in ids}
    if set(ledger)!=ids or set(baseline)!=ids or set(prior)!=ids:raise ValueError('Denominator drift')
    if any(ledger[q]['fold']!=f or baseline[q]['fold']!=f for q,f in scope):raise ValueError('Fold drift')
    labels={q:r['human_ground_truth_siret'] for q,r in ledger.items()}
    if any(baseline[q]['gt_siret']!=labels[q] for q in ids):raise ValueError('Truth mismatch')
    acceptable={};neutral={};active={}
    for r in pq.read_table(evaluator['files']['acceptable_operational.parquet']['path'],filters=[('fold','in',[2,3,4])]).to_pylist():
        q=r['query_id']
        if q not in ids:continue
        acceptable.setdefault(q,set()).add(r['siret'])
        if not r['exact_truth']:
            neutral.setdefault(q,set()).add(r['siret'])
            if r['administrative_state']=='A':active.setdefault(q,set()).add(r['siret'])
    results={}
    for variant,features in [('primary',FEATURES),('ablation_training_only',v35.FEATURES)]:
        dest=output/variant;dest.mkdir();outcomes=[];models=[]
        for outer in (2,3,4):
            t=time.monotonic();parts=[];stats=[]
            for shard in shards:
                for qfold in (f for f in (2,3,4) if f!=outer):
                    inner=v35.train_context(outer,qfold)
                    columns=['query_id','fold','siret','candidate_rank']+list(features)
                    frame=pq.read_table(verify(shard['files'][f'inner{inner}.parquet']),columns=columns,filters=[('fold','=',qfold)]).to_pandas()
                    sample,st=full_training(frame,labels,neutral);parts.append(sample);stats.append(st)
            train=pd.concat(parts,ignore_index=True).sort_values(['query_id','candidate_rank','siret']);del parts;gc.collect()
            if any(f==outer for f in train.fold):raise ValueError('Outer fold in fit')
            values=train[list(features)].to_numpy(dtype=np.float32)
            target=train.target.to_numpy(dtype=np.float32);groups=train.groupby('query_id',sort=False).size().to_numpy()
            trainids=sorted(set(train.query_id))
            if not np.isfinite(values).all() or len(trainids)!=len(groups):raise ValueError('Invalid grouped features')
            del train;gc.collect()
            model=xgb.XGBRanker(**PARAMS,callbacks=[Progress(variant,outer)])
            model.fit(values,target,group=groups,verbose=False)
            model.save_model(dest/f'outer{outer}.json');del values,target,groups;gc.collect()
            for i,shard in enumerate(shards):
                frame=pq.read_table(verify(shard['files']['outer.parquet']),filters=[('fold','=',outer)]).to_pandas()
                if not len(frame):continue
                scores=model.predict(frame[list(features)].to_numpy(dtype=np.float32))
                if not np.isfinite(scores).all():raise ValueError('Nonfinite score')
                frame['score']=scores;frame=frame.sort_values(['query_id','score','siret'],ascending=[True,False,True])
                frame['site_rank']=frame.groupby('query_id').cumcount()+1
                pq.write_table(pa.Table.from_pandas(frame[['query_id','fold','siren','siret','score','site_rank']],preserve_index=False),dest/f'scores_{outer}_{i:03d}.parquet',compression='zstd')
                for qid,g in frame.groupby('query_id',sort=False):
                    b=baseline[qid];ranks=g.loc[g.siret==labels[qid],'site_rank'].tolist();rank=ranks[0] if ranks else None
                    op=g.loc[g.siret.isin(acceptable[qid]),'site_rank'].tolist()
                    if bool(rank)!=bool(b['exact_rank']) or len(g)>2000 or g.siret.duplicated().any():raise ValueError('Master drift')
                    loss=('HIT' if rank is not None and rank<=100 else 'EVICTED_500_TO_100' if rank is not None and rank<=500
                          else 'EVICTED_MASTER_TO_500' if rank is not None else b['loss_stage'])
                    outcomes.append(dict(query_id=qid,fold=outer,exact_rank=rank,operational_rank=min(op) if op else None,
                        baseline_exact_rank=prior[qid]['exact_rank'],heuristic_exact_rank=b['exact_rank'],
                        label_kind=b['label_kind'],qualification_v2=b['qualification_v2'],qualification_v3=b['qualification_v3'],
                        loss_stage=loss,master_count=len(g),pre500_count=min(len(g),500),top100_count=min(len(g),100),
                        parent_in_top100=bool(((g.siren==b['gt_siren']) & (g.site_rank<=100)).any()),
                        active_equivalent_when_exact_closed=bool(b['exact_state']=='F' and ((g.siret.isin(active.get(qid,set()))) & (g.site_rank<=100)).any()),
                        multiple_active_equivalents=b['multiple_active_equivalents']))
            models.append(dict(outer=outer,trained_query_ids=trainids,model=record(dest/f'outer{outer}.json'),seconds=time.monotonic()-t,
                               sampling={k:sum(s[k] for s in stats) for k in stats[0]}))
            print(json.dumps(dict(stage='outer_complete',variant=variant,fold=outer,seconds=time.monotonic()-t)),flush=True)
            del model;gc.collect()
        if len(outcomes)!=len(ids) or {r['query_id'] for r in outcomes}!=ids:raise ValueError('Incomplete OOF')
        pq.write_table(pa.Table.from_pylist(outcomes),dest/'outcomes.parquet',compression='zstd')
        segments={col:{str(v):v35.summarize([r for r in outcomes if r[col]==v]) for v in sorted({r[col] for r in outcomes},key=str)}
                  for col in ('fold','label_kind','qualification_v2','qualification_v3')}
        quality=v35.summarize(outcomes)
        result=dict(quality=quality,segments=segments,models=models,outcomes=record(dest/'outcomes.parquet'),
                    losses={s:sum(r['loss_stage']==s for r in outcomes) for s in sorted({r['loss_stage'] for r in outcomes})},
                    parent_hits100=sum(r['parent_in_top100'] for r in outcomes),
                    active_equivalent_when_exact_closed=sum(r['active_equivalent_when_exact_closed'] for r in outcomes),
                    multiple_active_equivalents=sum(r['multiple_active_equivalents'] for r in outcomes),
                    exact_vs_operational_promotions100=sum(r['operational_rank'] is not None and r['operational_rank']<=100 and not(r['exact_rank'] is not None and r['exact_rank']<=100) for r in outcomes))
        write_json(dest/'manifest.json',result);results[variant]=result
        print(json.dumps(dict(stage='variant_complete',variant=variant,quality=quality)),flush=True)
    return dict(status='MEASURED_BOUNDED_OFFICIAL_EVIDENCE_FULL_LIST',decision='PIVOT_GENERATION',variants=results,
                primary_recipe='primary',fold0_opened=False,fold1_opened=False,positive_injection=False,
                production_slo_measured=False,full_population_measured=False,ready_for_fold1=False,
                excluded_missing_human_labels=evaluator['excluded_missing_human_labels'],label_side=evaluator['label_side'])


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    try:
        with Budget(RESOURCE) as budget:
            previous=v35.load_manifest(BASE/'manifest.json',BASE_SHA)
            base_features=json.loads(verify(previous['features_manifest']).read_text())
            prepared,models,sources,scope,paths=v35.discover()
            prereg=dict(implementation={str(p.relative_to(ROOT)):record(p) for p in
                [Path(__file__),ROOT/'scripts/retrieval_v35_site_fusion.py',ROOT/'docs/retrieval_v36_evidence_contract.md']},
                base=record(BASE/'manifest.json'),base_features=previous['features_manifest'],scope=scope,
                features=list(FEATURES),params=PARAMS,resource=RESOURCE,variants=['primary','ablation_training_only'],truth_opened=False)
            write_json(output/'preregistration.json',prereg)
            t=time.monotonic();c=v35.db(output);cache=project_sources(c,prepared['sources'],sources,output);c.close();gc.collect()
            print(json.dumps(dict(stage='official_cache',seconds=time.monotonic()-t)),flush=True)
            shards=[]
            for i,(source,base_shard) in enumerate(zip(sources,base_features['shards'],strict=True)):
                t=time.monotonic();shards.append(enrich(source,base_shard,cache,output/f'shard{i:05d}'))
                print(json.dumps(dict(stage='enrichment',done=i+1,total=32,seconds=time.monotonic()-t)),flush=True)
            write_json(output/'feature_manifest.json',dict(status='SEALED_BEFORE_LABELS',files=cache,shards=shards,features=list(FEATURES)))
            result=evaluate(shards,scope,output)
        result.update(resources=budget.report(),preregistration=record(output/'preregistration.json'),features_manifest=record(output/'feature_manifest.json'))
        write_json(output/'manifest.json',result)
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),exception=type(exc).__name__))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    run(parser.parse_args().output)
