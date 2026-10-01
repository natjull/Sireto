"""Physical recovery of the frozen v36 ablation: one fresh process per fold."""
import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb

from scripts import retrieval_v35_site_fusion as v35
from scripts import retrieval_v36_evidence_ranker as v36
from src.xgb_matcher.retrieval_v20_workbench import Budget, record, verify, write_json

RESOURCE=dict(max_seconds=300,max_rss_bytes=5*1024**3,max_swapout_pages_delta=0)


def worker(source,output,fold,launch):
    output.mkdir(parents=True,exist_ok=False)
    budget=Budget(RESOURCE)
    try:
        with budget:
            authority=json.loads(launch.read_text())
            for ref in authority['implementation'].values():verify(ref)
            old=json.loads(verify(authority['source_preregistration']).read_text())
            for ref in old['implementation'].values():verify(ref)
            feature=json.loads(verify(authority['source_features']).read_text())
            scope=dict(old['scope']);targetids={q for q,f in scope.items() if f==fold}
            if set(scope.values())!={2,3,4} or len(scope)!=1024:raise ValueError('Forbidden scope')
            ev=v35.load_manifest(v35.EVAL,v35.EVAL_SHA)
            for ref in ev['files'].values():verify(ref)
            verify(dict(path=v35.LEDGER,sha256=v35.LEDGER_SHA))
            rows=pq.read_table(v35.LEDGER,filters=[('fold','in',[2,3,4])],columns=['query_id','fold','human_ground_truth_siret']).to_pylist()
            labels={r['query_id']:r['human_ground_truth_siret'] for r in rows if r['query_id'] in scope}
            if set(labels)!=set(scope):raise ValueError('Denominator drift')
            neutral={}
            for r in pq.read_table(ev['files']['acceptable_operational.parquet']['path'],filters=[('fold','in',[2,3,4])]).to_pylist():
                if r['query_id'] in scope and not r['exact_truth']:neutral.setdefault(r['query_id'],set()).add(r['siret'])
            features=v35.FEATURES;parts=[];stats=[]
            for shard in feature['shards']:
                for qfold in (f for f in (2,3,4) if f!=fold):
                    inner=v35.train_context(fold,qfold)
                    frame=pq.read_table(verify(shard['files'][f'inner{inner}.parquet']),
                        columns=['query_id','fold','siret','candidate_rank']+list(features),filters=[('fold','=',qfold)]).to_pandas()
                    sample,st=v36.full_training(frame,labels,neutral);parts.append(sample);stats.append(st)
            train=pd.concat(parts,ignore_index=True).sort_values(['query_id','candidate_rank','siret']);del parts;gc.collect()
            trainids=sorted(set(train.query_id))
            if set(trainids)&targetids or any(fold==f for f in train.fold):raise ValueError('Outer fold leakage')
            values=train[list(features)].to_numpy(dtype=np.float32);target=train.target.to_numpy(dtype=np.float32)
            groups=train.groupby('query_id',sort=False).size().to_numpy();del train;gc.collect()
            if not np.isfinite(values).all():raise ValueError('Nonfinite training values')
            model=xgb.XGBRanker(**v36.PARAMS,callbacks=[v36.Progress('ablation_training_only',fold)])
            model.fit(values,target,group=groups,verbose=False)
            model.save_model(output/'model.json');del values,target,groups;gc.collect()
            files=[];observed=set();parity=[]
            for i,shard in enumerate(feature['shards']):
                frame=pq.read_table(verify(shard['files']['outer.parquet']),filters=[('fold','=',fold)]).to_pandas()
                if not len(frame):continue
                frame['score']=model.predict(frame[list(features)].to_numpy(dtype=np.float32))
                if not np.isfinite(frame.score).all():raise ValueError('Nonfinite predictions')
                frame=frame.sort_values(['query_id','score','siret'],ascending=[True,False,True])
                frame['site_rank']=frame.groupby('query_id').cumcount()+1
                cols=['query_id','fold','siren','siret','score','site_rank'];result=pa.Table.from_pandas(frame[cols],preserve_index=False)
                prior=source/'ablation_training_only'/f'scores_{fold}_{i:03d}.parquet'
                key=str(prior)
                if key in authority['partial_scores']:
                    previous=pq.read_table(verify(authority['partial_scores'][key]))
                    if not result.equals(previous,check_metadata=False):raise ValueError('Physical recovery prediction drift')
                    parity.append(key)
                path=output/f'scores_{i:03d}.parquet';pq.write_table(result,path,compression='zstd');files.append(record(path))
                observed.update(frame.query_id)
                for q,g in frame.groupby('query_id'):
                    if len(g)>2000 or g.siret.duplicated().any() or scope[q]!=fold:raise ValueError('Output cap/scope drift')
            if observed!=targetids:raise ValueError('Incomplete heldout fold')
        result=dict(status='PASS_ISOLATED_FOLD',fold=fold,features=list(features),params=v36.PARAMS,
                    trained_query_ids=trainids,sampling={k:sum(st[k] for st in stats) for k in stats[0]},
                    model=record(output/'model.json'),scores=files,resources=budget.report(),parity_with_partial_scores=parity)
        write_json(output/'manifest.json',result)
        print(json.dumps(dict(stage='isolated_fold_complete',fold=fold,resources=result['resources'],parity_shards=len(parity))),flush=True)
    except BaseException as exc:
        resource=budget.report() if hasattr(budget,'elapsed') else {}
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),resources=resource))
        raise


def aggregate(source,output,workers,scope):
    ev=v35.load_manifest(v35.EVAL,v35.EVAL_SHA)
    for ref in ev['files'].values():verify(ref)
    ids={q for q,f in scope}
    baseline={r['query_id']:r for r in pq.read_table(ev['files']['v30_stable_outcomes.parquet']['path'],filters=[('fold','in',[2,3,4])]).to_pylist() if r['query_id'] in ids}
    previous=v35.load_manifest(v36.BASE/'manifest.json',v36.BASE_SHA)
    prior={r['query_id']:r for r in pq.read_table(verify(previous['outcomes'])).to_pylist()}
    acceptable={};active={}
    for r in pq.read_table(ev['files']['acceptable_operational.parquet']['path'],filters=[('fold','in',[2,3,4])]).to_pylist():
        if r['query_id'] in ids:
            acceptable.setdefault(r['query_id'],set()).add(r['siret'])
            if not r['exact_truth'] and r['administrative_state']=='A':active.setdefault(r['query_id'],set()).add(r['siret'])
    outcomes=[]
    for worker_result in workers:
        for ref in worker_result['scores']:
            frame=pq.read_table(verify(ref)).to_pandas()
            for q,g in frame.groupby('query_id',sort=False):
                b=baseline[q];exact=g.loc[g.siret==b['gt_siret'],'site_rank'].tolist();rank=exact[0] if exact else None
                op=g.loc[g.siret.isin(acceptable[q]),'site_rank'].tolist()
                if bool(rank)!=bool(b['exact_rank']):raise ValueError('Master coverage drift')
                loss=('HIT' if rank is not None and rank<=100 else 'EVICTED_500_TO_100' if rank is not None and rank<=500
                      else 'EVICTED_MASTER_TO_500' if rank is not None else b['loss_stage'])
                outcomes.append(dict(query_id=q,fold=int(g.fold.iloc[0]),exact_rank=rank,operational_rank=min(op) if op else None,
                    baseline_exact_rank=prior[q]['exact_rank'],heuristic_exact_rank=b['exact_rank'],loss_stage=loss,
                    label_kind=b['label_kind'],qualification_v2=b['qualification_v2'],qualification_v3=b['qualification_v3'],
                    parent_in_top100=bool(((g.siren==b['gt_siren']) & (g.site_rank<=100)).any()),
                    active_equivalent_when_exact_closed=bool(b['exact_state']=='F' and ((g.siret.isin(active.get(q,set()))) & (g.site_rank<=100)).any()),
                    multiple_active_equivalents=b['multiple_active_equivalents']))
    if len(outcomes)!=1024 or {r['query_id'] for r in outcomes}!=ids:raise ValueError('Incomplete OOF recovery')
    pq.write_table(pa.Table.from_pylist(outcomes),output/'ablation_outcomes.parquet',compression='zstd')
    segments={col:{str(v):v35.summarize([r for r in outcomes if r[col]==v]) for v in sorted({r[col] for r in outcomes},key=str)} for col in ('fold','label_kind','qualification_v2','qualification_v3')}
    primary=json.loads((source/'primary/manifest.json').read_text())
    verify(primary['outcomes'])
    for m in primary['models']:verify(m['model'])
    return dict(status='MEASURED_V36_WITH_ISOLATED_ABLATION_RECOVERY',decision='PIVOT_GENERATION',primary=primary,
        primary_authority=record(source/'primary/manifest.json'),original_run_status=record(source/'status.json'),
        ablation=dict(quality=v35.summarize(outcomes),segments=segments,outcomes=record(output/'ablation_outcomes.parquet'),
            parent_hits100=sum(r['parent_in_top100'] for r in outcomes),
            active_equivalent_when_exact_closed=sum(r['active_equivalent_when_exact_closed'] for r in outcomes),
            multiple_active_equivalents=sum(r['multiple_active_equivalents'] for r in outcomes),
            promotions100=sum(r['operational_rank'] is not None and r['operational_rank']<=100 and not(r['exact_rank'] is not None and r['exact_rank']<=100) for r in outcomes),
            losses={s:sum(r['loss_stage']==s for r in outcomes) for s in {r['loss_stage'] for r in outcomes}},workers=workers),
        fold0_opened=False,fold1_opened=False,full_population_measured=False,ready_for_fold1=False,
        resources_original_run='STOP_RESOURCE_POLICY_EXACT_TRIGGER_NOT_RECORDED',
        excluded_missing_human_labels=ev['excluded_missing_human_labels'],label_side=ev['label_side'])


def recover(source,output):
    output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    prereg=json.loads((source/'preregistration.json').read_text())
    for ref in prereg['implementation'].values():verify(ref)
    feature=json.loads((source/'feature_manifest.json').read_text())
    for shard in feature['shards']:
        for ref in shard['files'].values():verify(ref)
    launch=dict(source_preregistration=record(source/'preregistration.json'),source_features=record(source/'feature_manifest.json'),
        implementation={str(p):record(p) for p in [Path(__file__),Path(v36.__file__),Path(v35.__file__)]},
        physical_change='FRESH_SEQUENTIAL_PROCESS_PER_FOLD_SAME_FEATURES_GROUPS_PARAMS_SEED',
        partial_scores={str(p):record(p) for p in (source/'ablation_training_only').glob('scores_*.parquet')})
    write_json(output/'launch.json',launch);workers=[]
    for fold in (2,3,4):
        cmd=[sys.executable,'-u','-m','scripts.retrieval_v36_isolated_fit','worker','--source',str(source),'--output',str(output/f'fold{fold}'),'--fold',str(fold),'--launch',str(output/'launch.json')]
        subprocess.run(cmd,check=True,timeout=min(330,900-(time.monotonic()-started)))
        workers.append(json.loads((output/f'fold{fold}/manifest.json').read_text()))
    result=aggregate(source,output,workers,prereg['scope']);result['seconds']=time.monotonic()-started
    result['launch']=record(output/'launch.json');write_json(output/'manifest.json',result)
    print(json.dumps(dict(stage='recovery_complete',ablation=result['ablation']['quality'],seconds=result['seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['recover','worker'])
    p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--fold',type=int,choices=[2,3,4]);p.add_argument('--launch',type=Path);args=p.parse_args()
    if args.mode=='worker':worker(args.source,args.output,args.fold,args.launch)
    else:recover(args.source,args.output)
