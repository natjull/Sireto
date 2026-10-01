"""Restore every preregistered discovery-parent channel to the learned admission."""
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
from scripts.retrieval_v36_isolated_fit import aggregate, RESOURCE
from src.xgb_matcher.retrieval_v20_workbench import Budget, ROOT, record, verify, write_json

SOURCE=v35.DATA/'audits/retrieval_v36_evidence_ranker/pilot_20261002_a'
RECOVERY=v35.DATA/'audits/retrieval_v36_evidence_ranker/recovery_20261002_a/manifest.json'
RECOVERY_SHA='97abf5e886068ccd878304f8f44ddec4efe470a5ddd7798949c9a9e602e471db'
CONFIG=ROOT/'config/retrieval_all_human_v16_parent_workbench.json'
CHANNELS=tuple(s['name'] for s in json.loads(CONFIG.read_text())['channels'])
EXTRA=tuple(f'discovery_{channel}_{stat}' for channel in CHANNELS for stat in ('rank_rrf','relative_score'))
FEATURES=v36.FEATURES+EXTRA


def parent_table(signals):
    required={'query_id','fold','siren','channel','channel_rank','score'}
    if not required<=set(signals):raise ValueError('Missing parent evidence fields')
    if not set(signals.channel)<=set(CHANNELS):raise ValueError('Unknown parent channel')
    if not set(signals.fold)<={2,3,4}:raise ValueError('Forbidden discovery fold')
    if (signals.channel_rank<1).any() or not np.isfinite(signals.score).all():raise ValueError('Invalid discovery score/rank')
    rows=signals[['query_id','fold','siren']].drop_duplicates().set_index(['query_id','fold','siren'])
    s=signals.copy();s['rank_rrf']=1./(60+s.channel_rank.astype(np.float64))
    score=s.score.clip(lower=0);denominator=score.groupby([s.query_id,s.fold,s.channel]).transform('max')
    s['relative_score']=np.divide(score,denominator,out=np.zeros(len(s)),where=denominator.to_numpy()>0)
    for channel in CHANNELS:
        one=s.loc[s.channel==channel].groupby(['query_id','fold','siren'])[['rank_rrf','relative_score']].max()
        for stat in ('rank_rrf','relative_score'):
            rows[f'discovery_{channel}_{stat}']=one[stat].reindex(rows.index).fillna(0).astype(np.float32)
    return rows.reset_index()


def attach(frame,evidence):
    if evidence.duplicated(['query_id','fold','siren']).any():raise ValueError('Duplicate parent evidence')
    result=frame.merge(evidence,on=['query_id','fold','siren'],how='left',validate='many_to_one',sort=False)
    result[list(EXTRA)]=result[list(EXTRA)].fillna(0).astype(np.float32)
    if len(result)!=len(frame):raise ValueError('Master changed by parent attachment')
    return result


def worker(launch_path,output,fold):
    output.mkdir(parents=True,exist_ok=False);budget=Budget(RESOURCE)
    try:
        with budget:
            launch=json.loads(launch_path.read_text())
            for ref in launch['implementation'].values():verify(ref)
            parent_shards=launch['parent_shards'];source=json.loads(verify(launch['source_features']).read_text())
            prereg=json.loads(verify(launch['source_preregistration']).read_text())
            for ref in prereg['implementation'].values():verify(ref)
            scope=dict(prereg['scope']);targetids={q for q,f in scope.items() if f==fold}
            if len(scope)!=1024 or set(scope.values())!={2,3,4}:raise ValueError('Scope drift')
            evaluator=v35.load_manifest(v35.EVAL,v35.EVAL_SHA)
            verify(dict(path=v35.LEDGER,sha256=v35.LEDGER_SHA))
            labels={r['query_id']:r['human_ground_truth_siret'] for r in pq.read_table(v35.LEDGER,filters=[('fold','in',[2,3,4])],columns=['query_id','human_ground_truth_siret']).to_pylist() if r['query_id'] in scope}
            if set(labels)!=set(scope):raise ValueError('Label scope drift')
            neutral={}
            for r in pq.read_table(verify(evaluator['files']['acceptable_operational.parquet']),filters=[('fold','in',[2,3,4])]).to_pylist():
                if r['query_id'] in scope and not r['exact_truth']:neutral.setdefault(r['query_id'],set()).add(r['siret'])
            parts=[];stats=[]
            for shard,parent in zip(source['shards'],parent_shards,strict=True):
                evidence=pq.read_table(verify(parent)).to_pandas()
                for qfold in (f for f in (2,3,4) if f!=fold):
                    inner=v35.train_context(fold,qfold)
                    frame=pq.read_table(verify(shard['files'][f'inner{inner}.parquet']),filters=[('fold','=',qfold)]).to_pandas()
                    frame=attach(frame,evidence);sample,st=v36.full_training(frame,labels,neutral);parts.append(sample);stats.append(st)
            train=pd.concat(parts,ignore_index=True).sort_values(['query_id','candidate_rank','siret']);del parts;gc.collect()
            trainids=sorted(set(train.query_id))
            if set(trainids)&targetids or (train.fold==fold).any():raise ValueError('Outer labels in fit')
            values=train[list(FEATURES)].to_numpy(dtype=np.float32);target=train.target.to_numpy(dtype=np.float32)
            groups=train.groupby('query_id',sort=False).size().to_numpy();del train;gc.collect()
            if not np.isfinite(values).all():raise ValueError('Nonfinite features')
            model=xgb.XGBRanker(**v36.PARAMS,callbacks=[v36.Progress('parent_evidence',fold)])
            model.fit(values,target,group=groups,verbose=False);model.save_model(output/'model.json')
            del values,target,groups;gc.collect();scores=[];observed=set()
            for i,(shard,parent) in enumerate(zip(source['shards'],parent_shards,strict=True)):
                frame=pq.read_table(verify(shard['files']['outer.parquet']),filters=[('fold','=',fold)]).to_pandas()
                if not len(frame):continue
                frame=attach(frame,pq.read_table(verify(parent)).to_pandas())
                frame['score']=model.predict(frame[list(FEATURES)].to_numpy(dtype=np.float32))
                if not np.isfinite(frame.score).all():raise ValueError('Nonfinite prediction')
                frame=frame.sort_values(['query_id','score','siret'],ascending=[True,False,True])
                frame['site_rank']=frame.groupby('query_id').cumcount()+1;observed.update(frame.query_id)
                for q,g in frame.groupby('query_id'):
                    if len(g)>2000 or g.siret.duplicated().any() or scope[q]!=fold:raise ValueError('Natural Master invalid')
                dest=output/f'scores_{i:03d}.parquet'
                pq.write_table(pa.Table.from_pandas(frame[['query_id','fold','siren','siret','score','site_rank']],preserve_index=False),dest,compression='zstd')
                scores.append(record(dest))
            if observed!=targetids:raise ValueError('Heldout population incomplete')
        write_json(output/'manifest.json',dict(status='PASS_ISOLATED_PARENT_EVIDENCE_FOLD',fold=fold,
            model=record(output/'model.json'),scores=scores,trained_query_ids=trainids,features=list(FEATURES),
            sampling={k:sum(s[k] for s in stats) for k in stats[0]},resources=budget.report()))
        print(json.dumps(dict(stage='v37_fold_complete',fold=fold,resources=budget.report())),flush=True)
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),resources=budget.report() if hasattr(budget,'elapsed') else {}));raise


def run(output):
    output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    recovery=v35.load_manifest(RECOVERY,RECOVERY_SHA)
    primary=json.loads(verify(recovery['primary_authority']).read_text());verify(primary['outcomes'])
    # Reuse the verified v36 snapshot rather than rerun any discovery or proof build.
    prep,models,source_shards,scope,paths=v35.discover()
    feature_ref=record(SOURCE/'feature_manifest.json');features=json.loads(verify(feature_ref).read_text())
    if len(features['shards'])!=32:raise ValueError('Wrong feature snapshot')
    evidence=[];source_refs=[]
    for i,s in enumerate(source_shards):
        ref=s['files']['parent_signals.parquet'];parent=pq.read_table(verify(ref)).to_pandas()
        table=parent_table(parent);dest=output/f'parent_evidence_{i:03d}.parquet'
        pq.write_table(pa.Table.from_pandas(table,preserve_index=False),dest,compression='zstd');evidence.append(record(dest));source_refs.append(ref)
    implementation={str(p):record(p) for p in [Path(__file__),Path(v35.__file__),Path(v36.__file__),ROOT/'scripts/retrieval_v36_isolated_fit.py',CONFIG,ROOT/'docs/retrieval_v37_parent_evidence_contract.md']}
    launch=dict(implementation=implementation,source_features=feature_ref,source_preregistration=record(SOURCE/'preregistration.json'),
        parent_shards=evidence,parent_sources=source_refs,features=list(FEATURES),params=v36.PARAMS,resources=RESOURCE,
        previous=record(RECOVERY),primary_v36=recovery['primary_authority'],scope=scope,truth_opened=False)
    write_json(output/'launch.json',launch);workers=[]
    for fold in (2,3,4):
        cmd=[sys.executable,'-u','-m','scripts.retrieval_v37_parent_evidence','worker','--output',str(output/f'fold{fold}'),'--fold',str(fold),'--launch',str(output/'launch.json')]
        subprocess.run(cmd,check=True,timeout=min(330,900-(time.monotonic()-start)))
        workers.append(json.loads((output/f'fold{fold}/manifest.json').read_text()))
    audit=aggregate(SOURCE,output,workers,scope)
    current=audit['ablation'];current['baseline_comparison']='v35'
    outcomes=pq.read_table(verify(current['outcomes'])).to_pylist()
    v36_outcomes={r['query_id']:r for r in pq.read_table(verify(primary['outcomes'])).to_pylist()}
    paired=v35.summarize([dict(r,baseline_exact_rank=v36_outcomes[r['query_id']]['exact_rank']) for r in outcomes])
    result=dict(status='MEASURED_PARENT_DISCOVERY_EVIDENCE_RESTORED',decision='PIVOT_GENERATION',primary=current,
        paired_vs_v36=paired,previous_v36=recovery['primary_authority'],launch=record(output/'launch.json'),seconds=time.monotonic()-start,
        fold0_opened=False,fold1_opened=False,full_population_measured=False,production_slo_measured=False,ready_for_fold1=False,
        excluded_missing_human_labels=audit['excluded_missing_human_labels'],label_side=audit['label_side'])
    write_json(output/'manifest.json',result)
    print(json.dumps(dict(stage='v37_complete',quality=current['quality'],paired_vs_v36=paired,seconds=result['seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['run','worker']);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--fold',type=int,choices=[2,3,4]);p.add_argument('--launch',type=Path);a=p.parse_args()
    if a.mode=='worker':worker(a.launch,a.output,a.fold)
    else:run(a.output)
