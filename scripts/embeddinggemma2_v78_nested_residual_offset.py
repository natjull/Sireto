"""Nested TRAIN offsets for frozen native100 baseline + source residual.

Outer held-SIREN k: serving V32 coefficients k remain EXACTLY unchanged.
For residual TRAIN example j, baseline offset is fit excluding BOTH k and j.
This corrects in-sample overconfidence without exposing outer labels through
another fold's baseline weights. Same source features/pairs as V71, no mining
again, no new encoder, no parameter grid, no capped-pool reranking.
"""
import argparse
from pathlib import Path
import signal
import sys
import time
import numpy as np
from scripts import embeddinggemma2_v71_frozen_residual as pipeline
from scripts import embeddinggemma2_v32_learned_query as learned
from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v14_full_pipeline as engine
from src.xgb_matcher.retrieval_v20_workbench import record,verify

OUTPUT=io.HOME/'v78/nested_residual_offset_20261013_a'
PREVIOUS=pipeline.OUTPUT
BASE=pipeline.base.OUTPUT
FEATURE_FIELDS=pipeline.FEATURE_FIELDS
PROOF_FIELDS=pipeline.PROOF_FIELDS
LEXICAL_FIELDS=pipeline.LEXICAL_FIELDS
CACHE=pipeline.CACHE
BASE_AT=pipeline.BASE_AT
EXTRA_AT=pipeline.EXTRA_AT
RECIPE=dict(pipeline.RECIPE,hypothesis='EXACT_FROZEN_V32_SERVING_NESTED_OOF_TRAIN_MARGIN_RESIDUAL',
    residual_training_offsets='INNER_BASELINE_EXCLUDES_OUTER_HELD_SIREN_FOLD_AND_EACH_TRAIN_QUERY_FOLD',
    baseline_training_margin_in_sample_not_nested_oof=False,
    encoder_full_oof=False,no_new_encoder_fit=True,no_new_mining=True,
    same_v71_source_features_and_pairs=True,old_serving_coefficients_unchanged=True,
    inner_baseline_uses_original_v32_training_pairs=True,no_hyperparameter_grid=True)
PURE_BASE_FITTER=learned.fit_pairwise
ACTIVE=OUTPUT
guard=pipeline.guard
feature_path=pipeline.feature_path
lexical_parts=pipeline.lexical_parts
native_search=pipeline.native_search


def bind_runtime(out):
    global ACTIVE
    ACTIVE=out;pipeline.bind_runtime(out)


def initialize(out):
    pipeline.initialize(out);launch=io.read(out/'launch.json')
    io.write(out/'recipe.json',RECIPE)
    launch.update(implementation=record(Path(__file__)),recipe=record(out/'recipe.json'),
        previous_features=record(PREVIOUS/'features_generation.json'),
        original_baseline_calibration=record(BASE/'calibration/generation.json'))
    launch['helpers'].append(record(Path(pipeline.__file__)));io.write(out/'launch.json',launch)


def authority(out):
    launch=pipeline.authority(out)
    verify(launch['previous_features']);verify(launch['original_baseline_calibration'])
    return launch


def prepare(out):
    launch=authority(out);ex=io.read(PREVIOUS/'prepare_exit.json')
    if ex['exit_code'] or not ex['own_child_exit_confirmed']:raise ValueError('Full source features owned0')
    gen=io.read(verify(launch['previous_features']))
    for ref in gen['features'].values():verify(ref)
    io.write(out/'features_generation.json',dict(gen,immutable_source=launch['previous_features'],labels_used=False))


def fit_residual(difference,weights,l2):
    from scipy.optimize import minimize
    from scipy.special import expit
    fold=min(k for k in range(5) if not engine.sealed(ACTIVE/'calibration'/f'model_{k}.json'))
    launch=authority(ACTIVE);original=io.read(verify(launch['original_baseline_calibration']))
    assignment=io.read(verify(original['assignment']));source=io.read(ACTIVE/'calibration/pair_binding.json')
    ordered=source['ordered_training_queries'];shards=source['training_shards']
    inner_dir=ACTIVE/'calibration/inner_baselines';inner_dir.mkdir(exist_ok=True)
    inner={};receipts={}
    for j in range(5):
        if j==fold:continue
        guard(launch['deadline_unix']);path=inner_dir/f'outer_{fold}_inner_{j}.json'
        if not engine.sealed(path):
            train=[q for q in original['training_shards'] if assignment[q] not in (fold,j)]
            if not train:raise ValueError('Nonempty doubly purged baseline TRAIN')
            d=[];w=[]
            for q in train:
                with np.load(verify(original['training_shards'][q]),allow_pickle=False) as f:d.append(f['difference']);w.append(f['weights'])
            coefficients,info=PURE_BASE_FITTER(np.concatenate(d),np.concatenate(w),l2)
            io.write(path,dict(coefficients=coefficients.tolist(),feature_fields=pipeline.base.FEATURE_FIELDS,
                excluded_outer_fold=fold,excluded_inner_fold=j,training_queries=train,
                source_calibration=launch['original_baseline_calibration'],**info))
            engine.seal(path)
        receipt=record(engine.sealed(path));inner[j]=np.asarray(io.read(verify(receipt))['coefficients']);receipts[str(j)]=receipt
    margin=[];at=0;train_queries=[]
    for q in ordered:
        if assignment[q]==fold:continue
        with np.load(verify(shards[q]),allow_pickle=False) as f:
            d=f['difference'];n=len(d)
        if not np.array_equal(d,difference[at:at+n]):raise ValueError('Exact ordered TRAIN pair offset binding')
        margin.append(d[:,BASE_AT]@inner[assignment[q]]);at+=n;train_queries.append(q)
    if at!=len(difference):raise ValueError('Every residual TRAIN pair gets a nested offset')
    margin=np.concatenate(margin);weights=weights/weights.sum();a=difference[:,EXTRA_AT].astype(float)
    scale=np.maximum(np.sqrt(np.sum(a*a*weights[:,None],axis=0)),1e-4);a/=scale
    def objective(theta):
        z=margin+a@theta
        return float(weights@np.logaddexp(0.,-z)+.5*l2*np.dot(theta,theta)), -(a.T@(weights*expit(-z)))+l2*theta
    result=minimize(objective,np.zeros(len(EXTRA_AT)),jac=True,bounds=[(0,None)]*len(EXTRA_AT),method='L-BFGS-B',options={'maxiter':500,'ftol':1e-10,'gtol':1e-7})
    if not result.success:raise ValueError(result.message)
    old=io.read(verify(launch['frozen_models'][str(fold)]))
    coefficient=np.zeros(len(FEATURE_FIELDS));coefficient[BASE_AT]=old['coefficients'];coefficient[EXTRA_AT]=result.x/scale
    return coefficient,dict(training_pairwise_loss=float(result.fun),iterations=int(result.nit),
        nested_baseline_models=receipts,fixed_serving_baseline=launch['frozen_models'][str(fold)],
        baseline_weights_unchanged=True,nested_offsets_outer_label_disjoint=True,
        baseline_training_margin_in_sample_not_nested_oof=False,train_pairs=len(difference))


def calibrate(out):
    while True:
        guard(pipeline.corporate.DEADLINE)
        ex=io.read(PREVIOUS/'calibrate_exit.json') if (PREVIOUS/'calibrate_exit.json').exists() else None
        if (PREVIOUS/'calibration/generation.json').exists() and ex and ex['exit_code']==0 and ex['own_child_exit_confirmed']:break
        if (PREVIOUS/'run_error.json').exists() or (PREVIOUS/'calibrate_error.json').exists():raise ValueError('Original pair producer failed')
        time.sleep(10)
    source=io.read(PREVIOUS/'calibration/generation.json');original=io.read(BASE/'calibration/generation.json')
    if (source['assignment']['sha256']!=original['assignment']['sha256']
            or source['encoder_control_queries']['sha256']!=original['encoder_control_queries']['sha256']
            or source['feature_fields']!=FEATURE_FIELDS):raise ValueError('Exact source/partition/controls binding')
    root=out/'calibration';root.mkdir(exist_ok=True)
    panel=io.read(verify(authority(out)['panel']));ordered=[q['query_id'] for q in panel if q['query_id'] in source['training_shards']]
    io.write(root/'pair_binding.json',dict(producer=record(PREVIOUS/'calibration/generation.json'),
        owned_exit=record(PREVIOUS/'calibrate_exit.json'),training_shards=source['training_shards'],ordered_training_queries=ordered,
        same_full_source_pair_values=True,no_remine=True))
    import os
    for q in ordered:
        source_path=verify(source['training_shards'][q]);dest=root/source_path.name
        if not dest.exists():os.link(source_path,dest)
        engine.seal(dest)
    learned.calibrate(out)


def stage(action,out):
    bind_runtime(out);learned.proof=sys.modules[__name__];learned.fit_pairwise=fit_residual
    learned.RECIPE=dict(learned.RECIPE,model='FROZEN_BASELINE_NESTED_OFFSET_SOURCE_RESIDUAL',
        feature_fields=FEATURE_FIELDS,nested_baseline_offsets=True,encoder_full_oof=False)
    if action=='prepare':prepare(out)
    elif action=='calibrate':calibrate(out)
    else:getattr(learned,action)(out)
    if action=='evaluate':
        report=io.read(out/'report.json');report.update(nested_baseline_offsets=True,full_encoder_oof=False,
            eligibility=RECIPE['eligibility'],production_promotion_allowed=False,frozen_v32_serving_coefficients=True)
        io.write(out/'report.json',report)


def run(out):
    # Same owned supervisor, but process entrypoint is this immutable module.
    import fcntl,os,subprocess
    lock=(io.HOME/(out.name+'.owned.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    deadline=pipeline.corporate.DEADLINE
    for action,artifact in [('prepare','features_generation.json'),('calibrate','calibration/generation.json'),('generate','generation.json')]:
        if (out/artifact).exists():
            ex=io.read(out/(action+'_exit.json'))
            if ex['exit_code'] or not ex['own_child_exit_confirmed']:raise ValueError('Owned complete stage')
            continue
        while True:
            guard(deadline);command=[str(engine.CPU),'-u','-m','scripts.embeddinggemma2_v78_nested_residual_offset',action,'--output',str(out)]
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',RAYON_NUM_THREADS='2',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TMPDIR=str(out))
            with (out/(action+'.log')).open('ab',buffering=0) as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
                io.write(out/(action+'_process.json'),dict(pid=child.pid,command=command,owned=True,started_unix=time.time()))
                while child.poll() is None:
                    if engine.INTERRUPTED or time.time()>=deadline-30:
                        child.terminate()
                        try:child.wait(timeout=25)
                        except subprocess.TimeoutExpired:child.kill();child.wait()
                        break
                    time.sleep(5)
                rc=child.wait()
            io.write(out/(action+'_exit.json'),dict(exit_code=rc,pid=child.pid,own_child_exit_confirmed=True,finished_unix=time.time()))
            if engine.INTERRUPTED or time.time()>=deadline-30:raise TimeoutError('EIGHT_HOUR_END')
            if rc==75:time.sleep(30);continue
            if rc:raise RuntimeError(f'{action} exit {rc}, no partial recall')
            break
    stage('evaluate',out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['initialize','run','prepare','calibrate','generate','evaluate']);p.add_argument('--output',type=Path,default=OUTPUT);a=p.parse_args()
    signal.signal(signal.SIGTERM,engine.interrupted);signal.signal(signal.SIGINT,engine.interrupted)
    try:
        if a.action=='initialize':initialize(a.output)
        elif a.action=='run':run(a.output)
        else:stage(a.action,a.output)
    except engine.ResourceYield as e:print(str(e),flush=True);sys.exit(75)
    except Exception as e:
        if a.output.exists():io.write(a.output/(a.action+'_error.json'),dict(error=repr(e),updated_unix=time.time()))
        raise
