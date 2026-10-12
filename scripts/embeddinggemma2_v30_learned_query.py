"""Seven-hour, owned first-stage learned-query experiment. No unit tests.

Calibration is SIREN-cross-fitted, with Gemma TRAIN SIRENs disjoint and its
checkpoint-control SIRENs excluded from calibration fitting. ALL526 remain in
evaluation. Training examples may include their labeled positive; search never
does. All score features are public evidence, no CRM alias memorization.
"""
import argparse
from collections import defaultdict
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

from scripts import embeddinggemma2_v30_native_features as proof
from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v14_full_pipeline as engine
from scripts import embeddinggemma2_v17_structural_retrieval as structural
from scripts import embeddinggemma2_v18_identity_retrieval as identity
from scripts import embeddinggemma2_v15_probability_pilot as pilot
from src.xgb_matcher.hierarchical_retrieval import TantivyBackend
from src.xgb_matcher.retrieval_v20_workbench import record,verify

RECIPE=dict(model='RNE_CORPORATE_ACTOR_CONTEXT_NONNEGATIVE_PAIRWISE_L2',l2=.01,folds=5,
    group='HUMAN_TARGET_SIREN_PARTITION_ONLY_NOT_FEATURE',
    group_seed='sireto-native-calibration-20261012-v1',
    training_negatives='12_HARD_PER_PROOF_FIELD_32_UNIFORM_AND_HISTORICAL_NATIONAL_LEXICAL100',
    negatives_are_training_sampling_not_production_pool=True,
    training_positive='LABELED_TRAINING_ONLY_NO_SEARCH_INJECTION',
    query_weights='ORIGINAL_COMMUNE_STRATIFIED_PROBABILITY_WEIGHTS',
    encoder_control_sirens_excluded_from_calibration_training=True,
    ancestor_cached64_and_current256_control_sirens_both_excluded=True,
    every526_query_retained_in_evaluation=True,
    standardization='TRAIN_ONLY_WEIGHTED_PAIR_DIFFERENCE_RMS',
    no_hyperparameter_grid=True,native_quantization_bits=proof.RECIPE['quantization_bits'],
    scope='CONSUMED_DEVELOPMENT_CROSSFITTED_CALIBRATOR_NOT_FULL_ENCODER_OOF',
    hierarchical_owner_geometry_and_official_name_portfolio=True,
    corporate_role_name_context_not_identity_or_own_alias=True,
    no_common_auditor_to_group_collapse=True,
    closed_test_opened=False,downstream_models_changed=False)


def calibration_sources():
    source=io.read(structural.SOURCE/'launch.json');manifest=io.read(verify(source['training_manifest']))
    membership=pilot.population.SOURCE/'validation_membership.json'
    if io.digest(membership)!=manifest['split_hashes']['validation']:raise ValueError('Original validation label authority')
    fit_launch=io.read(io.HOME/'v6/deadline_recovery_20261008_a/fit/launch.json')
    data=io.read(verify(fit_launch['data']));config=io.read(verify(data['config']))
    artifacts=data['artifacts'];training=io.read(verify(artifacts['train.json']));validation=io.read(verify(artifacts['validation.json']))
    train_sirens={r['siren'] for r in training}
    controls=sorted(validation,key=lambda r:hashlib.sha256(('gemma2-v2-control|'+r['query_id']).encode()).digest())[:config['validation_queries']]
    control_sirens={r['siren'] for r in controls}
    ancestor_launch=io.read(io.HOME/'cached_20261007_a/launch.json');ancestor=Path(ancestor_launch['source'])
    ancestor_refs=[dict(path=str(ancestor/name),sha256=ancestor_launch['inputs'][name]) for name in ('train.json','validation.json')]
    ancestor_train=io.read(verify(ancestor_refs[0]));ancestor_validation=io.read(verify(ancestor_refs[1]))
    train_sirens.update(r['siren'] for r in ancestor_train)
    ancestor_controls=sorted((r for r in ancestor_validation if r['negatives']),
        key=lambda r:hashlib.sha256(r['query_id'].encode()).digest())[:64]
    if len(ancestor_controls)!=64:raise ValueError('Exact initial cached checkpoint controls')
    control_sirens.update(r['siren'] for r in ancestor_controls)
    labels=io.read(membership)
    if {r['siren'] for r in labels}&train_sirens:raise ValueError('Gemma training identity leakage into calibration')
    return labels,control_sirens,[record(membership),fit_launch['data'],data['config'],artifacts['train.json'],artifacts['validation.json'],
        record(io.HOME/'cached_20261007_a/launch.json'),*ancestor_refs]


def masked_lexical_features(backend,q,keys):
    """Actual national BM25 scores, looked up for TRAIN examples only.

    A score-zero identifier mask is a lookup, never a predictive feature. All
    statistics remain those of the global index. No unmatched row is removed.
    """
    x=np.zeros((len(keys),len(proof.LEXICAL_FIELDS)),dtype=np.float32)
    if not len(keys):return x
    tv=backend._tantivy;refs=[f'{int(k):014d}' for k in keys];positions={k:i for i,k in enumerate(refs)}
    lookup=tv.Query.const_score_query(tv.Query.term_set_query(backend.index.schema,'siret',refs),0.)
    leaf=tv.Query.const_score_query(tv.Query.term_query(backend.index.schema,'document_type','siret','basic'),0.)
    for col,part in enumerate(proof.lexical_parts(backend,q)):
        if part is None:continue
        query=tv.Query.boolean_query([(tv.Occur.Must,lookup),(tv.Occur.Must,leaf),(tv.Occur.Should,part)])
        for score,loc in backend.searcher.search(query,limit=len(keys)+1,count=False).hits:
            key=backend._first(backend.searcher.doc(loc).to_dict(),'siret')
            if key not in positions or not np.isfinite(score) or score<0:raise ValueError('Exact native lookup scope/score')
            x[positions[key],col]=score
    return x


def fit_pairwise(difference,weights,l2):
    from scipy.optimize import minimize
    from scipy.special import expit
    weights=weights/weights.sum();scale=np.sqrt(np.sum(difference.astype(float)**2*weights[:,None],axis=0))
    scale=np.maximum(scale,1e-4);a=difference/scale
    def objective(theta):
        margin=a@theta
        loss=float(weights@np.logaddexp(0.,-margin)+.5*l2*np.dot(theta,theta))
        gradient=-(a.T@(weights*expit(-margin)))+l2*theta
        return loss,gradient
    result=minimize(objective,np.zeros(difference.shape[1]),jac=True,bounds=[(0,None)]*difference.shape[1],
        method='L-BFGS-B',options={'maxiter':500,'ftol':1e-10,'gtol':1e-7})
    if not result.success or not np.isfinite(result.fun) or not np.isfinite(result.x).all():raise ValueError(f'Calibration optimizer: {result.message}')
    return result.x/scale,dict(optimizer_message=str(result.message),iterations=int(result.nit),
        training_pairwise_loss=float(result.fun),training_pairs=len(difference),scale=scale.tolist())


def calibrate(out):
    launch=proof.authority(out);gen=io.read(out/'features_generation.json')
    if gen['status']!='COMPLETE_SEALED_QUERY_ONLY_ALL526_FULL_PROOFS':raise ValueError('Complete label-free features first')
    labels,controls,sources=calibration_sources();panel=io.read(verify(launch['panel']));scope={q['query_id'] for q in panel}
    labels={r['query_id']:r for r in labels if r['query_id'] in scope}
    if set(labels)!=scope or any(r['fold'] not in (2,3,4) for r in labels.values()):raise ValueError('All526 safe calibration scope')
    root=out/'calibration';root.mkdir(exist_ok=True)
    if not (root/'launch.json').exists():
        io.write(root/'recipe.json',RECIPE)
        assignment={qid:int.from_bytes(hashlib.sha256((RECIPE['group_seed']+'|'+r['siren']).encode()).digest()[:8],'big')%5 for qid,r in labels.items()}
        io.write(root/'assignment.json',assignment)
        excluded=sorted(q for q,r in labels.items() if r['siren'] in controls);io.write(root/'encoder_control_queries.json',excluded)
        io.write(root/'launch.json',dict(implementation=record(Path(__file__)),feature_implementation=launch['implementation'],
            feature_generation=record(out/'features_generation.json'),recipe=record(root/'recipe.json'),
            assignment=record(root/'assignment.json'),encoder_control_queries=record(root/'encoder_control_queries.json'),
            label_sources=sources,encoder_train_disjoint=True,deadline_unix=launch['deadline_unix'],
            calibration_labels_read_for_training_only=True,encoder_control_queries_kept_in_primary_evaluation=True))
    cal=io.read(root/'launch.json')
    for ref in (cal['implementation'],cal['feature_implementation'],cal['feature_generation'],cal['recipe'],cal['assignment'],cal['encoder_control_queries'],*cal['label_sources']):verify(ref)
    assignment=io.read(verify(cal['assignment']));excluded=set(io.read(verify(cal['encoder_control_queries'])))
    source=io.read(verify(launch['source_launch']));old=io.read(verify(io.read(proof.CACHE/'launch.json')['baseline_outputs']))
    sampling=io.read(verify(source['sampling']));backend=TantivyBackend(io.INDEX)
    if backend.searcher.num_docs!=82772691:raise ValueError('National corpus drift')
    shards=[];missing=[]
    for i,q in enumerate(panel):
        proof.guard(launch['deadline_unix']);qid=q['query_id']
        if qid in excluded:continue
        path=root/(hashlib.sha256(qid.encode()).hexdigest()+'.npz');saved=engine.sealed(path)
        if saved:shards.append((qid,saved));continue
        with np.load(verify(gen['features'][qid]),allow_pickle=False) as f:keys=f['keys'];x=f['features']
        gt=int(labels[qid]['siret']);selected=set()
        rng=np.random.default_rng(int.from_bytes(hashlib.sha256(('training-negatives|'+qid).encode()).digest()[:8],'big'))
        if len(keys):
            k=min(12,len(keys))
            for col in range(x.shape[1]):
                if x[:,col].max()>0:selected.update(map(int,keys[np.argpartition(x[:,col],len(keys)-k)[-k:]]))
            selected.update(map(int,keys[rng.choice(len(keys),size=min(32,len(keys)),replace=False)]))
        selected.update(int(r['siret']) for r in old['lexical'][qid]);selected.discard(gt)
        candidates=np.array([gt]+sorted(selected),dtype=np.uint64)
        numeric=np.zeros((len(candidates),len(proof.PROOF_FIELDS)),dtype=np.float32)
        loc=np.searchsorted(keys,candidates);inside=loc<len(keys);inside[inside]&=keys[loc[inside]]==candidates[inside]
        numeric[inside]=x[loc[inside]]
        if not inside[0]:missing.append(qid)
        lex=masked_lexical_features(backend,q,candidates);matrix=np.column_stack((lex,numeric))
        if not np.isfinite(matrix).all():raise ValueError('Finite official training proof features')
        difference=matrix[0:1]-matrix[1:]
        if not len(difference):raise ValueError('No training negatives; cannot silently drop query')
        zone=q['crm_insee'];weight=sampling['zone_weights'][zone]['calibrated_query_weight']
        pair_weights=np.full(len(difference),weight/len(difference),dtype=np.float64)
        temp=path.with_suffix('.pending.npz');np.savez(temp,difference=difference,weights=pair_weights)
        os.replace(temp,path);engine.seal(path);shards.append((qid,path))
        io.write(root/'state.json',dict(status='BUILDING_OFFICIAL_TRAINING_PROOF_PAIRS',completed=len(shards),
            total=len(panel)-len(excluded),query_id=qid,encoder_control_queries_not_fit=len(excluded),updated_unix=time.time()))
        del keys,x,numeric,lex,matrix,difference;gc.collect()
    models={};fold_receipts={}
    for fold in range(5):
        proof.guard(launch['deadline_unix']);model=root/f'model_{fold}.json'
        train=[q for q,p in shards if assignment[q]!=fold];held=[q for q in scope if assignment[q]==fold]
        train_sirens={labels[q]['siren'] for q in train};held_sirens={labels[q]['siren'] for q in held}
        if train_sirens&held_sirens:raise ValueError('Calibration SIREN overlap')
        if not engine.sealed(model):
            values=[];weights=[]
            for q,path in shards:
                if assignment[q]==fold:continue
                with np.load(engine.sealed(path),allow_pickle=False) as f:values.append(f['difference']);weights.append(f['weights'])
            d=np.concatenate(values);w=np.concatenate(weights);coef,info=fit_pairwise(d,w,RECIPE['l2'])
            io.write(model,dict(feature_fields=proof.FEATURE_FIELDS,coefficients=coef.tolist(),
                **info,heldout_fold=fold,train_queries=sorted(train),heldout_queries=sorted(held),
                train_siren_hashes=sorted(hashlib.sha256(s.encode()).hexdigest() for s in train_sirens),
                heldout_siren_hashes=sorted(hashlib.sha256(s.encode()).hexdigest() for s in held_sirens),
                no_group_overlap=True,encoder_control_sirens_not_fit=True,calibration_launch=record(root/'launch.json')))
            engine.seal(model);del values,weights,d,w
        models[str(fold)]=record(engine.sealed(model));fold_receipts[str(fold)]=dict(train_queries=len(train),heldout_queries=len(held),no_siren_overlap=True)
    io.write(root/'generation.json',dict(status='SEALED_ALL_FIVE_SIREN_CROSS_FITTED_QUERY_MODELS',
        models=models,assignment=cal['assignment'],encoder_control_queries=cal['encoder_control_queries'],
        folds=fold_receipts,feature_fields=proof.FEATURE_FIELDS,missing_nonlexical_positive_queries=missing,
        all526_evaluation_queries_retained=True,no_eval_positive_injection=True,full_encoder_oof=False,
        implementation=cal['implementation'],training_shards={q:record(p) for q,p in shards}))


def generate(out):
    launch=proof.authority(out);cal=io.read(out/'calibration/generation.json');verify(cal['implementation'])
    if cal['status']!='SEALED_ALL_FIVE_SIREN_CROSS_FITTED_QUERY_MODELS':raise ValueError('All calibration models sealed first')
    models={k:io.read(verify(v)) for k,v in cal['models'].items()};assignment=io.read(verify(cal['assignment']))
    features=io.read(out/'features_generation.json');panel=io.read(verify(launch['panel']));backend=TantivyBackend(io.INDEX)
    if backend.searcher.num_docs!=82772691:raise ValueError('Actual national index count')
    root=out/'native';root.mkdir(exist_ok=True);rows={};timings={};started=time.monotonic()
    if not (root/'launch.json').exists():io.write(root/'launch.json',dict(implementation=record(Path(__file__)),
        feature_launch=record(out/'launch.json'),calibration=record(out/'calibration/generation.json'),
        feature_generation=record(out/'features_generation.json'),query_only_panel=launch['panel'],
        models=cal['models'],assignment=cal['assignment'],deadline_unix=launch['deadline_unix'],
        evaluation_labels_read=False,no_fixed_prepool=True,native_cap=100))
    for q in panel:
        proof.guard(launch['deadline_unix']);qid=q['query_id'];model=models[str(assignment[qid])]
        if qid not in model['heldout_queries'] or qid in model['train_queries'] or not model['no_group_overlap']:
            raise ValueError('This query is genuinely held out of its calibrator')
        path=root/(hashlib.sha256(qid.encode()).hexdigest()+'.json');saved=engine.sealed(path)
        if saved:
            part=io.read(saved)
            if part['query']!=q or part['model']!=cal['models'][str(assignment[qid])]:raise ValueError('Exact native shard binding')
        else:
            with np.load(verify(features['features'][qid]),allow_pickle=False) as f:
                candidates,cost=proof.native_search(backend,q,f['keys'],f['features'],model['coefficients'])
            part=dict(query=q,model=cal['models'][str(assignment[qid])],candidates=candidates,cost=cost,
                first_stage_native100=True,labels_used=False,full_corpus_proof_features=features['features'][qid])
            io.write(path,part);engine.seal(path)
        rows[qid]=part['candidates'];timings[qid]=part['cost']
        io.write(root/'state.json',dict(status='NATIVE_NATIONAL_CROSS_FITTED_INITIAL100',queries=len(rows),total=len(panel),
            elapsed_s=time.monotonic()-started,query_id=qid,latest_cost=part['cost'],updated_unix=time.time()))
    if set(rows)!={q['query_id'] for q in panel} or len(rows)!=526:raise ValueError('All526 no reduced denominator')
    original=io.read(verify(io.read(proof.CACHE/'launch.json')['baseline_outputs']))
    previous=io.read(verify(io.read(identity.OUTPUT/'generation.json')['outputs']))
    outputs=dict(baseline=original['gemma2_lexical_geo'],v18_reference=previous['joint_lexical_oldgeo'],learned_native_ooc=rows)
    proof.authority(out);verify(cal['implementation'])
    io.write(out/'outputs.json',outputs);io.write(out/'native_timings.json',timings)
    io.write(out/'generation.json',dict(status='COMPLETE_SEALED_ALL526',outputs=record(out/'outputs.json'),queries=526,
        native_launch=record(root/'launch.json'),calibration=record(out/'calibration/generation.json'),
        timings=record(out/'native_timings.json'),no_prepool=True,no_downstream_reranking=True,
        labels_opened_in_native_worker=False,calibration_labels_used_elsewhere_for_training=True,
        no_positive_injected=True,closed_test_opened=False,full_encoder_oof=False))


def evaluate(out):
    proof.authority(out);verify(io.read(out/'calibration/generation.json')['implementation']);structural.evaluate(out)
    report=io.read(out/'report.json');rows=io.read(out/'outcomes.json');old={r['query_id']:r for r in rows['v18_reference']}
    fresh=rows['learned_native_ooc'];cost=list(io.read(out/'native_timings.json').values())
    controls=set(io.read(verify(io.read(out/'calibration/generation.json')['encoder_control_queries'])))
    report.update(status='MEASURED_LEARNED_NATIVE_NATIONAL_DEVELOPMENT_ALL526',
        paired_vs_v18=dict(gained=[r['query_id'] for r in fresh if r['exact'] and not old[r['query_id']]['exact']],
            lost=[r['query_id'] for r in fresh if not r['exact'] and old[r['query_id']]['exact']]),
        calibration_siren_crossfitted=True,encoder_train_sirens_disjoint=True,
        encoder_checkpoint_controls_not_fit_by_calibrator=True,checkpoint_control_queries=len(controls),
        primary_denominator_unchanged=526,full_encoder_oof=False,pristine_external_holdout=False,
        native_quantization_bits=proof.RECIPE['quantization_bits'],
        runtime=dict(p50=float(np.quantile([v['total_seconds'] for v in cost],.50)),
            p95=float(np.quantile([v['total_seconds'] for v in cost],.95)),p99=float(np.quantile([v['total_seconds'] for v in cost],.99)),
            cold_warm_not_controlled=True,includes_numeric_query_construction=True,
            excludes_one_shot_feature_preparation=True,not_full_production_slo=True))
    io.write(out/'report.json',report);io.write(out/'state.json',dict(status='COMPLETE_LEARNED_NATIVE_DEVELOPMENT',report=record(out/'report.json')))


def owned_stage(out,action):
    launch=proof.authority(out)
    module='scripts.embeddinggemma2_v30_native_features' if action=='prepare' else 'scripts.embeddinggemma2_v30_learned_query'
    command=[str(engine.CPU),'-u','-m',module,action,'--output',str(out)]
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',RAYON_NUM_THREADS='2',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TMPDIR=str(out))
    while True:
        proof.guard(launch['deadline_unix'])
        with (out/(action+'.log')).open('ab',buffering=0) as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
            io.write(out/(action+'_process.json'),dict(pid=child.pid,command=command,owned=True,started_unix=time.time()))
            while True:
                try:rc=child.wait(timeout=5);break
                except subprocess.TimeoutExpired:
                    if engine.INTERRUPTED or time.time()>=launch['deadline_unix']-30:
                        child.terminate()
                        try:rc=child.wait(timeout=25)
                        except subprocess.TimeoutExpired:child.kill();rc=child.wait()
                        io.write(out/(action+'_exit.json'),dict(exit_code=rc,pid=child.pid,own_child_exit_confirmed=True,deadline_or_user_stop=True))
                        raise TimeoutError('CONTROLLER_SIGNAL_ARTIFACTS_PRESERVED' if engine.INTERRUPTED else 'SEVEN_HOUR_END_CHECKPOINTS_PRESERVED')
        io.write(out/(action+'_exit.json'),dict(exit_code=rc,pid=child.pid,own_child_exit_confirmed=True,finished_unix=time.time()))
        if rc==75:
            io.write(out/'state.json',dict(status='REAL_RESOURCE_YIELD_SAME_BINDING',phase=action,updated_unix=time.time()));time.sleep(30);continue
        if rc:raise RuntimeError(f'{action} exit{rc}, no partial native recall; artifacts retained')
        return


def run(out,prepare_only=False):
    import fcntl
    lock=(io.HOME/(out.name+'.owned.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for action in (('prepare',) if prepare_only else ('prepare','calibrate','generate')):
        done={'prepare':'features_generation.json','calibrate':'calibration/generation.json','generate':'generation.json'}[action]
        if (out/done).exists():
            ex=io.read(out/(action+'_exit.json'))
            if ex['exit_code']!=0 or not ex['own_child_exit_confirmed']:raise ValueError('Completed artifacts require owned exit0')
            continue
        owned_stage(out,action)
    if not prepare_only:evaluate(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','prepare_only','calibrate','generate','evaluate']);p.add_argument('--output',type=Path,default=proof.OUTPUT);args=p.parse_args()
    signal.signal(signal.SIGTERM,engine.interrupted);signal.signal(signal.SIGINT,engine.interrupted)
    try:
        if args.action in ('run','prepare_only'):run(args.output,args.action=='prepare_only')
        else:globals()[args.action](args.output)
    except engine.ResourceYield as error:print(str(error),flush=True);sys.exit(75)
    except Exception as error:
        if args.output.exists():io.write(args.output/(args.action+'_error.json'),dict(error=repr(error),updated_unix=time.time()))
        raise


