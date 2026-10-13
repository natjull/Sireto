"""Source-only coherent street-content proofs, native national initial100.

Current own public addresses, never CRM/GT aliases, HQ/sibling coordinates or
synthetic points. Original historical evidence and V32 serving weights stay.
No partial-ratio street matching, name-independent address proof. New learner
residual uses nested SIREN-held-out baseline offsets, not in-sample margins.
"""
import argparse
import gc
import hashlib
import os
from pathlib import Path
import signal
import sys
import time
import numpy as np
from scripts import embeddinggemma2_v81_indirect_lexical_query as source
from scripts import embeddinggemma2_v78_nested_residual_offset as nested
from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v14_full_pipeline as engine
from src.xgb_matcher.retrieval_v20_workbench import record,verify

OUTPUT=io.HOME/'v94/coherent_address_recovery_20261013_a'
CACHE=source.CACHE
LEXICAL_FIELDS=source.LEXICAL_FIELDS
PROOF_FIELDS=source.PROOF_FIELDS+['own_current_content_street_x_commune',
    'own_current_content_street_x_number_suffix_commune','own_current_content_street_x_direct_geo']
FEATURE_FIELDS=LEXICAL_FIELDS+PROOF_FIELDS
BASE_AT=source.pipeline.BASE_AT
EXTRA_AT=np.array([6]+list(range(29,len(FEATURE_FIELDS))))
RECIPE=dict(source.RECIPE,hypothesis='NAME_INDEPENDENT_COHERENT_OWN_ADDRESS_CONTENT_RESIDUAL',
    address_source='CURRENT_PUBLIC_OWN_STOCK_ESTABLISHMENT_ONLY_NO_INHERITANCE',
    address_content='TOKEN_SORT_FULL_RATIO_NOT_PARTIAL_RATIO',
    generic_street_type_and_articles_removed_from_both_sides=True,
    number_suffix_grade='SAME_NUMBER_EXACT_SUFFIX_1_MISSING_SUFFIX_POINT8_ELSE0_NO_EXCLUSION',
    current_address_disagreement_not_a_hard_filter=True,
    old_historical_views_and_full_universe_unchanged=True,
    residual_training_offsets='NESTED_OUTER_AND_INNER_SIREN_DISJOINT',
    baseline_training_margin_in_sample_not_nested_oof=False,
    no_new_encoder_fit=True,full_encoder_oof=False,
    source_recipe_diagnostic_only=True,production_promotion_allowed=False)
PREPARED=io.HOME/'v89/coherent_address_content_20261013_a'
ACTIVE=OUTPUT
guard=source.guard
feature_path=source.feature_path
lexical_parts=source.lexical_parts
native_search=source.native_search
TYPES=set('RUE AVENUE BOULEVARD ROUTE CHEMIN IMPASSE PLACE ALLEE SQUARE QUAI PASSAGE SENTIER RUELLE'.split())
ARTICLES=set('DE DU DES LA LE LES D L'.split())


def content(street):
    words=source.original.structural.normalize(street or '',street=True).split()
    if words and words[0] in TYPES:words.pop(0)
    return ' '.join(sorted(w for w in words if w not in ARTICLES))


def bind_runtime(out):
    global ACTIVE
    ACTIVE=out;source.bind_runtime(out)


def initialize(out):
    source.initialize(out);launch=io.read(out/'launch.json');io.write(out/'recipe.json',RECIPE)
    ex=io.read(source.OUTPUT/'prepare_exit.json')
    if ex['exit_code'] or not ex['own_child_exit_confirmed']:raise ValueError('Complete official source')
    nx=io.read(nested.OUTPUT/'calibrate_exit.json')
    if nx['exit_code'] or not nx['own_child_exit_confirmed']:raise ValueError('Complete nested source')
    launch.update(implementation=record(Path(__file__)),recipe=record(out/'recipe.json'),
        prepared_v81=record(source.OUTPUT/'features_generation.json'),
        original_baseline_calibration=record(source.pipeline.base.OUTPUT/'calibration/generation.json'),
        original_nested_calibration=record(nested.OUTPUT/'calibration/generation.json'),nested_exit=record(nested.OUTPUT/'calibrate_exit.json'))
    launch['helpers'].extend([record(Path(source.__file__)),record(Path(nested.__file__))])
    px=io.read(PREPARED/'prepare_exit.json');cx=io.read(PREPARED/'calibrate_exit.json')
    if px['exit_code'] or not px['own_child_exit_confirmed'] or cx['exit_code']!=1 or not cx['own_child_exit_confirmed']:raise ValueError('Owned failed fit after complete preparation')
    launch.update(recovered_features=record(PREPARED/'features_generation.json'),recovered_calibration_launch=record(PREPARED/'calibration/launch.json'),recovered_calibration_exit=record(PREPARED/'calibrate_exit.json'))
    io.write(out/'launch.json',launch)


def authority(out):
    launch=source.authority(out)
    for n in ('prepared_v81','original_baseline_calibration','original_nested_calibration','nested_exit','recovered_features','recovered_calibration_launch','recovered_calibration_exit'):verify(launch[n])
    return launch


def prepare(out):
    launch=authority(out);fg=io.read(verify(launch['recovered_features']))
    for ref in fg['features'].values():verify(ref)
    io.write(out/'features_generation.json',fg)


def fit_residual(difference,weights,l2):
    # Called only after all TRAIN shards exist; bind their exact order, reuse
    # already completed20 doubly-purged baselines, never another fold's label.
    launch=authority(ACTIVE);root=ACTIVE/'calibration';original=io.read(verify(launch['original_baseline_calibration']))
    excluded=set(io.read(verify(original['encoder_control_queries'])))
    ordered=[q['query_id'] for q in io.read(verify(launch['panel'])) if q['query_id'] not in excluded]
    shards={q:record(engine.sealed(root/(hashlib.sha256(q.encode()).hexdigest()+'.npz'))) for q in ordered}
    io.write(root/'pair_binding.json',dict(ordered_training_queries=ordered,training_shards=shards,
        newly_mined_train_only=True,old_baseline_pair_values_not_modified=True))
    inner=root/'inner_baselines';inner.mkdir(exist_ok=True)
    for p in (nested.OUTPUT/'calibration/inner_baselines').glob('outer_*_inner_*.json'):
        if p.name.endswith('.seal.json'):continue
        dest=inner/p.name
        if not dest.exists():os.link(engine.sealed(p),dest);engine.seal(dest)
        d=io.read(dest)
        if d['source_calibration']['sha256']!=launch['original_baseline_calibration']['sha256']:raise ValueError('Same doubly-purged baseline authority')
    nested.ACTIVE=ACTIVE;nested.authority=authority;nested.BASE_AT=BASE_AT;nested.EXTRA_AT=EXTRA_AT;nested.FEATURE_FIELDS=FEATURE_FIELDS
    return nested.fit_residual(difference,weights,l2)


def stage(action,out):
    bind_runtime(out);source.pipeline.learned.proof=sys.modules[__name__]
    source.pipeline.learned.fit_pairwise=fit_residual
    source.pipeline.learned.RECIPE=dict(source.pipeline.learned.RECIPE,model='FROZEN_V32_NESTED_ADDRESS_CONTENT_RESIDUAL',
        feature_fields=FEATURE_FIELDS,nested_baseline_offsets=True,no_hyperparameter_grid=True)
    if action=='prepare':prepare(out)
    elif action=='calibrate':
        launch=authority(out);prior=io.read(verify(launch['recovered_calibration_launch']))
        if io.read(verify(prior['recipe']))['feature_fields']!=FEATURE_FIELDS:raise ValueError('Exact same prepared TRAIN schema')
        excluded=set(io.read(verify(prior['encoder_control_queries'])))
        root=out/'calibration';root.mkdir(exist_ok=True)
        for q in io.read(verify(launch['panel'])):
            if q['query_id'] in excluded:continue
            name=hashlib.sha256(q['query_id'].encode()).hexdigest()+'.npz';p=engine.sealed(PREPARED/'calibration'/name)
            if not p:raise ValueError('All375 original TRAIN shards required')
            dest=root/name
            if not dest.exists():os.link(p,dest);engine.seal(dest)
        source.pipeline.learned.calibrate(out)
    else:getattr(source.pipeline.learned,action)(out)
    if action=='evaluate':
        r=io.read(out/'report.json');r.update(eligibility=RECIPE['eligibility'],production_promotion_allowed=False,
            old_baseline_exact=True,address_content_is_own_source_only=True,nested_offsets=True)
        io.write(out/'report.json',r)


def run(out):
    import fcntl,subprocess
    lock=(io.HOME/(out.name+'.owned.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    deadline=source.pipeline.corporate.DEADLINE
    for action,artifact in [('prepare','features_generation.json'),('calibrate','calibration/generation.json'),('generate','generation.json')]:
        if (out/artifact).exists():
            ex=io.read(out/(action+'_exit.json'))
            if ex['exit_code'] or not ex['own_child_exit_confirmed']:raise ValueError('Owned complete stage')
            continue
        while True:
            guard(deadline);command=[str(engine.CPU),'-u','-m','scripts.embeddinggemma2_v94_coherent_address_recovery',action,'--output',str(out)]
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
            if engine.INTERRUPTED or time.time()>=deadline-30:raise TimeoutError('USER_STOP_OR_EIGHT_HOUR_END')
            if rc==75:time.sleep(30);continue
            if rc:raise RuntimeError(f'{action} exit {rc}; no partial recall')
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

