"""Independent own-site BAN geometry diagnostic before ONE native100.

V32 full proof universe and frozen Gemma; append missing-coordinate evidence.
This measures a diagnostic pipeline, not production authorization for BAN.
No target-directed point, site pool, label-side facts or downstream reranker.
"""
import argparse
import gc
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np

from scripts import embeddinggemma2_v56_corporate_site as mandate
from scripts import embeddinggemma2_v32_native_features as base
from scripts import embeddinggemma2_v32_learned_query as learned
from scripts import embeddinggemma2_v23_native_features as original
from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v14_full_pipeline as engine
from src.xgb_matcher.retrieval_v20_workbench import record,verify

OUTPUT=io.HOME/'v68/filled_own_site_geo_20261013_a'
ATLAS=io.HOME/'v67/public_site_ban_hash_finish_20261013_a'
CACHE=original.CACHE
LEXICAL_FIELDS=list(base.LEXICAL_FIELDS)
PROOF_FIELDS=list(original.PROOF_FIELDS)+['filled_own_site_geo','name_x_filled_own_geo',
    'site_x_filled_own_geo','parent_identity_x_filled_own_geo','own_address_x_filled_own_geo']
FEATURE_FIELDS=LEXICAL_FIELDS+PROOF_FIELDS
RECIPE=dict(original.RECIPE,feature_fields=FEATURE_FIELDS,
    hypothesis='OWN_SITE_MISSING_COORDINATE_FILL_ONLY_NO_CORPORATE_FEATURES',
    source_atlas_scope='ALL_NATIONAL_MISSING_PUBLIC_OWN_ADDRESSES_NOT_GT_SEEDED',
    new_numeric_scope='ORIGINAL_FULL_UNPRUNED_PROOF_UNIVERSE_NOT_NEW_FOOTPRINT_EXPANSION',
    missing_coordinates_not_negative_features=True, original_coordinates_not_overwritten=True,
    own_geo_score='SOURCE_RELIABILITY_X_QUERY_RELIABILITY_DIV_1_PLUS_DISTANCE_OVER_300_PLUS_BOTH_UNCERTAINTIES',
    weak_street_not_house_precision=True,eligibility='DIAGNOSTIC_ONLY_BAN_SCORE_INELIGIBLE',
    production_promotion_allowed=False, no_new_encoder_fit=True,deadline_unix=mandate.DEADLINE)


def guard(deadline):mandate.guard(deadline)
def feature_path(out,qid):return original.feature_path(out,qid)
def bind_runtime(out):base.bind_runtime(out)
def lexical_parts(backend,q):return base.lexical_parts(backend,q)


def initialize(out):
    generation=io.read(ATLAS/'generation.json');ex=io.read(ATLAS/'owned_exit.json')
    if generation['status']!='COMPLETE_NATIONAL_PUBLIC_OWN_ADDRESS_GEOMETRY_ATLAS' or ex['exit_code'] or not ex['own_child_exit_confirmed']:
        raise ValueError('Complete owned source atlas before experiment')
    mandate.initialize(out)
    io.write(out/'recipe.json',RECIPE);launch=io.read(out/'launch.json')
    launch.update(implementation=record(Path(__file__)),recipe=record(out/'recipe.json'),
        atlas_generation=record(ATLAS/'generation.json'),atlas_exit=record(ATLAS/'owned_exit.json'),
        atlas_geometry=generation['geometry'],eligibility=RECIPE['eligibility'],production_promotion_allowed=False)
    launch['helpers'].append(record(Path(mandate.__file__)));io.write(out/'launch.json',launch)


def authority(out):
    launch=mandate.authority(out)
    for field in ('atlas_generation','atlas_exit','atlas_geometry'):verify(launch[field])
    return launch


def prepare(out):
    import pyarrow.parquet as pq
    launch=authority(out);source=io.read(verify(launch['prepared_generation']))
    table=pq.read_table(verify(launch['atlas_geometry']),use_threads=False)
    atlas_keys=table['site_key'].to_numpy();ax=table['x'].to_numpy();ay=table['y'].to_numpy()
    uncertainty=table['uncertainty_m'].to_numpy();reliability=table['reliability'].to_numpy()
    if np.any(atlas_keys[1:]<=atlas_keys[:-1]) or len(atlas_keys)!=3310914:raise ValueError('Exact all-source unique ordered atlas')
    geometries={r['query_id']:r['geometry'] for r in io.read(verify(launch['geometry']))}
    panel=io.read(verify(launch['panel']));refs={};total=0;started=time.monotonic()
    for q in panel:
        guard(launch['deadline_unix']);qid=q['query_id'];path=feature_path(out,qid)
        if not engine.sealed(path):
            with np.load(verify(source['features'][qid]),allow_pickle=False) as f:keys=f['keys'];old=f['features']
            x=np.zeros((len(keys),len(PROOF_FIELDS)),dtype=np.float32);x[:,:len(original.PROOF_FIELDS)]=old
            g=geometries[qid]
            at=np.searchsorted(atlas_keys,keys);found=at<len(atlas_keys);found[found]&=atlas_keys[at[found]]==keys[found]
            loc=np.flatnonzero(found)
            if g['spatial_crs_compatible'] and len(loc):
                indices=at[loc];distance=np.hypot(ax[indices]-g['ban_x_l93'],ay[indices]-g['ban_y_l93'])
                query_quality=.5 if g['ban_declared_weak'] else 1.
                values=reliability[indices]*query_quality/(1+distance/(300+max(0,g['ban_uncertainty_m'])+uncertainty[indices]))
                start=len(original.PROOF_FIELDS);x[loc,start]=values
                for offset,field in enumerate((2,0,1,3),1):x[loc,start+offset]=values*old[loc,field]
            if not np.isfinite(x).all() or (x<0).any() or (x>1.000001).any():raise ValueError('Finite bounded own-site evidence')
            temp=path.with_suffix('.pending.npz');np.savez(temp,keys=keys,features=x);os.replace(temp,path);engine.seal(path)
            total+=len(keys);del keys,old,x;gc.collect()
        refs[qid]=record(path);engine.progress(out,'OWN_SITE_SOURCE_ATLAS_DIAGNOSTIC_PROOFS',len(refs),total=526,elapsed_s=time.monotonic()-started)
    authority(out)
    io.write(out/'features_generation.json',dict(source,features=refs,full_feature_rows=total,
        source_atlas=launch['atlas_generation'],labels_used=False,production_promotion_allowed=False,
        eligibility=RECIPE['eligibility'],only_missing_original_coordinates=True))


def native_search(backend,q,keys,features,coefficients):
    old=(base.FEATURE_FIELDS,base.PROOF_FIELDS)
    try:
        base.FEATURE_FIELDS=FEATURE_FIELDS;base.PROOF_FIELDS=PROOF_FIELDS
        return base.native_search(backend,q,keys,features,coefficients)
    finally:base.FEATURE_FIELDS,base.PROOF_FIELDS=old


def stage(action,out):
    learned.proof=sys.modules[__name__]
    learned.RECIPE=dict(learned.RECIPE,model='MISSING_OWN_SITE_GEOMETRY_NATIVE_QUERY_DIAGNOSTIC',
        feature_fields=FEATURE_FIELDS,eligibility=RECIPE['eligibility'],production_promotion_allowed=False)
    if action=='prepare':prepare(out)
    else:getattr(learned,action)(out)
    if action=='evaluate':
        report=io.read(out/'report.json');report.update(eligibility=RECIPE['eligibility'],production_promotion_allowed=False,
            source_atlas=record(ATLAS/'generation.json'),independent_geo_pipeline_not_corporate_combination=True)
        io.write(out/'report.json',report);io.write(out/'state.json',dict(status='COMPLETE_DIAGNOSTIC_ONLY_BAN_SCORE_INELIGIBLE',report=record(out/'report.json')))


def run(out):
    import fcntl
    lock=(io.HOME/(out.name+'.owned.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for action,artifact in [('prepare','features_generation.json'),('calibrate','calibration/generation.json'),('generate','generation.json')]:
        if (out/artifact).exists():
            ex=io.read(out/(action+'_exit.json'))
            if ex['exit_code'] or not ex['own_child_exit_confirmed']:raise ValueError('Owned completion required')
            continue
        while True:
            guard(mandate.DEADLINE)
            command=[str(engine.CPU),'-u','-m','scripts.embeddinggemma2_v68_filled_site_geo',action,'--output',str(out)]
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',RAYON_NUM_THREADS='2',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TMPDIR=str(out))
            with (out/(action+'.log')).open('ab',buffering=0) as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
                io.write(out/(action+'_process.json'),dict(pid=child.pid,command=command,owned=True,started_unix=time.time()))
                while child.poll() is None:
                    if engine.INTERRUPTED or time.time()>=mandate.DEADLINE-30:
                        child.terminate()
                        try:child.wait(timeout=25)
                        except subprocess.TimeoutExpired:child.kill();child.wait()
                        break
                    time.sleep(5)
                rc=child.wait()
            io.write(out/(action+'_exit.json'),dict(exit_code=rc,pid=child.pid,own_child_exit_confirmed=True,finished_unix=time.time()))
            if engine.INTERRUPTED or time.time()>=mandate.DEADLINE-30:raise TimeoutError('EIGHT_HOUR_END_ARTIFACTS_PRESERVED')
            if rc==75:time.sleep(30);continue
            if rc:raise RuntimeError(f'{action} owned exit {rc}; no partial recall')
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
