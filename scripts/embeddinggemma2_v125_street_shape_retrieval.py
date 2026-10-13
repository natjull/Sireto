"""One frozen full-panel geographic ablation, before the native collector100.

Replace point-only proximity by max(old, distance-to-actual-street-addresses).
Only existing full unpruned numeric proof coordinates are used. No brand hint,
target injection, fixed candidate pool, downstream ranker or new fitting.
Unchanged query inputs reuse the exact sealed baseline outputs by equivalence.
"""
import argparse
import fcntl
import gc
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np

from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v14_full_pipeline as engine
from scripts import embeddinggemma2_v115_streamed_activity_ablation as baseline
from scripts.embeddinggemma2_v123_street_geometry_probe import segment_distance
from src.xgb_matcher.hierarchical_retrieval import TantivyBackend
from src.xgb_matcher.retrieval_v20_workbench import record,verify

OUTPUT=io.HOME/'v125/street_cloud_retrieval_20261013_b'
SOURCE=io.HOME/'v126/ban_street_clouds_20261013_a'
BASE=baseline.OUTPUT
RECIPE=dict(hypothesis='FROZEN_V115_WITH_UNIFORM_WEAK_STREET_BAN_POSITION_CLOUD_GEOGRAPHY',
    eligibility='DIAGNOSTIC_ONLY_INHERITED_POSTCUTOFF_BAN_INELIGIBLE',
    own_geo='MAX_ORIGINAL_AND_HALF_DIV_ONE_PLUS_DISTANCE_TO_NEAREST_ACTUAL_STREET_POSITION_OVER_300M',
    no_interpolated_road_or_property_claim=True,
    original_strong_house_geometry_unchanged=True,source_names_not_company_aliases=True,
    numeric_scope='IDENTICAL_FULL_UNPRUNED_V115_NUMERIC_PROOF_UNIVERSE',
    original_national_lexical_unrestricted=True,no_new_external_footprint_expansion=True,
    changed_proof_fields=[5,6,7,16,17,19],own_and_parent_context_distinct=True,
    no_refitting=True,no_new_encoder_fit=True,no_fixed_prepool=True,native_cap=100,
    unchanged_queries='EXACT_SEALED_V115_SHARD_REUSE_PROVED_BY_EMPTY_STREET_GEOMETRY',
    no_partial_recall=True,production_promotion_allowed=False)
SHAPES=None;XY=None;KEYS=None


def guard(deadline):
    if time.time()>=deadline:raise TimeoutError('Owned experiment timeout, preserve artifacts')
    engine.guard()


def authority(out):
    launch=io.read(out/'launch.json')
    for field in ('implementation','recipe','panel','source_generation','source_exit','baseline_generation',
                  'baseline_exit','baseline_report','baseline_features','baseline_calibration'):
        verify(launch[field])
    guard(launch['deadline_unix']);return launch


def initialize(out):
    engine.guard()
    if out.exists():raise ValueError('Fresh frozen experiment')
    sg=io.read(SOURCE/'generation.json');sx=io.read(SOURCE/'owned_tool_exit.json')
    bg=io.read(BASE/'generation.json');bx=io.read(BASE/'generate_exit.json')
    if sg['status']!='COMPLETE_ALL526_QUERY_ONLY_STREET_SOURCE' or sx['exit_code']:
        raise ValueError('Complete observed source exit required')
    if bg['status']!='COMPLETE_SEALED_ALL526' or bx['exit_code'] or not bx['own_child_exit_confirmed']:
        raise ValueError('Complete owned original pipeline')
    verify(sg['shapes']);verify(bg['outputs'])
    out.mkdir(parents=True);(out/'spill').mkdir();(out/'calibration').mkdir()
    io.write(out/'recipe.json',RECIPE)
    launch=dict(io.read(BASE/'launch.json'),implementation=record(Path(__file__)),recipe=record(out/'recipe.json'),
        source_generation=record(SOURCE/'generation.json'),source_exit=record(SOURCE/'owned_tool_exit.json'),
        baseline_generation=record(BASE/'generation.json'),baseline_exit=record(BASE/'generate_exit.json'),
        baseline_report=record(BASE/'report.json'),baseline_features=record(BASE/'features_generation.json'),
        baseline_calibration=record(BASE/'calibration/generation.json'),deadline_unix=time.time()+6*3600,
        current_human_geographic_request=True,previous_night_not_extended=True,created_unix=time.time())
    launch['helpers'].extend([record(Path(baseline.__file__)),record(io.ROOT/'scripts/embeddinggemma2_v123_street_geometry_probe.py')])
    io.write(out/'launch.json',launch)
    io.write(out/'features_generation.json',io.read(BASE/'features_generation.json'))
    io.write(out/'calibration/generation.json',io.read(BASE/'calibration/generation.json'))


def modified_features(launch,q,keys,features,shapes):
    global KEYS,XY
    if KEYS is None:
        KEYS=np.load(verify(launch['keys']),mmap_mode='r',allow_pickle=False)
        XY=np.load(verify(launch['xy']),mmap_mode='r',allow_pickle=False)
    at=np.searchsorted(KEYS,keys);valid=at<len(KEYS);valid[valid]&=KEYS[at[valid]]==keys[valid]
    positions=np.flatnonzero(valid);coords=np.asarray(XY[at[positions]],dtype=float)
    good=np.isfinite(coords).all(axis=1)&(coords[:,0]>=0)&(coords[:,0]<=1300000)&(coords[:,1]>=6000000)&(coords[:,1]<=7300000)
    positions=positions[good];coords=coords[good]
    distance=np.full(len(coords),np.inf)
    for line in shapes[q['query_id']]['lines_l93']:
        distance=np.minimum(distance,segment_distance(coords,np.asarray(line,dtype=float)))
    f=features.copy()
    f[positions,5]=np.maximum(f[positions,5],.5/(1+distance/300.))
    f[:,6]=f[:,0]*f[:,5];f[:,7]=f[:,2]*f[:,5]
    _,group=np.unique(keys//100000,return_inverse=True);parent=np.zeros(int(group.max())+1,dtype=np.float32)
    np.maximum.at(parent,group,f[:,5]);f[:,16]=np.maximum(f[:,16],parent[group])
    f[:,17]=f[:,1]*f[:,16];f[:,19]=f[:,18]*f[:,16]
    if not np.isfinite(f).all() or (f<0).any() or (f>1.000001).any():raise ValueError('Bounded shape geography')
    return f,int(np.count_nonzero(f[:,5]>features[:,5]+1e-7))


def generate(out):
    launch=authority(out);baseline.bind_runtime(out)
    sg=io.read(verify(launch['source_generation']));shapes=io.read(verify(sg['shapes']))
    panel=io.read(verify(launch['panel']));cal=io.read(out/'calibration/generation.json')
    models={k:io.read(verify(v)) for k,v in cal['models'].items()};assignment=io.read(verify(cal['assignment']))
    features=io.read(verify(launch['baseline_features']));bg=io.read(verify(launch['baseline_generation']))
    original=io.read(verify(bg['outputs']));old_timings=io.read(verify(bg['timings']))
    backend=TantivyBackend(io.INDEX)
    if backend.searcher.num_docs!=82772691:raise ValueError('Original full national index')
    root=out/'native';root.mkdir(exist_ok=True);rows={};costs={};recomputed=0
    for q in panel:
        guard(launch['deadline_unix']);qid=q['query_id'];fold=str(assignment[qid]);model=models[fold]
        if qid not in model['heldout_queries'] or qid in model['train_queries']:raise ValueError('Heldout calibrator')
        path=root/(hashlib.sha256(qid.encode()).hexdigest()+'.json');saved=engine.sealed(path)
        if saved:
            part=io.read(saved)
            if part['query']!=q or part['model']!=cal['models'][fold]:raise ValueError('Exact saved binding')
        elif not shapes[qid]['lines_l93']:
            part=dict(query=q,model=cal['models'][fold],candidates=original['learned_native_ooc'][qid],
                cost=dict(old_timings[qid],unchanged_query_outputs_reused=True),
                source=record(BASE/'generation.json'),equivalence='NO_STREET_GEOMETRY_NO_CHANGED_QUERY_SCORE',labels_used=False)
            io.write(path,part);engine.seal(path)
        else:
            with np.load(verify(features['features'][qid]),allow_pickle=False) as f:
                start=time.monotonic();x,changed=modified_features(launch,q,f['keys'],f['features'],shapes)
                shape_seconds=time.monotonic()-start
                candidates,cost=baseline.native_search(backend,q,f['keys'],x,model['coefficients'])
                cost.update(street_shape_seconds=shape_seconds,street_shape_improved_leaves=changed,
                    total_seconds=cost['total_seconds']+shape_seconds,unchanged_query_outputs_reused=False)
                del x
            part=dict(query=q,model=cal['models'][fold],candidates=candidates,cost=cost,
                first_stage_native100=True,no_fixed_pool=True,labels_used=False)
            io.write(path,part);engine.seal(path);gc.collect()
        rows[qid]=part['candidates'];costs[qid]=part['cost'];recomputed+=not part['cost']['unchanged_query_outputs_reused']
        io.write(root/'state.json',dict(status='ALL526_STREET_SHAPE_INITIAL100',queries=len(rows),total=526,
            recomputed=recomputed,query_id=qid,updated_unix=time.time()))
    if len(rows)!=526 or any(len(r)>100 or len({p['siret'] for p in r})!=len(r) for r in rows.values()):
        raise ValueError('All526 unique100 without filtered denominator')
    authority(out);outputs=dict(original,learned_native_ooc=rows)
    io.write(out/'outputs.json',outputs);io.write(out/'native_timings.json',costs)
    io.write(out/'generation.json',dict(status='COMPLETE_SEALED_ALL526',queries=526,outputs=record(out/'outputs.json'),
        timings=record(out/'native_timings.json'),no_prepool=True,no_downstream_reranking=True,no_positive_injected=True,
        labels_opened_in_native_worker=False,closed_test_opened=False,full_encoder_oof=False,
        source=launch['source_generation'],recomputed_queries=recomputed,unchanged_reused_queries=526-recomputed))


def run(out):
    with (out/'owned.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            launch=authority(out)
            command=[str(engine.CPU),'-u','-m','scripts.embeddinggemma2_v125_street_shape_retrieval','generate','--output',str(out)]
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(out/'spill'),RAYON_NUM_THREADS='2',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
            with (out/'generate.log').open('ab',buffering=0) as log:
                child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
                io.write(out/'generate_process.json',dict(pid=child.pid,command=command,owned=True,started_unix=time.time()))
                while child.poll() is None:
                    if engine.INTERRUPTED or time.time()>=launch['deadline_unix']-30:
                        child.terminate()
                        try:child.wait(timeout=25)
                        except subprocess.TimeoutExpired:child.kill();child.wait()
                    time.sleep(5)
                rc=child.wait()
            io.write(out/'generate_exit.json',dict(exit_code=rc,pid=child.pid,own_child_exit_confirmed=True,finished_unix=time.time()))
            if rc==75:time.sleep(10);continue
            if rc:raise RuntimeError(f'Owned generation exit{rc}; no partial evaluation')
            break
        from scripts import embeddinggemma2_v32_learned_query as evaluator
        evaluator.proof=sys.modules[__name__];evaluator.evaluate(out)
        r=io.read(out/'report.json');r.update(eligibility=RECIPE['eligibility'],production_promotion_allowed=False,
            no_new_fitting=True,unchanged_inputs_reused_not_partial_evaluation=True,cold_warm_not_controlled=True)
        io.write(out/'report.json',r);io.write(out/'completion.json',dict(status='COMPLETE_OWNED_STREET_SHAPE_ABLATION_ALL526',report=record(out/'report.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['initialize','run','generate']);p.add_argument('--output',type=Path,default=OUTPUT);a=p.parse_args()
    signal.signal(signal.SIGTERM,engine.interrupted);signal.signal(signal.SIGINT,engine.interrupted)
    try:globals()[a.action](a.output)
    except engine.ResourceYield as e:print(str(e),flush=True);sys.exit(75)
