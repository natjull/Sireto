"""Lossless per-query representation of V114's identical activity hypothesis.

The 3.5GiB provenance JSON exceeded the real RSS guard on whole-file loading.
Keep it immutable; stream each query and store exact SIREN references/float64
scores only. No pruning, precision change, coefficients or scientific retune.
"""
import argparse
import gc
from pathlib import Path
import signal
import sys
import time
import numpy as np

from scripts import embeddinggemma2_v114_native_activity_ablation as original
from scripts import embeddinggemma2_v99_initial_alignment_native as baseline
from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v14_full_pipeline as engine
from src.xgb_matcher.retrieval_v20_workbench import record,verify

OUTPUT=io.HOME/'v115/streamed_activity_ablation_20261013_a'
CACHE=original.CACHE
LEXICAL_FIELDS=original.LEXICAL_FIELDS
PROOF_FIELDS=original.PROOF_FIELDS
FEATURE_FIELDS=original.FEATURE_FIELDS
RECIPE=dict(original.RECIPE,representation='LOSSLESS_QUERY_SHARDS_S9_AND_FLOAT64_NO_PRUNING',
    source_provenance_json_unchanged=True)
guard=original.guard
ACTIVE=OUTPUT


def bind_runtime(out):
    global ACTIVE
    ACTIVE=out;original.bind_runtime(out)


def initialize(out):
    original.initialize(out)
    launch=io.read(out/'launch.json');io.write(out/'recipe.json',RECIPE)
    launch.update(implementation=record(Path(__file__)),recipe=record(out/'recipe.json'),
        previous_interrupted_launch=record(original.OUTPUT/'launch.json'),
        previous_generate_exit=record(original.OUTPUT/'generate_exit.json'))
    launch['helpers'].append(record(Path(original.__file__)));io.write(out/'launch.json',launch)


def authority(out):
    launch=original.authority(out)
    verify(launch['previous_interrupted_launch']);verify(launch['previous_generate_exit'])
    return launch


def prepare(out):
    import hashlib
    import ijson
    original.prepare(out);launch=authority(out);fg=io.read(out/'features_generation.json')
    context=verify(fg['activity_context']);panel=io.read(verify(launch['panel']));scope={q['query_id'] for q in panel}
    target=out/'activity_shards';target.mkdir(exist_ok=True);shards={};total=0
    with context.open('rb') as stream:
        for qid,entries in ijson.kvitems(stream,'',use_float=True):
            guard(launch['deadline_unix'])
            if qid not in scope or qid in shards:raise ValueError('Exact whole query scope')
            refs=np.array([e['siren_ref'] for e in entries],dtype='S9')
            confidence=np.array([e['confidence'] for e in entries],dtype=np.float64)
            if len(refs)!=len(entries) or len(set(refs.tolist()))!=len(refs):raise ValueError('No truncated/duplicate references')
            if any(r.decode()!=e['siren_ref'] for r,e in zip(refs,entries)):raise ValueError('Exact lookup references')
            if not np.isfinite(confidence).all() or (confidence<0).any() or (confidence>.500001).any():raise ValueError('Identical bounded scores')
            path=target/(hashlib.sha256(qid.encode()).hexdigest()+'.npz')
            np.savez_compressed(path,refs=refs,confidence=confidence)
            with np.load(path,allow_pickle=False) as check:
                if not np.array_equal(refs,check['refs']) or not np.array_equal(confidence,check['confidence']):raise ValueError('Lossless projection')
            shards[qid]=record(path);total+=len(entries)
            io.write(out/'projection_state.json',dict(status='LOSSLESS_PER_QUERY_SOURCE_PROJECTION',queries=len(shards),
                total=526,owner_references=total,updated_unix=time.time()))
            del entries,refs,confidence;gc.collect()
    if set(shards)!=scope:raise ValueError('All526 projections, no partial denominator')
    io.write(out/'features_generation.json',dict(fg,activity_shards=shards,projected_owner_references=total,
        projection_complete=True,no_score_or_reference_changed=True))


def native_search(backend,q,keys,features,coefficients):
    fg=io.read(ACTIVE/'features_generation.json')
    with np.load(verify(fg['activity_shards'][q['query_id']]),allow_pickle=False) as f:
        entries=[dict(siren_ref=r.decode(),confidence=float(c)) for r,c in zip(f['refs'],f['confidence'])]
    proxy=baseline.BackendProxy(backend,original.BeforeOnlyCollector(backend,q,float(coefficients[6]),entries))
    rows,cost=baseline.native_search(proxy,q,keys,features,coefficients)
    cost.update(activity_owner_context_references=len(entries),historical_activity_not_own_alias=True,
        activity_scores_lossless_per_query=True,activity_quantization_abs_error_bound=float(coefficients[6])/((1<<RECIPE['quantization_bits'])-1))
    return rows,cost


def stage(action,out):
    bind_runtime(out);baseline.source.pipeline.learned.proof=sys.modules[__name__]
    if action=='prepare':prepare(out)
    else:getattr(baseline.source.pipeline.learned,action)(out)


def run(out):
    # Override only child-module routing in a private runtime, not hashed files.
    import os,subprocess,fcntl
    with (out/'owned.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);deadline=io.read(out/'launch.json')['deadline_unix']
        for action,artifact in [('prepare','features_generation.json'),('generate','generation.json')]:
            complete=(out/artifact).exists() and (out/(action+'_exit.json')).exists()
            if complete:
                ex=io.read(out/(action+'_exit.json'))
                if ex['exit_code'] or not ex['own_child_exit_confirmed']:raise ValueError('Owned complete stage')
                continue
            while True:
                guard(deadline)
                command=[str(engine.CPU),'-u','-m','scripts.embeddinggemma2_v115_streamed_activity_ablation',action,'--output',str(out)]
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',RAYON_NUM_THREADS='2',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TMPDIR=str(out/'spill'))
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
                if rc==75:time.sleep(10);continue
                if rc:raise RuntimeError(f'{action} exit{rc}, no partial evaluation')
                break
        stage('evaluate',out)
        r=io.read(out/'report.json');r.update(activity_context_not_identity=True,no_new_training=True,
            source_parser_single_object_scope=True,production_promotion_allowed=False,comparison_baseline='FROZEN_V99',
            exact_activity_hypothesis_same_as_v114=True,source_representation_only_changed=True)
        io.write(out/'report.json',r);io.write(out/'completion.json',dict(status='COMPLETE_OWNED_ALL526_FROZEN_ACTIVITY_ABLATION',
            report=record(out/'report.json'),finished_unix=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['initialize','run','prepare','generate','evaluate']);p.add_argument('--output',type=Path,default=OUTPUT);a=p.parse_args()
    signal.signal(signal.SIGTERM,engine.interrupted);signal.signal(signal.SIGINT,engine.interrupted)
    try:
        if a.action=='initialize':initialize(a.output)
        elif a.action=='run':run(a.output)
        else:stage(a.action,a.output)
    except engine.ResourceYield as e:print(str(e),flush=True);sys.exit(75)
    except Exception as e:
        if a.output.exists():io.write(a.output/(a.action+'_error.json'),dict(error=repr(e),updated_unix=time.time()))
        raise
