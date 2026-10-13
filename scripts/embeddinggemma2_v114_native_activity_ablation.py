"""One frozen-V99 ablation: typed historical activity before native100.

No capped channel pool, no model fitting or downstream changes. Source-wide
token union is historical company context, not a phrase/own-site/identity claim.
All526 generation sealed before labels; original heldout calibrators reused.
"""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np

from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v99_initial_alignment_native as baseline
from scripts import embeddinggemma2_v14_full_pipeline as engine
from src.xgb_matcher.retrieval_v20_workbench import record,verify

OUTPUT=io.HOME/'v114/native_activity_ablation_20261013_a'
SOURCE=io.HOME/'v112/bodacc_activity_context_20261013_a'
CACHE=baseline.CACHE
LEXICAL_FIELDS=baseline.LEXICAL_FIELDS
PROOF_FIELDS=baseline.PROOF_FIELDS
FEATURE_FIELDS=baseline.FEATURE_FIELDS
RECIPE=dict(baseline.RECIPE,hypothesis='FROZEN_V99_PLUS_HISTORICAL_PUBLISHED_ACTIVITY_TOKEN_CONTEXT',
    additional_context='QUERY_WORD_SOURCE_OWNER_IDF_COVERAGE_CAPPED_05',
    historical_owner_token_union=True,matching_words_not_necessarily_same_announcement=True,
    activity_is_not_name_or_identity=True,additional_admin_support=True,
    additional_weight='FROZEN_ACTOR_COEFFICIENT_NO_REFIT',source_single_object_parser_scope=True,
    no_fixed_prepool=True,no_model_training=True,all526_consumed_development=True)
ACTIVE=OUTPUT
ACTIVITY=None


def guard(deadline):
    if time.time()>=deadline:raise TimeoutError('Owned experiment safety timeout, preserve artifacts')
    engine.guard()


def bind_runtime(out):
    global ACTIVE
    ACTIVE=out;baseline.bind_runtime(out)


def initialize(out):
    if out.exists():raise ValueError('Fresh experiment namespace')
    engine.guard()
    prior=baseline.OUTPUT
    g=io.read(prior/'generation.json');ex=io.read(prior/'generate_exit.json')
    if g['status']!='COMPLETE_SEALED_ALL526' or ex['exit_code'] or not ex['own_child_exit_confirmed']:
        raise ValueError('Whole owned frozen baseline')
    verify(g['outputs'])
    out.mkdir(parents=True);(out/'calibration').mkdir();(out/'spill').mkdir()
    io.write(out/'recipe.json',RECIPE)
    launch=dict(io.read(prior/'launch.json'),implementation=record(Path(__file__)),
        recipe=record(out/'recipe.json'),created_unix=time.time(),deadline_unix=time.time()+6*3600,
        previous_mandate_not_extended=True,renewed_human_task=True,
        frozen_baseline_generation=record(prior/'generation.json'),
        frozen_baseline_exit=record(prior/'generate_exit.json'),
        frozen_features=record(prior/'features_generation.json'),
        frozen_models_generation=record(prior/'calibration/generation.json'),
        activity_source_directory=str(SOURCE),no_closed_test_opened=True)
    launch['helpers']=[*launch['helpers'],record(Path(baseline.__file__))]
    io.write(out/'launch.json',launch)


def authority(out):
    launch=baseline.authority(out)
    for key in ('frozen_baseline_generation','frozen_baseline_exit','frozen_features','frozen_models_generation'):
        verify(launch[key])
    guard(launch['deadline_unix'])
    return launch


def prepare(out):
    launch=authority(out)
    while not (SOURCE/'generation.json').exists() or not (SOURCE/'owned_tool_exit.json').exists():
        guard(launch['deadline_unix']);time.sleep(5)
    g=io.read(SOURCE/'generation.json');ex=io.read(SOURCE/'owned_tool_exit.json')
    if g['status']!='COMPLETE_WHOLE_SOURCE_QUERY_ONLY_BODACC_ACTIVITY_CONTEXT' or ex['exit_code'] or not ex['own_tool_exit_confirmed']:
        raise ValueError('Complete source and observed owned tool exit required')
    if g['query_panel']!=launch['panel']:raise ValueError('Exact same query panel')
    verify(g['context']);verify(g['implementation'])
    features=io.read(verify(launch['frozen_features']))
    io.write(out/'features_generation.json',dict(features,activity_context=g['context'],
        activity_generation=record(SOURCE/'generation.json'),activity_exit=record(SOURCE/'owned_tool_exit.json'),
        all_original_numeric_values_unchanged=True,labels_used=False))
    cal=io.read(verify(launch['frozen_models_generation']))
    for ref in cal['models'].values():verify(ref)
    verify(cal['assignment']);verify(cal['implementation'])
    io.write(out/'calibration/generation.json',cal)


class BeforeOnlyCollector:
    def __init__(self,backend,q,coefficient,entries):
        self.backend=backend;self.searcher=backend.searcher;self.q=q;self.coefficient=coefficient;self.entries=entries
    def doc(self,loc):return self.searcher.doc(loc)
    def search(self,query,**kwargs):
        if kwargs.get('limit')!=100:raise ValueError('One native collector100 only')
        tv=self.backend._tantivy;parts=[query]
        admin=baseline.source.original.lexical_parts(self.backend,self.q)[4]
        if admin is not None and self.coefficient>0 and self.entries:
            maximum=(1<<RECIPE['quantization_bits'])-1
            confidence=np.array([r['confidence'] for r in self.entries],dtype=float)
            if not np.isfinite(confidence).all() or (confidence<0).any() or (confidence>.500001).any():
                raise ValueError('Bounded weak activity context')
            integer=np.floor(confidence*maximum).astype(np.uint32);planes=[]
            for bit in range(RECIPE['quantization_bits']):
                chosen=np.flatnonzero(integer&(1<<bit))
                if len(chosen):planes.append(tv.Query.const_score_query(tv.Query.term_set_query(
                    self.backend.index.schema,'siren',[self.entries[i]['siren_ref'] for i in chosen]),
                    float(self.coefficient*(1<<bit)/maximum)))
            if planes:parts.append(tv.Query.boolean_query([(tv.Occur.Must,tv.Query.const_score_query(admin,0.)),
                (tv.Occur.Must,tv.Query.boolean_query([(tv.Occur.Should,p) for p in planes]))]))
        combined=tv.Query.boolean_query([(tv.Occur.Should,p) for p in parts])
        leaf=tv.Query.const_score_query(tv.Query.term_query(self.backend.index.schema,'document_type','siret','basic'),0.)
        return self.searcher.search(tv.Query.boolean_query([(tv.Occur.Must,leaf),(tv.Occur.Must,combined)]),**kwargs)


def native_search(backend,q,keys,features,coefficients):
    global ACTIVITY
    if ACTIVITY is None:ACTIVITY=io.read(verify(io.read(ACTIVE/'features_generation.json')['activity_context']))
    entries=ACTIVITY[q['query_id']]
    proxy=baseline.BackendProxy(backend,BeforeOnlyCollector(backend,q,float(coefficients[6]),entries))
    rows,cost=baseline.native_search(proxy,q,keys,features,coefficients)
    cost.update(activity_owner_context_references=len(entries),historical_activity_not_own_alias=True,
        activity_quantization_abs_error_bound=float(coefficients[6])/((1<<RECIPE['quantization_bits'])-1))
    return rows,cost


def stage(action,out):
    bind_runtime(out);baseline.source.pipeline.learned.proof=sys.modules[__name__]
    if action=='prepare':prepare(out)
    else:getattr(baseline.source.pipeline.learned,action)(out)


def run(out):
    with (out/'owned.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        deadline=io.read(out/'launch.json')['deadline_unix']
        for action,artifact in [('prepare','features_generation.json'),('generate','generation.json')]:
            if (out/artifact).exists():
                ex=io.read(out/(action+'_exit.json'))
                if ex['exit_code'] or not ex['own_child_exit_confirmed']:raise ValueError('Owned stage authority')
                continue
            while True:
                guard(deadline)
                command=[str(engine.CPU),'-u','-m','scripts.embeddinggemma2_v114_native_activity_ablation',action,'--output',str(out)]
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
        report=io.read(out/'report.json');report.update(activity_context_not_identity=True,no_new_training=True,
            source_parser_single_object_scope=True,production_promotion_allowed=False,comparison_baseline='FROZEN_V99')
        io.write(out/'report.json',report)
        io.write(out/'completion.json',dict(status='COMPLETE_OWNED_ALL526_FROZEN_ACTIVITY_ABLATION',report=record(out/'report.json'),finished_unix=time.time()))


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
