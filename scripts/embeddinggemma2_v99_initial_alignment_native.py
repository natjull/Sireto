"""Independent typed corporate/context native branch, ONE native100.

All whole-source corporate incidences, strict mother clauses and dated group
members used; no CRM/GT aliases or target selection. Relation != identity.
V81 proofs/coefficients/encoder frozen. Additional query components precede
the only collector; no capped pool, no candidate reranking, no new training.
"""
import argparse
from collections import defaultdict
import gc
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
from scripts import embeddinggemma2_v81_indirect_lexical_query as source
from scripts import embeddinggemma2_v72_corporate_incidence as graph
from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v14_full_pipeline as engine
from src.xgb_matcher.retrieval_v20_workbench import record,verify

from scripts import embeddinggemma2_v88_typed_context_native as context_source
INITIAL=io.HOME/'v98/official_initial_alignment_20261013_b'
OUTPUT=io.HOME/'v99/initial_alignment_native_20261013_a'
CACHE=source.CACHE
LEXICAL_FIELDS=source.LEXICAL_FIELDS
PROOF_FIELDS=source.PROOF_FIELDS
FEATURE_FIELDS=source.FEATURE_FIELDS
RECIPE=dict(source.RECIPE,hypothesis='FROZEN_V88_PLUS_DERIVED_LEGAL_NAME_INITIAL_TOKEN_ALIGNMENT',
    source_match='ALL_MEANINGFUL_PUBLISHED_NAME_TOKENS_PRESENT_IN_QUERY_MIN_TWO_OR_EXACT_SINGLE',
    source_match_not_entity_resolution=True,related_owner_names_not_site_aliases=True,
    corporate_scope='ONE_ROLE_EDGE_OR_DISTINCT_PEER_SHARING_ONE_ACTOR_NOT_OWNERSHIP',
    role_confidence='HALF_DIV_SQRT_ACTOR_SUBJECT_DEGREE_NO_PRUNING',
    dated_group_confidence=.5,reported_mother_confidence=.5,
    duplicate_context='MAX_NOT_ADDITIVE_VOTES',
    extra_admin_support_required=True,original_national_branch_unrestricted=True,
    context_weight='EXACT_V71_CORPORATE_ACTOR_WEIGHT_TIMES_SOURCE_CONFIDENCE',
    no_coefficient_refitting=True,no_production_promotion=True,
    initial_alignment_not_declared_company_alias=True,source_confidence_scaled_by_source_owner_idf=True,
    initials_weight='EXACT_FROZEN_V71_CORPORATE_ACTOR_WEIGHT_TIMES_DERIVED_SOURCE_CONFIDENCE')
ACTIVE=OUTPUT
CONTEXT=None
INITIAL_DATA=None
guard=source.guard
feature_path=source.feature_path
lexical_parts=source.lexical_parts


def bind_runtime(out):
    global ACTIVE
    ACTIVE=out;source.bind_runtime(out)


def initialize(out):
    source.initialize(out)
    launch=io.read(out/'launch.json');io.write(out/'recipe.json',RECIPE)
    for stage in ('prepare','generate'):
        ex=io.read(source.OUTPUT/(stage+'_exit.json'))
        if ex['exit_code'] or not ex['own_child_exit_confirmed']:raise ValueError('Complete owned V81')
    gx=io.read(graph.OUTPUT/'owned_exit.json')
    if gx['exit_code'] or not gx['own_child_exit_confirmed']:raise ValueError('Complete whole graph')
    launch.update(implementation=record(Path(__file__)),recipe=record(out/'recipe.json'),
        prepared_v81=record(source.OUTPUT/'features_generation.json'),
        v81_prepare_exit=record(source.OUTPUT/'prepare_exit.json'),graph_generation=record(graph.OUTPUT/'generation.json'),
        graph_exit=record(graph.OUTPUT/'owned_exit.json'))
    launch.update(prepared_context=record(context_source.OUTPUT/'features_generation.json'),
        context_exit=record(context_source.OUTPUT/'prepare_exit.json'),initial_source_directory=str(INITIAL),
        initial_implementation=record(io.ROOT/'scripts/embeddinggemma2_v98_official_initial_alignment.py'))
    launch['helpers'].extend([record(Path(source.__file__)),record(Path(graph.__file__))])
    io.write(out/'launch.json',launch)


def authority(out):
    launch=source.authority(out)
    for n in ('prepared_v81','v81_prepare_exit','graph_generation','graph_exit','prepared_context','context_exit','initial_implementation'):verify(launch[n])
    return launch


def prepare(out):
    launch=authority(out);fg=io.read(verify(launch['prepared_context']))
    ex=io.read(verify(launch['context_exit']))
    if ex['exit_code'] or not ex['own_child_exit_confirmed']:raise ValueError('Owned complete V88 source context')
    while True:
        guard(launch['deadline_unix'])
        if (INITIAL/'generation.json').exists() and (INITIAL/'owned_exit.json').exists():break
        time.sleep(10)
    ex=io.read(INITIAL/'owned_exit.json');initial=io.read(INITIAL/'generation.json')
    if ex['exit_code'] or not ex['own_child_exit_confirmed'] or initial['status']!='COMPLETE_WHOLE_SOURCE_INITIAL_ALIGNMENT':raise ValueError('Whole source compiler owned complete')
    if initial['implementation']!=launch['initial_implementation'] or initial['query_panel']!=launch['panel']:raise ValueError('Exact source implementation/query binding')
    verify(initial['alignment']);verify(initial['cog'])
    io.write(out/'features_generation.json',dict(fg,initial_generation=record(INITIAL/'generation.json'),
        initial_owned_exit=record(INITIAL/'owned_exit.json'),initial_alignment=initial['alignment'],
        no_original_context_values_changed=True,labels_used=False))



class BeforeOnlyCollector:
    def __init__(self,backend,q,w,entries):
        self.searcher=backend.searcher;self.backend=backend;self.q=q;self.w=w;self.entries=entries
    def doc(self,loc):return self.searcher.doc(loc)
    def search(self,query,**kwargs):
        if kwargs.get('limit')!=100:raise ValueError('Exactly one native100 collector')
        tv=self.backend._tantivy;admin=source.original.lexical_parts(self.backend,self.q)[4];parts=[query]
        if admin is not None:
            admin=tv.Query.const_score_query(admin,0.)
            if self.entries and self.w[6]>0:
                maximum=(1<<RECIPE['quantization_bits'])-1
                confidence=np.array([e['confidence'] for e in self.entries],dtype=float)
                if not np.isfinite(confidence).all() or (confidence<0).any() or (confidence>1).any():raise ValueError('Bounded source context')
                integers=np.floor(confidence*maximum).astype(np.uint32);planes=[]
                for bit in range(RECIPE['quantization_bits']):
                    selected=np.flatnonzero(integers & (1<<bit))
                    if len(selected):
                        owner=tv.Query.term_set_query(self.backend.index.schema,'siren',[self.entries[i]['siren'] for i in selected])
                        planes.append(tv.Query.const_score_query(owner,float(self.w[6]*(1<<bit)/maximum)))
                if planes:
                    parts.append(tv.Query.boolean_query([(tv.Occur.Must,admin),
                        (tv.Occur.Must,tv.Query.boolean_query([(tv.Occur.Should,p) for p in planes]))]))
        combined=tv.Query.boolean_query([(tv.Occur.Should,p) for p in parts])
        # Added branches still require leaf documents, because parent docs also
        # carry siren/name. No materialization before this single search call.
        leaf=tv.Query.const_score_query(tv.Query.term_query(self.backend.index.schema,'document_type','siret','basic'),0.)
        combined=tv.Query.boolean_query([(tv.Occur.Must,leaf),(tv.Occur.Must,combined)])
        return self.searcher.search(combined,**kwargs)


class InitialsBeforeOnlyCollector:
    def __init__(self,backend,q,w,entry):self.original=backend.searcher;self.backend=backend;self.q=q;self.w=w;self.entry=entry
    def doc(self,loc):return self.original.doc(loc)
    def search(self,query,**kwargs):
        if kwargs.get('limit')!=100:raise ValueError('Exactly one native100 collector')
        tv=self.backend._tantivy;admin=source.original.lexical_parts(self.backend,self.q)[4];parts=[query]
        if admin is not None and self.w[6]>0:
            planes=[];maximum=(1<<RECIPE['quantization_bits'])-1
            for field,entries in [('siren',[(e['siren_ref'],e['confidence']) for e in self.entry['parent_initials']]),
                    ('siret',[(ref,e['confidence']) for e in self.entry['own_city_initials'] for ref in e['siret_refs']])]:
                values={}
                for ref,c in entries:values[ref]=max(values.get(ref,0.),c)
                refs=list(values);scores=np.array([values[r] for r in refs],dtype=float)
                if not np.isfinite(scores).all() or (scores<0).any() or (scores>1).any():raise ValueError('Bounded initial-alignment confidence')
                integers=np.floor(scores*maximum).astype(np.uint32);bits=[]
                for bit in range(RECIPE['quantization_bits']):
                    chosen=np.flatnonzero(integers & (1<<bit))
                    if len(chosen):bits.append(tv.Query.const_score_query(tv.Query.term_set_query(self.backend.index.schema,field,[refs[i] for i in chosen]),float(self.w[6]*(1<<bit)/maximum)))
                if bits:planes.append(tv.Query.boolean_query([(tv.Occur.Should,p) for p in bits]))
            if planes:
                parts.append(tv.Query.boolean_query([(tv.Occur.Must,tv.Query.const_score_query(admin,0.)),
                    (tv.Occur.Must,tv.Query.disjunction_max_query(planes,tie_breaker=0.))]))
        combined=tv.Query.boolean_query([(tv.Occur.Should,p) for p in parts])
        leaf=tv.Query.const_score_query(tv.Query.term_query(self.backend.index.schema,'document_type','siret','basic'),0.)
        return self.original.search(tv.Query.boolean_query([(tv.Occur.Must,leaf),(tv.Occur.Must,combined)]),**kwargs)


class BackendProxy:
    def __init__(self,backend,searcher):self.backend=backend;self.searcher=searcher
    def __getattr__(self,name):return getattr(self.backend,name)


def native_search(backend,q,keys,features,coefficients):
    global CONTEXT,INITIAL_DATA
    if CONTEXT is None:CONTEXT=io.read(verify(io.read(ACTIVE/'features_generation.json')['context_expansion']))
    entries=CONTEXT[q['query_id']];proxy=BackendProxy(backend,BeforeOnlyCollector(backend,q,coefficients,entries))
    if INITIAL_DATA is None:INITIAL_DATA=io.read(verify(io.read(ACTIVE/'features_generation.json')['initial_alignment']))
    iproxy=BackendProxy(proxy,InitialsBeforeOnlyCollector(proxy,q,coefficients,INITIAL_DATA[q['query_id']]))
    rows,cost=source.native_search(iproxy,q,keys,features,coefficients)
    cost.update(typed_context_owner_expansions=len(entries),one_collector_before_materialization=True,
        added_relations_not_identity=True,source_scores_max_not_duplicate_votes=True,
        context_quantization_abs_error_bound=float(coefficients[6])/((1<<RECIPE['quantization_bits'])-1),
        initial_alignment_not_alias=True,initial_alignment_extra_error_bound=float(coefficients[6])/((1<<RECIPE['quantization_bits'])-1))
    return rows,cost


def stage(action,out):
    bind_runtime(out);source.pipeline.learned.proof=sys.modules[__name__]
    if action=='prepare':prepare(out)
    elif action=='bind_models':source.bind_models(out)
    else:getattr(source.pipeline.learned,action)(out)
    if action=='evaluate':
        r=io.read(out/'report.json');r.update(eligibility=RECIPE['eligibility'],production_promotion_allowed=False,
            typed_context_not_entity_identity=True,frozen_v81_coefficients=True,no_extra_fit=True)
        io.write(out/'report.json',r)


def run(out):
    import fcntl
    lock=(io.HOME/(out.name+'.owned.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    deadline=source.pipeline.corporate.DEADLINE
    for action,artifact in [('prepare','features_generation.json'),('bind_models','calibration/generation.json'),('generate','generation.json')]:
        if (out/artifact).exists():
            ex=io.read(out/(action+'_exit.json'))
            if ex['exit_code'] or not ex['own_child_exit_confirmed']:raise ValueError('Owned complete stage')
            continue
        while True:
            guard(deadline);command=[str(engine.CPU),'-u','-m','scripts.embeddinggemma2_v99_initial_alignment_native',action,'--output',str(out)]
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
            if rc:raise RuntimeError(f'{action} exit {rc}; no partial recall')
            break
    stage('evaluate',out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['initialize','run','prepare','bind_models','generate','evaluate']);p.add_argument('--output',type=Path,default=OUTPUT);a=p.parse_args()
    signal.signal(signal.SIGTERM,engine.interrupted);signal.signal(signal.SIGINT,engine.interrupted)
    try:
        if a.action=='initialize':initialize(a.output)
        elif a.action=='run':run(a.output)
        else:stage(a.action,a.output)
    except engine.ResourceYield as e:print(str(e),flush=True);sys.exit(75)
    except Exception as e:
        if a.output.exists():io.write(a.output/(a.action+'_error.json'),dict(error=repr(e),updated_unix=time.time()))
        raise



