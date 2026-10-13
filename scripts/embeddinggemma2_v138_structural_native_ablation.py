"""Independent predeclared structural-source initial100 comparison, NO OSM.

All six full526 generations and observed owned exits precede labels. Baseline
proofs/calibrators/encoder are frozen; new sources are contextual, not aliases
or target-directed rescue. No capped pool, new fitting or closed dataset.
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
from scripts import embeddinggemma2_v23_native_features as lexical
from src.xgb_matcher.hierarchical_retrieval import TantivyBackend
from src.xgb_matcher.retrieval_v20_workbench import record,verify

ROOT=io.HOME/'v138/structural_native_20261013_night'
BASE=baseline.OUTPUT
DEADLINE=1791954368.
SOURCES={
    'rna':io.HOME/'v130/structural_20261013_night',
    'banatic':io.HOME/'v130/banatic_memberships_20261013_night_b',
    'supplier':io.HOME/'v133/supplier_site_context_20261013_night',
    'address':io.HOME/'v137/continuity_address_source_20261013_night',
}
STATUSES={
    'rna':'COMPLETE_SEALED_WHOLE_JULY_RNA_QUERY_ONLY_CONTEXT_NOT_RECALL',
    'banatic':'COMPLETE_SEALED_WHOLE_BANATIC_PUBLIC_RELATIONS_QUERY_ONLY_NOT_RECALL',
    'supplier':'COMPLETE_QUERY_ONLY_ALL526_SUPPLIER_SITE_CONTEXT',
    'address':'COMPLETE_QUERY_ONLY_ALL526_CONTINUITY_ADDRESS_FEASIBILITY_NOT_RECALL',
}
VARIANTS=('rna_only','banatic_only','supplier_own_site_only','supplier_temporal_only',
    'continuity_address_only','structural_sources_max')
RECIPE=dict(variants=VARIANTS,primary_predeclared='structural_sources_max',
    scope='SAME_ALL526_ALREADY_CONSUMED_DEVELOPMENT_NOT_CONFIRMATION',
    no_osm_dependency_or_dummy_source=True,new_geography='NONE_BASELINE_AND_PREPARED_SOURCE_OWN_GEO_FROZEN',
    calibration='EXACT_FROZEN_V115_HELDOUT_SIREN_MODELS_58_EQUALS_7_PLUS_51',
    coefficient='EXACT_FROZEN_CONTEXT_COEFFICIENT6_ONCE',
    owner_context='RNA_OR_BANATIC_OR_MAX_BOTH_PER_OWNER_GATED_QUERY_ADMIN',
    leaf_context='SOURCE_OWN_OR_TYPED_TEMPORAL_OR_TYPED_ADDRESS_CONTINUITY_OR_MAX_ALL_PER_LEAF',
    max_same_supplier_provenance_not_sum=True,distinct_owner_and_leaf_sources_additive_not_independence_claim=True,
    source_semantics='CONTEXT_NOT_OWN_NAME_ALIAS_OR_CERTIFIED_IDENTITY',
    no_refit=True,no_grid=True,no_model_selection_on_labels=True,
    all_six_complete_owned_generations_before_labels=True,
    reuse='EXACT_BASELINE_ONLY_NEW_SOURCE_SCORE_IDENTICALLY_ZERO',
    initial_cap=100,no_fixed_prepool=True,no_reranking=True,no_positive_injected=True,
    quantization_bits=baseline.RECIPE['quantization_bits'],
    production_promotion_allowed=False,full_encoder_oof=False,pristine_tests_opened=False,
    eligibility='DIAGNOSTIC_ONLY_PARTIAL_PUBLISHED_SOURCE_POSTCUTOFF_BANATIC_AND_INHERITED_BASE',
    deadline_unix=DEADLINE,resources=dict(threads=1,min_memory_free_percent=25,max_rss_gib=7,reserve_catnat_gib=80))


def guard():
    import psutil
    if time.time()>=DEADLINE:raise TimeoutError('Hard eight-hour deadline; preserve sealed shards')
    free,level=io.pressure();rss=psutil.Process().memory_info().rss
    if engine.INTERRUPTED or free<25 or level not in (0,1,2) or rss>7*1024**3:
        raise engine.ResourceYield(f'V138 cooperative yield free={free} level={level} rss={rss}')
    if psutil.disk_usage(io.HOME).free<80*1024**3:
        raise engine.ResourceYield('CATNAT reserve below80GiB; preserve outputs')


def authority(out):
    launch=io.read(out/'launch.json')
    for k in ('implementation','recipe','baseline_launch','baseline_generation','baseline_exit','baseline_features','baseline_calibration'):
        verify(launch[k])
    for refs in launch['structural_sources'].values():
        verify(refs['generation']);verify(refs['owned_exit'])
    for ref in launch['helpers']:verify(ref)
    guard();return launch


def initialize(out):
    guard()
    if out.exists():raise ValueError('Fresh independent source comparison namespace')
    bg=io.read(BASE/'generation.json');bx=io.read(BASE/'generate_exit.json');old=io.read(BASE/'launch.json')
    if bg['status']!='COMPLETE_SEALED_ALL526' or bx['exit_code'] or not bx['own_child_exit_confirmed']:
        raise ValueError('Complete owned baseline before new experiment')
    verify(bg['outputs']);panel=io.read(verify(old['panel']));scope={q['query_id'] for q in panel}
    if len(scope)!=526 or any(q['fold'] not in (2,3,4) for q in panel):raise ValueError('Whole safe query panel')
    receipts={}
    for name,path in SOURCES.items():
        g=io.read(path/'generation.json');ex=io.read(path/'owned_tool_exit.json')
        if g['status']!=STATUSES[name] or ex['exit_code'] or not ex['own_tool_exit_confirmed']:
            raise ValueError('Whole source and observed owned tool exit required')
        if name in ('rna','banatic'):
            source_launch=io.read(verify(g['launch']));p=source_launch['query_panel'];values=io.read(verify(g['context']))
        else:p=g['query_panel'];values=io.read(verify(g['extras']))
        if p!=old['panel'] or set(values)!=scope:raise ValueError('Identical hash-bound all526 source population')
        receipts[name]=dict(generation=record(path/'generation.json'),owned_exit=record(path/'owned_tool_exit.json'))
    cal=io.read(BASE/'calibration/generation.json')
    for ref in cal['models'].values():
        m=io.read(verify(ref))
        if len(m['coefficients'])!=58 or len(m['feature_fields'])!=58:raise ValueError('Frozen58 field calibrator')
    out.mkdir(parents=True);(out/'spill').mkdir();(out/'calibration').mkdir()
    io.write(out/'recipe.json',RECIPE)
    launch=dict(old,implementation=record(Path(__file__)),recipe=record(out/'recipe.json'),
        baseline_launch=record(BASE/'launch.json'),baseline_generation=record(BASE/'generation.json'),
        baseline_exit=record(BASE/'generate_exit.json'),baseline_features=record(BASE/'features_generation.json'),
        baseline_calibration=record(BASE/'calibration/generation.json'),structural_sources=receipts,
        deadline_unix=DEADLINE,created_unix=time.time())
    launch['helpers'].extend([record(Path(baseline.__file__)),record(Path(lexical.__file__))])
    io.write(out/'launch.json',launch);io.write(out/'features_generation.json',io.read(BASE/'features_generation.json'))
    io.write(out/'calibration/generation.json',cal)


def projected(keys,ref):
    values=np.zeros(len(keys),dtype=np.float64)
    if ref is None:return values
    with np.load(verify(ref),allow_pickle=False) as f:
        refs=f['keys'];confidence=f['confidence'];at=np.searchsorted(keys,refs)
        if np.any(at>=len(keys)) or not np.array_equal(keys[at],refs):raise ValueError('Unchanged full numeric scope')
        if not np.isfinite(confidence).all() or np.any(confidence<=0) or np.any(confidence>.500001):raise ValueError('Frozen weak source confidence')
        values[at]=confidence
    return values


class EvidenceCollector:
    def __init__(self,backend,q,keys,score,owners,coefficient):
        self.backend=backend;self.actual=backend.searcher;self.q=q;self.keys=keys
        self.score=score;self.owners=owners;self.coefficient=coefficient
    def doc(self,location):return self.actual.doc(location)
    def search(self,query,**kwargs):
        guard()
        if kwargs.get('limit')!=100:raise ValueError('Only ONE native collector100')
        tv=self.backend._tantivy;maximum=(1<<20)-1;parts=[query]
        bound=float(self.score.max()) if len(self.score) else 0.
        if bound>0:
            integers=np.floor(np.clip(self.score/bound,0,1)*maximum).astype(np.uint32)
            for bit in range(20):
                at=np.flatnonzero(integers&(1<<bit))
                if len(at):parts.append(tv.Query.const_score_query(tv.Query.term_set_query(self.backend.index.schema,
                    'siret',[f'{int(self.keys[i]):014d}' for i in at]),bound*(1<<bit)/maximum))
        admin=lexical.lexical_parts(self.backend,self.q)[4]
        if admin is not None and self.owners:
            refs=[r['siren_ref'] for r in self.owners];value=np.asarray([r['confidence'] for r in self.owners])
            if len(set(refs))!=len(refs) or not np.isfinite(value).all() or np.any(value<=0) or np.any(value>.500001):
                raise ValueError('Unique bounded whole-source owner context')
            integers=np.floor(value*maximum).astype(np.uint32);planes=[]
            for bit in range(20):
                at=np.flatnonzero(integers&(1<<bit))
                if len(at):planes.append(tv.Query.const_score_query(tv.Query.term_set_query(self.backend.index.schema,
                    'siren',[refs[i] for i in at]),self.coefficient*(1<<bit)/maximum))
            if planes:parts.append(tv.Query.boolean_query([(tv.Occur.Must,tv.Query.const_score_query(admin,0.)),
                (tv.Occur.Must,tv.Query.boolean_query([(tv.Occur.Should,p) for p in planes]))]))
        leaf=tv.Query.const_score_query(tv.Query.term_query(self.backend.index.schema,'document_type','siret','basic'),0.)
        return self.actual.search(tv.Query.boolean_query([(tv.Occur.Must,leaf),(tv.Occur.Must,
            tv.Query.boolean_query([(tv.Occur.Should,p) for p in parts]))]),**kwargs)


def generate(out,variant):
    launch=authority(out);baseline.bind_runtime(out);panel=io.read(verify(launch['panel']))
    fg=io.read(out/'features_generation.json');cal=io.read(out/'calibration/generation.json')
    assignment=io.read(verify(cal['assignment']));models={k:io.read(verify(v)) for k,v in cal['models'].items()}
    sg={k:io.read(verify(v['generation'])) for k,v in launch['structural_sources'].items()}
    rna=io.read(verify(sg['rna']['context']));banatic=io.read(verify(sg['banatic']['context']))
    supplier=io.read(verify(sg['supplier']['extras']));address=io.read(verify(sg['address']['extras']))
    bg=io.read(verify(launch['baseline_generation']));old=io.read(verify(bg['outputs']));old_cost=io.read(verify(bg['timings']))
    backend=TantivyBackend(io.INDEX)
    if backend.searcher.num_docs!=82772691:raise ValueError('Original full national index')
    root=out/variant;root.mkdir(exist_ok=True);(root/'native').mkdir(exist_ok=True);rows={};costs={};changed=0
    for q in panel:
        guard();qid=q['query_id'];fold=str(assignment[qid]);model=models[fold]
        if qid not in model['heldout_queries'] or qid in model['train_queries']:raise ValueError('Frozen heldout calibrator')
        path=root/'native'/(hashlib.sha256(qid.encode()).hexdigest()+'.json')
        if engine.sealed(path):part=io.read(engine.sealed(path))
        else:
            entries=rna[qid] if variant=='rna_only' else banatic[qid] if variant=='banatic_only' else []
            if variant=='structural_sources_max':
                merged={r['siren_ref']:r['confidence'] for r in rna[qid]}
                for r in banatic[qid]:merged[r['siren_ref']]=max(merged.get(r['siren_ref'],0.),r['confidence'])
                entries=[dict(siren_ref=s,confidence=c) for s,c in sorted(merged.items())]
            refs=[]
            if variant in ('supplier_own_site_only','structural_sources_max'):refs.append(supplier[qid]['own'])
            if variant in ('supplier_temporal_only','structural_sources_max'):refs.append(supplier[qid]['temporal'])
            if variant in ('continuity_address_only','structural_sources_max'):refs.append(address[qid])
            leaf_nonzero=False
            for ref in refs:
                if ref is not None:
                    with np.load(verify(ref),allow_pickle=False) as f:leaf_nonzero|=bool(len(f['confidence']))
            if not entries and not leaf_nonzero:
                part=dict(query=q,model=cal['models'][fold],candidates=old['learned_native_ooc'][qid],
                    cost=dict(old_cost[qid],unchanged_outputs_reused=True),equivalence='EXACT_ZERO_ADDED_SOURCE_SCORE',labels_read=False)
            else:
                with np.load(verify(fg['features'][qid]),allow_pickle=False) as f:
                    keys=f['keys'];features=f['features']
                    if features.shape!=(len(keys),51):raise ValueError('Frozen51 numerical fields')
                    confidence=np.zeros(len(keys),dtype=np.float64)
                    for ref in refs:confidence=np.maximum(confidence,projected(keys,ref))
                    w=float(model['coefficients'][6]);score=w*confidence
                    proxy=baseline.baseline.BackendProxy(backend,EvidenceCollector(backend,q,keys,score,entries,w))
                    candidates,cost=baseline.native_search(proxy,q,keys,features,model['coefficients'])
                    cost.update(unchanged_outputs_reused=False,new_leaf_contexts=int(np.count_nonzero(score)),new_owner_contexts=len(entries),
                        extra_quantization_abs_error_bound=float(score.max(initial=0))/((1<<20)-1)+w*.5/((1<<20)-1))
                guard();part=dict(query=q,model=cal['models'][fold],candidates=candidates,cost=cost,labels_read=False)
            io.write(path,part);engine.seal(path);gc.collect()
        if part['query']!=q or part['model']!=cal['models'][fold] or len(part['candidates'])>100:
            raise ValueError('Exact resume query/model and native cap')
        rows[qid]=part['candidates'];costs[qid]=part['cost'];changed+=not part['cost']['unchanged_outputs_reused']
        io.write(root/'state.json',dict(status='ALL526_GENERATION_NO_LABELS',queries=len(rows),total=526,recomputed=changed,updated_unix=time.time()))
    if len(rows)!=526 or any(len({r['siret'] for r in cs})!=len(cs) for cs in rows.values()):raise ValueError('Whole unique100 population')
    authority(out);io.write(root/'outputs.json',dict(old,learned_native_ooc=rows));io.write(root/'native_timings.json',costs)
    io.write(root/'generation.json',dict(status='COMPLETE_SEALED_ALL526',queries=526,outputs=record(root/'outputs.json'),
        timings=record(root/'native_timings.json'),no_positive_injected=True,no_prepool=True,no_osm_sources=True,
        no_downstream_reranking=True,labels_opened_in_native_worker=False,closed_test_opened=False,
        full_encoder_oof=False,recomputed_queries=changed))


def run(out):
    import psutil
    with (out/'owned.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for variant in VARIANTS:
            root=out/variant;root.mkdir(exist_ok=True)
            if (root/'generate_exit.json').exists():
                ex=io.read(root/'generate_exit.json')
                if ex['exit_code']==0 and ex['own_child_exit_confirmed'] and (root/'generation.json').exists():continue
            while True:
                while True:
                    if engine.INTERRUPTED or time.time()>=DEADLINE-30:
                        raise TimeoutError('Night ended/user interrupted before admission')
                    try:authority(out);break
                    except engine.ResourceYield as error:
                        io.write(out/'supervisor_state.json',dict(status='WAITING_REAL_RESOURCE_ADMISSION',
                            variant=variant,reason=str(error),updated_unix=time.time()))
                        time.sleep(30)
                command=[str(engine.CPU),'-u','-m','scripts.embeddinggemma2_v138_structural_native_ablation',
                    'generate','--variant',variant,'--output',str(out)]
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(out/'spill'),RAYON_NUM_THREADS='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
                reason=None
                with (root/'generate.log').open('ab',buffering=0) as log:
                    child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
                    io.write(root/'owned_process.json',dict(pid=child.pid,command=command,owned=True,started_unix=time.time()))
                    while child.poll() is None:
                        free,level=io.pressure()
                        try:rss=psutil.Process(child.pid).memory_info().rss
                        except psutil.NoSuchProcess:rss=0
                        if engine.INTERRUPTED or time.time()>=DEADLINE-30 or free<25 or level not in (0,1,2) or rss>7*1024**3:
                            reason=dict(deadline=time.time()>=DEADLINE-30,user_stop=engine.INTERRUPTED,free=free,level=level,rss=rss)
                            child.terminate()
                            try:child.wait(timeout=25)
                            except subprocess.TimeoutExpired:child.kill();child.wait()
                            break
                        time.sleep(5)
                    rc=child.wait()
                io.write(root/'generate_exit.json',dict(exit_code=rc,pid=child.pid,own_child_exit_confirmed=True,
                    resource_or_deadline_reason=reason,finished_unix=time.time()))
                if reason and (reason['deadline'] or reason['user_stop']):raise TimeoutError('Night ended/user interrupted, preserve all outputs')
                if rc==75 or reason:
                    io.write(out/'supervisor_state.json',dict(status='COOPERATIVE_RESOURCE_YIELD',variant=variant,exit_code=rc,reason=reason))
                    time.sleep(30);continue
                if rc:raise RuntimeError(f'{variant} owned exit{rc}; no partial labels')
                break
        evaluate(out)


def evaluate(out):
    from scripts import embeddinggemma2_v32_learned_query as evaluator
    authority(out)
    for variant in VARIANTS:
        root=out/variant;ex=io.read(root/'generate_exit.json');gen=io.read(root/'generation.json')
        if ex['exit_code'] or not ex['own_child_exit_confirmed'] or gen['status']!='COMPLETE_SEALED_ALL526':
            raise ValueError('All six complete owned generations before any labels')
        verify(gen['outputs']);verify(gen['timings'])
    old={r['query_id']:r for r in io.read(BASE/'outcomes.json')['learned_native_ooc']};summary={}
    for variant in VARIANTS:
        root=out/variant;io.write(root/'launch.json',io.read(out/'launch.json'));(root/'calibration').mkdir(exist_ok=True)
        io.write(root/'calibration/generation.json',io.read(out/'calibration/generation.json'))
        evaluator.proof=sys.modules[__name__];evaluator.evaluate(root)
        fresh={r['query_id']:r for r in io.read(root/'outcomes.json')['learned_native_ooc']}
        if len(fresh)!=526 or set(fresh)!=set(old):raise ValueError('Unchanged all-human denominator')
        for qid in old:
            if any(old[qid][f]!=fresh[qid][f] for f in ('fold','zone','siren_group','calibrated_query_weight')):
                raise ValueError('Same evaluation units and weights')
        paired=dict(gains=[q for q in old if fresh[q]['exact'] and not old[q]['exact']],losses=[q for q in old if old[q]['exact'] and not fresh[q]['exact']])
        report=io.read(root/'report.json');report.update(paired_vs_v115=paired,eligibility=RECIPE['eligibility'],
            production_promotion_allowed=False,full_encoder_oof=False,no_full_encoder_oof=True,
            encoder_train_sirens_disjoint=None,independent_encoder_generalization_not_validated=True,
            runtime_contains_equivalent_baseline_reuse=True,no_new_full_cold_warm_slo=True,no_osm_source_dependency=True)
        io.write(root/'report.json',report);summary[variant]=dict(report=record(root/'report.json'),paired=paired,
            metrics=report['variants']['learned_native_ooc'])
    io.write(out/'report.json',dict(status='COMPLETE_SIX_PREDECLARED_STRUCTURAL_ABLATIONS_ALL526',
        primary_predeclared=RECIPE['primary_predeclared'],variants=summary,no_holdout_or_national_certification=True,
        no_source_or_model_admission=True,no_osm_dependency=True,finished_unix=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['initialize','generate','run','evaluate'])
    p.add_argument('--output',type=Path,default=ROOT);p.add_argument('--variant',choices=VARIANTS);a=p.parse_args()
    signal.signal(signal.SIGTERM,engine.interrupted);signal.signal(signal.SIGINT,engine.interrupted)
    try:
        if a.action=='generate':generate(a.output,a.variant)
        else:globals()[a.action](a.output)
    except engine.ResourceYield as error:print(str(error),flush=True);sys.exit(75)
    except Exception as error:
        if a.output.exists():io.write(a.output/(a.action+'_error.json'),dict(error=repr(error),updated_unix=time.time()))
        raise
