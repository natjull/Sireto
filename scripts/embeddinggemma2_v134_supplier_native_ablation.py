"""Four predeclared supplier/site context ablations, no extra fitting.

Published client relationships remain weak typed context, never legal identity.
Native outputs cover the same526 before labels. No per-query rescue or pool.
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

from scripts import embeddinggemma2_v143_empty_reference_continuation as parent
from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v14_full_pipeline as engine
from scripts import embeddinggemma2_v115_streamed_activity_ablation as baseline
from src.xgb_matcher.hierarchical_retrieval import TantivyBackend
from src.xgb_matcher.retrieval_v20_workbench import record,verify

ROOT=io.HOME/'v134/supplier_native_20261013_night'
SOURCE=io.HOME/'v133/supplier_site_context_20261013_night'
ADDRESS=io.HOME/'v137/continuity_address_source_20261013_night'
VARIANTS=('supplier_own_site_only','supplier_temporal_only','continuity_address_only','all_new_sources_max')
RECIPE=dict(variants=VARIANTS,scope='SAME_CONSUMED_ALL526_EXPLORATORY_NOT_CONFIRMATION',
    source='PREDECLARED_V133_PUBLISHED_CLIENT_SITE_CONTEXT_NOT_IDENTITY',
    coefficient='FROZEN_SAME_HELDOUT_CONTEXT_COEFFICIENT6_ONCE',
    all_new_sources='MAX_OSM_GEOMETRY_NAMED_SUPPLIER_OWN_SUPPLIER_TEMPORAL_CONTINUITY_ADDRESS_PER_LEAF_AND_MAX_RNA_BANATIC_PER_OWNER',
    continuity_address='EXACT_PREDECLARED_V137_PUBLIC_DIRECT_ECONOMIC_RELATION_AND_OWN_ADDRESS_COMPATIBILITY_NOT_XY_COPY',
    no_double_supplier_provenance_votes=True,no_fit=True,no_grid=True,
    reuse_upstream='ONLY_EXACT_EQUAL_ALL_LEAF_SCORE_AND_IDENTICAL_OWNER_CONTEXT_INPUTS',
    initial_cap=100,no_prepool=True,no_positive_injected=True,
    all_four_generations_and_owned_exits_before_labels=True,
    quantization_bits=baseline.RECIPE['quantization_bits'],
    upstream_primary_not_changed=True,source_eligibility='INHERITED_DIAGNOSTIC_ONLY_NOT_PRODUCTION',
    no_full_encoder_oof=True,pristine_tests_opened=False,deadline_unix=parent.DEADLINE)


def authority(out):
    launch=parent.authority(out)
    for k in ('supplier_generation','supplier_exit','address_generation','address_exit','parent_prepared','parent_launch',
              'parent_composite_generation','parent_composite_exit'):verify(launch[k])
    return launch


def initialize(out):
    old=parent.authority(parent.ROOT);prepared=io.read(parent.ROOT/'prepared.json')
    ex=io.read(SOURCE/'owned_tool_exit.json');source=io.read(SOURCE/'generation.json')
    ax=io.read(ADDRESS/'owned_tool_exit.json');ag=io.read(ADDRESS/'generation.json')
    if source['status']!='COMPLETE_QUERY_ONLY_ALL526_SUPPLIER_SITE_CONTEXT' or source['queries']!=526:
        raise ValueError('Complete supplier source all526')
    if ex['exit_code'] or not ex['own_tool_exit_confirmed']:raise ValueError('Observed actual source exit')
    if ax['exit_code'] or not ax['own_tool_exit_confirmed'] or ag['status']!='COMPLETE_QUERY_ONLY_ALL526_CONTINUITY_ADDRESS_FEASIBILITY_NOT_RECALL':
        raise ValueError('Complete observed new address-continuity source')
    if ag['query_panel']!=old['panel'] or set(io.read(verify(ag['extras'])))!={q['query_id'] for q in io.read(verify(old['panel']))}:
        raise ValueError('Exact whole source population, not source-selected labels')
    if source['query_panel']!=old['panel'] or prepared['status']!='COMPLETE_ALL526_LABEL_FREE_SOURCE_SCORES':
        raise ValueError('Exact source/cohort and prepared binding')
    composite=parent.ROOT/'osm_max_rna_banatic'
    cg=io.read(composite/'generation.json');cx=io.read(composite/'generate_exit.json')
    if cg['status']!='COMPLETE_SEALED_ALL526' or cx['exit_code'] or not cx['own_child_exit_confirmed']:
        raise ValueError('Owned complete upstream composite before equivalent reuse')
    if out.exists():raise ValueError('Fresh predeclared experiment namespace')
    verify(source['extras']);out.mkdir(parents=True);(out/'spill').mkdir();(out/'calibration').mkdir()
    io.write(out/'recipe.json',RECIPE)
    launch=dict(old,implementation=record(Path(__file__)),recipe=record(out/'recipe.json'),
        parent_launch=record(parent.ROOT/'launch.json'),parent_prepared=record(parent.ROOT/'prepared.json'),
        supplier_generation=record(SOURCE/'generation.json'),supplier_exit=record(SOURCE/'owned_tool_exit.json'),
        address_generation=record(ADDRESS/'generation.json'),address_exit=record(ADDRESS/'owned_tool_exit.json'),
        parent_composite_generation=record(composite/'generation.json'),parent_composite_exit=record(composite/'generate_exit.json'),
        created_unix=time.time())
    launch['helpers'].append(record(Path(parent.__file__)));io.write(out/'launch.json',launch)
    io.write(out/'features_generation.json',io.read(parent.ROOT/'features_generation.json'))
    io.write(out/'calibration/generation.json',io.read(parent.ROOT/'calibration/generation.json'))


def projected(keys,ref):
    scores=np.zeros(len(keys),dtype=float)
    if ref is None:return scores
    with np.load(verify(ref),allow_pickle=False) as f:
        refs=f['keys'];confidence=f['confidence'];at=np.searchsorted(keys,refs)
        if np.any(at>=len(keys)) or not np.array_equal(keys[at],refs):raise ValueError('Exact full original proof universe')
        if not np.isfinite(confidence).all() or np.any(confidence<=0) or np.any(confidence>.500001):raise ValueError('Frozen confidence')
        scores[at]=confidence
    return scores


def generate(out,variant):
    launch=authority(out);baseline.bind_runtime(out);panel=io.read(verify(launch['panel']))
    pg=io.read(verify(launch['parent_prepared']));fg=io.read(out/'features_generation.json')
    cal=io.read(out/'calibration/generation.json');assignment=io.read(verify(cal['assignment']))
    models={k:io.read(verify(v)) for k,v in cal['models'].items()}
    source=io.read(verify(launch['supplier_generation']));extras=io.read(verify(source['extras']))
    address=io.read(verify(io.read(verify(launch['address_generation']))['extras']))
    rna=io.read(verify(io.read(verify(launch['rna_generation']))['context']))
    banatic=io.read(verify(io.read(verify(launch['banatic_generation']))['context']))
    bg=io.read(verify(launch['baseline_generation']));old=io.read(verify(bg['outputs']));old_cost=io.read(verify(bg['timings']))
    cg=io.read(verify(launch['parent_composite_generation']))
    composite=io.read(verify(cg['outputs']))['learned_native_ooc'];composite_cost=io.read(verify(cg['timings']))
    backend=TantivyBackend(io.INDEX)
    if backend.searcher.num_docs!=82772691:raise ValueError('Identical whole national corpus')
    root=out/variant;root.mkdir(exist_ok=True);(root/'native').mkdir(exist_ok=True);rows={};costs={};modified=0
    for q in panel:
        parent.guard();qid=q['query_id'];fold=str(assignment[qid]);model=models[fold]
        if qid in model['train_queries'] or qid not in model['heldout_queries']:raise ValueError('Frozen heldout calibrator')
        path=root/'native'/(hashlib.sha256(qid.encode()).hexdigest()+'.json')
        if engine.sealed(path):part=io.read(engine.sealed(path))
        else:
            with np.load(verify(pg['shards'][qid]),allow_pickle=False) as f:
                keys=f['keys'];geom=f['geometry'];named=f['named']
            own=projected(keys,extras[qid]['own']);temporal=projected(keys,extras[qid]['temporal']);continuity=projected(keys,address[qid])
            w=float(model['coefficients'][6]);entries=[]
            if variant=='supplier_own_site_only':score=w*own
            elif variant=='supplier_temporal_only':score=w*temporal
            elif variant=='continuity_address_only':score=w*continuity
            else:
                score=np.maximum(np.maximum(geom,named),w*np.maximum(np.maximum(own,temporal),continuity))
                merged={r['siren_ref']:r['confidence'] for r in rna[qid]}
                for r in banatic[qid]:merged[r['siren_ref']]=max(merged.get(r['siren_ref'],0.),r['confidence'])
                entries=[dict(siren_ref=r,confidence=c) for r,c in sorted(merged.items())]
            if variant=='all_new_sources_max' and np.array_equal(score,np.maximum(geom,named)):
                # Owner-context merge is exactly the upstream recipe above.
                part=dict(query=q,model=cal['models'][fold],candidates=composite[qid],
                    cost=dict(composite_cost[qid],unchanged_outputs_reused=True,reused_exact_upstream_inputs=True),
                    equivalence='ALL_LEAF_SCORES_AND_RNA_BANATIC_CONTEXT_EXACTLY_IDENTICAL',labels_read=False)
            elif not np.any(score) and not entries:
                part=dict(query=q,model=cal['models'][fold],candidates=old['learned_native_ooc'][qid],
                    cost=dict(old_cost[qid],unchanged_outputs_reused=True),equivalence='EXACT_ZERO_NEW_SCORE',labels_read=False)
            else:
                with np.load(verify(fg['features'][qid]),allow_pickle=False) as f:
                    if not np.array_equal(f['keys'],keys):raise ValueError('Original unpruned leaf references')
                    proxy=baseline.baseline.BackendProxy(backend,parent.EvidenceCollector(backend,q,keys,score,entries,w))
                    candidates,cost=baseline.native_search(proxy,q,keys,f['features'],model['coefficients'])
                cost.update(unchanged_outputs_reused=False,supplier_supported_sites=int(np.count_nonzero(np.maximum(own,temporal))),
                    max_same_provenance_and_new_leaf_sources=True)
                part=dict(query=q,model=cal['models'][fold],candidates=candidates,cost=cost,labels_read=False)
            io.write(path,part);engine.seal(path);gc.collect()
        if part['query']!=q or part['model']!=cal['models'][fold] or len(part['candidates'])>100:
            raise ValueError('Exact resume binding and initial100 cap')
        rows[qid]=part['candidates'];costs[qid]=part['cost'];modified+=not part['cost']['unchanged_outputs_reused']
        io.write(root/'state.json',dict(status='ALL526_NATIVE_SOURCE_CONTEXT_NO_LABELS',queries=len(rows),total=526,
            recomputed=modified,updated_unix=time.time()))
    if len(rows)!=526 or any(len({p['siret'] for p in cs})!=len(cs) for cs in rows.values()):raise ValueError('All526 unique outputs')
    io.write(root/'outputs.json',dict(old,learned_native_ooc=rows));io.write(root/'native_timings.json',costs)
    io.write(root/'generation.json',dict(status='COMPLETE_SEALED_ALL526',queries=526,outputs=record(root/'outputs.json'),
        timings=record(root/'native_timings.json'),no_positive_injected=True,no_prepool=True,
        no_downstream_reranking=True,labels_opened_in_native_worker=False,recomputed_queries=modified))


def run(out):
    with (out/'owned.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for variant in VARIANTS:
            root=out/variant;root.mkdir(exist_ok=True)
            if (root/'generate_exit.json').exists() and io.read(root/'generate_exit.json')['exit_code']==0:continue
            while True:
                authority(out);command=[str(engine.CPU),'-u','-m','scripts.embeddinggemma2_v134_supplier_native_ablation',
                    'generate','--variant',variant,'--output',str(out)]
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(out/'spill'),RAYON_NUM_THREADS='2',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
                with (root/'generate.log').open('ab',buffering=0) as log:
                    child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
                    io.write(root/'owned_process.json',dict(pid=child.pid,command=command,owned=True,started_unix=time.time()))
                    while child.poll() is None:
                        if engine.INTERRUPTED or time.time()>=parent.DEADLINE-30:
                            child.terminate()
                            try:child.wait(timeout=25)
                            except subprocess.TimeoutExpired:child.kill();child.wait()
                        time.sleep(5)
                    rc=child.wait()
                io.write(root/'generate_exit.json',dict(exit_code=rc,pid=child.pid,own_child_exit_confirmed=True,finished_unix=time.time()))
                if rc==75:time.sleep(10);continue
                if rc:raise RuntimeError(f'Owned {variant} exit{rc}, no partial metrics')
                break
        evaluate(out)


def evaluate(out):
    from scripts import embeddinggemma2_v32_learned_query as evaluator
    authority(out)
    for v in VARIANTS:
        ex=io.read(out/v/'generate_exit.json');gen=io.read(out/v/'generation.json')
        if ex['exit_code'] or not ex['own_child_exit_confirmed'] or gen['status']!='COMPLETE_SEALED_ALL526':
            raise ValueError('All four complete owned generations before evaluation')
        verify(gen['outputs'])
    summary={};old={r['query_id']:r for r in io.read(baseline.OUTPUT/'outcomes.json')['learned_native_ooc']}
    for v in VARIANTS:
        root=out/v;io.write(root/'launch.json',io.read(out/'launch.json'));(root/'calibration').mkdir(exist_ok=True)
        io.write(root/'calibration/generation.json',io.read(out/'calibration/generation.json'))
        evaluator.proof=sys.modules[__name__];evaluator.evaluate(root)
        fresh={r['query_id']:r for r in io.read(root/'outcomes.json')['learned_native_ooc']}
        if set(fresh)!=set(old):raise ValueError('Same all-human population')
        report=io.read(root/'report.json');paired=dict(gains=[q for q in old if fresh[q]['exact'] and not old[q]['exact']],
            losses=[q for q in old if old[q]['exact'] and not fresh[q]['exact']])
        report.update(paired_vs_v115=paired,eligibility=RECIPE['source_eligibility'],production_promotion_allowed=False)
        report.update(runtime_contains_equivalent_output_reuse=True,no_full_new_cold_warm_slo=True)
        io.write(root/'report.json',report);summary[v]=dict(metrics=report['variants']['learned_native_ooc'],paired=paired,report=record(root/'report.json'))
    io.write(out/'report.json',dict(status='COMPLETE_FOUR_PREDECLARED_SUPPLIER_ABLATIONS_ALL526',variants=summary,
        no_holdout_or_national_certification=True,finished_unix=time.time()))


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
