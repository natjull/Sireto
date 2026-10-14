"""One frozen ablation: whole commune suffix only for weak named OSM context.

V142 full leaf MAX/core/owners/coefficients/names remain unchanged. The new
point bonus is MAX, never a pool/reranker/alias or copied site coordinate.
"""
import argparse
import fcntl
import gc
import hashlib
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import numpy as np
from scripts import embeddinggemma2_v142_bano_native_ablation as core
from scripts import embeddinggemma2_v161_osm_commune_suffix_preflight as source
from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v14_full_pipeline as engine
from src.xgb_matcher.hierarchical_retrieval import TantivyBackend
from src.xgb_matcher.retrieval_v20_workbench import record,verify

ROOT=io.HOME/'v162/commune_suffix_native_20261014_night'
VARIANT='commune_suffix_union'
RECIPE=dict(core.RECIPE,variants=[VARIANT],primary_predeclared=VARIANT,
    hypothesis='FROZEN_WHOLE_PUBLISHED_COMMUNE_SUFFIX_OSM_FULL_OR_REDUCED_MATCH',
    primary='MAX_EXACT_V142_LEAF_SCORE_AND_NEW_NAMED_POI_BONUS_SAME_RNA_BANATIC_OWNER_CONTEXT',
    named_bonus='FROZEN_COEFFICIENT6_TIMES_0.5_DIV_1_PLUS_OWN_XY_DISTANCE_TO_PUBLISHED_POINT_DIV300',
    same_public_mask_canonical_own_xy_required=True,no_new_parent_bonus_or_xy_copy=True,
    full_crm_name_for_bm25_dense_address_unchanged=True,
    incremental_named='NEW_MATCHED_OBJECTS_ONLY_EQUIVALENT_TO_FULL_UNION_OLD_NAMED_ALREADY_IN_CORE_MAX',
    reuse='EXACT_ORDERED_KEYS_FULL_LEAF_SCORE_COEFFICIENTS_OWNER_CONTEXTS_QUERY_MODEL_SOURCE_BINDINGS_NO_OUTCOMES',
    one_complete_generation_and_real_owned_exit_before_labels=True,
    prepare_rss_gib=1.5,resources=dict(threads=1,rss_gib=10,free_percent=25,pressure_levels=[0,1,2],disk_reserve_gib=40))


def guard(preparing=False):
    core.guard();engine.guard=guard
    if preparing:
        import psutil
        if psutil.Process().memory_info().rss>1.5*1024**3:raise engine.ResourceYield('Light preparation RSS1.5GiB yields, no scientific relaxation')


def authority(out):
    launch=core.authority(out)
    for k in ('core_launch','core_prepared','core_primary_generation','core_primary_exit','suffix_preflight','suffix_preflight_exit','suffix_rows','suffix_launch'):verify(launch[k])
    return launch


def initialize(out):
    guard();core.complete_family(core.ROOT,core.VARIANTS);old=core.authority(core.ROOT)
    sg=io.read(source.OUT/'report.json');sx=io.read(source.OUT/'owned_tool_exit.json');sl=io.read(source.OUT/'launch.json')
    if sx['exit_code'] or not sx['own_tool_exit_confirmed'] or sg['status']!='COMPLETE_ALL526_QUERY_ONLY_OSM_WHOLE_COMMUNE_SUFFIX_PREFLIGHT_NO_GT' or sl['query_panel']!=old['panel']:
        raise ValueError('Exact frozen unlabelled suffix source and real completion')
    if out.exists():raise ValueError('Fresh predeclared single ablation namespace')
    out.mkdir(parents=True);(out/'spill').mkdir();(out/'extras').mkdir();(out/'calibration').mkdir();(out/VARIANT).mkdir()
    io.write(out/'recipe.json',RECIPE)
    launch=dict(old,implementation=record(Path(__file__)),recipe=record(out/'recipe.json'),core_launch=record(core.ROOT/'launch.json'),
        core_prepared=record(core.ROOT/'prepared.json'),core_primary_generation=record(core.ROOT/'all_new_sources_max_bano/generation.json'),
        core_primary_exit=record(core.ROOT/'all_new_sources_max_bano/generate_exit.json'),suffix_preflight=record(source.OUT/'report.json'),
        suffix_preflight_exit=record(source.OUT/'owned_tool_exit.json'),suffix_rows=sg['all526'],suffix_launch=record(source.OUT/'launch.json'),
        source_context=sl['context'],created_unix=time.time())
    launch['helpers'].extend([record(Path(core.__file__)),record(Path(source.__file__))]);io.write(out/'launch.json',launch)
    io.write(out/'features_generation.json',io.read(core.ROOT/'features_generation.json'))
    io.write(out/'calibration/generation.json',io.read(core.ROOT/'calibration/generation.json'))


def inputs(launch,out):
    cal=io.read(out/'calibration/generation.json');models={k:io.read(verify(v)) for k,v in cal['models'].items()}
    return dict(cal=cal,models=models,assignment=io.read(verify(cal['assignment'])),
        pg=io.read(verify(launch['parent_prepared'])),bano=io.read(verify(launch['core_prepared'])),
        supplier=io.read(verify(io.read(verify(launch['supplier_generation']))['extras'])),
        address=io.read(verify(io.read(verify(launch['address_generation']))['extras'])),
        rna=io.read(verify(io.read(verify(launch['rna_generation']))['context'])),
        banatic=io.read(verify(io.read(verify(launch['banatic_generation']))['context'])))


def current_input(qid,state):
    model=state['models'][str(state['assignment'][qid])]
    if qid in model['train_queries'] or qid not in model['heldout_queries'] or len(model['coefficients'])!=58:
        raise ValueError('Exact frozen58 heldout calibrator')
    with np.load(verify(state['bano']['shards'][qid]),allow_pickle=False) as f:keys=f['keys'];score=f['bano_geometry']
    with np.load(verify(state['pg']['shards'][qid]),allow_pickle=False) as f:
        if not np.array_equal(keys,f['keys']):raise ValueError('Exact V142 parent ordered keys')
        previous_score=np.maximum(f['geometry'],f['named'])
    own=core.upstream.projected(keys,state['supplier'][qid]['own']);temporal=core.upstream.projected(keys,state['supplier'][qid]['temporal']);continuity=core.upstream.projected(keys,state['address'][qid])
    coefficient=float(model['coefficients'][6])
    previous_score=np.maximum(previous_score,coefficient*np.maximum(np.maximum(own,temporal),continuity));score=np.maximum(score,previous_score)
    merged={r['siren_ref']:r['confidence'] for r in state['rna'][qid]}
    for r in state['banatic'][qid]:merged[r['siren_ref']]=max(merged.get(r['siren_ref'],0.),r['confidence'])
    entries=[dict(siren_ref=r,confidence=c) for r,c in sorted(merged.items())]
    if keys.ndim!=1 or np.any(keys[1:]<=keys[:-1]) or score.shape!=keys.shape or not np.isfinite(score).all() or (score<0).any():raise ValueError('Exact finite full leaf proof universe')
    return keys,score,entries,model


def prepare(out):
    import ijson
    launch=authority(out);panel=io.read(verify(launch['panel']));queries={q['query_id']:q for q in panel};state=inputs(launch,out)
    suffix={r['query_id']:r for r in io.read(verify(launch['suffix_rows']))}
    if set(suffix)!=set(queries):raise ValueError('All526 suffix source decomposition')
    keys_all=np.load(verify(launch['keys']),mmap_mode='r');xy=np.load(verify(launch['xy']),mmap_mode='r');public=np.load(verify(launch['public_mask']),mmap_mode='r')
    if public.dtype!=bool or public.shape!=(len(keys_all),) or xy.shape!=(len(keys_all),2):raise ValueError('Exact public canonical coordinate universe')
    shards={};owner_context={};seen=set();modified=0;newobjects=0;start=time.monotonic()
    with verify(launch['source_context']).open('rb') as stream:
        for qid,c in ijson.kvitems(stream,'',use_float=True):
            guard(True)
            if qid not in queries or qid in seen:raise ValueError('Exact source scope')
            seen.add(qid);q=queries[qid];decomp=suffix[qid];nt=source.normalize(q['crm_name']).split();old=set(nt)-source.STOP
            ct=decomp['removed_commune_tokens'];enabled=decomp['reduced_match_enabled']
            if enabled and (not ct or nt[-len(ct):]!=ct):raise ValueError('Frozen entire suffix binding')
            reduced=set(nt[:-len(ct)])-source.STOP if enabled else set();points=[];oldmatch=0;newcount=0
            for obj in c['pois']:
                variants=[part for n in obj['names'] for part in re.split(r'[;|]',n) if part.strip()];ts=[set(source.normalize(n).split())-source.STOP for n in variants]
                full=bool(old and any(old<=t for t in ts));short=bool(enabled and reduced and any(reduced<=t for t in ts))
                oldmatch+=full
                if short and not full:
                    newcount+=1
                    for p in obj['points_l93']:
                        p=np.asarray(p,dtype=float)
                        if p.shape!=(2,) or not np.isfinite(p).all():raise ValueError('Published POI finite source point, no repair')
                        points.append(p)
            if oldmatch!=decomp['old_matches'] or newcount!=decomp['new_matches']:raise ValueError('Exact predeclared whole-source old/new POI predicates')
            keys,previous,entries,model=current_input(qid,state);named=np.zeros(len(keys),dtype=float)
            if points:
                if not c['spatial_crs_compatible']:raise ValueError('Same frozen spatial compatibility')
                at=np.searchsorted(keys_all,keys);valid=at<len(keys_all);valid[valid]&=keys_all[at[valid]]==keys[valid];valid[valid]&=public[at[valid]]
                pos=np.flatnonzero(valid);coords=np.asarray(xy[at[pos]],dtype=float)
                good=np.isfinite(coords).all(axis=1)&(coords[:,0]>=0)&(coords[:,0]<=1300000)&(coords[:,1]>=6000000)&(coords[:,1]<=7300000)
                pos,coords=pos[good],coords[good];distance=np.full(len(coords),np.inf)
                for p in points:distance=np.minimum(distance,np.hypot(coords[:,0]-p[0],coords[:,1]-p[1]))
                named[pos]=float(model['coefficients'][6])*.5/(1+distance/300.)
            score=np.maximum(previous,named)
            if not np.isfinite(score).all() or (score<previous).any():raise ValueError('Finite frozen MAX score no decreases')
            path=out/'extras'/(hashlib.sha256(qid.encode()).hexdigest()+'.npz');np.savez_compressed(path,keys=keys,previous=previous,score=score)
            engine.seal(path);shards[qid]=record(path);owner_context[qid]=entries;changed=int(np.count_nonzero(score!=previous));modified+=bool(changed);newobjects+=newcount
            io.write(path.with_suffix('.metadata.json'),dict(query_reference=launch['panel'],query_id=qid,model=state['cal']['models'][str(state['assignment'][qid])],
                full_ordered_keys=len(keys),new_poi_objects=newcount,new_poi_points=len(points),new_max_leaf_score_sites=changed,
                equivalent_to_v142=bool(np.array_equal(score,previous)),owner_contexts_identical=True,canonical_xy_only_no_parent_score_change=True,labels_read=False))
            gc.collect();io.write(out/'prepare_state.json',dict(status='ALL526_SUFFIX_SCORES_NO_LABELS',queries=len(shards),recomputed_native_needed=modified,updated_unix=time.time()))
    if seen!=set(queries) or newobjects!=io.read(verify(launch['suffix_preflight']))['counts']['new_object_appearances']:raise ValueError('Exact all526 source-only coverage')
    io.write(out/'owner_contexts.json',owner_context)
    io.write(out/'prepared.json',dict(status='COMPLETE_ALL526_SUFFIX_INPUTS_NO_LABELS',shards=shards,owner_contexts=record(out/'owner_contexts.json'),
        identical_core_sources=launch['core_launch'],queries=526,modified_queries=modified,new_poi_objects=newobjects,elapsed_seconds=time.monotonic()-start))
    authority(out);print('PREPARED',526,modified,newobjects,time.monotonic()-start,flush=True)


def generate(out):
    launch=authority(out);core.baseline.bind_runtime(out);panel=io.read(verify(launch['panel']));state=inputs(launch,out)
    prepared=io.read(out/'prepared.json');entries=io.read(verify(prepared['owner_contexts']));fg=io.read(out/'features_generation.json')
    gen=io.read(verify(launch['core_primary_generation']));old=io.read(verify(gen['outputs']));oldrows=old['learned_native_ooc'];oldcost=io.read(verify(gen['timings']))
    backend=TantivyBackend(io.INDEX)
    if backend.searcher.num_docs!=82772691:raise ValueError('Exact unrestricted national index')
    root=out/VARIANT;(root/'native').mkdir(exist_ok=True);rows={};costs={};recomputed=0
    for q in panel:
        guard();qid=q['query_id'];fold=str(state['assignment'][qid]);path=root/'native'/(hashlib.sha256(qid.encode()).hexdigest()+'.json')
        if engine.sealed(path):part=io.read(engine.sealed(path))
        else:
            keys,previous,owner,model=current_input(qid,state)
            with np.load(verify(prepared['shards'][qid]),allow_pickle=False) as f:
                if not np.array_equal(f['keys'],keys) or not np.array_equal(f['previous'],previous) or entries[qid]!=owner:raise ValueError('Full core score/key/coefficient/owner/source identity proof')
                score=f['score']
            original_part=io.read(engine.sealed(core.ROOT/'all_new_sources_max_bano/native'/path.name))
            if original_part['query']!=q or original_part['model']!=state['cal']['models'][fold]:raise ValueError('Original full query/model binding before reuse')
            if np.array_equal(score,previous):
                candidates=oldrows[qid];cost=dict(oldcost[qid],unchanged_outputs_reused=True,reused_exact_upstream_inputs=True)
                equivalence='EXACT_V142_FULL_ORDERED_KEYS_LEAF_SCORES_MODELS_OWNER_CONTEXTS_AND_QUERY'
            else:
                with np.load(verify(fg['features'][qid]),allow_pickle=False) as f:
                    if not np.array_equal(keys,f['keys']) or f['features'].shape!=(len(keys),51):raise ValueError('Unpruned original51 numerical proof')
                    proxy=core.baseline.baseline.BackendProxy(backend,core.parent.EvidenceCollector(backend,q,keys,score,owner,float(model['coefficients'][6])))
                    candidates,cost=core.baseline.native_search(proxy,q,keys,f['features'],model['coefficients'])
                cost.update(unchanged_outputs_reused=False,commune_suffix_max_only=True);equivalence=None
            part=dict(query=q,model=state['cal']['models'][fold],candidates=candidates,cost=cost,equivalence=equivalence,labels_read=False)
            io.write(path,part);engine.seal(path);gc.collect()
        if part['query']!=q or part['model']!=state['cal']['models'][fold] or len(part['candidates'])>100 or len({r['siret'] for r in part['candidates']})!=len(part['candidates']):raise ValueError('All526 exact binding absolute initial100')
        rows[qid]=part['candidates'];costs[qid]=part['cost'];recomputed+=not part['cost']['unchanged_outputs_reused']
        io.write(root/'state.json',dict(status='ALL526_INITIAL100_SUFFIX_NO_LABELS',queries=len(rows),recomputed=recomputed,updated_unix=time.time()))
    if len(rows)!=526 or recomputed!=prepared['modified_queries']:raise ValueError('All526 source-only recompute scope')
    authority(out);io.write(root/'outputs.json',dict(old,learned_native_ooc=rows));io.write(root/'native_timings.json',costs)
    io.write(root/'generation.json',dict(status='COMPLETE_SEALED_ALL526',queries=526,outputs=record(root/'outputs.json'),timings=record(root/'native_timings.json'),
        recomputed_queries=recomputed,no_positive_injected=True,no_prepool=True,no_downstream_reranking=True,labels_opened_in_native_worker=False,
        full_encoder_oof=False,runtime_reuse_not_full_new_slo=True))


def evaluate(out):
    from scripts import embeddinggemma2_v32_learned_query as evaluator
    authority(out);root=out/VARIANT;g=io.read(root/'generation.json');ex=io.read(root/'generate_exit.json')
    if g['status']!='COMPLETE_SEALED_ALL526' or g['queries']!=526 or ex['exit_code'] or not ex['own_child_exit_confirmed']:raise ValueError('Complete real owned generation before labels')
    verify(g['outputs']);io.write(root/'launch.json',io.read(out/'launch.json'));(root/'calibration').mkdir(exist_ok=True)
    io.write(root/'calibration/generation.json',io.read(out/'calibration/generation.json'));evaluator.proof=sys.modules[__name__];evaluator.evaluate(root)
    old={r['query_id']:r for r in io.read(core.ROOT/'all_new_sources_max_bano/outcomes.json')['learned_native_ooc']}
    new={r['query_id']:r for r in io.read(root/'outcomes.json')['learned_native_ooc']}
    if set(old)!=set(new) or any(old[q][f]!=new[q][f] for q in old for f in ('fold','zone','siren_group','calibrated_query_weight')):raise ValueError('Same allhuman population and calibrated weights')
    paired={metric:dict(gains=[q for q in old if new[q][metric] and not old[q][metric]],losses=[q for q in old if old[q][metric] and not new[q][metric]]) for metric in ('exact','siren','operational')}
    report=io.read(root/'report.json');report.update(paired_vs_v142=paired,eligibility=RECIPE['eligibility'],full_encoder_oof=False,production_promotion_allowed=False,runtime_contains_equivalent_output_reuse=True,no_full_new_cold_warm_slo=True)
    io.write(root/'report.json',report);io.write(out/'report.json',dict(status='COMPLETE_SINGLE_PREDECLARED_SUFFIX_ABLATION_ALL526',primary_predeclared=VARIANT,
        metrics=report['variants']['learned_native_ooc'],paired_vs_v142=paired,report=record(root/'report.json'),no_holdout_or_national_certification=True,finished_unix=time.time()))


def run(out):
    import psutil
    with (out/'owned.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);ex=io.read(out/'prepare_exit.json')
        if ex['exit_code'] or not ex['own_child_exit_confirmed']:raise ValueError('Observed complete light prepare first')
        root=out/VARIANT
        while True:
            authority(out);command=[str(engine.CPU),'-u','-m','scripts.embeddinggemma2_v162_commune_suffix_native','generate','--output',str(out)]
            with (root/'generate.log').open('ab',buffering=0) as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',RAYON_NUM_THREADS='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TMPDIR=str(out/'spill')))
                io.write(root/'owned_process.json',dict(pid=child.pid,command=command,owned=True,started_unix=time.time()));peak=0
                while child.poll() is None:
                    try:rss=psutil.Process(child.pid).memory_info().rss;peak=max(peak,rss)
                    except psutil.NoSuchProcess:break
                    free,level=io.pressure()
                    if engine.INTERRUPTED or time.time()>=core.parent.DEADLINE-20 or rss>10*1024**3 or free<25 or level not in (0,1,2):
                        child.terminate()
                        try:child.wait(timeout=15)
                        except subprocess.TimeoutExpired:child.kill();child.wait()
                        break
                    time.sleep(3)
                rc=child.wait()
            io.write(root/'generate_exit.json',dict(exit_code=rc,pid=child.pid,own_child_exit_confirmed=True,peak_rss_bytes=peak,finished_unix=time.time()))
            if rc==75 and time.time()<core.parent.DEADLINE-20:time.sleep(10);continue
            if rc:raise RuntimeError(f'Actual native exit{rc}; artifacts preserved, no partial recall')
            break
        evaluate(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['initialize','prepare','generate','evaluate','run']);p.add_argument('--output',type=Path,default=ROOT);a=p.parse_args()
    signal.signal(signal.SIGTERM,engine.interrupted);signal.signal(signal.SIGINT,engine.interrupted)
    try:globals()[a.action](a.output)
    except engine.ResourceYield as e:print(str(e),flush=True);sys.exit(75)
    except Exception as e:
        if a.output.exists():io.write(a.output/(a.action+'_error.json'),dict(error=repr(e),updated_unix=time.time()))
        raise
