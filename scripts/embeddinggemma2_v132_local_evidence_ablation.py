"""Frozen local OSM/RNA evidence before ONE national initial100 collector.

Source generation and public diffusion are checked before query-only scoring.
No labels, learned aliases, gradients, fixed prepool or downstream reordering.
The new OSM geometry and name evidence combine by MAX, not duplicate votes.
"""
import argparse
import fcntl
import gc
import hashlib
import os
import re
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np

from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v14_full_pipeline as engine
from scripts import embeddinggemma2_v115_streamed_activity_ablation as baseline
from scripts import embeddinggemma2_v125_street_shape_retrieval as geometry
from scripts import embeddinggemma2_v71_frozen_residual as native_proof
from scripts.embeddinggemma2_v17_structural_retrieval import normalize
from scripts.embeddinggemma2_v123_street_geometry_probe import segment_distance
from scripts.embeddinggemma2_v128_named_geo_feasibility import STOP
from src.xgb_matcher.hierarchical_retrieval import TantivyBackend
from src.xgb_matcher.retrieval_v20_workbench import record,verify

ROOT=io.HOME/'v132/local_evidence_20261013_night'
OSM=io.HOME/'v129/osm_local_20261013_night_c'
RNA=io.HOME/'v130/structural_20261013_night'
BANATIC=io.HOME/'v130/banatic_memberships_20261013_night_b'
BASE=baseline.OUTPUT
DEADLINE=1791954368.
VARIANTS=('geometry_only','named_places_only','osm_max','rna_only','osm_max_rna','banatic_only','osm_max_rna_banatic')
RECIPE=dict(variants=VARIANTS,scope='SAME_CONSUMED_ALL526_NOT_CONFIRMATION',
    primary_predeclared='osm_max_rna',other_variants='EXPLANATORY_ABLATIONS_NOT_HOLDOUT_MODEL_SELECTION',
    quantization_bits=baseline.RECIPE['quantization_bits'],
    baseline='EXACT_FROZEN_V115',initial_cap=100,no_prepool=True,no_new_fit=True,
    osm_names='ALL_INFORMATIVE_QUERY_TOKENS_IN_ONE_PUBLISHED_OSM_NAME_VARIANT',
    osm_name_variants='NAME_BRAND_ALT_OFFICIAL_SHORT_OLD_NOT_OPERATOR',
    osm_proximity='0.5/(1+distance_m/300)_MAX_MATCHING_PLACES',
    name_context_coefficient='SAME_FROZEN_COEFFICIENT6_NOT_TUNED_ON_OUTCOMES',
    geometry='MAX_OLD_AND_HALF_PROXIMITY_TO_OSM_STREET_OR_EXACT_ADDRESS',
    street_axis_only_when='EXPLICIT_STREET_FOOTPRINT_NOT_HOUSE',
    house_address='EXACT_NORMALIZED_STREET_AND_HOUSE_NUMBER_SUFFIX_WITHIN_QUERY_SOURCE_NEIGHBOURHOOD',
    geometry_interactions='SAME_EXISTING_FIELDS5_6_7_16_17_19',
    combined_osm='MAX_GEOMETRY_SCORE_INCREMENT_AND_NAMED_PLACE_CONTEXT_PER_SITE',
    rna='PREDECLARED_V130_CONTEXT_OWNER_REFS_AND_CONFIDENCES_GATED_BY_QUERY_ADMIN',
    banatic='PREDECLARED_WHOLE_PUBLIC_TYPED_MEMBERSHIP_CONTEXT_NOT_IDENTITY_GATED_BY_ADMIN',
    rna_banatic='MAX_CONFIDENCE_PER_OWNER_ACROSS_THESE_CONTEXT_SOURCES_NOT_DUPLICATE_VOTES',
    combined_rna_osm='ADDITIVE_DISTINCT_SOURCES_NOT_INDEPENDENCE_CLAIM',
    public_diffusion='BOTH_SITE_AND_OWNER_O_REQUIRED_FOR_OSM',
    coordinate_scope='EXISTING_ALL_UNPRUNED_NUMERIC_REFERENCE_UNIVERSE',
    original_national_lexical_unrestricted=True,no_new_spatial_expansion_claim=True,
    unchanged_query_reuse='ONLY_IDENTICALLY_ZERO_NEW_SCORE_BY_QUERY_INPUTS',
    labels_opened_only_after_all_predeclared_generations_and_owned_exits=True,
    exact_operational_separate=True,production_promotion_allowed=False,
    eligibility='DIAGNOSTIC_SOURCE_EXTENSION_CURRENT_COMMUNITY_OSM_AND_INHERITED_POSTCUTOFF_BASE',
    full_encoder_oof=False,closed_tests_opened=False,deadline_unix=DEADLINE,
    execution_resources=dict(max_rss_gib=10,min_memory_free_percent=25,
        admitted_pressure_levels=[0,1,2],reserve_catnat_gib=40,threads=2))


def guard():
    import psutil
    # Process-local execution policy; historical hash-bound helpers stay intact.
    # Their native scoring calls must use the same declared reserve, not80GiB.
    engine.guard=guard
    if time.time()>=DEADLINE:raise TimeoutError('Eight-hour mandate finished, preserve artifacts')
    free,level=io.pressure();rss=psutil.Process().memory_info().rss
    if engine.INTERRUPTED or free<25 or level not in (0,1,2) or rss>10*1024**3:
        raise engine.ResourceYield(f'V132 resource yield free={free} level={level} rss={rss}')
    if psutil.disk_usage(io.HOME).free<40*1024**3:
        raise engine.ResourceYield('CATNAT reserve below40GiB; preserve all artifacts')


def authority(out):
    launch=io.read(out/'launch.json')
    for k in ('implementation','recipe','baseline_launch','baseline_generation','baseline_exit',
              'baseline_features','baseline_calibration','osm_generation','osm_exit','rna_generation','rna_exit',
              'banatic_generation','banatic_exit','public_owners'):
        verify(launch[k])
    for ref in launch['helpers']:verify(ref)
    guard();return launch


def initialize(out):
    guard()
    if out.exists():raise ValueError('Fresh scientific namespace required')
    og=io.read(OSM/'generation.json');ox=io.read(OSM/'scan_owned_tool_exit.json')
    rg=io.read(RNA/'generation.json');rx=io.read(RNA/'owned_tool_exit.json')
    tg=io.read(BANATIC/'generation.json');tx=io.read(BANATIC/'owned_tool_exit.json')
    bg=io.read(BASE/'generation.json');bx=io.read(BASE/'generate_exit.json')
    if ox['exit_code'] or rx['exit_code'] or tx['exit_code'] or bx['exit_code'] or not bx['own_child_exit_confirmed']:
        raise ValueError('Observed complete sources and baseline required')
    if any(not ex.get('own_tool_exit_confirmed',ex.get('own_child_exit_confirmed',False)) for ex in (ox,rx,tx)):
        raise ValueError('Actual owned source tool exits required, not inferred from files')
    if bg['status']!='COMPLETE_SEALED_ALL526':raise ValueError('Complete baseline')
    if og['status']!='COMPLETE_ALL526_LOCAL_OSM_SOURCE' or og['all_queries']!=526:raise ValueError('All526 complete OSM source')
    for ref in og['files'].values():verify(ref)
    if rg['status']!='COMPLETE_SEALED_WHOLE_JULY_RNA_QUERY_ONLY_CONTEXT_NOT_RECALL' or rg['all_queries']!=526:
        raise ValueError('Whole complete public RNA source')
    verify(rg['context'])
    if tg['status']!='COMPLETE_SEALED_WHOLE_BANATIC_PUBLIC_RELATIONS_QUERY_ONLY_NOT_RECALL' or tg['all_queries']!=526:
        raise ValueError('Whole public typed membership source')
    verify(tg['context'])
    old_launch=io.read(BASE/'launch.json')
    ol=io.read(verify(og['launch']));rl=io.read(verify(rg['launch']));tl=io.read(verify(tg['launch']))
    if ol['recipe']['query_only']!=old_launch['panel'] or rl['query_panel']!=old_launch['panel'] or tl['query_panel']!=old_launch['panel']:
        raise ValueError('Identical hash-bound source panels, not just same query count')
    out.mkdir(parents=True);(out/'spill').mkdir();(out/'calibration').mkdir()
    io.write(out/'recipe.json',RECIPE)
    launch=dict(io.read(BASE/'launch.json'),implementation=record(Path(__file__)),recipe=record(out/'recipe.json'),
        baseline_launch=record(BASE/'launch.json'),baseline_generation=record(BASE/'generation.json'),
        baseline_exit=record(BASE/'generate_exit.json'),baseline_features=record(BASE/'features_generation.json'),
        baseline_calibration=record(BASE/'calibration/generation.json'),osm_generation=record(OSM/'generation.json'),
        osm_exit=record(OSM/'scan_owned_tool_exit.json'),rna_generation=record(RNA/'generation.json'),
        rna_exit=record(RNA/'owned_tool_exit.json'),banatic_generation=record(BANATIC/'generation.json'),
        banatic_exit=record(BANATIC/'owned_tool_exit.json'),
        public_owners=record(Path(old_launch['stock_source']['path']).with_name('StockUniteLegale.parquet')),
        deadline_unix=DEADLINE,created_unix=time.time())
    launch['helpers'].extend([record(Path(geometry.__file__)),record(Path(native_proof.__file__)),
        record(io.ROOT/'scripts/embeddinggemma2_v128_named_geo_feasibility.py')])
    io.write(out/'launch.json',launch)
    io.write(out/'features_generation.json',io.read(BASE/'features_generation.json'))
    io.write(out/'calibration/generation.json',io.read(BASE/'calibration/generation.json'))


def public_coordinates(out,launch):
    import duckdb
    keys=np.load(verify(launch['keys']),mmap_mode='r');xy=np.load(verify(launch['xy']),mmap_mode='r')
    path=out/'public_mask.npy'
    if engine.sealed(path):return keys,xy,np.load(engine.sealed(path),mmap_mode='r')
    mask=np.lib.format.open_memmap(path,mode='w+',dtype=bool,shape=(len(keys),));mask[:]=False
    con=duckdb.connect();con.execute('SET threads=1');con.execute("SET memory_limit='512MB'")
    con.execute('SET temp_directory=?',[str(out/'spill')])
    owners=verify(launch['public_owners']);verify(launch['stock_source'])
    cur=con.execute("""SELECT CAST(e.siret AS UBIGINT) site_key FROM read_parquet(?) e
      JOIN read_parquet(?) u USING(siren) WHERE e.statutDiffusionEtablissement='O'
      AND u.statutDiffusionUniteLegale='O'""",[launch['stock_source']['path'],str(owners)])
    count=0
    try:
        for chunk in cur.fetch_record_batch(65536):
            guard();ids=chunk.column(0).to_numpy();at=np.searchsorted(keys,ids)
            valid=at<len(keys);valid[valid]&=keys[at[valid]]==ids[valid];mask[at[valid]]=True;count+=int(valid.sum())
    finally:con.close()
    mask.flush();engine.seal(path)
    io.write(out/'public_source.json',dict(public_mask=record(path),site_source=launch['stock_source'],
        owner_source=record(owners),public_coordinate_refs=count,labels_read=False))
    return keys,xy,mask


def exact_address(q,obj):
    def number(value):
        # A range/list is not a single exact house number: never join6;8 into68.
        match=re.fullmatch(r'\s*(\d+)\s*(BIS|TER|QUATER|[A-Z])?\s*',normalize(value))
        return (int(match[1]),match[2] or '') if match and int(match[1])>0 else None
    tags=obj['tags'];query_number=number(q.get('crm_number','')+' '+q.get('crm_suffix',''))
    source_number=number(tags.get('addr:housenumber',''))
    return bool(query_number and query_number==source_number and normalize(q['crm_street'],street=True)==
                normalize(tags.get('addr:street',''),street=True))


def query_contexts(path,panel):
    import ijson
    queries={q['query_id']:q for q in panel};seen=set()
    with path.open('rb') as stream:
        for qid,context in ijson.kvitems(stream,'',use_float=True):
            if qid not in queries or qid in seen:raise ValueError('Exact unique query-only source scope')
            seen.add(qid);yield queries[qid],context
    if seen!=set(queries):raise ValueError('Every query retained, including source-empty cases')


def prepare(out):
    launch=authority(out);panel=io.read(verify(launch['panel']));fg=io.read(verify(launch['baseline_features']))
    source=io.read(verify(launch['osm_generation']))
    context_path=verify(source['files']['query_context']);shapes=io.read(verify(source['files']['shapes']))
    gs={r['query_id']:r['geometry'] for r in io.read(verify(launch['geometry']))}
    if set(shapes)!={q['query_id'] for q in panel}:raise ValueError('Whole same source population')
    keys_all,xy,public=public_coordinates(out,launch)
    cal=io.read(out/'calibration/generation.json');assignment=io.read(verify(cal['assignment']))
    models={k:io.read(verify(v)) for k,v in cal['models'].items()}
    root=out/'extras';root.mkdir(exist_ok=True);shards={}
    for q,context in query_contexts(context_path,panel):
        guard();qid=q['query_id'];path=root/(hashlib.sha256(qid.encode()).hexdigest()+'.npz')
        if engine.sealed(path):shards[qid]=record(path);continue
        model=models[str(assignment[qid])]
        if qid in model['train_queries'] or qid not in model['heldout_queries']:raise ValueError('Frozen heldout calibrator')
        w=np.asarray(model['coefficients'],dtype=float)
        with np.load(verify(fg['features'][qid]),allow_pickle=False) as f:
            keys=f['keys'];features=f['features'];at=np.searchsorted(keys_all,keys)
            valid=at<len(keys_all);valid[valid]&=keys_all[at[valid]]==keys[valid]
            valid[valid]&=public[at[valid]];pos=np.flatnonzero(valid);coords=np.asarray(xy[at[pos]],dtype=float)
            good=np.isfinite(coords).all(axis=1)&(coords[:,0]>=0)&(coords[:,0]<=1300000)&(coords[:,1]>=6000000)&(coords[:,1]<=7300000)
            pos=pos[good];coords=coords[good]
            g=gs[qid];lines=[]
            if g.get('precision')=='STREET_FOOTPRINT_NOT_HOUSE':lines.extend(shapes[qid]['lines_l93'])
            for obj in context['addresses']:
                if exact_address(q,obj):
                    for p in obj['points_l93']:lines.append([p,p])
            distance=np.full(len(coords),np.inf)
            for line in lines:distance=np.minimum(distance,segment_distance(coords,np.asarray(line,dtype=float)))
            changed=features.copy();changed[pos,5]=np.maximum(changed[pos,5],.5/(1+distance/300.))
            changed[:,6]=changed[:,0]*changed[:,5];changed[:,7]=changed[:,2]*changed[:,5]
            _,group=np.unique(keys//100000,return_inverse=True);parents=np.zeros(int(group.max())+1,dtype=np.float32)
            np.maximum.at(parents,group,changed[:,5]);changed[:,16]=np.maximum(changed[:,16],parents[group])
            changed[:,17]=changed[:,1]*changed[:,16];changed[:,19]=changed[:,18]*changed[:,16]
            if not lines:geom=np.zeros(len(keys),dtype=np.float64)
            else:
                geom=np.maximum(0.,(changed.astype(float)-features.astype(float))@w[len(baseline.LEXICAL_FIELDS):])
                geom[~valid]=0.
            del changed,group,parents
            qt=set(normalize(q['crm_name']).split())-STOP;matched=[]
            for obj in context['pois']:
                # Separate published aliases; a composite of two aliases is not a third name.
                variants=[part for n in obj['names'] for part in re.split(r'[;|]',n) if part.strip()]
                if qt and any(qt<=set(normalize(n).split())-STOP for n in variants):matched.append(obj)
            distance=np.full(len(coords),np.inf)
            for obj in matched:
                for p in obj['points_l93']:distance=np.minimum(distance,np.hypot(coords[:,0]-p[0],coords[:,1]-p[1]))
            named=np.zeros(len(keys),dtype=np.float64);named[pos]=float(w[6])*.5/(1+distance/300.)
            if not np.isfinite(geom).all() or not np.isfinite(named).all():raise ValueError('Finite bounded source scores')
            np.savez_compressed(path,keys=keys,geometry=geom,named=named)
            engine.seal(path);shards[qid]=record(path)
            io.write(root/(path.stem+'.metadata.json'),dict(query=q,matched_poi_refs=[(p['osm_type'],p['osm_id']) for p in matched],
                street_or_address_shapes=len(lines),geometry_changed_sites=int(np.count_nonzero(geom)),
                named_context_sites=int(np.count_nonzero(named)),same_full_numeric_scope=len(keys),labels_read=False))
        gc.collect();io.write(out/'prepare_state.json',dict(status='QUERY_ONLY_SOURCE_SCORES',queries=len(shards),total=526,updated_unix=time.time()))
    authority(out);io.write(out/'prepared.json',dict(status='COMPLETE_ALL526_LABEL_FREE_SOURCE_SCORES',shards=shards,
        source_context=record(OSM/'query_context.json'),shapes=record(OSM/'shapes.json'),public=record(out/'public_source.json')))


class EvidenceCollector:
    def __init__(self,backend,q,keys,scores,rna,coefficient):
        self.backend=backend;self.actual=backend.searcher;self.q=q;self.keys=keys;self.scores=scores;self.rna=rna;self.coefficient=coefficient
    def doc(self,loc):return self.actual.doc(loc)
    def search(self,query,**kwargs):
        if kwargs.get('limit')!=100:raise ValueError('One initial100 collector only')
        tv=self.backend._tantivy;maximum=(1<<20)-1;parts=[query]
        bound=float(self.scores.max()) if len(self.scores) else 0.
        if bound>0:
            integers=np.floor(np.clip(self.scores/bound,0,1)*maximum).astype(np.uint32)
            for bit in range(20):
                pos=np.flatnonzero(integers&(1<<bit))
                if len(pos):parts.append(tv.Query.const_score_query(tv.Query.term_set_query(self.backend.index.schema,'siret',
                    [f'{int(self.keys[i]):014d}' for i in pos]),bound*(1<<bit)/maximum))
        admin=native_proof.lexical_parts(self.backend,self.q)[4]
        if admin is not None and self.rna:
            refs=[r['siren_ref'] for r in self.rna];values=np.asarray([r['confidence'] for r in self.rna])
            integers=np.floor(values*maximum).astype(np.uint32);planes=[]
            for bit in range(20):
                at=np.flatnonzero(integers&(1<<bit))
                if len(at):planes.append(tv.Query.const_score_query(tv.Query.term_set_query(self.backend.index.schema,'siren',
                    [refs[i] for i in at]),float(self.coefficient)*(1<<bit)/maximum))
            if planes:parts.append(tv.Query.boolean_query([(tv.Occur.Must,tv.Query.const_score_query(admin,0.)),
                (tv.Occur.Must,tv.Query.boolean_query([(tv.Occur.Should,p) for p in planes]))]))
        leaf=tv.Query.const_score_query(tv.Query.term_query(self.backend.index.schema,'document_type','siret','basic'),0.)
        return self.actual.search(tv.Query.boolean_query([(tv.Occur.Must,leaf),(tv.Occur.Must,
            tv.Query.boolean_query([(tv.Occur.Should,p) for p in parts]))]),**kwargs)


def generate(out,variant):
    launch=authority(out);baseline.bind_runtime(out);panel=io.read(verify(launch['panel']))
    prepared=io.read(out/'prepared.json');fg=io.read(out/'features_generation.json');cal=io.read(out/'calibration/generation.json')
    assignment=io.read(verify(cal['assignment']));models={k:io.read(verify(v)) for k,v in cal['models'].items()}
    original=io.read(verify(io.read(verify(launch['baseline_generation']))['outputs']))
    old_cost=io.read(verify(io.read(verify(launch['baseline_generation']))['timings']))
    rna=io.read(verify(io.read(verify(launch['rna_generation']))['context']))
    memberships=io.read(verify(io.read(verify(launch['banatic_generation']))['context']))
    backend=TantivyBackend(io.INDEX);root=out/variant;root.mkdir(exist_ok=True);(root/'native').mkdir(exist_ok=True)
    if backend.searcher.num_docs!=82772691:raise ValueError('Unchanged full national corpus')
    rows={};costs={};changed=0
    for q in panel:
        guard();qid=q['query_id'];model=models[str(assignment[qid])];path=root/'native'/(hashlib.sha256(qid.encode()).hexdigest()+'.json')
        if engine.sealed(path):part=io.read(engine.sealed(path))
        else:
            with np.load(verify(prepared['shards'][qid]),allow_pickle=False) as f:
                keys=f['keys'];geom=f['geometry'];named=f['named']
            score=geom if variant=='geometry_only' else named if variant=='named_places_only' else np.maximum(geom,named)
            if variant in ('rna_only','banatic_only'):score=np.zeros_like(geom)
            entries=rna[qid] if variant in ('rna_only','osm_max_rna','osm_max_rna_banatic') else []
            if variant in ('banatic_only','osm_max_rna_banatic'):
                merged={r['siren_ref']:r['confidence'] for r in entries}
                for r in memberships[qid]:merged[r['siren_ref']]=max(merged.get(r['siren_ref'],0.),r['confidence'])
                entries=[dict(siren_ref=r,confidence=c) for r,c in sorted(merged.items())]
            if not np.any(score) and not entries:
                part=dict(query=q,model=cal['models'][str(assignment[qid])],candidates=original['learned_native_ooc'][qid],
                    cost=dict(old_cost[qid],unchanged_outputs_reused=True),equivalence='EXACT_ZERO_NEW_SOURCE_SCORE',labels_read=False)
            else:
                with np.load(verify(fg['features'][qid]),allow_pickle=False) as f:
                    if not np.array_equal(keys,f['keys']):raise ValueError('All original proof references, exact binding')
                    proxy=baseline.baseline.BackendProxy(backend,EvidenceCollector(backend,q,keys,score,entries,float(model['coefficients'][6])))
                    candidates,cost=baseline.native_search(proxy,q,keys,f['features'],model['coefficients'])
                cost.update(unchanged_outputs_reused=False,osm_extra_nonzero=int(np.count_nonzero(score)),
                    rna_owners=len(entries),osm_geometry_name_max_not_duplicate_votes=True)
                part=dict(query=q,model=cal['models'][str(assignment[qid])],candidates=candidates,cost=cost,labels_read=False)
            io.write(path,part);engine.seal(path);gc.collect()
        if (part['query']!=q or part['model']!=cal['models'][str(assignment[qid])]
                or len(part['candidates'])>100):
            raise ValueError('Exact query/model binding and native cap')
        rows[qid]=part['candidates'];costs[qid]=part['cost'];changed+=not part['cost']['unchanged_outputs_reused']
        io.write(root/'state.json',dict(status='INITIAL100_GENERATION_NO_LABELS',queries=len(rows),total=526,recomputed=changed,updated_unix=time.time()))
    if len(rows)!=526 or any(len({r['siret'] for r in cs})!=len(cs) for cs in rows.values()):raise ValueError('All526 unique outputs')
    outputs=dict(original,learned_native_ooc=rows);io.write(root/'outputs.json',outputs);io.write(root/'native_timings.json',costs)
    io.write(root/'generation.json',dict(status='COMPLETE_SEALED_ALL526',queries=526,outputs=record(root/'outputs.json'),
        timings=record(root/'native_timings.json'),no_positive_injected=True,no_prepool=True,no_downstream_reranking=True,
        labels_opened_in_native_worker=False,closed_test_opened=False,full_encoder_oof=False,recomputed_queries=changed))


def run(out):
    with (out/'owned.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        stages=[('prepare',None)]+[('generate',v) for v in VARIANTS]
        for action,variant in stages:
            target=out/variant if variant else out;target.mkdir(exist_ok=True)
            if (target/(action+'_exit.json')).exists() and io.read(target/(action+'_exit.json'))['exit_code']==0:continue
            while True:
                authority(out);command=[str(engine.CPU),'-u','-m','scripts.embeddinggemma2_v132_local_evidence_ablation',action,'--output',str(out)]
                if variant:command+=['--variant',variant]
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(out/'spill'),RAYON_NUM_THREADS='2',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
                with (target/(action+'.log')).open('ab',buffering=0) as log:
                    child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
                    io.write(target/'owned_process.json',dict(pid=child.pid,command=command,owned=True,started_unix=time.time()))
                    while child.poll() is None:
                        if engine.INTERRUPTED or time.time()>=DEADLINE-30:
                            child.terminate()
                            try:child.wait(timeout=25)
                            except subprocess.TimeoutExpired:child.kill();child.wait()
                        time.sleep(5)
                    rc=child.wait()
                io.write(target/(action+'_exit.json'),dict(exit_code=rc,pid=child.pid,own_child_exit_confirmed=True,finished_unix=time.time()))
                if rc==75:time.sleep(10);continue
                if rc:raise RuntimeError(f'{action}/{variant} failed exit{rc}, no partial metrics')
                break
        evaluate(out)


def evaluate(out):
    from scripts import embeddinggemma2_v32_learned_query as evaluator
    authority(out)
    for v in VARIANTS:
        root=out/v;ex=io.read(root/'generate_exit.json');g=io.read(root/'generation.json')
        if ex['exit_code'] or not ex['own_child_exit_confirmed'] or g['status']!='COMPLETE_SEALED_ALL526':raise ValueError('All predeclared generations complete before labels')
        verify(g['outputs']);verify(g['timings'])
    old={r['query_id']:r for r in io.read(BASE/'outcomes.json')['learned_native_ooc']};summary={}
    for v in VARIANTS:
        root=out/v
        io.write(root/'launch.json',io.read(out/'launch.json'));(root/'calibration').mkdir(exist_ok=True)
        io.write(root/'calibration/generation.json',io.read(out/'calibration/generation.json'))
        evaluator.proof=sys.modules[__name__];evaluator.evaluate(root)
        fresh={r['query_id']:r for r in io.read(root/'outcomes.json')['learned_native_ooc']}
        if set(fresh)!=set(old):raise ValueError('Exact full paired population')
        report=io.read(root/'report.json');paired=dict(gains=[q for q in old if fresh[q]['exact'] and not old[q]['exact']],
            losses=[q for q in old if old[q]['exact'] and not fresh[q]['exact']])
        report.update(paired_vs_v115=paired,eligibility=RECIPE['eligibility'],production_promotion_allowed=False,
            no_full_encoder_oof=True,unchanged_shard_reuse_not_filtered_denominator=True)
        io.write(root/'report.json',report);summary[v]=dict(report=record(root/'report.json'),paired=paired,
            metrics=report['variants']['learned_native_ooc'])
    io.write(out/'report.json',dict(status='COMPLETE_SEVEN_PREDECLARED_ALL526_EXPLORATORY_ABLATIONS',variants=summary,
        no_independent_certification=True,no_source_or_model_admission=True,finished_unix=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['initialize','prepare','generate','run','evaluate'])
    p.add_argument('--output',type=Path,default=ROOT);p.add_argument('--variant',choices=VARIANTS);a=p.parse_args()
    signal.signal(signal.SIGTERM,engine.interrupted);signal.signal(signal.SIGINT,engine.interrupted)
    try:
        if a.action=='generate':generate(a.output,a.variant)
        else:globals()[a.action](a.output)
    except engine.ResourceYield as error:print(str(error),flush=True);sys.exit(75)
    except Exception as error:
        if a.output.exists():io.write(a.output/(a.action+'_error.json'),dict(error=repr(error),updated_unix=time.time()))
        raise
