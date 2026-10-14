"""Whole-query source geography components, then separately postseal labels.

Never retrieval/model scores, top100 pools, coordinate fills or oracle unions.
Stage A reads no labels/outcomes. Stage B requires every526 shard + owned0.
"""
import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for _env in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_env] = '1'

import numpy as np
import psutil
from scripts import embeddinggemma2_night as io
from scripts.embeddinggemma2_v142_bano_native_ablation import compatible_points

HOME = Path('/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2')
V142 = HOME/'v142/bano_native_20261013_night'
OUT = HOME/'v157/geography_component_feasibility_20261014_night'
HELPER = Path(__file__).with_name('embeddinggemma2_v142_bano_native_ablation.py')
EXPECTED_HELPER = '568fadfc66f35ff3856290ee8a3a77466993d9ebba0b601d08975958f9cba9d7'
DEADLINE = 1791954368
DTYPE = np.dtype([('baseline5', '<f4'), ('bano_raw', '<f8'), ('flags', 'u1')])
RECIPE = dict(role='SOURCE_COMPONENT_AVAILABILITY_AND_RANK_INTERVALS_NOT_RETRIEVAL',
    queries='ALL526_ORIGINAL_FULL_NUMERIC_REFERENCES_INCLUDING_FOUR_EMPTY',
    baseline='EXACT_FROZEN_NUMERIC_FIELD5_ONLY_NOT_FULL_MODEL_SCORE',
    bano='EXACT_V142_COMPATIBLE_POINTS_ALL_HOUSE_ELSE_ALL_STREET_MIN_DISTANCE',
    bano_quality='0.5/(1+minimum_distance_m/300)_RAW_FLOAT64_NO_BASELINE_MAX',
    source_receivers='KEY_PRESENT_AND_eO_uO_AND_EXISTING_SUPPORTED_OWN_L93',
    baseline_query_guard='EXISTING_spatial_crs_compatible_ELSE_COMPONENT_ABSENT',
    bano_query_guard='UNCHANGED_V142_COMPATIBLE_POINTS_NO_EXTRA_BAN_GUARD',
    absent='NAN_COMPONENT_AND_EXPLICIT_FLAGS_NOT_FILTERED_QUERY_DENOMINATOR',
    packed='BASELINE_FLOAT32_BANO_FLOAT64_FLAGS_UINT8_KEYS_BOUND_TO_ORIGINAL_NPZ_HASH',
    flags=dict(canonical_key=1,both_public=2,supported_own_xy=4,baseline_query_geometry=8,bano_points=16),
    label_side='ONLY_AFTER_WHOLE_SOURCE_MANIFEST_AND_OWNED_EXIT0',
    ranks='BEST=1+STRICT_GREATER_WORST=GREATER_OR_EQUAL_NO_SIRET_TIE_BREAK',
    cutoffs=[50,100],strata=['ALL526','V115_BASELINE_MISSES','CURRENT_V142_PRIMARY_MISSES'],
    no_retrieval=True,no_fit=True,no_model_score=True,no_pool=True,no_oracle_union=True,
    metric_name='SOURCE_COMPONENT_POTENTIAL_NOT_RETRIEVAL_RECALL',
    baseline_precision='ORIGINAL_FLOAT32_TIES_NOT_EXACT_DISTANCE_RANKS',
    no_source_coordinate_label_recipe_change=True,closed_sets_opened=False,
    threads=1,max_rss_bytes=2*1024**3,max_component_disk_bytes=2*1024**3,
    free_percent=25,pressure_levels=[0,1,2],disk_reserve_gib=40,deadline_unix=DEADLINE)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()


def record(path):
    path=Path(path)
    return dict(path=str(path),sha256=digest(path),size_bytes=path.stat().st_size)


def verify(ref):
    p=Path(ref['path'])
    if digest(p)!=ref['sha256']:raise ValueError('Bound input drift: '+str(p))
    return p


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def guard():
    rss=psutil.Process().memory_info().rss
    free,pressure=io.pressure()
    if time.time()>=DEADLINE or rss>RECIPE['max_rss_bytes'] or free<25 or pressure not in (0,1,2):
        raise RuntimeError(f'STOP preserve resources/deadline rss={rss} free={free} pressure={pressure}')
    if psutil.disk_usage(HOME).free<40*1024**3:raise RuntimeError('CATNAT reserve below40GiB')
    return rss


def close_maps(*arrays):
    for array in arrays:
        if isinstance(array,np.memmap):array._mmap.close()


def source():
    guard()
    if OUT.exists():raise ValueError('Fresh namespace required')
    OUT.mkdir(parents=True);(OUT/'components').mkdir()
    write(OUT/'recipe.json',RECIPE)
    if digest(HELPER)!=EXPECTED_HELPER:raise ValueError('Exact V142 helper required')
    launch=read(V142/'launch.json')
    bindings={k:launch[k] for k in ('panel','baseline_features','geometry','keys','xy','public_mask','bano_generation')}
    bindings['implementation']=record(Path(__file__));bindings['helper']=record(HELPER)
    for ref in bindings.values():verify(ref)
    bg=read(bindings['bano_generation']['path']);bindings['bano_context']=bg['query_context'];verify(bindings['bano_context'])
    panel=read(bindings['panel']['path']);fg=read(bindings['baseline_features']['path'])
    contexts=read(bindings['bano_context']['path']);geometries={r['query_id']:r['geometry'] for r in read(bindings['geometry']['path'])}
    scope={q['query_id'] for q in panel}
    if len(panel)!=526 or len(scope)!=526 or set(contexts)!=scope or set(fg['features'])!=scope or set(geometries)!=scope:
        raise ValueError('Exact complete526 source scope')
    if any(q['fold'] not in (2,3,4) or any(k in q for k in ('label_siret','human_siret','ground_truth')) for q in panel):
        raise ValueError('Safe query-only panel required')
    write(OUT/'launch.json',dict(inputs=bindings,source_only=True,labels_read=False,created_unix=time.time()))
    queries=[];rows=0;disk=0;peak=0
    for q in panel:
        peak=max(peak,guard());qid=q['query_id'];ref=fg['features'][qid];path=verify(ref)
        with np.load(path,allow_pickle=False) as original:
            keys=original['keys'];x=original['features']
        peak=max(peak,guard())
        if keys.dtype!=np.uint64 or x.shape!=(len(keys),51) or np.any(keys[1:]<=keys[:-1]):
            raise ValueError('Sorted unique full51 numeric source')
        keyhash=hashlib.sha256(keys.tobytes()).hexdigest();baseline=x[:,5].copy();del x;gc.collect()
        allkeys=np.load(bindings['keys']['path'],mmap_mode='r');public=np.load(bindings['public_mask']['path'],mmap_mode='r')
        xy=np.load(bindings['xy']['path'],mmap_mode='r');at=np.searchsorted(allkeys,keys)
        found=at<len(allkeys);found[found]&=allkeys[at[found]]==keys[found]
        both=found.copy();both[both]&=public[at[both]]
        pos=np.flatnonzero(both);coords=np.asarray(xy[at[pos]],dtype=float)
        good=np.isfinite(coords).all(axis=1)&(coords[:,0]>=0)&(coords[:,0]<=1300000)&(coords[:,1]>=6000000)&(coords[:,1]<=7300000)
        pos,coords=pos[good],coords[good];supported=np.zeros(len(keys),dtype=bool);supported[pos]=True
        close_maps(allkeys,public,xy);del allkeys,public,xy,at,good
        points,precision=compatible_points(contexts[qid]);base_available=bool(geometries[qid]['spatial_crs_compatible'])
        stem=hashlib.sha256(qid.encode()).hexdigest();destination=OUT/'components'/(stem+'.npy')
        packed=np.lib.format.open_memmap(destination,mode='w+',dtype=DTYPE,shape=(len(keys),))
        packed['baseline5']=np.nan;packed['bano_raw']=np.nan
        packed['flags']=found.astype(np.uint8)+2*both.astype(np.uint8)+4*supported.astype(np.uint8)+(8 if base_available else 0)+(16 if points else 0)
        if base_available:packed['baseline5'][pos]=baseline[pos]
        if points:
            distance=np.full(len(pos),np.inf)
            for p in points:
                distance=np.minimum(distance,np.hypot(coords[:,0]-p[0],coords[:,1]-p[1]))
                peak=max(peak,guard())
            packed['bano_raw'][pos]=.5/(1+distance/300.)
            del distance
        packed.flush();peak=max(peak,guard());close_maps(packed)
        shard=record(destination);disk+=shard['size_bytes'];rows+=len(keys)
        if disk>RECIPE['max_component_disk_bytes']:raise RuntimeError('Component disk cap preserve partial source')
        queries.append(dict(query_id=qid,fold=q['fold'],original_numeric_ref=ref,ordered_keys_sha256=keyhash,
            rows=len(keys),canonical_key_present=int(found.sum()),both_public=int(both.sum()),supported_public_own_xy=len(pos),
            protected_or_not_both_public=int((found&~both).sum()),missing_canonical=int((~found).sum()),
            public_unsupported_xy=int((both&~supported).sum()),baseline_query_available=base_available,
            bano_points=len(points),bano_precision=precision,component=shard))
        del keys,baseline,found,both,supported,pos,coords,packed;gc.collect()
        if len(queries)%20==0:print('SOURCE',len(queries),rows,disk,flush=True)
        write(OUT/'source_state.json',dict(completed_queries=len(queries),total=526,rows=rows,labels_read=False))
    if len(queries)!=526 or sum(q['rows']==0 for q in queries)!=4:raise ValueError('ALL526 and four empty required')
    for ref in bindings.values():verify(ref)
    write(OUT/'source_report.json',dict(status='COMPLETE_ALL526_SOURCE_COMPONENTS_BEFORE_LABELS',launch=record(OUT/'launch.json'),
        recipe=record(OUT/'recipe.json'),queries=queries,rows=rows,empty_queries=4,packed_bytes=disk,
        original_coordinates_immutable=True,labels_read=False,outcomes_read=False,peak_guarded_rss=peak,completed_unix=time.time()))
    print('SOURCE_COMPLETE',rows,disk,peak,flush=True)


def interval(values,target):
    finite=np.isfinite(values)
    if not np.isfinite(target):return dict(available=False,eligible_candidates=int(finite.sum()))
    greater=int(np.count_nonzero(values>target));equal=int(np.count_nonzero(values==target))
    return dict(available=True,value=float(target),eligible_candidates=int(finite.sum()),strictly_greater=greater,equal=equal,
                best_rank=greater+1,worst_rank=greater+equal)


def summarize(rows):
    result=dict(queries=len(rows),scopes=dict(Counter(r['scope'] for r in rows)))
    for name in ('baseline5','bano_raw'):
        available=[r[name] for r in rows if r.get(name,{}).get('available')]
        result[name]=dict(available=len(available),absent=len(rows)-len(available),
            best_rank_min=min((r['best_rank'] for r in available),default=None),
            worst_rank_max=max((r['worst_rank'] for r in available),default=None))
        for key in ('best_rank','worst_rank','equal','eligible_candidates'):
            result[name][key+'_quantiles']={str(p):float(np.quantile([r[key] for r in available],p)) if available else None for p in (0,.25,.5,.75,.9,1)}
        for cap in RECIPE['cutoffs']:
            result[name][f'definitely_le_{cap}']=sum(r['worst_rank']<=cap for r in available)
            result[name][f'possibly_le_{cap}']=sum(r['best_rank']<=cap for r in available)
    return result


def label():
    guard()
    manifest=read(OUT/'source_manifest.json');ex=read(verify(manifest['owned_exit']))
    if ex['exit_code'] or not ex['own_child_exit_confirmed']:raise ValueError('Actual owned complete source before labels')
    sr=read(verify(manifest['source_report']));verify(sr['recipe']);verify(sr['launch'])
    if sr['status']!='COMPLETE_ALL526_SOURCE_COMPONENTS_BEFORE_LABELS' or len(sr['queries'])!=526:raise ValueError('Whole526 seal')
    for q in sr['queries']:verify(q['component'])
    # Label-side reads start strictly below this exhaustive whole-source seal.
    labeldir=OUT/'label_side'
    if labeldir.exists():raise ValueError('Fresh label-side directory')
    labeldir.mkdir()
    names=HOME/'v101/postseal_name_roles_20261013_a/report.json';status=HOME/'v103/raw_alias_coverage_20261013_a/report.json'
    nr=read(names);st=read(status);bindings=dict(names_report=record(names),statuses_report=record(status),
        labels=nr['cases'],statuses=st['statuses'],baseline_outcomes=record(HOME/'v115/streamed_activity_ablation_20261013_a/outcomes.json'),
        current_outcomes=record(V142/'all_new_sources_max_bano/outcomes.json'))
    labels={r['query_id']:r for r in read(verify(bindings['labels']))};statuses={r['qid']:r for r in read(verify(bindings['statuses']))}
    before={r['query_id']:r for r in read(verify(bindings['baseline_outcomes']))['learned_native_ooc']}
    current={r['query_id']:r for r in read(verify(bindings['current_outcomes']))['learned_native_ooc']}
    scope={r['query_id'] for r in sr['queries']}
    if any(set(items)!=scope for items in (labels,statuses,before,current)):raise ValueError('Exact ALL526 label-side denominator')
    write(labeldir/'launch.json',dict(source_manifest=record(OUT/'source_manifest.json'),bindings=bindings,
        labels_only_after_whole_source_and_owned_exit=True,created_unix=time.time()))
    rows=[]
    for q in sr['queries']:
        guard();qid=q['query_id'];s=statuses[qid]
        row=dict(query_id=qid,fold=q['fold'],baseline_exact=bool(before[qid]['exact']),current_exact=bool(current[qid]['exact']))
        rows.append(row)
        if s['site_diffusion']!='O' or s['owner_diffusion']!='O':
            row['scope']='PROTECTED_SKIP_BEFORE_TARGET_LOOKUP';continue
        target=np.uint64(int(labels[qid]['label_siret']))
        with np.load(verify(q['original_numeric_ref']),allow_pickle=False) as original:keys=original['keys']
        if len(keys)!=q['rows'] or hashlib.sha256(keys.tobytes()).hexdigest()!=q['ordered_keys_sha256']:raise ValueError('Original order binding')
        at=int(np.searchsorted(keys,target));inside=at<len(keys) and keys[at]==target
        del keys
        if not inside:row['scope']='NOT_IN_ORIGINAL_QUERY_NUMERIC_UNIVERSE';continue
        packed=np.load(q['component']['path'],mmap_mode='r');flags=int(packed['flags'][at])
        if not flags&1:row['scope']='CANONICAL_ENDPOINT_ABSENT'
        elif not flags&2:raise ValueError('Public label status versus source mask disagreement')
        elif not flags&4:row['scope']='OWN_XY_MISSING_OR_UNSUPPORTED'
        else:row['scope']='PUBLIC_SUPPORTED_OWN_XY'
        row['baseline5']=interval(packed['baseline5'],packed['baseline5'][at]);row['bano_raw']=interval(packed['bano_raw'],packed['bano_raw'][at])
        row['bano_source_points']=q['bano_points'];row['bano_precision']=q['bano_precision'];row['baseline_query_available']=q['baseline_query_available']
        close_maps(packed);del packed;gc.collect()
    summaries=dict(all526=summarize(rows),V115_baseline_misses=summarize([r for r in rows if not r['baseline_exact']]),
        current_V142_primary_misses=summarize([r for r in rows if not r['current_exact']]))
    if len(rows)!=526 or summaries['current_V142_primary_misses']['queries']!=19:raise ValueError('Current frozen19 denominator')
    write(labeldir/'all526.json',rows)
    write(labeldir/'report.json',dict(status='COMPLETE_SOURCE_COMPONENT_TARGET_RANK_INTERVALS_NOT_RECALL',
        source_manifest=record(OUT/'source_manifest.json'),launch=record(labeldir/'launch.json'),cases=record(labeldir/'all526.json'),
        summary=summaries,not_joint_model_admission_or_oracle_union=True,unsupported_protected_not_numeric_remain_absences=True,
        no_gain_label_error_national_SLO_or_ready_claim=True,closed_sets_opened=False,completed_unix=time.time()))
    print(json.dumps(summaries),flush=True)


def run(stage):
    guard()
    if stage=='label':read(OUT/'source_manifest.json')
    child=subprocess.Popen([sys.executable,'-u','-m','scripts.embeddinggemma2_v157_geography_component_feasibility',stage])
    try:
        while child.poll() is None:
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:guard()
    except BaseException:
        child.terminate();child.wait(timeout=10);raise
    directory=OUT if stage=='source' else OUT/'label_side'
    if not directory.exists():raise RuntimeError('No stage namespace')
    exitpath=directory/'owned_child_exit.json'
    write(exitpath,dict(exit_code=child.returncode,own_child_exit_confirmed=True,pid=child.pid,completed_unix=time.time()))
    if child.returncode:raise RuntimeError(f'{stage} child exited {child.returncode}; preserve artifacts')
    reportpath=directory/('source_report.json' if stage=='source' else 'report.json')
    manifestpath=OUT/'source_manifest.json' if stage=='source' else directory/'manifest.json'
    write(manifestpath,dict(status='COMPLETE_SEALED_'+stage.upper(),source_report=record(reportpath),owned_exit=record(exitpath)))
    print(json.dumps(record(manifestpath)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('run_source','run_label','source','label'));args=parser.parse_args()
    if args.action.startswith('run_'):run(args.action.removeprefix('run_'))
    else:globals()[args.action]()
