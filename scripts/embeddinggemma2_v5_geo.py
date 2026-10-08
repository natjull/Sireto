"""Independent national physical-neighbour initial100, no labels in generation."""
import argparse
from collections import Counter,defaultdict
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v4_fit as fit
from scripts import embeddinggemma2_v4_compare as paired
from src.xgb_matcher import retrieval_query_ban_v327 as ban
from src.xgb_matcher.retrieval_v20_workbench import record,verify

OUTPUT=io.HOME/'v5'


def geo_radius(g):
    if not g['spatial_crs_compatible']: return None
    sigma=float(g['ban_uncertainty_m'])
    if not math.isfinite(sigma) or sigma<=0: raise ValueError('Real positive BAN uncertainty')
    return max(1000. if g['ban_declared_weak'] else 300.,3*sigma)


def nearest100(previous,distances,keys):
    rows=previous+[(float(d),int(k)) for d,k in zip(distances,keys)]
    return sorted(rows,key=lambda row:(row[0],row[1]))[:100]


def weighted_fusion(channels):
    scores=Counter(); parents={}
    for rows,weight in channels:
        if not 0<=weight<=1 or len(rows)>100 or len({r['siret'] for r in rows})!=len(rows): raise ValueError('Initial100 source cap/weight')
        for rank,r in enumerate(rows,1):
            scores[r['siret']]+=weight/(60+rank); parents[r['siret']]=r['siren']
    return [dict(siret=k,siren=parents[k],score=scores[k]) for k in sorted(scores,key=lambda k:(-scores[k],k))[:100]]


def generate(out):
    import duckdb
    launch=io.read(out/'launch.json'); verify(launch['implementation']); source=io.read(verify(launch['comparison']))
    for ref in launch['helper_implementations']: verify(ref)
    queries=io.read(verify(source['panel'])); keys=np.load(verify(launch['keys']),mmap_mode='r',allow_pickle=False)
    xy=np.load(verify(launch['xy']),mmap_mode='r',allow_pickle=False)
    if keys.dtype!=np.uint64 or xy.dtype!=np.float64 or xy.shape!=(len(keys),2) or len(keys)!=43896818: raise ValueError('Official XY namespace')
    class Files:
        def file(self,ref):
            if ref!=launch['ban']: raise ValueError('Pinned BAN only')
            return verify(ref)
    con=duckdb.connect(); con.execute("SET threads=1"); con.execute("SET memory_limit='384MB'")
    temp=out/'duck_tmp'; temp.mkdir(); con.execute('SET temp_directory=?',[str(temp)])
    try: geometry=ban.resolve(con,queries,Files())
    finally: con.close()
    radii=[geo_radius(g) for g in geometry]; best=[[] for _ in queries]; matched=[0 for _ in queries]; previous_key=None
    start=time.monotonic()
    try:
        for lo in range(0,len(keys),262144):
            fit.resource_guard(launch['deadline_unix'],4*1024**3)
            chunk_keys=keys[lo:lo+262144]; coordinates=xy[lo:lo+262144]
            if (previous_key is not None and int(chunk_keys[0])<=previous_key) or np.any(chunk_keys[1:]<=chunk_keys[:-1]): raise ValueError('Keys not sorted unique')
            previous_key=int(chunk_keys[-1])
            valid=np.isfinite(coordinates).all(axis=1)&(coordinates[:,0]>=0)&(coordinates[:,0]<=1300000)&(coordinates[:,1]>=6000000)&(coordinates[:,1]<=7300000)
            for i,g in enumerate(geometry):
                if radii[i] is None: continue
                delta=coordinates-np.asarray([g['ban_x_l93'],g['ban_y_l93']]); d2=np.einsum('ij,ij->i',delta,delta)
                locations=np.flatnonzero(valid&(d2<=radii[i]**2))
                matched[i]+=len(locations)
                best[i]=nearest100(best[i],np.sqrt(d2[locations]),chunk_keys[locations])
            io.write(out/'state.json',dict(status='FULL_NATIONAL_GEOGRAPHIC_INITIAL100',scanned=min(lo+262144,len(keys)),total=len(keys),
                queries=len(queries),seconds=time.monotonic()-start))
    finally: keys._mmap.close(); xy._mmap.close()
    for ref in (launch['keys'],launch['xy'],launch['ban'],launch['implementation'],launch['comparison']): verify(ref)
    for ref in launch['helper_implementations']: verify(ref)
    outputs=[]
    for q,g,radius,top,n in zip(queries,geometry,radii,best,matched):
        candidates=[dict(siret=f'{key:014d}',siren=f'{key:014d}'[:9],distance_m=d,score=1/(1+d/500)) for d,key in top]
        outputs.append(dict(query_id=q['query_id'],fold=q['fold'],geometry=g,radius_m=radius,official_sites_in_radius=n,
            candidates=candidates,rrf_geo_weight=.5 if g['ban_declared_weak'] else 1.))
    io.write(out/'outputs.json',outputs)
    io.write(out/'generation.json',dict(status='SEALED_ALL80_NATIONAL_GEO_INITIAL100',outputs=record(out/'outputs.json'),
        queries=len(queries),scanned_sites=43896818,coordinate_namespace_appended_sites_missing=True,
        unresolved_queries=sum(r is None for r in radii),seconds=time.monotonic()-start,truth_opened=False,test_opened=False))


def evaluate(out):
    import pyarrow.parquet as pq
    launch=io.read(out/'launch.json'); source=io.read(verify(launch['comparison'])); comparison=Path(launch['comparison']['path']).parent
    previous=io.read(comparison/'report.json')
    if previous['status']!='MEASURED_DEVELOPMENT_PAIRED_INITIAL100_NOT_NATIONAL_GATE': raise ValueError('Complete paired report before labels')
    original=io.read(verify(previous['all_outputs']))['candidates']; report_ref=record(comparison/'report.json')
    ex=io.read(out/'generation_exit.json'); gen=io.read(out/'generation.json')
    if ex.get('exit_code')!=0 or not ex.get('own_child_exit_confirmed') or gen['status']!='SEALED_ALL80_NATIONAL_GEO_INITIAL100': raise ValueError('Owned complete geographic generation')
    geo={r['query_id']:r for r in io.read(verify(gen['outputs']))}; scope=set(geo)
    if scope!=set(original['national_lexical']) or len(scope)!=source['actual_queries']: raise ValueError('Exact same full cohort')
    candidates={'geo_only':{q:geo[q]['candidates'] for q in scope}}
    candidates['dense_lexical_geo_rrf']={q:weighted_fusion([(original['national_lexical'][q],1.),
        (original['v3_official512'][q],1.),(geo[q]['candidates'],geo[q]['rrf_geo_weight'])]) for q in scope}
    io.write(out/'fusion_sealed.json',dict(candidates=candidates,previous_report=report_ref,truth_opened=False))
    manifest=io.read(verify(source['training_manifest'])); path=io.HOME/'night_20261006_a/validation_membership.json'
    if io.digest(path)!=manifest['split_hashes']['validation']: raise ValueError('Validation membership drift')
    labels={r['query_id']:r for r in io.read(path) if r['query_id'] in scope}
    cfg=io.read(io.ROOT/'config/retrieval_v358_postseal_evaluation.json'); ev=io.read(verify(cfg['evaluation'])); side=io.read(verify(ev['label_side']))
    if side['never_used_in_retrieval'] is not True or side['ledger']!=cfg['ledger']: raise ValueError('Official sidecar authority')
    for ref in (side['policy'],*side['files'].values()): verify(ref)
    accepted={q:{labels[q]['siret']} for q in scope}
    for r in pq.read_table(side['files']['acceptable_operational.parquet']['path'],filters=[('fold','in',[2,3,4]),('query_id','in',sorted(scope))],use_threads=False).to_pylist(): accepted[r['query_id']].add(r['siret'])
    meta={r['query_id']:r for r in pq.read_table(verify(cfg['ledger']),filters=[('fold','in',[2,3,4]),('query_id','in',sorted(scope))],use_threads=False).to_pylist()}
    old_outcomes={r['query_id']:r for r in io.read(comparison/'outcomes.json')['v3_official512_rrf']}; outcomes={}; metrics={}
    for variant,values in candidates.items():
        rows=[]
        for q in sorted(scope):
            cs=values[q]; ids={r['siret'] for r in cs}; gt=labels[q]
            if len(cs)>100 or len(ids)!=len(cs): raise ValueError('Final100 cap')
            rows.append(dict(query_id=q,fold=gt['fold'],exact=gt['siret'] in ids,siren=gt['siren'] in {r['siren'] for r in cs},
                operational=bool(ids&accepted[q]),geometry_resolved=geo[q]['radius_m'] is not None,ban_declared_weak=geo[q]['geometry']['ban_declared_weak']))
        outcomes[variant]=rows; metrics[variant]=paired.summarize(rows)
        metrics[variant]['folds']={str(f):paired.summarize([r for r in rows if r['fold']==f]) for f in (2,3,4) if any(r['fold']==f for r in rows)}
        metrics[variant]['segments']={}
        for field in ('label_kind','qualification_v2','qualification_v3','data_origin','ground_truth_state'):
            grouped=defaultdict(list)
            for r in rows: grouped[str(meta.get(r['query_id'],{}).get(field,'UNKNOWN'))].append(r)
            metrics[variant]['segments'][field]={key:paired.summarize(values) for key,values in grouped.items()}
    fused={r['query_id']:r for r in outcomes['dense_lexical_geo_rrf']}
    io.write(out/'outcomes.json',outcomes)
    io.write(out/'report.json',dict(status='MEASURED_INDEPENDENT_GEO_AND_INITIAL100_UNION_DEVELOPMENT_ONLY',variants=metrics,
        gained=sum(fused[q]['exact'] and not old_outcomes[q]['exact'] for q in scope),lost=sum(old_outcomes[q]['exact'] and not fused[q]['exact'] for q in scope),
        previous_report=report_ref,all_queries_in_denominator=True,exact_operational_separate=True,
        national_geo_corpus_but_geographically_selected80_queries=True,full_oof_or_99percent_gate_validated=False,test_opened=False,downstream_models_changed=False))


def run(comparison,out):
    out.mkdir(parents=True,exist_ok=False); source=io.read(comparison/'launch.json'); paired.verify_launch(source)
    config=io.read(io.ROOT/'config/retrieval_v187_staged_initial.json'); structured=io.read(verify(config['structured'])); catalog=io.read(verify(structured['sources']['catalog']))
    deadline=io.read(verify(source['night']))['deadline_unix']
    io.write(out/'launch.json',dict(comparison=record(comparison/'launch.json'),keys=catalog['input_arrays']['endpoint_keys'],
        xy=structured['reused_arrays']['site_xy'],ban=ban.BAN,structured=config['structured'],catalog=structured['sources']['catalog'],
        implementation=record(Path(__file__)),deadline_unix=deadline,radii='MAX300M_RELIABLE_OR1000M_WEAK_AND3SIGMA',
        helper_implementations=[record(Path(ban.__file__)),record(Path(sys.modules[ban._resolve_ban.__module__].__file__)),
            record(io.ROOT/'src/xgb_matcher/retrieval_geo_v16.py')],
        score='PHYSICAL_DISTANCE_EXACT_NEAREST100_WITHIN_DECLARED_RADIUS',rrf_constant=60,weak_ban_weight=.5,truth_opened=False))
    with (out/'generation.log').open('x') as log:
        child=subprocess.run([sys.executable,'-u','-m','scripts.embeddinggemma2_v5_geo','generate','--output',str(out)],
            stdout=log,stderr=subprocess.STDOUT,timeout=max(1,deadline-time.time()))
    io.write(out/'generation_exit.json',dict(exit_code=child.returncode,own_child_exit_confirmed=True)); child.check_returncode()
    while time.time()<deadline:
        if (comparison/'report.json').exists():
            evaluate(out); io.write(out/'state.json',dict(status='COMPLETE_GEO_COMPARISON',test_opened=False)); return
        io.write(out/'state.json',dict(status='SEALED_GEO_WAITING_COMPLETE_DENSE_COMPARISON',test_opened=False)); time.sleep(30)
    io.write(out/'state.json',dict(status='SEALED_GEO_UNEVALUATED_NIGHT_DEADLINE',test_opened=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['run','generate','evaluate']); p.add_argument('--comparison',type=Path)
    p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    if not a.output.resolve().is_relative_to(OUTPUT): raise ValueError('Dedicated V5 geographic namespace')
    if a.action=='run': run(a.comparison,a.output)
    elif a.action=='generate': generate(a.output)
    else: evaluate(a.output)
