"""Query-only one-hop succession views BEFORE each native collector100.

Structural ablation, not the final learned multichannel search engine. Scan
ALL unpruned commune scores and ALL national sites inside each BAN footprint.
Never expand a top100 or rerank a fixed union. Historical national lexical
pipeline is preserved as reference ONLY. No labels/GT-directed edges in search.
"""
import argparse
from collections import defaultdict
from datetime import date
import gc
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time

import numpy as np

from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v14_full_pipeline as engine
from scripts import embeddinggemma2_v17_structural_retrieval as structural
from scripts import embeddinggemma2_v18_identity_retrieval as identity
from src.xgb_matcher.retrieval_v20_workbench import record, verify

CACHE = io.HOME/'v19/proof_engine_20261011_b'
SOURCE = structural.SOURCE
OUTPUT = io.HOME/'v21/relation_views_20261011_a'
SUCCESSION_CONFIG = io.ROOT/'config/retrieval_v133_all_succession.json'
RECIPE = dict(scope='ALL526_CONSUMED_DEVELOPMENT_NOT_PRISTINE_HOLDOUT', cap=100,
    relation='DIRECT_OFFICIAL_SUCCESSION_BOTH_ORIENTATIONS_ONE_HOP_ONLY',
    eligibility='EXPLICIT_ECONOMIC_CONTINUITY_TRUE_DATED_ON_OR_BEFORE_2026_08_01',
    score='MAX_OWN_VIEW_AND_DIRECT_NEIGHBOUR_VIEW_UNCHANGED_CHANNEL_SCORE',
    dense='UNCHANGED_V18_0.75_SITE_PLUS_0.25_PARENT_FULL_COMMUNES',
    geo='UNCHANGED_V17_0.90_OFFICIAL_NAME_PLUS_0.10_DISTANCE_FULL_NATIONAL_FOOTPRINT',
    tie_break='OWN_VIEW_BEFORE_CONTEXT_THEN_ASCENDING_OUTPUT_SIRET',
    context_is_not_own_alias=True, succession_is_not_legal_identity=True,
    moving_successions_not_same_site=True, no_transitive_closure=True,
    no_fixed_prepool=True, no_new_fusion_of_capped_channels=True,
    lexical='KEPT_IN_UNCHANGED_HISTORICAL_V18_REFERENCE_NOT_NEW_COMBINER',
    no_new_score_weights=True, no_fit=True, no_encoding=True, no_grid=True,
    learned_combiner_not_yet_measured=True, downstream_changed=False,
    variants=['baseline','v18_reference','joint_own','joint_relation','geo_own','geo_relation'])


def valid_link(continuity, effective_date):
    return continuity is True and isinstance(effective_date, date) and effective_date <= date(2026,8,1)


def footprint_radius(geometry):
    # Exactly the original V17 rule: street-fallback geometries carry an
    # explicit footprint; ordinary BAN geometries do not have this field.
    return geometry.get('footprint_radius_m',max(
        1000 if geometry['ban_declared_weak'] else 300,
        3*max(0,geometry['ban_uncertainty_m'])))


def collect(keys, scores, edges, cap=100):
    """All views first, one native top-k. No propagation of propagated scores.

    edges[source] = (target, original row) tuples. IDs join/tie-break ONLY.
    Neighbour scores retain their role and original official link provenance.
    """
    if len(keys)!=len(scores) or len(set(map(int,keys)))!=len(keys):
        raise ValueError('Unique full source corpus and exact score alignment')
    if not np.isfinite(scores).all(): raise ValueError('Finite full scores')
    values={int(k):(float(s),0,None,None) for k,s in zip(keys,scores)}
    for k,s in zip(keys,scores):
        source=int(k)
        for target, link in edges.get(source,()):
            view=(float(s),1,source,link)
            old=values.get(target)
            if old is None or view[0]>old[0] or (view[0]==old[0] and
                (view[1],source)<(old[1],old[2] if old[2] is not None else -1)):
                values[target]=view
    selected=sorted(values,key=lambda k:(-values[k][0],values[k][1],k))[:cap]
    result=[]
    for k in selected:
        score,depth,peer,link=values[k]; key=f'{k:014d}'
        result.append(dict(siret=key,siren=key[:9],score=score,view_role='DIRECT_SUCCESSION_CONTEXT' if depth else 'OWN_VIEW',
            context_endpoint=f'{peer:014d}' if peer is not None else None,official_link=link,
            context_is_not_own_alias=True))
    return result


def initialize(out,prepared_from=None):
    from src.xgb_matcher.retrieval_query_ban_v327 import surface
    if out.exists(): raise ValueError('Fresh immutable namespace required')
    source=engine.authority(SOURCE); panel=io.read(verify(source['panel'])); surface(panel)
    if len(panel)!=526: raise ValueError('Full526 query-only scope')
    cache_launch=io.read(CACHE/'launch.json'); verify(cache_launch['implementation'])
    exit_record=io.read(CACHE/'generate_exit.json')
    if exit_record['exit_code']!=0 or not exit_record['own_child_exit_confirmed']:
        raise ValueError('Owned full feature generation exit0 required')
    if io.read(CACHE/'generation.json')['status']!='COMPLETE_SEALED_ALL526':
        raise ValueError('Complete unpruned feature cache')
    if cache_launch['source_launch']!=record(SOURCE/'launch.json') or cache_launch['adapter']!=source['new_adapter']:
        raise ValueError('Identical query corpus and frozen encoder')
    config=io.read(SUCCESSION_CONFIG); verify(config['source'])
    out.mkdir(parents=True);io.write(out/'recipe.json',RECIPE)
    io.write(out/'launch.json',dict(implementation=record(Path(__file__)),recipe=record(out/'recipe.json'),
        helpers=[record(Path(m.__file__)) for m in (io,engine,structural,identity)]+
            [record(io.ROOT/p) for p in ('src/xgb_matcher/retrieval_v20_workbench.py',
              'src/xgb_matcher/retrieval_query_ban_v327.py','scripts/embeddinggemma2_v15_probability_pilot.py',
              'scripts/embeddinggemma2_v3_compare.py')],
        source_launch=record(SOURCE/'launch.json'),feature_launch=record(CACHE/'launch.json'),
        source_successions=config['source'],source_successions_config=record(SUCCESSION_CONFIG),
        baseline_outputs=record(SOURCE/'all_outputs.json'),
        v18_outputs=io.read(identity.OUTPUT/'generation.json')['outputs'],
        v17_outputs=io.read(structural.OUTPUT/'generation.json')['outputs'],
        geo=cache_launch['geo'],names=cache_launch['names'],created_unix=time.time(),truth_opened=False))
    if prepared_from is not None:
        old=io.read(prepared_from/'launch.json');current=io.read(out/'launch.json')
        if io.read(verify(old['recipe']))!=RECIPE or any(old[k]!=current[k] for k in
                ('source_launch','geo','names','source_successions','baseline_outputs','feature_launch')):
            raise ValueError('Prepared geometry must have identical scientific bindings')
        ex=io.read(prepared_from/'generate_exit.json')
        if ex['exit_code']==0 or not ex['own_child_exit_confirmed'] or (prepared_from/'generation.json').exists():
            raise ValueError('Only fully exited failed prelabel execution may supply geometry')
        path=engine.sealed(prepared_from/'geo_sources.json')
        if not path:raise ValueError('Complete sealed query-only geometry required')
        io.write(out/'geo_sources.json',io.read(path));engine.seal(out/'geo_sources.json')
        current['prepared_geometry']=record(path);io.write(out/'launch.json',current)


def authority(out):
    launch=io.read(out/'launch.json')
    for r in (launch['implementation'],launch['recipe'],launch['source_launch'],launch['feature_launch'],
              launch['source_successions_config'],*launch['helpers']): verify(r)
    source=engine.authority(SOURCE)
    return launch,source,io.read(verify(launch['feature_launch']))


def geometry_sources(panel, launch, source, out):
    """All national footprint leaves; not V17 top100 used as graph seeds."""
    saved=engine.sealed(out/'geo_sources.json')
    if saved:return io.read(saved)
    keys=np.load(verify(source['geographic']['keys']),mmap_mode='r',allow_pickle=False)
    xy=np.load(verify(source['geographic']['xy']),mmap_mode='r',allow_pickle=False)
    if len(keys)!=43896818 or len(keys)!=len(xy):raise ValueError('Full national XY alignment')
    geos={r['query_id']:r['geometry'] for r in io.read(verify(launch['geo']))}
    found={q['query_id']:[] for q in panel}
    try:
        for lo in range(0,len(keys),1048576):
            engine.guard();a=np.asarray(xy[lo:lo+1048576]);kk=keys[lo:lo+len(a)]
            valid=np.isfinite(a).all(axis=1)&(a[:,0]>=0)&(a[:,0]<=1300000)&(a[:,1]>=6000000)&(a[:,1]<=7300000)
            for q in panel:
                g=geos[q['query_id']]
                if not g['spatial_crs_compatible']:continue
                x,y=g['ban_x_l93'],g['ban_y_l93'];radius=footprint_radius(g)
                mask=valid&(np.abs(a[:,0]-x)<=radius)&(np.abs(a[:,1]-y)<=radius)
                loc=np.flatnonzero(mask)
                if not len(loc):continue
                dist=np.hypot(a[loc,0]-x,a[loc,1]-y);inside=dist<=radius
                found[q['query_id']].extend((int(k),float(d)) for k,d in zip(kk[loc][inside],dist[inside]))
            engine.progress(out,'ALL_NATIONAL_GEOMETRY_BEFORE_ANY_TOP100',lo+len(a),total=len(keys))
    finally:keys._mmap.close();xy._mmap.close()
    con=sqlite3.connect(f"file:{verify(launch['names'])}?mode=ro&immutable=1",uri=True)
    try:
        for q in panel:
            rows=found[q['query_id']];ks=[k for k,d in rows]
            parents=engine.lookup(con,1,{f'{k:014d}'[:9] for k in ks})
            sites=engine.lookup(con,2,[f'{k:014d}' for k in ks])
            n=structural.normalize(q['crm_name']);values=[]
            for k,d in rows:
                names=tuple({structural.normalize(v) for v in parents[int(f'{k:014d}'[:9])]|sites[k] if structural.normalize(v)})
                sim=structural.similarity(n,names)
                values.append([k,.90*sim+.10/(1+d/300)])
            found[q['query_id']]=values
    finally:con.close()
    io.write(out/'geo_sources.json',found);engine.seal(out/'geo_sources.json');return found


def write_selection(connection,sql,parameters,path):
    import pyarrow.parquet as pq
    reader=connection.execute(sql,parameters).fetch_record_batch(8192)
    with pq.ParquetWriter(path,reader.schema) as writer:
        for batch in reader:
            engine.guard();writer.write_batch(batch)


def compile_links(out, source_keys, launch, source):
    """Semijoin on ALL query-only search views, never labels or top100 seeds."""
    import duckdb
    import pyarrow as pa
    import pyarrow.parquet as pq
    saved=engine.sealed(out/'links.parquet')
    if not saved:
        path=verify(launch['source_successions'])
        if pq.ParquetFile(path).metadata.num_rows!=9730972:raise ValueError('Full original succession source')
        con=duckdb.connect();con.execute('SET threads=2');con.execute("SET memory_limit='1GB'")
        tmp=out/'duck_tmp';tmp.mkdir(exist_ok=True);con.execute('SET temp_directory=?',[str(tmp)])
        con.register('selected',pa.table({'key':pa.array(sorted(source_keys),type=pa.uint64())}))
        sql='''SELECT DISTINCT s.* FROM read_parquet(?) s WHERE
            continuiteEconomique IS TRUE AND dateLienSuccession <= DATE '2026-08-01' AND
            regexp_full_match(siretEtablissementPredecesseur, '[0-9]{14}') AND
            regexp_full_match(siretEtablissementSuccesseur, '[0-9]{14}') AND
            siretEtablissementPredecesseur <> siretEtablissementSuccesseur AND
            (CAST(siretEtablissementPredecesseur AS UBIGINT) IN (SELECT key FROM selected) OR
             CAST(siretEtablissementSuccesseur AS UBIGINT) IN (SELECT key FROM selected))
             ORDER BY siretEtablissementPredecesseur,siretEtablissementSuccesseur,
                      dateLienSuccession,transfertSiege,continuiteEconomique,
                      dateDernierTraitementLienSuccession'''
        try:
            write_selection(con,sql,[str(path)],out/'links.pending.parquet')
        finally:con.close()
        os.replace(out/'links.pending.parquet',out/'links.parquet');engine.seal(out/'links.parquet')
        saved=engine.sealed(out/'links.parquet')
    keys=np.load(verify(source['geographic']['keys']),mmap_mode='r',allow_pickle=False)
    edges=defaultdict(list);observations=0;missing=0
    try:
        for batch in pq.ParquetFile(saved).iter_batches(batch_size=8192):
            engine.guard()
            for raw in batch.to_pylist():
                if not valid_link(raw['continuiteEconomique'],raw['dateLienSuccession']):raise ValueError('Typed eligible source only')
                p=int(raw['siretEtablissementPredecesseur']);s=int(raw['siretEtablissementSuccesseur'])
                link={k:v.isoformat() if isinstance(v,date) else v for k,v in raw.items()}
                observations+=1
                for a,b in ((p,s),(s,p)):
                    if a not in source_keys:continue
                    i=np.searchsorted(keys,np.uint64(b))
                    if i>=len(keys) or int(keys[i])!=b:missing+=1;continue
                    edges[a].append((b,link))
    finally:keys._mmap.close()
    io.write(out/'link_counts.json',dict(full_source_observations=9730972,eligible_selected_observations=observations,
        query_only_source_sites=len(source_keys),directed_views=sum(map(len,edges.values())),
        targets_outside_official_catalog=missing,labels_used=False))
    return edges


def generate(out):
    launch,source,cache=authority(out);panel=io.read(verify(source['panel']))
    old=io.read(verify(launch['baseline_outputs']));v18=io.read(verify(launch['v18_outputs']))
    v17=io.read(verify(launch['v17_outputs']))
    geo=geometry_sources(panel,launch,source,out)
    source_keys={int(k) for rows in geo.values() for k,s in rows};bindings={}
    for zone,ref in sorted(cache['corpora'].items()):
        directory=CACHE/'zones'/(zone or 'UNKNOWN')
        if (directory/'features_binding.json').exists():
            b=io.read(engine.sealed(directory/'features_binding.json'))
            qs=[q for q in panel if q['crm_insee']==zone]
            if b['queries']!=qs or b['corpus']!=ref or b['recipe']!=cache['recipe'] or b['adapter']!=cache['adapter'] or not b['full_unpruned_scores']:
                raise ValueError('Exact unpruned query/document/model binding')
            bindings[zone]=b;source_keys.update(map(int,b['sirets']))
        elif io.read(verify(ref)):raise ValueError('Missing nonempty full commune scores')
    edges=compile_links(out,source_keys,launch,source);del source_keys
    outputs={v:{} for v in RECIPE['variants']};outputs['baseline']=old['gemma2_lexical_geo'];outputs['v18_reference']=v18['joint_lexical_oldgeo']
    total=0
    for zone in sorted(cache['corpora']):
        engine.guard();qs=[q for q in panel if q['crm_insee']==zone];b=bindings.get(zone)
        if b:
            dest=CACHE/'zones'/(zone or 'UNKNOWN');keys=np.array(b['sirets'],dtype=np.uint64)
            arrays={k:np.load(engine.sealed(dest/(k+'.npy')),mmap_mode='r',allow_pickle=False) for k in ('site','identity')}
            if any(a.shape!=(len(qs),len(keys)) for a in arrays.values()):raise ValueError('Full aligned matrix')
            try:
                for i,q in enumerate(qs):
                    scores=.75*arrays['site'][i]+.25*arrays['identity'][i];qid=q['query_id']
                    own=collect(keys,scores,{})
                    if [r['siret'] for r in own]!=[r['siret'] for r in v18['joint'][qid]] or any(abs(a['score']-z['score'])>1e-7 for a,z in zip(own,v18['joint'][qid])):
                        raise ValueError('All526 native joint replay before evaluation')
                    outputs['joint_own'][qid]=own;outputs['joint_relation'][qid]=collect(keys,scores,edges)
            finally:
                for a in arrays.values():a._mmap.close()
            total+=len(keys)
        else:
            for q in qs:
                outputs['joint_own'][q['query_id']]=[];outputs['joint_relation'][q['query_id']]=[]
        engine.progress(out,'ALL_COMMUNE_RELATIONAL_VIEWS_BEFORE_COLLECTOR100',total,total=2554848,zone=zone);gc.collect()
    for q in panel:
        qid=q['query_id'];rows=geo[qid];keys=np.array([k for k,s in rows],dtype=np.uint64);scores=np.array([s for k,s in rows])
        outputs['geo_own'][qid]=collect(keys,scores,{});outputs['geo_relation'][qid]=collect(keys,scores,edges)
        if [r['siret'] for r in outputs['geo_own'][qid]] != [r['siret'] for r in v17['geo'][qid]] or any(
                abs(a['score']-b['score'])>1e-7 for a,b in zip(outputs['geo_own'][qid],v17['geo'][qid])):
            raise ValueError('Exact full national geo replay before evaluation')
    # Do not merge capped channels into a new fixed candidate pool. This is
    # a structural ablation; the learned full-stream combiner remains future work.
    scope={q['query_id'] for q in panel}
    if total!=2554848 or any(set(v)!=scope for v in outputs.values()):raise ValueError('Complete all526 scope')
    for values in outputs.values():
        if any(len(v)>100 or len({r['siret'] for r in v})!=len(v) for v in values.values()):raise ValueError('Unique absolute100')
    authority(out);verify(launch['source_successions']);verify(launch['names'])
    io.write(out/'outputs.json',outputs)
    io.write(out/'generation.json',dict(status='COMPLETE_SEALED_ALL526',outputs=record(out/'outputs.json'),
        queries=526,commune_sites=total,national_geo_sites_scanned=43896818,links=record(out/'links.parquet'),
        no_new_fusion=True,truth_opened=False,all526_joint_replayed=True,no_gradient_steps=True))


def evaluate(out):
    authority(out);structural.evaluate(out)
    report=io.read(out/'report.json');rows=io.read(out/'outcomes.json');paired={}
    for channel in ('joint','geo'):
        before={r['query_id']:r for r in rows[channel+'_own']};after=rows[channel+'_relation']
        paired[channel]=dict(gained=[r['query_id'] for r in after if r['exact'] and not before[r['query_id']]['exact']],
            lost=[r['query_id'] for r in after if not r['exact'] and before[r['query_id']]['exact']])
    report.update(status='MEASURED_RELATIONAL_VIEW_ABLATION_ALL526',paired_view_ablation=paired,
        learned_combiner_measured=False,new_full_pipeline_measured=False,
        historical_national_lexical_reference_preserved=True,recipe=record(out/'recipe.json'))
    io.write(out/'report.json',report);io.write(out/'state.json',dict(status='COMPLETE_RELATIONAL_VIEW_ABLATION',report=record(out/'report.json')))


def run(out,resume=False,prepared_from=None):
    import fcntl
    lock=(io.HOME/(out.name+'.owned.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if resume:authority(out)
    else:initialize(out,prepared_from)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(out),RAYON_NUM_THREADS='2')
    while True:
        child=subprocess.Popen([str(engine.CPU),'-u','-m','scripts.embeddinggemma2_v21_relation_views','generate','--output',str(out)],env=env)
        io.write(out/'worker_process.json',dict(pid=child.pid,owned=True,started_unix=time.time()))
        while True:
            try:rc=child.wait(timeout=5);break
            except subprocess.TimeoutExpired:
                if engine.INTERRUPTED:
                    child.terminate()
                    try:rc=child.wait(timeout=45)
                    except subprocess.TimeoutExpired:child.kill();rc=child.wait()
                    io.write(out/'generate_exit.json',dict(exit_code=rc,pid=child.pid,own_child_exit_confirmed=True,interrupted=True))
                    raise engine.ResourceYield('Owned child stopped, artifacts preserved')
        io.write(out/'generate_exit.json',dict(exit_code=rc,pid=child.pid,own_child_exit_confirmed=True))
        if rc==75:
            io.write(out/'state.json',dict(status='WAITING_FOR_REAL_RESOURCES',updated_unix=time.time()))
            time.sleep(60);continue
        if rc:raise RuntimeError(f'Generation exit{rc}; artifacts preserved, no labels')
        break
    evaluate(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','resume','generate','evaluate']);p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--prepared-from',type=Path);a=p.parse_args()
    signal.signal(signal.SIGTERM,engine.interrupted);signal.signal(signal.SIGINT,engine.interrupted)
    try:
        if a.action in ('run','resume'):run(a.output,a.action=='resume',a.prepared_from)
        else:globals()[a.action](a.output)
    except engine.ResourceYield as e:print(str(e),flush=True);sys.exit(75)
    except Exception as e:
        if a.output.exists():io.write(a.output/'execution_error.json',dict(error=repr(e),updated_unix=time.time()))
        raise
