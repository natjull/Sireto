"""Post-audit hypotheses, frozen before new outcomes; no serving/training change.

Generation reads query-only CRM, official BAN/XY/name dictionaries and COMPLETE
commune documents. It never reads V16 cases, membership or label-side artifacts.
Each new channel searches its entire eligible corpus directly to <=100. There
is no fixed prepool, candidate injection or downstream model. Evaluation is a
separate process, gated by a complete generation and owned exit0.
"""
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import heapq
import math
from pathlib import Path
import re
import sqlite3
import time
import unicodedata

from scripts import embeddinggemma2_night as io
from src.xgb_matcher.retrieval_v20_workbench import record, verify

SOURCE = io.HOME / 'v14/representative_20261008_b'
OUTPUT = io.HOME / 'v17/structural_20261010_a'
BASELINE_SHA = '357031e7f79d190ba6b68c8dfecbbd3f945178a475f8354e087442593d94a9ff'
RECIPE = dict(
    scope='ALL_526_ALREADY_SEEN_DEVELOPMENT_QUERIES_NOT_NEW_HOLDOUT',
    street_fallback=dict(min_similarity=.90, min_margin=.03,
                         radius='ALL_BAN_STREET_POINTS_CENTROID_MAX_SPREAD_PLUS_300M',
                         precision='STREET_FOOTPRINT_NOT_HOUSE', weak=True),
    geo=dict(score='0.90_MAX_OFFICIAL_NAME_WRATIO_PLUS_0.10_DISTANCE_DECAY',
             distance_scale_m=300, entire_radius_scored=True),
    family=dict(min_parent_name_similarity=.85,
                score='0.55_PARENT_CONTEXT_NAME_PLUS_0.35_OWN_ADDRESS_PLUS_0.10_OWN_NAME',
                corpus='EVERY_OFFICIAL_SITE_IN_EACH_COMPLETE_SELECTED_COMMUNE',
                siblings_are_parent_context_not_site_aliases=True,
                untyped_links_used=False),
    fusion=dict(k=60, compare='SUM_VERSUS_MAX_SINGLE_CHANNEL_EVIDENCE',
                weights='DENSE_LEXICAL_FAMILY_1_GEO_ORIGINAL_RELIABILITY',
                no_fixed_prepool=True, output_cap=100),
    selection='NO_GRID_NO_POST_OUTCOME_RECIPE_CHANGE',
    variants=['baseline', 'geo_sum', 'max_old_geo', 'max_new_geo', 'max_new_geo_family'],
    final_test_opened=False, downstream_changed=False)

LEGAL = {'SA', 'SAS', 'SASU', 'SARL', 'EURL', 'SCI', 'STE', 'ETS'}
STREET = {'AV':'AVENUE', 'BD':'BOULEVARD', 'BLD':'BOULEVARD', 'R':'RUE',
          'RTE':'ROUTE', 'CHE':'CHEMIN', 'CH':'CHEMIN', 'IMP':'IMPASSE',
          'PL':'PLACE', 'ALL':'ALLEE', 'ST':'SAINT', 'STE':'SAINTE'}


def normalize(value, street=False):
    # Identifiers may join/output records but must never be matching features.
    value = re.sub(r'(?<!\d)(?:\d{14}|\d{9})(?!\d)', ' ', str(value))
    value = ''.join(c for c in unicodedata.normalize('NFKD', value.upper())
                    if not unicodedata.combining(c)).replace('&', ' ET ')
    words = re.findall(r'[A-Z0-9]+', value)
    if street:
        while words and (words[0].isdigit() or words[0] in ('BIS', 'TER', 'QUATER')):
            words.pop(0)
        return ' '.join(STREET.get(w, w) for w in words)
    return ' '.join(w for w in words if w not in LEGAL and w != 'ND')


def similarity(query, names):
    from rapidfuzz import process, fuzz
    if not query or not names: return 0.
    hit = process.extractOne(query, names, scorer=fuzz.WRatio, processor=None)
    return hit[1]/100 if hit else 0.


def choose_street(query, groups):
    """Street footprint, never pretend an unknown number has house precision.

    groups: (commune, street) -> ALL official projected BAN points. Prefer
    exact commune evidence, refuse fuzzy ties, and retain all points of the
    accepted street including distant endpoints.
    """
    from rapidfuzz.distance import JaroWinkler
    street = normalize(query['crm_street'], street=True)
    if not street or not query['crm_insee'] or query['crm_insee'].startswith(('97','98')):
        return None
    ranked = []
    for (commune, name), points in groups.items():
        name = normalize(name, street=True)
        if not name or not points: continue
        score = JaroWinkler.normalized_similarity(street, name)
        ranked.append((commune == query['crm_insee'], score, commune, name, points))
    if not ranked: return None
    # Merge equivalent normalized street spellings before checking ambiguity.
    merged = {}
    for local, score, commune, name, points in ranked:
        key = (local, score, commune, name)
        merged.setdefault(key, []).extend(points)
    ranked = sorted([(*k, v) for k,v in merged.items()], key=lambda x:(-x[0],-x[1],x[2],x[3]))
    local, score, commune, name, points = ranked[0]
    competitors = [r for r in ranked[1:] if r[0] == local]
    if score < .90 or (competitors and score-competitors[0][1] < .03): return None
    points = [(x,y) for x,y in points if math.isfinite(x) and math.isfinite(y)
              and 0<=x<=1300000 and 6000000<=y<=7300000]
    if not points: return None
    x = sum(p[0] for p in points)/len(points); y = sum(p[1] for p in points)/len(points)
    spread = max(math.hypot(a-x,b-y) for a,b in points)
    return dict(ban_resolved=True, spatial_crs_compatible=True,
                ban_x_l93=x, ban_y_l93=y, ban_declared_weak=True,
                ban_uncertainty_m=(spread+300)/3, footprint_radius_m=spread+300,
                precision='STREET_FOOTPRINT_NOT_HOUSE', matched_commune=commune,
                matched_street=name, street_similarity=score, source_points=len(points))


def push100(heap, candidate):
    # Numeric SIRET is ONLY a deterministic tie-break, never a score feature.
    entry = (candidate['score'], -int(candidate['siret']), candidate)
    if len(heap)<100: heapq.heappush(heap, entry)
    elif entry[:2] > heap[0][:2]: heapq.heapreplace(heap, entry)


def finish(heap):
    return [r[2] for r in sorted(heap, key=lambda e:(-e[0],-e[1]))]


def fuse(channels, mode):
    if mode not in ('sum','max'): raise ValueError('Frozen fusion mode')
    scores = {}; parents = {}
    for rows, weight in channels:
        if len(rows)>100 or len({r['siret'] for r in rows})!=len(rows) or not 0<=weight<=1:
            raise ValueError('Independent initial100 cap/weight')
        for rank, row in enumerate(rows, 1):
            key = row['siret']; score = weight/(60+rank); parents[key] = row['siren']
            scores[key] = scores.get(key, 0)+score if mode=='sum' else max(scores.get(key,0),score)
    return [dict(siret=k,siren=parents[k],score=scores[k])
            for k in sorted(scores,key=lambda k:(-scores[k],k))[:100]]


def fallback_geometries(panel, original, launch, out):
    import duckdb
    import pyarrow as pa
    from src.xgb_matcher.retrieval_geo_v16 import lonlat_to_l93
    from scripts import embeddinggemma2_v14_full_pipeline as engine
    missing = [q for q in panel if q['crm_insee'] and not q['crm_insee'].startswith(('97','98'))
               and normalize(q['crm_street'],street=True)
               and not original[q['query_id']]['geometry']['spatial_crs_compatible']]
    con = duckdb.connect(); con.execute("SET threads=2"); con.execute("SET memory_limit='1GB'")
    tmp = out/'duck_tmp'; tmp.mkdir(); con.execute('SET temp_directory=?',[str(tmp)])
    geometry = {q:r['geometry'] for q,r in original.items()}
    if not missing: con.close(); return geometry
    con.register('queries', pa.Table.from_pylist(missing)); groups = defaultdict(lambda:defaultdict(list))
    try:
        reader = con.execute('''SELECT q.query_id, b.code_commune, b.nom_voie, b.longitude, b.latitude
            FROM queries q JOIN read_parquet(?) b ON
            ((q.crm_insee<>'' AND b.code_commune=q.crm_insee) OR
             (q.crm_postcode<>'' AND b.code_postal=q.crm_postcode))
            WHERE b.longitude IS NOT NULL AND b.latitude IS NOT NULL''',
            [str(verify(launch['geographic']['ban']))]).fetch_record_batch(8192)
        for batch in reader:
            engine.guard()
            for row in batch.to_pylist():
                if not (-6<=float(row['longitude'])<=10 and 41<=float(row['latitude'])<=52): continue
                point = lonlat_to_l93(float(row['longitude']),float(row['latitude']))
                groups[row['query_id']][(row['code_commune'],row['nom_voie'])].append(point)
        for q in missing:
            footprint = choose_street(q, groups[q['query_id']])
            if footprint: geometry[q['query_id']] = {**footprint, 'query_id':q['query_id'],'fold':q['fold']}
    finally: con.close()
    io.write(out/'geometries.json', geometry)
    return geometry


def geo_generate(panel, original, launch, con, out):
    import numpy as np
    from scripts import embeddinggemma2_v14_full_pipeline as engine
    from scripts.embeddinggemma2_v5_geo import geo_radius
    geometry = fallback_geometries(panel, original, launch, out)
    geo = launch['geographic']; keys=np.load(verify(geo['keys']),mmap_mode='r',allow_pickle=False)
    xy=np.load(verify(geo['xy']),mmap_mode='r',allow_pickle=False)
    if len(keys)!=43896818 or keys.dtype!=np.uint64 or xy.shape!=(len(keys),2) or xy.dtype!=np.float64:
        raise ValueError('Frozen full national XY namespace')
    radii={q:g.get('footprint_radius_m',geo_radius(g)) for q,g in geometry.items()}
    heaps={q['query_id']:[] for q in panel}; counts=Counter(); previous=None
    names={q['query_id']:normalize(q['crm_name']) for q in panel}
    started=time.monotonic()
    try:
        for lo in range(0,len(keys),262144):
            engine.guard(); kk=keys[lo:lo+262144]; coord=xy[lo:lo+262144]
            if (previous is not None and int(kk[0])<=previous) or np.any(kk[1:]<=kk[:-1]):
                raise ValueError('National unique sorted keys')
            previous=int(kk[-1]); valid=np.isfinite(coord).all(axis=1)&(coord[:,0]>=0)&(coord[:,0]<=1300000)&(coord[:,1]>=6000000)&(coord[:,1]<=7300000)
            eligible={}
            for q,g in geometry.items():
                radius=radii[q]
                if radius is None: continue
                delta=coord-np.asarray([g['ban_x_l93'],g['ban_y_l93']]); d2=np.einsum('ij,ij->i',delta,delta)
                ii=np.flatnonzero(valid&(d2<=radius**2)); counts[q]+=len(ii)
                eligible[q]=[(int(kk[j]),float(math.sqrt(d2[j]))) for j in ii]
            all_keys={k for rows in eligible.values() for k,d in rows}
            own=engine.lookup(con,2,all_keys); legal=engine.lookup(con,1,{k//100000 for k in all_keys})
            aliases={k:tuple(normalize(n) for n in own[k]|legal[k//100000] if normalize(n)) for k in all_keys}
            for q,rows in eligible.items():
                for key,distance in rows:
                    match=similarity(names[q],aliases[key]); score=.90*match+.10/(1+distance/300)
                    s=f'{key:014d}'; push100(heaps[q],dict(siret=s,siren=s[:9],score=score,
                        name_similarity=match,distance_m=distance))
            io.write(out/'state.json',dict(status='NATIONAL_NAME_AWARE_GEO',scanned=min(lo+262144,len(keys)),
                total=len(keys),eligible_scored=sum(counts.values()),seconds=time.monotonic()-started))
    finally: keys._mmap.close(); xy._mmap.close()
    rows=[dict(query_id=q['query_id'],geometry=geometry[q['query_id']],radius_m=radii[q['query_id']],
               official_sites_in_radius=counts[q['query_id']],candidates=finish(heaps[q['query_id']]),
               rrf_geo_weight=.5 if geometry[q['query_id']]['ban_declared_weak'] else 1.) for q in panel]
    io.write(out/'geo.json',rows)
    return {r['query_id']:r for r in rows}


def family_generate(panel, launch, con, out):
    import ijson
    from scripts import embeddinggemma2_v14_full_pipeline as engine
    groups=defaultdict(list)
    for q in panel: groups[q['crm_insee']].append(q)
    heaps={q['query_id']:[] for q in panel}; count=0
    query_names={q['query_id']:normalize(q['crm_name']) for q in panel}
    @lru_cache(maxsize=16384)
    def portfolio(parent):
        legal={n for _,n in con.execute('SELECT owner_key,name FROM aliases WHERE kind=1 AND owner_key=?',[parent])}
        # All sites nationwide, not only those in the selected commune. Same
        # SIREN is official ownership, NOT proof each sibling name names this site.
        sites={n for _,n in con.execute('SELECT owner_key,name FROM aliases WHERE kind=2 AND owner_key>=? AND owner_key<?',
                                       [parent*100000,(parent+1)*100000])}
        return tuple(sorted({normalize(n) for n in legal|sites if normalize(n)}))
    @lru_cache(maxsize=32768)
    def parent_similarity(qid,parent):
        return similarity(query_names[qid],portfolio(parent))
    for zone,queries in sorted(groups.items()):
        addresses={q['query_id']:normalize(' '.join(q[k] for k in ('crm_number','crm_suffix','crm_street'))) for q in queries}
        directory=engine.zone_path(SOURCE,zone)
        exited=io.read(directory/'prepare_exit.json')
        if exited.get('exit_code')!=0 or not exited.get('own_child_exit_confirmed'):
            raise ValueError('Complete owned commune preparation exit0')
        ref=io.read(directory/'documents.seal.json'); n=0
        with verify(ref).open('rb') as stream:
            for doc in ijson.items(stream,'item'):
                n+=1; parent=int(doc['siren'])
                own=tuple(normalize(v) for v in doc['own_names'] if normalize(v))
                addr=tuple(normalize(v) for v in doc['all_addresses'] if normalize(v))
                for q in queries:
                    qid=q['query_id']; p=parent_similarity(qid,parent)
                    if p < .85: continue
                    a=similarity(addresses[qid],addr); site=similarity(query_names[qid],own)
                    push100(heaps[qid],dict(siret=doc['key'],siren=doc['siren'],score=.55*p+.35*a+.10*site,
                        parent_context_name_similarity=p,own_site_address_similarity=a,own_site_name_similarity=site,
                        siblings_are_parent_context_not_aliases=True))
                if n%8192==0: engine.guard()
        count+=n; engine.guard()
        io.write(out/'state.json',dict(status='COMPLETE_COMMUNE_PARENT_SITE_SEARCH',zone=zone,sites_scored=count))
    if count!=2554848: raise ValueError('Every complete selected-commune site required')
    values={q:finish(h) for q,h in heaps.items()}; io.write(out/'family.json',values)
    return values


def generate(out):
    from scripts import embeddinggemma2_v14_full_pipeline as engine
    from src.xgb_matcher.retrieval_query_ban_v327 import surface
    if out.exists(): raise ValueError('Immutable new namespace required; no overwrite/relaunch')
    source=engine.authority(SOURCE)
    if io.read(SOURCE/'state.json')['status']!='COMPLETE_PROBABILITY_PILOT': raise ValueError('Complete baseline required')
    panel=io.read(verify(source['panel'])); scope={q['query_id'] for q in panel}
    surface(panel)  # exact typed query-only field whitelist, no label/features.
    if len(panel)!=526 or len(scope)!=526 or any(q['fold'] not in (2,3,4) for q in panel): raise ValueError('All526 safe-fold queries')
    original=io.read(verify(io.read(SOURCE/'generation.json')['outputs']))
    if io.digest(SOURCE/'all_outputs.json')!=BASELINE_SHA: raise ValueError('Immutable completed V15 baseline binding')
    geo={r['query_id']:r for r in io.read(verify(source['projected_geo']))}
    out.mkdir(parents=True); io.write(out/'recipe.json',RECIPE)
    names=io.read(SOURCE/'names/generation.json')['database']
    launch=dict(source_launch=record(SOURCE/'launch.json'),baseline_outputs=record(SOURCE/'all_outputs.json'),
        recipe=record(out/'recipe.json'),implementation=record(Path(__file__)),names=names,
        helpers=[record(Path(engine.__file__)),record(Path(io.__file__)),
                 record(io.ROOT/'src/xgb_matcher/retrieval_geo_v16.py'),
                 record(io.ROOT/'src/xgb_matcher/retrieval_query_ban_v327.py'),
                 record(io.ROOT/'src/xgb_matcher/retrieval_query_population_v192.py'),
                 record(io.ROOT/'src/xgb_matcher/retrieval_v20_workbench.py'),
                 record(io.ROOT/'scripts/embeddinggemma2_v5_geo.py'),
                 record(io.ROOT/'scripts/embeddinggemma2_v15_probability_pilot.py'),
                 record(io.ROOT/'scripts/embeddinggemma2_v3_compare.py')],
        created_unix=time.time(),labels_opened=False)
    io.write(out/'launch.json',launch)
    con=sqlite3.connect(f'file:{verify(names)}?mode=ro&immutable=1',uri=True)
    try:
        new_geo=geo_generate(panel,geo,source,con,out)
        family=family_generate(panel,source,con,out)
    finally: con.close()
    outputs={'baseline':original['gemma2_lexical_geo'],'geo':{q:new_geo[q]['candidates'] for q in scope},'family':family}
    for mode,g,tag,include_family in [('sum',new_geo,'geo_sum',False),('max',geo,'max_old_geo',False),
                                     ('max',new_geo,'max_new_geo',False),('max',new_geo,'max_new_geo_family',True)]:
        outputs[tag]={q:fuse([(original['gemma2'][q],1.),(original['lexical'][q],1.),
            (g[q]['candidates'],g[q]['rrf_geo_weight'])]+([(family[q],1.)] if include_family else []),mode) for q in scope}
    for values in outputs.values():
        if set(values)!=scope: raise ValueError('Complete denominator before labels')
        for rows in values.values():
            if len(rows)>100 or len({r['siret'] for r in rows})!=len(rows): raise ValueError('Initial100 cap/unique')
    for ref in (*launch['helpers'],launch['recipe'],launch['implementation'],launch['source_launch'],launch['baseline_outputs'],names,
                source['geographic']['ban'],source['geographic']['keys'],source['geographic']['xy']): verify(ref)
    engine.authority(SOURCE); io.write(out/'outputs.json',outputs)
    io.write(out/'generation.json',dict(status='COMPLETE_SEALED_ALL526',outputs=record(out/'outputs.json'),
        queries=526,commune_sites=2554848,national_sites=43896818,truth_opened=False,test_opened=False))


def evaluate(out):
    from scripts import embeddinggemma2_v15_probability_pilot as pilot
    from scripts.embeddinggemma2_v3_compare import summarize
    launch=io.read(out/'launch.json'); verify(launch['implementation']); verify(launch['recipe'])
    for ref in launch['helpers']: verify(ref)
    ex=io.read(out/'generate_exit.json'); gen=io.read(out/'generation.json')
    if ex.get('exit_code')!=0 or not ex.get('own_child_exit_confirmed') or gen['status']!='COMPLETE_SEALED_ALL526':
        raise ValueError('Complete sealed generation owned exit0 BEFORE labels')
    outputs=io.read(verify(gen['outputs'])); scope=set(outputs['baseline'])
    # Existing complete baseline outcomes are already post-seal, all-human,
    # official operational sidecar verified. Reuse its EXACT metadata/weights.
    old=io.read(SOURCE/'outcomes.json')['gemma2_lexical_geo']
    if {r['query_id'] for r in old}!=scope or len(old)!=526: raise ValueError('Exact all526 original denominator')
    source=io.read(verify(launch['source_launch'])); manifest=io.read(verify(source['training_manifest']))
    membership=pilot.population.SOURCE/'validation_membership.json'
    if io.digest(membership)!=manifest['split_hashes']['validation']: raise ValueError('Label authority drift')
    labels={r['query_id']:r for r in io.read(membership) if r['query_id'] in scope}
    if set(labels)!=scope or any(r['fold'] not in (2,3,4) for r in labels.values()): raise ValueError('All526 safe-fold labels')
    import pyarrow.parquet as pq
    cfg=io.read(io.ROOT/'config/retrieval_v358_postseal_evaluation.json'); ev=io.read(verify(cfg['evaluation'])); side=io.read(verify(ev['label_side']))
    if side['never_used_in_retrieval'] is not True or side['ledger']!=cfg['ledger']: raise ValueError('Official label-side authority')
    for ref in (side['policy'],*side['files'].values()): verify(ref)
    meta={r['query_id']:r for r in pq.read_table(verify(cfg['ledger']),
        filters=[('fold','in',[2,3,4]),('query_id','in',sorted(scope))],use_threads=False).to_pylist()}
    accepted={q:{labels[q]['siret']} for q in scope}
    for r in pq.read_table(side['files']['acceptable_operational.parquet']['path'],
        filters=[('fold','in',[2,3,4]),('query_id','in',sorted(scope))],use_threads=False).to_pylist(): accepted[r['query_id']].add(r['siret'])
    sampling=io.read(verify(source['sampling'])); reports={}; outcomes={}
    for variant,values in outputs.items():
        rows=[]
        if set(values)!=scope: raise ValueError('Complete generation scope')
        for original in old:
            q=original['query_id']; ids={r['siret'] for r in values[q]}; gt=labels[q]
            rows.append({**original,'exact':gt['siret'] in ids,'operational':bool(ids&accepted[q]),
                'siren':gt['siren'] in {r['siren'] for r in values[q]}})
        gains=[r['query_id'] for r,o in zip(rows,old) if r['exact'] and not o['exact']]
        losses=[r['query_id'] for r,o in zip(rows,old) if o['exact'] and not r['exact']]
        if variant=='baseline' and any(r[f]!=o[f] for r,o in zip(rows,old) for f in ('exact','siren','operational')):
            raise ValueError('Baseline must replay every outcome')
        reports[variant]=dict(raw_sample=summarize(rows),weighted={f:pilot.weighted_report(rows,sampling,f) for f in ('exact','siren','operational')},
            gained=gains,lost=losses,segments={f:{v:summarize([r for r in rows if str(r[f])==v]) for v in sorted({str(r[f]) for r in rows})} for f in ('fold','stratum','role')})
        for f in ('label_kind','qualification_v2','qualification_v3','data_origin','ground_truth_state'):
            groups=defaultdict(list)
            for row in rows: groups[str(meta.get(row['query_id'],{}).get(f,'UNKNOWN'))].append(row)
            reports[variant]['segments'][f]={v:summarize(values) for v,values in groups.items()}
        outcomes[variant]=rows
    io.write(out/'outcomes.json',outcomes)
    io.write(out/'report.json',dict(status='MEASURED_STRUCTURAL_DEVELOPMENT_ALL526',variants=reports,
        generation=record(out/'generation.json'),already_seen_development=True,full_oof_measured=False,
        production_slo_measured=False,pristine_test_opened=False,no_99_2pct_certification=True,
        no_automatic_serving_promotion=True,all_human_denominator_526=True))
    io.write(out/'state.json',dict(status='COMPLETE_STRUCTURAL_COMPARISON',report=record(out/'report.json')))


def run(out):
    import subprocess, sys
    child=subprocess.Popen([sys.executable,'-u','-m','scripts.embeddinggemma2_v17_structural_retrieval','generate','--output',str(out)])
    exit_code=child.wait()
    if out.exists(): io.write(out/'generate_exit.json',dict(exit_code=exit_code,pid=child.pid,own_child_exit_confirmed=True))
    if exit_code: raise RuntimeError(f'Generation exit{exit_code}; retain artifacts, no labels')
    evaluate(out)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=['run','generate','evaluate']); parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args(); {'run':run,'generate':generate,'evaluate':evaluate}[args.action](args.output)
