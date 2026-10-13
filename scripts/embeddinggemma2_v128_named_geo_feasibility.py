"""Generic one-query feasibility of public named POI geography, not recall.

No brand-specific rule, SIRET hint or label enters source/scoring. All shops in
the CRM neighbourhood are queried. ONE national native collector emits100.
The separately invoked postseal evaluator can report rank, never national recall.
"""
import argparse
import math
from pathlib import Path
import json
import time
import urllib.parse
import urllib.request
import numpy as np

from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v14_full_pipeline as engine
from scripts import embeddinggemma2_v125_street_shape_retrieval as shapes
from scripts import embeddinggemma2_v115_streamed_activity_ablation as original
from scripts.embeddinggemma2_v17_structural_retrieval import normalize
from scripts.embeddinggemma2_v123_street_geometry_probe import inverse_l93
from src.xgb_matcher.retrieval_geo_v16 import lonlat_to_l93
from src.xgb_matcher.hierarchical_retrieval import TantivyBackend
from src.xgb_matcher.retrieval_v20_workbench import record,verify

BASE=shapes.OUTPUT
STOP={'DE','DU','DES','LA','LE','LES','ET','D','L','AU','AUX'}


class NamedPlaceCollector:
    def __init__(self,backend,keys,scores,coefficient):
        self.backend=backend;self.searcher=backend.searcher;self.keys=keys;self.scores=scores;self.coefficient=coefficient
    def doc(self,loc):return self.searcher.doc(loc)
    def search(self,query,**kwargs):
        if kwargs.get('limit')!=100:raise ValueError('One collector100, no extra pool')
        tv=self.backend._tantivy;maximum=(1<<20)-1;values=np.floor(self.scores*maximum).astype(np.uint32);planes=[]
        for bit in range(20):
            pos=np.flatnonzero(values&(1<<bit))
            if len(pos):planes.append(tv.Query.const_score_query(tv.Query.term_set_query(
                self.backend.index.schema,'siret',[f'{int(self.keys[i]):014d}' for i in pos]),
                self.coefficient*(1<<bit)/maximum))
        if not planes:return self.searcher.search(query,**kwargs)
        extra=tv.Query.boolean_query([(tv.Occur.Should,p) for p in planes])
        leaf=tv.Query.const_score_query(tv.Query.term_query(self.backend.index.schema,'document_type','siret','basic'),0.)
        combined=tv.Query.boolean_query([(tv.Occur.Must,leaf),(tv.Occur.Must,
            tv.Query.boolean_query([(tv.Occur.Should,query),(tv.Occur.Should,extra)]))])
        return self.searcher.search(combined,**kwargs)


def generate(qid,out,reuse_source=None):
    import duckdb
    import pyarrow as pa
    engine.guard()
    if out.exists():raise ValueError('Fresh independent feasibility namespace')
    launch=io.read(BASE/'launch.json');panel=io.read(verify(launch['panel']))
    q=next(q for q in panel if q['query_id']==qid)
    g=next(r['geometry'] for r in io.read(verify(launch['geometry'])) if r['query_id']==qid)
    if not g['spatial_crs_compatible']:raise ValueError('Known query geography required')
    lon,lat=inverse_l93(g['ban_x_l93'],g['ban_y_l93']);radius=min(2000.,max(500.,g.get('footprint_radius_m',0)+300.))
    dy=radius/110000.;dx=radius/(110000.*math.cos(math.radians(lat)))
    bb=','.join(f'{v:.7f}' for v in (lat-dy,lon-dx,lat+dy,lon+dx))
    query=f'[out:json][timeout:20];nwr({bb})[shop][name];out center;'
    out.mkdir(parents=True);(out/'spill').mkdir()
    io.write(out/'launch.json',dict(implementation=record(Path(__file__)),query=q,original_launch=record(BASE/'launch.json'),
        baseline_generation=record(BASE/'generation.json'),recipe=dict(scope='ONE_EXPLICIT_QUERY_DIAGNOSTIC_NOT_RECALL',
            source_request_has_no_company_name_or_id=True,all_neighbourhood_named_shops=True,
            exact_query_name_token_coverage=True,poi_context_not_own_alias=True,
            score='HALF_DIV_ONE_PLUS_DISTANCE_OVER300_MAX_SAME_FROZEN_ACTOR_COEFFICIENT',
            no_fixed_prepool=True,no_positive_injected=True,no_refitting=True),created_unix=time.time()))
    if reuse_source:
        source_ref=record(reuse_source);source=io.read(verify(source_ref))
        if source['query']!=query:raise ValueError('Exact unchanged query-only source request')
        pois=source['pois'];io.write(out/'source.json',dict(source,reused_source=source_ref))
    else:
        req=urllib.request.Request('https://overpass-api.de/api/interpreter',data=urllib.parse.urlencode({'data':query}).encode(),
            headers={'User-Agent':'SIRETO-readonly-named-geography/1.0'})
        with urllib.request.urlopen(req,timeout=30) as response:raw=response.read(5*1024*1024+1)
        if len(raw)>5*1024*1024:raise ValueError('Bounded source body')
        body=json.loads(raw)
        if body.get('remark'):raise ValueError('Incomplete neighbourhood')
        pois=[]
        for e in body['elements']:
            p=e.get('center',e);name=e.get('tags',{}).get('name','')
            if name and 'lon' in p and 'lat' in p:
                x,y=lonlat_to_l93(p['lon'],p['lat']);pois.append(dict(osm_type=e['type'],osm_id=e['id'],name=name,x=x,y=y))
        io.write(out/'source.json',dict(query=query,pois=pois,osm_timestamp=body.get('osm3s',{}).get('timestamp_osm_base'),
            current_community_not_official_identity=True))
    tokens=set(normalize(q['crm_name']).split())-STOP
    matched=[p for p in pois if tokens and tokens<=set(normalize(p['name']).split())-STOP and
             math.hypot(p['x']-g['ban_x_l93'],p['y']-g['ban_y_l93'])<=radius]
    fg=io.read(verify(launch['baseline_features']))
    with np.load(verify(fg['features'][qid]),allow_pickle=False) as f:keys=f['keys'];features=f['features']
    con=duckdb.connect();con.execute('SET threads=2');con.execute("SET memory_limit='512MB'")
    con.execute('SET temp_directory=?',[str(out/'spill')]);con.register('wanted',pa.table({'site_key':keys}))
    owners='/Volumes/CATNAT_DATA/SIRETO_RECALL100/sources/sirene/2026-08-01/StockUniteLegale.parquet'
    public=con.execute("""SELECT CAST(e.siret AS UBIGINT) AS site_key FROM read_parquet(?) e
       JOIN wanted w ON CAST(e.siret AS UBIGINT)=w.site_key JOIN read_parquet(?) u USING(siren)
       WHERE e.statutDiffusionEtablissement='O' AND u.statutDiffusionUniteLegale='O' ORDER BY site_key""",
       [launch['stock_source']['path'],owners]).fetchnumpy()['site_key'];con.close()
    at=np.searchsorted(public,keys);valid=at<len(public);valid[valid]&=public[at[valid]]==keys[valid]
    official=np.load(verify(launch['keys']),mmap_mode='r');xy=np.load(verify(launch['xy']),mmap_mode='r')
    loc=np.searchsorted(official,keys);good=loc<len(official);good[good]&=official[loc[good]]==keys[good];valid&=good
    pos=np.flatnonzero(valid);coords=np.asarray(xy[loc[pos]],dtype=float)
    good=np.isfinite(coords).all(axis=1)&(coords[:,0]>=0)&(coords[:,0]<=1300000)&(coords[:,1]>=6000000)&(coords[:,1]<=7300000)
    pos=pos[good];coords=coords[good];distance=np.full(len(pos),np.inf)
    for p in matched:distance=np.minimum(distance,np.hypot(coords[:,0]-p['x'],coords[:,1]-p['y']))
    confidence=.5/(1+distance/300.)
    sg=io.read(verify(launch['source_generation']));clouds=io.read(verify(sg['shapes']))
    original.bind_runtime(BASE);modified,_=shapes.modified_features(launch,q,keys,features,clouds)
    cal=io.read(BASE/'calibration/generation.json');assignment=io.read(verify(cal['assignment']));fold=str(assignment[qid]);model=io.read(verify(cal['models'][fold]))
    if qid not in model['heldout_queries'] or qid in model['train_queries']:raise ValueError('Heldout calibrator')
    backend=TantivyBackend(io.INDEX)
    proxy=original.baseline.BackendProxy(backend,NamedPlaceCollector(backend,keys[pos],confidence,float(model['coefficients'][6])))
    candidates,cost=original.native_search(proxy,q,keys,modified,model['coefficients'])
    io.write(out/'candidates.json',candidates);engine.seal(out/'candidates.json')
    io.write(out/'generation.json',dict(status='COMPLETE_SEALED_ONE_QUERY_FEASIBILITY_NOT_RECALL',query=q,
        implementation=record(Path(__file__)),source=record(out/'source.json'),candidates=record(out/'candidates.json'),
        neighbourhood_pois=len(pois),exact_name_matched_pois=matched,full_numeric_references=len(keys),
        protected_sites_not_scored=True,model=cal['models'][fold],cost=cost,labels_read=False,production_promotion_allowed=False))
    print('SEALED_ONE_QUERY_DIAGNOSTIC',len(candidates),len(matched),flush=True)


def evaluate(qid,out):
    g=io.read(out/'generation.json');ex=io.read(out/'owned_tool_exit.json')
    if g['status']!='COMPLETE_SEALED_ONE_QUERY_FEASIBILITY_NOT_RECALL' or ex['exit_code']:
        raise ValueError('Complete observed generation exit first')
    candidates=io.read(verify(g['candidates']))
    row=next(r for r in io.read(io.HOME/'v111/five_steps_ledger_20261013_a/cases.json') if r['query_id']==qid)
    rank=next((i for i,p in enumerate(candidates,1) if p['siret']==row['human_target_siret']),None)
    io.write(out/'postseal_diagnostic.json',dict(query_id=qid,exact_target_rank=rank,source_generation=record(out/'generation.json'),
        target_siret=row['human_target_siret'],NOT_A_RECALL_MEASUREMENT=True,no_main_metric_updated=True,
        no_generalization_or_serving_claim=True,observed_unix=time.time()))
    print('POSTSEAL_TARGET_RANK',rank,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['generate','evaluate']);p.add_argument('--query-id',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--reuse-source',type=Path);a=p.parse_args()
    if a.action=='generate':generate(a.query_id,a.output,a.reuse_source)
    else:evaluate(a.query_id,a.output)
