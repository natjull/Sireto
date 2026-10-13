"""Uniform named-place geography, not a hand-written brand/company alias.

All shop/office/amenity names in query-defined neighbourhoods are collected.
An exact informative query-name match may give nearby PUBLIC SIRENE sites
weak typed context. A nearby POI is not a certified operator or own-site name.
"""
from pathlib import Path
import json
import time
import math
import urllib.parse
import urllib.request
import numpy as np
from collections import defaultdict

from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v14_full_pipeline as engine
from scripts.embeddinggemma2_v123_street_geometry_probe import inverse_l93
from scripts.embeddinggemma2_v17_structural_retrieval import normalize
from src.xgb_matcher.retrieval_geo_v16 import lonlat_to_l93
from src.xgb_matcher.retrieval_v20_workbench import record,verify

OUTPUT=io.HOME/'v127/geo_named_pois_20261013_a'
BASE=io.HOME/'v125/street_cloud_retrieval_20261013_b'
STOP={'DE','DU','DES','LA','LE','LES','ET','D','L','AU','AUX'}


def main():
    import duckdb
    import pyarrow as pa
    engine.guard()
    if OUTPUT.exists():raise ValueError('Fresh named-geography namespace')
    launch=io.read(BASE/'launch.json');panel=io.read(verify(launch['panel']))
    gs={r['query_id']:r['geometry'] for r in io.read(verify(launch['geometry']))}
    wanted=[q for q in panel if gs[q['query_id']].get('precision')=='STREET_FOOTPRINT_NOT_HOUSE' and gs[q['query_id']]['spatial_crs_compatible']]
    OUTPUT.mkdir(parents=True);(OUTPUT/'batches').mkdir();(OUTPUT/'shards').mkdir();(OUTPUT/'spill').mkdir()
    recipe=dict(scope='ALL526_WITH_QUERY_ONLY_WEAK_STREET_NEIGHBOURHOODS',poi_types=['shop','office','amenity'],
        exact_informative_name_tokens_required=True,source_context_not_own_alias=True,
        confidence='HALF_DIV_ONE_PLUS_POINT_DISTANCE_OVER_EXISTING300METRES_MAX_NOT_SUM',
        national_public_status_guard_before_score=True,no_candidate_pool_limit=True,
        source_live_community_not_official_operator=True,no_new_model_fitting=True)
    io.write(OUTPUT/'launch.json',dict(implementation=record(Path(__file__)),recipe=recipe,panel=launch['panel'],
        original_geometry=launch['geometry'],original_features=launch['baseline_features'],
        queries=526,weak_queries=len(wanted),labels_read=False,created_unix=time.time()))
    all_pois={};batches=[]
    for batchno,start in enumerate(range(0,len(wanted),8)):
        engine.guard();batch=wanted[start:start+8];parts=[]
        for q in batch:
            g=gs[q['query_id']];lon,lat=inverse_l93(g['ban_x_l93'],g['ban_y_l93'])
            radius=min(2000.,max(500.,float(g.get('footprint_radius_m',0))+300.))
            dy=radius/110000.;dx=radius/(110000.*math.cos(math.radians(lat)))
            bb=','.join(f'{v:.7f}' for v in (lat-dy,lon-dx,lat+dy,lon+dx))
            for kind in recipe['poi_types']:parts.append(f'nwr({bb})[{kind}][name];')
        query='[out:json][timeout:25];('+''.join(parts)+');out center;'
        req=urllib.request.Request('https://overpass-api.de/api/interpreter',
            data=urllib.parse.urlencode({'data':query}).encode(),headers={'User-Agent':'SIRETO-readonly-geographic-context/1.0'})
        with urllib.request.urlopen(req,timeout=35) as response:raw=response.read(10*1024*1024+1)
        if len(raw)>10*1024*1024:raise ValueError('Bounded named-place source')
        body=json.loads(raw)
        if body.get('remark'):raise ValueError('Incomplete source response')
        pois=[]
        for e in body['elements']:
            p=e.get('center',e);name=e.get('tags',{}).get('name','')
            if not name or 'lon' not in p or 'lat' not in p:continue
            x,y=lonlat_to_l93(p['lon'],p['lat'])
            item=dict(osm_type=e['type'],osm_id=e['id'],name=name,x=x,y=y,
                lon=p['lon'],lat=p['lat'],kind=[k for k in recipe['poi_types'] if k in e.get('tags',{})])
            pois.append(item);all_pois[(e['type'],e['id'])]=item
        path=OUTPUT/'batches'/f'{batchno:03d}.json'
        io.write(path,dict(query=query,pois=pois,osm_timestamp=body.get('osm3s',{}).get('timestamp_osm_base')));batches.append(record(path))
        io.write(OUTPUT/'state.json',dict(status='ALL_WEAK_NEIGHBOURHOOD_NAMED_GEOGRAPHY',completed=min(start+8,len(wanted)),total=len(wanted),updated_unix=time.time()))
        print('batch',batchno,'weak',min(start+8,len(wanted)),len(wanted),flush=True)
    # All source collection completes before any label/evaluation access.
    poi_values=list(all_pois.values());io.write(OUTPUT/'all_pois.json',poi_values)
    con=duckdb.connect();con.execute('SET threads=2');con.execute("SET memory_limit='512MB'")
    con.execute('SET temp_directory=?',[str(OUTPUT/'spill')])
    owners='/Volumes/CATNAT_DATA/SIRETO_RECALL100/sources/sirene/2026-08-01/StockUniteLegale.parquet'
    records=con.execute("""SELECT CAST(e.siret AS UBIGINT) ref FROM read_parquet(?) e JOIN read_parquet(?) u USING(siren)
        WHERE e.statutDiffusionEtablissement='O' AND u.statutDiffusionUniteLegale='O' ORDER BY ref""",
        [launch['stock_source']['path'],owners]).fetchnumpy();con.close();public=records['ref']
    fg=io.read(verify(launch['baseline_features']));original_keys=np.load(verify(launch['keys']),mmap_mode='r');xy=np.load(verify(launch['xy']),mmap_mode='r')
    shards={};matched_queries=0
    weak_ids={q['query_id'] for q in wanted}
    for q in panel:
        engine.guard();tokens=set(normalize(q['crm_name']).split())-STOP;selected=[]
        if q['query_id'] in weak_ids and tokens:
            g=gs[q['query_id']];radius=min(2000.,max(500.,float(g.get('footprint_radius_m',0))+300.))
            selected=[p for p in poi_values if tokens<=set(normalize(p['name']).split())-STOP and
                math.hypot(p['x']-g['ban_x_l93'],p['y']-g['ban_y_l93'])<=radius]
        if not selected:
            shards[q['query_id']]=dict(matched_poi_count=0,no_extra_query_component=True);continue
        matched_queries+=1
        with np.load(verify(fg['features'][q['query_id']]),allow_pickle=False) as f:keys=f['keys']
        at=np.searchsorted(public,keys);eligible=at<len(public);eligible[eligible]&=public[at[eligible]]==keys[eligible]
        loc=np.searchsorted(original_keys,keys);valid=loc<len(original_keys);valid[valid]&=original_keys[loc[valid]]==keys[valid];valid&=eligible
        pos=np.flatnonzero(valid);coords=np.asarray(xy[loc[pos]],dtype=float)
        good=np.isfinite(coords).all(axis=1)&(coords[:,0]>=0)&(coords[:,0]<=1300000)&(coords[:,1]>=6000000)&(coords[:,1]<=7300000)
        pos=pos[good];coords=coords[good];distance=np.full(len(pos),np.inf)
        for p in selected:distance=np.minimum(distance,np.hypot(coords[:,0]-p['x'],coords[:,1]-p['y']))
        # Whole original proof universe. No nearest-k, forced slot or exact-ID boost.
        confidence=.5/(1+distance/300.)
        path=OUTPUT/'shards'/(str(len(shards))+'.npz')
        np.savez_compressed(path,keys=keys[pos],confidence=confidence)
        shards[q['query_id']]=dict(matched_poi_count=len(selected),scores=record(path),poi_refs=[(p['osm_type'],p['osm_id']) for p in selected],
            all_public_coordinate_references=len(pos),not_identity_claim=True)
    io.write(OUTPUT/'contexts.json',shards);engine.seal(OUTPUT/'contexts.json')
    io.write(OUTPUT/'generation.json',dict(status='COMPLETE_ALL526_NAMED_GEOGRAPHIC_CONTEXT',launch=record(OUTPUT/'launch.json'),
        contexts=record(OUTPUT/'contexts.json'),pois=record(OUTPUT/'all_pois.json'),batches=batches,
        queries=526,weak_queries=len(wanted),matched_queries=matched_queries,public_sites=len(public),
        no_labels_read=True,no_new_recall=True,production_promotion_allowed=False,completed_unix=time.time()))
    print('COMPLETE',matched_queries,flush=True)


if __name__=='__main__':main()
