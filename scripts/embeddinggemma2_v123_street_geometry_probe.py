"""Query-led full neighbourhood geometry archive and postseal distance audit.

OSM roads/buildings are community geometry, not an official business identity.
No shop-name/SIRET tag is queried. No label enters the map request or rules.
This diagnostic does not generate candidates, change a label or claim recall.
"""
from pathlib import Path
import json
import math
import time
import urllib.request
import urllib.parse
import numpy as np

from scripts import embeddinggemma2_night as io
from src.xgb_matcher import retrieval_geo_v16 as geo
from src.xgb_matcher.retrieval_v20_workbench import record, verify

OUTPUT = io.HOME/'v123/street_geometry_probe_20261013_a'
BASE = io.HOME/'v115/streamed_activity_ablation_20261013_a'
QUERY_ID = '5130'  # The user's explicit diagnostic case; never a learned feature.


def inverse_l93(x, y):
    dx, dy = x-700000., 6600000.+geo._L93_RHO0-y
    rho = math.hypot(dx,dy)
    t = (rho/(geo._GRS80_A*geo._L93_F))**(1/geo._L93_N)
    lat = math.pi/2-2*math.atan(t)
    for _ in range(12):
        s = geo._GRS80_E*math.sin(lat)
        lat = math.pi/2-2*math.atan(t*((1-s)/(1+s))**(geo._GRS80_E/2))
    lon = geo._L93_LON0+math.atan2(dx,dy)/geo._L93_N
    return math.degrees(lon), math.degrees(lat)


def segment_distance(points, line):
    """Euclidean L93 distance to a real polyline, no inferred building linkage."""
    result = np.full(len(points), np.inf)
    for a,b in zip(line[:-1],line[1:]):
        delta = b-a
        den = float(delta@delta)
        t = np.clip(((points-a)@delta)/den,0,1) if den else np.zeros(len(points))
        result = np.minimum(result,np.linalg.norm(points-(a+t[:,None]*delta),axis=1))
    return result


def main():
    import duckdb
    if OUTPUT.exists(): raise ValueError('Fresh geometry namespace')
    launch=io.read(BASE/'launch.json')
    q=next(r for r in io.read(verify(launch['panel'])) if r['query_id']==QUERY_ID)
    g=next(r['geometry'] for r in io.read(verify(launch['geometry'])) if r['query_id']==QUERY_ID)
    lon,lat=inverse_l93(g['ban_x_l93'],g['ban_y_l93'])
    if math.hypot(*(np.array(geo.lonlat_to_l93(lon,lat))-
                    np.array([g['ban_x_l93'],g['ban_y_l93']])))>1e-4:
        raise ValueError('Projection roundtrip integrity')
    # All neighbourhood roads and buildings, no name/target-selected objects.
    bbox=(lat-.012,lon-.019,lat+.012,lon+.019)
    bb=','.join(f'{v:.7f}' for v in bbox)
    query=f'[out:json][timeout:30];(way[highway]({bb});way[building]({bb}););out geom;'
    OUTPUT.mkdir(parents=True);(OUTPUT/'spill').mkdir()
    io.write(OUTPUT/'launch.json',dict(implementation=record(Path(__file__)),query=q,
        query_geometry=g,query_panel=launch['panel'],bbox=bbox,overpass_query=query,
        name_or_business_id_in_osm_request=False,diagnostic_only=True,created_unix=time.time()))
    endpoint='https://overpass-api.de/api/interpreter'
    request=urllib.request.Request(endpoint,data=urllib.parse.urlencode({'data':query}).encode(),
        headers={'User-Agent':'SIRETO-readonly-geometry-audit/1.0'})
    with urllib.request.urlopen(request,timeout=45) as response:
        raw=response.read(20*1024*1024+1)
    if len(raw)>20*1024*1024:raise ValueError('Bounded neighbourhood response')
    body=json.loads(raw)
    if body.get('remark'):raise ValueError('Incomplete Overpass response: '+body['remark'])
    io.write(OUTPUT/'osm.json',body)
    roads=[e for e in body['elements'] if 'highway' in e.get('tags',{}) and e.get('geometry')]
    buildings=[e for e in body['elements'] if 'building' in e.get('tags',{}) and e.get('geometry')]
    from scripts.embeddinggemma2_v17_structural_retrieval import normalize
    target_street=normalize(q['crm_street'],street=True)
    matching=[e for e in roads if normalize(e.get('tags',{}).get('name',''),street=True)==target_street]
    con=duckdb.connect();con.execute('SET threads=2');con.execute("SET memory_limit='256MB'")
    con.execute('SET temp_directory=?',[str(OUTPUT/'spill')])
    con.execute("CREATE MACRO textkey(v) AS trim(regexp_replace(upper(strip_accents(coalesce(v,''))), '[^A-Z0-9]+',' ','g'))")
    ban='/Volumes/CATNAT_DATA/oubusiness/silver/ban_addresses_2026-08-14_semantics-v1.parquet'
    points=con.execute("""SELECT longitude,latitude,numero,rep,type_position
        FROM read_parquet(?) WHERE code_commune=? AND textkey(nom_voie)=?""",
        [ban,q['crm_insee'],target_street]).fetchall();con.close()
    ban_points=[dict(lon=float(a),lat=float(b),number=n,suffix=s,position=p) for a,b,n,s,p in points]
    io.write(OUTPUT/'query_street_points.json',ban_points)
    io.write(OUTPUT/'source_generation.json',dict(status='COMPLETE_QUERY_ONLY_NEIGHBOURHOOD_GEOMETRY',
        osm=record(OUTPUT/'osm.json'),ban_points=record(OUTPUT/'query_street_points.json'),
        roads=len(roads),buildings=len(buildings),query_named_road_segments=len(matching),
        source_osm_timestamp=body.get('osm3s',{}).get('timestamp_osm_base'),
        community_source_not_official_identity=True,no_label_read_in_source_generation=True))
    # Label-side interpretation only after complete map source has been archived.
    case=next(r for r in io.read(io.HOME/'v111/five_steps_ledger_20261013_a/cases.json')
              if r['query_id']==QUERY_ID)
    if case['protected_not_backfilled']:raise ValueError('No masked target reconstruction')
    keys=np.load(verify(launch['keys']),mmap_mode='r');xy=np.load(verify(launch['xy']),mmap_mode='r')
    key=int(case['human_target_siret']);at=np.searchsorted(keys,key)
    if at>=len(keys) or keys[at]!=key:raise ValueError('Official target point absent')
    target=np.asarray(xy[at],dtype=float)
    osm_lines=[np.array([geo.lonlat_to_l93(p['lon'],p['lat']) for p in e['geometry']]) for e in matching]
    d=min((float(segment_distance(target[None,:],line)[0]) for line in osm_lines),default=None)
    bd=min((float(np.linalg.norm(target-np.array(geo.lonlat_to_l93(p['lon'],p['lat']))))
            for p in ban_points),default=None)
    io.write(OUTPUT/'postseal_distance_report.json',dict(query_id=QUERY_ID,target_siret=case['human_target_siret'],
        target_l93=target.tolist(),target_lonlat=inverse_l93(*target),old_query_point_l93=[g['ban_x_l93'],g['ban_y_l93']],
        old_center_distance_m=float(np.linalg.norm(target-np.array([g['ban_x_l93'],g['ban_y_l93']]))),
        nearest_osm_query_road_axis_m=d,nearest_actual_ban_query_road_address_point_m=bd,
        no_same_building_or_identity_claim=True,no_recall_measurement=True,observed_unix=time.time()))
    print(dict(roads=len(roads),buildings=len(buildings),matching_roads=len(matching),
               road_distance=d,ban_point_distance=bd),flush=True)


if __name__=='__main__':main()
