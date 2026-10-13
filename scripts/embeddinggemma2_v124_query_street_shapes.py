"""Complete label-blind street geometry for all weak-geocoded panel queries.

Community OSM geometry is diagnostic-only, not own-business-name evidence.
Exact strong-house geometries are unchanged. No target drives source selection.
"""
from pathlib import Path
import gzip
import json
import math
import time
import urllib.parse
import urllib.request
from collections import defaultdict
import numpy as np

from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v14_full_pipeline as engine
from scripts.embeddinggemma2_v123_street_geometry_probe import inverse_l93,segment_distance
from scripts.embeddinggemma2_v17_structural_retrieval import normalize
from src.xgb_matcher.retrieval_geo_v16 import lonlat_to_l93
from src.xgb_matcher.retrieval_v20_workbench import record,verify

OUTPUT=io.HOME/'v124/query_street_shapes_20261013_a'
BASE=io.HOME/'v115/streamed_activity_ablation_20261013_a'
RECIPE=dict(scope='ALL526_QUERY_ONLY_WEAK_STREET_GEOMETRY',weak_queries_only=True,
    osm_objects='ALL_NAMED_HIGHWAY_WAYS_WITHIN_QUERY_UNCERTAINTY_NEIGHBOURHOOD',
    radius='MIN_2000_MAX_500_EXISTING_FOOTPRINT_PLUS_300_METRES',
    road_names='EXACT_SHARED_NORMALIZATION_OF_PRIOR_BAN_MATCHED_STREET',
    road_axis_not_house_or_property=True,no_shop_names_or_business_ids_read=True,
    original_strong_house_points_unchanged=True,no_model_or_label_read=True,
    community_live_osm_and_postcutoff_ban_diagnostic_only=True)


def main():
    engine.guard()
    if OUTPUT.exists():raise ValueError('Fresh source namespace')
    launch=io.read(BASE/'launch.json');panel=io.read(verify(launch['panel']))
    geometries={r['query_id']:r['geometry'] for r in io.read(verify(launch['geometry']))}
    if len(panel)!=526 or any(q['fold'] not in (2,3,4) for q in panel):raise ValueError('Same526 safe queries')
    wanted=[]
    for q in panel:
        g=geometries[q['query_id']]
        if g['spatial_crs_compatible'] and g['ban_declared_weak'] and g.get('matched_street'):
            lon,lat=inverse_l93(g['ban_x_l93'],g['ban_y_l93'])
            wanted.append(dict(query=q,geometry=g,lon=lon,lat=lat,
                radius=min(2000.,max(500.,float(g.get('footprint_radius_m',0))+300.))))
    OUTPUT.mkdir(parents=True);(OUTPUT/'batches').mkdir()
    io.write(OUTPUT/'launch.json',dict(implementation=record(Path(__file__)),recipe=RECIPE,
        panel=launch['panel'],geometries=launch['geometry'],queries=526,weak_queries=len(wanted),
        helper=record(io.ROOT/'scripts/embeddinggemma2_v123_street_geometry_probe.py'),
        no_labels_read=True,created_unix=time.time()))
    shapes={q['query_id']:dict(status='ORIGINAL_STRONG_OR_UNRESOLVED_UNCHANGED',lines_l93=[]) for q in panel}
    timestamps=[];source_refs=[]
    for batchno,start in enumerate(range(0,len(wanted),8)):
        engine.guard();batch=wanted[start:start+8]
        statements=[f'way(around:{r["radius"]:.2f},{r["lat"]:.7f},{r["lon"]:.7f})[highway][name];' for r in batch]
        query='[out:json][timeout:30];('+''.join(statements)+');out geom;'
        body=None;attempts=[]
        for endpoint in ('https://overpass-api.de/api/interpreter','https://overpass.private.coffee/api/interpreter'):
            try:
                req=urllib.request.Request(endpoint,data=urllib.parse.urlencode({'data':query}).encode(),
                    headers={'User-Agent':'SIRETO-readonly-street-geometry/1.0'})
                with urllib.request.urlopen(req,timeout=45) as response:raw=response.read(20*1024*1024+1)
                if len(raw)>20*1024*1024:raise ValueError('Bounded response')
                value=json.loads(raw)
                if value.get('remark'):raise ValueError(value['remark'])
                body=value;attempts.append(dict(endpoint=endpoint,status='COMPLETE'));break
            except Exception as e:attempts.append(dict(endpoint=endpoint,error=repr(e)))
        io.write(OUTPUT/'batches'/f'{batchno:03d}_request.json',dict(query=query,
            query_ids=[r['query']['query_id'] for r in batch],attempts=attempts))
        if body is None:raise RuntimeError('No complete source response; no partial metrics')
        # Persist only road identity/geometry, never contacts or business metadata.
        elements=[dict(type=e['type'],id=e['id'],name=e.get('tags',{}).get('name',''),
                       highway=e.get('tags',{}).get('highway',''),geometry=e.get('geometry',[]))
                  for e in body['elements'] if e.get('geometry') and e.get('tags',{}).get('name')]
        path=OUTPUT/'batches'/f'{batchno:03d}_roads.json.gz'
        with gzip.open(path,'wt',encoding='utf8') as f:json.dump(elements,f)
        source_refs.append(record(path));timestamps.append(body.get('osm3s',{}).get('timestamp_osm_base'))
        named=defaultdict(list)
        for e in elements:
            if len(e['geometry'])<2:continue
            named[normalize(e['name'],street=True)].append((e['id'],
                np.array([lonlat_to_l93(p['lon'],p['lat']) for p in e['geometry']])))
        for r in batch:
            g=r['geometry'];name=normalize(g['matched_street'],street=True)
            center=np.array([[g['ban_x_l93'],g['ban_y_l93']]])
            lines=[(key,line) for key,line in named[name] if segment_distance(center,line)[0]<=r['radius']+2]
            shapes[r['query']['query_id']]=dict(status='MATCHED_OSM_STREET_AXIS' if lines else 'NO_EXACT_OSM_STREET_UNCHANGED',
                normalized_street=name,osm_way_ids=[key for key,line in lines],lines_l93=[line.tolist() for key,line in lines],
                query_uncertainty=g['ban_uncertainty_m'],road_axis_not_house=True,source_batch=batchno)
        io.write(OUTPUT/'state.json',dict(status='QUERY_ONLY_STREET_SOURCE_COMPILATION',weak_completed=min(start+8,len(wanted)),
            weak_total=len(wanted),all_queries=526,updated_unix=time.time()))
        print('batch',batchno,'weak',min(start+8,len(wanted)),len(wanted),flush=True)
    io.write(OUTPUT/'shapes.json',shapes);engine.seal(OUTPUT/'shapes.json')
    io.write(OUTPUT/'generation.json',dict(status='COMPLETE_ALL526_QUERY_ONLY_STREET_SOURCE',
        launch=record(OUTPUT/'launch.json'),shapes=record(OUTPUT/'shapes.json'),source_batches=source_refs,
        weak_queries=len(wanted),matched_queries=sum(bool(s['lines_l93']) for s in shapes.values()),
        all_queries=526,osm_timestamps=timestamps,no_labels_read=True,not_a_recall_measurement=True,
        diagnostic_only=True,completed_unix=time.time()))


if __name__=='__main__':main()
