"""Local OSM query-neighbourhood source, ALL526, never reads labels.

One immutable national PBF download, one streaming pass. Only node locations
in query-only buffered neighbourhoods are retained; there is no national RAM
location cache, Nominatim server, per-client request or SIRET lookup here.
Community place context is NOT an official operator/identity or house proof.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import sys
import time
import urllib.request

from scripts import embeddinggemma2_night as io
from scripts.embeddinggemma2_v123_street_geometry_probe import inverse_l93
from scripts.embeddinggemma2_v17_structural_retrieval import normalize
from src.xgb_matcher.retrieval_geo_v16 import lonlat_to_l93
from src.xgb_matcher.retrieval_v20_workbench import record, verify

ROOT = io.HOME/'v129/osm_local_20261013_night'
BASE = io.HOME/'v115/streamed_activity_ablation_20261013_a'
URL = 'https://download.geofabrik.de/europe/france-261012.osm.pbf'
SIZE = 5097546976
DEADLINE = 1791934200.  # 2026-10-13 23:30 UTC: source preparation stop.
POI_KEYS = ('shop','office','amenity','healthcare','tourism','craft','leisure','industrial')
TAG_KEYS = ('name','brand','operator','alt_name','official_name','short_name','old_name',
            'addr:street','addr:housenumber','addr:postcode','addr:city',
            'highway','building') + POI_KEYS


def guard():
    if time.time() >= DEADLINE: raise RuntimeError('Source preparation deadline reached')
    if shutil.disk_usage(ROOT).free < 80*1024**3: raise RuntimeError('CATNAT reserve 80 GiB')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 5*1024**3:
        raise RuntimeError('Source worker RSS exceeded 5 GiB')


def freeze():
    ROOT.mkdir(parents=True, exist_ok=True)
    launch = io.read(BASE/'launch.json')
    panel = io.read(verify(launch['panel']))
    geometry = {r['query_id']:r['geometry'] for r in io.read(verify(launch['geometry']))}
    if len(panel) != 526 or any(q['fold'] not in (2,3,4) for q in panel):
        raise ValueError('Only complete526 development query-only panel')
    queries = []
    for q in panel:
        g=geometry[q['query_id']]
        compatible=bool(g['spatial_crs_compatible'])
        item=dict(query_id=q['query_id'], spatial_crs_compatible=compatible,
                  street_keys=sorted({normalize(s,street=True) for s in
                    (q.get('crm_street',''),g.get('matched_street','')) if s}),
                  search_radius_m=2000., node_buffer_radius_m=3500.)
        if compatible:
            lon,lat=inverse_l93(g['ban_x_l93'],g['ban_y_l93'])
            item.update(lon=lon,lat=lat,x=float(g['ban_x_l93']),y=float(g['ban_y_l93']))
        queries.append(item)
    recipe=dict(all_queries=526,no_labels_read=True,query_only=launch['panel'],
        geometry=launch['geometry'],source_url=URL,expected_size=SIZE,
        context_radius_m=2000.,node_buffer_radius_m=3500.,
        names_not_official_operator_aliases=True,missing_geometry_stays_in_denominator=True,
        community_current_source_not_cutoff_admissible=True,house_position_not_certified=True,
        feature_identifier_tags_not_exported=True,contributor_metadata_not_exported=True,
        relations_exported_only_when_complete_member_geometry=True,
        recipe_version='ALL526_NEIGHBOURHOOD_STREAM_V1')
    if (ROOT/'launch.json').exists():
        old=io.read(ROOT/'launch.json')
        if old['recipe'] != recipe: raise ValueError('Frozen source recipe differs')
        verify(old['implementation'])
        if io.read(ROOT/'queries.json') != queries: raise ValueError('Frozen query geometry differs')
    else:
        io.write(ROOT/'queries.json',queries)
        io.write(ROOT/'launch.json',dict(recipe=recipe,queries=record(ROOT/'queries.json'),
            implementation=record(Path(__file__)),created_unix=time.time()))
    return queries


def download():
    freeze();guard()
    path=ROOT/'france-261012.osm.pbf';part=ROOT/'france-261012.osm.pbf.partial'
    if (ROOT/'download.json').exists(): verify(io.read(ROOT/'download.json')['pbf']);return
    with urllib.request.urlopen(URL+'.md5',timeout=30) as response:
        checksum_body=response.read(1024).decode('ascii')
    expected=checksum_body.split()[0]
    if len(expected)!=32: raise ValueError('Publisher checksum format')
    offset=part.stat().st_size if part.exists() else 0
    headers={'User-Agent':'SIRETO-local-OSM-research/1.0'}
    if offset:headers['Range']=f'bytes={offset}-'
    req=urllib.request.Request(URL,headers=headers)
    started=time.time()
    with urllib.request.urlopen(req,timeout=60) as response:
        if offset and response.status!=206:raise ValueError('Resume requires content-range')
        provenance=dict(url=URL,http_headers=dict(response.headers),status=response.status,
                        publisher_md5_body=checksum_body,started_unix=started)
        io.write(ROOT/'download_request.json',provenance)
        with part.open('ab' if offset else 'wb') as f:
            seen=offset;last=time.monotonic()
            while data:=response.read(4*1024**2):
                f.write(data);seen+=len(data)
                if seen>SIZE:raise ValueError('Source exceeds frozen expected bytes')
                if time.monotonic()-last>15:
                    guard();io.write(ROOT/'download_state.json',dict(bytes=seen,expected=SIZE,
                        elapsed_s=time.time()-started,updated_unix=time.time()));last=time.monotonic()
    if part.stat().st_size != SIZE:raise ValueError('Incomplete PBF')
    h=hashlib.md5();sha=hashlib.sha256()
    with part.open('rb') as f:
        while chunk:=f.read(8*1024**2):h.update(chunk);sha.update(chunk)
    if h.hexdigest()!=expected:raise ValueError('Publisher MD5 mismatch')
    os.replace(part,path)
    io.write(ROOT/'download.json',dict(provenance,pbf=dict(path=str(path),sha256=sha.hexdigest(),
        size_bytes=SIZE),publisher_md5=expected,md5_verified=True,completed_unix=time.time()))
    print('DOWNLOAD_COMPLETE',SIZE,sha.hexdigest(),flush=True)


class Neighbourhoods:
    """Small query-coordinate grid; no national location cache."""
    def __init__(self,queries):
        self.grid=defaultdict(list);self.queries=queries
        for q in queries:
            if not q['spatial_crs_compatible']:continue
            r=q['node_buffer_radius_m'];dy=r/110000;dx=r/(110000*math.cos(math.radians(q['lat'])))
            for ix in range(math.floor((q['lon']-dx)*20),math.floor((q['lon']+dx)*20)+1):
                for iy in range(math.floor((q['lat']-dy)*20),math.floor((q['lat']+dy)*20)+1):
                    self.grid[ix,iy].append(q)
    def nearby(self,lon,lat,buffer=False):
        result=[]
        for q in self.grid.get((math.floor(lon*20),math.floor(lat*20)),()):
            distance=math.hypot((lon-q['lon'])*110000*math.cos(math.radians(q['lat'])),
                                (lat-q['lat'])*110000)
            if distance <= q['node_buffer_radius_m' if buffer else 'search_radius_m']:
                result.append(q['query_id'])
        return result


def scan():
    queries=freeze();guard();download_ref=io.read(ROOT/'download.json');pbf=verify(download_ref['pbf'])
    if (ROOT/'generation.json').exists():raise ValueError('Complete immutable source already exists')
    sys.path.insert(0,str(ROOT/'deps'));import osmium
    lookup=Neighbourhoods(queries);nodes={};ways={};pois={};roads={};addresses={};buildings={}
    counts=defaultdict(int);started=time.time();last=[time.monotonic()]
    def state(force=False):
        if force or time.monotonic()-last[0]>20:
            guard();io.write(ROOT/'state.json',dict(status='STREAMING_NATIONAL_QUERY_ONLY',
                counters=dict(counts),retained_nodes=len(nodes),elapsed_s=time.time()-started,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,updated_unix=time.time()))
            last[0]=time.monotonic()
    def tags(obj):return {k:obj.tags[k] for k in TAG_KEYS if k in obj.tags}
    def object_record(kind,obj,t,xy,lines=None):
        return dict(osm_type=kind,osm_id=obj.id,tags=t,name=t.get('name',''),
            names=sorted({t[k] for k in ('name','brand','alt_name','official_name','short_name','old_name') if t.get(k)}),
            lon=xy[0],lat=xy[1],x=xy[2],y=xy[3],
            geometry_kind='NODE' if kind=='node' else 'ALL_REFERENCED_NODES_MEAN_NOT_ENTRANCE',
            lines_l93=lines or [],points_l93=[[xy[2],xy[3]]],
            source_role='COMMUNITY_GEOGRAPHIC_CONTEXT_NOT_OPERATOR_IDENTITY',not_official_identity=True)
    def register(kind,obj,t,coords,lines=None):
        if not coords:return
        lon=sum(p[0] for p in coords)/len(coords);lat=sum(p[1] for p in coords)/len(coords)
        if not lookup.nearby(lon,lat):return
        x,y=lonlat_to_l93(lon,lat);key=f'{kind}/{obj.id}'
        value=object_record(kind,obj,t,(lon,lat,x,y),lines)
        if any(k in t for k in POI_KEYS) and value['names']:pois[key]=value
        if t.get('addr:street') and t.get('addr:housenumber'):addresses[key]=value
        if t.get('building') and lines:buildings[key]=value
    class Handler(osmium.SimpleHandler):
        def node(self,obj):
            counts['nodes_scanned']+=1
            if counts['nodes_scanned']%1000000==0:state()
            if not obj.location.valid():return
            lon,lat=obj.location.lon,obj.location.lat
            if not lookup.nearby(lon,lat,buffer=True):return
            nodes[obj.id]=(lon,lat)
            t=tags(obj)
            if t:register('node',obj,t,[(lon,lat)])
        def way(self,obj):
            counts['ways_scanned']+=1
            if counts['ways_scanned']%100000==0:state()
            if not any(k in obj.tags for k in ('highway','building','addr:street')+POI_KEYS):return
            refs=[n.ref for n in obj.nodes]
            if not any(n in nodes for n in refs):return
            t=tags(obj);complete=all(n in nodes for n in refs)
            coords=[nodes[n] for n in refs if n in nodes]
            parts=[];part=[]
            for n in refs:
                if n in nodes:part.append(list(lonlat_to_l93(*nodes[n])))
                else:
                    if len(part)>1:parts.append(part)
                    part=[]
            if len(part)>1:parts.append(part)
            if complete:ways[obj.id]=dict(coords=coords,lines=parts)
            else:counts['boundary_partial_ways']+=1
            if t.get('highway') and parts:
                key=f'way/{obj.id}';lon,lat=coords[len(coords)//2];x,y=lonlat_to_l93(lon,lat)
                roads[key]=dict(object_record('way',obj,t,(lon,lat,x,y),parts),
                    node_ids=refs,complete_geometry=complete,name_key=normalize(t.get('name',''),street=True))
            if complete:register('way',obj,t,coords,parts)
        def relation(self,obj):
            counts['relations_scanned']+=1
            if counts['relations_scanned']%10000==0:state()
            if not any(k in obj.tags for k in ('building','addr:street')+POI_KEYS):return
            member_refs=[m.ref for m in obj.members if m.type=='w' and m.role in ('','outer')]
            if not member_refs or not any(n in ways for n in member_refs):return
            if not all(n in ways for n in member_refs):counts['incomplete_relevant_relations']+=1;return
            coords=[p for n in member_refs for p in ways[n]['coords']]
            lines=[p for n in member_refs for p in ways[n]['lines']]
            register('relation',obj,tags(obj),coords,lines)
    reader=osmium.io.Reader(str(pbf));header=reader.header()
    source_timestamp=header.get('osmosis_replication_timestamp')
    reader.close();Handler().apply_file(str(pbf),locations=False)
    state(True)
    # Source-global records never contain a CRM alias, candidate or label.
    for name,data in [('pois',pois),('roads',roads),('addresses',addresses),('buildings',buildings)]:
        io.write(ROOT/(name+'.json'),list(data.values()))
    shards={};shapes={};byqid={q['query_id']:q for q in queries}
    query_objects={qid:dict(pois=[],addresses=[],roads=[],buildings=[]) for qid in byqid}
    for kind,objects in [('pois',pois),('addresses',addresses),('buildings',buildings)]:
        for value in objects.values():
            for qid in lookup.nearby(value['lon'],value['lat']):query_objects[qid][kind].append(value)
    # Roads use actual retained vertices, not midpoint membership: long roads
    # whose centre falls outside a query must not disappear from its geometry.
    for value in roads.values():
        qids=set()
        for node_id in value['node_ids']:
            if node_id in nodes:qids.update(lookup.nearby(*nodes[node_id]))
        for qid in qids:query_objects[qid]['roads'].append(value)
    for qid,q in byqid.items():
        data=query_objects[qid]
        matched=[r for r in data['roads'] if r['name_key'] and r['name_key'] in q['street_keys']]
        shapes[qid]=dict(status='OSM_ACTUAL_ROAD_LINES' if matched else 'NO_MATCHED_OSM_STREET',
            lines_l93=[line for r in matched for line in r['lines_l93']],
            points_l93=[],osm_road_ids=[r['osm_id'] for r in matched],
            not_house_precision=True,not_official_operator_identity=True)
        shards[qid]=dict(spatial_crs_compatible=q['spatial_crs_compatible'],**data)
    io.write(ROOT/'query_context.json',shards);io.write(ROOT/'shapes.json',shapes)
    files={n:record(ROOT/(n+'.json')) for n in
           ('pois','roads','addresses','buildings','query_context','shapes')}
    guard()
    io.write(ROOT/'generation.json',dict(status='COMPLETE_ALL526_LOCAL_OSM_SOURCE',
        implementation=record(Path(__file__)),launch=record(ROOT/'launch.json'),
        download=record(ROOT/'download.json'),files=files,all_queries=526,
        located_queries=sum(q['spatial_crs_compatible'] for q in queries),
        matched_street_queries=sum(bool(v['lines_l93']) for v in shapes.values()),
        queries_with_named_pois=sum(bool(v['pois']) for v in shards.values()),
        counts={**dict(counts),'pois':len(pois),'roads':len(roads),'addresses':len(addresses),
                'buildings':len(buildings),'retained_nodes':len(nodes)},
        source_replication_timestamp=source_timestamp,source_cutoff_policy='CURRENT_COMMUNITY_DIAGNOSTIC_ONLY',
        no_labels_or_siret_read=True,no_recall_measurement=True,
        elapsed_s=time.time()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        completed_unix=time.time()))
    print('COMPLETE_LOCAL_OSM_SOURCE',json.dumps(dict(counts)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['download','scan']);a=p.parse_args()
    if a.action=='download':download()
    else:scan()
