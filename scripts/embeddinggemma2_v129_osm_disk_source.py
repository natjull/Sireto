"""Bounded local OSM source: persistent node cache, disk refs/object spool.

Same ALL526 neighbourhoods, POIs, addresses and street geometry as V129.
Anonymous decorative building polygons remain explicitly omitted. No labels,
business identifiers, candidate injection or external per-query requests.
After nodes_ready, a failed way/output stage can resume without node callbacks.
"""
from __future__ import annotations
import argparse
from array import array
from collections import defaultdict
import gc
import json
import os
from pathlib import Path
import resource
import sqlite3
import sys
import time

from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v129_osm_local_source as original
from scripts.embeddinggemma2_v17_structural_retrieval import normalize
from src.xgb_matcher.retrieval_geo_v16 import lonlat_to_l93
from src.xgb_matcher.retrieval_v20_workbench import record,verify

OLD=io.HOME/'v129/osm_local_20261013_night'
PREVIOUS=io.HOME/'v129/osm_local_20261013_night_c'
ROOT=io.HOME/'v129/osm_local_20261013_night_d'
sys.path.insert(0,str(OLD/'deps'))
import osmium


class Locations:
    def __init__(self,tracker,resume=False):
        self.path=ROOT/'node_locations.cache';self.ids_path=ROOT/'node_ids.u64'
        self.table=osmium.index.create_map('sparse_file_array,'+str(self.path))
        self.tracker=tracker;self.buffer=array('Q');self.ready=resume;self.count=0
        self.ids_file=None if resume else self.ids_path.open('wb')
        if resume:
            ready=io.read(ROOT/'nodes_ready.json');verify(ready['cache']);verify(ready['ids'])
            with self.ids_path.open('rb') as f:
                while chunk:=f.read(8*131072):
                    values=array('Q');values.frombytes(chunk)
                    for key in values:tracker.add_node(key)
                    self.count+=len(values)
            if self.count!=ready['retained_nodes']:raise ValueError('Node cache count binding')
    def __len__(self):return self.count
    def __setitem__(self,key,value):
        self.table.set(key,osmium.osm.Location(*value));self.tracker.add_node(key)
        self.buffer.append(key);self.count+=1
        if len(self.buffer)>=131072:self.flush()
    def __contains__(self,key):
        try:return self.table.get(key).valid()
        except KeyError:return False
    def __getitem__(self,key):
        value=self.table.get(key);return value.lon,value.lat
    def flush(self):
        if self.buffer:self.ids_file.write(self.buffer.tobytes());self.buffer=array('Q')
    def finalize(self,db,counts):
        if self.ready:return
        self.flush();self.ids_file.flush();os.fsync(self.ids_file.fileno());self.ids_file.close()
        db.commit();self.table=None;gc.collect()
        # Destruction trims/flushes the sparse file vector. Reopening uses the
        # documented same native file index API, not an assumed binary layout.
        self.table=osmium.index.create_map('sparse_file_array,'+str(self.path))
        io.write(ROOT/'nodes_ready.json',dict(status='COMPLETE_QUERY_BUFFER_NODE_CACHE',
            cache=record(self.path),ids=record(self.ids_path),retained_nodes=self.count,
            counters=dict(counts),launch=record(ROOT/'launch.json'),complete_unix=time.time()))
        self.ready=True


def main(resume=False):
    old_launch=io.read(OLD/'launch.json');verify(old_launch['implementation'])
    if io.read(PREVIOUS/'owned_tool_exit.json')['exit_code']!=1:
        raise ValueError('Observed previous guard exit required')
    download=io.read(OLD/'download.json');pbf=verify(download['pbf'])
    if resume:
        if not (ROOT/'nodes_ready.json').exists() or (ROOT/'generation.json').exists():
            raise ValueError('Resume requires complete node cache and no final generation')
        verify(io.read(ROOT/'launch.json')['implementation'])
    else:
        if ROOT.exists():raise ValueError('Fresh disk namespace required')
        ROOT.mkdir(parents=True)
        io.write(ROOT/'download.json',dict(download,reused_verified_download=record(OLD/'download.json')))
        io.write(ROOT/'migration.json',dict(original_implementation=old_launch['implementation'],
            previous_exit=record(PREVIOUS/'owned_tool_exit.json'),implementation=record(Path(__file__)),
            unchanged_scientific_fields=['ALL526_QUERY_GEOMETRY','POIS','ADDRESSES','MATCHED_STREET_LINES'],
            optional_omission='ANONYMOUS_BUILDINGS_NOT_USED_BY_V132',
            node_storage='PERSISTENT_CPP_SPARSE_FILE_ARRAY_WITH_NODE_IDS_FOR_RESUME',
            way_storage='SQLITE_REF_BLOBS',object_storage='SQLITE_JSON_WITH_QUERY_MEMBERSHIP',
            no_redownload=True,no_label_read=True,no_new_scientific_filter=True))
    # Reuse the exact historical query-source resolver without changing its file.
    original.ROOT=ROOT;original.__file__=str(Path(__file__).resolve())
    queries=original.freeze();original.guard();lookup=original.Neighbourhoods(queries)
    tracker=osmium.IdTracker();nodes=Locations(tracker,resume)
    db=sqlite3.connect(ROOT/'objects.sqlite');db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA synchronous=NORMAL');db.execute('PRAGMA cache_size=-65536')
    db.execute('PRAGMA temp_store=FILE')
    db.executescript('''CREATE TABLE IF NOT EXISTS ways(id INTEGER PRIMARY KEY,refs BLOB NOT NULL);
      CREATE TABLE IF NOT EXISTS objects(object_key TEXT PRIMARY KEY,data TEXT NOT NULL,name_key TEXT NOT NULL) WITHOUT ROWID;
      CREATE TABLE IF NOT EXISTS kinds(kind TEXT NOT NULL,object_key TEXT NOT NULL,PRIMARY KEY(kind,object_key)) WITHOUT ROWID;
      CREATE TABLE IF NOT EXISTS members(qid TEXT NOT NULL,kind TEXT NOT NULL,object_key TEXT NOT NULL,
        PRIMARY KEY(qid,kind,object_key)) WITHOUT ROWID;''')
    if resume:
        # Node objects are already complete and query-bound. Restart only the
        # incomplete way/relation object stage, at identical source binding.
        db.execute('DELETE FROM members WHERE object_key NOT LIKE ?',('node/%',))
        db.execute('DELETE FROM kinds WHERE object_key NOT LIKE ?',('node/%',))
        db.execute('DELETE FROM objects WHERE object_key NOT LIKE ?',('node/%',))
        db.execute('DELETE FROM ways');db.commit()
    counts=defaultdict(int,io.read(ROOT/'nodes_ready.json')['counters'] if resume else {})
    counts['ways_scanned']=0;counts['relations_scanned']=0
    counts['boundary_partial_ways']=0;counts['incomplete_relevant_relations']=0
    started=time.time();last=[time.monotonic()];writes=[0]
    def state(force=False):
        if force or time.monotonic()-last[0]>20:
            original.guard();db.commit()
            io.write(ROOT/'state.json',dict(status='STREAMING_NATIONAL_TO_DISK',owned_pid=os.getpid(),
                counters=dict(counts),retained_nodes=len(nodes),elapsed_s=time.time()-started,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,updated_unix=time.time()))
            last[0]=time.monotonic()
    def tags(obj):return {k:obj.tags[k] for k in original.TAG_KEYS if k in obj.tags}
    def has_name(t):return any(t.get(k) for k in ('name','brand','alt_name','official_name','short_name','old_name'))
    def value(kind,obj,t,lon,lat,lines=None):
        x,y=lonlat_to_l93(lon,lat)
        return dict(osm_type=kind,osm_id=obj.id,tags=t,name=t.get('name',''),
            names=sorted({t[k] for k in ('name','brand','alt_name','official_name','short_name','old_name') if t.get(k)}),
            lon=lon,lat=lat,x=x,y=y,points_l93=[[x,y]],lines_l93=lines or [],
            geometry_kind='NODE' if kind=='node' else 'ALL_REFERENCED_NODES_MEAN_NOT_ENTRANCE',
            source_role='COMMUNITY_GEOGRAPHIC_CONTEXT_NOT_OPERATOR_IDENTITY',not_official_identity=True)
    def store(v,categories,qids):
        key=f"{v['osm_type']}/{v['osm_id']}"
        db.execute('INSERT OR REPLACE INTO objects VALUES(?,?,?)',
            (key,json.dumps(v,ensure_ascii=False,separators=(',',':')),v.get('name_key','')))
        for cat in categories:
            db.execute('INSERT OR IGNORE INTO kinds VALUES(?,?)',(cat,key))
            db.executemany('INSERT OR IGNORE INTO members VALUES(?,?,?)',[(qid,cat,key) for qid in qids])
        writes[0]+=1
        if writes[0]%8192==0:db.commit()
    def register(kind,obj,t,coords,lines=None):
        if not coords:return
        categories=[];named=has_name(t)
        if any(k in t for k in original.POI_KEYS) and named:categories.append('pois')
        if t.get('addr:street') and t.get('addr:housenumber'):categories.append('addresses')
        if t.get('building') and lines and named:categories.append('buildings')
        if not categories:return
        lon=sum(p[0] for p in coords)/len(coords);lat=sum(p[1] for p in coords)/len(coords)
        qids=lookup.nearby(lon,lat)
        if qids:store(value(kind,obj,t,lon,lat,lines),categories,qids)
    class Handler(osmium.SimpleHandler):
        def node(self,obj):
            if resume:return
            counts['nodes_scanned']+=1
            if counts['nodes_scanned']%1000000==0:state()
            if not obj.location.valid():return
            lon,lat=obj.location.lon,obj.location.lat
            if not lookup.nearby(lon,lat,buffer=True):return
            nodes[obj.id]=(lon,lat);t=tags(obj)
            if t:register('node',obj,t,[(lon,lat)])
        def way(self,obj):
            nodes.finalize(db,counts)
            counts['ways_scanned']+=1
            if counts['ways_scanned']%100000==0:state()
            if not any(k in obj.tags for k in ('highway','building','addr:street')+original.POI_KEYS):return
            refs=[n.ref for n in obj.nodes]
            if not any(n in nodes for n in refs):return
            t=tags(obj);complete=all(n in nodes for n in refs)
            if complete:
                db.execute('INSERT OR REPLACE INTO ways VALUES(?,?)',(obj.id,array('Q',refs).tobytes()))
                tracker.add_way(obj.id)
            else:counts['boundary_partial_ways']+=1
            need_lines=t.get('highway') or (t.get('addr:street') and t.get('addr:housenumber')) or (any(k in t for k in original.POI_KEYS) and has_name(t)) or (t.get('building') and has_name(t))
            if not need_lines:return
            coords=[];parts=[];part=[];qids=set()
            for n in refs:
                if n in nodes:
                    p=nodes[n];coords.append(p);part.append(list(lonlat_to_l93(*p)));qids.update(lookup.nearby(*p))
                else:
                    if len(part)>1:parts.append(part)
                    part=[]
            if len(part)>1:parts.append(part)
            if t.get('highway') and parts:
                lon,lat=coords[len(coords)//2];v=value('way',obj,t,lon,lat,parts)
                v.update(node_ids=refs,complete_geometry=complete,name_key=normalize(t.get('name',''),street=True))
                store(v,['roads'],qids)
            if complete:register('way',obj,t,coords,parts)
        def relation(self,obj):
            nodes.finalize(db,counts)
            counts['relations_scanned']+=1
            if counts['relations_scanned']%10000==0:state()
            if not any(k in obj.tags for k in ('building','addr:street')+original.POI_KEYS):return
            members=[m.ref for m in obj.members if m.type=='w' and m.role in ('','outer')]
            if not members:return
            data=[]
            for member in members:
                row=db.execute('SELECT refs FROM ways WHERE id=?',(member,)).fetchone()
                if row:
                    refs=array('Q');refs.frombytes(row[0]);data.append(refs)
                else:data.append(None)
            if not any(x is not None for x in data):return
            if any(x is None for x in data):counts['incomplete_relevant_relations']+=1;return
            coords=[nodes[r] for refs in data for r in refs]
            lines=[[list(lonlat_to_l93(*nodes[r])) for r in refs] for refs in data]
            register('relation',obj,tags(obj),coords,lines)
    reader=osmium.io.Reader(str(pbf));timestamp=reader.header().get('osmosis_replication_timestamp');reader.close()
    filters=[tracker.contains_filter()]
    if resume:filters.insert(0,osmium.filter.EntityFilter(osmium.osm.WAY|osmium.osm.RELATION))
    Handler().apply_file(str(pbf),locations=False,filters=filters)
    nodes.finalize(db,counts);state(True);db.commit();db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    # No more geometry calls: free native indices before JSON serialization.
    nodes.table=None;tracker.node_ids().clear();tracker.way_ids().clear();gc.collect()
    def stream_list(path,sql,args=()):
        tmp=path.with_suffix('.json.tmp')
        with tmp.open('w') as out:
            out.write('[');first=True
            for row in db.execute(sql,args):
                if not first:out.write(',')
                out.write(row[0]);first=False
            out.write(']')
        os.replace(tmp,path)
    totals={}
    for kind in ('pois','roads','addresses','buildings'):
        stream_list(ROOT/(kind+'.json'),'SELECT o.data FROM kinds k JOIN objects o USING(object_key) WHERE k.kind=? ORDER BY object_key',(kind,))
        totals[kind]=db.execute('SELECT count(*) FROM kinds WHERE kind=?',(kind,)).fetchone()[0]
        original.guard()
    shapes={};with_pois=0
    tmp=ROOT/'query_context.json.tmp'
    with tmp.open('w') as out:
        out.write('{');first=True
        for q in queries:
            qid=q['query_id']
            if not first:out.write(',')
            first=False;out.write(json.dumps(qid)+':{"spatial_crs_compatible":'+json.dumps(q['spatial_crs_compatible']))
            matched=[];poi_count=0
            for kind in ('pois','addresses','roads','buildings'):
                out.write(','+json.dumps(kind)+':[');empty=True
                for (body,) in db.execute('SELECT o.data FROM members m JOIN objects o USING(object_key) WHERE m.qid=? AND m.kind=? ORDER BY object_key',(qid,kind)):
                    if not empty:out.write(',')
                    out.write(body);empty=False
                    if kind=='pois':poi_count+=1
                    if kind=='roads':
                        v=json.loads(body)
                        if v['name_key'] and v['name_key'] in q['street_keys']:matched.append(v)
                out.write(']')
            out.write('}');with_pois+=bool(poi_count)
            shapes[qid]=dict(status='OSM_ACTUAL_ROAD_LINES' if matched else 'NO_MATCHED_OSM_STREET',
                lines_l93=[line for r in matched for line in r['lines_l93']],points_l93=[],
                osm_road_ids=[r['osm_id'] for r in matched],not_house_precision=True,not_official_operator_identity=True)
            original.guard()
        out.write('}')
    os.replace(tmp,ROOT/'query_context.json');io.write(ROOT/'shapes.json',shapes)
    db.close();original.guard()
    files={n:record(ROOT/(n+'.json')) for n in ('pois','roads','addresses','buildings','query_context','shapes')}
    io.write(ROOT/'generation.json',dict(status='COMPLETE_ALL526_LOCAL_OSM_SOURCE',
        implementation=record(Path(__file__)),launch=record(ROOT/'launch.json'),migration=record(ROOT/'migration.json'),
        download=record(ROOT/'download.json'),files=files,all_queries=526,
        located_queries=sum(q['spatial_crs_compatible'] for q in queries),
        matched_street_queries=sum(bool(v['lines_l93']) for v in shapes.values()),queries_with_named_pois=with_pois,
        counts={**dict(counts),**totals,'retained_nodes':len(nodes)},source_replication_timestamp=timestamp,
        source_cutoff_policy='CURRENT_COMMUNITY_DIAGNOSTIC_ONLY',no_labels_or_siret_read=True,no_recall_measurement=True,
        anonymous_buildings_explicitly_omitted=True,counters_after_native_filter_are_callback_counts=True,
        elapsed_s=time.time()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,completed_unix=time.time()))
    print('COMPLETE_LOCAL_OSM_SOURCE',json.dumps(dict(counts)),totals,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resume-ways',action='store_true');a=p.parse_args();main(a.resume_ways)
